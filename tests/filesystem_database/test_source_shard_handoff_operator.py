"""Operator cannot expand a source bucket before BOTH live worker ACKs."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from file_system_catalog.source_shard_handoff_operator import (
    compute_step, UNIT, verify_releasing_worker_acknowledged,
    verify_releasing_q4_acknowledged,
)
from file_system_catalog.source_shard_runtime import SCHEMA, KNOWN_STAGES, write_manifest_bytes

NOW=datetime(2026,11,21,17,0,0,tzinfo=timezone.utc)


def cli():
    target=Path(__file__).resolve().parents[2]/'scripts/raspi_source_shard_handoff.py'
    spec=importlib.util.spec_from_file_location('isolated_shard_handoff_operator',target)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class ShardHandoffOperatorTests(unittest.TestCase):
    def setUp(self):
        t=TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.home=Path(t.name)
        self.manifest=self.home/'handoff.json'
        self.states={
            'coder':self.status('coder',0,None,None),
            'q4':self.status('q4',0,None,None),
        }

    @staticmethod
    def status(worker,generation,stage,digest):
        ts=(NOW-timedelta(seconds=30)).isoformat()
        return {
            'worker_pid':1111 if worker=='coder' else 2222,
            'started_utc':ts,'allocation_runtime_started_utc':ts,
            'updated_utc':NOW.isoformat(),
            'allocation_protocol':SCHEMA,
            'allocation_generation':generation,
            'allocation_stage':stage,
            'allocation_manifest_sha256':digest,
            'assigned_shards':sorted(KNOWN_STAGES[stage or 'baseline'][worker]),
            'phase':'running','current_original':None,
        }

    @staticmethod
    def live_pid(unit):
        return 1111 if unit==UNIT['coder'] else 2222

    @staticmethod
    def proc_command(pid):
        cmd=b'/venv/python\x00-u\x00/repo/scripts/raspi_hierarchical_worker.py\x00--shard-count\x004\x00'
        if pid==2222:return cmd+b'--worker-label\x00q4\x00'
        return cmd

    def step(self,target='balanced',**kw):
        return compute_step(manifest_path=self.manifest,
                            worker_statuses=kw.get('statuses',self.states),
                            target=target,live_pid=kw.get('live_pid',self.live_pid),
                            proc_command=kw.get('proc_command',self.proc_command),
                            now_utc=kw.get('now',NOW))

    def stage_and_acknowledge(self,stage,generation):
        data=write_manifest_bytes(stage,generation)
        self.manifest.write_bytes(data)
        self.manifest.chmod(0o600)
        digest=hashlib.sha256(data).hexdigest()
        self.states={key:self.status(key,generation,stage,digest) for key in ('coder','q4')}

    def test_complete_five_stage_release_acquire_and_rollback(self):
        stages=[('balanced','baseline'),('balanced','release_three'),
                ('balanced','balanced'),('baseline','release_three'),
                ('baseline','baseline')]
        for generation,(target,expected) in enumerate(stages,1):
            with self.subTest(generation=generation):
                plan=self.step(target)
                self.assertTrue(plan['requires_change'])
                self.assertEqual(expected,plan['source_next_stage'])
                self.assertEqual(generation,plan['source_next_generation'])
                self.assertEqual(set(),set(plan['source_next_assignments']['coder']) & set(plan['source_next_assignments']['q4']))
                self.stage_and_acknowledge(expected,generation)
        self.assertFalse(self.step('baseline')['requires_change'])
        self.assertEqual('baseline',self.step('baseline')['source_next_stage'])

    def test_expansion_requires_coder_source_boundary_ack_and_q4_ack(self):
        self.stage_and_acknowledge('release_three',2)
        for owner in ('coder','q4'):
            broken=deepcopy(self.states)
            broken[owner]['allocation_generation']=1
            with self.subTest(owner=owner),self.assertRaisesRegex(ValueError,'not yet acknowledged'):
                self.step('balanced',statuses=broken)
        good=self.step('balanced')
        self.assertEqual('balanced',good['source_next_stage'])
        self.assertEqual(3,good['source_next_generation'])

    def test_legacy_and_stale_processes_are_not_authorized_to_apply(self):
        broken=deepcopy(self.states)
        broken['coder'].pop('allocation_protocol')
        with self.assertRaisesRegex(ValueError,'not running'):
            self.step(statuses=broken)
        broken=deepcopy(self.states)
        broken['coder']['allocation_runtime_started_utc']='old-version-stale-marker'
        with self.assertRaisesRegex(ValueError,'not running'):
            self.step(statuses=broken)
        broken=deepcopy(self.states)
        broken['q4']['updated_utc']=(NOW-timedelta(hours=1)).isoformat()
        with self.assertRaisesRegex(ValueError,'stale'):
            self.step(statuses=broken)
        with self.assertRaisesRegex(ValueError,'live main PID'):
            self.step(live_pid=lambda unit:9999)
        with self.assertRaisesRegex(ValueError,'unexpected live'):
            self.step(proc_command=lambda pid:b'old_script_or_wrong_model')

    def test_acquiring_worker_checks_real_coder_release_independently_of_manifest(self):
        self.stage_and_acknowledge('release_three',2)
        current=self.states['coder']
        evidence=verify_releasing_worker_acknowledged(
            coder_state=current,requested_generation=3,
            live_pid=self.live_pid,proc_command=self.proc_command,now_utc=NOW)
        self.assertEqual([1,2],evidence['assigned_shards'])
        self.assertEqual(1111,evidence['pid'])
        broken=deepcopy(current)
        broken.update(allocation_stage='baseline',assigned_shards=[1,2,3])
        with self.assertRaisesRegex(ValueError,'not acknowledged release'):
            verify_releasing_worker_acknowledged(coder_state=broken,requested_generation=3,
                live_pid=self.live_pid,proc_command=self.proc_command,now_utc=NOW)
        with self.assertRaisesRegex(ValueError,'not acknowledged release'):
            verify_releasing_worker_acknowledged(coder_state=current,requested_generation=5,
                live_pid=self.live_pid,proc_command=self.proc_command,now_utc=NOW)
        with self.assertRaisesRegex(ValueError,'live main PID'):
            verify_releasing_worker_acknowledged(coder_state=current,requested_generation=3,
                live_pid=lambda service:9999,proc_command=self.proc_command,now_utc=NOW)
        with self.assertRaisesRegex(ValueError,'unexpected live'):
            verify_releasing_worker_acknowledged(coder_state=current,requested_generation=3,
                live_pid=self.live_pid,proc_command=lambda pid:b'older-worker-no-ack',now_utc=NOW)

    def test_coder_rollback_reacquisition_requires_q4_boundary_ack(self):
        self.stage_and_acknowledge('release_three',4)
        current=self.states['q4']
        proof=verify_releasing_q4_acknowledged(
            q4_state=current,requested_generation=5,
            live_pid=self.live_pid,proc_command=self.proc_command,now_utc=NOW)
        self.assertEqual([0],proof['assigned_shards'])
        self.assertEqual(2222,proof['pid'])
        stale=deepcopy(current)
        stale['allocation_stage']='balanced'
        stale['assigned_shards']=[0,3]
        with self.assertRaisesRegex(ValueError,'Q4 has not acknowledged'):
            verify_releasing_q4_acknowledged(q4_state=stale,requested_generation=5,
                live_pid=self.live_pid,proc_command=self.proc_command,now_utc=NOW)
        with self.assertRaisesRegex(ValueError,'live main PID'):
            verify_releasing_q4_acknowledged(q4_state=current,requested_generation=5,
                live_pid=lambda service:9999,proc_command=self.proc_command,now_utc=NOW)
        with self.assertRaisesRegex(ValueError,'unexpected live'):
            verify_releasing_q4_acknowledged(q4_state=current,requested_generation=5,
                live_pid=self.live_pid,proc_command=lambda pid:b'old Q4 worker',now_utc=NOW)

    def test_busy_source_in_released_bucket_blocks_next_stage(self):
        self.stage_and_acknowledge('release_three',2)
        broken=deepcopy(self.states)
        from file_system_catalog.summary_shard_planner import compare_assignments
        import hashlib as _hashlib
        target=next(f'docs/file-{n}.pdf' for n in range(100)
                    if int.from_bytes(_hashlib.sha256(f'docs/file-{n}.pdf'.encode()).digest()[:8],'big')%4==3)
        broken['coder']['current_original']=target
        with self.assertRaisesRegex(ValueError,'released bucket'):
            self.step(statuses=broken)

    def test_atomic_operator_waits_for_both_worker_acknowledgements(self):
        module=cli()
        status_files={key:self.home/(key+'.json') for key in self.states}
        def save_status():
            for key,data in self.states.items():
                status_files[key].write_text(json.dumps(data))
                status_files[key].chmod(0o600)
        save_status()
        from file_system_catalog.source_shard_handoff_operator import compute_step as verified_plan
        with (mock.patch.multiple(module,ROOT=self.home,MANIFEST=self.manifest,
                                  LOCK=self.home/'operator.lock',STATUS=status_files,
                                  JOURNAL=self.home/'handoff-events.jsonl',TARGET='balanced'),
              mock.patch.object(module,'live_pid',side_effect=self.live_pid),
              mock.patch.object(module,'proc_command',side_effect=self.proc_command),
              mock.patch.object(module,'compute_step',side_effect=lambda **kw:
                                verified_plan(now_utc=NOW,**kw))):
            first=module.safe_apply()
            self.assertEqual('baseline',first['source_next_stage'])
            self.assertEqual('owner_only_manifest_stage_persisted',first['action'])
            self.assertEqual(0o600,self.manifest.stat().st_mode&0o777)
            content=self.manifest.read_bytes()
            with self.assertRaisesRegex(ValueError,'not yet acknowledged'):
                module.safe_apply()
            self.assertEqual(content,self.manifest.read_bytes())
            self.stage_and_acknowledge('baseline',1)
            save_status()
            second=module.safe_apply()
            self.assertEqual('release_three',second['source_next_stage'])
            content=self.manifest.read_bytes()
            with self.assertRaisesRegex(ValueError,'not yet acknowledged'):
                module.safe_apply()
            self.assertEqual(content,self.manifest.read_bytes())
            self.stage_and_acknowledge('release_three',2)
            save_status()
            third=module.safe_apply()
            self.assertEqual('balanced',third['source_next_stage'])
            self.stage_and_acknowledge('balanced',3)
            save_status()
            noop=module.safe_apply()
            self.assertFalse(noop['requires_change'])
            self.assertEqual('no_change',noop['action'])
            self.assertEqual(write_manifest_bytes('balanced',3),self.manifest.read_bytes())
            events=[json.loads(s) for s in (self.home/'handoff-events.jsonl').read_text().splitlines()]
            self.assertEqual(['baseline','release_three','balanced'],[z['new_stage'] for z in events])
            self.assertEqual([1,2,3],[z['new_generation'] for z in events])
            self.assertTrue(all(z['any_service_signalled'] is False for z in events))
            self.assertEqual(0o600,(self.home/'handoff-events.jsonl').stat().st_mode&0o777)


if __name__=='__main__':
    unittest.main()
