"""Strict private source-boundary controls. No live marker or worker signals."""
from __future__ import annotations
from datetime import datetime,timedelta,timezone
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from file_system_catalog.source_page_quiescence import (
    SCHEMA,SCOPE,SourceBoundaryQuiescence,marker_bytes,read_control,
    hold_at_source_boundary,read_only_worker_acknowledgements,
)


class PublisherBoundaryQuiescenceTests(unittest.TestCase):
    def setUp(self):
        t=TemporaryDirectory();self.addCleanup(t.cleanup)
        self.root=Path(t.name)
        self.control=self.root/'optional-publisher-hold.json'

    def write(self,action,generation):
        blob=marker_bytes(action,generation)
        temp=self.control.with_suffix('.next')
        temp.write_bytes(blob);temp.chmod(0o600)
        os.replace(temp,self.control)
        return hashlib.sha256(blob).hexdigest()

    def test_no_marker_preserves_original_active_worker_behavior(self):
        default=read_control(self.control)
        self.assertEqual(SourceBoundaryQuiescence(0,'release',None),default)
        updates=[]
        returned=hold_at_source_boundary(marker=self.control,worker_label='coder',
                 worker_pid=44,prior_generation=0,prior_digest=None,
                 update_status=lambda **kw:updates.append(kw),should_stop=lambda:False,
                 wait=lambda _:self.fail('legacy no-hold worker unexpectedly waits'))
        self.assertEqual(default,returned)
        self.assertEqual([],updates)

    def test_hold_and_higher_generation_release_only_at_next_source_boundary(self):
        old=self.write('hold',1)
        updates=[];sleeps=[]
        def clock_sleep(_):
            sleeps.append('idle-not-a-new-source')
            if len(sleeps)==3:
                self.write('release',2)
        result=hold_at_source_boundary(marker=self.control,worker_label='coder',
                     worker_pid=44,prior_generation=0,prior_digest=None,
                     update_status=lambda **kw:updates.append(kw),
                     should_stop=lambda:False,wait=clock_sleep,heartbeat_seconds=0.05)
        self.assertEqual(2,result.generation)
        self.assertEqual('release',result.action)
        self.assertEqual(['publisher_revision_quiesced']*3+
             ['publisher_revision_released'],[x['phase'] for x in updates])
        self.assertTrue(all(x['current_original'] is None for x in updates))
        self.assertTrue(all(x['quiesce_manifest_sha256']==old for x in updates[:3]))
        self.assertEqual(2,updates[-1]['quiesce_generation'])
        self.assertEqual(3,len(sleeps))

    def test_stop_while_held_does_not_start_next_source(self):
        digest=self.write('hold',1)
        ack=[]
        result=hold_at_source_boundary(marker=self.control,worker_label='q4',
               worker_pid=3,prior_generation=0,prior_digest=None,
               update_status=lambda **kw:ack.append(kw),should_stop=lambda:True,
               wait=lambda _:self.fail('stop should return at source boundary'))
        self.assertEqual(SourceBoundaryQuiescence(1,'hold',digest),result)
        self.assertEqual('publisher_revision_quiesced',ack[0]['phase'])
        self.assertEqual(1,len(ack))

    def test_release_same_generation_mutation_or_delete_rejected(self):
        first_sha=self.write('hold',3)
        first=read_control(self.control)
        self.assertEqual(first_sha,first.digest)
        self.write('release',3)
        with self.assertRaisesRegex(ValueError,'rewrite'):
            read_control(self.control,prior_generation=3,prior_digest=first_sha)
        self.write('release',4)
        with self.assertRaisesRegex(ValueError,'replay'):
            read_control(self.control,prior_generation=5,prior_digest=first_sha)
        self.control.unlink()
        with self.assertRaisesRegex(ValueError,'removed'):
            read_control(self.control,prior_generation=3,prior_digest=first_sha)

    def test_symlinks_wrong_owner_permissions_and_malformed_generations_refused(self):
        valid=marker_bytes('hold',1)
        self.control.write_bytes(valid);self.control.chmod(0o644)
        with self.assertRaisesRegex(ValueError,'owner-only'):
            read_control(self.control)
        self.control.chmod(0o600)
        self.assertEqual(1,read_control(self.control).generation)
        self.control.unlink()
        target=self.root/'valid.json';target.write_bytes(valid);target.chmod(0o600)
        self.control.symlink_to(target)
        with self.assertRaises(OSError):read_control(self.control)
        self.control.unlink()
        self.control.write_bytes(b' '*(2050));self.control.chmod(0o600)
        with self.assertRaisesRegex(ValueError,'owner-only'):
            read_control(self.control)
        for data in ({'action':'hold'}, {'schema':SCHEMA,'scope':SCOPE,'generation':True,'action':'hold'},
                     {'schema':SCHEMA,'scope':SCOPE,'generation':1,'action':'unknown'},
                     {'schema':SCHEMA,'scope':SCOPE,'generation':1,'action':'hold','unexpected':1}):
            self.control.write_text(json.dumps(data));self.control.chmod(0o600)
            with self.assertRaises(ValueError):read_control(self.control)
        with self.assertRaises(ValueError):marker_bytes('pause',1)
        with self.assertRaises(ValueError):marker_bytes('hold',True)

    def test_both_workers_genuine_live_pid_heartbeat_and_ack_required(self):
        sha=self.write('hold',8)
        ctrl=read_control(self.control)
        now=datetime.now(timezone.utc)
        def state(pid):
            return {'worker_pid':pid,'quiesce_worker_pid':pid,
                'quiesce_protocol':SCHEMA,'quiesce_scope':SCOPE,
                'quiesce_generation':8,'quiesce_manifest_sha256':sha,
                'phase':'publisher_revision_quiesced','current_original':None,
                'started_utc':'2026-11-22T17:00:00+00:00',
                'quiesce_runtime_started_utc':'2026-11-22T17:00:00+00:00',
                'updated_utc':now.isoformat(),'quiesce_ack_utc':now.isoformat()}
        statuses={'coder':state(101),'q4':state(102)}
        def pid(unit):return 101 if 'q4' not in unit else 102
        def cmdline(p):
            return (b'/usr/bin/python\0raspi_hierarchical_worker.py\0--shard-count\x004\x00'
                  + (b'--worker-label\0q4\0' if p==102 else b''))
        result=read_only_worker_acknowledgements(control=ctrl,states=statuses,
                   current_pid=pid,current_cmdline=cmdline,now_utc=now)
        self.assertTrue(result['both_resident_summary_workers_paused_at_source_boundaries'])
        self.assertFalse(result['safe_to_cutover_live'])
        mutated={k:dict(v) for k,v in statuses.items()}
        for overrides in (
            {'worker_pid':999}, {'quiesce_generation':7},
            {'quiesce_manifest_sha256':'f'*64},{'phase':'embedding'},
            {'current_original':'source/in_progress.pdf'},
            {'quiesce_runtime_started_utc':'2026-11-22T15:00:00+00:00'},
            {'updated_utc':(now-timedelta(seconds=40)).isoformat()},
        ):
            with self.subTest(bad=overrides):
                mutated['q4']={**statuses['q4'],**overrides}
                with self.assertRaises(ValueError):
                    read_only_worker_acknowledgements(control=ctrl,states=mutated,
                        current_pid=pid,current_cmdline=cmdline,now_utc=now)
        with self.assertRaisesRegex(ValueError,'both'):
            read_only_worker_acknowledgements(control=ctrl,states={'coder':statuses['coder']},
                        current_pid=pid,current_cmdline=cmdline,now_utc=now)
        with self.assertRaises(ValueError):
            read_only_worker_acknowledgements(control=ctrl,states=statuses,
                current_pid=lambda _:0,current_cmdline=lambda _:self.fail('invalid PID cannot read /proc'),now_utc=now)

if __name__=='__main__':unittest.main()
