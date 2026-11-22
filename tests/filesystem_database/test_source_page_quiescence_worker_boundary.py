"""Real worker loop synthetic original boundary test. No services/models/DB touched."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from file_system_catalog.source_page_quiescence import hold_at_source_boundary,marker_bytes


def load_worker():
    path=Path(__file__).resolve().parents[2]/'scripts/raspi_hierarchical_worker.py'
    spec=importlib.util.spec_from_file_location('private_reindex_boundary_worker',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RealLoopPrivateQuiescenceTests(unittest.TestCase):
    def synthetic_run(self,*,release:bool):
        worker=load_worker()
        tmp=TemporaryDirectory();self.addCleanup(tmp.cleanup)
        root=Path(tmp.name)
        paths={
            'RUNROOT':root,
            'STATE':root/'worker-state.json',
            'LOCK':root/'.worker.lock',
            'FILEDB':root/'native.sqlite3',
            'SOURCE_MAP':root/'source-map.sqlite3',
            'CLI':root/'fake_cli.py',
            'ALLOCATION_MANIFEST':root/'old-allocation.json',
            'PUBLISHER_QUIESCE_MANIFEST':root/'new-hold.json',
        }
        for key in ('FILEDB','SOURCE_MAP','CLI'):
            paths[key].touch()
        tree=root/'existing_tree.json';tree.write_text('{}')
        candidates=[f'originals/harness-{i}.pdf' for i in range(1000)
            if worker.shard_for_original(f'originals/harness-{i}.pdf',4)==1][:2]
        self.assertEqual(2,len(candidates))
        records=[(p,'a'*64,'b'*64,70000) for p in candidates]
        events=[];actual_status=worker.state_update
        def recorder(**kw):
            actual_status(**kw)
            if kw.get('phase') in ('publisher_revision_quiesced',
                                    'publisher_revision_released','summarizing'):
                events.append((kw['phase'],kw.get('current_original')))
        def save_control(action,generation):
            file=paths['PUBLISHER_QUIESCE_MANIFEST']
            next_file=file.with_suffix('.next')
            next_file.write_bytes(marker_bytes(action,generation))
            next_file.chmod(0o600)
            os.replace(next_file,file)
        def import_no_action(parts,label,path):
            self.assertEqual('EMBED_AND_STORE',label)
            events.append(('simulated_import',path))
            if path==candidates[0]:
                save_control('hold',1)
            elif path==candidates[1]:
                self.assertTrue(release)
                worker.stop=True
            else:raise AssertionError('unknown source in synthetic work')
            return 'CANONICAL_KMD_UPDATED no-action'
        fake_summary={'eligible_text_characters':140000,'indexed_text_characters':0,
                      'complete_originals':0,'total_eligible_originals':2,'text_percentage':0.0}
        wait_count=[]
        def clock(_duration):
            wait_count.append('hold_observed')
            self.assertNotIn(('simulated_import',candidates[1]),events,
                             'second source started before hold was acknowledged')
            if release:
                save_control('release',2)
            else:
                worker.stop=True
        def injected_hold(**kw):
            return hold_at_source_boundary(**kw,wait=clock,heartbeat_seconds=0.001)
        worker.stop=False
        self.addCleanup(setattr,worker,'stop',False)
        fake_cli=SimpleNamespace(payload_path=lambda *args:tree)
        with mock.patch.multiple(worker,**paths), \
             mock.patch.object(worker,'state_update',side_effect=recorder), \
             mock.patch.object(worker,'hold_at_source_boundary',side_effect=injected_hold), \
             mock.patch.object(worker,'scope',return_value=(records,set())), \
             mock.patch.object(worker,'balanced_backlog',side_effect=lambda rows,count:iter(rows)), \
             mock.patch.object(worker,'summary_progress_metrics',return_value=fake_summary), \
             mock.patch.object(worker,'run_command',side_effect=import_no_action), \
             mock.patch.object(worker.signal,'signal'), \
             mock.patch.dict(sys.modules,{'raspi_hierarchical_library':fake_cli}):
            worker.run(sleep_seconds=0,max_calls=2,once=False,dry_run=False,
                       model_url='http://10.8.0.7:14833',model_name='no-action-fake-model',
                       output_tokens=800,shard_count=4,shard_index=1,
                       assigned_shards='1,2,3')
        state=json.loads(paths['STATE'].read_text())
        self.assertEqual('stopped',state['phase'])
        self.assertEqual(1,len(wait_count))
        self.assertEqual(candidates[1] if release else candidates[0],
                         state['last_completed_original'])
        self.assertEqual([
            ('summarizing',candidates[0]),('simulated_import',candidates[0]),
            ('publisher_revision_quiesced',None),
            *([('publisher_revision_released',None),('summarizing',candidates[1]),
               ('simulated_import',candidates[1])] if release else [])
        ],events)
        self.assertEqual(2 if release else 1,state['quiesce_generation'])
        self.assertEqual(state['quiesce_runtime_started_utc'],state['started_utc'])
        return events

    def test_held_worker_finishes_import_before_ack_then_release_allows_next_source(self):
        self.synthetic_run(release=True)

    def test_hold_during_import_then_requested_stop_never_starts_second_source(self):
        self.synthetic_run(release=False)

    def test_deleted_manifest_during_hold_never_forgets_in_memory_generation(self):
        worker=load_worker()
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name)
        paths={'RUNROOT':root,'STATE':root/'worker-status.json',
               'LOCK':root/'.worker.lock','FILEDB':root/'native.sqlite3',
               'SOURCE_MAP':root/'source-map.sqlite3','CLI':root/'fake-cli.py',
               'ALLOCATION_MANIFEST':root/'shards.json',
               'PUBLISHER_QUIESCE_MANIFEST':root/'owner-pause.json'}
        for name in ('FILEDB','SOURCE_MAP','CLI'):paths[name].touch()
        summary=root/'prebuilt.json';summary.write_text('{}')
        candidates=[f'books/hold-{i}.pdf' for i in range(300)
            if worker.shard_for_original(f'books/hold-{i}.pdf',4)==1][:2]
        records=[(path,'a'*64,'b'*64,40000) for path in candidates]
        calls=[];errors=[];old_update=worker.state_update;idle=[]
        def recorder(**fields):
            old_update(**fields)
            if fields.get('phase')=='error_recovering':
                errors.append(fields.get('last_error'))
        def no_action_import(args,label,path):
            calls.append((label,path))
            self.assertEqual((label,path),('EMBED_AND_STORE',candidates[0]))
            control=paths['PUBLISHER_QUIESCE_MANIFEST']
            control.write_bytes(marker_bytes('hold',1));control.chmod(0o600)
            return 'CANONICAL_KMD_UPDATED fake'
        def unlink_during_heartbeat(_):
            paths['PUBLISHER_QUIESCE_MANIFEST'].unlink()
        def holder(**kwargs):
            return hold_at_source_boundary(**kwargs,wait=unlink_during_heartbeat,
                                           heartbeat_seconds=0.01)
        def idle_sleep(_):
            idle.append('outer-error-retry')
            # Let one complete additional outer loop prove that deletion after
            # an ACK has not reset prior_generation to zero.
            if len(idle)==2:
                worker.stop=True
        summary_metrics={'eligible_text_characters':80000,'indexed_text_characters':0,
                         'complete_originals':0,'total_eligible_originals':2,'text_percentage':0.0}
        worker.stop=False
        self.addCleanup(setattr,worker,'stop',False)
        fake_cli=SimpleNamespace(payload_path=lambda *args:summary)
        with mock.patch.multiple(worker,**paths),\
             mock.patch.object(worker,'state_update',side_effect=recorder),\
             mock.patch.object(worker,'hold_at_source_boundary',side_effect=holder),\
             mock.patch.object(worker,'scope',return_value=(records,set())),\
             mock.patch.object(worker,'balanced_backlog',side_effect=lambda rows,count:iter(rows)),\
             mock.patch.object(worker,'summary_progress_metrics',return_value=summary_metrics),\
             mock.patch.object(worker,'run_command',side_effect=no_action_import),\
             mock.patch.object(worker.signal,'signal'),\
             mock.patch.object(worker.time,'sleep',side_effect=idle_sleep),\
             mock.patch.dict(sys.modules,{'raspi_hierarchical_library':fake_cli}):
            worker.run(sleep_seconds=0,max_calls=2,once=False,dry_run=False,
                       model_url='http://10.8.0.7:14833',model_name='fake-only',
                       output_tokens=800,shard_count=4,shard_index=1,
                       assigned_shards='1,2,3')
        self.assertEqual([('EMBED_AND_STORE',candidates[0])],calls)
        self.assertEqual(2,len(errors))
        self.assertTrue(all('removed' in error for error in errors))
        self.assertEqual(2,len(idle))
        status=json.loads(paths['STATE'].read_text())
        self.assertEqual('stopped',status['phase'])
        self.assertEqual(1,status['quiesce_generation'])
        self.assertIsInstance(status['quiesce_manifest_sha256'],str)
        self.assertEqual(64,len(status['quiesce_manifest_sha256']))

    def test_worker_restart_refuses_missing_previously_seen_control_manifest(self):
        worker=load_worker()
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name)
        paths={'RUNROOT':root,'STATE':root/'worker-status.json',
               'LOCK':root/'.worker.lock','FILEDB':root/'native.sqlite3',
               'SOURCE_MAP':root/'source-map.sqlite3','CLI':root/'fake-cli.py',
               'ALLOCATION_MANIFEST':root/'shards.json',
               'PUBLISHER_QUIESCE_MANIFEST':root/'deleted-hold.json'}
        for f in ('FILEDB','SOURCE_MAP','CLI'):
            paths[f].touch()
        saved={'quiesce_generation':2,'quiesce_manifest_sha256':'a'*64}
        original=json.dumps(saved)
        paths['STATE'].write_text(original)
        worker.stop=False
        self.addCleanup(setattr,worker,'stop',False)
        with mock.patch.multiple(worker,**paths),\
             mock.patch.object(worker.signal,'signal'),\
             mock.patch.object(worker,'run_command',side_effect=AssertionError('no source may start')):
            with self.assertRaisesRegex(ValueError,'manifest removed'):
                worker.run(sleep_seconds=0,max_calls=1,once=True,dry_run=True,
                           model_url='http://10.8.0.7:14833',model_name='test-model',
                           output_tokens=800,shard_count=4,shard_index=1,
                           assigned_shards='1,2,3')
        self.assertEqual(original,paths['STATE'].read_text())
        self.assertFalse(paths['PUBLISHER_QUIESCE_MANIFEST'].exists())

if __name__=='__main__':unittest.main()
