"""A staged allocation applies ONLY after the current source import completes.

This test invokes the actual worker loop under fully isolated fake source,
status, lock, command and FileDB objects. It never calls a model or systemd.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from file_system_catalog.source_shard_runtime import write_manifest_bytes


def load_worker():
    path=Path(__file__).resolve().parents[2]/'scripts/raspi_hierarchical_worker.py'
    spec=importlib.util.spec_from_file_location('isolated_summary_worker',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkerBoundaryHandoffTests(unittest.TestCase):
    def test_restart_never_reacquires_bucket_three_from_legacy_cli_without_ack(self):
        worker=load_worker()
        from file_system_catalog.source_shard_runtime import Assigned
        tmp=TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root=Path(tmp.name)
        (root/'hierarchical-worker-q4-status.json').write_text('{}')
        (root/'hierarchical-worker-status.json').write_text('{}')
        coder_old=frozenset((1,2,3))  # Actual installed Coder CLI at every restart.
        coder_rollback=Assigned(frozenset((1,2,3)),5,'b'*64,'baseline')
        with (mock.patch.object(worker,'RUNROOT',root),
              mock.patch.object(worker,'verify_releasing_q4_acknowledged',
                                side_effect=ValueError('Q4 release not acknowledged')) as q4_guard):
            with self.assertRaisesRegex(ValueError,'Q4 release not acknowledged'):
                worker.verify_source_shard_acquisition(
                    coder_rollback,coder_old,'coder',startup=True)
            q4_guard.assert_called_once()
        q4_initial=frozenset((0,))
        q4_acquire=Assigned(frozenset((0,3)),3,'c'*64,'balanced')
        with (mock.patch.object(worker,'RUNROOT',root),
              mock.patch.object(worker,'verify_releasing_worker_acknowledged',
                                side_effect=ValueError('Coder release not acknowledged')) as coder_guard):
            with self.assertRaisesRegex(ValueError,'Coder release not acknowledged'):
                worker.verify_source_shard_acquisition(
                    q4_acquire,q4_initial,'q4',startup=True)
            coder_guard.assert_called_once()
        # The real installed baseline before any manifest is still usable.
        self.assertIsNone(worker.verify_source_shard_acquisition(
            Assigned(coder_old,0,None,None),coder_old,'coder',startup=True))
        self.assertIsNone(worker.verify_source_shard_acquisition(
            Assigned(coder_old,1,'d'*64,'baseline'),coder_old,'coder',startup=True))

    def test_release_is_applied_after_current_import_and_before_next_source(self):
        worker=load_worker()
        tmp=TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root=Path(tmp.name)
        params={
            'RUNROOT':root,
            'STATE':root/'worker-status.json',
            'LOCK':root/'.worker.lock',
            'FILEDB':root/'native.sqlite3',
            'SOURCE_MAP':root/'source-map.sqlite3',
            'CLI':root/'fake_cli.py',
            'ALLOCATION_MANIFEST':root/'handoff.json',
        }
        for f in ('FILEDB','SOURCE_MAP','CLI'):
            params[f].touch()
        tree=root/'already_generated_tree.json'
        tree.write_text('{}')
        names=[]
        for i in range(250):
            name=f'originals/fixture-{i:03d}.pdf'
            if worker.shard_for_original(name,4)==3:
                names.append(name)
                if len(names)==2:
                    break
        self.assertEqual(2,len(names))
        records=[(p,'a'*64,'b'*64,75000) for p in names]
        events=[]
        original_state_update=worker.state_update
        def recorder(**kwargs):
            original_state_update(**kwargs)
            if kwargs.get('phase')=='allocation_applied':
                events.append(('boundary_ack',kwargs['assigned_shards']))
                worker.stop=True
        def simulated_import(parts,label,path):
            events.append((label,path))
            self.assertEqual(names[0],path,'must not start a second 3-bucket source after release')
            if label!='EMBED_AND_STORE':
                raise AssertionError('cached source tree should skip generation')
            # The manifest changes *during* first source import. The worker
            # must finish that whole import before acknowledging the release.
            file=params['ALLOCATION_MANIFEST']
            file.write_bytes(write_manifest_bytes('release_three',1))
            file.chmod(0o600)
            return 'CANONICAL_KMD_UPDATED'
        fake_summary={'eligible_text_characters':150000,'indexed_text_characters':0,'complete_originals':0,
                      'total_eligible_originals':2,'text_percentage':0.0}
        worker.stop=False
        self.addCleanup(setattr,worker,'stop',False)
        fake_cli=SimpleNamespace(payload_path=lambda *args:tree)
        with mock.patch.multiple(worker,**params),\
             mock.patch.object(worker,'state_update',side_effect=recorder),\
             mock.patch.object(worker,'scope',return_value=(records,set())),\
             mock.patch.object(worker,'balanced_backlog',side_effect=lambda rows,count:iter(rows)),\
             mock.patch.object(worker,'summary_progress_metrics',return_value=fake_summary),\
             mock.patch.object(worker,'run_command',side_effect=simulated_import),\
             mock.patch.object(worker.signal,'signal'),\
             mock.patch.object(worker.time,'sleep',side_effect=AssertionError('unexpected idle sleep')) ,\
             mock.patch.dict(sys.modules,{'raspi_hierarchical_library':fake_cli}):
            worker.run(sleep_seconds=0,max_calls=2,once=False,dry_run=False,
                       model_url='http://10.8.0.7:14833',model_name='test-model',
                       output_tokens=800,shard_count=4,shard_index=1,
                       assigned_shards='1,2,3')
        self.assertEqual([('EMBED_AND_STORE',names[0]),('boundary_ack',[1,2])],events)
        state=json.loads(params['STATE'].read_text())
        self.assertEqual([1,2],state['assigned_shards'])
        self.assertEqual(1,state['allocation_generation'])
        self.assertEqual('release_three',state['allocation_stage'])
        self.assertEqual(names[0],state['last_completed_original'])
        self.assertEqual('stopped',state['phase'])
        self.assertIsNone(state['current_original'])


if __name__=='__main__':
    unittest.main()
