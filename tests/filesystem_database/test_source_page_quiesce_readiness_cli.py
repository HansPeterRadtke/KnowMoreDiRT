"""Read-only pause status must never mark a live PDF migration safe."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from file_system_catalog.source_page_quiescence import SourceBoundaryQuiescence,SCHEMA,SCOPE


def load_cli():
    path=Path(__file__).resolve().parents[2]/'scripts/raspi_publisher_pdf_quiesce_readiness.py'
    spec=importlib.util.spec_from_file_location('isolated_pause_readiness_cli',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReadOnlyPauseStatusTests(unittest.TestCase):
    def setUp(self):
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.cli=load_cli()
        self.statuses={label:self.root/(label+'.json') for label in ('coder','q4')}
        self.marker=self.root/'optional-control.json'
        for label,path in self.statuses.items():
            payload={'phase':'summarizing','current_original':'publisher/current_source.pdf',
                     'worker_pid':101 if label=='coder' else 102,
                     'started_utc':'2026-11-08T09:00:00+00:00',
                     'allocation_protocol':None,'updated_utc':'2026-11-22T18:00:00+00:00'}
            path.write_text(json.dumps(payload));path.chmod(0o600)

    def check_no_effect(self,control):
        from contextlib import redirect_stdout
        from io import StringIO
        output=StringIO()
        with mock.patch.object(self.cli,'MARKER',self.marker),\
             mock.patch.object(self.cli,'WORKERS',self.statuses),\
             mock.patch.object(self.cli,'read_control',return_value=control),\
             mock.patch.object(self.cli,'current_main_pid',side_effect=lambda unit:102 if '-q4' in unit else 101),\
             mock.patch.object(self.cli.sys,'argv',['readiness.py']),\
             redirect_stdout(output):
            self.cli.main()
        return json.loads(output.getvalue())

    def test_unrequested_pause_does_not_infer_old_workers_quiescent(self):
        data=self.check_no_effect(SourceBoundaryQuiescence(0,'release',None))
        self.assertFalse(data['both_new_code_workers_confirmed'])
        self.assertFalse(data['both_workers_quiesced_at_complete_original_boundary'])
        self.assertFalse(data['safe_to_cutover_live'])
        self.assertFalse(self.marker.exists())

    def test_held_manifest_without_acknowledgements_still_refused(self):
        data=self.check_no_effect(SourceBoundaryQuiescence(3,'hold','a'*64))
        self.assertFalse(data['both_workers_quiesced_at_complete_original_boundary'])
        self.assertTrue(data['hold_acknowledgements_refused'])
        self.assertFalse(data['safe_to_cutover_live'])
        self.assertFalse(self.marker.exists())

    def test_release_during_worker_ack_check_does_not_claim_workers_held(self):
        from contextlib import redirect_stdout
        from io import StringIO
        captured=StringIO()
        held=SourceBoundaryQuiescence(5,'hold','a'*64)
        released=SourceBoundaryQuiescence(6,'release','b'*64)
        with mock.patch.object(self.cli,'MARKER',self.marker),\
             mock.patch.object(self.cli,'WORKERS',self.statuses),\
             mock.patch.object(self.cli,'read_control',side_effect=[held,released]),\
             mock.patch.object(self.cli,'read_only_worker_acknowledgements',
                   return_value={'both_resident_summary_workers_paused_at_source_boundaries':True}),\
             mock.patch.object(self.cli,'current_main_pid',side_effect=lambda unit:102 if '-q4' in unit else 101),\
             mock.patch.object(self.cli.sys,'argv',['readiness.py']),\
             redirect_stdout(captured):
            self.cli.main()
        data=json.loads(captured.getvalue())
        self.assertFalse(data['both_workers_quiesced_at_complete_original_boundary'])
        self.assertFalse(data['safe_to_cutover_live'])
        self.assertIn('advanced',data['hold_acknowledgements_refused'])
        self.assertFalse(self.marker.exists())

    def test_does_not_accept_apply_or_mutation_arguments(self):
        with mock.patch.object(self.cli.sys,'argv',['readiness.py','--apply']):
            with self.assertRaisesRegex(ValueError,'no action flags'):
                self.cli.main()
        self.assertFalse(self.marker.exists())

    def test_private_status_reader_rejects_symlink_and_world_readable_data(self):
        path=self.statuses['coder']
        path.chmod(0o644)
        with self.assertRaisesRegex(ValueError,'untrusted'):
            self.cli.read_private_worker_status(path)
        path.chmod(0o600)
        self.assertEqual('summarizing',self.cli.read_private_worker_status(path)['phase'])
        symlink=self.root/'status-link.json';symlink.symlink_to(path)
        with self.assertRaisesRegex(ValueError,'symlink'):
            self.cli.read_private_worker_status(symlink)

if __name__=='__main__':unittest.main()
