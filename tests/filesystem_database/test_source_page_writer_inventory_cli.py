"""Read-only actual systemd unit inventory and private-only receipt admission."""
from __future__ import annotations
from contextlib import redirect_stdout
from io import StringIO
import importlib.util
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from file_system_catalog.source_page_writer_inventory import WRITERS,CATALOG_ONLY


def load_cli():
    path=Path(__file__).resolve().parents[2]/'scripts/raspi_pdf_repair_additional_writer_readiness.py'
    spec=importlib.util.spec_from_file_location('test_only_extra_writer_readiness',path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class NativeSourceWriterReadinessCLITests(unittest.TestCase):
    def setUp(self):
        t=TemporaryDirectory();self.addCleanup(t.cleanup)
        self.root=Path(t.name)
        self.folder=self.root/'private'
        self.folder.mkdir(mode=0o700)
        self.output=self.folder/'new-writer-status.json'
        self.cli=load_cli()
        self.current={'schema':'kmd.publisher_native_full_source_document_boundary_readiness.v1',
            'both_new_code_workers_confirmed':False,
            'both_workers_quiesced_at_complete_original_boundary':False,
            'safe_to_cutover_live':False}
        self.units={u:{'load_state':'loaded','active_state':'inactive','main_pid':0}
                    for values in WRITERS.values() for u in values[:2]}
        self.units.update({u:{'load_state':'loaded','active_state':'active','main_pid':0}
                           for u in CATALOG_ONLY})
        for _,timer,_ in WRITERS.values():self.units[timer]['active_state']='active'

    def fake_subprocess(self,command,**_):
        if command[0]=='systemctl':
            name=command[2];s=self.units[name]
            lines=[f'LoadState={s["load_state"]}',f'ActiveState={s["active_state"]}']
            if name.endswith('.service'):lines.append(f'MainPID={s["main_pid"]}')
            return subprocess.CompletedProcess(command,0,stdout='\n'.join(lines)+'\n',stderr='')
        self.assertEqual(command[0],self.cli.sys.executable)
        return subprocess.CompletedProcess(command,0,stdout=json.dumps(self.current)+'\n',stderr='')

    def test_actual_systemd_timer_without_mainpid_is_valid_not_an_invented_pid(self):
        timer=WRITERS['native_direct_embedding'][1]
        with mock.patch.object(self.cli.subprocess,'run',side_effect=self.fake_subprocess):
            status=self.cli.show_unit(timer)
            self.assertEqual(0,status['main_pid'])
            self.assertEqual('active',status['active_state'])
            with self.assertRaisesRegex(ValueError,'unrecognized'):
                self.cli.show_unit('unrelated-ssh.service')

    def test_missing_mainpid_from_real_service_is_not_accepted(self):
        unit=WRITERS['native_direct_embedding'][0]
        def drop_pid(command,**kwargs):
            response=self.fake_subprocess(command,**kwargs)
            return subprocess.CompletedProcess(command,0,stdout='LoadState=loaded\nActiveState=inactive\n',stderr='')
        with mock.patch.object(self.cli.subprocess,'run',side_effect=drop_pid):
            with self.assertRaisesRegex(ValueError,'incomplete actual'):
                self.cli.show_unit(unit)

    def test_only_private_report_written_when_all_current_writers_observed(self):
        capture=StringIO()
        with mock.patch.object(self.cli,'PRIVATE',self.root), \
             mock.patch.object(self.cli.subprocess,'run',side_effect=self.fake_subprocess), \
             mock.patch.object(self.cli.sys,'argv',['readiness','--new-private-output',str(self.output)]), \
             redirect_stdout(capture):
            self.cli.main()
        summary=json.loads(capture.getvalue())
        status=json.loads(self.output.read_bytes())
        self.assertEqual(2,summary['known_native_writer_services_checked'])
        self.assertFalse(status['safe_to_cutover_live'])
        self.assertIn('native_direct_embedding_scheduled_timer_armed',status['explicit_release_blockers'])
        self.assertIn('semantic_source_extraction_scheduled_timer_armed',status['explicit_release_blockers'])
        self.assertEqual(0o600,self.output.stat().st_mode & 0o777)

    def test_existing_private_receipt_cannot_be_overwritten(self):
        self.output.write_text('original evidence preserved')
        with mock.patch.object(self.cli,'PRIVATE',self.root), \
             mock.patch.object(self.cli.sys,'argv',['readiness','--new-private-output',str(self.output)]):
            with self.assertRaisesRegex(ValueError,'NEW'):
                self.cli.main()
        self.assertEqual('original evidence preserved',self.output.read_text())

    def test_unreadable_independent_summary_readiness_refuses_output(self):
        def failed(command,**kwargs):
            return subprocess.CompletedProcess(command,2,stdout='',stderr='negative')
        with mock.patch.object(self.cli,'PRIVATE',self.root), \
             mock.patch.object(self.cli.subprocess,'run',side_effect=failed), \
             mock.patch.object(self.cli.sys,'argv',['readiness','--new-private-output',str(self.output)]):
            with self.assertRaisesRegex(RuntimeError,'unavailable'):
                self.cli.main()
        self.assertFalse(self.output.exists())

if __name__=='__main__':unittest.main()
