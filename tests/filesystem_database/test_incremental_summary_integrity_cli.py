"""CLI no-loss original-completion cursor, provenance and fail-closed tests.

Uses only isolated disposable SQLite databases and deterministic synthetic
journal events. The real worker journal and real canonical native FileDB
are not opened by this test.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase,mock
import unittest

from file_system_catalog.incremental_source_completion import CompletionEvent


def load_cli():
    script=Path(__file__).resolve().parents[2]/'scripts/raspi_incremental_summary_integrity.py'
    spec=importlib.util.spec_from_file_location('kmd_incremental_source_cli_test',script)
    module=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def event(i:int,path:str)->CompletionEvent:
    return CompletionEvent(
        cursor=f's=aa123456;event={i:05x};time={i:08x}',
        recorded_utc='2026-11-28T11:00:00+00:00',
        journal_timestamp_us=1795863600000000+i,
        source_path=path,
    )


class IncrementalNativeOriginalJournalCursorTests(TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.root.chmod(0o700)
        self.org=self.root/'publisher'
        self.der=self.root/'derived'
        self.org.mkdir(mode=0o700)
        self.der.mkdir(mode=0o700)
        self.source=self.root/'source.sqlite3'
        self.native=self.root/'native.sqlite3'
        self.ledger=self.root/'approved.json'
        for name in (self.source,self.native):
            with sqlite3.connect(name) as db:
                db.execute('CREATE TABLE original_anchor(path TEXT)')
        self.ledger.write_text(json.dumps({'items':[]}))
        self.cursor=self.root/'source-completions-cursor.json'
        self.mod=load_cli()
        self.mock_receipt=mock.patch.object(self.mod,'ledger_receipts',
                                            return_value=({'books/a.pdf':'a'*64,'books/b.pdf':'b'*64},'l'*64))
        self.mock_receipt.start()
        self.addCleanup(self.mock_receipt.stop)

    def cli(self,file:str,*,first:bool=False,events:list[CompletionEvent],
            audit_failure:str|None=None):
        out=self.root/file
        args=['raspi_incremental_summary_integrity.py',
              '--source-map',str(self.source),'--native-db',str(self.native),
              '--original-root',str(self.org),'--derived-root',str(self.der),
              '--completed-ledger',str(self.ledger),
              '--cursor-file',str(self.cursor),'--output',str(out),
              '--batch-size','3','--max-original-mib','64']
        if first:args.extend(['--initial-since','2026-11-28 10:40:00'])
        def audit(**kwargs):
            path=kwargs['original_path']
            if path==audit_failure:
                raise ValueError('the current original source checksum changed')
            return {'original_path':path,
                'current_native_summary_nodes_fully_verified':3,
                'publisher_original_sha256_reverified':'a'*64,
                'derived_original_sha256_reverified':'b'*64,
                'all_original_text_covered_by_recursive_summary_graph':True}
        with mock.patch.object(self.mod,'journal_page',return_value=events) as page,\
             mock.patch.object(self.mod,'audit_current_source_tree',side_effect=audit),\
             mock.patch.object(sys,'argv',args),\
             mock.patch('builtins.print'):
            if audit_failure:
                with self.assertRaises(SystemExit) as ctx:self.mod.main()
                self.assertEqual(2,ctx.exception.code)
            else:
                self.mod.main()
        return json.loads(out.read_bytes()),page.call_args

    def test_two_pages_never_skip_or_replay_previous_original_completion(self):
        a=event(1,'books/a.pdf');b=event(2,'books/b.pdf');c=event(3,'books/c.pdf')
        first,args=self.cli('page1.json',first=True,events=[a,b])
        self.assertTrue(first['all_selected_originals_verified'])
        self.assertTrue(first['cursor_advanced'])
        self.assertEqual(2,first['selected_completion_events'])
        self.assertEqual(6,first['native_summary_nodes_physically_verified'])
        self.assertEqual('2026-11-28 10:40:00',args.kwargs['initial_since'])
        current=json.loads(self.cursor.read_bytes())
        self.assertEqual(b.cursor,current['last_successful_event_cursor'])
        self.assertEqual(2,current['total_events_audited'])
        second,args=self.cli('page2.json',events=[c])
        self.assertEqual(b.cursor,args.kwargs['previous']['last_successful_event_cursor'])
        self.assertEqual(b.cursor,second['previous_cursor'])
        self.assertEqual(c.cursor,second['last_event_cursor'])
        self.assertEqual(1,second['selected_completion_events'])
        self.assertEqual(3,json.loads(self.cursor.read_bytes())['total_events_audited'])
        self.assertEqual(3,second['native_summary_nodes_physically_verified'])
        self.assertFalse(set(z['source_event_cursor'] for z in first['checks']) &
                         set(z['source_event_cursor'] for z in second['checks']))

    def test_partial_page_failure_never_advances_provenance_cursor(self):
        a=event(1,'books/a.pdf');b=event(2,'books/b.pdf')
        self.cli('page1.json',first=True,events=[a])
        before=self.cursor.read_bytes()
        failed,args=self.cli('failed-page.json',events=[b],audit_failure='books/b.pdf')
        self.assertEqual(a.cursor,args.kwargs['previous']['last_successful_event_cursor'])
        self.assertEqual(1,failed['selected_completion_events'])
        self.assertEqual(0,failed['new_complete_originals_physically_verified'])
        self.assertFalse(failed['cursor_advanced'])
        self.assertFalse(failed['all_selected_originals_verified'])
        self.assertEqual(before,self.cursor.read_bytes())
        self.assertIn('source checksum changed',failed['checks'][0]['reason'])

    def test_empty_source_completion_page_keeps_cursor_and_issues_no_fake_success(self):
        a=event(1,'books/a.pdf')
        self.cli('page1.json',first=True,events=[a])
        old=self.cursor.read_bytes()
        result,args=self.cli('empty-page.json',events=[])
        self.assertTrue(result['no_new_original_completions_observed'])
        self.assertEqual(0,result['selected_completion_events'])
        self.assertFalse(result['cursor_advanced'])
        self.assertFalse(result['all_selected_originals_verified'])
        self.assertEqual(old,self.cursor.read_bytes())
        self.assertEqual(a.cursor,args.kwargs['previous']['last_successful_event_cursor'])

    def test_tampered_previous_receipt_does_not_resync_by_guessing(self):
        a=event(1,'books/a.pdf')
        self.cli('page1.json',first=True,events=[a])
        prior=self.root/'page1.json'
        prior.write_bytes(prior.read_bytes()+b'\nmodified')
        before=self.cursor.read_bytes()
        with mock.patch.object(self.mod,'journal_page') as journal:
            with self.assertRaisesRegex(ValueError,'previous original source audit receipt digest changed'):
                self.cli('should-not-create.json',events=[event(2,'books/b.pdf')])
            journal.assert_not_called()
        self.assertFalse((self.root/'should-not-create.json').exists())
        self.assertEqual(before,self.cursor.read_bytes())

    def test_two_concurrent_owner_auditors_fail_closed_and_preserve_cursor(self):
        a=event(1,'books/a.pdf')
        with self.mod.exclusive_private_cursor_auditor(self.cursor):
            with self.assertRaisesRegex(ValueError,'another original-source cursor audit is active'):
                self.cli('must-not-exist.json',first=True,events=[a])
            self.assertFalse((self.root/'must-not-exist.json').exists())
            self.assertFalse(self.cursor.exists())
        result,_=self.cli('page1.json',first=True,events=[a])
        self.assertTrue(result['cursor_advanced'])
        old=self.cursor.read_bytes()
        with self.mod.exclusive_private_cursor_auditor(self.cursor):
            with self.assertRaisesRegex(ValueError,'another original-source cursor audit is active'):
                self.cli('must-not-exist2.json',events=[event(2,'books/b.pdf')])
        self.assertEqual(old,self.cursor.read_bytes())

    def test_corrupt_private_cursor_lock_permissions_fail_closed(self):
        lock=self.cursor.with_name('.'+self.cursor.name+'.audit-lock')
        lock.write_text('owned')
        lock.chmod(0o644)
        with self.assertRaisesRegex(ValueError,'owner-private original cursor audit lock is unsafe'):
            self.cli('must-not-be-published.json',first=True,events=[event(1,'books/a.pdf')])
        self.assertFalse((self.root/'must-not-be-published.json').exists())
        self.assertFalse(self.cursor.exists())

    def test_missing_or_wrong_journald_cursor_cannot_be_mistaken_for_no_new_events(self):
        """A vacuumed journal cursor must not silently produce an empty page."""
        prior=event(1,'books/a.pdf')
        another=event(2,'books/b.pdf')
        def json_line(e):
            return json.dumps({'__CURSOR':e.cursor,
                '__REALTIME_TIMESTAMP':str(e.journal_timestamp_us),
                'MESSAGE':'KMD_HIERARCHICAL_DOCUMENT_COMPLETE '+
                    e.recorded_utc+' '+e.source_path})+'\n'
        class DoneProcess:
            def __init__(self,body):
                self.stdout=io.StringIO(body)
                self.stderr=io.StringIO('')
            def poll(self):return 0
            def wait(self,timeout=None):return 0
        for response,should_fail in (
            (json_line(prior),False),
            (json_line(another),True),
            ('',True),
        ):
            with self.subTest(response=response[:26]):
                with mock.patch.object(self.mod.subprocess,'Popen',
                    side_effect=lambda *args,**kwargs: DoneProcess(response)) as child:
                    if should_fail:
                        with self.assertRaisesRegex(ValueError,'cursor'):
                            self.mod.confirm_retained_journal_cursor(
                                {'last_successful_event_cursor':prior.cursor})
                    else:
                        self.mod.confirm_retained_journal_cursor(
                            {'last_successful_event_cursor':prior.cursor})
                    args=child.call_args.args[0]
                    self.assertTrue(any(z=='--cursor='+prior.cursor for z in args))
                    self.assertFalse(any(z.startswith('--after-cursor=') for z in args))

    def test_public_cursor_or_modified_audit_permissions_are_rejected(self):
        a=event(1,'books/a.pdf')
        self.cli('page1.json',first=True,events=[a])
        self.cursor.chmod(0o644)
        with self.assertRaisesRegex(ValueError,'source cursor is unsafe'):
            self.cli('must-not-create.json',events=[event(2,'books/b.pdf')])
        self.assertFalse((self.root/'must-not-create.json').exists())
        self.cursor.chmod(0o600)
        prior=self.root/'page1.json'
        prior.chmod(0o644)
        with self.assertRaisesRegex(ValueError,'previous private original source audit'):
            self.cli('must-not-create2.json',events=[event(2,'books/b.pdf')])
        self.assertFalse((self.root/'must-not-create2.json').exists())

    def test_missing_cursor_refuses_implicit_start_and_existing_receipt(self):
        with mock.patch.object(sys,'argv',[
              'raspi_incremental_summary_integrity.py',
              '--source-map',str(self.source),'--native-db',str(self.native),
              '--original-root',str(self.org),'--derived-root',str(self.der),
              '--completed-ledger',str(self.ledger),'--cursor-file',str(self.cursor),
              '--output',str(self.root/'unrecorded.json')
        ]):
            with self.assertRaisesRegex(ValueError,'initial source journal UTC'):
                self.mod.main()
        existing=self.root/'existing.json'
        existing.write_text('{}')
        with mock.patch.object(sys,'argv',[
              'raspi_incremental_summary_integrity.py',
              '--source-map',str(self.source),'--native-db',str(self.native),
              '--original-root',str(self.org),'--derived-root',str(self.der),
              '--completed-ledger',str(self.ledger),'--cursor-file',str(self.cursor),
              '--initial-since','2026-11-28 10:40:00','--output',str(existing)
        ]):
            with self.assertRaisesRegex(ValueError,'existing/unsafe incremental source audit receipt'):
                self.mod.main()


if __name__=='__main__':
    unittest.main()
