"""Strict journal and completed-source receipt preconditions for read-only audit."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def load_cli():
    path=Path(__file__).resolve().parents[2]/'scripts/raspi_recent_summary_integrity.py'
    spec=importlib.util.spec_from_file_location('kmd_recent_integrity_cli',path)
    tool=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


class OriginalReceiptAndJournalSafetyTests(unittest.TestCase):
    def setUp(self):
        self.cli=load_cli()
        tmp=TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp=Path(tmp.name)
        self.ledger=self.tmp/'completed-originals.json'
        self.complete={'status':'complete','path':'references/original-1.html','sha256':'a'*64}
        self.pending={'status':'error','path':'references/visual-only.pdf','sha256':None}

    def write_ledger(self,items):
        self.ledger.write_text(json.dumps({'items':items})+'\n')
        self.ledger.chmod(0o600)

    def test_recent_worker_journal_scope_requires_actual_event_prefix(self):
        lines=[
            'KMD_HIERARCHICAL_DOCUMENT_COMPLETE 2026-11-21T16:00:00+00:00 references/a.html',
            'DEBUG KMD_HIERARCHICAL_DOCUMENT_COMPLETE 2026-11-21T16:00:01+00:00 references/forged.html',
            'KMD_HIERARCHICAL_RESULT COMPLETE_TREE KMD_HIERARCHICAL_DOCUMENT_COMPLETE 2026-11-21T16:00:01+00:00 references/forged.html',
            'KMD_HIERARCHICAL_DOCUMENT_COMPLETE 2026-11-21T16:00:02+00:00 references/b.html',
            'KMD_HIERARCHICAL_DOCUMENT_COMPLETE 2026-11-21T16:00:03+00:00 references/a.html',
            'KMD_HIERARCHICAL_DOCUMENT_COMPLETE 2026-11-21T16:00:04+00:00 references/c.html',
        ]
        result=self.cli.completed_paths_from_lines(lines,limit=2)
        self.assertEqual(['references/a.html','references/c.html'],result)
        self.assertNotIn('references/forged.html',result)
        self.assertEqual(['references/b.html','references/a.html','references/c.html'],
                         self.cli.completed_paths_from_lines(lines,limit=25))

    def test_reject_unbounded_selected_originals_and_invalid_events(self):
        for limit in (0,-3,26,False):
            with self.subTest(limit=limit),self.assertRaises(ValueError):
                self.cli.completed_paths_from_lines([],limit=limit)
        self.assertEqual([],self.cli.completed_paths_from_lines(
            ['KMD_HIERARCHICAL_DOCUMENT_COMPLETE 2026-11-21T16:00:00+00:00',
             'KMD_HIERARCHICAL_DOCUMENT_COMPLETE 2026-11-21T16:00:00+00:00 '+ 'a'*5000],limit=5))

    def test_longest_native_roots_are_ordered_distinct_and_bounded(self):
        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE TABLE filesystem_entries(id INTEGER,relative_path_display TEXT,entry_type TEXT)')
            db.execute('CREATE TABLE content_chunks(chunk_id TEXT,filesystem_entry_id INTEGER,collection_id TEXT,start_char INTEGER,end_char INTEGER)')
            db.execute('CREATE TABLE content_representations(chunk_id TEXT,prompt_version TEXT,analysis_status TEXT,representation_kind TEXT)')
            for name,ident,position in (('paper-a',1,200),('paper-b',2,500),('paper-c',3,70)):
                db.execute('INSERT INTO filesystem_entries VALUES(?,?,?)',(ident,name,'file'))
                db.execute('INSERT INTO content_chunks VALUES(?,?,?,?,?)',(name,ident,'kmd:summary',0,position))
                db.execute('INSERT INTO content_representations VALUES(?,?,?,?)',(
                    name,'kmd-hierarchical-source-summary-v1','complete','summary_long'))
            # Historical shorter root of same exact publisher path is never an extra source.
            db.execute('INSERT INTO content_chunks VALUES(?,?,?,?,?)',('paper-b-old',2,'kmd:summary',0,140))
            db.execute('INSERT INTO content_representations VALUES(?,?,?,?)',(
                'paper-b-old','kmd-hierarchical-source-summary-v1','complete','summary_long'))
            self.assertEqual(['paper-b','paper-a'],self.cli.largest_current_source_roots(db,limit=2))
            self.assertEqual(['paper-b','paper-a','paper-c'],self.cli.largest_current_source_roots(db,limit=25))
            for invalid in (0,26,True):
                with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                    self.cli.largest_current_source_roots(db,limit=invalid)

    def test_no_native_hierarchy_is_not_a_verified_large_source_sample(self):
        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE TABLE filesystem_entries(id INTEGER,relative_path_display TEXT,entry_type TEXT)')
            db.execute('CREATE TABLE content_chunks(chunk_id TEXT,filesystem_entry_id INTEGER,collection_id TEXT,start_char INTEGER,end_char INTEGER)')
            db.execute('CREATE TABLE content_representations(chunk_id TEXT,prompt_version TEXT,analysis_status TEXT,representation_kind TEXT)')
            with self.assertRaisesRegex(ValueError,'no native original-linked'):
                self.cli.largest_current_source_roots(db,limit=1)

    def test_verified_completed_original_ledger_rejects_duplicates_and_bad_sha(self):
        self.write_ledger([self.complete,self.pending])
        got,digest=self.cli.ledger_receipts(self.ledger)
        self.assertEqual({self.complete['path']:'a'*64},got)
        self.assertEqual(hashlib.sha256(self.ledger.read_bytes()).hexdigest(),digest)
        for entries in ([self.complete,self.complete],
                        [{**self.complete,'sha256':'z'*64}],
                        [{**self.complete,'sha256':'x'}],
                        ['untrusted'],):
            with self.subTest(entries=entries),self.assertRaises(ValueError):
                self.write_ledger(entries)
                self.cli.ledger_receipts(self.ledger)
        self.write_ledger([self.complete])
        self.ledger.unlink()
        self.ledger.symlink_to(self.tmp/'unknown-ledger.json')
        with self.assertRaises(ValueError):self.cli.ledger_receipts(self.ledger)


if __name__=='__main__':
    unittest.main()
