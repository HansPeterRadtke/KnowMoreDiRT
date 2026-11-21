"""Strict journal and completed-source receipt preconditions for read-only audit."""
from __future__ import annotations

import hashlib
import importlib.util
import json
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
