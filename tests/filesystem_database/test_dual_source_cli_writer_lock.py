"""Actual dual-source CLI remains read-only and current during source-map writer commits.

These are isolated disposable SQLite source metadata databases and simulated
already-authenticated native search payloads; no production provider,
download, model inference or native FileDB writes occur.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import hashlib
from importlib import util
import io
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest import mock


def load_cli():
    root=Path(__file__).resolve().parents[2]
    file=root/"scripts/raspi_dual_source_discovery.py"
    spec=util.spec_from_file_location("kmd_dual_source_writer_lock_test_cli",file)
    module=util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha(raw:bytes)->str:
    return hashlib.sha256(raw).hexdigest()


class DualSourceReaderLiveWriterLockTests(unittest.TestCase):
    def setUp(self):
        self.temporary=TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.originals=self.root/"originals"
        self.derivatives=self.root/"derived"
        self.originals.mkdir()
        self.derivatives.mkdir()
        self.name="research/approved-physics.pdf"
        self.original=self.originals/self.name
        self.derived=self.derivatives/(self.name+".txt")
        self.original.parent.mkdir()
        self.derived.parent.mkdir()
        self.text="A real publisher states its source physics finding in this exact sentence."
        self.original.write_bytes(b'%PDF-1.7 native approved original source bytes')
        self.derived.write_text(
            'KMD generated source provenance header'+chr(10)+
            '[[SOURCE UNIT: PDF page 2]]'+chr(10)+
            self.text+chr(10))
        self.source_map=self.root/"source-map.sqlite3"
        with sqlite3.connect(self.source_map) as conn:
            self.assertEqual("delete",conn.execute("PRAGMA journal_mode").fetchone()[0])
            conn.execute("CREATE TABLE originals (source_path TEXT, status TEXT, "
                         "source_sha256 TEXT, derived_sha256 TEXT, derived_path TEXT)")
            conn.execute("INSERT INTO originals VALUES(?,?,?,?,?)",
                         (self.name,"complete",sha(self.original.read_bytes()),
                          sha(self.derived.read_bytes()),self.name+".txt"))
        self.ledger=self.root/"ledger.json"
        self.ledger.write_text(json.dumps({"items":[{
            "status":"complete","path":self.name,
            "sha256":sha(self.original.read_bytes())}]}))
        self.native_cli=self.root/"summary-cli.py"
        self.raw_cli=self.root/"original-cli.py"
        self.native_cli.write_text("synthetic existing test source CLI")
        self.raw_cli.write_text("synthetic existing test source CLI")
        self.query="What original source describes its physics finding?"
        self.summary={
            "query":self.query,"documents":[{
                "original_path":self.name,"best_summary":"A model-derived navigation summary",
                "similarity":0.77,
                "matched_summary_nodes":[{"original_sha256":sha(self.original.read_bytes()),
                                          "summary_kind":"summary"}]
            }],
            "provenance_policy":"publisher_and_derived_sha256_checked;summary_is_not_primary_evidence",
        }
        self.raw={
            "query":self.query,"retrieval_mode":"overview",
            "results":[{
                "original_path":self.name,"original_source_sha256":sha(self.original.read_bytes()),
                "exact_passage_hash_verified":True,
                "source_passage":self.text,
                "source_passage_sha256":sha(self.text.encode()),
                "source_excerpt":self.text,
                "semantic_similarity":0.81,
                "source_section":"PDF page 2",
                "publisher_url":"https://publisher.example/approved",
                "source_page_citation_warning":None,
            }],
        }
        self.cli=load_cli()
        self.args=[
            "raspi_dual_source_discovery.py",self.query,"--top","3",
            "--summary-cli",str(self.native_cli),
            "--original-passage-cli",str(self.raw_cli),
            "--source-map",str(self.source_map),
            "--original-root",str(self.originals),
            "--derivative-root",str(self.derivatives),
            "--download-ledger",str(self.ledger),
        ]

    def _run_main(self):
        capture=io.StringIO()
        with mock.patch.object(self.cli,"_command",side_effect=[self.summary,self.raw]) as commands:
            with mock.patch.object(sys,"argv",list(self.args)):
                with redirect_stdout(capture):
                    self.cli.main()
            self.assertEqual(2,commands.call_count)
        return json.loads(capture.getvalue())

    def test_real_source_map_exclusive_writer_over_five_seconds_recovers_current_original(self):
        writer=sqlite3.connect(self.source_map,isolation_level=None,timeout=2)
        self.addCleanup(writer.close)
        writer.execute("BEGIN EXCLUSIVE")
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future=pool.submit(self._run_main)
                try:
                    time.sleep(5.55)
                    self.assertFalse(future.done(),
                        "the dual original-source reader timed out at SQLite default five seconds")
                finally:
                    writer.execute("ROLLBACK")
                results=future.result(timeout=8)
            self.assertEqual(1,results["summary_original_count"])
            self.assertEqual(1,results["raw_original_count"])
            self.assertEqual(1,results["union_original_count"])
            self.assertTrue(results["no_ranking_fusion_or_score_threshold_applied"])
            self.assertEqual(self.name,results["original_passage_channel"][0]["original_path"])
            self.assertEqual(self.text,results["original_passage_channel"][0]["passage_preview"])
            self.assertTrue(results["original_passage_channel"][0]["publisher_revision_verified"])
            writer.execute("BEGIN EXCLUSIVE")
            writer.execute("ROLLBACK")
            self.assertEqual(self.text,self.derived.read_text().split("]]"+chr(10))[1].strip())
        finally:
            if writer.in_transaction:
                writer.execute("ROLLBACK")

    def test_source_map_revision_committed_while_dual_reader_waits_fails_closed(self):
        writer=sqlite3.connect(self.source_map,isolation_level=None,timeout=2)
        self.addCleanup(writer.close)
        writer.execute("BEGIN EXCLUSIVE")
        writer.execute("UPDATE originals SET derived_sha256=? WHERE source_path=?",
                       ("f"*64,self.name))
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future=pool.submit(self._run_main)
                try:
                    time.sleep(5.55)
                    self.assertFalse(future.done(),"dual read falsely accepted an old source")
                finally:
                    writer.execute("COMMIT")
                with self.assertRaisesRegex(ValueError,"bytes changed"):
                    future.result(timeout=8)
            with sqlite3.connect(self.source_map) as db:
                self.assertEqual("f"*64,db.execute(
                    "SELECT derived_sha256 FROM originals WHERE source_path=?",
                    (self.name,)).fetchone()[0])
        finally:
            if writer.in_transaction:
                writer.execute("ROLLBACK")


if __name__=="__main__":
    unittest.main()
