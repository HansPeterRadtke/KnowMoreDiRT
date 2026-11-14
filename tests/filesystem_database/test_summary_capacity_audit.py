"""Audit of recent source-linked storage, never a fake full-database prediction."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch


def capacity_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "raspi_hierarchical_capacity_audit.py"
    spec = importlib.util.spec_from_file_location("kmd_capacity_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StorageAuditTests(TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.db = root / "filedb.sqlite3"
        self.sources = root / "sources.sqlite3"
        with sqlite3.connect(self.db) as conn:
            conn.execute("CREATE TABLE filesystem_entries (id INTEGER PRIMARY KEY,relative_path_display TEXT)")
            conn.execute("CREATE TABLE content_chunks (chunk_id TEXT PRIMARY KEY,filesystem_entry_id INTEGER,collection_id TEXT)")
            conn.execute("CREATE TABLE content_representations (chunk_id TEXT,embedding_blob BLOB,generation_json TEXT,representation_text TEXT,created_at_ns INTEGER,prompt_version TEXT,analysis_status TEXT)")
            for number in range(1, 13):
                path = f"original/paper-{number}.pdf"
                conn.execute("INSERT INTO filesystem_entries VALUES (?,?)",(number,path))
                conn.execute("INSERT INTO content_chunks VALUES (?,?,?)",(str(number),number,"archive:summary"))
                conn.execute("INSERT INTO content_representations VALUES (?,?,?,?,?,?,?)",
                             (str(number),b"1"*12,"json","text",2_000_000_000,
                              "kmd-hierarchical-source-summary-v1","complete"))
            # Exclude this incidental row, even when its timestamp is recent.
            conn.execute("INSERT INTO content_representations VALUES (?,?,?,?,?,?,?)",
                         ("1",b"1"*1000,"huge","uncited",2_000_000_000,"other","complete"))
            conn.commit()
        with sqlite3.connect(self.sources) as conn:
            conn.execute("CREATE TABLE originals (source_path TEXT,source_sha256 TEXT,derived_sha256 TEXT,text_characters INTEGER,status TEXT)")
            for number in range(1,13):
                conn.execute("INSERT INTO originals VALUES (?,?,?,?,?)",
                             (f"original/paper-{number}.pdf","a"*64,"b"*64,1000,"complete"))
            conn.commit()

    def test_readonly_sample_is_not_full_database_prediction(self):
        module=capacity_module()
        with patch.object(module,"current_completed_originals",return_value={"one"}),              patch.object(module,"summary_progress_metrics",return_value={
                 "complete_originals":1,"remaining_originals":11,
                 "eligible_text_characters":12000,"indexed_text_characters":1000,
             }):
            out=module.measure(self.db,self.sources,now_ns=3_000_000_000,window_hours=48)
        self.assertEqual(12,out["recent_indexed_originals"])
        self.assertEqual(12,out["recent_representation_rows"])
        self.assertEqual(12000,out["recent_source_text_characters"])
        expected=12*(12+4+4)
        self.assertEqual(expected,out["recent_partial_payload_bytes"])
        self.assertEqual(expected*11000//12000,out["estimated_remaining_partial_payload_bytes_at_recent_density"])
        self.assertIn("Partial-size",out["projection_scope"])
        with sqlite3.connect(f"file:{self.db}?mode=ro",uri=True) as db:
            self.assertEqual(13,db.execute("SELECT COUNT(*) FROM content_representations").fetchone()[0])

    def test_invalid_window_does_not_open_or_mutate_database(self):
        module=capacity_module()
        for value in (0,169,False,"48"):
            with self.subTest(value=value),self.assertRaises(ValueError):
                module.measure(self.db,self.sources,window_hours=value)


if __name__=="__main__":
    main()
