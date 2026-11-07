"""Regression checks for the Raspberry Pi continuous native summary worker.

All tests use disposable SQLite/status fixtures and never generate summaries,
call the LLM, change originals or write the production FileDB.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
import sqlite3
import stat
import tempfile
import unittest
from unittest import mock


def load_worker():
    path = Path(__file__).resolve().parents[2] / "scripts" / "raspi_hierarchical_worker.py"
    spec = importlib.util.spec_from_file_location("kmd_hierarchical_worker_fixture", path)
    worker = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(worker)
    return worker


class ContinuousWorkerTest(unittest.TestCase):
    def test_resume_scope_uses_original_file_id_and_only_summary_representations(self):
        worker = load_worker()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / "source-map.sqlite3"
            filedb = root / "filesystem.sqlite3"
            with sqlite3.connect(index) as db:
                db.execute(
                    "CREATE TABLE originals "
                    "(source_path TEXT,source_sha256 TEXT,derived_sha256 TEXT,"
                    "text_characters INTEGER,status TEXT)"
                )
                db.executemany(
                    "INSERT INTO originals VALUES(?,?,?,?,?)",
                    [
                        ("docs/a.pdf", "a" * 64, "b" * 64, 500, "complete"),
                        ("docs/b.pdf", "c" * 64, "d" * 64, 501, "complete"),
                        ("docs/c.pdf", "e" * 64, "f" * 64, 700, "failed"),
                    ],
                )
            with sqlite3.connect(filedb) as db:
                db.executescript(
                    "CREATE TABLE filesystem_entries"
                    "(id INTEGER,relative_path_display TEXT);"
                    "CREATE TABLE content_chunks "
                    "(chunk_id TEXT,filesystem_entry_id INTEGER,collection_id TEXT);"
                    "CREATE TABLE content_representations "
                    "(chunk_id TEXT,prompt_version TEXT);"
                )
                db.executemany(
                    "INSERT INTO filesystem_entries VALUES(?,?)",
                    [(1, "docs/a.pdf"), (2, "docs/b.pdf")],
                )
                db.executemany(
                    "INSERT INTO content_chunks VALUES(?,?,?)",
                    [
                        ("summary-a", 1, "original-library:summary"),
                        ("overview-b", 2, "original-library:overview"),
                    ],
                )
                db.executemany(
                    "INSERT INTO content_representations VALUES(?,?)",
                    [
                        ("summary-a", "kmd-hierarchical-source-summary-v1"),
                        ("overview-b", "legacy-overview-v1"),
                    ],
                )
            with mock.patch.object(worker, "SOURCE_MAP", index), \
                 mock.patch.object(worker, "FILEDB", filedb):
                available, processed = worker.scope()
            self.assertEqual(2, len(available))
            self.assertEqual({"docs/a.pdf"}, processed)
            self.assertEqual({"docs/b.pdf"}, {r[0] for r in available} - processed)

    def test_atomic_progress_file_persists_incremental_complete_count(self):
        worker = load_worker()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(worker, "RUNROOT", root), \
                 mock.patch.object(worker, "STATE", root / "worker-status.json"):
                worker.state_update(
                    phase="summarizing", complete_originals=2,
                    total_eligible_originals=17_714, percentage=0.011291,
                )
                worker.state_update(
                    phase="embedding", complete_originals=3, percentage=0.016936,
                )
                found = json.loads((root / "worker-status.json").read_text())
                self.assertEqual("embedding", found["phase"])
                self.assertEqual(3, found["complete_originals"])
                self.assertEqual(17_714, found["total_eligible_originals"])
                self.assertEqual(0.016936, found["percentage"])
                self.assertEqual(stat.S_IMODE((root / "worker-status.json").stat().st_mode), 0o600)
                self.assertEqual([], list(root.glob("*.tmp.*")))

    def test_command_uses_fixed_local_python_bounded_timeout(self):
        worker = load_worker()
        class Response:
            returncode = 0
            stdout = "PLAN checked\nCOMPLETE_TREE okay\n"
        with mock.patch.object(worker.subprocess, "run", return_value=Response()) as runner:
            result = worker.run_command(
                ["build", "--original-path", "docs/test.pdf", "--max-calls", "15"],
                "BUILD", "docs/test.pdf",
            )
        self.assertIn("COMPLETE_TREE", result)
        self.assertEqual(sys.executable, runner.call_args.args[0][0])
        self.assertEqual(14400, runner.call_args.kwargs["timeout"])
        self.assertTrue(runner.call_args.kwargs["check"] is False)

    def test_command_failure_is_not_reported_as_document_completion(self):
        worker = load_worker()
        class Response:
            returncode = 1
            stdout = "a bounded model failure"
        with mock.patch.object(worker.subprocess, "run", return_value=Response()):
            with self.assertRaisesRegex(RuntimeError, "bounded model failure"):
                worker.run_command(["build", "--original-path", "a"], "BUILD", "a")


if __name__ == "__main__":
    unittest.main()
