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
    def test_completed_sources_are_not_reported_as_outstanding_failures(self):
        worker = load_worker()
        old = {
            "docs/completed.pdf": {"attempts": 5},
            "docs/retry.pdf": {"attempts": 2},
            "docs/retired.pdf": {"attempts": 1},
        }
        active = [
            ("docs/completed.pdf", "a"*64, "b"*64, 100),
            ("docs/retry.pdf", "c"*64, "d"*64, 200),
        ]
        result = worker.pending_failures(old, active, {"docs/completed.pdf"})
        self.assertEqual({"docs/retry.pdf": {"attempts": 2}}, result)
        self.assertEqual(3, len(old))

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
                    "(id INTEGER,relative_path_display TEXT,entry_type TEXT);"
                    "CREATE TABLE content_chunks "
                    "(chunk_id TEXT,filesystem_entry_id INTEGER,collection_id TEXT,"
                    " content_sha256 TEXT,start_char INTEGER,end_char INTEGER);"
                    "CREATE TABLE content_representations "
                    "(chunk_id TEXT,prompt_version TEXT,representation_kind TEXT,"
                    " analysis_status TEXT,generation_json TEXT,"
                    " representation_text TEXT,representation_text_sha256 TEXT);"
                )
                import hashlib
                source_summary = "An accurate full original summary."
                summary_sha = hashlib.sha256(source_summary.encode()).hexdigest()
                metadata = json.dumps({
                    "original_sha256": "a"*64, "derived_sha256": "b"*64,
                    "source_tree_sha256": "b"*64, "source_start_char": 0,
                    "source_end_char": 555, "node_id": "root-a",
                })
                db.executemany(
                    "INSERT INTO filesystem_entries VALUES(?,?,?)",
                    [(1, "docs/a.pdf","file"), (2, "docs/b.pdf","file")],
                )
                db.executemany(
                    "INSERT INTO content_chunks VALUES(?,?,?,?,?,?)",
                    [
                        ("summary-a", 1, "original-library:summary","a"*64,0,555),
                        ("overview-b", 2, "original-library:overview","c"*64,0,100),
                    ],
                )
                db.executemany(
                    "INSERT INTO content_representations VALUES(?,?,?,?,?,?,?)",
                    [
                        ("summary-a", "kmd-hierarchical-source-summary-v1",
                         "summary_long", "complete", metadata,source_summary,summary_sha),
                        ("overview-b", "legacy-overview-v1", "summary",
                         "complete","{}",source_summary,summary_sha),
                    ],
                )
            with mock.patch.object(worker, "SOURCE_MAP", index), \
                 mock.patch.object(worker, "FILEDB", filedb):
                available, processed = worker.scope()
            self.assertEqual(2, len(available))
            self.assertEqual({"docs/a.pdf"}, processed)
            self.assertEqual({"docs/b.pdf"}, {r[0] for r in available} - processed)

    def test_revision_or_partial_only_root_reenters_worker_backlog(self):
        worker = load_worker()
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root/"source-map.sqlite3"
            filedb = root/"filedb.sqlite3"
            records = [
                ("docs/current.pdf","a"*64,"b"*64,1500,"complete"),
                ("docs/changed.pdf","c"*64,"e"*64,2200,"complete"),
                ("docs/partial.pdf","f"*64,"g"*64,1800,"complete"),
            ]
            with sqlite3.connect(index) as db:
                db.execute("CREATE TABLE originals(source_path TEXT,source_sha256 TEXT,"
                           "derived_sha256 TEXT,text_characters INTEGER,status TEXT)")
                db.executemany("INSERT INTO originals VALUES(?,?,?,?,?)",records)
            with sqlite3.connect(filedb) as db:
                db.executescript(
                    "CREATE TABLE filesystem_entries(id INTEGER,relative_path_display TEXT,entry_type TEXT);"
                    "CREATE TABLE content_chunks(chunk_id TEXT,filesystem_entry_id INTEGER,"
                    "collection_id TEXT,content_sha256 TEXT,start_char INTEGER,end_char INTEGER);"
                    "CREATE TABLE content_representations(chunk_id TEXT,prompt_version TEXT,"
                    "representation_kind TEXT,analysis_status TEXT,generation_json TEXT,"
                    "representation_text TEXT,representation_text_sha256 TEXT);"
                )
                db.executemany("INSERT INTO filesystem_entries VALUES(?,?,?)",
                               [(i, p, "file") for i,(p,*_) in enumerate(records,1)])
                text="Source-grounded summary"
                text_sha=hashlib.sha256(text.encode()).hexdigest()
                for i,(path,orig,derived,_,_) in enumerate(records,1):
                    if path=="docs/changed.pdf":
                        derived="d"*64  # a genuine stale summary revision
                    marker="summary_long" if path!="docs/partial.pdf" else "summary"
                    db.execute("INSERT INTO content_chunks VALUES(?,?,?,?,?,?)",
                               (str(i),i,"collection:summary",orig,0,2600))
                    metadata=json.dumps({
                        "original_sha256":orig,"derived_sha256":derived,
                        "source_tree_sha256":derived,"source_start_char":0,
                        "source_end_char":2600,"node_id":str(i),
                    })
                    db.execute("INSERT INTO content_representations VALUES(?,?,?,?,?,?,?)",
                               (str(i),"kmd-hierarchical-source-summary-v1",marker,
                                "complete",metadata,text,text_sha))
            with mock.patch.object(worker,"SOURCE_MAP",index),                  mock.patch.object(worker,"FILEDB",filedb):
                available,done=worker.scope()
            self.assertEqual({"docs/current.pdf"},done)
            self.assertEqual(3,len(available))
            self.assertEqual({"docs/changed.pdf","docs/partial.pdf"},
                             {x[0] for x in available}-done)

    def test_balanced_schedule_includes_medium_and_large_documents(self):
        worker = load_worker()
        sources = (
            [(f"small/{i:03d}", "a"*64, "b"*64, 1000+i)
             for i in range(70)]
            + [(f"medium/{i:03d}", "c"*64, "d"*64, 70000+i)
               for i in range(12)]
            + [(f"large/{i:03d}", "e"*64, "f"*64, 250000+i)
               for i in range(3)]
        )
        scheduled=list(worker.balanced_backlog(sources,0))
        self.assertEqual(len(sources),len(scheduled))
        self.assertEqual(len(set(x[0] for x in sources)),len(set(x[0] for x in scheduled)))
        self.assertEqual({x[0] for x in sources},{x[0] for x in scheduled})
        self.assertTrue(scheduled[7][0].startswith("medium/"))
        self.assertTrue(scheduled[15][0].startswith("medium/"))
        self.assertTrue(scheduled[39][0].startswith("large/"))
        self.assertTrue(scheduled[0][0].startswith("small/"))
        # Within each class, shortest sources should be first, avoiding
        # unexpectedly picking the largest book before bounded samples.
        for kind in ("small/","medium/","large/"):
            found=[r[3] for r in scheduled if r[0].startswith(kind)]
            self.assertEqual(found,sorted(found))

    def test_balanced_schedule_resume_modulo_and_empty_bucket_fallback(self):
        worker=load_worker()
        sources=(
            [(f"short/{i}", "a"*64, "b"*64, 2200+i) for i in range(38)]
            + [(f"mid/{i}", "c"*64, "d"*64, 60000+i) for i in range(3)]
            + [("long/0", "e"*64, "f"*64, 400000)]
        )
        schedule=list(worker.balanced_backlog(sources,7))
        self.assertEqual("mid/0",schedule[0][0])  # eighth source
        self.assertEqual("long/0",schedule[32][0])  # fortieth source
        self.assertEqual(len(sources),len(schedule))
        self.assertEqual({x[0] for x in sources},{x[0] for x in schedule})
        only_small=[("x/1","a","b",100),("x/2","c","d",200)]
        self.assertEqual(only_small,list(worker.balanced_backlog(only_small,39)))
        with self.assertRaises(ValueError):
            list(worker.balanced_backlog(only_small,-1))
        with self.assertRaises(ValueError):
            list(worker.balanced_backlog([("bad","a","b","wrong")],0))

    def test_source_text_weighted_progress_is_reported_separately(self):
        worker = load_worker()
        corpus = {
            "small.pdf":("a"*64,"b"*64,2000),
            "huge.pdf":("c"*64,"d"*64,998000),
        }
        result=worker.summary_progress_metrics(corpus,{"small.pdf"})
        self.assertEqual(50.0,result["percentage"])
        self.assertEqual(0.2,result["text_percentage"])
        self.assertEqual(2000,result["indexed_text_characters"])
        self.assertEqual(1000000,result["eligible_text_characters"])

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

    def test_coder_voice_slot_flags_forwarded_without_second_worker(self):
        worker=load_worker()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"source-map.sqlite3"
            filedb=root/"filedb.sqlite3"
            cli=root/"test-cli.py"
            source.touch();filedb.touch();cli.touch()
            recorded=[]
            with mock.patch.object(worker,"RUNROOT",root), \
                 mock.patch.object(worker,"STATE",root/"progress.json"), \
                 mock.patch.object(worker,"LOCK",root/".lock"), \
                 mock.patch.object(worker,"SOURCE_MAP",source), \
                 mock.patch.object(worker,"FILEDB",filedb), \
                 mock.patch.object(worker,"CLI",cli), \
                 mock.patch.object(worker,"scope",return_value=(
                    [("docs/source.txt","a"*64,"b"*64,1200)],set(),
                 )), \
                 mock.patch.object(worker,"run_command",
                    side_effect=lambda args,label,path:recorded.append((args,label,path))):
                worker.run(
                    sleep_seconds=0,max_calls=50,once=True,dry_run=True,
                    model_url="http://10.8.0.7:14833",
                    model_name="qwen3-coder-30b-a3b-instruct-q4-k-m",
                    output_tokens=2400,context_cap_tokens=16384,
                    shared_slot_idle_seconds=8,
                    compact_native_facets=True,
                )
            self.assertEqual(1,len(recorded))
            parts,label,path=recorded[0]
            self.assertEqual("PLAN",label)
            self.assertEqual("http://10.8.0.7:14833",
                             parts[parts.index("--model-url")+1])
            self.assertEqual("qwen3-coder-30b-a3b-instruct-q4-k-m",
                             parts[parts.index("--model")+1])
            self.assertEqual("2400",parts[parts.index("--output-tokens")+1])
            self.assertEqual("16384",parts[parts.index("--context-cap-tokens")+1])
            self.assertEqual("8",parts[parts.index("--shared-slot-idle-seconds")+1])
            self.assertIn("--native-facets",parts)
            self.assertIn("--compact-native-facets",parts)
            self.assertIn("--plan-only",parts)
            self.assertEqual("docs/source.txt",path)
            state=json.loads((root/"progress.json").read_text())
            self.assertEqual("qwen3-coder-30b-a3b-instruct-q4-k-m",state["model"])
            self.assertEqual(8,state["shared_voice_slot_idle_seconds"])
            self.assertTrue(state["compact_native_facets"])

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
