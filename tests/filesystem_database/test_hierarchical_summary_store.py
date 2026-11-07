"""Native FileDB storage for hierarchy nodes: original rows, real KMD schema."""
from __future__ import annotations

import base64
import hashlib
import sqlite3
import unittest
from copy import deepcopy

import numpy as np

from file_system_catalog.content_schema import (
    CONTENT_CREATE_SQL, CONTENT_INDEX_SQL, CHUNK_TABLE_NAME,
    REPRESENTATION_TABLE_NAME,
)
from file_system_catalog.content_pipeline import search_semantic_entries
from file_system_catalog.hierarchical_summary import HierarchicalSummarizer
from file_system_catalog.hierarchical_summary_store import (
    summary_rows, append_summary_rows,
)
from test_hierarchical_summary import FakeSummaryModel, long_document


class FakeEmbeddings:
    model = "test-embedding-384"
    revision = "test-sha"
    embedding_dimension = 384

    def embed(self, texts):
        results = []
        for text in texts:
            arr = np.frombuffer(hashlib.sha256(text.encode()).digest() * 48, dtype=np.uint8)[:384].astype("<f4")
            arr /= np.linalg.norm(arr)
            results.append(arr)
        return results


def fixture():
    db = sqlite3.connect(":memory:")
    db.execute("PRAGMA foreign_keys=ON")
    db.execute(
        "CREATE TABLE filesystem_entries (id INTEGER PRIMARY KEY, "
        "relative_path_display TEXT NOT NULL, relative_path_b64 TEXT NOT NULL, "
        "entry_type TEXT NOT NULL, size_bytes INTEGER NOT NULL)"
    )
    for sql in CONTENT_CREATE_SQL + CONTENT_INDEX_SQL:
        db.execute(sql)
    original = "books/guide.pdf"
    db.execute(
        "INSERT INTO filesystem_entries VALUES (1,?,?,?,?)",
        (original, base64.b64encode(original.encode()).decode(), "file", 12345),
    )
    db.commit()
    return db


class SummaryStorageTest(unittest.TestCase):
    def prepare(self):
        db = fixture()
        original_sha = "a" * 64
        source = long_document(12, 100)
        tree = HierarchicalSummarizer(
            FakeSummaryModel(750), output_tokens=64, safety_tokens=16
        ).summarize(title="Guide", text=source)
        kwargs = {
            "original_path": "books/guide.pdf",
            "original_sha256": original_sha,
            "original_bytes": 12345,
            "derived_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "derived_text": source,
            "summary_start_char": 0,
            "tree": tree,
            "embedding_client": FakeEmbeddings(),
            "analysis_model": "test-local-llm",
            "collection_id": "kmd-local-summary-v1",
        }
        return db, kwargs

    def test_full_tree_written_to_native_representations(self):
        db, kwargs = self.prepare()
        c, r = summary_rows(db, **kwargs)
        self.assertEqual(len(kwargs["tree"].nodes), len(r))
        self.assertEqual(len(c), len(r))
        self.assertEqual("summary_long", next(x["representation_kind"] for x in r
                                            if x["generation_json"].find(kwargs["tree"].root_id) >= 0))
        self.assertTrue(all(len(bytes(x["embedding_blob"])) == 384 * 2 for x in r))
        result = append_summary_rows(db, chunk_rows=c, representation_rows=r)
        self.assertEqual(len(c)*2, result["new_rows"])
        self.assertEqual(len(c), db.execute("SELECT count(*) FROM content_chunks").fetchone()[0])
        self.assertEqual(len(r), db.execute("SELECT count(*) FROM content_representations").fetchone()[0])
        self.assertEqual("ok", db.execute("PRAGMA integrity_check").fetchone()[0])
        self.assertEqual([], db.execute("PRAGMA foreign_key_check").fetchall())
        query = FakeEmbeddings().embed([kwargs["tree"].root.summary])[0]
        found = search_semantic_entries(db, query)
        self.assertEqual(1, len(found))
        self.assertEqual("books/guide.pdf", found[0]["relative_path_display"])
        self.assertIn(found[0]["analysis_kind"], ("summary", "summary_long"))
        self.assertTrue(any(x["representation_kind"] == "summary_long" for x in r))
        self.assertGreater(found[0]["score"], 0.99)
        again = append_summary_rows(db, chunk_rows=c, representation_rows=r)
        self.assertEqual(0, again["new_rows"])
        db.close()

    def test_corrupt_original_and_derivative_rejected_before_write(self):
        db, kwargs = self.prepare()
        for changed in [
            {"original_bytes": 12346},
            {"derived_sha256": "b" * 64},
            {"original_path": "books/unknown.pdf"},
            {"summary_start_char": 500},
            {"original_path": "../books/guide.pdf"},
        ]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                summary_rows(db, **dict(kwargs, **changed))
        self.assertEqual(0, db.execute("SELECT count(*) FROM content_chunks").fetchone()[0])
        db.close()

    def test_silent_overwrite_is_refused_and_rolled_back(self):
        db, kwargs = self.prepare()
        chunks, rep = summary_rows(db, **kwargs)
        append_summary_rows(db, chunk_rows=chunks, representation_rows=rep)
        tampered = [dict(r) for r in rep]
        tampered[0]["embedding_sha256"] = "c" * 64
        with self.assertRaisesRegex(RuntimeError, "conflicting"):
            append_summary_rows(db, chunk_rows=chunks, representation_rows=tampered)
        self.assertEqual(rep[0]["embedding_sha256"], db.execute(
            "SELECT embedding_sha256 FROM content_representations WHERE representation_id=?",
            (rep[0]["representation_id"],),
        ).fetchone()[0])
        db.close()

    def test_requires_foreign_keys_and_bounded_storage(self):
        db, kwargs = self.prepare()
        c, r = summary_rows(db, **kwargs)
        db.execute("PRAGMA foreign_keys=OFF")
        with self.assertRaisesRegex(RuntimeError, "foreign keys"):
            append_summary_rows(db, chunk_rows=c, representation_rows=r)
        db.execute("PRAGMA foreign_keys=ON")
        with self.assertRaises(ValueError):
            append_summary_rows(db, chunk_rows=c, representation_rows=r, max_database_bytes=10)
        db.close()


if __name__ == "__main__":
    unittest.main()
