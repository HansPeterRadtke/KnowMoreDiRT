"""Search all native KMD summary levels and return original documents."""
from __future__ import annotations

import hashlib
import sqlite3
import unittest

from test_hierarchical_summary_store import SummaryStorageTest, FakeEmbeddings
from file_system_catalog.hierarchical_summary_search import (
    search_summary_entries, summary_index_status,
)
from file_system_catalog.hierarchical_summary_store import (
    summary_rows, append_summary_rows,
)


class SummarySearchTest(unittest.TestCase):
    def setup_doc(self):
        db, kwargs = SummaryStorageTest().prepare()
        chunk_rows, reps = summary_rows(db, **kwargs)
        append_summary_rows(db, chunk_rows=chunk_rows, representation_rows=reps)
        return db, kwargs, chunk_rows, reps

    def test_discover_original_with_all_levels_in_same_database(self):
        db, kwargs, chunks, reps = self.setup_doc()
        embedded = FakeEmbeddings()
        query = embedded.embed([kwargs["tree"].root.summary])[0]
        result = search_summary_entries(
            db, query, limit=4, nodes_per_document=5,
            expected_model=embedded.model,
        )
        self.assertEqual(1, len(result))
        self.assertEqual("books/guide.pdf", result[0]["original_path"])
        self.assertEqual(1, result[0]["filesystem_entry_id"])
        self.assertGreater(result[0]["similarity"], 0.99)
        self.assertEqual(len(kwargs["tree"].nodes), result[0]["indexed_summary_nodes_for_document"])
        self.assertLessEqual(len(result[0]["matched_summary_nodes"]), 5)
        self.assertEqual(kwargs["original_sha256"], result[0]["original_sha256"])
        self.assertTrue(all(x["source_scope"].startswith("summary_candidate") for x in result[0]["matched_summary_nodes"]))
        summary = summary_index_status(db)
        self.assertEqual(len(kwargs["tree"].nodes), summary["native_summary_vectors"])
        self.assertEqual(1, summary["original_documents_with_summaries"])
        self.assertGreater(summary["maximum_summary_level"], 0)
        db.close()

    def test_source_guard_blocks_document(self):
        db, kwargs, chunks, reps = self.setup_doc()
        query = FakeEmbeddings().embed([kwargs["tree"].root.summary])[0]
        checked = []
        found = search_summary_entries(db, query, require_source=lambda h: checked.append(h) or False)
        self.assertEqual([], found)
        self.assertEqual(len(reps), len(checked))
        self.assertEqual([], search_summary_entries(db, query, expected_model="some-other-model"))
        db.close()

    def test_corrupt_representation_is_skipped(self):
        db, kwargs, chunks, reps = self.setup_doc()
        query = FakeEmbeddings().embed([kwargs["tree"].root.summary])[0]
        count = summary_index_status(db)["native_summary_vectors"]
        db.execute(
            "UPDATE content_representations SET embedding_sha256=? WHERE representation_id=?",
            ("0"*64, reps[0]["representation_id"]),
        )
        remaining = search_summary_entries(db, query)
        self.assertEqual(count-1, remaining[0]["indexed_summary_nodes_for_document"])
        db.close()

    def test_invalid_search_bounds_and_nonfinite_query(self):
        db, kwargs, chunks, reps = self.setup_doc()
        query = FakeEmbeddings().embed([kwargs["tree"].root.summary])[0]
        with self.assertRaises(ValueError):
            search_summary_entries(db, query, limit=0)
        with self.assertRaises(ValueError):
            search_summary_entries(db, query[:2])
        db.close()


if __name__ == "__main__":
    unittest.main()
