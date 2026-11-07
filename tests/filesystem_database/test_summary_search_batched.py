"""Native KMD bounded batched search: exact-result and failure regressions."""
from __future__ import annotations

import hashlib
import unittest

from test_hierarchical_summary_store import SummaryStorageTest, FakeEmbeddings
from file_system_catalog.hierarchical_summary_search import search_summary_entries
from file_system_catalog.summary_search_batched import (
    search_summary_entries_batched, _scan_vectors,
)


class BatchedNativeSearchTest(unittest.TestCase):
    def prepare(self):
        t=SummaryStorageTest()
        db,kwargs=t.prepare()
        from file_system_catalog.hierarchical_summary_store import summary_rows,append_summary_rows
        chunks,reps=summary_rows(db,**kwargs)
        append_summary_rows(db,chunk_rows=chunks,representation_rows=reps)
        return db,kwargs,chunks,reps

    def compare(self,db,query,*,source_check=None):
        first=search_summary_entries(
            db,query,limit=10,nodes_per_document=5,
            expected_model="test-embedding-384",require_source=source_check,
        )
        second=search_summary_entries_batched(
            db,query,limit=10,nodes_per_document=5,
            expected_model="test-embedding-384",require_source=source_check,
        )
        self.assertEqual([x["original_path"] for x in first],
                         [x["original_path"] for x in second])
        for left,right in zip(first,second):
            for field in (
                "original_path","original_bytes","original_sha256",
                "best_summary","best_level","indexed_summary_nodes_for_document",
            ):
                self.assertEqual(left[field],right[field],field)
            self.assertAlmostEqual(left["similarity"],right["similarity"],places=5)
            self.assertEqual(
                [h["summary_node_id"] for h in left["matched_summary_nodes"]],
                [h["summary_node_id"] for h in right["matched_summary_nodes"]],
            )
            for l,r in zip(left["matched_summary_nodes"],right["matched_summary_nodes"]):
                self.assertEqual(l["summary_sha256"],r["summary_sha256"])
                self.assertEqual(l["source_start_char"],r["source_start_char"])
                self.assertEqual(l["source_end_char"],r["source_end_char"])
                self.assertAlmostEqual(l["similarity"],r["similarity"],places=5)
        return first,second

    def test_existing_native_multi_level_summary_matches_scalar(self):
        db,kwargs,chunks,reps=self.prepare()
        query=FakeEmbeddings().embed([kwargs["tree"].root.summary])[0]
        try:
            first,second=self.compare(db,query)
            self.assertEqual(1,len(second))
            self.assertEqual(len(reps),second[0]["indexed_summary_nodes_for_document"])
            self.assertEqual("summary_long",second[0]["matched_summary_nodes"][0]["summary_kind"])
        finally:
            db.close()

    def test_input_limits_and_nonfinite_vectors_fail_closed(self):
        db,kwargs,chunks,reps=self.prepare()
        query=FakeEmbeddings().embed([kwargs["tree"].root.summary])[0]
        try:
            import numpy as np
            for invalid in (query[:10],np.full((384,),np.nan)):
                with self.assertRaises(ValueError):
                    search_summary_entries_batched(db,invalid)
            for batch in (0,1,63,8193):
                with self.assertRaises(ValueError):
                    search_summary_entries_batched(db,query,vector_batch_size=batch)
            self.assertEqual([],search_summary_entries_batched(
                db,query,expected_model="unrelated-model"
            ))
        finally:
            db.close()

    def test_corrupt_stored_embedding_digest_and_text_fail_closed(self):
        db,kwargs,chunks,reps=self.prepare()
        query=FakeEmbeddings().embed([kwargs["tree"].root.summary])[0]
        try:
            rep=reps[0]["representation_id"]
            db.execute(
                "UPDATE content_representations SET embedding_sha256=? WHERE representation_id=?",
                ("0"*64,rep),
            )
            old,new=self.compare(db,query)
            self.assertEqual(len(reps)-1,new[0]["indexed_summary_nodes_for_document"])
            rep=reps[1]["representation_id"]
            db.execute(
                "UPDATE content_representations SET representation_text_sha256=? WHERE representation_id=?",
                ("0"*64,rep),
            )
            old,new=self.compare(db,query)
            self.assertEqual(len(reps)-2,new[0]["indexed_summary_nodes_for_document"])
        finally:
            db.close()

    def test_source_callback_filters_before_returning_any_original(self):
        db,kwargs,chunks,reps=self.prepare()
        query=FakeEmbeddings().embed([kwargs["tree"].root.summary])[0]
        try:
            self.assertEqual([],search_summary_entries_batched(
                db,query,require_source=lambda r: False
            ))
            old,new=self.compare(db,query,source_check=lambda r: r["summary_kind"]=="summary_long")
            self.assertEqual(1,len(new))
            self.assertEqual(1,new[0]["indexed_summary_nodes_for_document"])
            self.assertTrue(all(node["summary_kind"]=="summary_long"
                for node in new[0]["matched_summary_nodes"]))
        finally:
            db.close()

    def test_sqlite_is_read_only_no_temp_tables_or_source_writes(self):
        db,kwargs,chunks,reps=self.prepare()
        query=FakeEmbeddings().embed([kwargs["tree"].root.summary])[0]
        try:
            before=db.total_changes
            objects=db.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall()
            self.compare(db,query)
            self.assertEqual(before,db.total_changes)
            self.assertEqual(objects,db.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall())
            self.assertEqual([],db.execute("SELECT name FROM sqlite_temp_master").fetchall())
        finally:
            db.close()


if __name__=="__main__":
    unittest.main()
