"""Revision-aware native KMD summary completeness checks; no production writes."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import unittest
from file_system_catalog.summary_coverage import (
    current_completed_originals, summary_progress_metrics,
)

class RevisionCoverageTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.executescript("""
          CREATE TABLE filesystem_entries(
            id INTEGER PRIMARY KEY,relative_path_display TEXT,entry_type TEXT);
          CREATE TABLE content_chunks(
            chunk_id TEXT PRIMARY KEY,filesystem_entry_id INTEGER,collection_id TEXT,
            content_sha256 TEXT,start_char INTEGER,end_char INTEGER);
          CREATE TABLE content_representations(
            chunk_id TEXT,prompt_version TEXT,representation_kind TEXT,
            analysis_status TEXT,generation_json TEXT,representation_text TEXT,
            representation_text_sha256 TEXT);
        """)
        self.names = [
            "current.pdf","revised.pdf","leaf.pdf","truncated.pdf",
            "invalid-summary.pdf","wrong-source.pdf","wrong-root.pdf"
        ]
        self.original = "a"*64
        self.derived = "b"*64
        self.sources = {p:(self.original,self.derived,(i+1)*1000)
                        for i,p in enumerate(self.names)}
        for i,p in enumerate(self.names,1):
            self.db.execute("INSERT INTO filesystem_entries VALUES(?,?,?)",(i,p,"file"))
        self.db.commit()

    def add(self,path,*,orig=None,derived=None,tree=None,start=0,end=10000,
            kind="summary_long",source_kind="file",text="Source summary",
            text_hash=None,analysis_status="complete",metadata_override=None):
        i=self.names.index(path)+1
        orig=self.original if orig is None else orig
        derived=self.derived if derived is None else derived
        tree=derived if tree is None else tree
        self.db.execute("INSERT INTO content_chunks VALUES(?,?,?,?,?,?)",
                        (path,i,"library:summary",orig,start,end))
        meta = {
            "original_sha256":orig,
            "derived_sha256":derived,
            "source_tree_sha256":tree,
            "source_start_char":start,
            "source_end_char":end,
            "node_id":"root:"+path,
        }
        if metadata_override:
            meta.update(metadata_override)
        self.db.execute("INSERT INTO content_representations VALUES(?,?,?,?,?,?,?)",
                        (path,"kmd-hierarchical-source-summary-v1",kind,
                        analysis_status,json.dumps(meta),text,
                        text_hash or hashlib.sha256(text.encode()).hexdigest()))
        self.db.commit()

    def test_root_complete_revision_rules(self):
        self.add("current.pdf")
        self.add("revised.pdf",derived="f"*64,tree="f"*64)
        self.add("leaf.pdf",kind="summary")
        self.add("truncated.pdf",tree="c"*64)
        self.add("invalid-summary.pdf",text_hash="0"*64)
        self.add("wrong-source.pdf",orig="d"*64)
        self.add("wrong-root.pdf",metadata_override={"source_start_char":8})
        got=current_completed_originals(self.db,self.sources)
        self.assertEqual({"current.pdf"},got)

    def test_multiple_versions_only_current_one_qualifies(self):
        self.add("current.pdf",derived="f"*64,tree="f"*64)
        self.assertEqual(set(),current_completed_originals(self.db,self.sources))
        self.db.execute("DELETE FROM content_representations WHERE chunk_id=?",("current.pdf",))
        self.db.execute("DELETE FROM content_chunks WHERE chunk_id=?",("current.pdf",))
        self.add("current.pdf")
        self.assertEqual({"current.pdf"},current_completed_originals(self.db,self.sources))
        self.sources["current.pdf"]=(self.original,"c"*64,1000)
        self.assertEqual(set(),current_completed_originals(self.db,self.sources))

    def test_text_percent_and_document_percent_are_not_interchangeable(self):
        scores=summary_progress_metrics(
            {"small":("a","b",100),"large":("a","b",9900)},
            {"small","outside"}
        )
        self.assertEqual(50.0,scores["percentage"])
        self.assertEqual(1.0,scores["text_percentage"])
        self.assertEqual(100,scores["indexed_text_characters"])
        self.assertEqual(10000,scores["eligible_text_characters"])
        self.assertEqual(1,scores["remaining_originals"])
        self.assertIn("excluding_provenance",scores["content_progress_basis"])

    def test_empty_corpus_is_well_defined(self):
        self.assertEqual(set(),current_completed_originals(self.db,{}))
        out=summary_progress_metrics({},set())
        self.assertEqual(0.0,out["percentage"])
        self.assertEqual(0.0,out["text_percentage"])

if __name__=="__main__":
    unittest.main()
