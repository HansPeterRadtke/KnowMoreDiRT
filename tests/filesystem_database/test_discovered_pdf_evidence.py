"""Corpus discovery must bind every PDF excerpt to the discovered native source and revision."""
import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from test_source_evidence import cli


class DiscoveredPdfEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.cli = cli()
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        db = Path(self.temp.name) / "native.sqlite3"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE filesystem_entries (id INTEGER, entry_type TEXT, relative_path_display TEXT)")
            conn.execute("INSERT INTO filesystem_entries VALUES (8,'file','paper.pdf')")
        self.source = dict(original_path="paper.pdf", original_sha256="a"*64, derived_sha256="b"*64, title="Paper", derived_text="test")
        self.hit = dict(original_path="paper.pdf", original_source_sha256="a"*64, kmd_native_chunk_id="c7",
                        kmd_original_file_entry_id=8, semantic_similarity=0.7, source_kind="pdf")
        self.args = SimpleNamespace(query="How are scores verified?", documents=2, passages=1, snippet_characters=1600)
        self.ranked = [{"source_excerpt":"first", "excerpt_sha256":"first-sha"},
                       {"source_excerpt":"second", "excerpt_sha256":"second-sha"}]
        versions = ({"paper.pdf":{"path":"paper.pdf","kind":"pdf","source_url":"https://publisher.example/paper"}},
                    {"paper.pdf":{"sha256":"a"*64}}, {"paper.pdf":("paper.pdf","a"*64,"b"*64,"paper.txt","complete")})
        for name,value in (("DB",db),):
            ctx=patch.object(self.cli,name,value)
            ctx.start();self.addCleanup(ctx.stop)
        self.mocked={}
        for name,value in (("_current_versions",versions),("checked_original",self.source),
                           ("_native_passage_search",{"results":[self.hit]}),
                           ("source_evidence_candidates",{"passages":self.ranked}),
                           ("publisher_path",Path(self.temp.name)/"paper.pdf"),
                           ("verify_pdf_page_excerpts",[{"status":"unverified","reason":"excerpt_absent"},
                                                      {"status":"verified","page":7}]),
                           ("append_query_audit",Path(self.temp.name)/"receipt")):
            ctx=patch.object(self.cli,name,return_value=value)
            self.mocked[name]=ctx.start();self.addCleanup(ctx.stop)
    def call(self):
        with patch("builtins.print") as output:
            self.cli.find_pdf_evidence(self.args)
        return json.loads(output.call_args.args[0])
    def test_only_physically_verified_excerpts_survive_and_keep_document_rank(self):
        self.mocked["_native_passage_search"].return_value={"results":[dict(self.hit, original_path="other.html",source_kind="html"),self.hit]}
        out=self.call()
        first,second=out["documents"]
        self.assertEqual((1,"not_pdf",[]),(first["rank"],first["status"],first["passages"]))
        self.assertEqual((2,"verified"),(second["rank"],second["status"]))
        self.assertEqual("second",second["passages"][0]["source_excerpt"])
        self.assertEqual(7,second["passages"][0]["publisher_pdf_page"]["page"])
        record=self.mocked["append_query_audit"].call_args.args[0]
        self.assertEqual(record["documents"][1]["passages"][0]["excerpt_sha256"],"second-sha")
        self.assertNotIn("source_excerpt",record["documents"][1]["passages"][0])
        self.assertEqual("c7",second["discovery_chunk_id"])
    def test_changed_revision_or_reassigned_native_entry_returns_no_evidence(self):
        for key,value in (("original_source_sha256","0"*64),("kmd_original_file_entry_id",11)):
            with self.subTest(key=key):
                self.mocked["_native_passage_search"].return_value={"results":[dict(self.hit,**{key:value})]}
                self.assertEqual(self.call()["documents"][0]["status"],"source_unavailable")
        self.mocked["_native_passage_search"].return_value={"results":[self.hit]}
        self.mocked["checked_original"].return_value=dict(self.source,derived_sha256="0"*64)
        self.assertEqual(self.call()["documents"][0]["passages"],[])
        self.mocked["source_evidence_candidates"].assert_not_called()
    def test_empty_or_unverifiable_pages_are_not_returned_as_answers(self):
        self.mocked["verify_pdf_page_excerpts"].return_value=[{"status":"unverified","reason":"missing"}]*2
        self.assertEqual(self.call()["documents"][0]["status"],"no_verified_excerpt")
        self.mocked["_native_passage_search"].return_value={"results":[]}
        result=self.call()
        self.assertEqual(result["documents"],[])
        self.assertEqual(result["verified_excerpts"],0)
    def test_more_distinct_query_terms_recover_a_later_verified_methods_excerpt(self):
        hits=[
            {"source_excerpt":"generic", "excerpt_sha256":"g", "matched_excerpt_terms":["scores"]},
            {"source_excerpt":"full answer", "excerpt_sha256":"f",
             "matched_excerpt_terms":["scores","rankings","crrf","confidence","quality"]},
            {"source_excerpt":"partial", "excerpt_sha256":"p",
             "matched_excerpt_terms":["crrf","quality"]},
        ]
        self.args.passages=1
        self.mocked["source_evidence_candidates"].return_value={"passages":hits}
        self.mocked["verify_pdf_page_excerpts"].return_value=[
            {"status":"verified","page":9},{"status":"verified","page":3},
            {"status":"verified","page":11},
        ]
        doc=self.call()["documents"][0]
        self.assertEqual("verified",doc["status"])
        self.assertEqual("full answer",doc["passages"][0]["source_excerpt"])
        self.assertEqual(2,doc["passages"][0]["source_lexical_rank"])
        self.assertEqual(5,doc["passages"][0]["matched_excerpt_query_terms"])
        self.assertEqual(3,doc["source_candidates"])

    def test_unreadable_publisher_excerpts_are_reported_not_silently_promoted(self):
        self.mocked["source_evidence_candidates"].return_value={
            "passages": [], "unreadable_pdf_excerpts_dropped": 3,
        }
        self.mocked["verify_pdf_page_excerpts"].return_value=[]
        doc=self.call()["documents"][0]
        self.assertEqual("no_verified_excerpt",doc["status"])
        self.assertEqual(0,doc["source_candidates"])
        self.assertEqual(3,doc["unreadable_pdf_excerpts_dropped"])
        self.assertEqual(0,doc["physical_page_unverified_candidates"])

    def test_physical_page_misses_are_counted_without_disclosing_a_false_page(self):
        doc=self.call()["documents"][0]
        self.assertEqual(2,doc["source_candidates"])
        self.assertEqual(1,doc["physical_page_unverified_candidates"])
        self.assertEqual(0,doc["unreadable_pdf_excerpts_dropped"])
        self.assertEqual(7,doc["passages"][0]["publisher_pdf_page"]["page"])

    def test_bad_bounds_fail_before_model_or_source_access(self):
        self.args.documents=0
        with self.assertRaises(ValueError):
            self.call()
        self.mocked["_native_passage_search"].assert_not_called()
        self.mocked["checked_original"].assert_not_called()
