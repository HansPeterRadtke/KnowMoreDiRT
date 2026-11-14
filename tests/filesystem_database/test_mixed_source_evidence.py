"""Native mixed-source evidence must not promote an HTML derivative into a PDF page quote."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_source_evidence import cli


class MixedSourceEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.cli = cli()
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "native.sqlite3"
        self.paths = {
            "html": "docs/paper.html",
            "pdf": "docs/paper.pdf",
            "text": "docs/raw.txt",
        }
        self.ids = {"html": 8, "pdf": 9, "text": 10}
        with sqlite3.connect(self.db) as connection:
            connection.execute("CREATE TABLE filesystem_entries (id INTEGER, entry_type TEXT, relative_path_display TEXT)")
            for kind, path in self.paths.items():
                connection.execute("INSERT INTO filesystem_entries VALUES (?,'file',?)",
                                   (self.ids[kind], path))
        self.hits = [self.hit("html"), self.hit("pdf")]
        self.args = SimpleNamespace(
            query="How are glass material coatings physically represented?",
            documents=3, passages=2, snippet_characters=1000,
        )
        versions = (
            {p: {"path":p, "kind":k, "source_url":f"https://example.org/{k}"} for k,p in self.paths.items()},
            {p: {"sha256":"a"*64} for p in self.paths.values()},
            {p: (p, "a"*64, "b"*64, p+".txt", "complete") for p in self.paths.values()},
        )
        self.mocks = {}
        values = {
            "DB": self.db,
        }
        for name, value in values.items():
            context = patch.object(self.cli, name, value)
            context.start();self.addCleanup(context.stop)
        for name, value in (
            ("_current_versions", versions),
            ("_native_passage_search", {"results": self.hits}),
            ("checked_original", None),
            ("source_evidence_candidates", None),
            ("verify_pdf_page_excerpts", None),
            ("publisher_path", self.db),
            ("append_query_audit", Path(self.temp.name)/"audit"),
        ):
            kw = {"side_effect": self.checked} if name == "checked_original" else (
                {"side_effect": self.candidates} if name == "source_evidence_candidates" else (
                    {"return_value": [{"status":"verified","page":4,"page_text_sha256":"f"*64}]} if name == "verify_pdf_page_excerpts"
                    else {"return_value":value}
                )
            )
            context = patch.object(self.cli, name, **kw)
            self.mocks[name] = context.start()
            self.addCleanup(context.stop)

    def hit(self, kind):
        return {
            "original_path": self.paths[kind],
            "source_kind": kind,
            "original_source_sha256": "a"*64,
            "kmd_native_chunk_id": f"chunk-{kind}",
            "kmd_original_file_entry_id": self.ids[kind],
            "semantic_similarity": 0.9 if kind=="html" else 0.7,
        }

    def checked(self, path, **kwargs):
        return {
            "original_path": path, "original_sha256":"a"*64,
            "derived_sha256":"b"*64, "title":"Verified sample",
            "derived_text":"sample",
            "publisher_text_bytes": b"first party byte source" if path.endswith(".txt") else None,
        }

    def candidates(self, **kwargs):
        path = kwargs["original_path"]
        if path.endswith(".pdf"):
            return {"passages":[{"source_excerpt":"PDF excerpt", "excerpt_sha256":"0"*64,
                                 "matched_excerpt_terms":["material","coatings"],
                                 "source_section":"PDF page 4"}]}
        if path.endswith(".txt"):
            return {"passages":[{"source_excerpt":"Direct UTF-8", "excerpt_sha256":"1"*64,
                                 "matched_excerpt_terms":["material"],
                                 "publisher_excerpt":{"start_byte":0,"end_byte":12,"sha256":"2"*64}}]}
        return {"passages":[{"source_excerpt":"Extracted HTML", "excerpt_sha256":"3"*64,
                             "matched_excerpt_terms":["material","glass","coatings"]}]}

    def call(self):
        with patch("builtins.print") as output:
            self.cli.find_source_evidence(self.args)
        return json.loads(output.call_args.args[0])

    def test_mixed_document_rank_distinguishes_derived_html_from_physically_verified_pdf(self):
        result = self.call()
        self.assertEqual("native_discovery_mixed_source_evidence_v1", result["retrieval_engine"])
        self.assertEqual((1,1), (result["verified_pdf_excerpts"],result["verified_non_pdf_derived_excerpts"]))
        self.assertEqual({"physical_pdf_pages":1, "publisher_utf8_byte_intervals":0,
                          "extracted_text_only":1},result["evidence_kind_counts"])
        html,pdf = result["documents"]
        self.assertEqual((1,"verified_derived_text","docs/paper.html"),
                         (html["rank"],html["status"],html["original_path"]))
        self.assertEqual("verified_extracted_text_only",html["passages"][0]["evidence_provenance"]["scope"])
        self.assertFalse(html["passages"][0]["evidence_provenance"]["physical_pdf_page_verified"])
        self.assertNotIn("publisher_pdf_page",html["passages"][0])
        self.assertEqual((2,"verified",4),(pdf["rank"],pdf["status"],pdf["passages"][0]["publisher_pdf_page"]["page"]))
        self.assertEqual("chunk-html",html["discovery_chunk_id"])
        audit = self.mocks["append_query_audit"].call_args.args[0]
        self.assertEqual("find_source_evidence",audit["operation"])
        self.assertNotIn("source_excerpt",audit["documents"][0]["passages"][0])
        self.assertNotIn("source_excerpt",audit["documents"][1]["passages"][0])
        self.assertIn("HTML",result["evidence_scope"])
        self.assertEqual(2,self.mocks["checked_original"].call_count)
        self.assertEqual(1,self.mocks["verify_pdf_page_excerpts"].call_count)

    def test_publisher_utf8_plain_text_proof_kept_distinct_from_normalized_html(self):
        self.mocks["_native_passage_search"].return_value={"results":[self.hit("text")]}
        result=self.call()
        d=result["documents"][0]
        self.assertEqual("verified_derived_text",d["status"])
        self.assertEqual("publisher_utf8_byte_interval",
                         d["passages"][0]["evidence_provenance"]["scope"])
        self.assertEqual("verified",d["passages"][0]["evidence_provenance"]["status"])
        self.assertEqual(12,d["passages"][0]["publisher_excerpt"]["end_byte"])
        self.assertEqual(1,result["evidence_kind_counts"]["publisher_utf8_byte_intervals"])
        self.assertFalse(d["passages"][0]["evidence_provenance"]["physical_pdf_page_verified"])
        self.mocks["checked_original"].assert_called_once()
        self.assertTrue(self.mocks["checked_original"].call_args.kwargs["include_publisher_text"])
        self.mocks["verify_pdf_page_excerpts"].assert_not_called()

    def test_explicit_catalog_name_selects_only_one_original_without_model_ranking(self):
        self.args.source_name = "docs/paper.html"
        result = self.call()
        self.assertEqual("explicit_named_catalog_original",result["selection_basis"])
        self.assertEqual(1,len(result["documents"]))
        doc=result["documents"][0]
        self.assertEqual("docs/paper.html",doc["original_path"])
        self.assertEqual("explicit_named_catalog_original",doc["selection_basis"])
        self.assertIsNone(doc["discovery_similarity"])
        self.assertIsNone(doc["discovery_chunk_id"])
        self.mocks["_native_passage_search"].assert_not_called()
        self.assertEqual("verified_derived_text",doc["status"])

    def test_explicit_pdf_source_gets_physical_page_proof_without_semantic_claim(self):
        self.args.source_name="docs/paper.pdf"
        result=self.call()
        d=result["documents"][0]
        self.assertEqual("explicit_named_catalog_original",result["selection_basis"])
        self.assertEqual("verified",d["status"])
        self.assertIsNone(d["discovery_similarity"])
        self.assertEqual(4,d["passages"][0]["publisher_pdf_page"]["page"])
        self.assertEqual(1,result["evidence_kind_counts"]["physical_pdf_pages"])
        self.mocks["_native_passage_search"].assert_not_called()

    def test_affirmative_named_question_uses_topic_not_false_semantic_score(self):
        self.args.query = "What does RFC 9110 say about safe HTTP methods?"
        with patch.object(self.cli,"resolve_source_name",return_value={
            "status":"unique","original_path":"docs/raw.txt",
            "kind":"catalog_rfc","subject_query":"safe HTTP methods?"
        }):
            result=self.call()
        self.assertEqual("unambiguous_named_source_question",result["selection_basis"])
        self.assertEqual("safe HTTP methods?",result["evidence_query"])
        self.assertEqual("docs/raw.txt",result["documents"][0]["original_path"])
        self.assertIsNone(result["documents"][0]["discovery_similarity"])
        self.assertEqual(1,result["verified_non_pdf_derived_excerpts"])
        self.mocks["_native_passage_search"].assert_not_called()

    def test_ambiguous_or_absent_named_original_never_falls_back_to_unrelated_vectors(self):
        self.args.source_name="Nonexistent publisher source"
        with self.assertRaisesRegex(ValueError,"not uniquely recognized"):
            self.call()
        with patch.object(self.cli,"resolve_source_name",return_value={
            "status":"ambiguous","candidate_paths":["docs/paper.html","docs/raw.txt"]
        }):
            with self.assertRaisesRegex(ValueError,"ambiguous catalog original"):
                self.call()
        with patch.object(self.cli,"resolve_source_name",return_value={
            "status":"no_catalog_match","matching_catalog_paths":0
        }):
            with self.assertRaisesRegex(ValueError,"not found"):
                self.call()
        self.mocks["_native_passage_search"].assert_not_called()
        self.mocks["checked_original"].assert_not_called()

    def test_drift_and_reassigned_entries_never_return_evidence(self):
        for changed in (
            dict(original_source_sha256="0"*64),
            dict(kmd_original_file_entry_id=987),
            dict(source_kind="text"),
        ):
            with self.subTest(changed=changed):
                self.mocks["_native_passage_search"].return_value={
                    "results":[dict(self.hit("html"),**changed)],
                }
                self.assertEqual("source_unavailable",self.call()["documents"][0]["status"])
        self.mocks["source_evidence_candidates"].assert_not_called()

    def test_derivative_revision_mismatch_after_original_verification(self):
        self.mocks["_native_passage_search"].return_value={"results":[self.hit("html")]}
        self.mocks["checked_original"].side_effect=lambda *a,**kw: {
            **self.checked(*a,**kw),"derived_sha256":"0"*64,
        }
        d=self.call()["documents"][0]
        self.assertEqual("source_unavailable",d["status"])
        self.assertEqual([],d["passages"])
        self.mocks["source_evidence_candidates"].assert_not_called()

    def test_unsupported_kind_and_no_candidates_are_not_invented(self):
        self.mocks["_native_passage_search"].return_value={
            "results":[dict(self.hit("html"),source_kind="exe")],
        }
        self.assertEqual("unsupported_source_kind",self.call()["documents"][0]["status"])
        self.mocks["source_evidence_candidates"].assert_not_called()
        self.mocks["_native_passage_search"].return_value={"results":[self.hit("html")]}
        self.mocks["source_evidence_candidates"].side_effect=lambda **kw: {"passages":[]}
        self.assertEqual("no_verified_excerpt",self.call()["documents"][0]["status"])

    def test_invalid_bounds_fail_before_native_discovery(self):
        for kwargs in (
            {"documents":0}, {"documents":6}, {"passages":0}, {"passages":6},
            {"snippet_characters":20}, {"snippet_characters":2500},
            {"query":" "}, {"query":"x"*601},
        ):
            with self.subTest(kwargs=kwargs):
                values={**vars(self.args),**kwargs}
                with self.assertRaises(ValueError),patch("builtins.print"):
                    self.cli.find_source_evidence(SimpleNamespace(**values))
        self.mocks["_native_passage_search"].assert_not_called()


if __name__=="__main__":
    unittest.main()
