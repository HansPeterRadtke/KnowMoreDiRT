"""Exact RFC section routing in mixed-source retrieval must not invent lexical evidence."""
import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock

from test_source_evidence import cli


class RfcStructuralDispatchTests(TestCase):
    def setUp(self):
        self.cli=cli()
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        self.db=root/'native.sqlite3'
        self.original='archive/rfc9110.txt'
        with sqlite3.connect(self.db) as c:
            c.execute("CREATE TABLE filesystem_entries(id INTEGER,relative_path_display TEXT,entry_type TEXT)")
            c.execute("INSERT INTO filesystem_entries VALUES(10,?,'file')",(self.original,))
        self.args=SimpleNamespace(query="RFC 9110, Section 9.2.2",documents=3,
             passages=2,snippet_characters=520,scope_context_characters=320,
             source_name=None,continue_section=None)
        self.named={"status":"unique","original_path":self.original,
                    "kind":"catalog_rfc","subject_query":"Section 9.2.2"}
        self.source={"original_path":self.original,"title":"HTTP Semantics",
            "original_sha256":"a"*64,"derived_sha256":"b"*64,
            "verified_rfc_number":"9110","derived_text":"verified"}
        self.versions=({self.original:{"kind":"text","url":"https://www.rfc-editor.org/rfc/rfc9110.txt"}},
                       {self.original:{"sha256":"a"*64}},
                       {self.original:(self.original,"a"*64,"b"*64,"rfc9110.txt","complete")})
        self.lookup={"status":"verified","rfc":"9110","section":"9.2.2",
                     "title":"Idempotent Methods",
                     "context":{"start_offset":0,"end_offset":333,"next_cursor":"333."+"c"*64}}
        self.publisher={"start_byte":100,"end_byte":126,"sha256":"e"*64}
        self.rows=[{"source_excerpt":"9.2.2. Idempotent Methods",
                    "status":"verified_rfc_section_opening",
                    "source_section":"RFC 9110, Section 9.2.2. Idempotent Methods",
                    "excerpt_sha256":"d"*64,"publisher_excerpt":self.publisher}]
        self.mocked={}
        for name,value in (("DB",self.db),
           ("_current_versions",self.versions),
           ("resolve_source_name",self.named),
           ("checked_original",self.source),
           ("rfc_section_evidence",{"passages":self.rows,
                  "rfc_section_lookup":self.lookup,
                  "evidence_warning":"Continuation may qualify the opening"}),
           ("append_query_audit",root/"receipt")):
            patch=mock.patch.object(self.cli,name,return_value=value) if name!="DB" else mock.patch.object(self.cli,name,value)
            self.mocked[name]=patch.start();self.addCleanup(patch.stop)
        for name in ("_native_passage_search","source_evidence_candidates"):
            p=mock.patch.object(self.cli,name)
            self.mocked[name]=p.start();self.addCleanup(p.stop)

    def call(self):
        with mock.patch("builtins.print") as printed:
            self.cli.find_source_evidence(self.args)
        return json.loads(printed.call_args.args[0])

    def test_named_section_is_not_lexical_toc_and_preserves_exact_rfc_metadata(self):
        result=self.call()
        self.assertEqual("unambiguous_named_source_question",result["selection_basis"])
        self.assertEqual("Section 9.2.2",result["evidence_query"])
        self.assertEqual(1,result["evidence_kind_counts"]["publisher_utf8_byte_intervals"])
        d=result["documents"][0]
        self.assertEqual("verified_derived_text",d["status"])
        self.assertEqual("verified_rfc_body_section_boundary",d["passage_selection_basis"])
        self.assertEqual("Idempotent Methods",d["rfc_section_lookup"]["title"])
        self.assertEqual(self.lookup["context"]["next_cursor"],
                         d["rfc_section_lookup"]["context"]["next_cursor"])
        self.assertEqual("9.2.2. Idempotent Methods",d["passages"][0]["source_excerpt"])
        self.assertIsNone(d["passages"][0]["source_lexical_rank"])
        self.assertIsNone(d["passages"][0]["matched_excerpt_query_terms"])
        self.mocked["rfc_section_evidence"].assert_called_once()
        self.assertEqual("9.2.2",self.mocked["rfc_section_evidence"].call_args.args[3])
        self.assertIsNone(self.mocked["rfc_section_evidence"].call_args.kwargs["cursor"])
        self.mocked["_native_passage_search"].assert_not_called()
        self.mocked["source_evidence_candidates"].assert_not_called()
        audit=self.mocked["append_query_audit"].call_args.args[0]
        self.assertNotIn("source_excerpt",audit["documents"][0]["passages"][0])
        self.assertEqual("RFC 9110, Section 9.2.2. Idempotent Methods",audit["documents"][0]["passages"][0]["source_section"])
        self.assertEqual(self.lookup,audit["documents"][0]["rfc_section_lookup"])

    def test_source_bound_continuation_is_routed_to_structural_reader(self):
        self.args.continue_section="333."+"c"*64
        self.rows[0]["status"]="verified_rfc_section_continuation"
        self.lookup["context"]["start_offset"]=333
        out=self.call()
        self.assertEqual("verified_rfc_body_section_boundary",out["documents"][0]["passage_selection_basis"])
        self.assertEqual(self.args.continue_section,
                         self.mocked["rfc_section_evidence"].call_args.kwargs["cursor"])
        self.mocked["source_evidence_candidates"].assert_not_called()

    def test_missing_section_never_falls_back_to_other_words(self):
        self.lookup["status"]="section_not_verified"
        self.rows.clear()
        d=self.call()["documents"][0]
        self.assertEqual("rfc_section_not_verified",d["status"])
        self.assertEqual([],d["passages"])
        self.mocked["source_evidence_candidates"].assert_not_called()

    def test_invalid_publisher_proof_is_refused(self):
        self.rows[0].pop("publisher_excerpt")
        d=self.call()["documents"][0]
        self.assertEqual("source_unavailable",d["status"])
        self.assertEqual([],d["passages"])
        self.mocked["source_evidence_candidates"].assert_not_called()

    def test_borrowed_cursor_requires_exact_rfc_selection(self):
        self.args.query="How are unrelated artifacts processed?"
        self.args.continue_section="333."+"c"*64
        self.mocked["resolve_source_name"].return_value=None
        with self.assertRaisesRegex(ValueError,"one exact named original"):
            self.call()
        self.mocked["_native_passage_search"].assert_not_called()
        self.args.source_name="docs/paper.html"
        self.mocked["resolve_source_name"].return_value={
            "status":"unique","original_path":"docs/paper.html","kind":"catalog_title"}
        with self.assertRaisesRegex(ValueError,"not found"):
            self.call()
        self.args.continue_section="oversized"*100
        with self.assertRaisesRegex(ValueError,"invalid RFC continuation cursor"):
            self.call()


if __name__=="__main__":
    import unittest
    unittest.main()
