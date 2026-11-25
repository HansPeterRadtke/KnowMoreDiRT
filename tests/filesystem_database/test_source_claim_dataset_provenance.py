"""Current-source provenance checks for frozen, independently prepared claims."""
from __future__ import annotations

import copy
import hashlib
import unittest
from unittest.mock import Mock

from file_system_catalog.source_claim_dataset_provenance import attest_frozen_claim_sources


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fixture():
    body="First line.\nA condition applies.\n\nVersion two is excluded.\n"
    normalized=" ".join(body.split())
    path="approved/manual.pdf"
    url="https://example.invalid/original"
    title="Manual revision two"
    original="a"*64
    derived="b"*64
    layout="c"*64
    page=3
    def span(text):
        start=normalized.index(text)
        return {"kind":"publisher_pdf_reading_order_span","text":text,
                "title":title,"source_path":path,"source_url":url,
                "sha256":digest(text),"original_sha256":original,
                "derived_sha256":derived,"page_text_sha256":digest(body),
                "physical_page":page,"start_char":start,
                "end_char":start+len(text)}
    sources={"one":span("A condition applies."),
             "two":span("Version two is excluded."),
             "synthetic":{"kind":"synthetic_fixture","title":"Test control",
                          "text":"A hypothetical counterexample.",
                          "sha256":digest("A hypothetical counterexample.")}}
    d={"schema":"kmd.source_claim_assessment_frozen_dataset.v1",
       "frozen_before_model_inference":True,"source_text_is_private":True,
       "sources":sources,"cases":[
          {"id":"a","claim":"A condition applies.","evidence_ids":["one","two"],
           "expected_label":"supports","expected_evidence_ids":["one"],
           "group":"prior_papers"},
          {"id":"b","claim":"An unrelated current implementation exists.",
           "evidence_ids":["synthetic"],"expected_label":"insufficient",
           "expected_evidence_ids":[],"group":"synthetic_scope"}]}
    source={"original_path":path,"source_url":url,"title":title,
            "original_sha256":original,"current_derived_sha256":derived,
            "publisher_original_actual_sha256_verified":True,
            "source_map_derivative_actual_sha256_verified":True}
    ctx={"source_map_revision_reverified_after_physical_page_extraction":True,
         "source":source,"physical_pages":{
           "original_pdf_sha256":original,
           "requested_center_physical_page":page,
           "original_pdf_revision_verified_before_and_after":True,
           "pages":[{"physical_page":page,"status":"verified_original_page_text",
                     "physical_page_poppler_sha256":layout}],
           "optional_center_reading_order_projection":{
               "physical_page":page,"status":"same_physical_page_alternate_reading_order_text",
               "extraction_mode":"poppler_default_reading_order_utf8_not_layout",
               "exact_layout_page_poppler_sha256":layout,
               "complete_extractor_output_not_truncated":True,
               "reading_order_poppler_sha256":digest(body),
               "extracted_text_sha256":digest(body),"extracted_page_text":body}}}
    return d,ctx,path


class FrozenClaimProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.d,self.ctx,self.path=fixture()
        self.reader=Mock(return_value=self.ctx)
        self.ids=Mock(return_value={self.path:37})
        self.final=Mock()
    def attest(self):
        return attest_frozen_claim_sources(self.d,read_current_pdf_context=self.reader,
            native_identities=self.ids,final_current_source_check=self.final)

    def test_two_exact_spans_one_physical_read_and_final_current_checks(self):
        receipt=self.attest()
        self.reader.assert_called_once_with(self.path,3)
        self.assertEqual(self.ids.call_count,2)
        self.final.assert_called_once_with({self.path:self.ctx["source"]})
        self.assertEqual(receipt["publisher_originals_reverified"],1)
        self.assertEqual(receipt["physical_original_pages_reverified"],1)
        self.assertEqual(receipt["publisher_spans_reverified"],2)
        self.assertEqual(receipt["synthetic_fixture_spans_not_publisher_attested"],1)
        self.assertTrue(receipt["semantic_gold_labels_not_proven_by_source_hash"])
        self.assertFalse(receipt["cross_store_immutable_generation_pinned"])
        self.assertFalse(receipt["safe_to_cutover_live"])
        self.assertNotIn("A condition applies.",str(receipt))
        self.assertEqual({r["evidence_id"] for r in receipt["spans"]},{"one","two"})

    def test_each_current_source_identity_and_verification_flag_required(self):
        for key,value in (("original_path","different.pdf"),
                          ("source_url","https://wrong.invalid"),
                          ("title","Wrong version"),
                          ("original_sha256","d"*64),
                          ("current_derived_sha256","d"*64),
                          ("publisher_original_actual_sha256_verified",False),
                          ("source_map_derivative_actual_sha256_verified",False)):
            with self.subTest(key=key):
                ctx=copy.deepcopy(self.ctx);ctx["source"][key]=value
                self.reader.return_value=ctx
                with self.assertRaisesRegex(ValueError,"current approved publisher revision"):
                    self.attest()

    def test_wrong_physical_page_or_truncated_alternate_rejected(self):
        for key,value in (("physical_page",4),
                          ("status","unverified"),
                          ("extraction_mode","layout"),
                          ("exact_layout_page_poppler_sha256","d"*64),
                          ("complete_extractor_output_not_truncated",False),
                          ("reading_order_poppler_sha256","d"*64),
                          ("extracted_text_sha256","d"*64)):
            with self.subTest(key=key):
                ctx=copy.deepcopy(self.ctx)
                ctx["physical_pages"]["optional_center_reading_order_projection"][key]=value
                self.reader.return_value=ctx
                with self.assertRaisesRegex(ValueError,"reading-order page proof"):
                    self.attest()

    def test_missing_layout_digests_do_not_match_each_other(self):
        del self.ctx["physical_pages"]["pages"][0]["physical_page_poppler_sha256"]
        del self.ctx["physical_pages"]["optional_center_reading_order_projection"]["exact_layout_page_poppler_sha256"]
        with self.assertRaisesRegex(ValueError,"missing original layout page digest"):
            self.attest()

    def test_tampered_text_with_original_receipt_digest_rejected(self):
        self.ctx["physical_pages"]["optional_center_reading_order_projection"]["extracted_page_text"]="Different page."
        with self.assertRaisesRegex(ValueError,"page text digest"):
            self.attest()

    def test_valid_digest_but_wrong_offsets_rejected(self):
        s=self.d["sources"]["one"]
        s["start_char"]+=1;s["end_char"]+=1
        with self.assertRaisesRegex(ValueError,"span differs"):
            self.attest()

    def test_frozen_span_self_consistency_does_not_prove_actual_text(self):
        s=self.d["sources"]["one"]
        s["text"]="Fabricated claim."
        s["sha256"]=digest(s["text"])
        s["end_char"]=s["start_char"]+len(s["text"])
        with self.assertRaisesRegex(ValueError,"span differs"):
            self.attest()

    def test_final_native_id_change_rejected(self):
        self.ids.side_effect=[{self.path:37},{self.path:38}]
        with self.assertRaisesRegex(ValueError,"native original identity changed"):
            self.attest()

    def test_initial_native_identity_must_be_exact(self):
        for bad in ({},{self.path:True},{self.path:0},{self.path:37,"extra.pdf":38}):
            with self.subTest(bad=bad):
                self.ids.return_value=bad
                with self.assertRaisesRegex(ValueError,"native original identities"):
                    self.attest()
        self.reader.assert_not_called()

    def test_final_revision_guard_failure_is_not_a_success_receipt(self):
        self.final.side_effect=ValueError("current source changed")
        with self.assertRaisesRegex(ValueError,"current source changed"):
            self.attest()

    def test_no_publisher_attestation_for_synthetic_only(self):
        self.d["sources"]={"synthetic":self.d["sources"]["synthetic"]}
        self.d["cases"]=self.d["cases"][1:]
        result=self.attest()
        self.reader.assert_not_called();self.ids.assert_not_called();self.final.assert_not_called()
        self.assertEqual(result["publisher_originals_reverified"],0)
        self.assertEqual(result["synthetic_fixture_spans_not_publisher_attested"],1)

    def test_unverified_or_missing_context_rejected(self):
        for bad in (None,{},{"source":self.ctx["source"],"physical_pages":self.ctx["physical_pages"]}):
            with self.subTest(bad=bad):
                self.reader.return_value=bad
                with self.assertRaisesRegex(ValueError,"missing independently verified"):
                    self.attest()


if __name__=="__main__":unittest.main()
