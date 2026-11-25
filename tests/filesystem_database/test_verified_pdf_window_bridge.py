"""Page expansion supplements and never replaces independently verified source results."""
from __future__ import annotations
import copy
import hashlib
import json
from types import SimpleNamespace
from unittest import TestCase, mock

from file_system_catalog.verified_pdf_window_bridge import (
    expand_verified_page_evidence, expansion_audit_summary, validate_expansion_request,
)
from file_system_catalog.verified_pdf_neighbor_context import PhysicalPdfPageContextUnavailable
from test_reading_order_page_passages import context
from test_source_evidence import cli

def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()

def fixture(path="approved/paper.pdf", page=3, rank=1, native=7,
            layout="Private method. The measurement applies to scenario A only. No general claim follows.",
            reading=None, excerpt="The measurement applies to scenario A only."):
    reading=layout if reading is None else reading
    c=context(reading)
    c["source"]["original_path"]=path
    block=c["physical_pages"]
    block["requested_center_physical_page"]=page
    block["pages"]=[{"physical_page":page,"status":"verified_original_page_text",
                     "extracted_page_text":layout,"extracted_text_sha256":sha(layout),
                     "physical_page_poppler_sha256":sha(layout),
                     "text_projection_dropped_non_whitespace":False,
                     "display_layout_policy":"verbatim_poppler_page_text"}]
    block["optional_center_reading_order_projection"].update(
        physical_page=page,exact_layout_page_poppler_sha256=sha(layout))
    norm=" ".join(layout.split())
    assert excerpt in norm
    start=norm.index(excerpt)
    anchor={"original_path":path,"filesystem_entry_id":native,
            "original_sha256":"a"*64,"derived_sha256":"b"*64,
            "source_excerpt":excerpt,"excerpt_sha256":sha(excerpt),
            "publisher_pdf_page":{"status":"verified","page":page,
                "page_text_sha256":sha(layout),"start_char":start,"end_char":start+len(excerpt)}}
    doc={"rank":rank,"status":"verified","original_path":path,"original_sha256":"a"*64,
         "derived_sha256":"b"*64,"filesystem_entry_id":native,"passages":[anchor]}
    return doc,c

class VerifiedPageExpansionTests(TestCase):
    def expand(self,docs,contexts,budget=5):
        return expand_verified_page_evidence(docs,"measurement scenario A only",
            page_budget=budget,snippet_characters=200,
            read_context=lambda path,page:contexts[path,page])

    def test_deduplicated_pages_round_robin_preserves_originals(self):
        a,ca=fixture()
        b,cb=fixture(page=4)
        other,co=fixture(path="approved/other.pdf",page=2,rank=2,native=8)
        a["passages"] += [copy.deepcopy(a["passages"][0]),b["passages"][0]]
        prior=copy.deepcopy([a,other])
        result=self.expand([a,other],{("approved/paper.pdf",3):ca,
            ("approved/paper.pdf",4):cb,("approved/other.pdf",2):co},budget=2)
        self.assertEqual(prior,[a,other])
        self.assertEqual(result["distinct_verified_candidate_pages"],3)
        self.assertEqual(result["candidate_pages_not_examined_due_to_budget"],1)
        self.assertEqual([(x["document_rank"],x["physical_page"]) for x in result["pages"]],[(1,3),(2,2)])
        self.assertEqual([1,2],[x["passage_rank"] for x in result["pages"][0]["source_layout_anchors"]])
        self.assertTrue(result["native_document_ranking_and_original_passages_unchanged"])
        self.assertTrue(result["not_automatic_factual_entailment_or_answer_truth"])
        p=result["pages"][0]
        self.assertEqual(p["windows"][0]["span_sha256"],sha(p["windows"][0]["text"]))
        self.assertTrue(p["source_layout_anchors"][0]["layout_anchor_also_contiguous_in_reading_order_page"])
        audit=expansion_audit_summary(result)
        self.assertNotIn("Private method.",str(audit))
        self.assertNotIn("normalized_excerpt",str(audit))

    def test_reordered_reading_text_does_not_invent_layout_anchor(self):
        a,c=fixture(layout="First. The measurement applies to scenario A only. End.",
            reading="First. The measurement applies to scenario A. Other column. Only an example. End.")
        r=self.expand([a],{(a["original_path"],3):c})
        proof=r["pages"][0]["source_layout_anchors"][0]
        self.assertFalse(proof["layout_anchor_also_contiguous_in_reading_order_page"])
        self.assertEqual(proof["window_ranks_containing_complete_normalized_layout_anchor"],[])
        self.assertTrue(r["pages"][0]["windows"])

    def test_unverified_originals_never_read_physical_page(self):
        a,c=fixture()
        a["passages"][0]["publisher_pdf_page"]={"status":"unverified","page":3}
        a["passages"][0]["source_section"]="PDF page 3"
        r=self.expand([a],{})
        self.assertEqual(r["pages"],[])
        self.assertEqual(r["distinct_verified_candidate_pages"],0)

    def test_explicit_extraction_unavailability_not_identity_mismatch(self):
        a,c=fixture()
        r=expand_verified_page_evidence([a],"measurement",page_budget=1,
             snippet_characters=200,read_context=mock.Mock(side_effect=PhysicalPdfPageContextUnavailable("bounded extraction")))
        self.assertEqual(r["pages"][0]["status"],"physical_page_context_unavailable")
        self.assertEqual(r["candidate_windows"],0)
        c["physical_pages"]["optional_center_reading_order_projection"].update(
             status="center_reading_order_text_unavailable",extracted_page_text=None)
        r=self.expand([a],{(a["original_path"],3):c})
        self.assertEqual(r["pages"][0]["status"],"reading_order_context_unavailable")

    def test_wrong_revision_layout_offsets_and_tampered_anchor_refuse(self):
        for change in [
            lambda a,c:a["passages"][0].update(excerpt_sha256="e"*64),
            lambda a,c:a["passages"][0].update(filesystem_entry_id=8),
            lambda a,c:c["source"].update(current_derived_sha256="c"*64),
            lambda a,c:c["physical_pages"]["pages"][0].update(physical_page_poppler_sha256="d"*64),
            lambda a,c:c["physical_pages"]["pages"][0].update(extracted_page_text="forged"),
            lambda a,c:a["passages"][0]["publisher_pdf_page"].update(
                start_char=a["passages"][0]["publisher_pdf_page"]["start_char"]+1,
                end_char=a["passages"][0]["publisher_pdf_page"]["end_char"]+1),
            lambda a,c:c["physical_pages"]["optional_center_reading_order_projection"].update(extracted_page_text="forged"),
            lambda a,c:a.update(rank=True),
        ]:
            with self.subTest(change=change):
                a,c=fixture();change(a,c)
                with self.assertRaises(ValueError):
                    self.expand([a],{("approved/paper.pdf",3):c})

    def test_global_budget_and_duplicate_native_id_refuse(self):
        for budget in (False,True,-1,6):
            with self.assertRaises(ValueError):validate_expansion_request(budget,5,900)
        validate_expansion_request(0,50,900)
        with self.assertRaises(ValueError):validate_expansion_request(1,50,900)
        a,c=fixture();b,bc=fixture(path="approved/b.pdf",rank=2,native=7)
        with self.assertRaises(ValueError):self.expand([a,b],{})

    def test_unavailable_or_budget_excluded_page_does_not_emit_any_unverified_text(self):
        a,c=fixture()
        a["passages"][0]["publisher_pdf_page"]["page_text_sha256"]="e"*64
        with self.assertRaises(ValueError):self.expand([a],{(a["original_path"],3):c})

class OptionalDispatchTests(TestCase):
    def test_opt_in_helper_passes_exact_query_and_fixed_page_context(self):
        mod=cli()
        a,c=fixture()
        import raspi_original_pdf_page_neighbors as pages
        with mock.patch.object(pages,"original_pdf_context",return_value=c) as reader:
            out=mod._optional_verified_pdf_page_windows(
                SimpleNamespace(expand_pdf_pages=1,snippet_characters=200),
                [a],evidence_query="measurement scenario")
        reader.assert_called_once_with(a["original_path"],3,neighbor_radius=0,include_center_reading_order=True)
        self.assertEqual(1,out["pages_examined"])
        self.assertIsNone(mod._optional_verified_pdf_page_windows(
            SimpleNamespace(),[a],evidence_query="measurement"))

    def test_invalid_or_incompatible_request_refuses_prior_to_selection(self):
        mod=cli()
        for args in (SimpleNamespace(expand_pdf_pages=6,snippet_characters=900),
                     SimpleNamespace(expand_pdf_pages=1,snippet_characters=900,continue_section="cursor"),
                     SimpleNamespace(expand_pdf_pages=1,snippet_characters=900,private_staged_pdf_preflight="old.json")):
            with self.assertRaises(ValueError):mod._validate_pdf_page_expansion(args,5)

if __name__=="__main__":
    import unittest
    unittest.main()
