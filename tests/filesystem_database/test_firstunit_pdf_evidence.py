from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
from unittest import TestCase,mock

from file_system_catalog.firstunit_pdf_evidence import (
    SCHEMA,select_lexical_pdf_candidates,require_lexical_current_identity,
)
from file_system_catalog.reading_order_document_pages import BoundedPdfDiscoveryUnavailable

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
import raspi_firstunit_selected_pdf_evidence as bridge


def hit(path,rank,channel,native=1):
    return {"original_path":path,"source_rank":rank,
            "rank_channel":channel+"_title_firstpage_bm25",
            "publisher_original_sha256":"a"*64,"current_derivative_sha256":"b"*64,
            "native_original_filedb_entry_id":native,
            "current_approved_publisher_and_extracted_bytes_rehashed":True}


def receipt():
    query="Test original question?"
    channels={
        "literal":[hit("first.html",1,"literal",3),hit("a.pdf",2,"literal",1),
                   hit("b.pdf",3,"literal",2)],
        "porter":[hit("b.pdf",1,"porter",2),hit("a.pdf",2,"porter",1)],
    }
    return {"schema":SCHEMA,"independent_rank_channels_no_fusion":True,
            "native_filedb_and_source_map_read_only":True,
            "all_displayed_candidates_original_and_full_derivative_sha_reverified":True,
            "query_count":1,"queries":[{"id":"case","query":query,
                "query_sha256":hashlib.sha256(query.encode()).hexdigest(),
                "rank_fusion_applied":False,"independent_rank_channels":{
                    c:{"source_originals":r,"returned_current_originals":len(r)}
                    for c,r in channels.items()}}]}


class LexicalSelectionTests(TestCase):
    def test_preserve_ranks_deduplicate_and_skip_non_pdf_without_compressing_rank(self):
        result=select_lexical_pdf_candidates(receipt(),originals_per_query=2)[0]
        a,b=result["selected_originals"]
        self.assertEqual(("b.pdf",{"literal":3,"porter":1}),
                         (a["original_path"],a["independent_lexical_ranks"]))
        self.assertEqual(("a.pdf",{"literal":2,"porter":2}),
                         (b["original_path"],b["independent_lexical_ranks"]))
        self.assertEqual(2,result["available_distinct_pdf_candidates"])

    def test_changed_receipt_rejected_even_if_unselected(self):
        for edit in (
            lambda d:d["queries"][0].update(query="changed"),
            lambda d:d["queries"][0].update(rank_fusion_applied=True),
            lambda d:d["queries"][0]["independent_rank_channels"]["porter"]["source_originals"][1].update(
                current_derivative_sha256="c"*64),
            lambda d:d["queries"][0]["independent_rank_channels"]["literal"]["source_originals"][0].update(
                original_path="../escape.html"),
            lambda d:d["queries"][0]["independent_rank_channels"]["literal"]["source_originals"][2].update(
                source_rank=True),
            lambda d:d["queries"][0]["independent_rank_channels"]["literal"]["source_originals"][2].update(
                native_original_filedb_entry_id=1),
        ):
            with self.subTest(edit=edit):
                d=receipt();edit(d)
                with self.assertRaises(ValueError):select_lexical_pdf_candidates(d)

    def test_current_original_rechecked_not_trusted_from_receipt(self):
        c=select_lexical_pdf_candidates(receipt())[0]["selected_originals"][0]
        s={"original_path":"b.pdf","original_sha256":"a"*64,"current_derived_sha256":"b"*64,
           "publisher_original_actual_sha256_verified":True,
           "source_map_derivative_actual_sha256_verified":True}
        require_lexical_current_identity(c,s,2)
        with self.assertRaises(ValueError):require_lexical_current_identity(c,s,4)
        with self.assertRaises(ValueError):require_lexical_current_identity(c,{**s,"original_sha256":"c"*64},2)


class LexicalEvidenceBridgeTests(TestCase):
    def identity(self,path):
        return ({"original_path":path,"original_sha256":"a"*64,"current_derived_sha256":"b"*64,
                 "publisher_original_actual_sha256_verified":True,
                 "source_map_derivative_actual_sha256_verified":True},2 if path=="b.pdf" else 1)

    def evidence(self,path,query,**kwargs):
        return {"original_path":path,"original_sha256":"a"*64,"derived_sha256":"b"*64,
                "native_original_id":2 if path=="b.pdf" else 1,
                "selected_source_revision_and_native_id_rechecked":True,
                "pages":[{"physical_page":1,"windows":[]}]}

    def test_collective_check_after_all_reads(self):
        with mock.patch.object(bridge,"current_document_identity",side_effect=self.identity), \
             mock.patch.object(bridge,"current_document_passages",side_effect=self.evidence) as search, \
             mock.patch.object(bridge,"final_sources") as final, \
             mock.patch.object(bridge,"current_native_ids",return_value={"a.pdf":1,"b.pdf":2}):
            out=bridge.verified_lexical_pdf_evidence(receipt(),originals_per_query=2)
        self.assertEqual(["b.pdf","a.pdf"],[x.args[0] for x in search.call_args_list])
        self.assertEqual({"a.pdf","b.pdf"},set(final.call_args.args[0]))
        self.assertEqual(2,out["unique_selected_originals_rechecked"])
        self.assertTrue(out["independent_lexical_ranks_preserved"])

    def test_late_identity_change_refuses_whole_result(self):
        with mock.patch.object(bridge,"current_document_identity",side_effect=self.identity), \
             mock.patch.object(bridge,"current_document_passages",side_effect=self.evidence), \
             mock.patch.object(bridge,"final_sources"), \
             mock.patch.object(bridge,"current_native_ids",return_value={"b.pdf":5}):
            with self.assertRaises(ValueError):bridge.verified_lexical_pdf_evidence(receipt())

    def test_bounded_failure_preserves_source_slot_and_still_rechecks_identity(self):
        with mock.patch.object(bridge,"current_document_identity",side_effect=self.identity), \
             mock.patch.object(bridge,"current_document_passages",side_effect=BoundedPdfDiscoveryUnavailable("test limit")), \
             mock.patch.object(bridge,"final_sources") as final, \
             mock.patch.object(bridge,"current_native_ids",return_value={"b.pdf":2}):
            out=bridge.verified_lexical_pdf_evidence(receipt())
        row=out["queries"][0]["selected_originals"][0]
        self.assertEqual("complete_document_discovery_unavailable",row["status"])
        self.assertIsNone(row["verified_page_evidence"])
        final.assert_called_once()

if __name__=="__main__":
    import unittest
    unittest.main()
