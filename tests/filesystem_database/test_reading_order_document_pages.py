from __future__ import annotations
import copy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase,mock
from file_system_catalog.reading_order_document_pages import (
    split_complete_reading_order_pages,read_complete_original_pdf_pages,
    read_original_pdf_page_range,validate_explicit_page_range,
    rank_document_page_candidates,verify_selected_document_pages,
    BoundedPdfDiscoveryUnavailable,MAX_PAGES,
)
from file_system_catalog.verified_pdf_neighbor_context import PhysicalPdfPageContextUnavailable
from test_reading_order_page_passages import context

def sha(b):return hashlib.sha256(b).hexdigest()

def scan(raw,count):
    return {"original_sha256":"a"*64,"complete_extraction_sha256":sha(raw),
            "physical_page_count":count,"pages":split_complete_reading_order_pages(raw,count)}

class BoundedWholePdfDiscoveryTests(TestCase):
    def test_physical_page_separators_exact_and_unavailable_pages_explicit(self):
        pages=split_complete_reading_order_pages(b'one\n\f\fno\x00good\f\xff\f',4)
        self.assertEqual([1,2,3,4],[x["physical_page"] for x in pages])
        self.assertEqual(["usable_candidate","page_text_empty","invalid_glyphs","invalid_utf8"],
                         [x["status"] for x in pages])
        self.assertEqual(sha(b"one\n\f"),pages[0]["reading_order_page_sha256"])
        self.assertIsNone(pages[1]["text"])
        for raw,count in ((b"a\f",2),(b"a\ftrailing",1),(b"a\f\f",1),(b"",1),(b"a\f",MAX_PAGES+1)):
            with self.subTest(raw=raw,count=count):
                with self.assertRaises(BoundedPdfDiscoveryUnavailable):
                    split_complete_reading_order_pages(raw,count)

    def test_original_byte_digest_and_real_fd_binding(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/"paper.pdf";data=b"%PDF-1.4 test only"
            p.write_bytes(data)
            with mock.patch("file_system_catalog.reading_order_document_pages._poppler",
                    side_effect=[b"Pages: 2\n",b"first\fsecond\f"]) as pop:
                out=read_complete_original_pdf_pages(p,sha(data))
            self.assertEqual(2,out["physical_page_count"])
            self.assertEqual(sha(b"first\fsecond\f"),out["complete_extraction_sha256"])
            fds=[call.args[1] for call in pop.call_args_list]
            self.assertEqual(fds[0],fds[1])
            self.assertIn("/proc/self/fd/",pop.call_args_list[0].args[0][-1])
            with self.assertRaises(ValueError):read_complete_original_pdf_pages(p,"b"*64)

    def test_embedded_form_feed_recovered_only_with_explicit_page_reads(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/"paper.pdf";data=b"%PDF-1.4 test only";p.write_bytes(data)
            module="file_system_catalog.reading_order_document_pages._poppler"
            with mock.patch(module,side_effect=[b"Pages: 2\n",b"first\finside\fsecond\f",
                    b"first\finside\f",b"second\f"]) as pop:
                out=read_complete_original_pdf_pages(p,sha(data))
            self.assertEqual("explicit_numbered_physical_page_reads_after_separator_mismatch",
                             out["extraction_method"])
            self.assertEqual("first\finside\f",out["pages"][0]["text"])
            self.assertEqual(sha(b"first\finside\f"),out["pages"][0]["reading_order_page_sha256"])
            self.assertEqual(2,len(out["pages"]))
            self.assertIn("-f",pop.call_args_list[2].args[0])
            self.assertEqual("1",pop.call_args_list[2].args[0][2])
            with mock.patch(module,side_effect=[b"Pages: 2\n",b"first\finside\fsecond\f",
                    b"first\finside\f",None]):
                with self.assertRaises(BoundedPdfDiscoveryUnavailable):
                    read_complete_original_pdf_pages(p,sha(data))

    def test_changed_original_while_extracting_fails_closed(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/"paper.pdf";data=b"%PDF-1.4 test only";p.write_bytes(data)
            def changed(args,fd,deadline):
                if "pdfinfo" in args[0]:return b"Pages: 1\n"
                p.write_bytes(b"%PDF-1.4 changed")
                return b"answer\f"
            with mock.patch("file_system_catalog.reading_order_document_pages._poppler",side_effect=changed):
                with self.assertRaisesRegex(ValueError,"changed during"):
                    read_complete_original_pdf_pages(p,sha(data))

    def test_rank_uses_shared_document_term_weights_not_old_page_labels(self):
        s=scan(b"A broad unrelated comparison.\fManual annotation is not required for the method. "
               b"It obtains temporal alignment.\fA different annotation method.\f",3)
        result=rank_document_page_candidates(s,"manually annotated temporal alignment",top_pages=2)
        self.assertEqual(2,result["private_candidates"][0]["physical_page"])
        self.assertEqual(3,result["physical_page_count"])
        self.assertEqual(3,result["usable_machine_text_pages"])
        self.assertEqual(["alignment","annotated","manually","temporal"],sorted(result["term_weights"]))
        self.assertIn("manually",result["private_candidates"][0]["windows"][0]["matched_query_inflection_terms"])
        self.assertIn("not required",result["private_candidates"][0]["windows"][0]["text"])

    def test_invalid_page_candidate_proof_refused(self):
        s=scan(b"Valid text.\fOther text.\f",2)
        for modify in (
            lambda d:d["pages"][0].update(physical_page=True),
            lambda d:d["pages"][0].update(text="Forged."),
            lambda d:d["pages"].pop(),
        ):
            d=copy.deepcopy(s);modify(d)
            with self.assertRaises(ValueError):rank_document_page_candidates(d,"text")

class IndependentPageConfirmationTests(TestCase):
    def setUp(self):
        self.raw=b"Unrelated.\fA measurement applies only under condition A.\f"
        self.scan=scan(self.raw,2)
        self.ranked=rank_document_page_candidates(self.scan,"measurement condition",top_pages=1)
        self.source={"original_path":"approved/paper.pdf","original_sha256":"a"*64,"current_derived_sha256":"b"*64}
        self.ctx=context("A measurement applies only under condition A.\f")
        b=self.ctx["physical_pages"];b["requested_center_physical_page"]=2;b["actual_original_pdf_page_count"]=2
        b["pages"][0]["physical_page"]=2;b["optional_center_reading_order_projection"]["physical_page"]=2

    def verify(self):
        return verify_selected_document_pages(self.ranked,source=self.source,
                  read_context=mock.Mock(return_value=self.ctx))

    def test_exact_page_offsets_reread_required(self):
        result=self.verify()
        p=result["pages"][0]
        self.assertEqual(2,p["physical_page"])
        self.assertEqual("verified_reading_order_windows",p["status"])
        self.assertTrue(result["independent_original_physical_page_reread_required"])
        self.assertFalse(result["visual_reading_order_or_factual_claim_truth_verified"])
        self.assertNotIn("private_candidates",result)
        self.assertEqual(sha(p["windows"][0]["text"].encode()),p["windows"][0]["span_sha256"])

    def test_mismatching_page_read_or_window_bytes_fail(self):
        for modify in (
            lambda s:s.ctx["source"].update(current_derived_sha256="d"*64),
            lambda s:s.ctx["physical_pages"].update(actual_original_pdf_page_count=3),
            lambda s:s.ranked["private_candidates"][0].update(reading_order_page_sha256="d"*64),
            lambda s:s.ranked["private_candidates"][0]["windows"][0].update(text="Different"),
            lambda s:s.ranked["private_candidates"][0].update(page_candidate_rank=True),
            lambda s:s.ctx.update(source_map_revision_reverified_after_physical_page_extraction=False),
        ):
            with self.subTest(modify=modify):
                self.setUp();modify(self)
                with self.assertRaises(ValueError):self.verify()

    def test_unavailable_selected_page_remains_slot_without_text(self):
        r=verify_selected_document_pages(self.ranked,source=self.source,
               read_context=mock.Mock(side_effect=PhysicalPdfPageContextUnavailable("test")))
        self.assertEqual("physical_page_context_unavailable",r["pages"][0]["status"])
        self.assertEqual([],r["pages"][0]["windows"])

class ExplicitBookRangeTests(TestCase):
    def test_range_is_exact_physical_not_printed_page_and_never_full_document(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/"book.pdf";data=b"%PDF-1.4 fixture"
            p.write_bytes(data)
            raw=b"physical 326\f\fphysical 328\f"
            mod="file_system_catalog.reading_order_document_pages._poppler"
            with mock.patch(mod,side_effect=[b"Pages: 417\n",raw]) as pop:
                out=read_original_pdf_page_range(p,sha(data),first_page=326,last_page=328)
            self.assertEqual(417,out["physical_page_count"])
            self.assertEqual([326,327,328],[x["physical_page"] for x in out["pages"]])
            self.assertEqual("page_text_empty",out["pages"][1]["status"])
            self.assertEqual(sha(raw),out["range_extraction_sha256"])
            self.assertNotIn("complete_extraction_sha256",out)
            self.assertEqual("explicit_physical_page_range",out["scan_scope"])
            self.assertEqual(["-f","326","-l","328"],pop.call_args_list[1].args[0][1:5])
            with mock.patch(mod,return_value=b"Pages: 417\n") as full:
                with self.assertRaises(BoundedPdfDiscoveryUnavailable):
                    read_complete_original_pdf_pages(p,sha(data))
            full.assert_called_once()

    def test_range_separator_failure_rereads_only_requested_physical_pages(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/"book.pdf";data=b"%PDF-1.4 fixture";p.write_bytes(data)
            raw1=b"first\fembedded\f";raw2=b"second\f"
            mod="file_system_catalog.reading_order_document_pages._poppler"
            with mock.patch(mod,side_effect=[b"Pages: 417\n",raw1+raw2,raw1,raw2]) as pop:
                out=read_original_pdf_page_range(p,sha(data),first_page=326,last_page=327)
            self.assertEqual([326,327],[x["physical_page"] for x in out["pages"]])
            self.assertEqual(sha(raw1),out["pages"][0]["reading_order_page_sha256"])
            self.assertEqual(["-f","326","-l","326"],pop.call_args_list[2].args[0][1:5])
            self.assertEqual(["-f","327","-l","327"],pop.call_args_list[3].args[0][1:5])

    def test_invalid_ranges_rejected_without_pdf_extraction(self):
        for first,last in ((None,2),(1,None),(True,2),(0,2),(3,2),(1,65),(999999,1000000)):
            with self.subTest(first=first,last=last):
                with self.assertRaises(ValueError):validate_explicit_page_range(first,last)
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/"book.pdf";data=b"%PDF-1.4 fixture";p.write_bytes(data)
            with mock.patch("file_system_catalog.reading_order_document_pages._poppler",
                       return_value=b"Pages: 417\n") as pop:
                with self.assertRaisesRegex(ValueError,"exceeds actual"):
                    read_original_pdf_page_range(p,sha(data),first_page=417,last_page=418)
            pop.assert_called_once()

    def range_scan(self):
        raw=b"Unrelated.\fA measurement applies only under condition A.\f"
        result=scan(raw,2)
        result.update(physical_page_count=417,scan_scope="explicit_physical_page_range",
                      scanned_first_physical_page=326,scanned_last_physical_page=327,
                      range_extraction_sha256=result.pop("complete_extraction_sha256"))
        for n,x in enumerate(result["pages"],326):x["physical_page"]=n
        return result

    def test_ranked_range_retains_absolute_coordinates_and_unsearched_count(self):
        r=rank_document_page_candidates(self.range_scan(),"measurement condition",top_pages=1)
        self.assertEqual(327,r["private_candidates"][0]["physical_page"])
        self.assertEqual(417,r["physical_page_count"])
        self.assertEqual(2,r["scanned_physical_page_count"])
        self.assertEqual(415,r["physical_pages_outside_requested_range"])
        for change in (
            lambda d:d.update(scanned_first_physical_page=325),
            lambda d:d["pages"][0].update(physical_page=1),
            lambda d:d.update(complete_extraction_sha256="a"*64),
            lambda d:d.pop("scanned_last_physical_page"),
            lambda d:d.update(range_extraction_sha256=None),
        ):
            with self.subTest(change=change):
                d=self.range_scan();change(d)
                with self.assertRaises(ValueError):rank_document_page_candidates(d,"measurement")

    def test_verify_range_refuses_valid_physical_page_outside_requested_scope(self):
        r=rank_document_page_candidates(self.range_scan(),"measurement condition",top_pages=1)
        src={"original_path":"approved/book.pdf","original_sha256":"a"*64,
             "current_derived_sha256":"b"*64}
        ctx=context("A measurement applies only under condition A.\f")
        ctx["source"]["original_path"]=src["original_path"]
        b=ctx["physical_pages"];b["requested_center_physical_page"]=327
        b["actual_original_pdf_page_count"]=417
        b["pages"][0]["physical_page"]=327
        b["optional_center_reading_order_projection"]["physical_page"]=327
        reader=mock.Mock(return_value=ctx)
        out=verify_selected_document_pages(r,source=src,read_context=reader)
        self.assertEqual(327,out["pages"][0]["physical_page"])
        self.assertEqual(415,out["physical_pages_outside_requested_range"])
        reader.assert_called_once_with("approved/book.pdf",327)
        for change in (
            lambda d:d["private_candidates"][0].update(physical_page=328),
            lambda d:d.update(scanned_physical_page_count=3),
            lambda d:d.update(physical_pages_outside_requested_range=414),
        ):
            with self.subTest(change=change):
                d=copy.deepcopy(r);change(d)
                reader.reset_mock()
                with self.assertRaises(ValueError):
                    verify_selected_document_pages(d,source=src,read_context=reader)
                reader.assert_not_called()

if __name__=="__main__":
    import unittest
    unittest.main()
