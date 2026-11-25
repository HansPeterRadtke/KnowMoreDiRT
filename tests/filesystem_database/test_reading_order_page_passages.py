from __future__ import annotations
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from file_system_catalog.reading_order_page_passages import (
    reading_order_page_passages, _numeric_keys, _porter_tokens, _sentence_units,
)

def digest(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def context(body):
    h=digest(body)
    return {"source_map_revision_reverified_after_physical_page_extraction":True,
        "source":{"original_path":"approved/paper.pdf","title":"Paper","source_url":"https://example.invalid",
                  "original_sha256":"a"*64,"current_derived_sha256":"b"*64,
                  "publisher_original_actual_sha256_verified":True,
                  "source_map_derivative_actual_sha256_verified":True},
        "physical_pages":{"original_pdf_sha256":"a"*64,"requested_center_physical_page":3,
            "actual_original_pdf_page_count":9,"neighbor_radius":0,
            "original_pdf_revision_verified_before_and_after":True,
            "pages":[{"physical_page":3,"status":"verified_original_page_text",
                      "physical_page_poppler_sha256":"c"*64}],
            "optional_center_reading_order_projection":{"physical_page":3,
                "status":"same_physical_page_alternate_reading_order_text",
                "extraction_mode":"poppler_default_reading_order_utf8_not_layout",
                "exact_layout_page_poppler_sha256":"c"*64,
                "complete_extractor_output_not_truncated":True,
                "reading_order_poppler_sha256":h,"extracted_text_sha256":h,
                "extracted_page_text":body}}}

class PagePassagesTests(unittest.TestCase):
    def test_exact_contiguous_source_span_and_numeric_context(self):
        body=("Prior conditions are different. \nThe measured improvement was 52.00% "
              "under protocol B, not under protocol A. This does not establish an average. "
              "Unrelated material. More unrelated material. Another sentence.")
        c=context(body)
        r=reading_order_page_passages(c,"measured improvement 52 percent protocol B",max_characters=160)
        p=r["passages"][0];normalized=" ".join(body.split())
        self.assertEqual(normalized[p["start_char"]:p["end_char"]],p["text"])
        self.assertEqual(digest(p["text"]),p["span_sha256"])
        self.assertEqual(digest(normalized),r["normalized_reading_order_page_sha256"])
        self.assertIn("not under protocol A.",p["text"])
        self.assertIn("52%",p["matched_query_number_literals"])
        self.assertLessEqual(len(p["text"]),160)
        self.assertTrue(r["native_document_ranking_unchanged"])
        self.assertFalse(r["visual_reading_order_or_factual_claim_truth_verified"])

    def test_numeric_values_are_not_rounded_or_unit_substituted(self):
        self.assertEqual(_numeric_keys("52 percent; 52.00%; 1.69 times; 1.69×"),{"52%","1.69×"})
        self.assertEqual(_numeric_keys("5. -5 −5 +5 18.11%p and 18.11 percent and 10 percentage points."),
                         {"5","-5","18.11%p","18.11%","10%p"})
        self.assertEqual(_numeric_keys("1e-5; Qwen2.5-VL-32B"),set())
        self.assertEqual(_numeric_keys("1234567890123456789012345678901"),{"1234567890123456789012345678901"})

    def test_optional_inflection_only_never_changes_source_text(self):
        body="A method obtains results without manual annotation. Actions are extracted from recordings."
        exact=reading_order_page_passages(context(body),"manually annotated",top=1)
        self.assertEqual(exact["passages"],[])
        stem=reading_order_page_passages(context(body),"manually annotated",top=1,
                                        include_inflection_matches=True)
        p=stem["passages"][0]
        self.assertEqual(p["matched_query_terms"],[])
        self.assertEqual(p["matched_query_inflection_terms"],["annotated","manually"])
        self.assertEqual(p["text"],body)
        self.assertEqual(_porter_tokens({"annotation","annotated","manual","manually","id_1","Qwen2"}),
                         {"annotation":"annot","annotated":"annot","manual":"manual","manually":"manual"})

    def test_long_unit_partial_explicit_and_word_aligned(self):
        body=" ".join(["unrelated"]*60+["decisive","measurement","result"]+["other"]*60)
        r=reading_order_page_passages(context(body),"decisive measurement",max_characters=160)
        self.assertEqual(r["overlong_heuristic_units"],1)
        self.assertTrue(r["passages"])
        for p in r["passages"]:
            self.assertTrue(p["partial_overlong_heuristic_sentence"])
            self.assertFalse(p["heuristic_sentence_window"])
            self.assertLessEqual(len(p["text"]),160)
            self.assertEqual(body[p["start_char"]:p["end_char"]],p["text"])
            self.assertTrue(p["start_char"]==0 or body[p["start_char"]-1]==" ")
            self.assertTrue(p["end_char"]==len(body) or body[p["end_char"]]==" ")

    def test_boundaries_keep_decimal_and_abbreviation_context(self):
        text="Dr. Doe measured 1.69 times throughput. The result was not universal."
        self.assertEqual([text[s:e] for s,e in _sentence_units(text)],
                         ["Dr. Doe measured 1.69 times throughput.","The result was not universal."])

    def test_fail_closed_on_wrong_or_truncated_page_proof(self):
        c=context("A measurement is reported.")
        for edit in (
            lambda x:x["source"].update(original_sha256="d"*64),
            lambda x:x["physical_pages"].update(requested_center_physical_page=4),
            lambda x:x["physical_pages"].update(neighbor_radius=1),
            lambda x:x["physical_pages"]["pages"][0].update(physical_page_poppler_sha256=None),
            lambda x:x["physical_pages"]["optional_center_reading_order_projection"].update(complete_extractor_output_not_truncated=False),
            lambda x:x["physical_pages"]["optional_center_reading_order_projection"].update(extracted_page_text="Other text."),
            lambda x:x.update(source_map_revision_reverified_after_physical_page_extraction=False),
        ):
            with self.subTest(edit=edit):
                a=copy.deepcopy(c);edit(a)
                with self.assertRaises(ValueError):reading_order_page_passages(a,"measurement")

    def test_determinism_and_no_match(self):
        c=context("A distinct measurement was reported. A second measurement differs. A third unrelated sentence.")
        self.assertEqual(reading_order_page_passages(c,"measurement"),
                         reading_order_page_passages(c,"measurement"))
        self.assertEqual(reading_order_page_passages(c,"absent zyzzyva")["passages"],[])
        for args in ({"top":True},{"top":6},{"max_characters":159},{"include_inflection_matches":"yes"}):
            with self.assertRaises(ValueError):reading_order_page_passages(c,"measurement",**args)

    def test_cli_rechecks_native_and_source_identity_and_prior_digests(self):
        script=Path(__file__).resolve().parents[2]/"scripts/raspi_original_pdf_page_passages.py"
        spec=importlib.util.spec_from_file_location("test_page_passages_cli",script)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        c=context("A measurement is reported.");path=c["source"]["original_path"]
        h=c["physical_pages"]["optional_center_reading_order_projection"]["reading_order_poppler_sha256"]
        with patch.object(mod,"current_native_ids",return_value={path:7}) as ids, \
             patch.object(mod,"original_pdf_context",return_value=c),patch.object(mod,"final_sources") as guard:
            r=mod.current_page_passages(path,3,"measurement",expected_identity=("a"*64,"b"*64,h))
            self.assertEqual(r["native_original_id"],7)
            self.assertTrue(r["selected_source_revision_and_native_id_rechecked"])
            self.assertEqual(ids.call_count,2);guard.assert_called_once_with({path:c["source"]})
            with self.assertRaises(ValueError):mod.current_page_passages(path,3,"measurement",
                                                    expected_identity=("d"*64,"b"*64,h))
        with patch.object(mod,"current_native_ids",side_effect=[{path:7},{path:8}]), \
             patch.object(mod,"original_pdf_context",return_value=c),patch.object(mod,"final_sources"):
            with self.assertRaisesRegex(ValueError,"native source identity changed"):
                mod.current_page_passages(path,3,"measurement")

if __name__=="__main__":unittest.main()
