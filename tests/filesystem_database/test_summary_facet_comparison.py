"""Contract tests for the bounded summary/facet comparison benchmark."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BENCHMARKS = Path(__file__).resolve().parents[2] / "scripts" / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import compare_summary_facets as benchmark


def hit(path, score, **overrides):
    return {
        "original_path": path, "original_sha256": "o", "derived_sha256": "d",
        "similarity": score, "native_representation_id": path,
        "source_start_char": 1, "source_end_char": 2,
        "source_passage_sha256": benchmark.library.sha("é".encode()),
        **overrides,
    }


class FacetComparisonTest(unittest.TestCase):
    def test_single_strong_keyword_cannot_displace_a_supported_document(self):
        summary = [hit("menu", 0.9), hit("recipe", 0.5)]
        facets = [hit("recipe", 1.0), hit("absent-from-summary-pool", 0.99)]
        rows = benchmark.supported_ranking(summary, facets)
        self.assertEqual(["menu", "recipe"], [r["anchor"]["original_path"] for r in rows])
        self.assertAlmostEqual(0.625, rows[1]["score"])
        self.assertEqual(facets[0], rows[1]["support"])

    def test_corroborated_result_can_move_up(self):
        summary = [hit("first", 0.81), hit("next", 0.8)]
        rows = benchmark.supported_ranking(summary, [hit("next", 0.9)])
        self.assertEqual("next", rows[0]["anchor"]["original_path"])
        self.assertAlmostEqual(0.825, rows[0]["score"])

    def test_stale_original_or_derivative_and_weaker_support_do_not_count(self):
        for override in ({"original_sha256": "other"}, {"derived_sha256": "other"},
                         {"similarity": -0.6}):
            with self.subTest(override=override):
                old = hit("same-path", -0.3)
                candidate = {**old, "similarity": 1.0, **override}
                ranked = benchmark.supported_ranking([old], [candidate])
                self.assertEqual(-0.3, ranked[0]["score"])
                self.assertIsNone(ranked[0]["support"])

    def test_cases_digest_and_invalid_input(self):
        case = {"id": "a", "group": "control", "query": "what", "expected_path": "guide"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            raw = json.dumps([case]).encode()
            path.write_bytes(raw)
            self.assertEqual(([case], hashlib.sha256(raw).hexdigest()),
                             benchmark.load_cases(path))
            for invalid in ([], {}, [case, case], [{}], [{**case, "query": "a" * 1001}]):
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    benchmark.load_cases(path)
            path.write_bytes(b" " * (1024 * 1024 + 1))
            with self.assertRaises(ValueError):
                benchmark.load_cases(path)

    def test_ties_keep_deterministic_path_order(self):
        summaries = [hit("z", 0.0), hit("a", 0.0)]
        rows = benchmark.supported_ranking(summaries, [])
        self.assertEqual(["a", "z"], [r["anchor"]["original_path"] for r in rows])
        self.assertEqual([], benchmark.supported_ranking([], summaries))

    def test_verification_checks_both_scoring_spans_and_the_source_revision(self):
        anchor = hit("menu", 0.8)
        support = {**anchor, "native_representation_id": "facet"}
        source = {**anchor, "derived_text": "aéz"}
        row = {"anchor": anchor, "support": support}
        results = [{"modes": {"supported": [row]}}]
        with patch.object(benchmark.library, "checked_original", return_value=source):
            self.assertEqual({"originals": 1, "distinct_representations": 2},
                             benchmark.verify_returned_sources(results))
            for error in ({"source_end_char": 4}, {"derived_sha256": "old"},
                          {"source_passage_sha256": "bad"}):
                with self.subTest(error=error):
                    row["support"] = {**support, **error}
                    with self.assertRaises(ValueError):
                        benchmark.verify_returned_sources(results)


if __name__ == "__main__":
    unittest.main()
