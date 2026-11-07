"""Existing native KnowMoreDiRT facet contract used by future summary bridge.

No new keyword type/schema/database: this verifies the original content
pipeline already normalizes and flattens combined summary/keyword output.
"""
from __future__ import annotations

import unittest

from file_system_catalog.content_pipeline import (
    FILE_ANALYSIS_SCHEMA,
    _representation_schema,
    normalize_analysis,
    flatten_representations,
)
from file_system_catalog.content_schema import REPRESENTATION_KIND_VALUES


class ExistingNativeFacetContract(unittest.TestCase):
    def test_native_closed_json_schema_supports_keyword_and_keyphrase(self):
        self.assertEqual(
            {"document_summary", "facets"},
            set(FILE_ANALYSIS_SCHEMA["required"]),
        )
        kinds = _representation_schema()["properties"]["kind"]["enum"]
        for kind in ("keyword", "keyphrase", "entity", "description", "sentence", "topic"):
            self.assertIn(kind, kinds)
            self.assertIn(kind, REPRESENTATION_KIND_VALUES)

    def test_existing_native_normalizer_and_flatten_preserve_concurrent_fields(self):
        real_shape = {
            "document_summary": (
                "Hick's Law relates the number of choices to human decision time."
            ),
            "facets": [
                {
                    "facet_name": "decision time and choice complexity",
                    "facet_strength": "essential",
                    "representations": [
                        {"kind": "keyword", "item_strength": "strong",
                         "text": "decision time"},
                        {"kind": "keyword", "item_strength": "strong",
                         "text": "cognitive load"},
                        {"kind": "keyphrase", "item_strength": "strong",
                         "text": "Hick-Hyman Law"},
                        {"kind": "description", "item_strength": "strong",
                         "text": "Decision time increases with the complexity of alternatives."},
                    ],
                },
            ],
        }
        validated = normalize_analysis(real_shape)
        flattened = flatten_representations(validated)
        self.assertEqual(real_shape["document_summary"], flattened[0]["text"])
        self.assertEqual("summary", flattened[0]["kind"])
        self.assertEqual(2, sum(r["kind"] == "keyword" for r in flattened))
        self.assertEqual(1, sum(r["kind"] == "keyphrase" for r in flattened))
        self.assertEqual(1, sum(r["kind"] == "topic" for r in flattened))
        self.assertEqual(1, sum(r["kind"] == "description" for r in flattened))
        self.assertEqual(len(flattened), len({r["global_rank"] for r in flattened}))

    def test_native_normalization_rejects_non_source_or_duplicate_labels(self):
        raw = {
            "document_summary": "One true document.",
            "facets": [
                {"facet_name": "technical source",
                 "facet_strength": "strong",
                 "representations": [
                     {"kind": "keyword", "item_strength": "strong", "text": "algorithms"},
                     {"kind": "keyword", "item_strength": "strong", "text": "algorithms"},
                 ]},
                {"facet_name": "technical source",
                 "facet_strength": "weak",
                 "representations": [
                     {"kind": "keyword", "item_strength": "weak", "text": "algorithms"},
                 ]},
            ],
        }
        normalized = normalize_analysis(raw)
        self.assertEqual(1, len(normalized["facets"]))
        self.assertEqual(
            ["algorithms"],
            [r["text"] for r in normalized["facets"][0]["representations"]],
        )


if __name__ == "__main__":
    unittest.main()
