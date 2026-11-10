"""Regressions for native facets that contain an entire list of source codes."""
from dataclasses import replace
from hashlib import sha256
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from file_system_catalog.hierarchical_summary import HierarchicalSummarizer
from file_system_catalog.hierarchical_summary_store import append_summary_rows, summary_rows
from test_hierarchical_summary import FakeSummaryModel
from test_hierarchical_summary_store import FakeEmbeddings, fixture


def digest(text):
    return sha256(text.encode()).hexdigest()


class BoundedEmbeddings(FakeEmbeddings):
    def __init__(self, long_texts):
        self.long_texts = long_texts
        self.embedded = []

    def model_context(self):
        return SimpleNamespace(configured_tokens=512)

    def token_count(self, text):
        return 513 if text in self.long_texts else 512

    def embed(self, texts):
        self.embedded.extend(texts)
        return super().embed(texts)


class IdentifierListTests(unittest.TestCase):
    def setUp(self):
        self.db = fixture()
        self.addCleanup(self.db.close)
        self.body = "[[SOURCE UNIT: code_label]]\nPrefix café. Codes include bb_two, aa_one and cc_three.\n"
        self.prefix = "KMD source provenance header_only\n"
        self.text = self.prefix + self.body
        self.group = "aa_one, bb_two, cc_three"

    def rows(self, group, *, kind="keyphrase", source=None, start=0, stop=None, extra_long=()):
        text = self.text if source is None else source
        tree = HierarchicalSummarizer(FakeSummaryModel(32000), output_tokens=64).summarize(
            title="Table", text=text[start:stop])
        root = tree.root
        analysis = {"document_summary": root.summary, "facets": [
            {"label": "Codes", "strength": "strong", "representations": [
                {"kind": kind, "strength": "strong", "text": group},
                {"kind": "description", "strength": "moderate", "text": "Neighbor"}]},
            {"label": "More", "strength": "strong", "representations": [
                {"kind": "keyword", "strength": "strong", "text": "Tail"}]},
        ]}
        self.embedding = BoundedEmbeddings({group, *extra_long})
        return summary_rows(
            self.db, original_path="books/guide.pdf", original_sha256="a"*64,
            original_bytes=12345, derived_sha256=digest(text), derived_text=text,
            summary_start_char=start, tree=replace(tree, nodes=(replace(root, native_analysis=analysis),)),
            embedding_client=self.embedding, analysis_model="local", collection_id="collection")

    def test_split_preserves_member_order_original_hash_and_exact_absolute_spans(self):
        chunks, rows = self.rows(self.group, start=len(self.prefix))
        self.assertEqual([r["global_rank"] for r in rows], list(range(8)))
        terms = rows[2:5]
        self.assertEqual([r["representation_text"] for r in terms], self.group.split(", "))
        self.assertEqual([r["item_rank"] for r in rows[1:]], [0,1,2,3,4,0,1])
        for ordinal, row in enumerate(terms, 1):
            member = json.loads(row["generation_json"])["identifier_list_member"]
            self.assertEqual(member["list_sha256"], digest(self.group))
            self.assertEqual((member["index"], member["count"]), (ordinal, 3))
            self.assertEqual(member["original_kind"], "keyphrase")
            self.assertEqual(self.text[member["start_char"]:member["end_char"]], row["representation_text"])
            self.assertEqual(row["representation_kind"], "keyword")
        self.assertNotIn(self.group, self.embedding.embedded)
        self.assertNotIn("identifier_list_member", json.loads(rows[-1]["generation_json"]))
        self.assertEqual(9, append_summary_rows(self.db, chunk_rows=chunks, representation_rows=rows)["new_rows"])
        repeat_chunks, repeat_rows = self.rows(self.group, start=len(self.prefix))
        self.assertEqual(0, append_summary_rows(self.db, chunk_rows=repeat_chunks, representation_rows=repeat_rows)["new_rows"])

    def test_only_complete_source_backed_identifier_lists_are_accepted(self):
        for group, kind in [
            ("aa_one, header_only", "keyphrase"), ("aa_one, code_label", "keyphrase"),
            ("aa_one, cc_thr", "keyphrase"), ("aa_one, bb_two, aa_one", "keyword"),
            ("aa_one, an unsupported condition", "keyphrase"), (self.group, "description"),
            ("aa_one, bb_two,", "keyphrase"), ("aa_one", "keyword"),
        ]:
            with self.subTest(group=group, kind=kind), self.assertRaises(ValueError):
                self.rows(group, kind=kind)
            self.assertEqual([], self.embedding.embedded)

    def test_source_span_edges_and_missing_unit_marker(self):
        for start, stop in [(self.text.index("bb_two")+1, None), (0, self.text.index("cc_three")+3)]:
            with self.subTest(start=start, stop=stop), self.assertRaises(ValueError):
                self.rows(self.group, start=start, stop=stop)
        with self.assertRaises(ValueError):
            self.rows(self.group, source="aa_one, bb_two, cc_three")

    def test_long_summary_long_member_and_total_facet_limit_still_fail(self):
        with self.assertRaisesRegex(ValueError, "summary exceeds embedding context"):
            self.rows(self.group, extra_long=["Indexable document context: sections and topics"])
        with self.assertRaises(ValueError):
            self.rows(self.group, extra_long=["bb_two"])
        terms = ", ".join(f"code_{i}" for i in range(254))
        with self.assertRaises(ValueError):
            self.rows(terms, source="[[SOURCE UNIT: list]]\n"+terms)
        self.assertEqual([], self.embedding.embedded)

    def test_combined_node_limit_includes_all_other_facets(self):
        # Four normal facet rows plus exactly 252 code rows fit the existing cap.
        terms = ", ".join(f"code_{i}" for i in range(252))
        _, rows = self.rows(terms, source="[[SOURCE UNIT: list]]\n"+terms)
        self.assertEqual(257, len(rows))  # One summary plus 256 native facets.
        self.assertEqual(list(range(257)), [r["global_rank"] for r in rows])

    def test_an_identifier_cannot_end_at_a_false_boundary(self):
        for ending in ("aa_onex", "aa_one_extra", "aa_oneé"):
            source = "[[SOURCE UNIT: codes]]\nbb_two and " + ending
            with self.subTest(ending=ending), self.assertRaisesRegex(ValueError, "absent"):
                self.rows("aa_one, bb_two", source=source, stop=source.index("aa_one")+6)

    def test_list_at_the_exact_model_limit_keeps_native_text_and_identity(self):
        with patch.object(BoundedEmbeddings, "token_count", return_value=512):
            _, rows = self.rows(self.group)
        self.assertEqual([r["global_rank"] for r in rows], list(range(6)))
        self.assertEqual(rows[2]["representation_text"], self.group)
        self.assertEqual(rows[2]["representation_kind"], "keyphrase")
        self.assertNotIn("identifier_list_member", json.loads(rows[2]["generation_json"]))
