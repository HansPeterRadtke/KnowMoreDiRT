"""Complete-document hierarchical summary and context-budget regression tests."""
from __future__ import annotations

import hashlib
import json
import random
import unittest
from unittest.mock import patch

from file_system_catalog.hierarchical_summary import (
    HierarchicalSummarizer, SUMMARY_SYSTEM, Section, SummaryInput,
    split_sections,
)
from file_system_catalog.hierarchical_summary_model import LocalSummaryModel


class FakeSummaryModel:
    def __init__(self, context_tokens=360):
        self._context = context_tokens
        self.calls = []

    @property
    def context_tokens(self):
        return self._context

    def prompt_tokens(self, *, system, user):
        # Deterministic stand-in for a correctly measured model prompt.
        return (len(system) + len(user) + 3) // 4

    def text_tokens(self, text):
        return (len(text) + 3) // 4

    def generate_summary(self, *, system, user, output_tokens):
        self.calls.append((system, user, output_tokens))
        assert "untrusted source DATA" in system
        assert "SOURCE_BEGIN" in user and "SOURCE_END" in user
        # Fake semantic summarization only tests information routing, not quality.
        distinctive = [v for v in ("GUI_CHAPTER_28", "TIMMY_DREAM", "NEGATED_CLAIM")
                       if v in user]
        return "Indexable document context: " + (", ".join(distinctive) or "sections and topics")


def long_document(pages=75, length=150):
    return "".join(
        f"[[SOURCE UNIT: PDF page {n}]]\n"
        + ("Chapter 28 GUI_CHAPTER_28: graphical interface design. " if n == 37 else "")
        + ("Timmy described a TIMMY_DREAM with a NEGATED_CLAIM. " if n == 44 else "")
        + ("Substantive explanatory source text. " + "X" * length + "\n")
        for n in range(1, pages + 1)
    )


class HierarchicalSummaryTest(unittest.TestCase):
    def test_split_source_pages_keeps_every_character_and_marker(self):
        text = "KMD provenance header\n" + long_document(8, 8)
        sections = split_sections(text)
        self.assertEqual(text, "".join(s.text for s in sections))
        self.assertEqual(0, sections[0].start_char)
        self.assertEqual(len(text), sections[-1].end_char)
        self.assertEqual(len(sections), 9)
        self.assertIn("PDF page 4", sections[4].label)

    def test_chapter_structure_is_used_if_page_markers_are_absent(self):
        text = "Preface\n" + "Chapter 1: One\nIntro\n" + "Chapter 2: Two\nEnd\n"
        pieces = split_sections(text)
        self.assertEqual(3, len(pieces))
        self.assertIn("Chapter 2", pieces[-1].label)
        self.assertEqual(text, "".join(p.text for p in pieces))

    def test_complete_hierarchical_tree_and_source_links(self):
        model = FakeSummaryModel(context_tokens=750)
        engine = HierarchicalSummarizer(model, output_tokens=64, safety_tokens=16)
        text = long_document(75, 140)
        sections, planned = engine.plan_source(title="Complete book", text=text)
        self.assertEqual(text, "".join(s.text for group in planned for s in group))
        self.assertGreater(len(planned), 2)
        for group in planned:
            self.assertTrue(engine._fits(title="Complete book", level=0, sections=group))
        tree = engine.summarize(title="Complete book", text=text)
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), tree.source_sha256)
        self.assertEqual(len(text), tree.original_characters)
        self.assertEqual(75, tree.leaf_sections)
        self.assertGreater(len(set(n.level for n in tree.nodes)), 1)
        self.assertEqual((0, len(text)), (tree.root.start_char, tree.root.end_char))
        self.assertIn("GUI_CHAPTER_28", tree.root.summary)
        self.assertIn("TIMMY_DREAM", tree.root.summary)
        self.assertIn("NEGATED_CLAIM", tree.root.summary)
        known = {n.node_id: n for n in tree.nodes}
        for node in tree.nodes:
            self.assertLessEqual(node.prompt_tokens + node.allocated_output_tokens + 16, model.context_tokens)
            self.assertEqual(node.summary_sha256, hashlib.sha256(node.summary.encode()).hexdigest())
            self.assertEqual(node.source_sha256, tree.source_sha256)
            if node.child_ids:
                children = [known[x] for x in node.child_ids]
                self.assertEqual((node.start_char, node.end_char),
                                 (children[0].start_char, children[-1].end_char))
        self.assertEqual(len(model.calls), len(tree.nodes))

    def test_oversized_single_page_splits_without_losing_a_character(self):
        text = "[[SOURCE UNIT: PDF page 1]]\n" + "Words are separated by spaces. " * 400
        engine = HierarchicalSummarizer(FakeSummaryModel(context_tokens=360), output_tokens=64, safety_tokens=16)
        source, groups = engine.plan_source(title="long page", text=text)
        self.assertEqual(1, len(source))
        self.assertGreater(len(groups), 1)
        self.assertEqual(text, "".join(x.text for g in groups for x in g))
        self.assertEqual(0, groups[0][0].start_char)
        self.assertEqual(len(text), groups[-1][-1].end_char)

    def test_incomplete_model_budget_does_not_claim_a_summary(self):
        model = FakeSummaryModel(360)
        engine = HierarchicalSummarizer(model, output_tokens=64, safety_tokens=16)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            engine.summarize(title="long", text=long_document(20), max_calls=1)

    def test_document_input_validation_and_zero_length(self):
        with self.assertRaises(ValueError):
            split_sections("")
        with self.assertRaises(ValueError):
            Section(0, 99, "bad", "short")
        with self.assertRaises(ValueError):
            HierarchicalSummarizer(FakeSummaryModel(360), output_tokens=130)
        with self.assertRaises(ValueError):
            HierarchicalSummarizer(FakeSummaryModel(360), output_tokens=64).plan_source(
                title="x" * 301, text="Document")
        with self.assertRaises(ValueError):
            SummaryInput(10, 5, "bad", "text")

    def test_parent_summary_payload_keeps_original_offsets(self):
        model = FakeSummaryModel(360)
        engine = HierarchicalSummarizer(model, output_tokens=64, safety_tokens=16)
        piece = SummaryInput(200, 1200, "Chapter 17", "A brief child summary")
        self.assertTrue(engine._fits(title="A book", level=1, sections=[piece]))
        prompt = engine._prompt(title="A book", level=1, nodes=[piece])
        self.assertIn("200:1200", prompt)
        self.assertIn("Chapter 17", prompt)

    def test_model_outputs_must_not_be_silent_or_overbudget(self):
        engine = HierarchicalSummarizer(FakeSummaryModel(360), output_tokens=64, safety_tokens=16)
        with patch.object(engine.model, "generate_summary", return_value=""):
            with self.assertRaisesRegex(RuntimeError, "usable summary"):
                engine.summarize(title="small", text="a few words")
        with patch.object(engine.model, "generate_summary", return_value="word " * 100):
            with self.assertRaisesRegex(RuntimeError, "output reservation"):
                engine.summarize(title="small", text="a few words")

    def test_real_client_uses_explicit_max_tokens_and_private_cache(self):
        model = LocalSummaryModel("http://127.0.0.1:14829", model="mock")
        with patch.object(model.client, "prompt_token_count", return_value=20), \
             patch.object(model.client, "model_context") as context, \
             patch.object(model.client, "_render_prompt", return_value="prompt"), \
             patch.object(model.client, "effective_model", return_value="mock"), \
             patch("file_system_catalog.hierarchical_summary_model.read_model_call", return_value=None), \
             patch("file_system_catalog.hierarchical_summary_model.write_model_call") as save, \
             patch("file_system_catalog.hierarchical_summary_model.stream_chat_completion_json",
                   return_value={"content": '{"summary":"Grounded text"}', "finish_reason":"stop"}) as call, \
             patch.object(model.client, "token_count", return_value=6):
            context.return_value.configured_tokens = 1000
            result = model.generate_summary(system="prompt", user="data", output_tokens=100)
        self.assertEqual(result, "Grounded text")
        self.assertEqual(call.call_args.args[0], "http://127.0.0.1:14829/v1/chat/completions")
        self.assertEqual(call.call_args.args[1]["max_tokens"], 100)
        save.assert_called_once()

    def test_model_rejects_external_api_urls(self):
        with self.assertRaises(ValueError):
            LocalSummaryModel("https://api.some-cloud.com", model="cloud")


if __name__ == "__main__":
    unittest.main()
