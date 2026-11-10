"""Recovery from a native JSON response that exhausts the compact worker budget."""
import json
import unittest
from unittest.mock import patch

from file_system_catalog.hierarchical_summary import (
    HierarchicalSummarizer, SUMMARY_FACETS_COMPACT_SYSTEM,
)
from test_hierarchical_native_facets import FakeNativeFacetModel

LENGTH = "local summary did not finish within reserved generation tokens"


class CompactFacetRetryTest(unittest.TestCase):
    def setUp(self):
        self.model = FakeNativeFacetModel(16384)
        self.engine = HierarchicalSummarizer(
            self.model, output_tokens=2400, safety_tokens=384,
            native_facets=True, native_facets_compact=True,
        )
        self.source = "Documented procedure A applies only after B. Never use it for C."

    def run_source(self, on_node=None):
        return self.engine.summarize(title="Procedure", text=self.source, on_node=on_node)

    def test_cutoff_changes_request_and_retains_source_and_budget(self):
        generate = self.model.generate_retrieval_analysis
        def reject_verbose(**args):
            if args["system"] == SUMMARY_FACETS_COMPACT_SYSTEM:
                raise RuntimeError(LENGTH)
            return generate(**args)

        with patch.object(self.model, "generate_retrieval_analysis", side_effect=reject_verbose) as calls:
            tree = self.run_source()
        first, retry = [c.kwargs for c in calls.call_args_list]
        self.assertNotEqual(first["system"], retry["system"])
        self.assertEqual(first["user"], retry["user"])
        self.assertIn(self.source, retry["user"])
        self.assertEqual(2400, first["output_tokens"])
        self.assertEqual(2400, retry["output_tokens"])
        self.assertLess(tree.root.prompt_tokens, self.model.prompt_tokens(
            system=first["system"], user=first["user"]))
        self.assertEqual(tree.root.prompt_tokens, self.model.prompt_tokens(
            system=retry["system"], user=retry["user"]))
        self.assertLessEqual(tree.root.prompt_tokens + tree.root.allocated_output_tokens + 384,
                             self.model.context_tokens)
        self.assertEqual("native_facets_compact_bounded_after_length", tree.root.prompt_policy)
        self.assertEqual((0, len(self.source)), (tree.root.start_char, tree.root.end_char))
        self.assertEqual(tree.root.summary, tree.root.native_analysis["document_summary"])

    def test_successful_compact_and_unrelated_failures_do_not_take_fallback(self):
        tree = self.run_source()
        self.assertEqual("native_facets_compact", tree.root.prompt_policy)
        self.assertEqual(1, len(self.model.calls))
        with patch.object(self.model, "generate_retrieval_analysis",
                          side_effect=RuntimeError("source checksum mismatch")) as calls:
            with self.assertRaisesRegex(RuntimeError, "source checksum"):
                self.run_source()
            self.assertEqual(1, calls.call_count)

    def test_native_transport_and_cache_use_distinct_requests(self):
        import test_native_facet_model as native_fixture
        model = native_fixture.ControlledNativeFacetTest().fake_model()
        engine = HierarchicalSummarizer(
            model, output_tokens=2400, safety_tokens=384,
            native_facets=True, native_facets_compact=True,
        )
        generated = {
            "document_summary": "Procedure A requires B and excludes C.",
            "facets": [{"facet_name": "requirements", "facet_strength": "essential",
                        "representations": [{"kind": "keyphrase", "item_strength": "strong",
                                             "text": "Procedure A requires B"}]}],
        }
        module = "file_system_catalog.hierarchical_summary_model."
        responses = [{"finish_reason": "length", "content": "{"},
                     {"finish_reason": "stop", "content": json.dumps(generated)}]
        with patch(module + "stream_chat_completion_json", side_effect=responses) as transport, \
                patch(module + "read_model_call", return_value=None) as cache_reads, \
                patch(module + "write_model_call") as cache_write:
            tree = engine.summarize(title="Procedure", text=self.source)
        first, second = [call.args[1] for call in transport.call_args_list]
        self.assertNotEqual(first["messages"][0], second["messages"][0])
        self.assertEqual(first["messages"][1:], second["messages"][1:])
        self.assertEqual(first["response_format"], second["response_format"])
        self.assertEqual(first["max_tokens"], second["max_tokens"])
        first_key, second_key = [call.args[0] for call in cache_reads.call_args_list]
        self.assertNotEqual(first_key, second_key)
        cache_write.assert_called_once_with(second_key, {"native_analysis": tree.root.native_analysis})

    def test_second_cutoff_fails_without_emitting_an_incomplete_node(self):
        emitted = []
        with patch.object(self.model, "generate_retrieval_analysis",
                          side_effect=RuntimeError(LENGTH)) as calls:
            with self.assertRaisesRegex(RuntimeError, LENGTH):
                self.run_source(lambda node, count: emitted.append(node))
        self.assertEqual(2, calls.call_count)
        self.assertEqual([], emitted)
