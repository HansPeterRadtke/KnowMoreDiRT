"""Embedding checks on native generation, saved tree recovery and atomic publication."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from file_system_catalog.hierarchical_summary import (
    EmbeddingContextExceeded, HierarchicalSummarizer, SummaryActivity, SUMMARY_FACETS_COMPACT_SYSTEM,
)
from file_system_catalog.hierarchical_summary_store import prepare_node_facets, retain_fitting_native_fields
from test_hierarchical_native_facets import FakeNativeFacetModel
from test_summary_search_audit import load_cli


class CharacterBudget:
    limit = 150

    def token_count(self, text):
        return len(text)


def model_with_overflow(field):
    model = FakeNativeFacetModel(16384)
    generate = model.generate_retrieval_analysis

    def analysis(**args):
        value = generate(**args)
        if args["system"] == SUMMARY_FACETS_COMPACT_SYSTEM:
            if field == "summary":
                value["document_summary"] = "publisher technique " * 30
            elif field == "facet":
                value["facets"][0]["representations"][0]["text"] = "method and limitations " * 20
            elif field == "model":
                raise EmbeddingContextExceeded("native summary exceeds safe embedding context")
        return value

    model.generate_retrieval_analysis = analysis
    return model


class EmbeddingRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.cli = load_cli()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        text = "[[SOURCE UNIT: methods]]\nA bounded technique works only on the measured examples."
        self.source = dict(original_path="examples/guide.txt", title="Technique",
                           original_sha256="a"*64, derived_sha256=hashlib.sha256(text.encode()).hexdigest(),
                           derived_text=text)
        self.args = SimpleNamespace(original_path=self.source["original_path"], model="fixture",
            model_url="http://localhost:1234", model_timeout=30, output_tokens=2400, safety_tokens=384,
            context_cap_tokens=16384, max_levels=12, max_nodes=100, max_calls=4, native_facets=True,
            compact_native_facets=True, plan_only=False, recover_embedding_context=False)
        self.budget = CharacterBudget()

    def engine(self, model):
        return HierarchicalSummarizer(model, output_tokens=2400, safety_tokens=384,
                                      native_facets=True, native_facets_compact=True)

    def validate(self, analysis, start, end):
        prepare_node_facets(analysis["document_summary"], analysis, text=self.source["derived_text"],
                            start=start, end=end, embedding_client=self.budget, limit=self.budget.limit)

    def test_summary_facet_and_model_guard_retry_full_input_once(self):
        for field in ("summary", "facet", "model"):
            with self.subTest(field=field):
                model = model_with_overflow(field)
                progress = SummaryActivity()
                tree = self.engine(model).summarize(title="Technique", text=self.source["derived_text"],
                                                     validate_native=self.validate, activity=progress)
                self.assertEqual(len(model.calls), 2)
                self.assertEqual(progress.model_requests, 2)
                self.assertEqual(progress.completed_model_calls, 1 if field == "model" else 2)
                self.assertEqual(model.calls[0][1:], model.calls[1][1:])
                self.assertIn(self.source["derived_text"], model.calls[1][1])
                self.assertEqual(tree.root.prompt_policy, "native_facets_bounded_after_embedding_overflow")
                self.validate(tree.root.native_analysis, 0, len(self.source["derived_text"]))

    def test_preserves_every_admissible_field_without_changing_its_facet(self):
        model = FakeNativeFacetModel(16384)
        good = model.generate_retrieval_analysis(system="", user="", output_tokens=1000)
        strict = deepcopy(good)
        strict["document_summary"] = "shorter answer"
        strict["facets"] = [strict["facets"][1]]
        strict["facets"][0]["representations"].append(
            {"kind": "keyword", "strength": "strong", "text": "new native keyword"})
        def repair(prior):
            return retain_fitting_native_fields(prior, strict, text=self.source["derived_text"],
                start=0, end=len(self.source["derived_text"]), embedding_client=self.budget, limit=150)
        big_summary = dict(good, document_summary="overlong " * 30)
        self.assertEqual(repair(big_summary), dict(good, document_summary="shorter answer"))
        big_item = deepcopy(good)
        big_item["facets"][0]["representations"].append(
            {"text": "long prose " * 40, "kind":"description","strength":"strong"})
        final = repair(big_item)
        self.assertEqual(final["document_summary"], good["document_summary"])
        self.assertEqual(final["facets"][0], good["facets"][0])
        self.assertEqual(final["facets"][1], strict["facets"][0])
        self.validate(final, 0, len(self.source["derived_text"]))
        self.assertEqual(len(big_item["facets"][0]["representations"]), 4)

    def test_recovery_reuses_exact_prior_node_only_for_matching_source_and_children(self):
        m = FakeNativeFacetModel(16384)
        text = self.source["derived_text"]
        tree = self.engine(m).summarize(title="Technique", text=text)
        progress = SummaryActivity()
        replay = self.engine(m).summarize(title="Technique", text=text, previous_tree=tree,
                                         validate_native=self.validate, activity=progress)
        self.assertEqual(len(m.calls), 1)
        self.assertIs(replay.root, tree.root)
        self.assertEqual(progress, SummaryActivity(reused_saved_nodes=1))
        with self.assertRaisesRegex(ValueError, "same complete source"):
            self.engine(m).summarize(title="Technique", text=text + "change", previous_tree=tree,
                                     validate_native=self.validate)

    def test_retry_respects_the_actual_request_limit_before_requesting_again(self):
        model = model_with_overflow("summary")
        progress = SummaryActivity()
        with self.assertRaisesRegex(RuntimeError, "1 model requests exhausted"):
            self.engine(model).summarize(title="Technique", text=self.source["derived_text"], max_calls=1,
                                         validate_native=self.validate, activity=progress)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(progress, SummaryActivity(model_requests=1, completed_model_calls=1))

    def test_second_overflow_and_unrelated_errors_emit_no_node(self):
        for err, count in ((EmbeddingContextExceeded("too long"), 2), (ValueError("wrong source"), 1)):
            with self.subTest(err=err):
                model = FakeNativeFacetModel(16384)
                emitted = []
                with self.assertRaises(type(err)):
                    self.engine(model).summarize(title="Technique", text=self.source["derived_text"],
                        validate_native=lambda *args: self.raise_error(err), on_node=lambda *args: emitted.append(args))
                self.assertEqual(len(model.calls), count)
                self.assertEqual(emitted, [])

    @staticmethod
    def raise_error(err):
        raise err

    def base_file(self, field):
        model = model_with_overflow(field)
        tree = self.engine(model).summarize(title="Technique", text=self.source["derived_text"])
        payload = dict(schema="kmd-complete-hierarchical-summary-tree-v1", model="fixture", native_facets=True,
                       tree=asdict(tree), **{k:self.source[k] for k in ("original_path","original_sha256","derived_sha256")})
        base = self.cli.payload_path(self.source["original_path"], self.source["original_sha256"], self.source["derived_sha256"])
        self.assertTrue(self.cli.save_tree_once(base, payload))
        return base

    def build(self, model):
        with patch.object(self.cli, "checked_original", return_value=self.source), \
                patch.object(self.cli, "LocalSummaryModel", return_value=model), \
                patch.object(self.cli, "E5PassageBudget", return_value=self.budget), redirect_stdout(io.StringIO()):
            self.cli.build(self.args)

    def test_recovery_preserves_base_and_selects_only_matching_sidecar(self):
        with patch.object(self.cli, "OUT", self.root):
            base = self.base_file("facet")
            old = base.read_bytes()
            self.args.recover_embedding_context = True
            self.build(model_with_overflow("facet"))
            payload, tree = self.cli.import_tree(base, self.source)
            digest = hashlib.sha256(old).hexdigest()
            self.assertEqual(payload["recovered_from_tree_sha256"], digest)
            self.assertEqual(tree.root.prompt_policy, "native_facets_bounded_after_embedding_overflow_retained_fields")
            self.assertEqual(base.read_bytes(), old)
            repeat = model_with_overflow("facet")
            self.build(repeat)
            self.assertEqual(repeat.calls, [])
            other = self.cli.recovery_path(base, "b"*64)
            sidecar = self.cli.recovery_path(base, digest)
            sidecar.rename(other)
            original, _ = self.cli.import_tree(base, self.source)
            self.assertNotIn("recovered_from_tree_sha256", original)
            data = dict(payload, recovered_from_tree_sha256="c"*64)
            self.cli.save_tree_once(sidecar, data)
            with self.assertRaisesRegex(ValueError, "predecessor"):
                self.cli.import_tree(base, self.source)

    def test_fitting_tree_and_incompatible_model_are_not_superseded(self):
        with patch.object(self.cli, "OUT", self.root):
            base = self.base_file(None)
            self.args.recover_embedding_context = True
            with self.assertRaisesRegex(ValueError, "unnecessary"):
                self.build(model_with_overflow(None))
            self.args.model = "unrelated"
            with self.assertRaisesRegex(ValueError, "generation model"):
                self.build(model_with_overflow(None))
            self.assertEqual(list(self.root.iterdir()), [base])

    def test_recovery_rejects_stale_source_and_inconsistent_native_summary(self):
        with patch.object(self.cli, "OUT", self.root):
            base = self.base_file("summary")
            changed = dict(self.source, derived_sha256="b"*64)
            with self.assertRaisesRegex(ValueError, "stale"):
                self.cli.import_tree(base, changed)
            data = json.loads(base.read_bytes())
            data["tree"]["nodes"][0]["native_analysis"]["document_summary"] = "changed"
            base.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "digest"):
                self.cli.import_tree(base, self.source)

    def test_racing_distinct_completed_trees_have_one_winner(self):
        base = self.root/"tree.json"
        candidates = [{"attempt": i, "contents":"document"*5000} for i in range(8)]
        with ThreadPoolExecutor(8) as pool:
            won = list(pool.map(lambda data: self.cli.save_tree_once(base, data), candidates))
        self.assertEqual(sum(won), 1)
        self.assertEqual(json.loads(base.read_bytes()), candidates[won.index(True)])
        self.assertTrue(self.cli.save_tree_once(base, candidates[won.index(True)]))
        self.assertEqual(base.stat().st_mode & 0o777, 0o600)
        self.assertEqual(list(self.root.iterdir()), [base])
        tuple_path = self.root / "tree-with-tuples.json"
        self.assertTrue(self.cli.save_tree_once(tuple_path, {"child_ids": ("a", "b")}))
        self.assertTrue(self.cli.save_tree_once(tuple_path, {"child_ids": ("a", "b")}))

    def test_plan_only_does_not_load_encoder_or_write_anything(self):
        self.args.plan_only = True
        with patch.object(self.cli, "OUT", self.root), \
                patch.object(self.cli, "checked_original", return_value=self.source), \
                patch.object(self.cli, "LocalSummaryModel", return_value=FakeNativeFacetModel(16384)), \
                patch.object(self.cli, "E5PassageBudget", side_effect=AssertionError("unexpected tokenizer")), \
                redirect_stdout(io.StringIO()):
            self.cli.build(self.args)
        self.assertEqual(list(self.root.iterdir()), [])
