"""Context CLI checks using synthetic originals, never actual publisher text."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from test_publisher_html_article_discovery_cli import (
    QUERY, SCOPE, SCRIPT, SyntheticLibrary, load_cli, sha,
)


class PublisherHTMLArticleContextCLIIntegrationTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.fixture = SyntheticLibrary(self.root)
        self.neighbor = self.fixture.add_source("4.5/child", 3)
        self.fixture.persist_metadata()
        self.expected_articles = {}
        opening = " ".join(
            "Opening synthetic reference sentence %03d describes the sample fixture." % i
            for i in range(22))
        target = ("Normalization exposure energy appear together in this synthetic target. "
                  "The statement is only a retrieval fixture, with no claim about real software.")
        following = " ".join(
            "Following synthetic reference sentence %03d supplies neighboring sample prose." % i
            for i in range(22))
        self.replace_article(self.fixture.selected, [opening, target, following])
        self.replace_article(self.neighbor, [
            "Normalization exposure energy form a short synthetic alternative article. "
            "It contains the whole reference in fewer than five hundred characters."])
        self.cli = load_cli()

    def replace_article(self, path, paragraphs):
        record = next(row for row in self.fixture.catalog_rows if row["path"] == path)
        title = record["title"]
        text = " ".join([title] + paragraphs)
        raw = ("<!doctype html><html><nav>Normalization exposure energy excluded navigation.</nav>"
               "<main><article><h1>" + html.escape(title) + "</h1>" +
               "".join("<p>" + html.escape(p) + "</p>" for p in paragraphs) +
               "</article></main></html>").encode()
        original, derivative = self.fixture.files[path]
        original.write_bytes(raw)
        derivative.write_text("Synthetic derivative header.\n[[SOURCE UNIT: HTML article]]\n" + text)
        derived = derivative.read_bytes()
        with sqlite3.connect(self.fixture.source_map) as db:
            db.execute("UPDATE originals SET source_sha256=?,derived_sha256=?,"
                       "original_bytes=?,text_characters=? WHERE source_path=?",
                       (sha(raw), sha(derived), len(raw), len(derived.decode()), path))
        receipt = next(row for row in self.fixture.ledger_rows if row["path"] == path)
        receipt.update(sha256=sha(raw), bytes_received=len(raw))
        self.fixture.persist_metadata()
        self.expected_articles[path] = text

    def run_cli(self, extra):
        command = [sys.executable, str(SCRIPT)] + self.fixture.arguments()
        command += ["--catalog", str(self.fixture.catalog)]
        environment = os.environ.copy()
        environment.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1")
        return subprocess.run(command + extra, capture_output=True, text=True,
                              env=environment, timeout=20)

    def run_query(self, extra=(), *, scope=SCOPE, query=QUERY):
        run = self.run_cli(["--publisher-url-prefix", scope, "--query", query] + list(extra))
        self.assertEqual(0, run.returncode, run.stderr)
        return json.loads(run.stdout)

    @staticmethod
    def all_hits(result):
        return [hit for question in result["queries"]
                for channel in question["independent_rank_channels"].values()
                for hit in channel["source_originals"]]

    def assert_empty(self, result):
        self.assertEqual(0, result["selected_original_count"])
        self.assertFalse(result["out_of_scope_fallback_used"])
        self.assertEqual([], self.all_hits(result))
        for question in result["queries"]:
            for channel in question["independent_rank_channels"].values():
                self.assertEqual(0, channel["returned_current_originals"])

    def assert_context(self, hit, count):
        text = self.expected_articles[hit["original_path"]]
        context = hit["source_context"]
        self.assertIsInstance(context["basis"], str)
        self.assertTrue(context["basis"])
        self.assertEqual(count, context["requested_characters_per_neighbor"])
        self.assertEqual(hit["original_path"], context["original_path"])
        self.assertEqual(hit["publisher_original_sha256"], context["publisher_original_sha256"])
        self.assertEqual(sha(text.encode()), context["normalized_article_sha256"])
        self.assertEqual("article", context["article_selector"])
        self.assertFalse(context["governing_section_identified"])
        self.assertFalse(context["factual_answer_verified"])
        start, end = hit["normalized_article_start_char"], hit["normalized_article_end_char"]
        self.assertEqual(text[start:end], hit["source_excerpt"])
        self.assertLessEqual(len(hit["source_excerpt"]), 520)
        before = max(0, start - count)
        expected = [("article_opening", 0, min(count, before)),
                    ("preceding", before, start),
                    ("following", end, min(len(text), end + count))]
        expected = [span for span in expected if span[1] < span[2]]
        self.assertEqual({role: (start, end) for role, start, end in expected},
                         {role: (span["start_char"], span["end_char"])
                          for role, span in context["spans"].items()})
        covered = [False] * len(text)
        covered[start:end] = [True] * (end - start)
        for span in context["spans"].values():
            a, b = span["start_char"], span["end_char"]
            self.assertEqual(text[a:b], span["text"])
            self.assertEqual(sha(span["text"].encode()), span["sha256"])
            self.assertLessEqual(b - a, count)
            covered[a:b] = [True] * (b - a)
        omitted = []
        cursor = 0
        while cursor < len(covered):
            if covered[cursor]:
                cursor += 1
                continue
            begin = cursor
            while cursor < len(covered) and not covered[cursor]:
                cursor += 1
            omitted.append({"start_char": begin, "end_char": cursor})
        self.assertEqual(omitted, context["omitted_normalized_article_ranges"])
        self.assertEqual(sum(covered), context["covered_normalized_article_characters"])
        self.assertEqual(all(covered), context["entire_selected_container_covered"])

    def test_default_zero_preserves_excerpt_and_ranking_when_context_is_enabled(self):
        before = self.fixture.source_digests()
        default = self.run_query()
        explicit_zero = self.run_query(["--context-chars", "0"])
        expanded = self.run_query(["--context-chars", "80"])
        self.assertEqual(0, default["context_characters_per_side"])
        self.assertEqual(0, explicit_zero["context_characters_per_side"])
        self.assertEqual(80, expanded["context_characters_per_side"])
        self.assertEqual(default["queries"], explicit_zero["queries"])
        self.assertEqual({self.fixture.selected, self.neighbor},
                         {hit["original_path"] for hit in self.all_hits(default)})
        for hit in self.all_hits(default):
            self.assertNotIn("source_context", hit)
        without_context = json.loads(json.dumps(expanded["queries"]))
        for question in without_context:
            for channel in question["independent_rank_channels"].values():
                for hit in channel["source_originals"]:
                    self.assertIn("source_context", hit)
                    del hit["source_context"]
        self.assertEqual(default["queries"], without_context)
        self.assertEqual(before, self.fixture.source_digests())

    def test_minimum_and_maximum_context_have_exact_independent_source_coordinates(self):
        before = self.fixture.source_digests()
        for count in (80, 2000):
            with self.subTest(count=count):
                result = self.run_query(["--context-chars", str(count)])
                self.assertTrue(result["all_displayed_sources_freshly_revalidated_after_batch"])
                self.assertFalse(result["cross_store_atomic_generation_pinned"])
                for hit in self.all_hits(result):
                    self.assert_context(hit, count)
                long_hits = [hit for hit in self.all_hits(result)
                             if hit["original_path"] == self.fixture.selected]
                self.assertEqual(2, len(long_hits))
                for hit in long_hits:
                    context = hit["source_context"]
                    if count == 80:
                        self.assertEqual({"article_opening", "preceding", "following"},
                                         set(context["spans"]))
                        self.assertFalse(context["entire_selected_container_covered"])
                        self.assertTrue(context["omitted_normalized_article_ranges"])
                    else:
                        self.assertTrue(context["entire_selected_container_covered"])
                        self.assertEqual([], context["omitted_normalized_article_ranges"])
                for hit in self.all_hits(result):
                    if hit["original_path"] == self.neighbor:
                        self.assertEqual({}, hit["source_context"]["spans"])
                        self.assertTrue(hit["source_context"]["entire_selected_container_covered"])
        self.assertEqual(before, self.fixture.source_digests())

    def test_context_limits_reject_invalid_cli_values_without_output(self):
        before = self.fixture.source_digests()
        for value in (-1, 1, 79, 2001):
            with self.subTest(value=value):
                run = self.run_cli(["--publisher-url-prefix", SCOPE, "--query", QUERY,
                                    "--context-chars", str(value)])
                self.assertNotEqual(0, run.returncode)
                self.assertEqual("", run.stdout)
                self.assertIn("context", run.stderr.lower())
        self.assertEqual(before, self.fixture.source_digests())

    def test_context_cannot_turn_nonmatching_query_into_a_result(self):
        result = self.run_query(["--context-chars", "80"], query="quartz marmalade periscope")
        self.assertEqual(80, result["context_characters_per_side"])
        self.assertEqual(2, result["source_scan"]["verified_original_articles_indexed"])
        self.assert_empty(result)

    def test_context_unknown_version_is_empty_without_cross_version_fallback(self):
        result = self.run_query(["--context-chars", "2000"],
                                scope="https://manual.example.org/manual/en/99.9/")
        self.assertEqual(2000, result["context_characters_per_side"])
        self.assertEqual("no_archived_sources_in_requested_scope", result["source_scan"]["scope_status"])
        self.assert_empty(result)

    def test_context_does_not_recover_sources_that_fail_original_validation(self):
        for path in (self.fixture.selected, self.neighbor):
            original = self.fixture.files[path][0]
            original.write_bytes(original.read_bytes() + b" changed")
        before = self.fixture.source_digests()
        result = self.run_query(["--context-chars", "80"])
        self.assertEqual("no_usable_archived_html_articles", result["source_scan"]["scope_status"])
        self.assertEqual(2, result["source_scan"]["skipped_html_records"])
        self.assert_empty(result)
        self.assertEqual(before, self.fixture.source_digests())

    def test_context_still_refuses_changes_after_selection_and_final_hash_validation(self):
        for stage in (1, 2):
            for kind in ("url", "title", "derived_path", "status", "catalog", "receipt",
                         "native", "original", "derivative"):
                with self.subTest(stage=stage, kind=kind), TemporaryDirectory() as folder:
                    fixture = SyntheticLibrary(Path(folder))
                    verify = self.cli.verify_entry
                    calls = 0

                    def mutate_after_verification(*args, **kwargs):
                        nonlocal calls
                        proof = verify(*args, **kwargs)
                        calls += 1
                        if calls == stage:
                            fixture.mutate(kind)
                        return proof

                    with mock.patch.object(self.cli, "verify_entry", side_effect=mutate_after_verification):
                        with self.assertRaises(ValueError):
                            self.cli.search(**fixture.search_arguments(), context_chars=80)
                    self.assertGreaterEqual(calls, stage)


if __name__ == "__main__":
    unittest.main()
