"""Original-article context contracts independent of search ranks or gold queries."""
from __future__ import annotations

import hashlib
import unittest

from file_system_catalog.publisher_html_article_discovery import ArticleEntry, article_context


def sha(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def entry_for(text, *, path="manual/version45/example.html", selector="article"):
    return ArticleEntry(
        (path, "a" * 64, "b" * 64, path + ".txt", "html",
         "https://docs.example.test/manual/en/4.5/example.html",
         "Synthetic original context reference", 777, len(text), "complete"),
        "c" * 64, "d" * 64,
        "https://docs.example.test/manual/en/4.5/example.html",
        selector, text,
    )


def excerpt_for(text, start, end):
    value = text[start:end]
    return {
        "source_excerpt": value,
        "normalized_article_start_char": start,
        "normalized_article_end_char": end,
        "source_excerpt_sha256": sha(value),
        "matched_article_forms": [],
    }


class OriginalArticleContextTests(unittest.TestCase):
    def assert_partition(self, entry, excerpt, context, characters):
        body = entry.text
        intervals = [(excerpt["normalized_article_start_char"],
                      excerpt["normalized_article_end_char"])]
        self.assertEqual(entry.path, context["original_path"])
        self.assertEqual(entry.source_map_row[1], context["publisher_original_sha256"])
        self.assertEqual(sha(body), context["normalized_article_sha256"])
        self.assertEqual(entry.selector, context["article_selector"])
        self.assertFalse(context["governing_section_identified"])
        self.assertFalse(context["factual_answer_verified"])
        self.assertLessEqual(set(context["spans"]), {"article_opening", "preceding", "following"})
        for role, span in context["spans"].items():
            start, end = span["start_char"], span["end_char"]
            self.assertIs(type(start), int)
            self.assertIs(type(end), int)
            self.assertTrue(0 <= start < end <= len(body), role)
            self.assertEqual(body[start:end], span["text"])
            self.assertEqual(sha(body[start:end]), span["sha256"])
            self.assertLessEqual(end - start, characters)
            intervals.append((start, end))
        intervals.sort()
        for left, right in zip(intervals, intervals[1:]):
            self.assertLessEqual(left[1], right[0], "separate evidence spans must never overlap")
        omissions = context["omitted_normalized_article_ranges"]
        for span in omissions:
            self.assertEqual({"start_char", "end_char"}, set(span))
            self.assertIs(type(span["start_char"]), int)
            self.assertIs(type(span["end_char"]), int)
            self.assertTrue(0 <= span["start_char"] < span["end_char"] <= len(body))
        self.assertEqual(omissions, sorted(omissions, key=lambda p: p["start_char"]))
        partition = sorted(intervals + [(p["start_char"], p["end_char"]) for p in omissions])
        self.assertEqual(0, partition[0][0])
        self.assertEqual(len(body), partition[-1][1])
        for left, right in zip(partition, partition[1:]):
            self.assertEqual(left[1], right[0], "covered and omitted intervals must partition the entire container")
        covered = sum(end - start for start, end in intervals)
        self.assertEqual(covered, context["covered_normalized_article_characters"])
        self.assertEqual(not omissions, context["entire_selected_container_covered"])
        self.assertLessEqual(covered, 3 * characters + len(excerpt["source_excerpt"]))

    def test_distant_opening_negation_and_later_qualification_are_separate_exact_spans(self):
        opening = "This article describes a hypothetical example; it does not establish real-world findings. "
        before = "The following sentence is quoted as a rejected hypothesis. "
        anchor = "All materials preserve total power under every possible transformation."
        after = " That hypothesis is false outside the narrowly specified example conditions. "
        text = opening + ("Neutral background discussion. " * 120) + before + anchor + after + ("Later explanatory material. " * 40)
        start = text.index(anchor)
        excerpt = excerpt_for(text, start, start + len(anchor))
        original_excerpt = dict(excerpt)
        entry = entry_for(text)
        context = article_context(entry, excerpt, characters=160)
        self.assert_partition(entry, excerpt, context, 160)
        self.assertEqual(original_excerpt, excerpt, "context must not rewrite the short excerpt or its digest")
        self.assertIn("hypothetical", context["spans"]["article_opening"]["text"])
        self.assertIn("rejected hypothesis", context["spans"]["preceding"]["text"])
        self.assertIn("is false", context["spans"]["following"]["text"])
        self.assertEqual(
            [{"start_char": 160, "end_char": start - 160},
             {"start_char": start + len(anchor) + 160, "end_char": len(text)}],
            context["omitted_normalized_article_ranges"],
        )
        self.assertFalse(context["entire_selected_container_covered"])

    def test_near_start_does_not_invent_preceding_or_duplicate_opening(self):
        text = "A synthetic source explanation and its qualifications. " * 12
        entry = entry_for(text)
        excerpt = excerpt_for(text, 0, 100)
        context = article_context(entry, excerpt, characters=80)
        self.assert_partition(entry, excerpt, context, 80)
        self.assertEqual({"following"}, set(context["spans"]))
        self.assertEqual(100, context["spans"]["following"]["start_char"])
        self.assertEqual(180, context["spans"]["following"]["end_char"])

    def test_near_end_does_not_invent_following_context(self):
        text = "The original reference contains additional context and measured limitations. " * 12
        entry = entry_for(text)
        excerpt = excerpt_for(text, len(text) - 90, len(text))
        context = article_context(entry, excerpt, characters=80)
        self.assert_partition(entry, excerpt, context, 80)
        self.assertEqual({"article_opening", "preceding"}, set(context["spans"]))
        self.assertEqual(len(text) - 170, context["spans"]["preceding"]["start_char"])

    def test_opening_clipped_before_preceding_start_without_overlap(self):
        text = "x" * 260
        entry = entry_for(text)
        excerpt = excerpt_for(text, 120, 140)
        context = article_context(entry, excerpt, characters=80)
        self.assert_partition(entry, excerpt, context, 80)
        opening = context["spans"]["article_opening"]
        preceding = context["spans"]["preceding"]
        self.assertEqual((0, 40), (opening["start_char"], opening["end_char"]))
        self.assertEqual((40, 120), (preceding["start_char"], preceding["end_char"]))
        self.assertEqual([{"start_char": 220, "end_char": 260}],
                         context["omitted_normalized_article_ranges"])

    def test_small_whole_container_has_no_omissions_but_no_answer_claim(self):
        text = "A" * 190
        entry = entry_for(text, selector="main")
        excerpt = excerpt_for(text, 40, 100)
        context = article_context(entry, excerpt, characters=120)
        self.assert_partition(entry, excerpt, context, 120)
        self.assertTrue(context["entire_selected_container_covered"])
        self.assertEqual([], context["omitted_normalized_article_ranges"])
        self.assertEqual(190, context["covered_normalized_article_characters"])
        self.assertFalse(context["governing_section_identified"])
        self.assertFalse(context["factual_answer_verified"])

    def test_whole_container_excerpt_adds_no_fabricated_context(self):
        text = "This complete small normalized article contains one synthetic explanation."
        entry = entry_for(text)
        excerpt = excerpt_for(text, 0, len(text))
        context = article_context(entry, excerpt, characters=80)
        self.assert_partition(entry, excerpt, context, 80)
        self.assertEqual({}, context["spans"])
        self.assertTrue(context["entire_selected_container_covered"])

    def test_unicode_offsets_are_characters_and_digests_use_utf8(self):
        text = ("Original \u00e9vidence \u03b1\u03b2\u03b3 \U0001f9ed \u6f22\u5b57. " * 30 +
                "Quoted r\u00e9sum\u00e9 has a narrow source scope." +
                " Later \u00e9vidence and qualifications. " * 20)
        start = text.index("Quoted")
        end = start + len("Quoted r\u00e9sum\u00e9 has a narrow source scope.")
        self.assertGreater(len(text[:start].encode("utf8")), start)
        entry = entry_for(text)
        excerpt = excerpt_for(text, start, end)
        context = article_context(entry, excerpt, characters=80)
        self.assert_partition(entry, excerpt, context, 80)
        preceding = context["spans"]["preceding"]
        self.assertEqual(start - 80, preceding["start_char"])
        self.assertEqual(text[start - 80:start], preceding["text"])

    def test_context_limit_uses_strict_positive_bounded_integer(self):
        text = "A sufficiently long synthetic article with a scope-sensitive explanation."
        entry = entry_for(text)
        excerpt = excerpt_for(text, 0, 30)
        for value in (None, False, True, 0, -1, 79, 2001, 80.0, "80"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    article_context(entry, excerpt, characters=value)
        for valid in (80, 2000):
            context = article_context(entry, excerpt, characters=valid)
            self.assert_partition(entry, excerpt, context, valid)

    def test_changed_excerpt_text_hash_and_coordinates_are_refused(self):
        text = "A sufficiently long original article with meaningful context and qualifying explanation."
        entry = entry_for(text)
        valid = excerpt_for(text, 10, 40)
        mutations = [
            {"source_excerpt": "Different text", "source_excerpt_sha256": sha("Different text")},
            {"source_excerpt_sha256": "0" * 64},
            {"source_excerpt_sha256": None},
            {"source_excerpt_sha256": valid["source_excerpt_sha256"].upper()},
            {"normalized_article_start_char": -1},
            {"normalized_article_start_char": True},
            {"normalized_article_start_char": 10.0},
            {"normalized_article_start_char": "10"},
            {"normalized_article_end_char": False},
            {"normalized_article_end_char": len(text) + 1},
            {"normalized_article_end_char": 10},
            {"normalized_article_start_char": 40, "normalized_article_end_char": 10},
            {"source_excerpt": "", "source_excerpt_sha256": sha("")},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    article_context(entry, {**valid, **mutation}, characters=80)
        for missing in ("source_excerpt", "source_excerpt_sha256",
                        "normalized_article_start_char", "normalized_article_end_char"):
            broken = dict(valid)
            broken.pop(missing)
            with self.subTest(missing=missing):
                with self.assertRaises(ValueError):
                    article_context(entry, broken, characters=80)
        for malformed in (None, [], "not a mapping"):
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    article_context(entry, malformed, characters=80)

    def test_excerpt_from_changed_article_cannot_be_attached_to_stale_entry(self):
        old = "Before material discussion there is a valid original statement and source context."
        new = old.replace("valid original", "other revised!")
        entry = entry_for(old)
        start, end = old.index("valid"), old.index(" and")
        changed = excerpt_for(new, start, end)
        with self.assertRaises(ValueError):
            article_context(entry, changed, characters=80)

    def test_full_article_revision_digest_changes_even_when_excerpt_is_unchanged(self):
        text = "Opening reference. " * 20 + "The exact anchor." + " Closing explanation." * 20
        start = text.index("The exact anchor.")
        excerpt = excerpt_for(text, start, start + len("The exact anchor."))
        first = article_context(entry_for(text), excerpt, characters=80)
        revised = text + " Additional qualifying sentence."
        second = article_context(entry_for(revised), excerpt, characters=80)
        self.assertNotEqual(first["normalized_article_sha256"], second["normalized_article_sha256"])
        self.assertEqual(sha(text), first["normalized_article_sha256"])
        self.assertEqual(sha(revised), second["normalized_article_sha256"])


if __name__ == "__main__":
    unittest.main()
