"""RFC citations must identify one publisher file and the body heading printed in it."""

import hashlib
import unittest

from file_system_catalog.named_source import resolve_source_name
from file_system_catalog.rfc_section import rfc_section_evidence, rfc_header_matches


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def record(n):
    return {
        "title": "Publisher standard",
        "kind": "text",
        "url": f"https://www.rfc-editor.org/rfc/rfc{n}.txt",
    }


BODY = """\ufeff
Internet Engineering Task Force
Request for Comments: 9110           Example

Table of Contents

   1.  Scope
     1.1.  Métadonnées and Long
           Names
   Appendix A.  Notes

1.  Scope

1.1.  Métadonnées and Long
      Names

   Preserve the value across repeated requests.
   Check the original bytes.

   A second paragraph describes an exception.

Appendix A.  Notes

   An appendix example.
"""


class RfcSectionTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "r.txt": record("9110"),
            "related.html": {
                "title": "Other material",
                "kind": "html",
                "url": "https://example.org/review",
                "source_url": record("9110")["url"],
            },
        }

    def test_exact_names_urls_and_citations(self):
        for q in (
            "RFC 9110",
            "rfc9110",
            "https://www.rfc-editor.org/info/rfc9110",
            "https://www.rfc-editor.org/rfc/rfc9110.html",
        ):
            m = resolve_source_name(q, self.manifest)
            self.assertEqual(
                (m["kind"], m["original_path"], m["matched_identifier"]), ("catalog_rfc", "r.txt", "RFC 9110")
            )
        for q in (
            "RFC 9110, Section 1.1",
            '"RFC 9110, Section 1.1"',
            "RFC 9110 § 1.1",
            "https://www.rfc-editor.org/rfc/rfc9110.html#section-1.1",
        ):
            m = resolve_source_name(q, self.manifest)
            self.assertEqual((m["original_path"], m["subject_query"]), ("r.txt", "Section 1.1"))

    def test_incidental_comparison_ambiguous_and_conflicting_records(self):
        for q in ("Compare RFC 9110 and RFC 9113", "https://www.rfc-editor.org.evil/rfc/rfc9110.txt", "9110"):
            self.assertIsNone(resolve_source_name(q, self.manifest))
        self.assertEqual(resolve_source_name("RFC 9111", self.manifest)["status"], "no_catalog_match")
        self.assertEqual(
            resolve_source_name("RFC 9110", dict(self.manifest, second=record("9110")))["status"], "ambiguous"
        )
        self.manifest["r.txt"]["source_url"] = "https://www.rfc-editor.org/info/rfc9113"
        self.assertEqual(resolve_source_name("RFC 9110", self.manifest)["status"], "no_catalog_match")

    def evidence(self, section, body=BODY, limit=900, *, cursor=None):
        raw = body.replace("\n", "\r\n").encode()
        text = "KMD source provenance example\n[[SOURCE UNIT: Original plain text]]\n" + body.strip() + "\n"
        s = dict(
            original_path="r.txt",
            publisher_text_bytes=raw,
            original_sha256=sha(raw),
            derived_text=text,
            derived_sha256=sha(text.encode()),
        )
        args = {} if cursor is None else {"cursor": cursor}
        return rfc_section_evidence(s, 3, "9110", section, limit, **args), raw, text

    def test_body_not_contents_and_publisher_bytes_with_crlf_bom_and_unicode(self):
        d, raw, text = self.evidence("1.1")
        (row,) = d["passages"]
        self.assertIn("Preserve the value", row["source_excerpt"])
        self.assertNotIn("second paragraph", row["source_excerpt"])
        self.assertEqual(row["source_excerpt"], text[row["excerpt_start_char"] : row["excerpt_end_char"]])
        anchor = row["publisher_excerpt"]
        fragment = raw[anchor["start_byte"] : anchor["end_byte"]]
        self.assertEqual(sha(fragment), anchor["sha256"])
        self.assertEqual(fragment.decode().replace("\r\n", "\n"), row["source_excerpt"])
        self.assertEqual(row["source_start_char"], text.rindex("\n1.1.  ") + 1)

    def test_parent_without_intro_and_appendix(self):
        d, _, _ = self.evidence("1")
        self.assertEqual(d["passages"][0]["source_excerpt"], "1.  Scope")
        self.assertEqual(d["rfc_section_lookup"]["scope"], "heading_only")
        d, _, _ = self.evidence("A")
        self.assertIn("An appendix example.", d["passages"][0]["source_excerpt"])

    def test_fake_contents_and_duplicate_body_do_not_become_evidence(self):
        for section, body in (
            ("1.3", BODY),
            ("1.1", BODY.replace("           Names", "           Codes")),
            ("1.1", BODY + "\n1.1.  Métadonnées and Long Names\n\n   Fabrication.\n"),
        ):
            d, _, _ = self.evidence(section, body)
            self.assertEqual(d["returned_passages"], 0)
            self.assertEqual(d["rfc_section_lookup"]["status"], "section_not_verified")

    def test_unlisted_subsection_requires_its_unique_enclosing_publisher_heading(self):
        child = "1.1.2.  Timeouts\n\n   A bounded timeout is required.\n\n"
        body = BODY.replace(
            "Appendix A.  Notes\n\n   An appendix", child + "Appendix A.  Notes\n\n   An appendix"
        )
        d, _, _ = self.evidence("1.1.2", body)
        self.assertEqual(d["rfc_section_lookup"]["basis"], "body_under_verified_parent")
        self.assertIn("bounded timeout", d["passages"][0]["source_excerpt"])
        for bad in (
            body.replace(child, "2.  A different section\n\n" + child),
            body.replace("           Names", "           Codes"),
            body + child,
        ):
            missing, _, _ = self.evidence("1.1.2", bad)
            self.assertEqual(missing["returned_passages"], 0)

    def test_unlisted_sibling_still_belongs_to_the_same_parent(self):
        subsections = (
            "1.1.1.  One\n\n   First.\n\n1.1.1.1.  Detail\n\n   Detail.\n\n1.1.2.  Two\n\n   Second.\n\n"
        )
        body = BODY.replace(
            "Appendix A.  Notes\n\n   An appendix", subsections + "Appendix A.  Notes\n\n   An appendix"
        )
        result, _, _ = self.evidence("1.1.2", body)
        self.assertEqual(result["rfc_section_lookup"]["basis"], "body_under_verified_parent")
        self.assertIn("Second.", result["passages"][0]["source_excerpt"])

    def test_hashes_and_changed_extraction_fail_verification(self):
        _, raw, text = self.evidence("1")
        source = dict(
            original_path="r.txt",
            publisher_text_bytes=raw,
            original_sha256=sha(raw),
            derived_text=text,
            derived_sha256=sha(text.encode()),
        )
        changed = text.replace("Preserve", "Modify")
        for patch in (
            {"original_sha256": "0" * 64},
            {"derived_sha256": "0" * 64},
            {"derived_text": changed, "derived_sha256": sha(changed.encode())},
        ):
            with self.subTest(patch=list(patch)), self.assertRaises(ValueError):
                rfc_section_evidence({**source, **patch}, 3, "9110", "1.1", 900)

    def test_context_recovers_condition_and_stops_at_next_heading(self):
        text = BODY.replace(
            "   A second paragraph describes an exception.",
            "   " + ("Long qualification. " * 24) + "\n\n   This rule does not apply during testing.",
        )
        state, raw, derivative = self.evidence("1.1", text, limit=160)
        pieces = []
        previous_end = None
        while True:
            (row,) = state["passages"]
            cursor_info = state["rfc_section_lookup"]["context"]
            anchor = row["publisher_excerpt"]
            if previous_end is not None:
                self.assertEqual(previous_end, anchor["start_byte"])
            fragment = raw[anchor["start_byte"] : anchor["end_byte"]]
            self.assertEqual(sha(fragment), anchor["sha256"])
            self.assertEqual(row["source_excerpt"], fragment.decode().replace("\r\n", "\n"))
            self.assertEqual(
                derivative[row["excerpt_start_char"] : row["excerpt_end_char"]], row["source_excerpt"]
            )
            pieces.append(row["source_excerpt"])
            previous_end = anchor["end_byte"]
            cursor = cursor_info["next_cursor"]
            if cursor is None:
                break
            state, _, _ = self.evidence("1.1", text, limit=160, cursor=cursor)
            self.assertEqual(state["rfc_section_lookup"]["scope"], "section_continuation")
        expected = text[text.index("\n1.1.  Métadonnées") + 1 : text.rindex("\nAppendix A.")].rstrip()
        self.assertEqual("".join(pieces), expected)
        self.assertGreater(len(pieces), 2)

    def test_cursor_is_bound_to_the_file_revision_and_section(self):
        state, raw, text = self.evidence("1.1")
        cursor = state["rfc_section_lookup"]["context"]["next_cursor"]
        for section, body, token in (
            ("A", BODY, cursor),
            ("1.1", BODY + "\n", cursor),
            ("1.1", BODY, "1." + "0" * 64),
            ("99", BODY, cursor),
            ("1.1", BODY, ""),
            ("1.1", BODY, "9" * 5000),
        ):
            with self.subTest(section=section, token=str(token)[:12]), self.assertRaises(ValueError):
                self.evidence(section, body, cursor=token)
        source = dict(
            original_path="other.txt",
            publisher_text_bytes=raw,
            original_sha256=sha(raw),
            derived_text=text,
            derived_sha256=sha(text.encode()),
        )
        with self.assertRaises(ValueError):
            rfc_section_evidence(source, 3, "9110", "1.1", 900, cursor=cursor)
        text = "Revised extraction prologue\n" + text
        source.update(original_path="r.txt", derived_text=text, derived_sha256=sha(text.encode()))
        with self.assertRaises(ValueError):
            rfc_section_evidence(source, 3, "9110", "1.1", 900, cursor=cursor)

    def test_short_parent_and_unknown_margin_are_reported(self):
        parent, _, _ = self.evidence("1")
        context = parent["rfc_section_lookup"]["context"]
        self.assertIsNone(context["next_cursor"])
        self.assertEqual(context["boundary"], "numbered_heading")
        body = BODY.replace(
            "   A second paragraph describes an exception.", "Unclassified heading\n\n   A different topic."
        )
        res, _, _ = self.evidence("1.1", body)
        context = res["rfc_section_lookup"]["context"]
        self.assertIsNone(context["next_cursor"])
        self.assertEqual(context["boundary"], "unclassified_margin")
        self.assertNotIn("different topic", res["passages"][0]["source_excerpt"])

    def test_captioned_diagram_preserves_following_condition(self):
        drawing = "| frame | header |\n+-------+--------+\n\nFigure 1: Sample layout"
        body = BODY.replace(
            "   A second paragraph describes an exception.",
            drawing + "\n\n   Only confirmed requests may proceed.",
        )
        first, _, _ = self.evidence("1.1", body)
        context = first["rfc_section_lookup"]["context"]
        self.assertEqual(context["boundary"], "numbered_heading")
        next_page, _, _ = self.evidence("1.1", body, cursor=context["next_cursor"])
        quote = next_page["passages"][0]["source_excerpt"]
        self.assertIn(drawing, quote)
        self.assertIn("Only confirmed requests", quote)
        self.assertNotIn("Appendix A", quote)
        for invalid in (
            drawing.replace("Figure 1: Sample layout", "A separate heading"),
            drawing.replace("+-------+--------+", "2.  Other | heading |"),
            drawing.replace("+-------+--------+", "Unclassified title"),
        ):
            blocked, _, _ = self.evidence("1.1", body.replace(drawing, invalid))
            self.assertEqual(blocked["rfc_section_lookup"]["context"]["boundary"], "unclassified_margin")
            self.assertIsNone(blocked["rfc_section_lookup"]["context"]["next_cursor"])

    def test_long_opening_and_unnumbered_back_matter(self):
        body = BODY.replace("   Appendix A.  Notes\n", "   Appendix A.  Notes\n   Contributors\n")
        body = body.replace(
            "   An appendix example.", "   " + ("ÄBC " * 120) + "\n\nContributors\n\n   Editorial material."
        )
        result, _, _ = self.evidence("A", body, limit=160)
        info = result["rfc_section_lookup"]
        self.assertTrue(info["truncated"])
        self.assertEqual(info["context"]["boundary"], "contents_heading")
        follow, _, _ = self.evidence("A", body, limit=900, cursor=info["context"]["next_cursor"])
        self.assertIsNone(follow["rfc_section_lookup"]["context"]["next_cursor"])
        self.assertEqual(
            len(follow["passages"][0]["source_excerpt"]) + 160, info["context"]["length_characters"]
        )
        self.assertNotIn("Contributors", follow["passages"][0]["source_excerpt"])

    def test_mismatched_header_and_revision(self):
        self.assertFalse(rfc_header_matches(BODY.encode(), "9113"))
        self.assertFalse(rfc_header_matches((BODY[:80] + "\nRequest for Comments: 9113\n").encode(), "9110"))
        with self.assertRaises(ValueError):
            self.evidence("1", BODY.replace("Comments: 9110", "Comments: 9113"))
