"""Source-scope context must be bounded, exact and never treated as an answer."""
from __future__ import annotations

import hashlib
import unittest
from file_system_catalog.source_evidence import (
    source_evidence_candidates,
    _checked_scope_context,
)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source(text, *, query, path="fictional/scope-example.txt", context=320,
           width=260):
    return source_evidence_candidates(
        original_path=path, original_title="Fictional test source",
        filesystem_entry_id=71, original_sha256="a" * 64,
        derived_sha256=digest(text), derived_text=text,
        query=query, top=3, snippet_characters=width,
        scope_context_characters=context,
    )


class VerifiedContextTests(unittest.TestCase):
    def assert_context_spans(self, text, row):
        proof=row["scope_context"]
        self.assertEqual("verified_current_derivative_section_only",proof["basis"])
        self.assertFalse(proof["publisher_page_or_byte_proof"])
        self.assertIn("not an answer",proof["warning"])
        for label, span in proof["spans"].items():
            self.assertIn(label, ("section_opening", "preceding", "following"))
            l,r=span["start_char"],span["end_char"]
            self.assertGreaterEqual(l,0)
            self.assertLessEqual(r,len(text))
            self.assertEqual(text[l:r],span["text"])
            self.assertEqual(digest(span["text"]),span["sha256"])
            self.assertLessEqual(len(span["text"]),320)
        self.assertEqual(text[row["excerpt_start_char"]:row["excerpt_end_char"]],
                         row["source_excerpt"])
        self.assertEqual(digest(row["source_excerpt"]),row["excerpt_sha256"])

    def test_dream_quote_superseded_and_condition_scope_is_preserved_not_classified(self):
        fixtures=[
            ("dream", "DREAM ONLY: The next scene is not real legislation.",
             "The fictional narrator says dragon courier registration is mandatory.",
             "dragon courier registration mandatory"),
            ("superseded", "SUPERSEDED EDITION: The following is obsolete, not current.",
             "The old rule lists a late shipment penalty of fifty credits.",
             "late shipment penalty fifty credits"),
            ("conditional", "ONLY WHEN CALIBRATING: The following rule has this condition.",
             "The sample procedure says the plasma enclosure door must be locked.",
             "plasma enclosure door locked"),
            ("quoted", "DISPUTED QUOTE: The following is a speaker's unfounded assertion.",
             "The fictional speaker says the harmonic motor creates perpetual energy.",
             "harmonic motor perpetual energy"),
        ]
        for ident,warning,claim,query in fixtures:
            with self.subTest(ident=ident):
                text="[[SOURCE UNIT: Original plain text]]\n"+warning+"\n"+("background unrelated descriptions. "*75)+claim
                old=source(text,query=query,context=0)
                self.assertTrue(old["passages"])
                self.assertNotIn(warning,old["passages"][0]["source_excerpt"])
                new=source(text,query=query)
                self.assertTrue(new["passages"])
                self.assertEqual([v["excerpt_sha256"] for v in old["passages"]],
                                 [v["excerpt_sha256"] for v in new["passages"]])
                row=new["passages"][0]
                self.assert_context_spans(text,row)
                intro=row["scope_context"]["spans"]["section_opening"]["text"]
                self.assertIn(warning,intro)
                self.assertNotIn("scope_classification",row)
                self.assertIn("not proof",new["evidence_warning"])

    def test_adjacent_context_never_bleeds_across_physical_source_page(self):
        p1=("[[SOURCE UNIT: PDF page 1]]\n"
            "Fictional historical scenario: a municipal dragon rule was dreamed.\n"
            + "irrelevant unrelated background " * 35)
        p2=("[[SOURCE UNIT: PDF page 2]]\n"
            "This page gives unrelated background on a different experiment.\n"
            + "unrelated lab notes " * 42
            + "\nThe measured turbine displacement rate was ten units.\n")
        raw=p1+p2
        x=source(raw,query="measured turbine displacement rate",path="fictional/example.pdf")
        self.assertTrue(x["passages"])
        r=x["passages"][0]
        self.assertEqual("PDF page 2",r["source_section"])
        self.assert_context_spans(raw,r)
        for span in r["scope_context"]["spans"].values():
            self.assertGreaterEqual(span["start_char"],len(p1))
            self.assertNotIn("dragon rule",span["text"])
        self.assertFalse(r["scope_context"]["publisher_page_or_byte_proof"])

    def test_long_split_source_uses_original_page_opening_not_chunk_opening(self):
        intro="FICTIONAL HYPOTHESIS: The following is not a real safety policy.\n"
        big=("repeated technical context with no relevant terms. " * 1500)
        text="[[SOURCE UNIT: PDF page 8]]\n"+intro+big+"\nThe silver turbine must be isolated before the experiment."
        x=source(text,query="silver turbine isolated experiment",path="fake/book.pdf")
        self.assertTrue(x["passages"])
        row=x["passages"][0]
        self.assertGreaterEqual(row["source_start_char"],60000)
        self.assert_context_spans(text,row)
        self.assertIn("FICTIONAL HYPOTHESIS:",row["scope_context"]["spans"]["section_opening"]["text"])

    def test_markdown_headings_and_fenced_quote_are_not_promoted_to_truth(self):
        text=("[[SOURCE UNIT: Original plain text]]\n"
              "# Fictional examples\n"
              "## A reported but unverified statement\n"
              + ("Other dialogue and surrounding prose. "*70)
              + "\nIn the dialogue the spectrometer produces infinite energy.\n")
        out=source(text,query="spectrometer infinite energy",path="examples/manual.md")
        row=out["passages"][0]
        self.assert_context_spans(text,row)
        self.assertIn("unverified statement",row["source_section"].lower())
        self.assertIn("unverified statement",row["scope_context"]["spans"]["section_opening"]["text"].lower())

    def test_later_retraction_is_in_bounded_following_context_not_asserted_as_fact(self):
        claim=("The 2018 draft instructed operators to disable the emission alarms "
               "during nighttime maintenance.")
        retraction=("\nIMPORTANT: The 2024 revision explicitly reversed that practice. "
                    "Do not disable the alarms.")
        text=("[[SOURCE UNIT: Original plain text]]\n"
              "Historic operational bulletin.\n"
              + ("Older background, unrelated to the alarm policy. "*20)
              + "\n" + claim + retraction + "\n")
        x=source(text,query="disable emission alarms nighttime maintenance",width=160)
        row=x["passages"][0]
        self.assert_context_spans(text,row)
        # The answer still requires an interpretation; context is just data.
        context=row["scope_context"]["spans"]
        self.assertTrue(
            any("reversed" in piece["text"] for piece in context.values()),
            "A nearby retraction must remain visible next to the retrieved claim",
        )
        self.assertIn("not an answer",row["scope_context"]["warning"])

    def test_conflicting_statements_on_separate_pages_are_not_silently_merged(self):
        text=("[[SOURCE UNIT: PDF page 1]]\n"
              "Old test edition: protocol X required a red indicator.\n"
              + ("Unrelated first-page analysis. "*70)
              + "\n[[SOURCE UNIT: PDF page 2]]\n"
              "Current test edition explicitly says protocol X does not require red indicators.\n"
              + ("Unrelated other-page notes. "*30))
        result=source(text,query="protocol X require red indicators",path="draft/scope.pdf")
        self.assertTrue(result["passages"])
        self.assertIn("not proof",result["evidence_warning"])
        for item in result["passages"]:
            self.assert_context_spans(text,item)
            context=item["scope_context"]["spans"]
            part=item["source_section"]
            if part=="PDF page 1":
                self.assertTrue(all("Current test edition" not in c["text"] for c in context.values()))
            elif part=="PDF page 2":
                self.assertTrue(all("Old test edition" not in c["text"] for c in context.values()))
        self.assertNotIn("resolved_conflict",result)

    def test_invalid_context_requests_rejected_before_indexing_or_source_use(self):
        for n in (-100,-1,1,79,601,1.0,True,"320"):
            with self.subTest(n=n):
                with self.assertRaisesRegex(ValueError, "scope context"):
                    source("[[SOURCE UNIT: page]]\nVerified sample source.",
                           query="sample source",context=n)
        with self.assertRaisesRegex(ValueError,"digest"):
            text="[[SOURCE UNIT: page]]\nPublished sample evidence."
            source_evidence_candidates(
                original_path="report.pdf", original_title="Report",
                filesystem_entry_id=9, original_sha256="a"*64,
                derived_sha256="b"*64, derived_text=text,
                query="published evidence", scope_context_characters=320,
            )
        with self.assertRaisesRegex(ValueError,"outside exact original section"):
            _checked_scope_context(
                "tiny paragraph",section_start=0,section_end=5,
                excerpt_start=1,excerpt_end=12,characters=320,
            )

    def test_default_zero_does_not_change_legacy_evidence_shape(self):
        text="[[SOURCE UNIT: Original plain text]]\nOriginal signed measurement records show solid mass."
        x=source(text,query="signed solid mass",context=0)
        self.assertTrue(x["passages"])
        self.assertNotIn("scope_context",x["passages"][0])


if __name__=="__main__":
    unittest.main()
