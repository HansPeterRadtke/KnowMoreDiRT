"""Named publisher evidence, source offsets, and operator audit coverage."""
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from file_system_catalog.source_evidence import _publisher_sections, source_evidence_candidates
from test_source_evidence import cli


def digest(text):
    return sha256(text.encode()).hexdigest()


def evidence(text, query, path="guides/example.md"):
    return source_evidence_candidates(
        original_path=path, original_title="Publisher example", filesystem_entry_id=3,
        original_sha256="f" * 64, derived_sha256=digest(text), derived_text=text,
        query=query, top=8, snippet_characters=200,
    )["passages"]


class MarkdownEvidenceTests(unittest.TestCase):
    def test_generated_header_and_unknown_wrappers_cannot_supply_evidence(self):
        header = ('KMD source provenance example\n'
                  '{"note":"[[SOURCE UNIT: Example]] nebula penguins"}\n')
        for path in ("a.md", "a.pdf"):
            self.assertEqual([], evidence(header, "nebula penguins", path))
            for marker in ("Original plain text", "PDF page 3"):
                text = header + "[[SOURCE UNIT: "+marker+"]]\nField notes about pollen transport."
                self.assertEqual([], evidence(text, "nebula penguins", path))
                self.assertTrue(evidence(text, "pollen transport", path))

    def test_fenced_examples_yaml_and_unicode_preserve_all_publisher_bytes(self):
        header = "KMD source provenance example\n[[SOURCE UNIT: Original plain text]]\r\n"
        raw = ("---\r\nname: résumé\r\n# YAML comment\r\n...\r\n"
               "Preamble Ü🌱\r\n# Root ###\r\n\u00c4\r\n  ## Topic\r\n"
               "````sh\r\n# Fake heading\r\n```\r\n## Still a comment\r\n````\r\n"
               "~~~bash\r\n# Another example\r\n~~~~\r\n### Subtopic\r\nThe last word.")
        source = header + raw
        sections = list(_publisher_sections(source, True))
        self.assertEqual([s.label for s in sections],
                         ["Original plain text", "Root", "Root / Topic", "Topic / Subtopic"])
        self.assertEqual("".join(s.text for s in sections), raw)
        for section in sections:
            self.assertEqual(section.text, source[section.start_char:section.end_char])
        rows = evidence(source, "Subtopic unrelated nonmatching expressions")
        self.assertEqual([r["source_section"] for r in rows], ["Topic / Subtopic"])
        self.assertEqual(["subtopic"], rows[0]["matched_publisher_heading_terms"])
        self.assertEqual(rows[0]["excerpt_sha256"], digest(rows[0]["source_excerpt"]))

    def test_horizontal_rule_is_not_silently_treated_as_frontmatter(self):
        text = "---\n# Introduction\n---\n## Another section\nFinal body."
        self.assertEqual([s.label for s in _publisher_sections(text, True)],
                         ["Original plain text", "Introduction", "Introduction / Another section"])

    def test_repeated_sections_retain_distinct_physical_intervals(self):
        block = "## Resume\nResume when the validation succeeds.\n"
        text = "# Manual\n" + block * 2
        hits = evidence(text, "resume validation")
        self.assertEqual(2, len(hits))
        self.assertEqual(2, len({h["excerpt_start_char"] for h in hits}))
        self.assertEqual({digest(block)}, {h["source_span_sha256"] for h in hits})
        for row in hits:
            self.assertEqual(text[row["excerpt_start_char"]:row["excerpt_end_char"]], row["source_excerpt"])

    def test_heading_terms_do_not_bypass_literal_identifier_constraint(self):
        text = "# Credentials\nNo variable is defined.\n## Other\nUse APP_TOKEN_BACKUP.\n"
        self.assertFalse(evidence(text, "credentials APP_TOKEN"))
        rows = evidence(text + "## Runtime\nUse APP_TOKEN as a variable.", "credentials APP_TOKEN")
        self.assertEqual(["Credentials / Runtime"], [r["source_section"] for r in rows])


class NamedEvidenceOperatorTests(unittest.TestCase):
    def setUp(self):
        self.cli = cli()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.manifest = root / "sources.json"
        self.path = "guides/test.md"
        self.manifest.write_text(json.dumps({"sources": [{"path": self.path, "title": "Library Guide"}]}))
        dbpath = root / "test.sqlite3"
        with sqlite3.connect(dbpath) as db:
            db.execute("CREATE TABLE filesystem_entries(id,entry_type,relative_path_display)")
            db.execute("INSERT INTO filesystem_entries VALUES(5,'file',?)", (self.path,))
        text = "# Setup\nUse the endpoint for a connection."
        self.checked = mock.Mock(return_value=dict(
            title="Library Guide", original_sha256="a" * 64, derived_sha256=digest(text), derived_text=text))
        self.audit = mock.Mock(return_value=root / "audit.jsonl")
        self.printed = mock.Mock()
        for obj, key, value in [(self.cli, "DB", dbpath), (self.cli, "SOURCES", self.manifest),
                                (self.cli, "checked_original", self.checked),
                                (self.cli, "append_query_audit", self.audit)]:
            patch = mock.patch.object(obj, key, value)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch("builtins.print", self.printed)
        patch.start()
        self.addCleanup(patch.stop)

    def run_query(self, query, name=None):
        self.cli.original_source_evidence(SimpleNamespace(original_path=None, source_name=name,
            query=query, top=3, snippet_characters=200))
        return json.loads(self.printed.call_args.args[0])

    def test_quoted_source_question_and_explicit_name_keep_auditable_subject(self):
        for name, query in [(None, 'According to "Library Guide", which endpoint handles the connection?'),
                            ("Library Guide", "which endpoint handles the connection?")]:
            with self.subTest(name=name):
                result = self.run_query(query, name)
                self.assertEqual(self.path, result["original_path"])
                self.assertEqual(query, result["query"])
                self.assertEqual("which endpoint handles the connection?", result["evidence_query"])
                self.assertEqual("verified", result["source_name_match"]["status"])
                self.assertTrue(result["passages"])
                receipt = self.audit.call_args.args[0]
                self.assertEqual(query, receipt["query"])
                self.assertEqual(result["evidence_query"], receipt["evidence_query"])
                self.assertEqual(result["passages"][0]["excerpt_sha256"], receipt["ranked_source_spans"][0]["excerpt_sha256"])

    def test_bare_name_ambiguous_and_unrecognized_choices_never_read_an_original(self):
        for query, error in [("Library Guide", "supply the topic"), ("unlisted manual", "unrecognized")]:
            with self.assertRaisesRegex(ValueError, error):
                self.run_query(query)
            self.assertFalse(self.audit.call_args.args[0]["success"])
        self.manifest.write_text(json.dumps({"sources": [
            {"path":"older.md","title":"Library Guide"}, {"path":self.path,"title":"Library Guide"}]}))
        with self.assertRaisesRegex(ValueError, "2 catalog paths") as caught:
            self.run_query("endpoint", name="Library Guide")
        self.assertIn("older.md", str(caught.exception))
        self.assertIn(self.path, str(caught.exception))
        receipt = self.audit.call_args.args[0]
        self.assertEqual("ambiguous", receipt["source_name_match"]["status"])
        self.assertIsNone(receipt["original_path"])
        self.checked.assert_not_called()
