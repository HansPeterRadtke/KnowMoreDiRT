"""Exact catalog names, publisher-qualified archived skills, and ambiguity controls."""
import unittest
from file_system_catalog.named_source import resolve_source_name


def skill(slug, pin, vendor="nvidia"):
    path = f"skills/{vendor}/{slug}-{pin}/SKILL.md"
    return path, dict(path=path, title=f"{vendor} Agent Skill: {slug}", source_type="agent_skill_instructions")


class SourceNameTests(unittest.TestCase):
    def setUp(self):
        self.path, record = skill("nvidia-ontology-setup", "a123456")
        self.manifest = {self.path: record, "books/C.pdf": {"title": "C++ Reference"},
                         "books/D.pdf": {"title": "C# Reference"}}

    def test_direct_path_titles_and_affirmative_references(self):
        cases = [(self.path, "original_path"),
                 ("NVIDIA Agent Skill: nvidia-ontology-setup", "catalog_title"),
                 ("What does the NVIDIA ontology setup guide say about connecting?", "pinned_skill_name"),
                 ("Was steht im NVIDIA Ontology Setup Leitfaden zur Verbindung?", "pinned_skill_name"),
                 ('According to "nvidia-ontology-setup", how do we connect?', "pinned_skill_name")]
        for query, kind in cases:
            with self.subTest(query=query):
                result = resolve_source_name(query, self.manifest)
                self.assertEqual((result["original_path"], result["kind"]), (self.path, kind))
        self.assertIsNone(resolve_source_name(self.path.lower(), self.manifest))

    def test_unindexed_revision_and_duplicate_titles_are_still_ambiguous(self):
        self.manifest.update([skill("nvidia-ontology-setup", "b234567")])
        ambiguous = {"status": "ambiguous", "matching_catalog_paths": 2}
        self.assertEqual(resolve_source_name("nvidia ontology setup", self.manifest), ambiguous)
        self.assertEqual(resolve_source_name(self.path, self.manifest)["original_path"], self.path)
        self.manifest["other.pdf"] = {"title": "C++ Reference"}
        self.assertEqual(resolve_source_name('In the “C++ Reference”, what is constexpr?', self.manifest), ambiguous)

    def test_exact_path_has_priority_over_another_catalog_display_title(self):
        self.manifest["other.pdf"] = {"title": self.path}
        result = resolve_source_name(self.path, self.manifest)
        self.assertEqual((result["original_path"], result["kind"]), (self.path, "original_path"))

    def test_catalog_titles_keep_punctuation_and_unicode_identity(self):
        self.manifest["hci.pdf"] = {"title": "Fitts’s Law"}
        for query, path in [('According to "C++ Reference", how do templates work?', "books/C.pdf"),
                            ('According to "C# Reference", what is a delegate?', "books/D.pdf"),
                            ("Fitts's Law", "hci.pdf")]:
            self.assertEqual(resolve_source_name(query, self.manifest)["original_path"], path)
        self.assertIsNone(resolve_source_name("C Reference", self.manifest))

    def test_comparisons_exclusions_topic_mentions_and_partial_names(self):
        other, rec = skill("nvidia-ontology-query", "b234567")
        self.manifest[other] = rec
        for q in ("Compare the NVIDIA ontology setup guide with the NVIDIA ontology query guide.",
                  "What does the NVIDIA ontology setup guide and the NVIDIA ontology query guide say?",
                  "Exclude the NVIDIA ontology setup guide from the results.",
                  "How can I send MCP requests to the NVIDIA ontology setup runtime?",
                  "What does the ontology setup guide say?", "Similar to NVIDIA ontology setup",
                  "What does the Nvidia ontology setup wizard guide say?", "x"*1001):
            self.assertIsNone(resolve_source_name(q, self.manifest), q)

    def test_generic_skill_needs_catalog_identity_and_publisher(self):
        path, rec = skill("brand-guidelines", "b234567", "anthropic")
        self.manifest[path] = rec
        self.assertEqual(resolve_source_name("What does the Anthropic brand guidelines skill recommend?", self.manifest)["original_path"],path)
        self.assertIsNone(resolve_source_name("What does the brand guidelines skill recommend?", self.manifest))
        self.manifest[path] = {**rec, "title": "Other publisher catalog entry"}
        self.assertIsNone(resolve_source_name("What does the Anthropic brand guidelines skill recommend?", self.manifest))
