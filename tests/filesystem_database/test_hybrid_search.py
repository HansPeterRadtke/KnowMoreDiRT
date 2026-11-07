"""Native, nondestructive KnowMoreDiRT hybrid retrieval contract."""
from __future__ import annotations

import hashlib
import unittest

from file_system_catalog.hybrid_search import (
    BoundedPassageIndex,
    fuse_search_candidates,
    query_terms,
    search_hybrid_candidates,
)

A = "a" * 64
B = "b" * 64


def add(index, path, section, text, *, entry_id=1, title="Reference"):
    index.add_passage(
        filesystem_entry_id=entry_id,
        original_path=path,
        original_title=title,
        source_section=section,
        passage=text,
        original_sha256=A,
        derived_sha256=B,
    )


class HybridSearchTest(unittest.TestCase):
    def test_missing_gui_chapter_is_retrievable_without_embeddings(self):
        semantic = [
            {
                "relative_path_display": "ai-agents/gui-automation-agent.pdf",
                "score": 0.91,
                "chunk_id": "agent-vector",
            },
            {
                "relative_path_display": "skills/ai-agent-control.md",
                "score": 0.89,
                "chunk_id": "agent-skills-vector",
            },
        ]
        with BoundedPassageIndex() as idx:
            add(
                idx, "books/hci-textbook.pdf", "PDF page 503",
                "Graphical user interfaces: visual hierarchy, typography, "
                "spacing, contrast, and button layout govern the presentation of controls.",
                entry_id=401,
                title="Human Computer Interaction textbook",
            )
            add(
                idx, "books/hci-textbook.pdf", "PDF page 506",
                "Visual hierarchy and accessible touch targets on graphical "
                "user interfaces improve readability and control selection.",
                entry_id=401,
                title="Human Computer Interaction textbook",
            )
            result = search_hybrid_candidates(
                semantic, idx, "GUI design visual hierarchy spacing typography contrast buttons"
            )
        self.assertEqual("books/hci-textbook.pdf", result[0]["original_path"])
        self.assertIsNone(result[0]["semantic_rank"])
        self.assertEqual(1, result[0]["lexical_rank"])
        self.assertEqual(2, len(result[0]["lexical_passages"]))
        self.assertEqual(401, result[0]["lexical_passages"][0]["filesystem_entry_id"])
        self.assertTrue(all(
            item["evidence_status"].startswith("candidate")
            for item in result[0]["lexical_passages"]
        ))
        self.assertIn("candidates_only", result[0]["evidence_status"])
        self.assertIn(
            "PDF page 503",
            {x["source_section"] for x in result[0]["lexical_passages"]},
        )

    def test_lexical_does_not_throw_away_semantic_only_hits(self):
        semantic = [{"relative_path_display": "ai-agents/computer-use.pdf", "score": .83}]
        with BoundedPassageIndex() as idx:
            add(idx, "books/hci.pdf", "PDF page 9", "Typography and visual hierarchy")
            results = search_hybrid_candidates(semantic, idx, "Does the graph agree?")
        self.assertEqual(1, len(results))
        self.assertEqual("ai-agents/computer-use.pdf", results[0]["original_path"])
        self.assertIsNone(results[0]["lexical_rank"])

    def test_single_incidental_word_cannot_dominate(self):
        with BoundedPassageIndex() as idx:
            add(idx, "unrelated/generic.pdf", "page 1", "Design is a word used here.")
            add(
                idx, "design/guide.pdf", "page 2",
                "Visual hierarchy and typography are useful for design.",
            )
            rows = idx.search("visual hierarchy typography contrast design")
        self.assertEqual(["design/guide.pdf"], [r["original_path"] for r in rows])

    def test_fts_input_is_not_sql_or_sqlite_fts_syntax(self):
        with BoundedPassageIndex() as idx:
            add(idx, "d/doc.txt", "page 1", 'The "visual" hierarchy is important.')
            found = idx.search('visual" OR * ; DROP TABLE passage_fts; -- hierarchy')
            self.assertEqual(1, len(found))
            self.assertEqual(1, idx.passage_count)
            self.assertEqual(1, len(idx.search("visual hierarchy")))

    def test_unicode_and_exact_passage_digest(self):
        passage = "Barrierefreie Bedienung: Größe, Kontrast, Ästhetik und Tastflächen."
        with BoundedPassageIndex() as idx:
            add(idx, "u/handbuch.md", "Abschnitt 1", passage)
            row = idx.search("Kontrast und Tastflächen")[0]
        self.assertEqual(hashlib.sha256(passage.encode("utf-8")).hexdigest(), row["passage_sha256"])
        self.assertEqual(passage, row["passage"])

    def test_revision_and_path_preconditions(self):
        with BoundedPassageIndex(max_passages=1, max_characters=100) as idx:
            for unsafe in ("../private", "/absolute", "books/./file", "books//file", "books/../file", "books\\file", "books/\x00file"):
                with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                    add(idx, unsafe, "page 1", "Text")
            with self.assertRaises(ValueError):
                idx.add_passage(
                    filesystem_entry_id=1,
                    original_path="books/ref.txt",
                    original_title="Reference",
                    source_section="page 1",
                    passage="Text",
                    original_sha256="unverified",
                    derived_sha256=B,
                )
            add(idx, "books/ref.txt", "page 1", "Real semantic text")
            with self.assertRaises(ValueError):
                add(idx, "books/other.txt", "page 2", "Any content")
        with self.assertRaises(ValueError):
            BoundedPassageIndex(max_passages=0)

    def test_no_premature_confidence_or_equating_cosine_with_probability(self):
        sem = [{"relative_path_display": "chapter/law.pdf", "score": 0.995}]
        fusion = fuse_search_candidates(sem, [], top=1)
        self.assertEqual(0.995, fusion[0]["semantic_similarity"])
        self.assertNotIn("confidence", fusion[0])
        self.assertNotIn("answer", fusion[0])
        self.assertEqual(
            "candidates_only_requires_source_and_context_validation",
            fusion[0]["evidence_status"],
        )

    def test_generic_rank_fusion_is_deterministic_and_source_bounded(self):
        semantic = [
            {"relative_path_display": "a/file", "score": .90},
            {"relative_path_display": "b/file", "score": .89},
        ]
        lexical = [
            {"original_path": "b/file", "source_section": "page 3", "passage_sha256": "sha1"},
            {"original_path": "b/file", "source_section": "page 4", "passage_sha256": "sha2"},
            {"original_path": "b/file", "source_section": "page 5", "passage_sha256": "sha3"},
            {"original_path": "b/file", "source_section": "page 6", "passage_sha256": "sha4"},
        ]
        results = fuse_search_candidates(semantic, lexical, max_passages_per_source=3)
        self.assertEqual("b/file", results[0]["original_path"])
        self.assertEqual(3, len(results[0]["lexical_passages"]))
        self.assertEqual(results, fuse_search_candidates(semantic, lexical, max_passages_per_source=3))

    def test_query_terms_are_bounded_and_generic(self):
        self.assertEqual(("graphical", "user", "interface"), query_terms("The graphical user interface"))
        self.assertEqual(20, len(query_terms(" ".join("term" + str(i) for i in range(100)))))
        self.assertEqual((), query_terms("and the or"))
        with self.assertRaises(ValueError):
            query_terms("x" * 1001)
        with self.assertRaises(ValueError):
            query_terms("example", max_terms=0)

    def test_no_persistent_database_or_on_disk_cache(self):
        with BoundedPassageIndex() as idx:
            self.assertEqual("", idx._db.execute("pragma database_list").fetchone()[2])
            add(idx, "file/one.txt", "p1", "Testing layout hierarchy")
        with self.assertRaises(RuntimeError):
            idx.search("layout hierarchy")


if __name__ == "__main__":
    unittest.main()
