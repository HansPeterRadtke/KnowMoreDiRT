"""Explicit publication identifiers select only their own cataloged records."""
import unittest

from file_system_catalog.named_source import resolve_source_name


class PublicationIdentifierTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "papers/asr.pdf": {"kind": "pdf", "title": "Speech recognition", "doi": "https://doi.org/10.21437/Interspeech.2026-786"},
            "books/robot.pdf": {"kind": "pdf", "title": "Robot autonomy", "url": "https://arxiv.org/pdf/2608.03496v2",
                                "source_url": "https://arxiv.org/abs/2608.03496", "arxiv_id": "2608.03496"},
            "snapshots/related.html": {"kind": "html", "title": "Publisher interview",
                                      "url": "https://example.org/interview", "source_url": "https://doi.org/10.1234/interview"},
        }

    def assert_catalog_miss(self, query, manifest=None):
        match = resolve_source_name(query, self.manifest if manifest is None else manifest)
        self.assertEqual(match["status"], "no_catalog_match")
        self.assertEqual(match["matching_catalog_paths"], 0)
        self.assertNotIn("original_path", match)

    def test_exact_doi_forms(self):
        for query in ("10.21437/interspeech.2026-786", "DOI: 10.21437/Interspeech.2026-786",
                      "https://doi.org/10.21437/INTERSPEECH.2026-786"):
            with self.subTest(query=query):
                result = resolve_source_name(query, self.manifest)
                self.assertEqual((result["kind"], result["original_path"]), ("catalog_doi", "papers/asr.pdf"))

    def test_arxiv_exact_and_unversioned_forms(self):
        for query in ("arXiv:2608.03496v2", "arXiv: 2608.03496", "https://arxiv.org/pdf/2608.03496v2.pdf",
                      "https://arxiv.org/abs/2608.03496", "https://arxiv.org/pdf/2608.03496"):
            with self.subTest(query=query):
                result = resolve_source_name(query, self.manifest)
                self.assertEqual((result["kind"], result["original_path"]), ("catalog_arxiv", "books/robot.pdf"))
                self.assertEqual(result["matched_identifier"], "arXiv:2608.03496v2")
        self.assert_catalog_miss("arXiv:2608.03496v3", self.manifest)

    def test_quoted_and_direct_citations_return_only_subject(self):
        for query in ('According to "doi:10.21437/Interspeech.2026-786", what is WER?',
                      'According to 10.21437/Interspeech.2026-786, what is WER?'):
            match = resolve_source_name(query, self.manifest)
            self.assertEqual((match["subject_query"], match["original_path"]), ("what is WER?", "papers/asr.pdf"))

    def test_precise_versions_and_ambiguity_include_unindexed_editions(self):
        self.manifest["books/edition.pdf"] = dict(self.manifest["books/robot.pdf"],
                                                   url="https://arxiv.org/pdf/2608.03496v1")
        match = resolve_source_name("arXiv:2608.03496", self.manifest)
        self.assertEqual(match["candidate_paths"], ["books/edition.pdf", "books/robot.pdf"])
        self.assertEqual(resolve_source_name("arXiv:2608.03496v1", self.manifest)["original_path"], "books/edition.pdf")
        self.manifest["copies/asr.pdf"] = self.manifest["papers/asr.pdf"].copy()
        self.assertEqual(resolve_source_name("10.21437/Interspeech.2026-786", self.manifest)["matching_catalog_paths"], 2)

    def test_source_page_and_unversioned_download_do_not_prove_a_pinned_file(self):
        book = self.manifest["books/robot.pdf"]
        book.update(url="https://arxiv.org/pdf/2608.03496", source_url="https://arxiv.org/abs/2608.03496v2")
        self.assert_catalog_miss("arXiv:2608.03496v2", self.manifest)
        self.assertEqual(resolve_source_name("arXiv:2608.03496", self.manifest)["original_path"], "books/robot.pdf")
        book["url"] = "https://example.org/review.pdf"
        self.assert_catalog_miss("arXiv:2608.03496", self.manifest)
        self.assert_catalog_miss("10.1234/interview", self.manifest)

    def test_conflicting_catalog_metadata_suppresses_identifier_match(self):
        book = self.manifest["books/robot.pdf"]
        for field, wrong in (("arxiv_id", "2608.00000"), ("source_url", "https://arxiv.org/abs/2608.03496v1")):
            with self.subTest(field=field):
                m = dict(self.manifest, **{"books/robot.pdf": dict(book, **{field: wrong})})
                self.assert_catalog_miss("arXiv:2608.03496", m)
        self.manifest["papers/asr.pdf"]["source_url"] = "https://doi.org/10.1234/something-else"
        self.assert_catalog_miss("10.21437/interspeech.2026-786", self.manifest)

    def test_unknown_malformed_comparison_and_spoofed_queries_remain_unselected(self):
        for q in ("10.21437/interspeech.2026-78", "10.21437/interspeech.2026-786."):
            self.assert_catalog_miss(q)
        for q in ("arXiv:2608.03496v0", "2608.03496", "arXiv:2613.03496",
                  "https://arxiv.org.evil.example/pdf/2608.03496v2", "https://arxiv.org@evil.example/abs/2608.03496",
                  "https://arxiv.org/abs/2608.03496v2.pdf", "https://doi.org/10.21437/Interspeech.2026-786?via=other",
                  "Compare arXiv:2608.03496 with 10.21437/Interspeech.2026-786",
                  'According to "arXiv:2608.03496", and another study, what is WER?',
                  "Exclude arXiv:2608.03496", "According to arXiv:2608.03496 and other papers, what is WER?"):
            with self.subTest(q=q):
                self.assertIsNone(resolve_source_name(q, self.manifest))

    def test_authoritative_field_or_archive_url_is_required(self):
        path = "10.1234/path"
        self.manifest[path] = {"kind": "pdf", "title": "Exact path"}
        self.assertEqual(resolve_source_name(path, self.manifest)["kind"], "original_path")
        self.manifest["papers/mention.pdf"] = {"kind": "pdf", "title": "10.7777/not-owned", "abstract": "arXiv:2608.01234"}
        self.assert_catalog_miss("doi:10.7777/not-owned", self.manifest)
        self.assert_catalog_miss("arXiv:2608.01234", self.manifest)
