"""Independent synthetic contracts for scoped archived-HTML retrieval."""
from __future__ import annotations

import copy
from contextlib import closing
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from file_system_catalog import publisher_html_article_discovery as article


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class PublisherScopeTests(unittest.TestCase):
    def test_exact_directory_origin_and_version_boundary(self):
        scope = article.PublisherScope.parse("https://DOCS.EXAMPLE.TEST/manual/en/4.5/")
        self.assertEqual("https://docs.example.test/manual/en/4.5/", scope.url)
        self.assertTrue(scope.contains("https://docs.example.test/manual/en/4.5/render/page.html"))
        for url in (
            "https://docs.example.test/manual/en/4.50/render/page.html",
            "https://docs.example.test/manual/en/5.2/render/page.html",
            "https://docs.example.test.evil.test/manual/en/4.5/page.html",
            "https://other.example.test/manual/en/4.5/page.html",
            "https://docs.example.test/manual/en/4.5",
            "https://docs.example.test/MANUAL/en/4.5/page.html",
        ):
            with self.subTest(url=url):
                self.assertFalse(scope.contains(url))

    def test_ambiguous_urls_refused_before_normalization(self):
        bad = (
            "", None, 1, b"https://example.test/a/",
            "https://example.test/a", "http://example.test/a/",
            "//example.test/a/", "https://user@example.test/a/",
            "https://user:password@example.test/a/", "https://example.test:443/a/",
            "https://example.test/a/?q=other", "https://example.test/a/#other",
            "https://example.test/a/%2e%2e/", "https://example.test/a/%252f/",
            "https://example.test/a/../b/", "https://example.test/./a/",
            "https://example.test/a//b/", "https://example.test/a\\b/",
            " https://example.test/a/", "https://example.test/a/\n",
            "https://exam\tple.test/a/", "https://example.test/a/\x00",
            "https://example.test/a/\x7f", "https://example.test/caf\u00e9/",
            "https://localhost/a/", "https://example..test/a/",
        )
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    article.PublisherScope.parse(value)

    def test_invalid_source_url_is_not_a_match(self):
        scope = article.PublisherScope.parse("https://example.test/a/")
        for value in ("https://example.test/a/%2f", "https://example.test/a/?x=1",
                      "https://example.test/a/#x", "https://example.test@evil.test/a/"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    scope.contains(value)


class ArticleParserTests(unittest.TestCase):
    def test_article_selection_removes_known_navigation_and_hidden_text(self):
        raw = (
            '<main><nav>outside-navigation</nav><article><h1>Original &amp; approved</h1>'
            '<nav>navigation-decoy</nav><aside>aside-decoy</aside><footer>footer-decoy</footer>'
            '<script>script-decoy</script><style>style-decoy</style>'
            '<noscript>noscript-decoy</noscript><template>template-decoy</template>'
            '<svg><text>svg-decoy</text></svg><canvas>canvas-decoy</canvas>'
            '<button>button-decoy</button><div role="navigation">role-decoy</div>'
            '<div role="navigation region">multi-role-decoy</div>'
            '<div role=" navigation ">spaced-role-decoy</div>'
            '<div hidden>hidden-decoy</div><div inert>inert-decoy</div>'
            '<span aria-hidden="true">aria-decoy</span>'
            '<p style="display: none !important">display-decoy</p>'
            '<p style="visibility: hidden;">visibility-decoy</p>'
            '<p>The archived material method preserves measured surface detail.</p>'
            '</article><footer>outside-footer</footer></main>'
        ).encode()
        selector, body = article.extract_article(raw)
        self.assertEqual("article", selector)
        self.assertEqual(
            "Original & approved The archived material method preserves measured surface detail.",
            body,
        )
        self.assertNotIn("decoy", body)
        self.assertNotIn("outside", body)

    def test_unique_main_fallback_and_inline_entity_normalization(self):
        raw = b'<main><h1>Research</h1><p>A stable <em>source</em> keeps its &quot;exact&quot; archive identity.</p></main>'
        self.assertEqual(
            ("main", 'Research A stable source keeps its "exact" archive identity.'),
            article.extract_article(raw),
        )

    def test_ambiguous_hidden_incomplete_or_unusable_containers_refused(self):
        fact = "A sufficiently long synthetic article sentence about material provenance."
        bad = [
            "<div>" + fact + "</div>",
            "<article>" + fact,
            "<article><article>" + fact + "</article></article>",
            "<article>" + fact + "</article><article>" + fact + "</article>",
            "<main>" + fact + "</main><main>" + fact + "</main>",
            "<div><article>" + fact + "</div>",
            "<article hidden>" + fact + "</article>",
            "<article aria-hidden='true'>" + fact + "</article>",
            "<main>" + fact + "<article>short</article></main>",
            "<article>short</article>",
            "<article>" + fact + "\0" + "</article>",
        ]
        for text in bad:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    article.extract_article(text.encode())
        with self.assertRaises(ValueError):
            article.extract_article(b"<article>\xff</article>")

    def test_structure_character_and_input_bounds(self):
        with self.assertRaises(ValueError):
            article.extract_article("not bytes")
        with self.assertRaises(ValueError):
            article.extract_article(b"")
        with mock.patch.object(article, "MAX_HTML_BYTES", 40):
            with self.assertRaises(ValueError):
                article.extract_article(b"<article>" + b"x" * 50 + b"</article>")
        with mock.patch.object(article, "MAX_ARTICLE_CHARS", 80):
            with self.assertRaises(ValueError):
                article.extract_article(b"<article>" + b"x" * 81 + b"</article>")
        with self.assertRaises(ValueError):
            article.extract_article(("<article>" + "<div>" * 512 +
                                     "Synthetic text" + "</div>" * 512 + "</article>").encode())


class ScopedArticleFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.originals = self.base / "originals"
        self.derived = self.base / "derived"
        self.originals.mkdir()
        self.derived.mkdir()
        self.sm_path = self.base / "source-map.sqlite3"
        self.native_path = self.base / "native.sqlite3"
        self.writer = sqlite3.connect(self.sm_path)
        self.native_writer = sqlite3.connect(self.native_path)
        self.addCleanup(self.writer.close)
        self.addCleanup(self.native_writer.close)
        definitions = (
            "source_path TEXT,source_sha256 TEXT,derived_sha256 TEXT,derived_path TEXT,"
            "original_kind TEXT,original_url TEXT,original_title TEXT,original_bytes INTEGER,"
            "text_characters INTEGER,status TEXT"
        )
        self.writer.execute("CREATE TABLE originals(" + definitions + ")")
        self.native_writer.execute(
            "CREATE TABLE filesystem_entries(id INTEGER,relative_path_display TEXT,entry_type TEXT)"
        )
        self.catalog = {}
        self.ledger = {}
        self.original_bytes = {}
        self.derivative_bytes = {}
        self.scope = article.PublisherScope.parse("https://docs.example.test/manual/en/4.5/")
        self.path = "manual/version45/material.html"
        self.other_path = "manual/version52/material.html"
        self.add_source(
            self.path, self.scope.url + "material.html",
            b"<article><h1>Material reference</h1><p>Normalization preserves total power when the light shape changes.</p></article>",
            identifier=10,
        )
        self.add_source(
            self.other_path, "https://docs.example.test/manual/en/5.2/material.html",
            b"<article><h1>Material reference</h1><p>The alternate version has a separate archived identity.</p></article>",
            identifier=20,
        )
        self.add_source(
            "manual/version45/archive.zip", self.scope.url + "archive.zip",
            b"not HTML and never to be parsed", identifier=30, kind="zip",
        )
        self.sm = article.readonly(self.sm_path)
        self.native = article.readonly(self.native_path)
        self.addCleanup(self.sm.close)
        self.addCleanup(self.native.close)
        self.index = article.ArticleRAMIndex()
        self.addCleanup(self.index.db.close)

    def add_source(self, path, url, raw, *, identifier, kind="html", title="Synthetic material reference"):
        original = self.originals / path
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(raw)
        derived_raw = ("Synthetic derivative. Original path " + path).encode()
        target = self.derived / (path + ".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(derived_raw)
        row = (path, sha(raw), sha(derived_raw), path + ".txt", kind, url,
               title, len(raw), len(derived_raw), "complete")
        self.writer.execute("INSERT INTO originals VALUES(?,?,?,?,?,?,?,?,?,?)", row)
        self.writer.commit()
        self.native_writer.execute("INSERT INTO filesystem_entries VALUES(?,?,'file')",
                                   (identifier, path))
        self.native_writer.commit()
        self.catalog[path] = {
            "path": path, "title": title, "source_url": url, "url": url,
            "kind": kind, "rights_note": "Synthetic test fixture, authored for this test.",
        }
        self.ledger[path] = {
            "path": path, "status": "complete", "sha256": sha(raw),
            "bytes_received": len(raw),
        }
        self.original_bytes[path] = raw
        self.derivative_bytes[path] = derived_raw

    def scan(self, **extra):
        return article.load_scoped_articles(
            source_map=self.sm, original_root=self.originals,
            catalog=self.catalog, ledger=self.ledger, index=self.index,
            scope=self.scope, **extra
        )

    def entry(self):
        return self.index.entries[self.index.ids[self.path]]

    def verify(self, entry=None):
        return article.verify_entry(
            entry or self.entry(), source_map=self.sm, native_db=self.native,
            catalog=self.catalog, ledger=self.ledger, original_root=self.originals,
            derived_root=self.derived, scope=self.scope,
        )

    def final_check(self, entry, proof):
        with closing(article.readonly(self.sm_path)) as fresh_sm, closing(article.readonly(self.native_path)) as fresh_native:
            return article.final_metadata_fence(
                {entry.path: entry}, {entry.path: proof}, source_map=fresh_sm,
                native_db=fresh_native, catalog=self.catalog, ledger=self.ledger,
                scope=self.scope, original_root=self.originals, derived_root=self.derived,
            )


class ScopedArticleRetrievalTests(ScopedArticleFixture):
    def test_only_exact_scope_html_is_indexed_with_explicit_nonhtml_coverage(self):
        result = self.scan()
        self.assertEqual(2, result["catalog_records_matching_scope"])
        self.assertEqual(1, result["eligible_html_records"])
        self.assertEqual(1, result["non_html_records_excluded"])
        self.assertEqual(1, result["verified_original_articles_indexed"])
        self.assertEqual(0, result["skipped_html_records"])
        self.assertEqual("scoped_html_ready", result["scope_status"])
        self.assertFalse(result["out_of_scope_fallback_allowed"])
        self.assertEqual({self.path}, set(self.index.ids))
        for channel in ("literal", "porter"):
            hits = self.index.ranked("normalization total power", channel=channel)
            self.assertEqual(self.path, hits[0][0].path)
            self.assertLess(hits[0][2], 0)

    def test_unknown_scope_returns_empty_without_other_version_fallback(self):
        self.scope = article.PublisherScope.parse("https://docs.example.test/manual/en/9.9/")
        result = self.scan()
        self.assertEqual("no_archived_sources_in_requested_scope", result["scope_status"])
        self.assertEqual({}, self.index.entries)
        self.assertEqual([], self.index.ranked("material", channel="literal"))

    def test_late_fact_and_porter_forms_produce_exact_original_article_offsets(self):
        path = "manual/version45/late.html"
        body = ("Background neutral descriptive sentences. " * 210 +
                "Displacement preserves measured geometric detail after normalization.")
        self.add_source(path, self.scope.url + "late.html",
                        ("<article><nav>navigationonlysecret</nav><p>" + body +
                         "</p></article>").encode(), identifier=40, title="Late source")
        self.scan()
        self.assertEqual([], self.index.ranked("navigationonlysecret", channel="literal"))
        self.assertEqual([], self.index.ranked("displacements", channel="literal"))
        entry = self.index.ranked("displacements", channel="porter")[0][0]
        self.assertEqual(path, entry.path)
        proof = self.index.excerpt(entry, "displacements", channel="porter")
        self.assertGreater(proof["normalized_article_start_char"], 6500)
        self.assertIn("Displacement", proof["source_excerpt"])
        self.assertIn("displacement", proof["matched_article_forms"])
        self.assertEqual(
            entry.text[proof["normalized_article_start_char"]:proof["normalized_article_end_char"]],
            proof["source_excerpt"],
        )
        self.assertEqual(sha(proof["source_excerpt"].encode()), proof["source_excerpt_sha256"])
        self.assertLessEqual(len(proof["source_excerpt"]), 520)

    def test_query_channel_excerpt_and_duplicate_bounds(self):
        self.scan()
        for query in ("", "of the and", "x" * 601, None, 2):
            with self.subTest(query=query):
                with self.assertRaises(ValueError):
                    self.index.ranked(query, channel="literal")
        for limit in (0, -1, 101, True, 1.5):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    self.index.ranked("material", channel="literal", limit=limit)
        with self.assertRaises(ValueError):
            self.index.ranked("material", channel="unknown")
        with self.assertRaises(ValueError):
            self.index.add(self.entry())
        for length in (159, 2001, True):
            with self.assertRaises(ValueError):
                self.index.excerpt(self.entry(), "power", channel="literal", length=length)
        with self.assertRaises(ValueError):
            self.index.excerpt(self.entry(), "absentkeyword", channel="literal")

    def test_reserved_evidence_markers_cannot_forge_fts_coordinates(self):
        self.scan()
        forged = replace(self.entry(), text=self.entry().text + "\x1eKMD_START\x1e")
        with self.assertRaisesRegex(ValueError, "reserved"):
            self.index.excerpt(forged, "power", channel="literal")

    def test_invalid_metadata_is_excluded_and_identified(self):
        self.catalog[self.path]["url"] = "https://docs.example.test/manual/en/5.2/material.html"
        result = self.scan()
        self.assertEqual(0, result["verified_original_articles_indexed"])
        self.assertEqual(1, result["skipped_html_records"])
        self.assertEqual(self.path, result["skipped_sources"][0]["original_path"])
        self.assertIn("URLs disagree", result["skipped_sources"][0]["reason"])

    def test_scan_budget_and_record_bounds(self):
        for field, bad in (("max_records", 0), ("max_records", True),
                           ("max_records", 513), ("max_total_bytes", 0),
                           ("max_total_bytes", -1), ("max_total_bytes", True),
                           ("max_total_bytes", article.MAX_SCOPE_BYTES + 1)):
            with self.subTest(field=field, value=bad):
                with self.assertRaises(ValueError):
                    self.scan(**{field: bad})
        with self.assertRaises(RuntimeError):
            self.scan(max_total_bytes=len(self.original_bytes[self.path]) - 1)
        self.add_source("manual/version45/second.html", self.scope.url + "second.html",
                        b"<article>A separate article about rendering reference methods.</article>",
                        identifier=41)
        with self.assertRaisesRegex(ValueError, "cardinality"):
            self.scan(max_records=1)

    def test_false_small_metadata_size_cannot_overread_scope_budget(self):
        raw = self.original_bytes[self.path]
        budget = len(raw) // 2
        self.writer.execute("UPDATE originals SET original_bytes=1 WHERE source_path=?", (self.path,))
        self.writer.commit()
        self.ledger[self.path]["bytes_received"] = 1
        real_read = article.read_bounded
        observed = []
        def measured_read(path, limit):
            data = real_read(path, limit)
            observed.append(len(data))
            return data
        with mock.patch.object(article, "read_bounded", side_effect=measured_read):
            result = self.scan(max_total_bytes=budget)
        self.assertLessEqual(sum(observed), budget)
        self.assertLessEqual(result["original_html_byte_budget_used"], budget)
        self.assertEqual(0, result["verified_original_articles_indexed"])

    def test_duplicate_source_map_identity_is_rejected(self):
        self.writer.execute("INSERT INTO originals SELECT * FROM originals WHERE source_path=?",
                            (self.path,))
        self.writer.commit()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.scan()

    def test_unclosed_original_is_an_explicit_skip(self):
        self.add_source("manual/version45/broken.html", self.scope.url + "broken.html",
                        b"<article>This article has enough content but no closing container.",
                        identifier=44)
        result = self.scan()
        self.assertEqual(1, result["skipped_html_records"])
        self.assertEqual("partial_html_coverage", result["scope_status"])
        self.assertFalse(result["all_scoped_html_articles_available"])


class ScopedArticleProvenanceTests(ScopedArticleFixture):
    def setUp(self):
        super().setUp()
        self.scan()
        self.selected = self.entry()

    def test_full_byte_revision_native_identity_and_false_claim_flags(self):
        proof = self.verify()
        self.assertEqual(10, proof["native_original_filedb_entry_id"])
        self.assertEqual(sha(self.original_bytes[self.path]), proof["publisher_original_sha256"])
        self.assertEqual(sha(self.derivative_bytes[self.path]), proof["current_derivative_sha256"])
        self.assertTrue(proof["original_and_derivative_full_bytes_rehashed"])
        self.assertFalse(proof["quoted_text_is_original_publisher_bytes"])
        self.assertFalse(proof["browser_visibility_or_rendering_verified"])
        self.assertFalse(proof["factual_answer_verified"])
        self.assertIsNone(self.final_check(self.selected, proof))

    def test_catalog_receipt_and_url_mutations_after_indexing_refused(self):
        original_catalog = copy.deepcopy(self.catalog)
        original_ledger = copy.deepcopy(self.ledger)
        cases = [
            ("catalog", "title", "Changed provenance title"),
            ("catalog", "source_url", "https://other.example.test/manual/en/4.5/material.html"),
            ("catalog", "url", "https://docs.example.test/manual/en/5.2/material.html"),
            ("catalog", "rights_note", "Changed archival permission metadata"),
            ("ledger", "status", "pending"),
            ("ledger", "sha256", "0" * 64),
            ("ledger", "bytes_received", len(self.original_bytes[self.path]) + 1),
            ("ledger", "extra_note", "A changed receipt identity"),
        ]
        for target, key, value in cases:
            with self.subTest(target=target, key=key):
                self.catalog = copy.deepcopy(original_catalog)
                self.ledger = copy.deepcopy(original_ledger)
                (self.catalog if target == "catalog" else self.ledger)[self.path][key] = value
                with self.assertRaises(ValueError):
                    self.verify()
        self.catalog = original_catalog
        self.ledger = original_ledger

    def test_source_map_fields_rechecked_after_indexing(self):
        fields = [
            ("source_sha256", "0" * 64), ("derived_sha256", "0" * 64),
            ("derived_path", "other.txt"), ("original_kind", "pdf"),
            ("original_url", "https://docs.example.test/manual/en/5.2/material.html"),
            ("original_title", "Reassigned title"), ("original_bytes", 1),
            ("text_characters", 1), ("status", "pending"),
        ]
        for field, value in fields:
            original = self.writer.execute("SELECT " + field + " FROM originals WHERE source_path=?",
                                           (self.path,)).fetchone()[0]
            with self.subTest(field=field):
                self.writer.execute("UPDATE originals SET " + field + "=? WHERE source_path=?",
                                    (value, self.path))
                self.writer.commit()
                with self.assertRaisesRegex(ValueError, "source-map"):
                    self.verify()
                self.writer.execute("UPDATE originals SET " + field + "=? WHERE source_path=?",
                                    (original, self.path))
                self.writer.commit()

    def test_original_and_derivative_corruption_refused(self):
        for root, relative, raw in (
            (self.originals, self.path, self.original_bytes[self.path]),
            (self.derived, self.path + ".txt", self.derivative_bytes[self.path]),
        ):
            with self.subTest(root=root.name):
                path = root / relative
                path.write_bytes(b"X" + raw[1:])
                with self.assertRaises(ValueError):
                    self.verify()
                path.write_bytes(raw)

    def test_missing_or_duplicate_native_identity_refused(self):
        self.native_writer.execute("DELETE FROM filesystem_entries WHERE relative_path_display=?",
                                   (self.path,))
        self.native_writer.commit()
        with self.assertRaisesRegex(ValueError, "unique"):
            self.verify()
        self.native_writer.executemany("INSERT INTO filesystem_entries VALUES(?,?,'file')",
                                       [(10, self.path), (99, self.path)])
        self.native_writer.commit()
        with self.assertRaisesRegex(ValueError, "unique"):
            self.verify()

    def test_final_fence_detects_native_identity_change(self):
        proof = self.verify()
        self.native_writer.execute("UPDATE filesystem_entries SET id=99 WHERE relative_path_display=?",
                                   (self.path,))
        self.native_writer.commit()
        with self.assertRaisesRegex(ValueError, "native identity"):
            self.final_check(self.selected, proof)

    def test_final_fence_detects_source_map_url_change_in_fresh_view(self):
        proof = self.verify()
        self.writer.execute("UPDATE originals SET original_url=? WHERE source_path=?",
                            ("https://docs.example.test/manual/en/5.2/material.html", self.path))
        self.writer.commit()
        with self.assertRaisesRegex(ValueError, "source-map"):
            self.final_check(self.selected, proof)

    def test_final_fence_detects_catalog_and_receipt_change(self):
        proof = self.verify()
        self.catalog[self.path]["rights_note"] = "Changed after full byte verification"
        with self.assertRaisesRegex(ValueError, "catalog"):
            self.final_check(self.selected, proof)
        self.catalog[self.path]["rights_note"] = "Synthetic test fixture, authored for this test."
        self.ledger[self.path]["status"] = "pending"
        with self.assertRaises(ValueError):
            self.final_check(self.selected, proof)

    def test_final_fence_detects_file_replacement_even_with_identical_bytes(self):
        proof = self.verify()
        path = self.originals / self.path
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
        with self.assertRaisesRegex(ValueError, "source file changed"):
            self.final_check(self.selected, proof)

    def test_symlinked_original_and_derivative_are_refused(self):
        for root, relative in ((self.originals, self.path), (self.derived, self.path + ".txt")):
            path = root / relative
            saved = path.with_name(path.name + ".fixture")
            path.rename(saved)
            path.symlink_to(saved)
            with self.subTest(root=root.name):
                with self.assertRaisesRegex(ValueError, "symlink"):
                    self.verify()
            path.unlink()
            saved.rename(path)

    def test_query_only_requirement_and_incomplete_proof_set(self):
        with self.assertRaisesRegex(ValueError, "query-only"):
            article.verify_entry(
                self.selected, source_map=self.writer, native_db=self.native,
                catalog=self.catalog, ledger=self.ledger, original_root=self.originals,
                derived_root=self.derived, scope=self.scope,
            )
        with self.assertRaisesRegex(ValueError, "query-only"):
            article.native_identity(self.native_writer, self.path)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            article.final_metadata_fence(
                {self.path: self.selected}, {}, source_map=self.sm, native_db=self.native,
                catalog=self.catalog, ledger=self.ledger, scope=self.scope,
                original_root=self.originals, derived_root=self.derived,
            )


class BoundedMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_bounded_reads_validate_limits_and_file_identity(self):
        path = self.root / "data"
        path.write_bytes(b"known synthetic bytes")
        self.assertEqual(b"known synthetic bytes", article.read_bounded(path, 32))
        for limit in (0, -1, -2, True, 2.5, article.MAX_METADATA_BYTES + 1):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    article.read_bounded(path, limit)
        with self.assertRaises(ValueError):
            article.read_bounded(path, 3)
        link = self.root / "link"
        link.symlink_to(path)
        with self.assertRaises(ValueError):
            article.read_bounded(link, 32)
        before = article.file_identity(path)
        after = (*before[:-1], before[-1] + 1)
        with mock.patch.object(article, "file_identity", side_effect=[before, after]):
            with self.assertRaisesRegex(ValueError, "changed"):
                article.read_bounded(path, 32)

    def test_metadata_path_duplicates_unsafe_paths_and_population_bounds(self):
        path = self.root / "catalog.json"
        row = {"path": "manual/reference.html", "kind": "html"}
        path.write_text(json.dumps({"sources": [row]}))
        self.assertEqual({row["path"]: row}, article.load_records(path, "sources"))
        for rows in ([row, row], [{"path": "../escape.html"}],
                     [{"path": "/absolute.html"}], [{"path": "a\\b"}],
                     [{"path": "a//b"}], [{"path": "a\0b"}], [None]):
            with self.subTest(rows=rows):
                path.write_text(json.dumps({"sources": rows}))
                with self.assertRaises(ValueError):
                    article.load_records(path, "sources")
        path.write_text(json.dumps({"sources": [row, {"path": "manual/other.html"}]}))
        with mock.patch.object(article, "MAX_METADATA_ROWS", 1):
            with self.assertRaises(ValueError):
                article.load_records(path, "sources")

    def test_readonly_connection_cannot_write(self):
        path = self.root / "db.sqlite3"
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE sample(value TEXT)")
        db = article.readonly(path)
        self.addCleanup(db.close)
        with self.assertRaises(sqlite3.OperationalError):
            db.execute("INSERT INTO sample VALUES('forbidden')")


if __name__ == "__main__":
    unittest.main()
