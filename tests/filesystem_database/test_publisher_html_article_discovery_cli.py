"""Real CLI and fresh-authority checks using only synthetic archived sources."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/raspi_publisher_html_article_discovery.py"
LEGACY = REPO / "scripts/raspi_publisher_firstunit_discovery.py"
SCOPE = "https://manual.example.org/manual/en/4.5/"
QUERY = "normalization exposure energy"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_cli():
    spec = importlib.util.spec_from_file_location("scoped_article_cli_integration", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    scripts = str(REPO / "scripts")
    sys.path.insert(0, scripts)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(scripts)
    return module


class SyntheticLibrary:
    def __init__(self, root):
        self.root = root
        self.originals = root / "originals"
        self.derivatives = root / "derivatives"
        self.originals.mkdir()
        self.derivatives.mkdir()
        self.catalog = root / "catalog.json"
        self.ledger = root / "ledger.json"
        self.source_map = root / "sources.sqlite3"
        self.native = root / "native.sqlite3"
        self.catalog_rows = []
        self.ledger_rows = []
        self.files = {}
        with sqlite3.connect(self.source_map) as db:
            db.execute("CREATE TABLE originals(source_path TEXT PRIMARY KEY,source_sha256 TEXT,"
                       "derived_sha256 TEXT,derived_path TEXT,original_kind TEXT,original_url TEXT,"
                       "original_title TEXT,original_bytes INTEGER,text_characters INTEGER,status TEXT)")
        with sqlite3.connect(self.native) as db:
            db.execute("CREATE TABLE filesystem_entries(id INTEGER PRIMARY KEY,"
                       "relative_path_display TEXT,entry_type TEXT)")
        self.selected = self.add_source("4.5", 1)
        self.other_version = self.add_source("5.2", 2)
        self.persist_metadata()

    def add_source(self, version, ident):
        path = "manuals/" + version + "/lights.html"
        url = "https://manual.example.org/manual/en/" + version + "/lights.html"
        title = "Synthetic lamp reference " + version
        body = ("Lamp normalization preserves total energy while the shape changes. "
                "The exposure control scales emitted energy and this deliberately synthetic "
                "reference is not a factual statement about any real software.")
        raw = ("<!doctype html><html><nav>Navigation-only unrelated topic.</nav>"
               "<main><article><h1>" + title + "</h1><p>" + body +
               "</p></article></main></html>").encode()
        original = self.originals / path
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(raw)
        derivative = self.derivatives / (path + ".txt")
        derivative.parent.mkdir(parents=True, exist_ok=True)
        derivative.write_text("Generated provenance header.\n[[SOURCE UNIT: HTML article]]\n" + body)
        derived = derivative.read_bytes()
        self.files[path] = (original, derivative)
        self.catalog_rows.append({"path": path, "kind": "html", "title": title,
                                  "source_url": url, "url": url,
                                  "rights_note": "Synthetic unit-test text only."})
        self.ledger_rows.append({"path": path, "status": "complete", "sha256": sha(raw),
                                 "bytes_received": len(raw)})
        with sqlite3.connect(self.source_map) as db:
            db.execute("INSERT INTO originals VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (path, sha(raw), sha(derived), path + ".txt", "html", url, title,
                        len(raw), len(derived.decode()), "complete"))
        with sqlite3.connect(self.native) as db:
            db.execute("INSERT INTO filesystem_entries VALUES(?,?,?)", (ident, path, "file"))
        return path

    def persist_metadata(self):
        self.catalog.write_text(json.dumps({"sources": self.catalog_rows}))
        self.ledger.write_text(json.dumps({"items": self.ledger_rows}))

    def arguments(self):
        return ["--source-map", str(self.source_map), "--native-db", str(self.native),
                "--original-root", str(self.originals), "--derivative-root", str(self.derivatives),
                "--completed-ledger", str(self.ledger)]

    def search_arguments(self):
        return dict(publisher_url_prefix=SCOPE, queries=[{"id": "one", "query": QUERY}], top=5,
                    source_map=self.source_map, native_db=self.native, catalog_path=self.catalog,
                    completed_ledger=self.ledger, original_root=self.originals,
                    derivative_root=self.derivatives)

    def mutate(self, kind):
        if kind in ("url", "title", "derived_path", "status"):
            column, value = {
                "url": ("original_url", "https://manual.example.org/manual/en/5.2/lights.html"),
                "title": ("original_title", "Reassigned synthetic publisher title"),
                "derived_path": ("derived_path", self.other_version + ".txt"),
                "status": ("status", "pending"),
            }[kind]
            with sqlite3.connect(self.source_map) as db:
                db.execute("UPDATE originals SET " + column + "=? WHERE source_path=?",
                           (value, self.selected))
        elif kind == "catalog":
            self.catalog_rows[0]["source_url"] = "https://manual.example.org/manual/en/5.2/lights.html"
            self.persist_metadata()
        elif kind == "receipt":
            self.ledger_rows[0]["status"] = "pending"
            self.persist_metadata()
        elif kind == "native":
            with sqlite3.connect(self.native) as db:
                db.execute("UPDATE filesystem_entries SET id=101 WHERE id=1")
        elif kind in ("original", "derivative"):
            path = self.files[self.selected][0 if kind == "original" else 1]
            path.write_bytes(path.read_bytes() + b" changed")
        else:
            raise AssertionError(kind)

    def source_digests(self):
        paths = [self.catalog, self.ledger, self.source_map, self.native]
        paths += [path for pair in self.files.values() for path in pair]
        return {str(path): sha(path.read_bytes()) for path in paths}


class PublisherHTMLArticleCLIIntegrationTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.fixture = SyntheticLibrary(self.root)
        self.outputs = self.root / "search-audit"
        self.outputs.mkdir()
        self.cli = load_cli()

    def run_cli(self, extra, *, legacy=False):
        command = [sys.executable, str(LEGACY if legacy else SCRIPT)]
        command += self.fixture.arguments()
        if not legacy:
            command += ["--catalog", str(self.fixture.catalog)]
        environment = os.environ.copy()
        environment.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        return subprocess.run(command + extra, capture_output=True, text=True,
                              env=environment, timeout=20)

    def assert_scoped_result(self, result, query_count=1):
        self.assertEqual("kmd.readonly_scoped_publisher_html_article_discovery.v1", result["schema"])
        self.assertEqual(query_count, result["query_count"])
        self.assertTrue(result["all_displayed_sources_freshly_revalidated_after_batch"])
        self.assertFalse(result["cross_store_atomic_generation_pinned"])
        self.assertFalse(result["out_of_scope_fallback_used"])
        self.assertEqual(1, result["source_scan"]["verified_original_articles_indexed"])
        for question in result["queries"]:
            for channel in question["independent_rank_channels"].values():
                self.assertEqual([self.fixture.selected], [r["original_path"] for r in channel["source_originals"]])
                proof = channel["source_originals"][0]
                self.assertEqual(1, proof["native_original_filedb_entry_id"])
                self.assertTrue(proof["source_url"].startswith(SCOPE))
                self.assertFalse(proof["factual_answer_verified"])
                self.assertFalse(proof["quoted_text_is_original_publisher_bytes"])
                self.assertEqual(sha(proof["source_excerpt"].encode()), proof["source_excerpt_sha256"])
                self.assertNotIn("original_file_identity", proof)
                self.assertNotIn("derivative_file_identity", proof)

    def test_explicit_opt_in_local_semantic_channels_use_full_original_provenance(self):
        from tests.filesystem_database.test_publisher_html_article_semantic import FakeLocalE5
        before=self.fixture.source_digests()
        result=self.cli.search(**self.fixture.search_arguments(),
                               include_local_semantic=True,semantic_model=FakeLocalE5())
        self.assertEqual("kmd.readonly_scoped_publisher_html_article_semantic.v1",result["schema"])
        self.assertTrue(result['local_article_semantic_opt_in'])
        self.assertTrue(result['local_semantic_channels_are_independent_and_never_score_fused'])
        self.assertTrue(result['all_displayed_sources_freshly_revalidated_after_batch'])
        self.assertFalse(result['cross_store_atomic_generation_pinned'])
        channels=result['queries'][0]['independent_rank_channels']
        self.assertEqual({'literal','porter','semantic_article','semantic_catalog_title'},set(channels))
        self.assertEqual([self.fixture.selected],
                         [z['original_path'] for z in channels['semantic_article']['source_originals']])
        for ch in ('semantic_article','semantic_catalog_title'):
            rows=channels[ch]['source_originals']
            self.assertEqual(1,len(rows))
            r=rows[0]
            self.assertEqual(self.fixture.selected,r['original_path'])
            self.assertEqual(1,r['native_original_filedb_entry_id'])
            self.assertTrue(r['original_and_derivative_full_bytes_rehashed'])
            self.assertFalse(r['factual_answer_verified'])
            self.assertEqual(sha(r['source_excerpt'].encode()),r['source_excerpt_sha256'])
            self.assertTrue(r['source_excerpt_is_whitespace_normalized_publisher_article_not_literal_HTML_bytes'])
            if ch=='semantic_catalog_title':
                self.assertTrue(r['title_only_rank_is_not_original_article_content_evidence'])
            else:
                self.assertFalse(r['title_only_rank_is_not_original_article_content_evidence'])
        self.assertEqual(before,self.fixture.source_digests())

    def test_optin_semantic_source_changes_after_initial_index_refuse(self):
        from tests.filesystem_database.test_publisher_html_article_semantic import FakeLocalE5
        class MutateDuringSemantic(FakeLocalE5):
            def embed_queries(_self, ss):
                original=self.fixture.files[self.fixture.selected][0]
                original.write_bytes(original.read_bytes()+b' unexpected altered bytes')
                return super().embed_queries(ss)
        with self.assertRaisesRegex(ValueError,'changed after article indexing'):
            self.cli.search(**self.fixture.search_arguments(),
                            include_local_semantic=True,
                            semantic_model=MutateDuringSemantic())

    def test_legacy_article_cli_does_not_import_optional_numpy_semantic_module(self):
        # No local model/NumPy dependency is allowed to leak into the old
        # two-channel command when the opt-in flag is absent.
        with mock.patch.dict(sys.modules, {
            'file_system_catalog.publisher_html_article_semantic': None
        }):
            default=self.cli.search(**self.fixture.search_arguments())
        self.assertEqual('kmd.readonly_scoped_publisher_html_article_discovery.v1',
                         default['schema'])
        self.assertEqual({'literal','porter'},
                         set(default['queries'][0]['independent_rank_channels']))
        self.assertNotIn('local_semantic_model',default)
        self.assertNotIn('local_article_semantic_opt_in',default)

    def test_optin_local_semantic_rejects_post_query_source_map_revision_change(self):
        from tests.filesystem_database.test_publisher_html_article_semantic import FakeLocalE5
        class MutateSourceMapDuringSemantic(FakeLocalE5):
            def embed_queries(_self, queries):
                with sqlite3.connect(self.fixture.source_map) as conn:
                    conn.execute('UPDATE originals SET derived_sha256=? WHERE source_path=?',
                                 ('f'*64,self.fixture.selected))
                return super().embed_queries(queries)
        with self.assertRaisesRegex(ValueError,'current source-map revision|source-map revision'):
            self.cli.search(**self.fixture.search_arguments(),
                            include_local_semantic=True,
                            semantic_model=MutateSourceMapDuringSemantic())

    def test_optin_local_semantic_rejects_post_query_derivative_bytes_change(self):
        from tests.filesystem_database.test_publisher_html_article_semantic import FakeLocalE5
        class MutateDerivativeDuringSemantic(FakeLocalE5):
            def embed_queries(_self, queries):
                derivative=self.fixture.files[self.fixture.selected][1]
                derivative.write_bytes(derivative.read_bytes()+b' invalid extra bytes')
                return super().embed_queries(queries)
        with self.assertRaisesRegex(ValueError,'current derivative bytes differ'):
            self.cli.search(**self.fixture.search_arguments(),
                            include_local_semantic=True,
                            semantic_model=MutateDerivativeDuringSemantic())

    def test_optin_local_semantic_requires_boolean_and_does_not_change_default(self):
        original=self.fixture.search_arguments()
        with self.assertRaises(ValueError):
            self.cli.search(**original,include_local_semantic=1)
        with self.assertRaises(ValueError):
            self.cli.search(**original,semantic_model=object())
        default=self.cli.search(**original)
        self.assertNotIn('local_article_semantic_opt_in',default)
        self.assertEqual({'literal','porter'},
                         set(default['queries'][0]['independent_rank_channels']))

    def test_actual_single_query_stdout_preserves_all_source_bytes(self):
        before = self.fixture.source_digests()
        run = self.run_cli(["--publisher-url-prefix", SCOPE, "--query", QUERY])
        self.assertEqual(0, run.returncode, run.stderr)
        self.assert_scoped_result(json.loads(run.stdout))
        self.assertEqual(before, self.fixture.source_digests())

    def test_actual_batch_creates_new_private_output_with_both_independent_channels(self):
        questions = self.root / "queries.json"
        questions.write_text(json.dumps({"queries": [
            {"id": "energy", "query": QUERY},
            {"id": "shape", "query": "lamp shape changes"}]}))
        output = self.outputs / "result.json"
        run = self.run_cli(["--publisher-url-prefix", SCOPE, "--queries-file", str(questions),
                            "--output", str(output)])
        self.assertEqual(0, run.returncode, run.stderr)
        self.assertEqual("", run.stdout)
        self.assert_scoped_result(json.loads(output.read_text()), query_count=2)
        self.assertEqual(0o600, output.stat().st_mode & 0o777)
        self.assertIn(sha(output.read_bytes()), run.stderr)

    def test_unavailable_version_is_empty_and_never_falls_back(self):
        run = self.run_cli(["--publisher-url-prefix", "https://manual.example.org/manual/en/99.9/",
                            "--query", QUERY])
        self.assertEqual(0, run.returncode, run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual("no_archived_sources_in_requested_scope", result["source_scan"]["scope_status"])
        self.assertEqual(0, result["selected_original_count"])
        self.assertFalse(result["out_of_scope_fallback_used"])
        for channel in result["queries"][0]["independent_rank_channels"].values():
            self.assertEqual([], channel["source_originals"])

    def test_output_never_overwrites_existing_owner_content(self):
        output = self.outputs / "result.json"
        output.write_bytes(b"Existing owner evidence must remain exact.\n")
        before = output.read_bytes()
        run = self.run_cli(["--publisher-url-prefix", SCOPE, "--query", QUERY,
                            "--output", str(output)])
        self.assertNotEqual(0, run.returncode)
        self.assertEqual(before, output.read_bytes())

    def test_output_refuses_original_derivative_native_and_symlinked_source_paths(self):
        alias = self.root / "source-alias"
        alias.symlink_to(self.fixture.originals, target_is_directory=True)
        targets = [self.fixture.originals / "forbidden-result.json",
                   self.fixture.derivatives / "forbidden-result.json",
                   self.root / "forbidden-result.json",
                   alias / "forbidden-result.json"]
        before = self.fixture.source_digests()
        for output in targets:
            with self.subTest(output=output):
                run = self.run_cli(["--publisher-url-prefix", SCOPE, "--query", QUERY,
                                    "--output", str(output)])
                self.assertNotEqual(0, run.returncode)
                self.assertIn("protected original or native-data", run.stderr)
                self.assertEqual("", run.stdout)
                self.assertFalse(output.exists())
        self.assertEqual(before, self.fixture.source_digests())

    def test_scope_is_required_and_query_inputs_remain_mutually_exclusive(self):
        run = self.run_cli(["--query", QUERY])
        self.assertNotEqual(0, run.returncode)
        self.assertIn("--publisher-url-prefix", run.stderr)
        self.assertEqual("", run.stdout)
        run = self.run_cli(["--publisher-url-prefix", SCOPE, "--query", QUERY,
                            "--queries-file", str(self.root / "absent.json")])
        self.assertNotEqual(0, run.returncode)
        self.assertEqual("", run.stdout)

    def test_legacy_firstunit_command_keeps_unscoped_interface_and_schema(self):
        before = self.fixture.source_digests()
        run = self.run_cli(["--query", QUERY], legacy=True)
        self.assertEqual(0, run.returncode, run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual("kmd.readonly_publisher_first_source_unit_dual_lexical_discovery.v1", result["schema"])
        self.assertNotIn("publisher_url_prefix", result)
        for channel in result["queries"][0]["independent_rank_channels"].values():
            self.assertEqual({self.fixture.selected, self.fixture.other_version},
                             {r["original_path"] for r in channel["source_originals"]})
        self.assertEqual(before, self.fixture.source_digests())

    def test_initial_valid_result_cannot_escape_after_authority_or_source_changes(self):
        for kind in ("url", "title", "derived_path", "status", "catalog", "receipt", "native",
                     "original", "derivative"):
            with self.subTest(kind=kind), TemporaryDirectory() as folder:
                fixture = SyntheticLibrary(Path(folder))
                original_verify = self.cli.verify_entry
                calls = 0

                def mutate_after_initial_verification(*args, **kwargs):
                    nonlocal calls
                    proof = original_verify(*args, **kwargs)
                    calls += 1
                    if calls == 1:
                        fixture.mutate(kind)
                    return proof

                with mock.patch.object(self.cli, "verify_entry", side_effect=mutate_after_initial_verification):
                    with self.assertRaises(ValueError):
                        self.cli.search(**fixture.search_arguments())
                self.assertGreaterEqual(calls, 1)

    def test_changes_after_final_hash_pass_are_caught_by_fresh_metadata_fence(self):
        for kind in ("url", "title", "derived_path", "status", "catalog", "receipt", "native",
                     "original", "derivative"):
            with self.subTest(kind=kind), TemporaryDirectory() as folder:
                fixture = SyntheticLibrary(Path(folder))
                original_verify = self.cli.verify_entry
                calls = 0

                def mutate_after_final_verification(*args, **kwargs):
                    nonlocal calls
                    proof = original_verify(*args, **kwargs)
                    calls += 1
                    if calls == 2:
                        fixture.mutate(kind)
                    return proof

                with mock.patch.object(self.cli, "verify_entry", side_effect=mutate_after_final_verification):
                    with self.assertRaises(ValueError):
                        self.cli.search(**fixture.search_arguments())
                self.assertEqual(2, calls)


if __name__ == "__main__":
    unittest.main()
