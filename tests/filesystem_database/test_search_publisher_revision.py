"""Search must verify the physical revision behind a ranked native summary."""
from __future__ import annotations

import base64
from contextlib import ExitStack, closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from file_system_catalog.hierarchical_summary import HierarchicalSummarizer
from file_system_catalog.hierarchical_summary_store import append_summary_rows, summary_rows
from file_system_catalog.summary_search_audit import make_search_audit
from test_hierarchical_summary import FakeSummaryModel, long_document
from test_hierarchical_summary_store import FakeEmbeddings, fixture
from test_summary_search_audit import load_cli


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class PublisherRevisionSearchTests(unittest.TestCase):
    def setUp(self):
        self.cli = load_cli()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        corpus = root / "library"
        cache = root / "private"
        (corpus / "books").mkdir(parents=True)
        (cache / "text").mkdir(parents=True)
        paths = dict(LIBRARY=corpus, CACHE=cache, LEDGER=root/"ledger.json",
                     SOURCES=root/"manifest.json", SOURCE_MAP=root/"sources.sqlite", DB=root/"native.sqlite")
        patches = ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(mock.patch.multiple(self.cli, **paths))
        self.embeddings = FakeEmbeddings()
        self.embeddings.embed_queries = self.embeddings.embed
        adapter = SimpleNamespace(LocalE5=lambda **kw: self.embeddings)
        patches.enter_context(mock.patch.dict(sys.modules, knowledge_library_semantic_model=adapter))
        self.names = ["books/guide.pdf", "books/other.pdf"]
        ledger, manifest = [], []
        with closing(fixture()) as db, closing(sqlite3.connect(paths["SOURCE_MAP"])) as sources:
            sources.execute("CREATE TABLE originals(source_path,source_sha256,derived_sha256,derived_path,status)")
            for i, name in enumerate(self.names):
                raw = bytes([65+i])*12345
                (corpus/name).write_bytes(raw)
                text = long_document(12,100) + (" TIMMY_DREAM" if i==0 else " NEGATED_CLAIM")
                derived = f"{i}.txt"
                (cache/"text"/derived).write_text(text, encoding="utf-8")
                original_sha, derived_sha = digest(raw), digest(text.encode())
                ledger.append(dict(path=name, status="complete", bytes_received=len(raw), sha256=original_sha))
                manifest.append(dict(path=name, title=name))
                sources.execute("INSERT INTO originals VALUES(?,?,?,?,?)",
                                (name,original_sha,derived_sha,derived,"complete"))
                if i:
                    db.execute("INSERT INTO filesystem_entries VALUES(2,?,?,?,?)",
                               (name,base64.b64encode(name.encode()).decode(),"file",len(raw)))
                    db.commit()
                tree=HierarchicalSummarizer(FakeSummaryModel(750),output_tokens=64,safety_tokens=16).summarize(
                    title=name,text=text)
                chunks,reps=summary_rows(db,original_path=name, original_sha256=original_sha,
                    original_bytes=len(raw),derived_sha256=derived_sha,derived_text=text,
                    summary_start_char=0,tree=tree,embedding_client=self.embeddings,
                    analysis_model="test-local",collection_id=self.cli.COLLECTION)
                append_summary_rows(db,chunk_rows=chunks,representation_rows=reps)
                if i==0:
                    self.query=tree.root.summary
            sources.commit()
            with closing(sqlite3.connect(paths["DB"])) as disk:
                db.backup(disk)
        paths["LEDGER"].write_text(json.dumps({"items":ledger}))
        paths["SOURCES"].write_text(json.dumps({"sources":manifest}))
        self.original = corpus/self.names[0]
        self.derivative = cache/"text"/"0.txt"

    def search(self, *, top=1, facets=False):
        return self.cli._search_impl(SimpleNamespace(query=self.query, top=top,
            with_facets=facets, nodes_per_original=4))

    def paths(self, **options):
        return [doc["original_path"] for doc in self.search(**options)["documents"]]

    def test_named_source_precedes_a_better_cosine_only_after_revision_checks(self):
        ranked = self.search(top=2)["documents"]
        preferred = ranked[1]["original_path"]
        original_query = self.query
        self.query = preferred
        self.embeddings.embed_queries = lambda _: self.embeddings.embed([original_query])
        for facets in (False, True):
            result = self.search(top=1, facets=facets)
            self.assertEqual(result["source_name_match"]["status"], "verified")
            hit = result["documents"][0]
            self.assertEqual(hit["original_path"], preferred)
            self.assertEqual(hit["rank_basis"], "explicit_source_name")
            self.assertEqual(hit["similarity"], ranked[1]["similarity"])
            self.assertEqual(hit["matched_summary_nodes"], ranked[1]["matched_summary_nodes"])
            audit = make_search_audit(query=self.query, result=result, elapsed_seconds=0, model="fixture", database="fixture")
            self.assertEqual(audit["ranked_results"][0]["rank_basis"], "explicit_source_name")
            self.assertEqual(audit["source_name_match"], result["source_name_match"])
        publisher = self.cli.LIBRARY / preferred
        publisher.write_bytes(b"X" * publisher.stat().st_size)
        missing = self.search(top=1)
        self.assertEqual(missing["source_name_match"]["status"], "no_verified_index_match")
        self.assertEqual(missing["documents"][0]["original_path"], ranked[0]["original_path"])
        self.assertNotIn("rank_basis", missing["documents"][0])

    def test_named_source_with_changed_derivative_or_unavailable_vectors_gets_backfilled(self):
        self.query = self.names[0]
        self.derivative.write_bytes(b"X" + self.derivative.read_bytes()[1:])
        self.assertEqual(self.paths(top=1), self.names[1:])
        with sqlite3.connect(self.cli.DB) as db:
            db.execute("UPDATE content_representations SET embedding_sha256=? WHERE chunk_id IN "
                       "(SELECT chunk_id FROM content_chunks WHERE filesystem_entry_id=2)", ("0"*64,))
        self.query = self.names[1]
        result = self.search(top=2)
        self.assertEqual(result["documents"], [])
        self.assertEqual(result["source_name_match"]["status"], "no_verified_index_match")

    def test_changed_publisher_bytes_at_same_size_and_mtime_get_backfilled(self):
        self.assertEqual(self.paths(), self.names[:1])
        stamp=self.original.stat()
        self.original.write_bytes(b"X"+self.original.read_bytes()[1:])
        os.utime(self.original, ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
        for facets in (False,True):
            with self.subTest(facets=facets):
                self.assertEqual(self.paths(facets=facets), self.names[1:])
        self.original.write_bytes(b"A"*12345)
        self.assertEqual(self.paths(), self.names[:1])

    def test_ambiguous_manifest_or_completion_receipt_is_excluded(self):
        for path, key in ((self.cli.SOURCES, "sources"), (self.cli.LEDGER, "items")):
            with self.subTest(key=key):
                saved = path.read_bytes()
                data = json.loads(saved)
                data[key].append(dict(data[key][0]))
                path.write_text(json.dumps(data))
                self.assertEqual(self.paths(top=2), self.names[1:])
                path.write_bytes(saved)

    def test_past_failed_receipt_does_not_mask_a_unique_completion(self):
        ledger = json.loads(self.cli.LEDGER.read_bytes())
        ledger["items"].append(dict(ledger["items"][0], status="failed"))
        self.cli.LEDGER.write_text(json.dumps(ledger))
        self.assertEqual(self.paths(), self.names[:1])

    def test_missing_or_wrong_size_publisher_is_excluded(self):
        self.original.write_bytes(b"short")
        self.assertEqual(self.paths(top=2), self.names[1:])
        self.original.unlink()
        self.assertEqual(self.paths(top=2), self.names[1:])

    def test_symbolic_link_and_escaping_parent_are_excluded_even_when_bytes_match(self):
        aside=self.original.with_suffix(".backup")
        self.original.rename(aside)
        self.original.symlink_to(aside)
        self.assertEqual(self.paths(top=2), self.names[1:])
        self.original.unlink()
        aside.rename(self.original)
        outside=Path(self.temp.name)/"external"
        self.original.parent.rename(outside)
        self.original.parent.symlink_to(outside, target_is_directory=True)
        self.assertEqual(self.paths(top=2), [])

    def test_file_replaced_while_hashing_fails_even_with_identical_bytes(self):
        replacement = self.original.with_suffix(".replacement")
        replacement.write_bytes(self.original.read_bytes())
        original_stat = Path.stat

        def replace_before_final_stat(path, *, follow_symlinks=True):
            if path == self.original and not follow_symlinks:
                os.replace(replacement, self.original)
            return original_stat(path, follow_symlinks=follow_symlinks)

        with mock.patch.object(Path, "stat", replace_before_final_stat):
            with self.assertRaisesRegex(ValueError, "changed during verification"):
                self.cli.file_sha256(self.original)

    def test_build_and_evidence_share_the_publisher_revision_check(self):
        result = self.cli.checked_original(self.names[0])
        self.assertEqual(result["derived_text"].encode(), self.derivative.read_bytes())
        self.original.write_bytes(b"Z" * self.original.stat().st_size)
        with self.assertRaisesRegex(ValueError, "original file SHA"):
            self.cli.checked_original(self.names[0])

    def test_changed_derivative_and_invalid_source_map_path_are_excluded(self):
        backup=self.derivative.read_bytes()
        self.derivative.write_bytes(b"X"+backup[1:])
        self.assertEqual(self.paths(top=2), self.names[1:])
        self.derivative.write_bytes(backup)
        with sqlite3.connect(self.cli.SOURCE_MAP) as db:
            db.execute("UPDATE originals SET derived_path=NULL WHERE source_path=?",self.names[:1])
        self.assertEqual(self.paths(top=2), self.names[1:])

    def test_one_hash_per_file_per_query_with_multiple_nodes(self):
        with mock.patch.object(self.cli, "file_sha256", wraps=self.cli.file_sha256) as hasher:
            result=self.search(top=2)
        self.assertGreater(len(result["documents"][0]["matched_summary_nodes"]),1)
        files=[c.args[0] for c in hasher.call_args_list]
        self.assertCountEqual(files,[self.cli.LIBRARY/n for n in self.names] +
                              [self.cli.CACHE/"text"/f"{i}.txt" for i in range(2)])
        self.assertEqual(result["originals_validated_for_version"],2)

    def test_audit_records_the_checked_revisions_for_each_selected_node(self):
        result=self.search()
        node=result["documents"][0]["matched_summary_nodes"][0]
        audit=make_search_audit(query=self.query,result=result,elapsed_seconds=.1,model="test",database="test")
        logged=audit["ranked_results"][0]["matched_nodes"][0]
        self.assertEqual("publisher_and_derived_sha256_checked;summary_is_not_primary_evidence",audit["provenance_policy"])
        for key in ("original_sha256","derived_sha256","source_passage_sha256"):
            self.assertEqual(logged[key],node[key])
