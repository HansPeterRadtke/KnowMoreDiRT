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

    def test_publication_requires_matching_acquisition_receipt(self):
        raw = json.loads(self.cli.SOURCES.read_text())
        book = raw["sources"][0]
        book.update(kind="pdf", doi="10.1234/guide", url="https://example.org/archive/guide.pdf")
        self.cli.SOURCES.write_text(json.dumps(raw))
        self.query = book["doi"]
        self.assertEqual(self.search()["source_name_match"]["status"], "no_verified_index_match")
        with self.assertRaisesRegex(ValueError, "acquisition URL"):
            self.cli.checked_original(self.names[0], publication_record=book)
        ledger = json.loads(self.cli.LEDGER.read_text())
        ledger["items"][0]["url"] = book["url"]
        self.cli.LEDGER.write_text(json.dumps(ledger))
        self.assertEqual(self.search()["source_name_match"]["status"], "verified")
        self.assertEqual(self.cli.checked_original(self.names[0], publication_record=book)["original_path"], self.names[0])
        with self.assertRaisesRegex(ValueError, "catalog record changed"):
            self.cli.checked_original(self.names[0], publication_record=dict(book, doi="10.1234/old"))

    def test_final_reader_fence_rechecks_current_source_bytes_and_both_databases(self):
        source=self.cli.checked_original(self.names[0],include_publisher_text=True)
        with sqlite3.connect(self.cli.DB) as db:
            native_id=db.execute("SELECT id FROM filesystem_entries WHERE relative_path_display=?",
                                 (self.names[0],)).fetchone()[0]
        result=self.cli.final_original_revision_guard(source,original_path=self.names[0],
                                                       native_file_id=native_id)
        self.assertTrue(result['original_publisher_bytes_rechecked_after_search'])
        self.assertTrue(result['derived_source_bytes_rechecked_after_search'])
        self.assertTrue(result['latest_source_map_revision_and_native_file_id_rechecked'])
        self.assertFalse(result['cross_store_immutable_generation_pinned'])
        self.assertFalse(result['safe_to_cutover_live'])
        with self.assertRaisesRegex(ValueError,'native FileDB identity'):
            self.cli.final_original_revision_guard(source,original_path=self.names[0],
                                                   native_file_id=native_id+101)
        self.original.write_bytes(b'Z'*source['original_bytes'])
        with self.assertRaisesRegex(ValueError,'SHA differs'):
            self.cli.final_original_revision_guard(source,original_path=self.names[0],
                                                   native_file_id=native_id)

    def test_final_reader_fence_rejects_derived_byte_tamper_without_index_writes(self):
        source=self.cli.checked_original(self.names[0])
        original_bytes=self.original.read_bytes()
        self.derivative.write_bytes(b'fake-derived-text-'*200)
        with self.assertRaisesRegex(ValueError,'derived original text changed'):
            self.cli.final_original_revision_guard(source,original_path=self.names[0],
                                                   native_file_id=1)
        self.assertEqual(original_bytes,self.original.read_bytes())

    def test_final_reader_fence_rechecks_approved_ledger_and_original_title(self):
        source=self.cli.checked_original(self.names[0])
        ledger=json.loads(self.cli.LEDGER.read_text())
        ledger['items'][0]['sha256']='b'*64
        self.cli.LEDGER.write_text(json.dumps(ledger))
        with self.assertRaisesRegex(ValueError,'completed original publisher receipt changed'):
            self.cli.final_original_revision_guard(source,original_path=self.names[0],
                                                   native_file_id=1)
        ledger['items'][0]['sha256']=source['original_sha256']
        self.cli.LEDGER.write_text(json.dumps(ledger))
        manifest=json.loads(self.cli.SOURCES.read_text())
        manifest['sources'][0]['title']='wrong publication edition after original lookup'
        self.cli.SOURCES.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'curated original metadata changed'):
            self.cli.final_original_revision_guard(source,original_path=self.names[0],
                                                   native_file_id=1)

    def test_final_reader_fence_detects_source_map_change_during_derived_sha_stream(self):
        source=self.cli.checked_original(self.names[0])
        base=self.cli.file_sha256
        def source_map_writer_between_first_and_second_fence(path,**kwargs):
            digest=base(path,**kwargs)
            if path==self.derivative:
                with sqlite3.connect(self.cli.SOURCE_MAP) as db:
                    db.execute('UPDATE originals SET derived_sha256=? WHERE source_path=?',
                               ('a'*64,self.names[0]))
            return digest
        with mock.patch.object(self.cli,'file_sha256',side_effect=source_map_writer_between_first_and_second_fence):
            with self.assertRaisesRegex(ValueError,'source-map.*changed'):
                self.cli.final_original_revision_guard(source,original_path=self.names[0],
                                                       native_file_id=1)

    def test_reader_fence_rejects_new_source_map_revision_published_during_search(self):
        args=SimpleNamespace(query='source provenance verified', original_path=self.names[0],
                             top=3,snippet_characters=300)
        def change_after_first_original_read(*_args,**_kwargs):
            with sqlite3.connect(self.cli.SOURCE_MAP) as db:
                db.execute('UPDATE originals SET derived_sha256=? WHERE source_path=?',
                           ('f'*64,self.names[0]))
            return {'passages':[], 'evidence_warning':'Source labels remain provisional.'}
        with mock.patch.object(self.cli,'_original_passages',side_effect=change_after_first_original_read), \
             mock.patch.object(self.cli,'verify_pdf_page_excerpts',return_value=[]), \
             mock.patch.object(self.cli,'append_query_audit') as audit:
            with self.assertRaisesRegex(ValueError,'source-map.*changed'):
                self.cli.original_source_evidence(args)
        self.assertEqual(1,audit.call_count)
        self.assertFalse(audit.call_args.args[0]['success'])

    def test_exact_section_respects_source_evidence_passage_limit(self):
        args = SimpleNamespace(query="RFC 9110, Section 9.2.2", top=0, snippet_characters=900)
        with mock.patch.object(self.cli, "append_query_audit") as log, self.assertRaisesRegex(ValueError, "passage count"):
            self.cli.original_source_evidence(args)
        self.assertFalse(log.call_args.args[0]["success"])

    def test_rfc_pin_requires_the_actual_publisher_header_and_acquisition_url(self):
        manifest = json.loads(self.cli.SOURCES.read_text())
        book = manifest["sources"][0]
        book.update(kind="text", url="https://www.rfc-editor.org/rfc/rfc9110.txt")
        self.cli.SOURCES.write_text(json.dumps(manifest))
        self.query = "RFC 9110"
        self.assertEqual(self.search()["source_name_match"]["status"], "no_verified_index_match")
        with self.assertRaisesRegex(ValueError, "acquisition URL"):
            self.cli.checked_original(self.names[0], include_publisher_text=True, publication_record=book)
        ledger = json.loads(self.cli.LEDGER.read_text())
        ledger["items"][0]["url"] = book["url"]
        self.cli.LEDGER.write_text(json.dumps(ledger))
        self.assertEqual(self.search(facets=True)["source_name_match"]["status"], "no_verified_index_match")
        with self.assertRaisesRegex(ValueError, "publisher header"):
            self.cli.checked_original(self.names[0], include_publisher_text=True, publication_record=book)

    def test_missing_publication_identifier_returns_no_semantic_substitute(self):
        self.query = "arXiv:2608.03496v3"
        with mock.patch.object(self.embeddings, "embed_queries", side_effect=AssertionError("unexpected embedding")):
            result = self.search(top=2)
        self.assertEqual(result["source_name_match"]["status"], "no_catalog_match")
        self.assertEqual(result["retrieval_engine"], "exact_publication_catalog_lookup_v1")
        self.assertEqual(result["documents"], [])
        self.assertIn("no archived PDF", result["relevance_warning"])

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
        self.assertIn("no verified summary result", missing["relevance_warning"])
        self.assertIn("source-evidence", missing["relevance_warning"])
        self.assertEqual(missing["documents"][0]["original_path"], ranked[0]["original_path"])
        self.assertNotIn("rank_basis", missing["documents"][0])

    def test_explicit_short_publisher_name_never_substitutes_a_different_source(self):
        manifest = json.loads(self.cli.SOURCES.read_text())
        manifest["sources"][0]["title"] = "CiteGuard: An Archived Method Paper"
        self.cli.SOURCES.write_text(json.dumps(manifest))
        self.query = "According to CiteGuard, how is the evidence selected?"
        with mock.patch.object(self.cli,"search_summary_entries_batched",
                               wraps=self.cli.search_summary_entries_batched) as search_impl:
            good = self.search(top=2)
        self.assertEqual(self.names[0],search_impl.call_args.kwargs["restrict_original_path"])
        self.assertEqual("verified", good["source_name_match"]["status"])
        self.assertEqual("catalog_title_lead", good["source_name_match"]["kind"])
        self.assertEqual(self.names[0], good["documents"][0]["original_path"])
        # A publisher revision mismatch must never allow a semantically
        # similar and fully verified different source to replace this one.
        self.original.write_bytes(b"Z" * self.original.stat().st_size)
        missing = self.search(top=2)
        self.assertEqual("no_verified_index_match", missing["source_name_match"]["status"])
        self.assertEqual([], missing["documents"])
        self.assertEqual(0, missing["originals_validated_for_version"])
        self.assertIn("no different publication was substituted", missing["relevance_warning"])
        # Restoring the original publisher bytes restores source-identity
        # resolution without reindexing or changing its model embedding.
        self.original.write_bytes(b"A" * 12345)
        self.assertEqual(self.names[0], self.search(top=2)["documents"][0]["original_path"])

    def mixed_doc(self):
        source=self.cli.checked_original(self.names[0])
        with sqlite3.connect(self.cli.DB) as db:
            ident=db.execute("SELECT id FROM filesystem_entries WHERE relative_path_display=?",
                             (self.names[0],)).fetchone()[0]
            chunk=db.execute("SELECT chunk_id FROM content_chunks WHERE filesystem_entry_id=? LIMIT 1",
                             (ident,)).fetchone()[0]
        return {'original_path':self.names[0], 'status':'verified',
                'original_sha256':source['original_sha256'],
                'derived_sha256':source['derived_sha256'],
                'filesystem_entry_id':ident,'discovery_chunk_id':chunk,
                'passages':[{'source_excerpt':'verified synthetic citation marker'}]}

    def test_mixed_original_final_fence_verifies_source_and_native_identity(self):
        doc=self.mixed_doc();rec=self.cli.final_mixed_document_source_revision_guard(
            [doc],versions=self.cli._current_versions())
        self.assertEqual(1,rec['verified_source_originals_rechecked_after_combined_evidence'])
        self.assertTrue(rec['final_source_map_native_ID_and_original_revision_checked'])
        self.assertFalse(rec['safe_to_cutover_live'])
        self.assertFalse(rec['cross_store_immutable_generation_pinned'])

    def test_mixed_original_final_fence_refuses_changed_original_and_source_map(self):
        doc=self.mixed_doc();versions=self.cli._current_versions()
        self.original.write_bytes(b'X'*self.original.stat().st_size)
        with self.assertRaisesRegex(ValueError,'original|publisher|SHA'):
            self.cli.final_mixed_document_source_revision_guard([doc],versions=versions)
        # Independent scenario, unchanged publisher, changed derived revision.
        self.original.write_bytes(b'A'*12345)
        with sqlite3.connect(self.cli.SOURCE_MAP) as db:
            db.execute('UPDATE originals SET derived_sha256=? WHERE source_path=?',
                       ('f'*64,self.names[0]))
        with self.assertRaisesRegex(ValueError,'source-map|revision'):
            self.cli.final_mixed_document_source_revision_guard([doc],versions=versions)

    def test_mixed_original_final_fence_refuses_changed_native_discovery_chunk(self):
        doc=self.mixed_doc();versions=self.cli._current_versions()
        with sqlite3.connect(self.cli.DB) as db:
            db.execute('DELETE FROM content_chunks WHERE chunk_id=?',(doc['discovery_chunk_id'],))
        with self.assertRaisesRegex(ValueError,'native discovery chunk'):
            self.cli.final_mixed_document_source_revision_guard([doc],versions=versions)

    def test_mixed_guard_rejects_source_map_change_after_physical_hash_check(self):
        doc=self.mixed_doc();versions=self.cli._current_versions()
        original=self.cli.final_original_revision_guard
        def commit_after_file_sha(*args,**kwargs):
            proof=original(*args,**kwargs)
            with sqlite3.connect(self.cli.SOURCE_MAP) as db:
                db.execute('UPDATE originals SET derived_sha256=? WHERE source_path=?',
                           ('f'*64,kwargs['original_path']))
            return proof
        with mock.patch.object(self.cli,'final_original_revision_guard',side_effect=commit_after_file_sha):
            with self.assertRaisesRegex(ValueError,'source-map.*changed'):
                self.cli.final_mixed_document_source_revision_guard([doc],versions=versions)

    def test_mixed_guard_does_not_claim_citation_when_no_publisher_pages_verified(self):
        doc=dict(self.mixed_doc(),status='no_verified_excerpt',passages=[])
        proof=self.cli.final_mixed_document_source_revision_guard([doc],versions=self.cli._current_versions())
        self.assertEqual(0,proof['verified_source_originals_rechecked_after_combined_evidence'])
        self.assertFalse(proof['safe_to_cutover_live'])

    def test_mixed_source_guard_rejects_unsafe_and_unbounded_sources(self):
        doc=self.mixed_doc();versions=self.cli._current_versions()
        invalid=dict(doc,original_path='../unknown.pdf')
        with self.assertRaisesRegex(ValueError,'unique source identity'):
            self.cli.final_mixed_document_source_revision_guard([invalid],versions=versions)
        with mock.patch.object(self.cli,'_MIXED_SOURCE_FINAL_PHYSICAL_BUDGET',1):
            with self.assertRaisesRegex(ValueError,'bounded physical read budget'):
                self.cli.final_mixed_document_source_revision_guard([doc],versions=versions)
        result=self.cli.final_mixed_document_source_revision_guard([],versions=versions)
        self.assertEqual(0,result['verified_source_originals_rechecked_after_combined_evidence'])

    def test_batched_summary_search_must_reject_source_map_revision_changed_after_ranking(self):
        # The pre-return checks in the ranked source callback may all pass,
        # then the separate extractor commits a newer derivative revision.
        original_search=self.cli.search_summary_entries_batched
        selected=self.names[0]
        def stale_after_candidates(*args,**kwargs):
            ranked=original_search(*args,**kwargs)
            self.assertTrue(ranked)
            with sqlite3.connect(self.cli.SOURCE_MAP) as db:
                db.execute('UPDATE originals SET derived_sha256=? WHERE source_path=?',('f'*64,selected))
            return ranked
        with mock.patch.object(self.cli,'search_summary_entries_batched',side_effect=stale_after_candidates):
            with self.assertRaisesRegex(ValueError,'source-map|revision changed'):
                self.search(top=1)

    def test_batched_summary_search_must_reject_publisher_same_size_modified_after_ranking(self):
        original_search=self.cli.search_summary_entries_batched
        original_bytes=self.original.stat().st_size
        def stale_original_after_candidates(*args,**kwargs):
            ranked=original_search(*args,**kwargs)
            self.assertTrue(ranked)
            self.original.write_bytes(b'X'*original_bytes)
            return ranked
        with mock.patch.object(self.cli,'search_summary_entries_batched',side_effect=stale_original_after_candidates):
            with self.assertRaisesRegex(ValueError,'original|publisher|SHA'):
                self.search(top=1)

    def test_batched_summary_search_must_reject_native_summary_chunk_removed_after_ranking(self):
        original_search=self.cli.search_summary_entries_batched
        def stale_native_after_candidates(*args,**kwargs):
            ranked=original_search(*args,**kwargs)
            self.assertTrue(ranked)
            first=ranked[0]['matched_summary_nodes'][0]
            with sqlite3.connect(self.cli.DB) as db:
                db.execute('DELETE FROM content_representations WHERE representation_id=?',
                           (first['native_representation_id'],))
            return ranked
        with mock.patch.object(self.cli,'search_summary_entries_batched',side_effect=stale_native_after_candidates):
            with self.assertRaisesRegex(ValueError,'summary|native|revision'):
                self.search(top=1)

    def test_post_summary_hash_writer_race_rejects_late_native_representation_change(self):
        original_guard=self.cli.final_original_revision_guard
        def late_native_write(*args,**kwargs):
            proof=original_guard(*args,**kwargs)
            with sqlite3.connect(self.cli.DB) as db:
                db.execute("UPDATE content_representations SET representation_text_sha256=? "
                           "WHERE representation_id=(SELECT representation_id "
                           "FROM content_representations ORDER BY representation_id LIMIT 1)",('d'*64,))
            return proof
        with mock.patch.object(self.cli,'final_original_revision_guard',side_effect=late_native_write):
            with self.assertRaisesRegex(ValueError,'native summary.*changed'):
                self.search(top=2)

    def test_post_summary_hash_writer_race_rejects_late_source_map_revision(self):
        original_guard=self.cli.final_original_revision_guard
        def late_extract_write(*args,**kwargs):
            proof=original_guard(*args,**kwargs)
            with sqlite3.connect(self.cli.SOURCE_MAP) as db:
                db.execute('UPDATE originals SET derived_sha256=? WHERE source_path=?',
                           ('d'*64,kwargs['original_path']))
            return proof
        with mock.patch.object(self.cli,'final_original_revision_guard',side_effect=late_extract_write):
            with self.assertRaisesRegex(ValueError,'source-map.*changed'):
                self.search(top=2)

    def test_summary_candidate_without_current_original_is_never_pin_approved(self):
        self.query='arXiv:2608.03496v3'
        with mock.patch.object(self.cli,'final_original_revision_guard',
                               side_effect=AssertionError('must not read an unknown publisher file')):
            result=self.search(top=2)
        self.assertEqual([],result['documents'])
        proof=result['final_current_summary_source_revision_check']
        self.assertEqual(0,proof['returned_originals_rechecked_after_summary_ranking'])
        self.assertFalse(proof['safe_to_cutover_live'])

    def test_unindexed_short_title_never_loads_e5_or_scans_other_originals(self):
        manifest=json.loads(self.cli.SOURCES.read_text())
        manifest["sources"][0]["title"]="CiteGuard: Original Publisher Research"
        self.cli.SOURCES.write_text(json.dumps(manifest))
        self.query="According to CiteGuard, what is the main claim?"
        with sqlite3.connect(self.cli.DB) as db:
            db.execute("DELETE FROM content_representations WHERE chunk_id IN "
                       "(SELECT chunk_id FROM content_chunks WHERE filesystem_entry_id=1)")
        with mock.patch.object(self.embeddings,"embed_queries",side_effect=AssertionError("model should not load")):
            result=self.search(top=2)
        self.assertEqual("no_verified_index_match",result["source_name_match"]["status"])
        self.assertEqual([],result["documents"])
        self.assertEqual("exact_catalog_unindexed_original_v1",result["retrieval_engine"])
        self.assertEqual(0,result["originals_validated_for_version"])

    def test_duplicate_short_publisher_titles_fail_closed_without_model_calls(self):
        manifest = json.loads(self.cli.SOURCES.read_text())
        manifest["sources"][0]["title"] = "ToolScope: Tool-Based Reasoning"
        manifest["sources"][1]["title"] = "ToolScope: A Separate Original Method"
        self.cli.SOURCES.write_text(json.dumps(manifest))
        self.query = "According to ToolScope, which policy is described?"
        with mock.patch.object(self.embeddings,"embed_queries",side_effect=AssertionError("model should not run")):
            result = self.search(top=2)
        self.assertEqual("ambiguous",result["source_name_match"]["status"])
        self.assertEqual(self.names,result["source_name_match"]["candidate_paths"])
        self.assertEqual([],result["documents"])
        self.assertEqual(0,result["originals_validated_for_version"])
        self.assertEqual("exact_ambiguous_catalog_lookup_v1",result["retrieval_engine"])
        self.assertIn("No document was substituted",result["relevance_warning"])

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

    def assert_false_anchor_rejected(self, change):
        self.query = self.names[0]
        with sqlite3.connect(self.cli.DB) as db:
            rows = db.execute(
                "SELECT r.representation_id,r.chunk_id,r.generation_json FROM content_representations r "
                "JOIN content_chunks c ON c.chunk_id=r.chunk_id WHERE c.filesystem_entry_id=1"
            ).fetchall()
            for rid, cid, raw in rows:
                g = json.loads(raw)
                if change == "digest":
                    g["source_passage_sha256"] = "e" * 64
                    db.execute("UPDATE content_chunks SET text_sha256=? WHERE chunk_id=?", ("e"*64, cid))
                else:
                    g["source_start_char"] += 500000
                    g["source_end_char"] += 500000
                    db.execute("UPDATE content_chunks SET start_char=?,end_char=? WHERE chunk_id=?",
                               (g["source_start_char"], g["source_end_char"], cid))
                db.execute("UPDATE content_representations SET generation_json=? WHERE representation_id=?",
                           (json.dumps(g), rid))
        for mode in (False, True):
            result = self.search(top=1, facets=mode)
            self.assertEqual(result["source_name_match"]["status"], "no_verified_index_match")
            self.assertEqual([d["original_path"] for d in result["documents"]], self.names[1:])

    def test_consistent_but_incorrect_database_digest_cannot_support_a_named_hit(self):
        self.assert_false_anchor_rejected("digest")

    def test_out_of_bounds_offsets_cannot_support_a_named_hit(self):
        self.assert_false_anchor_rejected("past_end")

    def test_a_bad_node_does_not_hide_other_valid_passages_in_the_same_source(self):
        self.query = self.names[0]
        before = self.search(top=1)["documents"][0]
        bad = before["matched_summary_nodes"][0]["native_representation_id"]
        with sqlite3.connect(self.cli.DB) as db:
            cid, raw = db.execute("SELECT chunk_id,generation_json FROM content_representations "
                                  "WHERE representation_id=?", (bad,)).fetchone()
            g = json.loads(raw)
            g["source_passage_sha256"] = "f"*64
            db.execute("UPDATE content_chunks SET text_sha256=? WHERE chunk_id=?", ("f"*64, cid))
            db.execute("UPDATE content_representations SET generation_json=? WHERE representation_id=?",
                       (json.dumps(g), bad))
        result = self.search(top=1)
        self.assertEqual(result["source_name_match"]["status"], "verified")
        remaining = result["documents"][0]
        self.assertEqual(remaining["indexed_summary_nodes_for_document"], before["indexed_summary_nodes_for_document"]-1)
        self.assertNotIn(bad, [n["native_representation_id"] for n in remaining["matched_summary_nodes"]])

    def test_unreadable_vector_and_corrupt_metadata_leave_healthy_sources_searchable(self):
        for kind in ("odd_vector", "malformed_json"):
            with self.subTest(kind=kind), sqlite3.connect(self.cli.DB) as db:
                rid = db.execute("SELECT representation_id FROM content_representations "
                                 "WHERE json_valid(generation_json) LIMIT 1").fetchone()[0]
                if kind == "odd_vector":
                    db.execute("UPDATE content_representations SET embedding_blob=?,embedding_sha256=? "
                               "WHERE representation_id=?", (b"x",digest(b"x"),rid))
                else:
                    db.execute("UPDATE content_representations SET generation_json='{' WHERE representation_id=?", (rid,))
                db.commit()
                result = self.search(top=2, facets=True)
                self.assertCountEqual([x["original_path"] for x in result["documents"]], self.names)
                for doc in result["documents"]:
                    self.assertNotIn(rid, [n["native_representation_id"] for n in doc["matched_summary_nodes"]])

    def test_duplicate_current_source_map_rows_block_explicit_name(self):
        self.query = self.names[0]
        with sqlite3.connect(self.cli.SOURCE_MAP) as db:
            db.execute("INSERT INTO originals SELECT * FROM originals WHERE source_path=?", (self.query,))
        result = self.search(top=1)
        self.assertEqual(result["source_name_match"]["status"], "no_verified_index_match")
        self.assertEqual(result["documents"][0]["original_path"], self.names[1])

    def test_each_source_is_hashed_once_during_ranking_and_again_before_return(self):
        with mock.patch.object(self.cli, "regular_file_blocks", wraps=self.cli.regular_file_blocks) as hasher:
            result=self.search(top=2)
        self.assertGreater(len(result["documents"][0]["matched_summary_nodes"]),1)
        files=[c.args[0] for c in hasher.call_args_list]
        expected=[self.cli.LIBRARY/n for n in self.names] + [self.cli.CACHE/"text"/f"{i}.txt" for i in range(2)]
        # The new full-source fence requires a SECOND read after ranking; this
        # is intentional, not duplicate reads per summary node.
        self.assertCountEqual(files,expected*2)
        self.assertEqual(result["originals_validated_for_version"],2)
        self.assertEqual(result['final_current_summary_source_revision_check'][
            'returned_originals_rechecked_after_summary_ranking'],2)
        self.assertFalse(result['final_current_summary_source_revision_check'][
            'cross_store_immutable_generation_pinned'])

    def test_audit_records_the_checked_revisions_for_each_selected_node(self):
        result=self.search()
        node=result["documents"][0]["matched_summary_nodes"][0]
        audit=make_search_audit(query=self.query,result=result,elapsed_seconds=.1,model="test",database="test")
        logged=audit["ranked_results"][0]["matched_nodes"][0]
        self.assertEqual("publisher_and_derived_sha256_checked;summary_is_not_primary_evidence",audit["provenance_policy"])
        self.assertEqual(1,audit["final_current_summary_source_revision_check"][
            "returned_originals_rechecked_after_summary_ranking"])
        self.assertFalse(audit["final_current_summary_source_revision_check"][
            "cross_store_immutable_generation_pinned"])
        for key in ("original_sha256","derived_sha256","source_passage_sha256"):
            self.assertEqual(logged[key],node[key])
