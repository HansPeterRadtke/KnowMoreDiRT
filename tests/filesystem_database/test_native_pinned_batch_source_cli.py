"""Batch CLI with one real SQLite source snapshot still rejects stale evidence."""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys
sys.path.insert(0,'/data/infra/libexec')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
from file_system_catalog.native_pinned_batch_snapshot import NativeOneBatchSnapshot
from knowledge_library_semantic_query import _ScopedNativeConnection
import raspi_batch_low_memory_original_search as batch
from test_native_streamed_semantic import StreamedExactNativeTests


def sha(s):return hashlib.sha256(s.encode()).hexdigest()


class PinnedSourceBatchIntegrationTests(unittest.TestCase):
    def setUp(self):
        fixture=StreamedExactNativeTests('test_empty_files_and_empty_vector_rows')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.fixture=fixture
        for i in range(12):fixture.add('docs/original-'+str(i%6)+'.html',i,
                            dtype='float16' if i%2 else 'float32')
        fixture.connection.commit()
        db=Path(fixture.connection.execute('PRAGMA database_list').fetchone()[2])
        self.snapshot=NativeOneBatchSnapshot(db);self.addCleanup(self.snapshot.close)
        self.queries=['First material source question','Second independent source question']
        self.constructed=[];self.events=[]
        self.native=SimpleNamespace(search_semantic_entries=lambda *a,**kw:[])
        self.old_scanner=self.native.search_semantic_entries
        model_reference=self
        class TestE5:
            model='intfloat/multilingual-e5-small-onnx-int8'
            revision='test-pinned-model-revision'
            def __init__(this,*,threads,batch_size):
                if (threads,batch_size)!=(2,4):raise AssertionError('unexpected model configuration')
                model_reference.constructed.append(this)
            def embed_queries(this,questions):
                i=self.queries.index(questions[0]);q=fixture.query.copy()
                if i:q=np.roll(q,i)
                return [q]
        self.native.LocalE5=TestE5;self.model_class=TestE5
        self.native.search=self.search
        self.native._originals=lambda: {'docs/original-1.html':{'path':'docs/original-1.html'}}
        self.native._completed_original_ledger_hashes=lambda: {'docs/original-1.html':'a'*64}
        self.native._current_derived_versions=lambda: {'docs/original-1.html':{'source_sha256':'a'*64,'derived_sha256':'b'*64}}
        self.native._final_returned_source_generation_check=self.final_check

    def search(self,q,*,top,mode,original_path):
        self.assertEqual(1,len(self.constructed))
        model=self.native.LocalE5(threads=2,batch_size=4)
        self.assertIs(model,self.constructed[0])
        vectors=model.embed_queries([q])
        candidates=self.native.search_semantic_entries(_ScopedNativeConnection(self.fixture.connection,mode='any'),vectors[0])
        self.assertEqual(6,len(candidates))
        self.events.append(('source-query',q,candidates[0]['chunk_id']))
        excerpt='Currently attested original source excerpt for '+q
        return {'count':1,'results':[{'original_path':candidates[0]['relative_path_display'],
                      'kmd_native_chunk_id':candidates[0]['chunk_id'],
                      'source_passage':excerpt,'source_passage_sha256':sha(excerpt),
                      'exact_passage_hash_verified':True,
                      'original_source_sha256':'a'*64}],
                'final_current_original_source_revision_check':{
                    'selected_originals_rechecked_after_query':1,
                    'cross_store_immutable_generation_pinned':False,
                    'safe_to_cutover_live':False}}

    def final_check(self,candidates,**kwargs):
        self.events.append(('final-original-and-native-check',len(candidates)))
        self.assertEqual(2,len(candidates))
        return {'selected_originals_rechecked_after_query':len(candidates),
                'cross_store_immutable_generation_pinned':False,
                'safe_to_cutover_live':False}

    def run_batch(self,**kw):
        return batch.search_verified_source_batch(self.queries,top=2,native_module=self.native,
                prepared_snapshot=self.snapshot,**kw)

    def test_pinned_snapshot_is_reused_for_both_queries_and_final_source_guard(self):
        result=self.run_batch()
        self.assertEqual(1,len(self.constructed))
        self.assertTrue(result['ephemeral_single_batch_read_only_current_vector_snapshot_used'])
        self.assertEqual(6,result['current_native_vector_snapshot_originals'])
        self.assertEqual(24,result['current_native_vector_snapshot_candidate_count'])
        self.assertEqual(2,result['current_source_originals_returned'])
        self.assertEqual('final-original-and-native-check',self.events[-1][0])
        self.assertEqual(3,len(self.events))
        self.assertIs(self.native.LocalE5,self.model_class)
        self.assertIs(self.native.search_semantic_entries,self.old_scanner)
        self.snapshot.require_current()

    def test_native_index_commit_during_second_query_refuses_entire_batch(self):
        genuine=self.search
        def race(q,**kw):
            if q==self.queries[-1]:
                self.fixture.add('docs/new-writer-original.html',99)
                self.fixture.connection.commit()
            return genuine(q,**kw)
        self.native.search=race
        with self.assertRaisesRegex(ValueError,'FileDB changed'):
            self.run_batch()
        self.assertEqual(1,len([x for x in self.events if x[0]=='source-query']))
        self.assertIs(self.native.LocalE5,self.model_class)
        self.assertIs(self.native.search_semantic_entries,self.old_scanner)

    def test_source_map_revision_change_detected_by_final_original_guard(self):
        def stale(*a,**kw):raise ValueError('publisher source-map changed after earlier result')
        self.native._final_returned_source_generation_check=stale
        with self.assertRaisesRegex(ValueError,'source-map changed'):
            self.run_batch()
        self.assertEqual(2,len(self.events))
        self.assertIs(self.native.search_semantic_entries,self.old_scanner)
        self.assertIs(self.native.LocalE5,self.model_class)

    def test_database_snapshot_can_only_satisfy_unfiltered_native_any_scope(self):
        for kw in ({'mode':'full'},{'original_path':'docs/original-1.html'}):
            with self.subTest(kw=kw),self.assertRaisesRegex(ValueError,'unfiltered any-mode'):
                self.run_batch(**kw)
        self.assertEqual([],self.events)
        self.assertEqual([],self.constructed)

    def test_context_restores_model_and_scanner_even_if_final_guard_errors(self):
        self.native._final_returned_source_generation_check=lambda *args,**kw:{
            'cross_store_immutable_generation_pinned':True,'safe_to_cutover_live':True}
        with self.assertRaisesRegex(ValueError,'final source batch proof'):
            self.run_batch()
        self.assertIs(self.native.LocalE5,self.model_class)
        self.assertIs(self.native.search_semantic_entries,self.old_scanner)

if __name__=='__main__':unittest.main()
