"""A live one-shot source batch never skips late original revision proofs."""
from __future__ import annotations
import hashlib
import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

P=Path(__file__).resolve().parents[2]/'scripts'/'raspi_batch_low_memory_original_search.py'
SPEC=importlib.util.spec_from_file_location('kmd_batch_native_source',P)
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def digest(s):return hashlib.sha256(s.encode()).hexdigest()


class NativeBatchCurrentSourceTests(unittest.TestCase):
    def setUp(self):
        self.events=[]
        self.original_scanner=object()
        self.module=SimpleNamespace(search_semantic_entries=self.original_scanner)
        self.models=[]
        class FakeE5:
            model='intfloat/multilingual-e5-small-onnx-int8'
            revision='pinned-verified-test-revision'
            def __init__(model,threads,batch_size):
                if (threads,batch_size)!=(2,4):raise ValueError('source model settings changed')
                self.models.append(model)
            def embed_queries(model,queries):return ['test-query-vector']
        self.source_model_class=FakeE5
        self.module.LocalE5=FakeE5
        self.module._originals=lambda: {'docs/original.pdf':{'path':'docs/original.pdf'}}
        self.module._completed_original_ledger_hashes=lambda: {'docs/original.pdf':'a'*64}
        self.module._current_derived_versions=lambda: {'docs/original.pdf':{'source_sha256':'a'*64,'derived_sha256':'b'*64}}
        self.module._final_returned_source_generation_check=self.guard
        self.module.search=self.search
        self.queries=['First official evidence question','Second cited evidence question']

    def search(self,q,*,top,mode,original_path):
        self.assertEqual(1,len(self.models))
        self.assertIs(self.module.LocalE5(threads=2,batch_size=4),self.models[0])
        from file_system_catalog.native_streamed_semantic import stream_exact_native_semantic_entries
        self.assertIs(self.module.search_semantic_entries,stream_exact_native_semantic_entries)
        self.events.append(('search',q))
        snippet='Exact current source text for '+q
        return {'count':1,'results':[{'source_passage':snippet,
            'source_passage_sha256':digest(snippet),
            'exact_passage_hash_verified':True,
            'original_path':'docs/original.pdf',
            'kmd_native_chunk_id':'chunk:'+q,
            'original_source_sha256':'a'*64}],
            'final_current_original_source_revision_check':{
               'selected_originals_rechecked_after_query':1,
               'cross_store_immutable_generation_pinned':False,
               'safe_to_cutover_live':False}}

    def guard(self,results,*,original_map,source_hashes,derived_versions):
        self.events.append(('final_group',len(results)))
        self.assertEqual(len(self.queries),len(results))
        self.assertEqual({'docs/original.pdf':{'path':'docs/original.pdf'}},original_map)
        self.assertEqual('a'*64,source_hashes['docs/original.pdf'])
        return {'selected_originals_rechecked_after_query':len(results),
            'cross_store_immutable_generation_pinned':False,'safe_to_cutover_live':False}

    def test_two_queries_reuse_one_model_and_verify_all_sources_after_both(self):
        result=M.search_verified_source_batch(self.queries,top=2,native_module=self.module)
        self.assertEqual(1,len(self.models))
        self.assertEqual(2,result['query_count'])
        self.assertEqual(2,result['current_source_originals_returned'])
        self.assertTrue(result['final_all_earlier_result_revisions_rechecked_after_last_query'])
        self.assertEqual([('search',self.queries[0]),('search',self.queries[1]),('final_group',2)],self.events)
        self.assertIs(self.source_model_class,self.module.LocalE5)
        self.assertIs(self.original_scanner,self.module.search_semantic_entries)
        self.assertFalse(result['non_atomic_cross_SQLite_source_generation_pinned'])

    def test_late_source_mutation_rejects_entire_batch_not_partial_old_passages(self):
        def changed(*args,**kwargs):raise ValueError('earlier publisher original revision changed during later query')
        self.module._final_returned_source_generation_check=changed
        with self.assertRaisesRegex(ValueError,'original revision changed'):
            M.search_verified_source_batch(self.queries,top=2,native_module=self.module)
        self.assertEqual(2,len([x for x in self.events if x[0]=='search']))
        self.assertIs(self.source_model_class,self.module.LocalE5)
        self.assertIs(self.original_scanner,self.module.search_semantic_entries)

    def test_early_query_error_restores_model_and_scanner(self):
        original=self.search
        def fail(q,**kw):
            if q==self.queries[-1]:raise RuntimeError('a later native source was unavailable')
            return original(q,**kw)
        self.module.search=fail
        with self.assertRaisesRegex(RuntimeError,'unavailable'):
            M.search_verified_source_batch(self.queries,top=2,native_module=self.module)
        self.assertIs(self.source_model_class,self.module.LocalE5)
        self.assertIs(self.original_scanner,self.module.search_semantic_entries)

    def test_malformed_or_incomplete_source_SHA_is_not_returned(self):
        original=self.search
        def tamper(q,**kw):
            r=original(q,**kw)
            r['results'][0]['source_passage_sha256']='0'*64
            return r
        self.module.search=tamper
        with self.assertRaisesRegex(ValueError,'excerpt SHA'):
            M.search_verified_source_batch(self.queries,top=2,native_module=self.module)
        self.assertIs(self.source_model_class,self.module.LocalE5)

    def test_unsafe_parameters_rejected_before_model_or_source_read(self):
        for queries,top,mode,path in [
          ([],2,'any',None),(['only one'],2,'any',None),
          (self.queries*4,2,'any',None),(self.queries[:1]*2,2,'any',None),
          (self.queries,5,'any',None),(self.queries,True,'any',None),
          (self.queries,2,'future',None),(self.queries,2,'any','../unsafe-original.pdf'),
          (self.queries,2,'any','/etc/secret'),
          (self.queries[:1]+['X'*601],2,'any',None)]:
            with self.subTest(queries=queries[:1],top=top,mode=mode,path=path),self.assertRaises(ValueError):
                M.search_verified_source_batch(queries,top=top,mode=mode,original_path=path,native_module=self.module)
        self.assertEqual([],self.models)
        self.assertEqual([],self.events)

    def test_nonfinite_model_or_bad_model_configuration_fail_closed(self):
        self.module.LocalE5=lambda **kw:SimpleNamespace(model='unknown',revision='new',embed_queries=None)
        with self.assertRaises(ValueError):M.search_verified_source_batch(self.queries,native_module=self.module)
        self.assertIs(self.original_scanner,self.module.search_semantic_entries)

    def test_parallel_Python_thread_rejected_not_the_live_system_process(self):
        active=threading.Event();stop=threading.Event()
        worker=threading.Thread(target=lambda:(active.set(),stop.wait(timeout=1)))
        worker.start();active.wait(timeout=1)
        try:
            with self.assertRaisesRegex(RuntimeError,'single-threaded'):
                M.search_verified_source_batch(self.queries,native_module=self.module)
            self.assertIs(self.source_model_class,self.module.LocalE5)
            self.assertIs(self.original_scanner,self.module.search_semantic_entries)
        finally:stop.set();worker.join(timeout=2)

    def test_without_snapshot_up_to_six_queries_retains_existing_bounded_behavior(self):
        names=['Question '+str(i) for i in range(6)]
        self.module.search=lambda q,**kw:{'count':0,'results':[],
              'final_current_original_source_revision_check':{
                 'cross_store_immutable_generation_pinned':False,'safe_to_cutover_live':False}}
        self.module._final_returned_source_generation_check=lambda candidates,**kw:{
              'selected_originals_rechecked_after_query':0,
              'cross_store_immutable_generation_pinned':False,'safe_to_cutover_live':False}
        result=M.search_verified_source_batch(names,native_module=self.module)
        self.assertEqual(6,result['query_count'])
        self.assertFalse(result['ephemeral_single_batch_read_only_current_vector_snapshot_used'])

    def test_empty_retrieval_candidates_can_abstain_per_query(self):
        self.module.search=lambda q,**kw:{'count':0,'results':[],
             'final_current_original_source_revision_check':{
                 'cross_store_immutable_generation_pinned':False,'safe_to_cutover_live':False}}
        self.module._final_returned_source_generation_check=lambda candidates,**kw:{
            'selected_originals_rechecked_after_query':0,'cross_store_immutable_generation_pinned':False,
            'safe_to_cutover_live':False}
        result=M.search_verified_source_batch(self.queries,native_module=self.module)
        self.assertEqual(0,result['current_source_originals_returned'])
        self.assertEqual([0,0],[len(x['results']) for x in result['full_installed_source_verified_query_results']])
        self.assertTrue(result['source_passages_not_factually_verified_answers'])


if __name__=='__main__':unittest.main()
