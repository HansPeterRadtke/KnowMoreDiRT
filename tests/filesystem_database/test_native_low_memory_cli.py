"""The alternate streaming scanner must preserve the installed source reader."""
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('kmd_low_memory_native_search',ROOT/'scripts'/'raspi_low_memory_native_original_search.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def base_result():
 return {'count':1,'results':[{'original_path':'docs/original.pdf',
        'exact_passage_hash_verified':True,'source_passage_sha256':'a'*64}],
        'final_current_original_source_revision_check':{
           'safe_to_cutover_live':False,'cross_store_immutable_generation_pinned':False}}


class NativeStreamingReaderCliTests(unittest.TestCase):
    def setUp(self):
        self.original=object()
        self.module=SimpleNamespace(search_semantic_entries=self.original)
        def search(query,*,top,mode,original_path):
            self.assertIs(self.module.search_semantic_entries,M.stream_exact_native_semantic_entries)
            self.assertEqual('Approved query',query)
            self.assertEqual(2,top)
            return base_result()
        self.module.search=search

    def test_installed_verified_reader_still_runs_with_only_process_local_scanner_replaced(self):
        result=M.native_low_memory_search('Approved query',top=2,native_module=self.module)
        self.assertEqual(1,result['count'])
        self.assertIs(self.original,self.module.search_semantic_entries)

    def test_exception_during_search_does_not_leave_a_patched_module(self):
        def faulty(*a,**kw):raise ValueError('verified original bytes changed')
        self.module.search=faulty
        with self.assertRaisesRegex(ValueError,'original bytes changed'):
            M.native_low_memory_search('Approved query',top=2,native_module=self.module)
        self.assertIs(self.original,self.module.search_semantic_entries)

    def test_output_missing_exact_passage_SHA_is_never_accepted(self):
        sample=base_result();sample['results'][0]['exact_passage_hash_verified']=False
        self.module.search=lambda *a,**kw:sample
        with self.assertRaisesRegex(ValueError,'SHA proof'):
            M.native_low_memory_search('Approved query',top=2,native_module=self.module)
        self.assertIs(self.original,self.module.search_semantic_entries)

    def test_invalid_query_and_boundaries_before_any_scanner_change(self):
        for kw in ({'query':''},{'query':'x'*601},{'query':'Approved query','top':6},
                   {'query':'Approved query','top':True},
                   {'query':'Approved query','original_path':'../sensitive.pdf'},
                   {'query':'Approved query','original_path':'/private/source'}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):
                M.native_low_memory_search(native_module=self.module,**kw)
            self.assertIs(self.original,self.module.search_semantic_entries)

    def test_reject_concurrent_active_Python_threads_before_patching(self):
        active=threading.Event();stop=threading.Event()
        thread=threading.Thread(target=lambda:(active.set(),stop.wait(timeout=1)))
        thread.start();active.wait(timeout=1)
        try:
            with self.assertRaisesRegex(RuntimeError,'single-threaded'):
                M.native_low_memory_search('Approved query',top=2,native_module=self.module)
            self.assertIs(self.original,self.module.search_semantic_entries)
        finally:stop.set();thread.join()

    def test_missing_installed_scanner_releases_override_lock_for_a_later_safe_run(self):
        missing=SimpleNamespace(search=lambda *a,**kw:base_result())
        with self.assertRaisesRegex(ValueError,'no recognized existing scanner'):
            M.native_low_memory_search('Approved query',top=2,native_module=missing)
        self.assertEqual(1,M.native_low_memory_search('Approved query',top=2,
                                                       native_module=self.module)['count'])
        self.assertIs(self.original,self.module.search_semantic_entries)

    def test_bare_mock_search_without_verification_guard_refused(self):
        self.module.search=lambda *a,**kw:{'count':0,'results':[]}
        with self.assertRaisesRegex(ValueError,'verified result'):
            M.native_low_memory_search('Approved query',top=2,native_module=self.module)


if __name__=='__main__':unittest.main()
