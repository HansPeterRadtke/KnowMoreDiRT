"""Current native vector table one-batch snapshot has exact scores or refuses."""
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0,'/data/infra/libexec')
import threading
import unittest
from unittest import mock
import numpy as np

from test_native_streamed_semantic import StreamedExactNativeTests
from file_system_catalog.content_pipeline import search_semantic_entries
from file_system_catalog.native_pinned_batch_snapshot import NativeOneBatchSnapshot
from file_system_catalog.native_streamed_semantic import stream_exact_native_semantic_entries
from knowledge_library_semantic_query import _ScopedNativeConnection


class PinnedBatchExactCurrentTests(unittest.TestCase):
    def setUp(self):
        f=StreamedExactNativeTests(methodName='test_empty_files_and_empty_vector_rows')
        f.setUp();self.addCleanup(f.doCleanups)
        self.f=f
        self.dbfile=Path(f.connection.execute('PRAGMA database_list').fetchone()[2])

    def add_several(self):
        for i in range(26):
            self.f.add('docs/current-'+str(i%11)+'.html',i,
                dtype='float16' if i%3==0 else 'float32',
                rep_status='pending' if i%7==0 else 'complete',
                chunk_collection='kmd:summary' if i%9==0 else 'kmd:full')
        self.f.connection.commit()

    @staticmethod
    def adapter(conn,mode='any',path=None):
        return _ScopedNativeConnection(conn,mode=mode,original_path=path)

    def test_same_exact_all_native_rank_rows_for_three_query_embeddings_in_one_snapshot(self):
        self.add_several()
        vectors=[self.f.query,self.f.rng.normal(size=384).astype('<f4'),
                 self.f.rng.normal(size=384).astype('<f4')]
        with NativeOneBatchSnapshot(self.dbfile) as s:
            self.assertEqual(11,s.complete_originals)
            self.assertGreater(len(s.candidates),25)
            for q in vectors:
                reference=search_semantic_entries(self.adapter(self.f.connection),q)
                expected=stream_exact_native_semantic_entries(self.adapter(self.f.connection),q)
                actual=s.rank(self.adapter(self.f.connection),q)
                self.assertEqual(reference,expected)
                self.assertEqual(reference,actual)
            s.require_current()
        self.assertTrue(s.closed)
        self.assertEqual([],s.candidates)

    def test_runtime_fails_closed_when_a_writer_commits_any_new_native_row(self):
        self.add_several()
        with NativeOneBatchSnapshot(self.dbfile) as s:
            ref=s.rank(self.adapter(self.f.connection),self.f.query)
            self.assertTrue(ref)
            self.f.add('docs/changed-after-snapshot.html',500)
            self.f.connection.commit()
            with self.assertRaisesRegex(ValueError,'FileDB changed'):
                s.require_current()
            with self.assertRaisesRegex(ValueError,'FileDB changed'):
                s.rank(self.adapter(self.f.connection),self.f.query)

    def test_no_silent_full_or_exact_path_filter_bypass(self):
        self.add_several()
        with NativeOneBatchSnapshot(self.dbfile) as snap:
            for kw in ({'mode':'full'}, {'mode':'overview'}, {'path':'docs/current-1.html'}):
                with self.subTest(kw=kw),self.assertRaisesRegex(ValueError,'path/mode scope'):
                    snap.rank(self.adapter(self.f.connection,**kw),self.f.query)
            self.assertEqual(11,len(snap.rank(self.adapter(self.f.connection),self.f.query)))

    def test_missing_native_vectors_or_exceeded_candidate_budget_refuses_snapshot(self):
        with self.assertRaisesRegex(ValueError,'invalid or oversized'):
            NativeOneBatchSnapshot(self.dbfile)
        self.add_several()
        with mock.patch('file_system_catalog.native_pinned_batch_snapshot.MAX_TOTAL_ROWS',4):
            with self.assertRaisesRegex(ValueError,'safe one-shot memory bound'):
                NativeOneBatchSnapshot(self.dbfile)

    def test_nonfinite_vector_strict_first_winner_matches_unmodified_semantic_scanner(self):
        q=self.f.query.copy()
        nan=np.full(384,np.nan,dtype='<f4')
        self.f.add('docs/finite.html',1,chunk_vec=q,rep_vec=q,rep_status='pending')
        self.f.add('docs/finite.html',2,chunk_vec=nan,rep_vec=q,rep_status='pending')
        self.f.connection.commit()
        with NativeOneBatchSnapshot(self.dbfile) as snap:
            existing=search_semantic_entries(self.adapter(self.f.connection),q)
            candidate=snap.rank(self.adapter(self.f.connection),q)
            self.assertEqual(existing,candidate)
            self.assertEqual('chunk-1',candidate[0]['chunk_id'])

    def test_high_memory_snapshot_denied_when_other_workers_need_ram(self):
        self.add_several()
        with mock.patch.object(NativeOneBatchSnapshot,'_physical_ram_available_bytes',return_value=600*1024*1024):
            with self.assertRaisesRegex(ValueError,'insufficient physical RAM'):
                NativeOneBatchSnapshot(self.dbfile)

    def test_empty_source_embeddings_do_not_promote_non_source_text(self):
        self.f.connection.execute('INSERT INTO filesystem_entries VALUES (?,?,?)',
                                  (1,'docs/empty.pdf',b'docs/empty.pdf'))
        self.f.connection.commit()
        with self.assertRaisesRegex(ValueError,'invalid or oversized'):
            NativeOneBatchSnapshot(self.dbfile)

    def test_runtime_RAM_drop_aborts_snapshot_before_stale_result(self):
        self.add_several()
        with NativeOneBatchSnapshot(self.dbfile) as snap:
            with mock.patch.object(snap,'_physical_ram_available_bytes',return_value=300*1024*1024):
                with self.assertRaisesRegex(ValueError,'memory headroom exhausted'):
                    snap.require_current()

    def test_queries_from_other_threads_are_denied_before_SQLite_access(self):
        self.add_several()
        with NativeOneBatchSnapshot(self.dbfile) as snapshot:
            outcome=[]
            def attempt():
                try:snapshot.require_current()
                except Exception as e:outcome.append((type(e).__name__,str(e)))
            thread=threading.Thread(target=attempt)
            thread.start();thread.join(timeout=2)
            self.assertEqual(1,len(outcome))
            self.assertIn('another thread',outcome[0][1])

if __name__=='__main__':unittest.main()
