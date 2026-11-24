"""Experimental memory-bounded native semantic scan must be EXACT legacy ranking."""
import sqlite3
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest
import numpy as np
from file_system_catalog.content_pipeline import search_semantic_entries
from file_system_catalog.native_streamed_semantic import stream_exact_native_semantic_entries


class StreamedExactNativeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        p=Path(self.tmp.name)/'fixture.sqlite3'
        self.connection=sqlite3.connect(p)
        self.addCleanup(self.connection.close)
        self.connection.executescript('''
            CREATE TABLE filesystem_entries (id INTEGER PRIMARY KEY,relative_path_display TEXT,relative_path_b64 BLOB);
            CREATE TABLE content_chunks (
              chunk_id TEXT PRIMARY KEY,file_id TEXT,filesystem_entry_id INTEGER,
              chunk_kind TEXT,chunk_index INTEGER,start_char INTEGER,end_char INTEGER,
              embedding_dimension INTEGER,embedding_dtype TEXT,embedding_blob BLOB,collection_id TEXT);
            CREATE TABLE content_representations (
              representation_id TEXT PRIMARY KEY,representation_kind TEXT,representation_text TEXT,
              embedding_dimension INTEGER,embedding_dtype TEXT,embedding_blob BLOB,chunk_id TEXT,
              analysis_status TEXT);
        ''')
        self.rng=np.random.default_rng(20261124)
        self.query=self.rng.normal(size=384).astype('<f4')

    def add(self,filename,index,dtype='float32',*,chunk_vec=None,rep_vec=None,
            chunk_collection='kmd:full',rep_status='complete'):
        if chunk_vec is None:chunk_vec=self.rng.normal(size=384).astype('<f4')
        if rep_vec is None:rep_vec=self.rng.normal(size=384).astype('<f4')
        file_id='file-'+filename
        entry_id=next((x[0] for x in self.connection.execute('SELECT id FROM filesystem_entries WHERE relative_path_display=?',(filename,))),None)
        if entry_id is None:
            entry_id=self.connection.execute('SELECT count(*)+1 FROM filesystem_entries').fetchone()[0]
            self.connection.execute('INSERT INTO filesystem_entries VALUES (?,?,?)',(entry_id,filename,filename.encode('utf8')))
        no=self.connection.execute('SELECT count(*) FROM content_chunks').fetchone()[0]+1
        chunkid='chunk-'+str(no)
        v=np.asarray(chunk_vec,dtype='<f2' if dtype=='float16' else '<f4')
        self.connection.execute('INSERT INTO content_chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)',(
            chunkid,file_id,entry_id,'chunk',index,index*500,(index+1)*500,384,dtype,v.tobytes(),chunk_collection))
        v=np.asarray(rep_vec,dtype='<f2' if dtype=='float16' else '<f4')
        self.connection.execute('INSERT INTO content_representations VALUES (?,?,?,?,?,?,?,?)',(
            'representation-'+str(no),'summary_long','full distinct source passage '+str(no),
            384,dtype,v.tobytes(),chunkid,rep_status))
        return chunkid

    def compare(self, connection=None):
        db=connection or self.connection
        old=search_semantic_entries(db,self.query)
        new=stream_exact_native_semantic_entries(db,self.query)
        self.assertEqual(old,new)
        return new

    def test_mixed_float16_float32_representations_file_level_order_and_chunk_identity(self):
        for i in range(17):
            self.add('docs/original-'+str(i%7)+'.pdf',i,dtype='float16' if i%2 else 'float32')
        self.connection.commit()
        rank=self.compare()
        self.assertEqual(7,len(rank))
        self.assertTrue(all(x['file_id'].startswith('file-docs/') for x in rank))

    def test_exact_ties_keep_first_chunk_and_not_last_representation(self):
        q=self.query.copy()
        a=self.add('docs/first.html',1,chunk_vec=q,rep_vec=q)
        b=self.add('docs/first.html',2,chunk_vec=q,rep_vec=q)
        self.add('docs/second.html',1,chunk_vec=-q,rep_vec=-q)
        data=self.compare()
        self.assertEqual(a,data[0]['chunk_id'])
        self.assertEqual('chunk',data[0]['analysis_kind'])
        self.assertEqual('docs/first.html',data[0]['relative_path_display'])
        self.assertEqual(2,len(data))

    def test_only_complete_representations_are_allowed_to_replace_chunk_winner(self):
        q=self.query.copy()
        self.add('docs/pending.html',1,chunk_vec=q*0.1,rep_vec=q,rep_status='pending')
        self.add('docs/complete.html',1,chunk_vec=-q,rep_vec=q)
        data=self.compare()
        self.assertEqual('docs/complete.html',data[0]['relative_path_display'])
        self.assertEqual('summary_long',data[0]['analysis_kind'])
        self.assertEqual('chunk',next(x for x in data if x['relative_path_display']=='docs/pending.html')['analysis_kind'])

    def test_readonly_sqlite_connection(self):
        for i in range(13):self.add('docs/file-'+str(i)+'.pdf',i)
        self.connection.commit()
        database=self.connection.execute('PRAGMA database_list').fetchone()[2]
        with sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro',uri=True) as ro:
            ro.execute('PRAGMA query_only=ON')
            self.compare(ro)
            with self.assertRaises(sqlite3.OperationalError):
                ro.execute('DELETE FROM content_chunks')

    def test_scoped_CTE_preserves_same_visibility_as_legacy_full_reader(self):
        for i in range(14):
            self.add('docs/scope-'+str(i)+'.pdf',i,chunk_collection='kmd:summary' if i%3==0 else 'kmd:full')
        self.connection.commit()
        class Scoped:
            def __init__(self,db):self.db=db
            def execute(self,sql,params=()):
                if not sql.strip().upper().startswith('SELECT'):
                    raise ValueError('unsafe SQL')
                return self.db.execute("WITH content_chunks AS (SELECT * FROM main.content_chunks WHERE collection_id!='kmd:summary') "+sql,params)
        ranked=self.compare(Scoped(self.connection))
        self.assertTrue(ranked)
        self.assertEqual(set('docs/scope-'+str(i)+'.pdf' for i in range(14) if i%3),set(x['relative_path_display'] for x in ranked))

    def test_nonfinite_lower_rank_challenger_does_not_displace_previous_winner(self):
        # The legacy cosine scorer does not reject NaN blobs; preserve the
        # ORIGINAL strict (score > previous_score) logic, including NaN's
        # unordered comparison. A corrupt vector must not silently become
        # a newly winning source excerpt in the streaming alternative.
        q=self.query.copy()
        first=self.add('docs/nonfinite.html',1,chunk_vec=q,rep_vec=q,
                       rep_status='pending')
        nan=np.full(384,np.nan,dtype='<f4')
        self.add('docs/nonfinite.html',2,chunk_vec=nan,rep_vec=q,
                 rep_status='pending')
        best=self.compare()
        self.assertEqual(first,best[0]['chunk_id'])
        self.assertEqual('chunk',best[0]['analysis_kind'])

    def test_nonfinite_representation_does_not_override_a_finite_chunk_source(self):
        valid=self.query.copy()
        nan=np.full(384,np.nan,dtype='<f4')
        chunk=self.add('docs/nonfinite-representation.html',1,
                       chunk_vec=valid,rep_vec=nan,rep_status='complete')
        best=self.compare()
        self.assertEqual(chunk,best[0]['chunk_id'])
        self.assertEqual('chunk',best[0]['analysis_kind'])

    def test_empty_files_and_empty_vector_rows(self):
        self.assertEqual([],self.compare())


if __name__=='__main__':unittest.main()
