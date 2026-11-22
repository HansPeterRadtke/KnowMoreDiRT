"""Full-source original SHA / page metadata and dependent vector erasure: private clones."""
from __future__ import annotations
import hashlib,json,re,sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from file_system_catalog.source_page_forward_rehearsal import (
    private_forward_generation,private_restore_generation,InjectedPrivateForwardFailure)
from file_system_catalog.source_page_shadow_cohort import (table_fingerprint,_TABLES)
from file_system_catalog.summary_coverage import revision_identity


def sha(blob:bytes):return hashlib.sha256(blob).hexdigest()


class PrivateFullForwardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.stage=self.root/'stage';self.stage.mkdir()
        self.bundle=self.root/'old-history';self.bundle.mkdir()
        self.sqlite=self.bundle/'original_native_and_source_map_subset.sqlite3'
        self.old_derived=self.bundle/'old_derived';self.old_derived.mkdir()
        self.old_summary=self.bundle/'old_summaries';self.old_summary.mkdir()
        (self.old_summary/'progress').mkdir()
        self.records=[]
        self.manifest=[]
        with sqlite3.connect(self.sqlite) as db:
            db.execute('PRAGMA foreign_keys=ON')
            db.execute('CREATE TABLE originals (source_path TEXT PRIMARY KEY,source_sha256 TEXT,derived_path TEXT,derived_sha256 TEXT,original_kind TEXT,original_url TEXT,original_title TEXT,original_bytes INTEGER,extracted_pages INTEGER,nonempty_pages INTEGER,text_characters INTEGER,status TEXT,error TEXT,updated_utc TEXT)')
            db.execute('CREATE TABLE filesystem_entries (id INTEGER PRIMARY KEY,relative_path_display TEXT,entry_type TEXT)')
            db.execute('CREATE TABLE content_chunks (chunk_id TEXT PRIMARY KEY,filesystem_entry_id INTEGER NOT NULL, embedding_blob BLOB, FOREIGN KEY(filesystem_entry_id) REFERENCES filesystem_entries(id) ON DELETE CASCADE)')
            db.execute('CREATE TABLE content_representations (representation_id TEXT PRIMARY KEY, chunk_id TEXT, embedding_blob BLOB, FOREIGN KEY(chunk_id) REFERENCES content_chunks(chunk_id) ON DELETE CASCADE)')
            db.execute('CREATE TABLE snapshot_revisions (original_path TEXT PRIMARY KEY,original_sha256 TEXT,old_derived_sha256 TEXT,new_staged_derived_sha256 TEXT,native_file_id INTEGER UNIQUE)')
            for n in (1,2):
                path=f'publishers/paper-{n}.pdf'
                source=('publisher-pdf-original-'+str(n)).encode()
                old=('previous extracted source PDF '+str(n)).encode()
                original_sha,old_sha=sha(source),sha(old)
                generated='KMD source provenance (the original PDF/HTML/ZIP/XLSX, not a new publisher book):'
                metadata=json.dumps({'source_path':path,'source_sha256':original_sha,'kind':'pdf'},sort_keys=True)
                text=(generated+'\n'+metadata+'\n\n'+'\n\n[[SOURCE UNIT: PDF page 1]]\n'
                      'Evidence for audited source retrieval with original physical publisher pages.\n'
                      '\n\n[[SOURCE UNIT: PDF page 2]]\n'
                      'A second original publisher page describes private full forward rehearsal.\n')
                self.assertTrue(text.startswith(generated))
                stage=self.stage/(path+'.txt');stage.parent.mkdir(parents=True,exist_ok=True)
                stage.write_text(text)
                previous=self.old_derived/(path+'.txt');previous.parent.mkdir(parents=True,exist_ok=True)
                previous.write_bytes(old)
                record={'original_path':path,'original_sha256':original_sha,
                        'old_derived_sha256':old_sha,'new_staged_derived_sha256':sha(text.encode()),
                        'staged_private_file':str(stage),'native_original_filedb_id':10+n,
                        'original_pdf_physical_pages':2,'new_max_staged_page_label':2,
                        'new_staged_original_page_unit_count':2}
                self.records.append(record)
                db.execute('INSERT INTO snapshot_revisions VALUES(?,?,?,?,?)',
                            (path,original_sha,old_sha,record['new_staged_derived_sha256'],10+n))
                db.execute('INSERT INTO originals VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                            (path,original_sha,path+'.txt',old_sha,'pdf','https://publisher.example/paper','Paper',len(source),
                             12,12,60,'complete',None,'2026-11-21T00:00:00Z'))
                db.execute('INSERT INTO filesystem_entries VALUES(?,?,?)',(10+n,path,'file'))
                db.execute('INSERT INTO content_chunks VALUES(?,?,?)',(f'chunk{n}',10+n,b'original-embedding-data'))
                db.execute('INSERT INTO content_representations VALUES(?,?,?)',(f'rep{n}',f'chunk{n}',b'old-recursive-summary-vector'))
                name=re.sub(r'[^A-Za-z0-9_.-]+','-',Path(path).name[:64])+'.'+revision_identity(path,original_sha,old_sha)[:24]+'.summary-tree.json'
                tree=json.dumps({'original_path':path,'original_sha256':original_sha,'derived_sha256':old_sha}).encode()
                self._archive('old_derived/'+path+'.txt',old)
                self._archive('old_summaries/'+name,tree)
                self._archive('old_summaries/progress/'+name+'.progress.json',b'progress for '+source)
            db.commit()
            self.base_digests={table:table_fingerprint(db,table) for table in _TABLES}
        self.base_sha=sha(self.sqlite.read_bytes())
        native={'file':self.sqlite.name,'sha256':self.base_sha,
                'bytes':self.sqlite.stat().st_size,'logical_tables':self.base_digests}
        self.sealed={'schema':'kmd.private_cohort_original_pdf_rollback_shadowsnapshot.v1',
            'originals':2,
            'not_safe_as_a_future_recovery_point_without_fresh_quiesced_source_revision_check':True,
            'no_live_original_cache_source_map_vector_summary_or_worker_changes':True,
            'archived_file_sha_manifest':self.manifest,
            'private_saved_old_content_total_bytes':sum(row['bytes'] for row in self.manifest),
            'native_shadow_snapshot':native}
        (self.bundle/'sealed_read_only_recovery_rehearsal.json').write_text(json.dumps(self.sealed))
        self.output=self.root/'only-new-scratch-candidate'

    def _archive(self,relative,blob):
        file=self.bundle/relative;file.parent.mkdir(parents=True,exist_ok=True)
        if not file.exists():file.write_bytes(blob)
        self.manifest.append({'saved_relative_path':relative,'sha256':sha(blob),'bytes':len(blob)})

    def do_forward(self,**kwargs):
        return private_forward_generation(historical_db=self.sqlite,records=self.records,
             stage_root=self.stage,private_root=self.root,output=self.output,
             expected_count=2,**kwargs)

    def test_private_forward_revisions_and_page_count_correct_without_stale_vectors(self):
        output=self.do_forward()
        self.assertEqual(2,output['originals_simulated'])
        self.assertEqual(2,output['old_content_chunks_invalidated_in_private'])
        self.assertEqual(2,output['old_summary_representations_invalidated_in_private'])
        self.assertEqual('staged_needs_reindex',output['new_source_status_requires_reindex_before_native_search'])
        self.assertFalse(output['safe_to_cutover_live'])
        with sqlite3.connect(self.output/'private_after_transition_not_search_ready.sqlite3') as db:
            self.assertEqual((0,0),(db.execute('select count(*) from content_chunks').fetchone()[0],
                                   db.execute('select count(*) from content_representations').fetchone()[0]))
            self.assertEqual([],db.execute('pragma foreign_key_check').fetchall())
            for rec in self.records:
                row=db.execute('SELECT source_sha256,derived_sha256,extracted_pages,nonempty_pages,text_characters,status FROM originals WHERE source_path=?',(rec['original_path'],)).fetchone()
                copied=(self.output/'corrected_source_text'/(rec['original_path']+'.txt'))
                from file_system_catalog.source_revision_audit import _verified_derived_body_count
                expected_chars=_verified_derived_body_count(Path(rec['staged_private_file']).read_text(),
                                  original_path=rec['original_path'],original_sha=rec['original_sha256'],source_kind='pdf')
                self.assertEqual((rec['original_sha256'],rec['new_staged_derived_sha256'],2,2,
                                  expected_chars,'staged_needs_reindex'),row)
                self.assertEqual(rec['new_staged_derived_sha256'],sha(copied.read_bytes()))
        self.assertEqual(self.base_sha,sha(self.sqlite.read_bytes()))
        self.assertTrue(all((self.bundle/'old_derived'/(r['original_path']+'.txt')).exists() for r in self.records))

    def test_injected_partial_file_copy_cannot_mutate_original_backup(self):
        with self.assertRaisesRegex(InjectedPrivateForwardFailure,'copy 1 of 2'):
            self.do_forward(fail_after_copies=1)
        self.assertFalse((self.output/'private_after_transition_not_search_ready.sqlite3').exists())
        self.assertEqual(1,len(list((self.output/'corrected_source_text').rglob('*.txt'))))
        self.assertEqual(self.base_sha,sha(self.sqlite.read_bytes()))

    def test_commit_failure_restores_complete_old_private_files_and_all_vectors(self):
        with self.assertRaisesRegex(InjectedPrivateForwardFailure,'database commit'):
            self.do_forward(fail_after_commit=True)
        with sqlite3.connect(self.output/'private_after_transition_not_search_ready.sqlite3') as db:
            self.assertEqual(0,db.execute('SELECT COUNT(*) FROM content_representations').fetchone()[0])
        recovered=self.root/'recovered-new-private-copy'
        result=private_restore_generation(bundle_root=self.bundle,private_root=self.root,
                                           output=recovered,expected_count=2)
        self.assertEqual(6,result['sealed_historical_recovery_files_reproduced'])
        self.assertEqual(2,result['native_summary_representations_restored'])
        self.assertEqual(2,result['native_original_chunks_restored'])
        self.assertFalse(result['safe_to_replace_live_source_with_this_historical_bundle'])
        self.assertEqual(self.base_sha,sha((recovered/self.sqlite.name).read_bytes()))
        self.assertEqual(self.base_sha,sha(self.sqlite.read_bytes()))
        for row in self.manifest:
            self.assertEqual(row['sha256'],sha((recovered/row['saved_relative_path']).read_bytes()))

    def test_wrong_staged_sha_fails_before_any_private_generation_folder_exists(self):
        item=self.records[0];p=Path(item['staged_private_file']);p.write_bytes(p.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'SHA'):
            self.do_forward()
        self.assertFalse(self.output.exists())

    def test_wrong_physical_page_label_fails_before_creating_recovery_folder(self):
        item=self.records[0];source=Path(item['staged_private_file']);raw=source.read_text().replace('PDF page 2','PDF page 3')
        source.write_text(raw)
        item['new_staged_derived_sha256']=sha(raw.encode())
        with sqlite3.connect(self.sqlite) as db:
            db.execute('UPDATE snapshot_revisions SET new_staged_derived_sha256=? WHERE original_path=?',
                       (item['new_staged_derived_sha256'],item['original_path']))
        with self.assertRaisesRegex(ValueError,'page coverage'):
            self.do_forward()
        self.assertFalse(self.output.exists())

    def test_duplicate_source_is_not_forwarded(self):
        self.records=[self.records[0],self.records[0]]
        with self.assertRaisesRegex(ValueError,'duplicate'):
            self.do_forward()
        self.assertFalse(self.output.exists())

    def test_output_nested_in_sealed_recovery_bundle_refused(self):
        protected=self.bundle/'dangerous-child'
        with self.assertRaisesRegex(ValueError,'outside sealed baseline'):
            private_forward_generation(historical_db=self.sqlite,records=self.records,
                 stage_root=self.stage,private_root=self.root,output=protected,expected_count=2)
        self.assertFalse(protected.exists())
        with self.assertRaisesRegex(ValueError,'outside sealed baseline'):
            private_restore_generation(bundle_root=self.bundle,private_root=self.root,
                                       output=protected,expected_count=2)
        self.assertFalse(protected.exists())

    def test_output_nested_in_corrected_staging_tree_refused(self):
        dangerous=self.stage/'child-folder'
        with self.assertRaisesRegex(ValueError,'inside the staged'):
            private_forward_generation(historical_db=self.sqlite,records=self.records,
                 stage_root=self.stage,private_root=self.root,output=dangerous,expected_count=2)
        self.assertFalse(dangerous.exists())

    def test_outside_private_output_refused(self):
        out=Path(self.tmp.name+'-elsewhere')
        with self.assertRaisesRegex(ValueError,'exclusive private output'):
            private_forward_generation(historical_db=self.sqlite,records=self.records,
                 stage_root=self.stage,private_root=self.root,output=out,
                 expected_count=2)
        self.assertFalse(out.exists())

if __name__=='__main__':unittest.main()
