"""Only original-specific current metadata is copied; mutations must rollback in shadow."""
from __future__ import annotations
import hashlib
import json
import re
from file_system_catalog.summary_coverage import revision_identity
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from file_system_catalog.source_page_shadow_cohort import (
    export_cohort_snapshot,rehearse_isolated_sqlite_rollback,table_fingerprint,
    verify_private_shadow_bundle,rehearse_private_file_database_restore,
    compare_historical_shadow_with_current_live,
)


class ShadowSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.original=sqlite3.connect(':memory:');self.addCleanup(self.original.close)
        self.native=sqlite3.connect(':memory:');self.addCleanup(self.native.close)
        self.original.execute('CREATE TABLE originals (source_path TEXT PRIMARY KEY, source_sha256 TEXT,derived_sha256 TEXT,derived_path TEXT,status TEXT)')
        self.native.execute('CREATE TABLE filesystem_entries (id INTEGER PRIMARY KEY,relative_path_display TEXT,entry_type TEXT)')
        self.native.execute('CREATE TABLE content_chunks (chunk_id TEXT PRIMARY KEY,filesystem_entry_id INTEGER NOT NULL,embedding_blob BLOB,FOREIGN KEY(filesystem_entry_id) REFERENCES filesystem_entries(id) ON DELETE CASCADE)')
        self.native.execute('CREATE TABLE content_representations (representation_id TEXT PRIMARY KEY,chunk_id TEXT,embedding_blob BLOB,FOREIGN KEY(chunk_id) REFERENCES content_chunks(chunk_id) ON DELETE CASCADE)')
        self.rows=[]
        for n in (1,2):
            path=f'pdfs/original_{n}.pdf'
            o=hashlib.sha256(f'original{n}'.encode()).hexdigest()
            old=hashlib.sha256(f'old{n}'.encode()).hexdigest()
            new=hashlib.sha256(f'new{n}'.encode()).hexdigest()
            self.rows.append({'original_path':path,'original_sha256':o,'old_derived_sha256':old,'new_staged_derived_sha256':new,'native_original_file_id':100+n})
            self.original.execute('INSERT INTO originals VALUES(?,?,?,?,?)',(path,o,old,path+'.txt','complete'))
            self.native.execute('INSERT INTO filesystem_entries VALUES(?,?,?)',(100+n,path,'file'))
            self.native.execute('INSERT INTO content_chunks VALUES(?,?,?)',(f'chunk{n}',100+n,b'\x00\x01blob'))
            self.native.execute('INSERT INTO content_representations VALUES(?,?,?)',(f'rep{n}',f'chunk{n}',b'\x10\x11'))
        # An unrelated original, chunk and representation must NOT be exported.
        self.native.execute('INSERT INTO filesystem_entries VALUES(?,?,?)',(300,'unrelated.txt','file'))
        self.native.execute('INSERT INTO content_chunks VALUES(?,?,?)',('unrelated',300,b'private-other'))
        self.native.execute('INSERT INTO content_representations VALUES(?,?,?)',('unrelated','unrelated',b'should-not-be-exported'))
        self.original.commit();self.native.commit()
        self.original.execute('PRAGMA query_only=ON');self.native.execute('PRAGMA query_only=ON')
        self.destination=self.root/'private-shadow.sqlite3'

    def test_exact_cohort_export_excludes_all_other_entries(self):
        result=export_cohort_snapshot(original_receipts=self.rows,source_map=self.original,
                       native=self.native,dest=self.destination,expected_count=2)
        self.assertEqual({'originals':2,'filesystem_entries':2,'content_chunks':2,'content_representations':2},result['counts'])
        self.assertFalse(result['multi_file_live_cutover_approved'])
        with sqlite3.connect(self.destination) as clone:
            self.assertEqual(0,len(clone.execute("SELECT chunk_id FROM content_chunks WHERE chunk_id='unrelated'").fetchall()))
            self.assertEqual(b'\x10\x11',clone.execute("SELECT embedding_blob FROM content_representations WHERE representation_id='rep1'").fetchone()[0])
            self.assertEqual('ok',clone.execute('PRAGMA quick_check').fetchone()[0])
            indexes={z[0] for z in clone.execute("SELECT name FROM sqlite_master WHERE type='index'")}
            self.assertIn('shadow_representations_parent_id',indexes)
            self.assertIn('shadow_chunks_parent_id',indexes)
            self.assertEqual([],clone.execute('PRAGMA foreign_key_check').fetchall())
            for table,expected in result['logical_tables'].items():
                self.assertEqual(expected,table_fingerprint(clone,table))

    def test_forced_full_rehearsal_rollback_restores_every_native_row(self):
        exported=export_cohort_snapshot(original_receipts=self.rows,source_map=self.original,
                        native=self.native,dest=self.destination,expected_count=2)
        rehearsal=rehearse_isolated_sqlite_rollback(self.destination,batch_size=2)
        self.assertTrue(rehearsal['tables_identical_after_forced_rollback'])
        self.assertEqual(0,rehearsal['simulated_after_delete_chunk_rows'])
        self.assertEqual(0,rehearsal['simulated_after_delete_representation_rows'])
        with sqlite3.connect(self.destination) as db:
            self.assertEqual(exported['logical_tables'],{t:table_fingerprint(db,t) for t in exported['logical_tables']})

    def test_changed_source_map_revision_does_not_create_valid_shadow(self):
        self.original.execute('PRAGMA query_only=OFF')
        self.original.execute("UPDATE originals SET derived_sha256=? WHERE source_path=?",('0'*64,self.rows[0]['original_path']))
        self.original.commit();self.original.execute('PRAGMA query_only=ON')
        with self.assertRaisesRegex(ValueError,'original revision'):
            export_cohort_snapshot(original_receipts=self.rows,source_map=self.original,
                           native=self.native,dest=self.destination,expected_count=2)

    def test_duplicate_or_unbounded_source_list_rejected(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            export_cohort_snapshot(original_receipts=self.rows[:1]*2,source_map=self.original,
                           native=self.native,dest=self.destination,expected_count=2)
        with self.assertRaisesRegex(ValueError,'invalid original source cohort'):
            export_cohort_snapshot(original_receipts=self.rows[:1],source_map=self.original,
                           native=self.native,dest=self.destination,expected_count=2)
        self.assertFalse(self.destination.exists())

    def test_independent_bundle_verifies_every_archived_byte_and_fails_on_tampering(self):
        snapshot=export_cohort_snapshot(original_receipts=self.rows,source_map=self.original,
                       native=self.native,dest=self.destination,expected_count=2)
        bundle=self.root/'sealed';bundle.mkdir()
        (bundle/'old_derived/pdfs').mkdir(parents=True)
        archived=[]
        (bundle/'old_summaries/progress').mkdir(parents=True)
        for z in self.rows:
            rel='old_derived/'+z['original_path']+'.txt'
            number=z['original_path'].split('_')[-1].split('.')[0]
            data=('old'+number).encode()
            (bundle/rel).write_bytes(data)
            archived.append({'saved_relative_path':rel,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
            name=Path(z['original_path']).name
            prefix=re.sub(r'[^A-Za-z0-9_.-]+','-',name[:64])
            tree=prefix+'.'+revision_identity(z['original_path'],z['original_sha256'],z['old_derived_sha256'])[:24]+'.summary-tree.json'
            summary={'original_path':z['original_path'],'original_sha256':z['original_sha256'],
                     'derived_sha256':z['old_derived_sha256']}
            for location,text in (
                ('old_summaries/'+tree,json.dumps(summary).encode()),
                ('old_summaries/progress/'+tree+'.progress.json',b'safe old progress')):
                (bundle/location).write_bytes(text)
                archived.append({'saved_relative_path':location,
                                 'sha256':hashlib.sha256(text).hexdigest(),'bytes':len(text)})
        sqlite_path=bundle/'original_native_and_source_map_subset.sqlite3'
        sqlite_path.write_bytes(self.destination.read_bytes())
        sql=sqlite_path.read_bytes()
        receipt={'schema':'kmd.private_cohort_original_pdf_rollback_shadowsnapshot.v1','originals':2,
            'not_safe_as_a_future_recovery_point_without_fresh_quiesced_source_revision_check':True,
            'no_live_original_cache_source_map_vector_summary_or_worker_changes':True,
            'archived_file_sha_manifest':archived,
            'private_saved_old_content_total_bytes':sum(z['bytes'] for z in archived),
            'native_shadow_snapshot':{'file':sqlite_path.name,'sha256':hashlib.sha256(sql).hexdigest(),
                'bytes':len(sql),'logical_tables':snapshot['logical_tables']}}
        (bundle/'sealed_read_only_recovery_rehearsal.json').write_text(json.dumps(receipt))
        result=verify_private_shadow_bundle(bundle_root=bundle,expected_count=2)
        self.assertTrue(result['native_original_and_summary_logical_fingerprints_equal'])
        self.assertEqual(6,result['verified_old_source_and_summary_files'])
        self.assertEqual(2,result['summary_revisions_with_tree_and_progress_verified'])
        self.assertTrue(result['all_saved_source_paths_match_exact_old_revision_map'])
        (bundle/archived[0]['saved_relative_path']).write_text('changed old source contents')
        with self.assertRaisesRegex(ValueError,'recovery content SHA/length mismatch'):
            verify_private_shadow_bundle(bundle_root=bundle,expected_count=2)

    def test_historical_snapshot_always_lacks_live_cutover_authority_even_when_current_matches(self):
        export_cohort_snapshot(original_receipts=self.rows,source_map=self.original,
            native=self.native,dest=self.destination,expected_count=2)
        report=compare_historical_shadow_with_current_live(historical_db=self.destination,
             source_map=self.original,native=self.native,expected_originals=2)
        self.assertFalse(report['historical_snapshot_content_is_different_from_current_live'])
        self.assertTrue(report['historic_backup_is_not_a_current_coordinated_rollback_point'])
        self.assertFalse(report['live_cutover_permitted_by_this_audit'])
        self.assertTrue(report['old_native_generated_summary_rows_and_embedding_bytes_equal_current'])
        self.assertEqual(2,report['historical_native_summary_representations'])

    def test_newly_generated_summary_row_marks_historical_recovery_bundle_stale(self):
        export_cohort_snapshot(original_receipts=self.rows,source_map=self.original,
            native=self.native,dest=self.destination,expected_count=2)
        self.native.execute('PRAGMA query_only=OFF')
        self.native.execute('INSERT INTO content_representations VALUES(?,?,?)',
                            ('new-summarization','chunk1',b'new-live-summary-vector'))
        self.native.commit();self.native.execute('PRAGMA query_only=ON')
        report=compare_historical_shadow_with_current_live(historical_db=self.destination,
             source_map=self.original,native=self.native,expected_originals=2)
        self.assertTrue(report['historical_snapshot_content_is_different_from_current_live'])
        self.assertEqual(2,report['historical_native_summary_representations'])
        self.assertEqual(3,report['current_native_summary_representations'])
        self.assertFalse(report['old_native_generated_summary_rows_and_embedding_bytes_equal_current'])
        self.assertFalse(report['live_cutover_permitted_by_this_audit'])

    def test_equal_summary_count_but_changed_vector_bytes_is_detected_as_stale(self):
        export_cohort_snapshot(original_receipts=self.rows,source_map=self.original,
            native=self.native,dest=self.destination,expected_count=2)
        self.native.execute('PRAGMA query_only=OFF')
        self.native.execute('UPDATE content_representations SET embedding_blob=? WHERE representation_id=?',
                            (b'changed-vector-blob','rep2'))
        self.native.commit();self.native.execute('PRAGMA query_only=ON')
        report=compare_historical_shadow_with_current_live(historical_db=self.destination,
             source_map=self.original,native=self.native,expected_originals=2)
        self.assertEqual(2,report['current_native_summary_representations'])
        self.assertTrue(report['historical_snapshot_content_is_different_from_current_live'])
        self.assertFalse(report['old_native_generated_summary_rows_and_embedding_bytes_equal_current'])

    def test_live_source_revision_moved_marks_historical_bundle_stale(self):
        export_cohort_snapshot(original_receipts=self.rows,source_map=self.original,
            native=self.native,dest=self.destination,expected_count=2)
        self.original.execute('PRAGMA query_only=OFF')
        self.original.execute('UPDATE originals SET derived_sha256=? WHERE source_path=?',
                              ('b'*64,self.rows[1]['original_path']))
        self.original.commit();self.original.execute('PRAGMA query_only=ON')
        report=compare_historical_shadow_with_current_live(historical_db=self.destination,
             source_map=self.original,native=self.native,expected_originals=2)
        self.assertEqual(1,report['live_original_revision_mismatches'])
        self.assertTrue(report['historical_snapshot_content_is_different_from_current_live'])

    def test_missing_live_writer_query_only_rejected(self):
        export_cohort_snapshot(original_receipts=self.rows,source_map=self.original,
            native=self.native,dest=self.destination,expected_count=2)
        self.native.execute('PRAGMA query_only=OFF')
        with self.assertRaisesRegex(ValueError,'query-only'):
            compare_historical_shadow_with_current_live(historical_db=self.destination,
                 source_map=self.original,native=self.native,expected_originals=2)

    def test_private_old_and_current_summary_trees_growth_detected_without_database_growth(self):
        export_cohort_snapshot(original_receipts=self.rows,source_map=self.original,
            native=self.native,dest=self.destination,expected_count=2)
        summary=self.root/'summaries';summary.mkdir()
        (summary/'progress').mkdir()
        historic=self.root/'historic';historic.mkdir()
        (historic/'old_summaries/progress').mkdir(parents=True)
        for i,row in enumerate(self.rows):
            from file_system_catalog.summary_coverage import revision_identity
            name=Path(row['original_path']).name
            tree=name+'.'+revision_identity(row['original_path'],row['original_sha256'],row['old_derived_sha256'])[:24]+'.summary-tree.json'
            payload=json.dumps({'original_path':row['original_path'],
                                'original_sha256':row['original_sha256'],
                                'derived_sha256':row['old_derived_sha256']}).encode()
            (summary/tree).write_bytes(payload)
            (summary/'progress'/(tree+'.progress.json')).write_text('old progress')
            if i==0:
                (historic/'old_summaries'/tree).write_bytes(payload)
                (historic/'old_summaries/progress'/(tree+'.progress.json')).write_text('old progress')
        report=compare_historical_shadow_with_current_live(historical_db=self.destination,
           source_map=self.original,native=self.native,expected_originals=2,
           summary_root=summary,archive_root=historic)
        tree=report['summary_tree_and_progress_revision_counts']
        self.assertEqual(1,tree['historical_source_trees'])
        self.assertEqual(2,tree['current_source_trees'])
        self.assertEqual(1,tree['new_current_trees_not_in_historical_archive'])
        self.assertEqual(0,tree['current_trees_changed_since_historical_archive'])
        self.assertTrue(report['historical_snapshot_content_is_different_from_current_live'])

    def test_cross_file_and_sqlite_forced_failures_restore_immutable_archive(self):
        export_cohort_snapshot(original_receipts=self.rows,source_map=self.original,
                              native=self.native,dest=self.destination,expected_count=2)
        private=self.root/'protected-audit';private.mkdir()
        bundle=private/'bundle';bundle.mkdir()
        record=self.rows[0]
        olddir=bundle/'old_derived'/'pdfs';olddir.mkdir(parents=True)
        old=(olddir/'original_1.pdf.txt');old.write_bytes(b'old original text')
        record['old_derived_sha256']=hashlib.sha256(old.read_bytes()).hexdigest()
        # Fix this isolated fixture's native/sourcemap old revision to the newly
        # constructed old original-byte receipt rather than using guessed data.
        self.original.execute('PRAGMA query_only=OFF')
        self.original.execute('UPDATE originals SET derived_sha256=? WHERE source_path=?',
                              (record['old_derived_sha256'],record['original_path']))
        self.original.commit();self.original.execute('PRAGMA query_only=ON')
        target=bundle/'original_native_and_source_map_subset.sqlite3'
        self.destination.unlink()
        export_cohort_snapshot(original_receipts=self.rows,source_map=self.original,
                              native=self.native,dest=target,expected_count=2)
        summary_dir=bundle/'old_summaries';(summary_dir/'progress').mkdir(parents=True)
        prefix=re.sub(r'[^A-Za-z0-9_.-]+','-',Path(record['original_path']).name[:64])
        name=prefix+'.'+revision_identity(record['original_path'],record['original_sha256'],record['old_derived_sha256'])[:24]+'.summary-tree.json'
        (summary_dir/name).write_bytes(b'old publisher-sourced summary tree')
        (summary_dir/'progress'/(name+'.progress.json')).write_bytes(b'old summary progress state')
        staged=private/'corrected-source.txt';staged.write_bytes(b'new corrected physical PDF page text')
        record['new_staged_derived_sha256']=hashlib.sha256(staged.read_bytes()).hexdigest()
        with sqlite3.connect(target) as db:
            db.execute('UPDATE snapshot_revisions SET new_staged_derived_sha256=? WHERE original_path=?',
                      (record['new_staged_derived_sha256'],record['original_path']))
        proof=rehearse_private_file_database_restore(bundle_root=bundle,staged_new_text=staged,
                   original_path=record['original_path'],private_root=private,
                   scenarios_root=private/'fault-scenarios')
        self.assertTrue(proof['only_private_isolated_copies_changed_and_restored'])
        self.assertEqual(2,len(proof['failure_injection_scenarios']))
        self.assertEqual(['fail_after_file_swap','fail_after_database_commit'],
                         [z['injected_failure_point'] for z in proof['failure_injection_scenarios']])
        self.assertEqual(1,proof['original_native_chunk_rows'])
        self.assertEqual(1,proof['original_native_summary_rows'])
        for result in proof['failure_injection_scenarios']:
            self.assertTrue(result['native_original_rows_and_all_summary_rows_logically_restored'])
            self.assertTrue(result['old_source_summary_and_progress_shas_restored'])
            self.assertTrue(result['private_snapshot_sqlite_binary_restored'])
        self.assertEqual(b'old original text',old.read_bytes())

    def test_live_connection_must_be_read_only_and_output_must_be_new(self):
        self.original.execute('PRAGMA query_only=OFF')
        with self.assertRaisesRegex(ValueError,'query-only'):
            export_cohort_snapshot(original_receipts=self.rows,source_map=self.original,
                           native=self.native,dest=self.destination,expected_count=2)
        self.original.execute('PRAGMA query_only=ON')
        self.destination.write_text('not a sqlite database')
        with self.assertRaisesRegex(ValueError,'must be a new'):
            export_cohort_snapshot(original_receipts=self.rows,source_map=self.original,
                           native=self.native,dest=self.destination,expected_count=2)
        self.assertEqual('not a sqlite database',self.destination.read_text())


if __name__=='__main__':unittest.main()
