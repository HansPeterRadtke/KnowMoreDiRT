"""Source revision bound no-action migration-preflight regression tests."""
from __future__ import annotations
import hashlib
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from file_system_catalog.source_page_cutover_readiness import inspect_staged_cutover


def sha(data:bytes):return hashlib.sha256(data).hexdigest()


class SourcePageReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.original=self.root/'orig';self.der=self.root/'old';self.stage=self.root/'private';
        for p in (self.original,self.der,self.stage):p.mkdir()
        self.sm=sqlite3.connect(':memory:');self.addCleanup(self.sm.close)
        self.native=sqlite3.connect(':memory:');self.addCleanup(self.native.close)
        self.sm.execute('CREATE TABLE originals (source_path TEXT,source_sha256 TEXT,derived_sha256 TEXT,derived_path TEXT,status TEXT)')
        self.native.execute('CREATE TABLE filesystem_entries (id INTEGER,relative_path_display TEXT,entry_type TEXT)')
        self.native.execute('CREATE TABLE content_chunks (chunk_id TEXT, filesystem_entry_id INTEGER, embedding_blob BLOB)')
        self.native.execute('CREATE TABLE content_representations (representation_id TEXT, chunk_id TEXT,embedding_blob BLOB)')
        self.records=[];self.receipts={}
        for index in range(2):
            relative=f'papers/original_{index}.pdf';old=f'Legacy labelled PDF {index}'.encode();new=f'Repaired original physical page {index}'.encode();original=b'%PDF-approved-example-'+bytes([index])
            for root,data,suffix in ((self.original,original,''),(self.der,old,'.txt'),(self.stage,new,'.txt')):
                p=root/(relative+suffix);p.parent.mkdir(exist_ok=True);p.write_bytes(data)
            o=sha(original);d=sha(old);n=sha(new);self.receipts[relative]=o
            self.sm.execute('INSERT INTO originals VALUES (?,?,?,?,?)',(relative,o,d,relative+'.txt','complete'))
            self.native.execute('INSERT INTO filesystem_entries VALUES (?,?,?)',(100+index,relative,'file'))
            self.native.execute('INSERT INTO content_chunks VALUES (?,?,?)',(f'chunk{index}',100+index,b'embed'))
            if index==1:
                self.native.execute('INSERT INTO content_representations VALUES (?,?,?)',('summary1','chunk1',b'rep'))
            self.records.append({'original_path':relative,'original_sha256':o,'old_derived_sha256':d,
               'new_staged_derived_sha256':n,'staged_private_file':str(self.stage/(relative+'.txt')),
               'native_original_filedb_id':100+index})
        self.sm.commit();self.native.commit()
        self.sm.execute('PRAGMA query_only=ON');self.native.execute('PRAGMA query_only=ON')

    def evaluate(self,service='active'):
        return inspect_staged_cutover(self.records,source_map=self.sm,native=self.native,
               original_root=self.original,derived_root=self.der,stage_root=self.stage,
               completed_receipts=self.receipts,
               service_states={'kmd-summary-worker-coder':service,'kmd-summary-worker-q4':service},
               expected_count=2)

    def test_actual_native_dependents_listed_but_cutover_never_permitted(self):
        r=self.evaluate()
        self.assertFalse(r['safe_to_cutover']);self.assertTrue(r['workers_not_quiesced'])
        self.assertEqual(2,r['originals_reverified'])
        self.assertEqual(2,r['native_dependencies']['original_vectors'])
        self.assertEqual(1,r['native_dependencies']['summary_nodes'])
        self.assertEqual(1,r['native_dependencies']['affected_sources_with_summary_nodes'])
        self.assertTrue(r['no_source_cache_source_map_filedb_vector_or_summary_writes'])
        self.assertFalse(self.evaluate('inactive')['safe_to_cutover'])
        self.assertFalse(self.evaluate('inactive')['workers_not_quiesced'])

    def test_missing_service_or_unbounded_cohort_refused(self):
        with self.assertRaisesRegex(ValueError,'missing actually observed'):inspect_staged_cutover(
           self.records,source_map=self.sm,native=self.native,original_root=self.original,
           derived_root=self.der,stage_root=self.stage,completed_receipts=self.receipts,
           service_states={},expected_count=2)
        with self.assertRaisesRegex(ValueError,'incomplete or unbounded'):inspect_staged_cutover(
           self.records[:1],source_map=self.sm,native=self.native,original_root=self.original,
           derived_root=self.der,stage_root=self.stage,completed_receipts=self.receipts,
           service_states={'kmd-summary-worker-coder':'active','kmd-summary-worker-q4':'active'},expected_count=2)

    def test_duplicate_or_changed_revisions_fail_closed(self):
        self.records[1]['original_path']=self.records[0]['original_path']
        with self.assertRaisesRegex(ValueError,'ambiguous'):self.evaluate()
        self.records[1]['original_path']='papers/original_1.pdf'
        old=self.stage/'papers/original_0.pdf.txt';old.write_text('corrupted')
        with self.assertRaisesRegex(ValueError,'SHA changed'):self.evaluate()

    def test_stale_live_source_map_or_approved_receipt_refused(self):
        self.sm.execute('PRAGMA query_only=OFF')
        self.sm.execute('UPDATE originals SET derived_sha256=? WHERE source_path=?',('0'*64,'papers/original_0.pdf'))
        self.sm.commit();self.sm.execute('PRAGMA query_only=ON')
        with self.assertRaisesRegex(ValueError,'source-map no longer'):self.evaluate()
        self.sm.execute('PRAGMA query_only=OFF')
        self.sm.execute('UPDATE originals SET derived_sha256=? WHERE source_path=?',(self.records[0]['old_derived_sha256'],'papers/original_0.pdf'))
        self.sm.commit();self.sm.execute('PRAGMA query_only=ON')
        self.receipts['papers/original_0.pdf']='0'*64
        with self.assertRaisesRegex(ValueError,'approved acquisition'):self.evaluate()

    def test_no_metadata_read_write_connection_allowed(self):
        self.native.execute('PRAGMA query_only=OFF')
        with self.assertRaisesRegex(ValueError,'read-only'):self.evaluate()


if __name__=='__main__':unittest.main()
