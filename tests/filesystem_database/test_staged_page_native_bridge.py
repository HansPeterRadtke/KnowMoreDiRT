"""Unified native-selected PDF private-page evidence is opt-in and fail closed."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from file_system_catalog.staged_page_native_bridge import (
  pinned_private_corrected_sources,verified_native_selected_private_page_overlay,
)


def digest(data):return hashlib.sha256(data).hexdigest()


class NativeSelectedPrivatePageBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.original=self.root/'original';self.old=self.root/'old';self.stage=self.root/'stage'
        for p in (self.original,self.old,self.stage):p.mkdir()
        self.path='ai-agents/approved-original.pdf'
        self.publisher=b'PDF original fixture for source selection only'
        self.old_text=b'prior wrongly labeled physical PDF pages'
        self.new_text=('KMD source provenance publisher original fixture\n'
              '[[SOURCE UNIT: PDF page 1]]\nSpeech synthesis needs pronunciation controls and rhythmic timing.\n'
              '[[SOURCE UNIT: PDF page 2]]\nPublisher verified physical PDF page provenance for search citations and reproducible scholarly evidence.\n').encode()
        for folder,suffix,data in ((self.original,'',self.publisher),(self.old,'.txt',self.old_text),
                                   (self.stage,'.txt',self.new_text)):
            p=folder/(self.path+suffix);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        self.record={'original_path':self.path,'original_sha256':digest(self.publisher),
           'old_derived_sha256':digest(self.old_text),'new_staged_derived_sha256':digest(self.new_text),
           'staged_private_file':str(self.stage/(self.path+'.txt')),
           'native_original_filedb_id':55,'original_pdf_physical_pages':2,'new_max_staged_page_label':2}
        self.doc={'original_path':self.path,'status':'no_verified_excerpt',
                  'original_sha256':self.record['original_sha256'],
                  'derived_sha256':self.record['old_derived_sha256'],
                  'filesystem_entry_id':55}
        self.map=sqlite3.connect(':memory:');self.addCleanup(self.map.close)
        self.map.execute('CREATE TABLE originals (source_path TEXT,source_sha256 TEXT,derived_sha256 TEXT,derived_path TEXT,status TEXT)')
        self.map.execute('INSERT INTO originals VALUES (?,?,?,?,?)',
           (self.path,digest(self.publisher),digest(self.old_text),self.path+'.txt','complete'))
        self.native=sqlite3.connect(':memory:');self.addCleanup(self.native.close)
        self.native.execute('CREATE TABLE filesystem_entries (id INTEGER,entry_type TEXT,relative_path_display TEXT)')
        self.native.execute('INSERT INTO filesystem_entries VALUES (?,?,?)',(55,'file',self.path))
        self.map.commit();self.native.commit()
        self.map.execute('PRAGMA query_only=ON');self.native.execute('PRAGMA query_only=ON')

    def search(self,documents=None,verifier=None,**kwargs):
        return verified_native_selected_private_page_overlay(
            documents=documents if documents is not None else [self.doc],
            query='publisher physical PDF page provenance citations',
            staged_records={self.path:self.record},stage_root=self.stage,
            original_root=self.original,derived_root=self.old,
            completed_receipts={self.path:digest(self.publisher)},
            source_map=self.map,native=self.native,
            verifier=verifier or (lambda *_:[{'status':'verified','page':2,
                'basis':'derivative_label_confirmed','page_text_sha256':'a'*64}]),**kwargs)

    def test_native_selected_original_physical_page_proof_survives_unverified_old_page(self):
        result=self.search()
        self.assertEqual(1,result['staged_originals_in_native_selection'])
        self.assertEqual(1,result['publisher_verified_corrected_passages'])
        doc=result['source_originals_with_private_page_evidence'][0]
        self.assertEqual('publisher_original_physical_pages_verified',doc['status'])
        proof=doc['corrected_private_pdf_excerpts'][0]
        self.assertEqual(2,proof['physical_pdf_page'])
        self.assertEqual(self.path,proof['original_path'])
        self.assertFalse(proof['factual_answer_verified'])
        self.assertTrue(doc['no_claim_of_native_reindexing'])
        self.assertEqual(self.old_text,(self.old/(self.path+'.txt')).read_bytes())

    def test_old_revision_or_different_native_id_refused(self):
        for key,bad in (('derived_sha256','0'*64),('filesystem_entry_id',5000)):
            broken=dict(self.doc,**{key:bad})
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'identity diverge'):
                self.search(documents=[broken])

    def test_known_pdf_without_native_revision_is_not_silently_verified(self):
        result=self.search(documents=[{'original_path':self.path,'status':'source_unavailable'}])
        row=result['source_originals_with_private_page_evidence'][0]
        self.assertEqual('not_checked_native_revision_unavailable',row['status'])
        self.assertEqual([],row['corrected_private_pdf_excerpts'])

    def test_unapproved_or_unrelated_pdf_is_not_added_to_native_search(self):
        result=self.search(documents=[{'original_path':'some/unrelated.pdf','status':'verified'}])
        self.assertEqual(0,result['staged_originals_in_native_selection'])

    def test_false_page_verification_refused(self):
        result=self.search(verifier=lambda *_:[{'status':'unverified','reason':'excerpt_absent'}])
        row=result['source_originals_with_private_page_evidence'][0]
        self.assertEqual('no_independently_verified_corrected_pdf_page',row['status'])
        self.assertEqual([],row['corrected_private_pdf_excerpts'])

    def test_query_only_live_connections_are_mandatory(self):
        self.native.execute('PRAGMA query_only=OFF')
        with self.assertRaisesRegex(ValueError,'read-only'):
            self.search()

    def test_preflight_requires_complete_approved_cohort_and_no_duplicate(self):
        sample=dict(self.record)
        records=[]
        for i in range(327):records.append({**sample,'original_path':f'originals/book{i}.pdf'})
        receipt={'schema':'kmd.all_327_approved_original_pdf_private_stage_revision_and_physical_page_preflight.v1',
            'source_pdf_originals_verified_and_staged':327,
            'all_original_old_derived_and_new_staged_full_sha_reverified_after_staging':True,
            'current_source_map_old_derivative_revisions_and_native_ids_reverified':True,
            'status':'ALL_PRIVATE_STAGE_CANDIDATES_VERIFIED_NOT_PUBLISHED_TO_LIVE_SEARCH',
            'original_sources_and_corrected_private_derivative_candidates':records}
        path=self.stage/'approved.json';path.write_text(json.dumps(receipt))
        self.assertEqual(327,len(pinned_private_corrected_sources(stage_preflight=path,private_root=self.stage)))
        receipt['original_sources_and_corrected_private_derivative_candidates'][-1]=records[0]
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            pinned_private_corrected_sources(stage_preflight=path,private_root=self.stage)
        receipt['source_pdf_originals_verified_and_staged']=326
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError,'unapproved'):
            pinned_private_corrected_sources(stage_preflight=path,private_root=self.stage)


if __name__=='__main__':unittest.main()
