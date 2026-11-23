"""Source-bound provisional original PDF glyph rescue search hard negatives."""
from __future__ import annotations
import hashlib
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from file_system_catalog.alternate_pdf_page_search import (
    exact_skipped_staged_physical_pages,require_current_approved_staged_pdf_source,
    provisional_alternate_query_hits,
)
from file_system_catalog.alternate_pdf_page_text import quality_of_alternate_page

def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()

class ProvisionalOriginalPdfSearchTests(unittest.TestCase):
    def setUp(self):
        t=TemporaryDirectory();self.addCleanup(t.cleanup)
        root=Path(t.name)
        self.original=root/'original';self.old=root/'old';self.stage=root/'stage'
        for p in (self.original,self.old,self.stage):p.mkdir()
        self.rel='scientific/approved-publisher.pdf'
        self.pdf=b'original-approved-publisher-physical-pdf-bytes'
        self.oldtext=b'old source incorrectly attributed page labels'
        self.corrected=(
           'KMD source provenance original synthetic PDF\n\n'
           '[[SOURCE UNIT: PDF page 1]]\nPublisher advances a hypothetical technique with a clear source location.\n'
           '[[SOURCE UNIT: PDF page 2]]\n'+('X\x00'*220)+'\n'
           '[[SOURCE UNIT: PDF page 3]]\nLater paragraph denies that experiment established a real-world law.\n').encode()
        for root,suffix,data in ((self.original,'',self.pdf),(self.old,'.txt',self.oldtext),
                                  (self.stage,'.txt',self.corrected)):
            p=root/(self.rel+suffix);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        self.item={'original_path':self.rel,'original_sha256':sha(self.pdf),
           'old_derived_sha256':sha(self.oldtext),'new_staged_derived_sha256':sha(self.corrected),
           'staged_private_file':str(self.stage/(self.rel+'.txt')),
           'native_original_filedb_id':99,'original_pdf_physical_pages':3,'new_max_staged_page_label':3}
        self.sm=sqlite3.connect(':memory:');self.addCleanup(self.sm.close)
        self.sm.execute('CREATE TABLE originals (source_path TEXT,source_sha256 TEXT,derived_sha256 TEXT,derived_path TEXT,status TEXT)')
        self.sm.execute('INSERT INTO originals VALUES (?,?,?,?,?)',
              (self.rel,sha(self.pdf),sha(self.oldtext),self.rel+'.txt','complete'))
        self.db=sqlite3.connect(':memory:');self.addCleanup(self.db.close)
        self.db.execute('CREATE TABLE filesystem_entries(id INTEGER,entry_type TEXT,relative_path_display TEXT)')
        self.db.execute('INSERT INTO filesystem_entries VALUES (?,?,?)',(99,'file',self.rel))
        self.sm.commit();self.db.commit()
        self.sm.execute('PRAGMA query_only=ON');self.db.execute('PRAGMA query_only=ON')
        self.alternate_text=(
         'Experimental alternate extraction from the official publisher PDF physically identifies '
         'the page where figure panels describe a hypothetical benchmark but do not establish '
         'that an actual law exists. A specific source interpretation remains unknown until a '
         'scientist verifies the original publisher PDF visual and exact physical page. ')*3
        self.alt=quality_of_alternate_page(self.alternate_text.encode())
        self.assertEqual('source_bounded_provisional_machine_text',self.alt['status'])
        self.alt['physical_pdf_page']=2

    def bound(self,item=None):
        return require_current_approved_staged_pdf_source(item or self.item,
              original_root=self.original,derived_root=self.old,stage_root=self.stage,
              completed_acquisitions={self.rel:sha(self.pdf)},
              source_map=self.sm,native=self.db)

    def test_malformed_or_omitted_physical_pages_not_invented(self):
        self.assertEqual([2],exact_skipped_staged_physical_pages(self.corrected.decode(),3,3))
        for bad in (self.corrected.decode().replace('PDF page 3','PDF page 5'),
                    self.corrected.decode().replace('PDF page 3','PDF page 2'),
                    self.corrected.decode().replace('PDF page 1','PDF page 2')):
            with self.subTest(bad=bad[-25:]),self.assertRaises(ValueError):
                exact_skipped_staged_physical_pages(bad,3,3)
        with self.assertRaises(ValueError):
            exact_skipped_staged_physical_pages(self.corrected.decode(),4,3)

    def test_staged_old_original_and_native_revision_all_bound_before_ranking(self):
        result=self.bound()
        self.assertEqual(self.item['native_original_filedb_id'],result['native_id'])
        self.assertEqual(self.item['new_staged_derived_sha256'],result['staged_derived_sha256'])

    def test_changed_old_or_staged_citation_file_refused(self):
        for target in ('old','stage'):
            with self.subTest(target=target):
                root=self.old if target=='old' else self.stage
                p=root/(self.rel+'.txt');saved=p.read_bytes()
                p.write_bytes(b'changed during source bound check')
                with self.assertRaisesRegex(ValueError,'SHA changed'):self.bound()
                p.write_bytes(saved)

    def test_modified_source_map_or_native_original_identity_refused(self):
        self.sm.execute('PRAGMA query_only=OFF')
        self.sm.execute("UPDATE originals SET derived_sha256=? WHERE source_path=?",('0'*64,self.rel))
        self.sm.commit();self.sm.execute('PRAGMA query_only=ON')
        with self.assertRaisesRegex(ValueError,'source-map'):self.bound()
        self.sm.execute('PRAGMA query_only=OFF')
        self.sm.execute('UPDATE originals SET derived_sha256=? WHERE source_path=?',(sha(self.oldtext),self.rel))
        self.sm.commit();self.sm.execute('PRAGMA query_only=ON')
        self.db.execute('PRAGMA query_only=OFF')
        self.db.execute('UPDATE filesystem_entries SET id=100 WHERE relative_path_display=?',(self.rel,))
        self.db.commit();self.db.execute('PRAGMA query_only=ON')
        with self.assertRaisesRegex(ValueError,'FileDB identity'):self.bound()

    def test_missing_approval_or_writeable_native_metadata_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'approved revision'):
            require_current_approved_staged_pdf_source(self.item,
                original_root=self.original,derived_root=self.old,stage_root=self.stage,
                completed_acquisitions={},source_map=self.sm,native=self.db)
        self.db.execute('PRAGMA query_only=OFF')
        with self.assertRaisesRegex(ValueError,'read-only'):self.bound()

    def test_outside_stage_original_path_and_symlink_rejected(self):
        item=dict(self.item,original_path='../private.pdf')
        with self.assertRaises(ValueError):self.bound(item)
        stage=self.stage/(self.rel+'.txt')
        stage.unlink();stage.symlink_to(self.original/self.rel)
        with self.assertRaises(ValueError):self.bound()

    def test_provisional_extracted_hit_does_not_become_confirmed_fact(self):
        r=provisional_alternate_query_hits('figure panels hypothetical benchmark',
          [self.alt],top=3,corroborate_with_poppler=False)
        self.assertEqual(1,r['temporary_alternate_page_candidates_indexed'])
        self.assertEqual(1,len(r['accepted_provisional_ranked_pages']))
        hit=r['accepted_provisional_ranked_pages'][0]
        self.assertEqual(2,hit['physical_pdf_page'])
        self.assertEqual('provisional_alternate_machine_text_only_not_a_verified_quote',hit['status'])
        self.assertFalse(hit['independent_poppler_same_physical_page_passage_corroborated'])
        self.assertFalse(hit['factual_answer_verified'])
        self.assertFalse(hit['publisher_original_visual_layout_correctness_verified'])
        self.assertTrue(r['no_factual_answers_asserted'])

    def test_cross_extractor_corroboration_requires_exact_same_physical_page(self):
        calls=[]
        def fake(_pdf,_sha,passages):
            calls.append(passages)
            self.assertEqual('PDF page 2',passages[0]['source_section'])
            return [{'status':'verified','page':2,'basis':'derivative_label_confirmed',
                     'page_text_sha256':'b'*64}]
        r=provisional_alternate_query_hits('figure panels hypothetical benchmark',
          [self.alt],top=1,corroborate_with_poppler=True,
          original_pdf=self.original/self.rel,expected_original_sha256=sha(self.pdf),
          independent_verifier=fake)
        hit=r['accepted_provisional_ranked_pages'][0]
        self.assertEqual('same_page_second_extractor_exact_phrase_corroborated',hit['status'])
        self.assertTrue(hit['independent_poppler_same_physical_page_passage_corroborated'])
        self.assertFalse(hit['factual_answer_verified'])
        self.assertEqual(1,len(calls))

    def test_unique_match_elsewhere_is_not_corroboration_of_original_page(self):
        def elsewhere(*_):return [{'status':'verified','page':1,'basis':'unique_across_pdf'}]
        x=provisional_alternate_query_hits('figure panels hypothetical benchmark',
          [self.alt],corroborate_with_poppler=True,
          original_pdf=self.original/self.rel,expected_original_sha256=sha(self.pdf),
          independent_verifier=elsewhere)
        self.assertFalse(x['accepted_provisional_ranked_pages'][0]['independent_poppler_same_physical_page_passage_corroborated'])

    def test_incomplete_or_tampered_alternate_page_candidate_not_indexed(self):
        bad=dict(self.alt,display_text_sha256='c'*64)
        with self.assertRaisesRegex(ValueError,'altered or untrusted'):
            provisional_alternate_query_hits('figure panels',
                [bad],corroborate_with_poppler=False)
        bad=dict(self.alt,status='unavailable',text='plausible fake')
        x=provisional_alternate_query_hits('figure panels',
                [bad],corroborate_with_poppler=False)
        self.assertEqual(0,x['temporary_alternate_page_candidates_indexed'])
        self.assertEqual(1,x['excluded_unusable_alternate_pages'])
        self.assertEqual([],x['accepted_provisional_ranked_pages'])

    def test_low_query_coverage_returns_no_citation(self):
        x=provisional_alternate_query_hits('figure extraterrestrial nebula',
               [self.alt],corroborate_with_poppler=False)
        self.assertEqual([],x['accepted_provisional_ranked_pages'])
        self.assertEqual(1,x['weak_word_coverage_candidates_excluded'])

if __name__=='__main__':unittest.main()
