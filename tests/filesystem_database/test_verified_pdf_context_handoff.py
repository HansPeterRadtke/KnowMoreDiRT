"""Explicit source and physical PDF citation handoff never invents authority."""
from __future__ import annotations
import hashlib
import unittest

from file_system_catalog.verified_pdf_context_handoff import (
    verified_physical_page_candidate,bind_original_pdf_context_to_source_evidence,
)


class VerifiedPhysicalPageHandoffTests(unittest.TestCase):
    def setUp(self):
        self.path='ai-agents/original_publisher_paper.pdf'
        self.excerpt='This is a benchmark question, not an assertion of the real-world birth date.'
        self.sha=hashlib.sha256(self.excerpt.encode()).hexdigest()
        self.page_sha='5'*64
        self.entry={'original_path':self.path,'original_sha256':'a'*64,
                    'derived_sha256':'b'*64,'source_section':'PDF page 2',
                    'source_excerpt':self.excerpt,'excerpt_sha256':self.sha,
                    'publisher_pdf_page':{'status':'verified','page':2,
                       'page_text_sha256':self.page_sha,'basis':'derivative_label_confirmed'}}
        self.payload={'original_path':self.path,'original_sha256':'a'*64,
                      'derived_sha256':'b'*64,'passages':[self.entry]}

    def test_verified_source_exact_physical_page_anchor(self):
        anchor=verified_physical_page_candidate(self.payload,self.path)
        self.assertEqual(2,anchor['physical_pdf_page'])
        self.assertEqual(self.page_sha,anchor['physical_pdf_page_text_sha256'])
        self.assertEqual(self.sha,anchor['source_excerpt_sha256'])
        self.assertIn('not a factual answer',anchor['citation_scope'])

    def test_only_bounded_same_original_pages_no_wrong_substitution(self):
        for changed in (
            {**self.payload,'original_path':'another/paper.pdf'},
            {**self.payload,'passages':[]},
            {**self.payload,'passages':[self.entry]*4},
            {**self.payload,'original_sha256':'other'},
        ):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    verified_physical_page_candidate(changed,self.path)

    def test_excerpt_sha_or_source_revision_mismatch_rejected(self):
        for patch in ({'excerpt_sha256':'0'*64},{'derived_sha256':'0'*64},
                      {'original_path':'other.pdf'},{'source_excerpt':'not the actual literal'}):
            fake={**self.entry,**patch}
            with self.subTest(patch=patch):
                with self.assertRaises(ValueError):
                    verified_physical_page_candidate({**self.payload,'passages':[fake]},self.path)

    def test_ocr_and_fake_physical_page_markers_rejected(self):
        for proof in ({'status':'unverified','reason':'no_text_page_label'},
                      {'status':'verified','page':0,'page_text_sha256':self.page_sha},
                      {'status':'verified','page':2,'page_text_sha256':'broken'}):
            row={**self.entry,'publisher_pdf_page':proof}
            with self.subTest(proof=proof):
                with self.assertRaises(ValueError):
                    verified_physical_page_candidate({**self.payload,'passages':[row]},self.path)

    def test_verified_second_same_original_passage_permitted_not_another_document(self):
        previous={**self.entry,'publisher_pdf_page':{'status':'unverified','reason':'ambiguous_pdf_text'}}
        second={**self.entry,'source_section':'PDF page 3',
                'publisher_pdf_page':{'status':'verified','page':3,
                   'page_text_sha256':'8'*64,'basis':'unique_across_pdf'}}
        anchor=verified_physical_page_candidate({**self.payload,'passages':[previous,second]},self.path)
        self.assertEqual(2,anchor['first_verified_source_excerpt_rank'])
        self.assertEqual(3,anchor['physical_pdf_page'])

    def test_same_page_poppler_digest_must_match_independent_reread(self):
        a=verified_physical_page_candidate(self.payload,self.path)
        good={'source':{'original_path':self.path,'original_sha256':'a'*64},
              'source_map_revision_reverified_after_physical_page_extraction':True,
              'physical_pages':{
                 'original_pdf_sha256':'a'*64,'requested_center_physical_page':2,
                 'required_center_excerpt_verified':True,
                 'pages':[{'physical_page':2,'physical_page_poppler_sha256':self.page_sha,
                           'extracted_page_text':'Primary source benchmark example.'}]}}
        ok=bind_original_pdf_context_to_source_evidence(a,good)
        self.assertTrue(ok['physical_source_page_sha256_cross_verified_twice'])
        self.assertFalse(not ok['source_context_factual_answer_not_established'])
        wrong={**good,'physical_pages':{**good['physical_pages'],
                    'pages':[{'physical_page':2,'physical_page_poppler_sha256':'0'*64}]}}
        with self.assertRaisesRegex(ValueError,'digest diverges'):
            bind_original_pdf_context_to_source_evidence(a,wrong)

    def test_stale_map_missing_excerpt_other_original_never_cite(self):
        a=verified_physical_page_candidate(self.payload,self.path)
        good={'source':{'original_path':self.path,'original_sha256':'a'*64},
              'source_map_revision_reverified_after_physical_page_extraction':True,
              'physical_pages':{'original_pdf_sha256':'a'*64,
                'requested_center_physical_page':2,'required_center_excerpt_verified':True,
                'pages':[{'physical_page':2,'physical_page_poppler_sha256':self.page_sha}]}}
        for changed in ({**good,'source':{'original_path':'other/paper.pdf','original_sha256':'a'*64}},
                        {**good,'source_map_revision_reverified_after_physical_page_extraction':False},
                        {**good,'physical_pages':{**good['physical_pages'],'required_center_excerpt_verified':False}},
                        {**good,'physical_pages':{**good['physical_pages'],'requested_center_physical_page':4}},
                        {**good,'physical_pages':{**good['physical_pages'],'original_pdf_sha256':'f'*64}}):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):bind_original_pdf_context_to_source_evidence(a,changed)

if __name__=='__main__':unittest.main()
