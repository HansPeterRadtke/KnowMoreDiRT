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
        self.assertEqual(2,anchor['original_derived_page_label_number'])
        self.assertFalse(anchor['original_derived_page_label_disagrees_with_physical_page'])
        self.assertFalse(anchor['corrected_physical_page_citation_required'])

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

    def test_mislabeled_derivative_page_correction_preserves_actual_publisher_page(self):
        # Real-world old extractor sometimes has 34 labels for 19 physical
        # pages. A verified unique relocation is source proof, not an excuse
        # to keep citing the wrong derivative page number.
        old={**self.entry,'source_section':'PDF page 19',
             'publisher_pdf_page':{'status':'verified','page':13,
                                  'page_text_sha256':self.page_sha,
                                  'basis':'unique_across_pdf'}}
        anchor=verified_physical_page_candidate({**self.payload,'passages':[old]},self.path)
        self.assertEqual(13,anchor['physical_pdf_page'])
        self.assertEqual(19,anchor['original_derived_page_label_number'])
        self.assertTrue(anchor['original_derived_page_label_disagrees_with_physical_page'])
        self.assertTrue(anchor['corrected_physical_page_citation_required'])
        x={'source':{'original_path':self.path,'original_sha256':'a'*64,'current_derived_sha256':'b'*64},
           'source_map_revision_reverified_after_physical_page_extraction':True,
           'physical_pages':{'original_pdf_sha256':'a'*64,
               'requested_center_physical_page':13,'required_center_excerpt_verified':True,
               'pages':[{'physical_page':13,'physical_page_poppler_sha256':self.page_sha}]}}
        result=bind_original_pdf_context_to_source_evidence(anchor,x)
        self.assertEqual(19,result['original_derived_page_label_number'])
        self.assertEqual(13,result['correct_physical_original_pdf_page'])
        self.assertTrue(result['physical_page_citation_replaces_old_derivative_label_only_if_disagreement'])

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

    def test_explicit_lower_rank_requires_exact_current_revision_and_excerpt(self):
        text='A second source excerpt explicitly contradicts the first selected example.'
        digest=hashlib.sha256(text.encode()).hexdigest()
        second={**self.entry,'source_excerpt':text,'excerpt_sha256':digest,
                'publisher_pdf_page':{'status':'verified','page':3,'page_text_sha256':'8'*64}}
        payload={**self.payload,'passages':[self.entry,second]}
        a=verified_physical_page_candidate(payload,self.path,passage_rank=2,
                expected_identity=('a'*64,'b'*64,digest,3,'8'*64))
        self.assertEqual(2,a['selected_source_excerpt_rank'])
        self.assertIsNone(a['first_verified_source_excerpt_rank'])
        self.assertEqual('explicit_rank_exact_prior_identity',a['source_excerpt_selection'])
        self.assertTrue(a['explicit_prior_passage_identity_matched'])
        self.assertEqual(3,a['physical_pdf_page'])
        self.assertEqual(1,verified_physical_page_candidate(payload,self.path)['selected_source_excerpt_rank'])
        for wrong in (('0'*64,'b'*64,digest,3,'8'*64),('a'*64,'0'*64,digest,3,'8'*64),('a'*64,'b'*64,self.sha,2,self.page_sha)):
            with self.assertRaisesRegex(ValueError,'no longer matches'):
                verified_physical_page_candidate(payload,self.path,passage_rank=2,expected_identity=wrong)
        # Repeated exact text on another page of the same revision is not
        # the reviewed context, even when rank and excerpt digest agree.
        for wrong_page in (('a'*64,'b'*64,digest,2,'8'*64),('a'*64,'b'*64,digest,3,'0'*64)):
            with self.assertRaisesRegex(ValueError,'physical page no longer matches'):
                verified_physical_page_candidate(payload,self.path,passage_rank=2,expected_identity=wrong_page)
        with self.assertRaisesRegex(ValueError,'no longer exists'):
            verified_physical_page_candidate(payload,self.path,passage_rank=3,expected_identity=('a'*64,'b'*64,digest,3,'8'*64))
        payload['passages'][1]={**second,'publisher_pdf_page':{'status':'unverified'}}
        with self.assertRaisesRegex(ValueError,'no fallback'):
            verified_physical_page_candidate(payload,self.path,passage_rank=2,expected_identity=('a'*64,'b'*64,digest,3,'8'*64))

    def test_explicit_rank_validation_is_strict(self):
        for rank,identity in ((1,None),(True,('a'*64,'b'*64,self.sha,2,self.page_sha)),
                              (4,('a'*64,'b'*64,self.sha,2,self.page_sha)),(None,('a'*64,'b'*64,self.sha,2,self.page_sha)),
                              (1,('a'*64,'b'*64,'invalid',2,self.page_sha))):
            with self.subTest(rank=rank,identity=identity):
                with self.assertRaises(ValueError):
                    verified_physical_page_candidate(self.payload,self.path,passage_rank=rank,expected_identity=identity)

    def test_same_page_poppler_digest_must_match_independent_reread(self):
        a=verified_physical_page_candidate(self.payload,self.path)
        good={'source':{'original_path':self.path,'original_sha256':'a'*64,'current_derived_sha256':'b'*64},
              'source_map_revision_reverified_after_physical_page_extraction':True,
              'physical_pages':{
                 'original_pdf_sha256':'a'*64,'requested_center_physical_page':2,
                 'required_center_excerpt_verified':True,
                 'pages':[{'physical_page':2,'physical_page_poppler_sha256':self.page_sha,
                           'extracted_page_text':'Primary source benchmark example.'}]}}
        ok=bind_original_pdf_context_to_source_evidence(a,good)
        self.assertTrue(ok['physical_source_page_sha256_cross_verified_twice'])
        self.assertTrue(ok['source_original_and_derived_revision_match_across_operators'])
        self.assertFalse(ok['cross_store_immutable_generation_pinned'])
        self.assertFalse(not ok['source_context_factual_answer_not_established'])
        wrong={**good,'physical_pages':{**good['physical_pages'],
                    'pages':[{'physical_page':2,'physical_page_poppler_sha256':'0'*64}]}}
        with self.assertRaisesRegex(ValueError,'digest diverges'):
            bind_original_pdf_context_to_source_evidence(a,wrong)

    def test_equal_physical_page_cannot_mask_replaced_original_or_derivative(self):
        anchor=verified_physical_page_candidate(self.payload,self.path)
        self.assertEqual('a'*64,anchor['original_sha256'])
        self.assertEqual('b'*64,anchor['derived_sha256'])
        for original,derived in (('c'*64,'b'*64),('a'*64,'d'*64),('a'*64,None)):
            with self.subTest(original=original,derived=derived):
                changed={'source':{'original_path':self.path,'original_sha256':original,'current_derived_sha256':derived},
                    'source_map_revision_reverified_after_physical_page_extraction':True,
                    'physical_pages':{'original_pdf_sha256':original,'requested_center_physical_page':2,
                        'required_center_excerpt_verified':True,
                        'pages':[{'physical_page':2,'physical_page_poppler_sha256':self.page_sha}]}}
                with self.assertRaisesRegex(ValueError,'revision changed'):
                    bind_original_pdf_context_to_source_evidence(anchor,changed)

    def test_incomplete_anchor_cannot_be_used_as_revision_proof(self):
        anchor=verified_physical_page_candidate(self.payload,self.path)
        del anchor['derived_sha256']
        with self.assertRaisesRegex(ValueError,'missing exact source revision'):
            bind_original_pdf_context_to_source_evidence(anchor,{'source':{},'physical_pages':{}})

    def test_stale_map_missing_excerpt_other_original_never_cite(self):
        a=verified_physical_page_candidate(self.payload,self.path)
        good={'source':{'original_path':self.path,'original_sha256':'a'*64,'current_derived_sha256':'b'*64},
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
