"""The exact source-evidence to actual physical page operator fails closed."""
from __future__ import annotations
import importlib.util
import hashlib
from pathlib import Path
import json
import unittest
from unittest import mock
from types import SimpleNamespace


def module():
    p=Path(__file__).resolve().parents[2]/'scripts/raspi_source_evidence_pdf_neighbors.py'
    spec=importlib.util.spec_from_file_location('test_only_source_pdf_neighbor_operator',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


class SourceEvidenceToPdfNeighborsTests(unittest.TestCase):
    def setUp(self):
        self.mod=module()
        self.path='ai-agents/2026-original-study.pdf'
        self.text='A study displays a question inside its benchmark example, not official biographical evidence.'
        self.sha=hashlib.sha256(self.text.encode()).hexdigest()
        self.page_sha='1'*64
        self.payload={'original_path':self.path,'original_sha256':'a'*64,'derived_sha256':'b'*64,
          'passages':[{'original_path':self.path,'original_sha256':'a'*64,'derived_sha256':'b'*64,
          'source_section':'PDF page 2','source_excerpt':self.text,'excerpt_sha256':self.sha,
          'publisher_pdf_page':{'status':'verified','page':2,'page_text_sha256':self.page_sha,'basis':'derivative_label_confirmed'}}]}
        self.context={'source':{'original_path':self.path,'original_sha256':'a'*64,'current_derived_sha256':'b'*64},
                      'source_map_revision_reverified_after_physical_page_extraction':True,
                      'physical_pages':{'original_pdf_sha256':'a'*64,
                           'requested_center_physical_page':2,'required_center_excerpt_verified':True,
                           'pages':[{'physical_page':2,'physical_page_poppler_sha256':self.page_sha}]}}

    def call(self,**patches):
        payload=patches.get('payload',self.payload)
        context=patches.get('context',self.context)
        proc=SimpleNamespace(returncode=patches.get('exit_code',0),
                stdout=json.dumps(payload),stderr='')
        with mock.patch.object(self.mod.subprocess,'run',return_value=proc) as source, \
             mock.patch.object(self.mod,'original_pdf_context',return_value=context) as phys:
            out=self.mod.context_from_current_source_evidence('original study benchmark question',self.path)
        return out,source,phys

    def test_exact_original_question_to_verified_page_uses_existing_source_operator_only(self):
        got,source,phys=self.call()
        args=source.call_args.args[0]
        self.assertIn('source-evidence',args)
        self.assertIn('--original-path',args)
        self.assertIn(self.path,args)
        self.assertIn('500',args)
        self.assertNotIn('restart',args)
        phys.assert_called_once_with(self.path,2,neighbor_radius=1,required_center_excerpt=self.text,include_center_reading_order=False)
        self.assertTrue(got['physical_source_page_sha256_cross_verified_twice'])
        self.assertTrue(got['source_context_factual_answer_not_established'])
        self.assertEqual(2,got['physical_pages']['requested_center_physical_page'])

    def test_optional_reading_order_only_applies_after_verified_source_anchor(self):
        alternate={'physical_page':2,'status':'same_physical_page_alternate_reading_order_text',
                   'extraction_mode':'poppler_default_reading_order_utf8_not_layout',
                   'exact_layout_page_poppler_sha256':self.page_sha,
                   'reading_order_poppler_sha256':self.sha,'extracted_text_sha256':self.sha,
                   'extracted_page_text':self.text,'text_characters':len(self.text),
                   'complete_extractor_output_not_truncated':True,
                   'does_not_replace_verified_layout_excerpt_or_offsets':True,
                   'visual_reading_order_or_factual_claim_truth_verified':False,
                   'required_layout_excerpt_also_contiguous_in_reading_order':True}
        self.context['physical_pages']['optional_center_reading_order_projection']=alternate
        proc=SimpleNamespace(returncode=0,stdout=json.dumps(self.payload),stderr='')
        with mock.patch.object(self.mod.subprocess,'run',return_value=proc), \
             mock.patch.object(self.mod,'original_pdf_context',return_value=self.context) as phys:
            got=self.mod.context_from_current_source_evidence('original study benchmark question',self.path,
                      neighbor_radius=0,include_center_reading_order=True)
        phys.assert_called_once_with(self.path,2,neighbor_radius=0,required_center_excerpt=self.text,
                    include_center_reading_order=True)
        self.assertEqual(alternate,got['physical_pages']['optional_center_reading_order_projection'])
        self.assertTrue(got['source_original_and_derived_revision_match_across_operators'])

    def test_rejects_wrong_page_or_corrupt_alternate_context(self):
        good={'physical_page':2,'status':'same_physical_page_alternate_reading_order_text',
              'extraction_mode':'poppler_default_reading_order_utf8_not_layout',
              'exact_layout_page_poppler_sha256':self.page_sha,
              'reading_order_poppler_sha256':self.sha,'extracted_text_sha256':self.sha,
              'extracted_page_text':self.text,'text_characters':len(self.text),
              'complete_extractor_output_not_truncated':True,
              'does_not_replace_verified_layout_excerpt_or_offsets':True,
              'visual_reading_order_or_factual_claim_truth_verified':False,
              'required_layout_excerpt_also_contiguous_in_reading_order':True}
        for patch in ({'physical_page':3},{'extracted_page_text':'changed'},
                      {'exact_layout_page_poppler_sha256':'0'*64},
                      {'required_layout_excerpt_also_contiguous_in_reading_order':False},
                      {'visual_reading_order_or_factual_claim_truth_verified':True}):
            with self.subTest(patch=patch):
                self.context['physical_pages']['optional_center_reading_order_projection']={**good,**patch}
                with self.assertRaisesRegex(ValueError,'alternate reading-order'):
                    self.call()

    def test_explicit_rank_is_rechecked_before_original_page_read(self):
        second={**self.payload['passages'][0],'publisher_pdf_page':{
            'status':'verified','page':3,'page_text_sha256':self.page_sha}}
        payload={**self.payload,'passages':[self.payload['passages'][0],second]}
        self.context['physical_pages']['requested_center_physical_page']=3
        self.context['physical_pages']['pages'][0]['physical_page']=3
        proc=SimpleNamespace(returncode=0,stdout=json.dumps(payload),stderr='')
        with mock.patch.object(self.mod.subprocess,'run',return_value=proc), \
             mock.patch.object(self.mod,'original_pdf_context',return_value=self.context) as phys:
            got=self.mod.context_from_current_source_evidence('original study benchmark question',self.path,
                    passage_rank=2,expected_identity=('a'*64,'b'*64,self.sha,3,self.page_sha))
        phys.assert_called_once_with(self.path,3,neighbor_radius=1,required_center_excerpt=self.text,
                                    include_center_reading_order=False)
        self.assertEqual(2,got['selected_source_excerpt_rank'])
        self.assertEqual(3,got['correct_physical_original_pdf_page'])
        self.assertIsNone(got['first_verified_source_excerpt_rank'])
        self.assertTrue(got['explicit_prior_passage_identity_matched'])
        with mock.patch.object(self.mod.subprocess,'run',return_value=proc), \
             mock.patch.object(self.mod,'original_pdf_context',side_effect=AssertionError('bad call')):
            with self.assertRaisesRegex(ValueError,'no longer matches'):
                self.mod.context_from_current_source_evidence('original study benchmark question',self.path,
                        passage_rank=2,expected_identity=('a'*64,'b'*64,'0'*64,3,self.page_sha))
        with mock.patch.object(self.mod.subprocess,'run',side_effect=AssertionError('reader launched')):
            with self.assertRaisesRegex(ValueError,'requires all four'):
                self.mod.context_from_current_source_evidence('original study benchmark question',self.path,passage_rank=2)

    def test_invalid_source_or_query_never_launches_existing_reader(self):
        with mock.patch.object(self.mod.subprocess,'run',side_effect=AssertionError('reader launched')):
            for name,q in (('../wrong.pdf','normal query'),('/tmp/source.pdf','normal query'),
                           (self.path,'a'),(self.path,'bad query'*150)):
                with self.subTest(name=name,query=q[:20]):
                    with self.assertRaises(ValueError):
                        self.mod.context_from_current_source_evidence(q,name)

    def test_no_verified_page_cannot_be_replaced_with_nearest_other_document(self):
        wrong={**self.payload,'passages':[{**self.payload['passages'][0],
                             'publisher_pdf_page':{'status':'unverified','reason':'no_text_page_label'}}]}
        proc=SimpleNamespace(returncode=0,stdout=json.dumps(wrong),stderr='')
        with mock.patch.object(self.mod.subprocess,'run',return_value=proc), \
             mock.patch.object(self.mod,'original_pdf_context',side_effect=AssertionError('page inferred')):
            with self.assertRaisesRegex(ValueError,'no verified original physical PDF source excerpt'):
                self.mod.context_from_current_source_evidence('original study benchmark',self.path)

    def test_existing_reader_failure_is_not_a_valid_source_citation(self):
        proc=SimpleNamespace(returncode=1,stdout='partial',stderr='not trusted')
        with mock.patch.object(self.mod.subprocess,'run',return_value=proc), \
             mock.patch.object(self.mod,'original_pdf_context',side_effect=AssertionError('bad call')):
            with self.assertRaisesRegex(ValueError,'native current-source citation lookup failed'):
                self.mod.context_from_current_source_evidence('original study benchmark',self.path)

    def test_later_original_revision_or_page_sha_mismatch_rejected(self):
        changed={**self.context,'physical_pages':{**self.context['physical_pages'],
                  'pages':[{'physical_page':2,'physical_page_poppler_sha256':'0'*64}]}}
        proc=SimpleNamespace(returncode=0,stdout=json.dumps(self.payload),stderr='')
        with mock.patch.object(self.mod.subprocess,'run',return_value=proc), \
             mock.patch.object(self.mod,'original_pdf_context',return_value=changed):
            with self.assertRaisesRegex(ValueError,'digest diverges'):
                self.mod.context_from_current_source_evidence('original study benchmark',self.path)

if __name__=='__main__':unittest.main()
