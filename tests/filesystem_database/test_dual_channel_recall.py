"""Native source-channel recall is exact-identity discovery, not answer truth."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from file_system_catalog.dual_channel_recall import (
    evaluate_verified_dual_case,aggregate_verified_dual_cases,
)


class AttestedDualChannelRecallTests(unittest.TestCase):
    def setUp(self):
        self.query='Which original papers describe rough surface reflection?'
        self.case={'id':'roughness','query':self.query,'category':'multi_source',
                   'relevant_paths':['books/microfacet.html','studies/roughness.html']}
        preview='Original publisher text about roughness and microfacet reflectance.'
        self.payload={
            'schema':'kmd.two_channel_original_discovery.v1','query':self.query,
            'no_database_or_original_mutation':True,
            'no_ranking_fusion_or_score_threshold_applied':True,
            'summary_channel':[{'original_path':'books/microfacet.html','summary_rank':1,
                                'publisher_revision_verified':True}],
            'original_passage_channel':[{'original_path':'studies/roughness.html',
                 'original_source_rank':1,'publisher_revision_verified':True,
                 'passage_preview':preview,'passage_preview_sha256':hashlib.sha256(preview.encode()).hexdigest()}],
        }

    def test_frozen_multi_source_labels_recovered_by_complementary_channels(self):
        result=evaluate_verified_dual_case(self.case,self.payload,top=10)
        self.assertEqual((2,1,1,2),(result['known_labels'],result['raw_topk_hits'],
                                       result['summary_topk_hits'],result['either_channel_topk_hits']))
        self.assertTrue(result['complete_known_labels_in_either_channel'])
        self.assertEqual(2,len(result['known_original_ranks']))
        self.assertEqual(1,result['known_original_ranks'][0]['hierarchical_summary_rank'])
        self.assertIsNone(result['known_original_ranks'][0]['original_passage_rank'])
        self.assertEqual(1,result['known_original_ranks'][1]['original_passage_rank'])
        summary=aggregate_verified_dual_cases([result],top=10)
        self.assertEqual(2,summary['known_source_labels'])
        self.assertEqual(2,summary['either_channel_source_hits'])
        self.assertEqual(1,summary['cases_with_all_known_sources_in_either_channel'])
        self.assertEqual(1.0,summary['either_source_recall'])
        self.assertIn('NOT exhaustive relevance',summary['notes'].replace('Not exhaustive relevance','NOT exhaustive relevance'))

    def test_named_publisher_digest_and_rank_are_verified_before_scoring(self):
        for mutate in (
            lambda x:x.update(schema='unsafe'),
            lambda x:x.update(query='Different question'),
            lambda x:x.update(no_database_or_original_mutation=False),
            lambda x:x.update(no_ranking_fusion_or_score_threshold_applied=False),
            lambda x:x['original_passage_channel'][0].update(passage_preview_sha256='f'*64),
            lambda x:x['original_passage_channel'][0].update(publisher_revision_verified=False),
            lambda x:x['original_passage_channel'][0].update(original_source_rank=3),
            lambda x:x['summary_channel'][0].update(summary_rank=0),
            lambda x:x['summary_channel'].append(deepcopy(x['summary_channel'][0])),
        ):
            with self.subTest(mutation=mutate):
                x=deepcopy(self.payload)
                mutate(x)
                with self.assertRaises(ValueError):
                    evaluate_verified_dual_case(self.case,x,top=10)

    def test_unverified_and_missing_gold_says_missing_not_unanswerable(self):
        x=deepcopy(self.payload)
        x['original_passage_channel']=[]
        row=evaluate_verified_dual_case(self.case,x,top=10)
        self.assertEqual(1,row['either_channel_topk_hits'])
        self.assertFalse(row['complete_known_labels_in_either_channel'])
        self.assertEqual(0,row['raw_topk_hits'])
        report=aggregate_verified_dual_cases([row],top=10)
        self.assertEqual(.5,report['either_source_recall'])
        self.assertEqual(0,report['cases_with_all_known_sources_in_either_channel'])
        self.assertIn('Not exhaustive relevance',report['notes'])

    def test_cli_wrong_gold_sha_refuses_before_model_or_file_creation(self):
        program=Path(__file__).resolve().parents[2]/'scripts/raspi_dual_channel_recall.py'
        spec=importlib.util.spec_from_file_location('testable_dual_recall_cli',program)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            gold=root/'frozen.json'
            gold.write_text(json.dumps({'schema':'kmd.frozen_multirelevant_original_gold.v1',
                                        'frozen_before_any_new_queries':True,'cases':[self.case]}))
            result=root/'should-not-exist.json'
            argv=['raspi_dual_channel_recall.py','--gold',str(gold),
                  '--gold-sha256','0'*64,
                  '--dual-cli',str(root/'dummy_dual.py'),
                  '--source-map',str(root/'dummy.sqlite'),
                  '--original-root',str(root/'originals'),
                  '--derived-root',str(root/'derived'),
                  '--completed-ledger',str(root/'ledger.json'),
                  '--output',str(result)]
            with mock.patch('sys.argv',argv),\
                 mock.patch.object(module.subprocess,'run',side_effect=AssertionError('no model allowed')):
                with self.assertRaisesRegex(ValueError,'frozen gold hash changed'):
                    module.main()
            self.assertFalse(result.exists())

    def test_inconsistent_manual_counts_are_rejected(self):
        row=evaluate_verified_dual_case(self.case,self.payload,top=10)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            aggregate_verified_dual_cases([row,row],top=10)
        bad=deepcopy(row)
        bad['either_channel_topk_hits']=0
        with self.assertRaisesRegex(ValueError,'union recall'):
            aggregate_verified_dual_cases([bad],top=10)
        bad=deepcopy(row)
        bad['raw_topk_hits']=True
        with self.assertRaisesRegex(ValueError,'untrusted'):
            aggregate_verified_dual_cases([bad],top=10)
        for k in (0,21,True):
            with self.subTest(top=k),self.assertRaises(ValueError):
                evaluate_verified_dual_case(self.case,self.payload,top=k)


if __name__=='__main__':
    unittest.main()
