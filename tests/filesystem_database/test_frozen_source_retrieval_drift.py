"""Private same-snapshot frozen source recall drift, not answer evaluation."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from file_system_catalog.frozen_source_retrieval_drift import (
    compare_frozen_source_ranks,report_from_files,CHANNELS,TOPS,SCHEMA,
)


def sha(text):return hashlib.sha256(text.encode()).hexdigest()


def fixture(*,new=False):
    path='ai-agents/current-official-publisher.pdf'
    row={'id':'agent_source_frozen','group':'independent_27','family':'unseen_independent_27',
         'query_sha256':sha('How do I verify a grounded agent citation?'),
         'known_originals':[path],
         'source_revisions':{path:{'original_sha256':sha('original'),
                                   'derived_sha256':sha('derived')}},
         'label_ranks':{k:[8 if not new else 7] for k in CHANNELS}}
    return {'schema':SCHEMA,'captured_utc':('2026-11-24T06:00Z' if new else '2026-11-22T20:00Z'),
          'all_search_channels_and_questions_use_one_captured_native_in_memory_embedding_set':True,
          'nonproduction_original_chunk_ranking_excludes_original_analysis_facet_vectors':True,
          'native_snapshot':{'complete_original_paths':9614 if new else 9108},
          'gold_original_bytes':{
              'all_frozen_known_original_revisions_match_current_source_map_and_download_ledger':True},
          'frozen_gold_file_sha256':{k:sha(k) for k in ('historical','independent','multiple','heldout')},
          'query_encoder_revision':'local-e5-v1',
          'lexical_policy':{'method':'temporary-original-title-FTS5'},
          'rrf_policy':{'k':60},
          'cases':[row]}


class TwoEraSourceRankDriftTests(unittest.TestCase):
    def test_identical_source_labels_rank_changed_but_no_top10_regression(self):
        report=compare_frozen_source_ranks(fixture(),fixture(new=True))
        self.assertEqual(1,report['comparable_source_labels'])
        self.assertEqual(0,report['incomparable_source_labels'])
        self.assertEqual(9614,report['newer_complete_original_roots'])
        self.assertEqual(1,report['changed_rank_case_count'])
        self.assertTrue(report['identical_top10_3channel_source_label_recall'])
        for channel in CHANNELS:
            self.assertEqual({'before_hits':1,'after_hits':1,'gains':0,'regressions':0},
                             report['channel_scores'][channel]['10'])
        self.assertTrue(report['not_answer_accuracy_or_original_pdf_physical_page_truth'])

    def test_reliable_top10_gains_and_regressions_scored_only_for_matching_source_revision(self):
        old=fixture();new=fixture(new=True)
        old['cases'][0]['label_ranks']['rrf_three']=[9]
        new['cases'][0]['label_ranks']['rrf_three']=[12]
        first=compare_frozen_source_ranks(old,new)
        self.assertEqual(1,first['channel_scores']['rrf_three']['10']['regressions'])
        self.assertEqual(0,first['channel_scores']['rrf_three']['10']['after_hits'])
        self.assertEqual('regression',first['per_source_rank_changes'][0]['top10_navigation_changes']['rrf_three'])
        old['cases'][0]['label_ranks']['rrf_three']=[13]
        new['cases'][0]['label_ranks']['rrf_three']=[4]
        second=compare_frozen_source_ranks(old,new)
        self.assertEqual(1,second['channel_scores']['rrf_three']['10']['gains'])
        self.assertEqual('gain',second['per_source_rank_changes'][0]['top10_navigation_changes']['rrf_three'])
        self.assertEqual(0,second['channel_scores']['rrf_three']['1']['after_hits'])
        self.assertEqual(1,second['channel_scores']['rrf_three']['5']['after_hits'])

    def test_new_original_revision_is_incomparable_not_a_false_source_ranking_regression(self):
        old=fixture();new=fixture(new=True)
        new['cases'][0]['source_revisions'][new['cases'][0]['known_originals'][0]]['derived_sha256']=sha('changed derived')
        new['cases'][0]['label_ranks']['rrf_three']=[None]
        report=compare_frozen_source_ranks(old,new)
        self.assertEqual(0,report['comparable_source_labels'])
        self.assertEqual(1,report['incomparable_source_labels'])
        self.assertEqual(0,report['channel_scores']['rrf_three']['10']['regressions'])
        self.assertEqual('source_or_derived_revision_changed',report['changed_source_labels_not_scored'][0]['reason'])

    def test_frozen_question_mutation_or_changed_gold_set_is_not_comparable(self):
        for which in ('gold_hash','query','original','encoder','policy','missing_case'):
            old=fixture();new=fixture(new=True)
            if which=='gold_hash':new['frozen_gold_file_sha256']['historical']=sha('changed gold')
            elif which=='query':new['cases'][0]['query_sha256']=sha('different query')
            elif which=='original':new['cases'][0]['known_originals']=['different/source.pdf']
            elif which=='encoder':new['query_encoder_revision']='changed-embedder'
            elif which=='policy':new['rrf_policy']={'k':100}
            else:new['cases']=[]
            with self.subTest(which=which),self.assertRaises(ValueError):
                compare_frozen_source_ranks(old,new)

    def test_malformed_ranks_labels_source_identity_or_non_snapshot_run_refused(self):
        for which in ('missing_rank','range_zero','bool_rank','duplicate_id','bad_path','wrong_sha','missing_proof','not_same_snapshot'):
            old=fixture();new=fixture(new=True)
            row=new['cases'][0];p=row['known_originals'][0]
            if which=='missing_rank':row['label_ranks'].pop('summary')
            elif which=='range_zero':row['label_ranks']['rrf_three']=[0]
            elif which=='bool_rank':row['label_ranks']['rrf_three']=[True]
            elif which=='duplicate_id':new['cases'].append(copy.deepcopy(row))
            elif which=='bad_path':row['known_originals']=['../private.pdf'];row['source_revisions']={'../private.pdf':row['source_revisions'][p]}
            elif which=='wrong_sha':row['source_revisions'][p]['original_sha256']='aaa'
            elif which=='missing_proof':new['gold_original_bytes']['all_frozen_known_original_revisions_match_current_source_map_and_download_ledger']=False
            else:new['all_search_channels_and_questions_use_one_captured_native_in_memory_embedding_set']=False
            with self.subTest(which=which),self.assertRaises(ValueError):
                compare_frozen_source_ranks(old,new)

    def test_case_labels_with_multiple_originals_are_scored_independently(self):
        old=fixture();new=fixture(new=True)
        extra='ai-agents/additional.pdf'
        for d in (old,new):
            row=d['cases'][0]
            row['known_originals'].append(extra)
            row['source_revisions'][extra]={'original_sha256':sha('source-2'),'derived_sha256':sha('derived-2')}
            for name in CHANNELS:
                row['label_ranks'][name].append(15)
        new['cases'][0]['label_ranks']['lexical'][1]=2
        report=compare_frozen_source_ranks(old,new)
        self.assertEqual(2,report['comparable_source_labels'])
        self.assertEqual(1,report['channel_scores']['lexical']['10']['gains'])
        self.assertEqual(2,report['channel_scores']['lexical']['10']['after_hits'])
        self.assertEqual(0,report['channel_scores']['rrf_three']['10']['gains'])

    def test_file_evidence_hashes_bound_both_private_input_snapshots(self):
        with TemporaryDirectory() as dirname:
            root=Path(dirname);old=root/'old.json';new=root/'new.json'
            old_raw=json.dumps(fixture()).encode();new_raw=json.dumps(fixture(new=True)).encode()
            old.write_bytes(old_raw);new.write_bytes(new_raw)
            report=report_from_files(old,new)
            self.assertEqual(hashlib.sha256(old_raw).hexdigest(),report['older_private_report_sha256'])
            self.assertEqual(hashlib.sha256(new_raw).hexdigest(),report['newer_private_report_sha256'])
            old.write_bytes(b'not JSON')
            with self.assertRaises((ValueError,json.JSONDecodeError)):
                report_from_files(old,new)
            old.write_bytes(old_raw)
            old.unlink();old.symlink_to(new)
            with self.assertRaisesRegex(ValueError,'missing or unbounded'):
                report_from_files(old,new)


if __name__=='__main__':unittest.main()
