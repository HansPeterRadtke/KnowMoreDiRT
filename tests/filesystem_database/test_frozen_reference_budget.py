"""Fixed-ten-slot source-label test excludes unsupported ranking improvements."""
from __future__ import annotations
import copy
import unittest

from file_system_catalog.frozen_reference_budget import compare_frozen_reference_budgets

SHA='a'*64


def fixture():
    native={'schema':'kmd.actual_current_same_read_snapshot_triple_channel_predeclared_gold_recall.v1',
       'gold_original_bytes':{'all_frozen_known_original_revisions_match_current_source_map_and_download_ledger':True},
       'all_search_channels_and_questions_use_one_captured_native_in_memory_embedding_set':True,
       'cases':[]}
    html={'schema':'kmd.private_unseen_73_frozen_same_snapshot_retrieval_official_HTML_archive_link_supplementary_recall.v1',
       'no_live_source_index_writes_or_acquisition':True,
       'frozen_original_results_SHA256':SHA,'frozen_question_count':73,
       'frozen_gold_attested_labels':79,'cases':[]}
    md={'schema':'kmd.private_frozen_73_original_label_approved_markdown_source_reference_navigation.v1',
       'no_source_download_or_live_native_index_change':True,
       'source_frozen_rank_report_sha256':SHA,
       'frozen_gold_case_count':73,'known_original_label_count':79,'cases':[]}
    for i in range(73):
        id=f'case_{i:02d}';group='previous_46' if i<46 else 'independent_27'
        paths=[f'docs/rrf-{i}-{k}.html' for k in range(5)]
        gold=[f'docs/gold-{i}.html']
        if i<6:gold.append(f'docs/additional-{i}.html')
        ranks=[10 if i in (0,1) else 1]+([None] if len(gold)>1 else [])
        native['cases'].append({'id':id,'group':group,'family':'QA',
             'query_sha256':SHA,'original_top5':{'rrf_three':paths},
             'known_originals':gold,'label_ranks':{'rrf_three':ranks},
             'source_revisions':{p:{'original_sha256':SHA,'derived_sha256':SHA} for p in gold}})
        html['cases'].append({'id':id,'known_originals':gold,'approved_source_link_proofs':[]})
        md['cases'].append({'id':id,'firstparty_approved_link_candidates':[]})
    # First source case: native #10 target displaced when one publisher link
    # takes source slot #10; that linked source is otherwise #30.
    html['cases'][0]['approved_source_link_proofs'].append({
         'parent_original_path':'docs/rrf-0-0.html','target_original_path':'docs/additional-0.html',
         'source_link_original_SHA':SHA,'target_original_receipt_SHA':SHA})
    return native,html,md


class FixedBudgetReferenceOriginalTests(unittest.TestCase):
    def run_evaluation(self,parts):
        return compare_frozen_reference_budgets(*parts,native_input_sha256=SHA)

    def test_native_top_ten_recall_is_not_replaced_by_free_additional_links(self):
        result=self.run_evaluation(fixture())
        self.assertEqual(73,result['frozen_original_questions'])
        self.assertEqual(79,result['source_known_original_labels'])
        baseline=result['group_scores']['previous_46']['10']
        substituted=result['group_scores']['previous_46']['9']
        self.assertEqual(46,baseline['source_known_label_hits'])
        self.assertEqual(45,substituted['source_known_label_hits'])
        self.assertEqual(-1,substituted['net_delta_hits'])
        self.assertEqual(1,len(substituted['gains']))
        self.assertEqual(2,len(substituted['losses']))
        self.assertEqual(10,substituted['maximum_source_candidate_slots_per_question'])
        self.assertTrue(result['not_a_factual_answer_quality_or_page_truth_measure'])
        self.assertTrue(result['not_an_authority_to_change_installed_source_rankings'])

    def test_without_links_adding_reserved_slots_never_invents_more_known_source_hits(self):
        n,h,m=fixture()
        h['cases'][0]['approved_source_link_proofs']=[]
        result=self.run_evaluation((n,h,m))
        self.assertEqual([],result['group_scores']['all']['9']['gains'])
        self.assertLessEqual(result['group_scores']['all']['9']['source_known_label_hits'],result['group_scores']['all']['10']['source_known_label_hits'])

    def test_new_markdown_firstparty_link_is_admitted_only_as_an_extra_candidate(self):
        n,h,m=fixture()
        h['cases'][0]['approved_source_link_proofs']=[]
        m['cases'][0]['firstparty_approved_link_candidates']=[{
            'parent_original':'docs/rrf-0-0.html','target_original':'docs/additional-0.html',
            'publisher_original_SHA256':SHA,'target_receipt_SHA256':SHA,
            'link_original_byte_sha256':SHA,'original_markdown_start_byte':120}]
        result=self.run_evaluation((n,h,m))
        self.assertEqual(['docs/additional-0.html'],result['cases'][0]['methods']['9']['added_original_link_candidates'])
        self.assertEqual(10,result['cases'][0]['methods']['9']['maximum_total_source_candidate_slots'])

    def test_duplicate_frozen_question_source_identity_rejected(self):
        n,h,m=fixture();n['cases'][1]['id']=n['cases'][0]['id']
        with self.assertRaisesRegex(ValueError,'duplicate'):
            self.run_evaluation((n,h,m))

    def test_wrong_frozen_native_digest_is_not_inferred_from_matching_titles(self):
        n,h,m=fixture();m['source_frozen_rank_report_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'different frozen'):
            self.run_evaluation((n,h,m))

    def test_forced_gold_original_identity_change_rejected(self):
        n,h,m=fixture();h['cases'][0]['known_originals']=['docs/forged.html']
        with self.assertRaisesRegex(ValueError,'ranking shape changed'):
            self.run_evaluation((n,h,m))

    def test_untrusted_html_link_parent_must_be_in_old_native_topfive(self):
        n,h,m=fixture();h['cases'][0]['approved_source_link_proofs'][0]['parent_original_path']='docs/unknown.html'
        with self.assertRaisesRegex(ValueError,'HTML link parent'):
            self.run_evaluation((n,h,m))

    def test_original_traversal_paths_and_ambiguous_link_digests_rejected(self):
        for source,field,value in [('html','target_original_path','../secret.pdf'),
                                    ('html','target_original_receipt_SHA','fake'),
                                    ('markdown','original_markdown_start_byte',True)]:
            n,h,m=fixture()
            if source=='html':h['cases'][0]['approved_source_link_proofs'][0][field]=value
            else:
                m['cases'][0]['firstparty_approved_link_candidates']=[{
                    'parent_original':'docs/rrf-0-0.html','target_original':'docs/additional-0.html',
                    'publisher_original_SHA256':SHA,'target_receipt_SHA256':SHA,
                    'link_original_byte_sha256':SHA,'original_markdown_start_byte':value}]
            with self.subTest(source=source,field=field),self.assertRaises(ValueError):
                self.run_evaluation((n,h,m))

    def test_unverified_or_unapproved_source_study_rejected(self):
        n,h,m=fixture();h['no_live_source_index_writes_or_acquisition']=False
        with self.assertRaisesRegex(ValueError,'unrecognized or unsupported'):
            self.run_evaluation((n,h,m))

    def test_source_budget_does_not_exceed_ten_even_with_repeated_children(self):
        n,h,m=fixture()
        h['cases'][0]['approved_source_link_proofs']*=50
        a=self.run_evaluation((n,h,m))
        self.assertLessEqual(len(a['cases'][0]['methods']['5']['added_original_link_candidates']),5)
        self.assertEqual(1,a['cases'][0]['available_distinct_same_origin_link_destinations'])

if __name__=='__main__':unittest.main()
