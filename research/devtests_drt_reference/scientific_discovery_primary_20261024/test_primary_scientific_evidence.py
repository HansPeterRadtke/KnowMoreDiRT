"""Source integrity, published label continuity and historical evidence tests.

No inference model, no source downloads, no rewrites to previous outcomes.
"""
import hashlib,json,math,unittest
from pathlib import Path
from real_neutrino_source_reasoning import parse_numeric, dated_cern_claims
from real_neutrino_source_reasoning_corrected import corrected_result,primary_measurement_verified

HERE=Path(__file__).resolve().parent
DATA=Path('/data/tmp/kmd-scifact-release-20261024/data')

def jsonl(filename):
 return [json.loads(s) for s in (DATA/filename).read_text().splitlines() if s]

class RealScientificEvidenceTests(unittest.TestCase):
    def test_preserve_original_bad_identity_and_v2_correct_identity(self):
        old=json.loads((HERE/'real_neutrino_source_analysis_results_20261024.json').read_text())
        self.assertEqual(old['measurements']['opera_2012']['measured_experiment'],'ICARUS')
        current=json.loads((HERE/'real_neutrino_source_analysis_corrected_20261024.json').read_text())
        self.assertEqual(current['old_result_sha256'],hashlib.sha256((HERE/'real_neutrino_source_analysis_results_20261024.json').read_bytes()).hexdigest())
        self.assertEqual(current['measurements']['opera_2012']['measured_experiment'],'OPERA')
        self.assertEqual(current['measurements']['icarus_2012']['measured_experiment'],'ICARUS')
        self.assertIn('OPERA',current['measurements']['opera_2012']['paper_author'])
        self.assertEqual(current['measurements']['opera_2012']['value_ns'],old['measurements']['opera_2012']['value_ns'])
        self.assertEqual(current['measurements']['icarus_2012']['value_ns'],old['measurements']['icarus_2012']['value_ns'])

    def test_real_publication_vintage_blocks_future_knowledge(self):
        c=dated_cern_claims()
        oldest=[x for x in c if x['date']<='2011-09-23']
        self.assertTrue(any(x['claim_type']=='initial_early_arrival' for x in oldest))
        self.assertFalse(any(x['claim_type']=='four_luminal_measurements' for x in oldest))
        self.assertFalse(any(x['claim_type']=='possible_fault' for x in oldest))
        up_to_march=[x for x in c if x['date']<='2012-03-16']
        self.assertTrue(any(x['claim_type']=='icarus_independent_measurement' for x in up_to_march))
        self.assertFalse(any(x['claim_type']=='four_luminal_measurements' for x in up_to_march))
        self.assertTrue(any(x['claim_type']=='four_luminal_measurements' for x in c if x['date']<='2012-06-08'))
        cond=[x for x in oldest if x['claim_type']=='hypothetical_future']
        self.assertEqual(len(cond),1)
        self.assertTrue(cond[0]['is_quote_or_conditional'])

    def test_hypothesis_is_conditional_and_paper_numerics_not_invented(self):
        r=corrected_result()
        self.assertAlmostEqual(r['derived_hypothesis_check']['derived_expected_early_arrival_ns'],48.6994,places=3)
        self.assertEqual(r['measurements']['opera_2012']['value_ns'],0.6)
        self.assertEqual(r['measurements']['opera_2012']['systematic_ns'],3.0)
        self.assertEqual(r['measurements']['icarus_2012']['value_ns'],0.1)
        self.assertEqual(r['measurements']['icarus_2012']['systematic_ns'],2.39)
        self.assertGreater(r['derived_hypothesis_check']['derived_expected_early_arrival_ns'],10*r['measurements']['icarus_2012']['combined_ns_assuming_independent_errors'])
        self.assertNotEqual(r['measurements']['opera_2012']['paper_author'],r['measurements']['icarus_2012']['paper_author'])

    def test_scifact_full_corpus_retrieval_independent_gold(self):
        r=json.loads((HERE/'scifact_fixed_retrieval_results_20261024.json').read_text())
        self.assertEqual((r['corpus_docs'],r['claims'],r['claims_with_labeled_evidence']),(5183,300,188))
        for key,expect in [('title_tfidf',126),('title_abstract_tfidf',171),('abstract_bm25',178)]:
            self.assertEqual(r['methods'][key]['one_relevant_in_top_10'],expect)
        self.assertEqual((r['rationale_oracle_gold_doc']['hit_top2'],r['rationale_oracle_gold_doc']['hit_top2_plus_neighbors']),(139,161))
        self.assertEqual(hashlib.sha256((DATA/'corpus.jsonl').read_bytes()).hexdigest(),r['source_sha256']['corpus.jsonl'])

    def test_generic_scientific_nli_wrong_polarity_is_visible(self):
        r=json.loads((HERE/'scifact_distinct_train_nli_results_20261024.json').read_text())
        self.assertEqual((r['total_fixed_pairs'],r['correct'],r['evaluable_unambiguous_gold_pairs']),(96,39,96))
        wrong=[x for x in r['results'] if x['label_predicted']!='NEUTRAL' and x['label_predicted']!=x['human_annotated_label']]
        self.assertEqual(len(wrong),14)
        c=[x for x in wrong if x['claim_id']==1288]
        self.assertEqual(len(c),1)
        self.assertEqual((c[0]['human_annotated_label'],c[0]['label_predicted']),('CONTRADICT','SUPPORT'))
        self.assertEqual(r['provenance']['sha256_train'],hashlib.sha256((DATA/'claims_train.jsonl').read_bytes()).hexdigest())

    def test_scope_counterfactual_changes_evidence_validity_not_original(self):
        r=json.loads((HERE/'real_neutrino_source_analysis_corrected_20261024.json').read_text())
        original_claim=next(c for c in r['historical_cern_claims'] if c['claim_type']=='initial_early_arrival')
        assert original_claim['date']=='2011-09-23'
        # Explicit metadata intervention. A dream wrapper would be invalid
        # evidence for actual arrival, without changing raw recorded text.
        def eligible_for_real_world(claim,world,as_of):
            return world=='real' and claim['date']<=as_of and not claim['is_quote_or_conditional']
        self.assertTrue(eligible_for_real_world(original_claim,'real','2011-09-23'))
        self.assertFalse(eligible_for_real_world(original_claim,'dream','2011-09-23'))
        new=next(c for c in r['historical_cern_claims'] if c['claim_type']=='four_luminal_measurements')
        self.assertFalse(eligible_for_real_world(new,'real','2011-09-23'))
        self.assertTrue(eligible_for_real_world(new,'real','2012-06-08'))

if __name__=='__main__': unittest.main()
