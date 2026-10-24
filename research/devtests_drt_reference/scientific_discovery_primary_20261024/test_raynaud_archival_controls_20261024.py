"""Scientific source-group, historical-time and negation controls only.

Reads pre-existing immutable original-source snapshots and completed results.
Does not regenerate model inferences, mutate source documents or retune gold.
"""
from pathlib import Path
import hashlib,json,unittest

BASE=Path(__file__).resolve().parent
SRC=Path('/data/tmp/kmd-raynaud-originals-20261024')

class RaynaudSourceGroundingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a=json.loads((BASE/'raynaud_archival_discovery_results_20261024.json').read_text())
        cls.b=json.loads((BASE/'raynaud_author_provenance_results_20261024.json').read_text())

    def test_original_published_records_not_substituted_or_rewritten(self):
        for row in self.a['sources']:
            raw=(SRC/(row['id']+'.json')).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),row['source_record_sha256'])
            self.assertEqual(json.loads(raw)['title'],row['title'])
        self.assertEqual(len(self.a['sources']),7)

    def test_pre1986_mechanistic_path_is_hypothesis_not_tracked_as_truth(self):
        join=self.a['pre1986_automatic_lexical_literature_bridge']
        self.assertEqual(join['inference_type'],'plausible_mediated_hypothesis_not_logical_entailment')
        self.assertTrue(join['no_direct_claim_in_curated_pre1986_abstracts'])
        self.assertTrue(join['not_exhaustive_world_prior_art_search'])
        x=self.a['as_of_date_controls']
        self.assertEqual(len(x['as_of_1985_12_31_source_pmids']),5)
        self.assertEqual(set(x['excluded_future_pmids']),{'3797213','2536517'})

    def test_independently_published_negative_primary_and_nontransportable_cohorts(self):
        controls=self.a['negative_and_mismatch_controls']
        self.assertTrue(controls['1977_primary_null_venesection_negative_retained'])
        self.assertTrue(controls['1975_cohort_dominated_by_secondary_not_primary'])
        claims=self.a['pre1986_automatic_pattern_extractions']
        p=next(x for x in claims if x['claim_category']=='primary_raynaud_no_detected_viscosity_difference')
        self.assertIn('no significant difference',p['source_provenance']['exact_evidence'])
        self.assertIn('did not change',p['additional_intervention_negative']['exact_evidence'])
        self.assertIn('primary',p['population_scope'])

    def test_trial_is_later_independent_partial_response_not_mechanism_proof(self):
        x=self.a['independent_followup_randomized_study']
        self.assertEqual((x['publication_year'],x['total_randomized']),(1989,32))
        self.assertAlmostEqual(x['descriptive_change_minutes'],15.2)
        self.assertEqual(x['reported_subgroup_result'],'benefit in primary, not established in secondary')
        self.assertTrue(x['no_causal_mechanism_proven'])
        self.assertNotIn('swanson',x['trial_authors'].lower())

    def test_first_person_retrospective_is_not_original_source_or_blinded_trial(self):
        self.assertEqual(self.b['year'],1989)
        self.assertTrue(self.b['not_the_original_1986_paper'])
        self.assertTrue(self.b['not_independent_clinical_replication'])
        self.assertEqual(hashlib.sha256((SRC/'swanson_1989_first_person_1989_letter.html').read_bytes()).hexdigest(),self.b['source_sha256'])
        self.assertFalse(next(x for x in self.a['sources'] if x['id']=='3797213')['abstract_available'])

if __name__=='__main__':unittest.main()
