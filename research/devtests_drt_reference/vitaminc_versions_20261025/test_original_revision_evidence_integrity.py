"""Read-only integrity verification of independent publisher originals and outputs.

No inference, no original source changes, no gold overwriting and no special
'gold repairing' or semantic conversion. Tests only observed factual outputs.
"""
from pathlib import Path
import hashlib,json,unittest
HERE=Path(__file__).parent
SOURCE=Path('/data/tmp/kmd-vitaminc-original-test-20261025.jsonl')
F=HERE/'frozen_real_revision_source_only_20261025.json'
G=HERE/'raw_source_version_nli_predictions_20261025.jsonl'
S=HERE/'human_gold_revision_contrast_results_20261025.json'
C=HERE/'real_revision_provenance_cluster_audit_20261025.json'
SPECIALIST=HERE/'paired_generic_vs_contrastive_specialist_20261025.json'
SPECIAL_RAW=HERE/'raw_source_revision_vitaminc_specialist_predictions_20261025.jsonl'
class OriginalVitaminCVersionEvidence(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.f=json.loads(F.read_text());cls.s=json.loads(S.read_text());cls.c=json.loads(C.read_text())
 def test_original_publisher_original_license_and_hash(self):
  expected='7ad1808dbc30c62e0a1427a53022d0dfaff668a1fde3c4b612a2d266edd753ad'
  self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(),expected)
  self.assertEqual(self.f['source_sha256'],expected)
  self.assertEqual(self.s['original_test_sha256'],expected)
  self.assertEqual(self.f['original_license'],'CC BY-SA 3.0; raw original text retained externally, no wholesale dataset copy to project Git')
 def test_source_only_sample_frozen_and_all_model_predictions_unchanged(self):
  self.assertEqual(hashlib.sha256(F.read_bytes()).hexdigest(),self.s['frozen_model_input_selection_sha256'])
  self.assertEqual(hashlib.sha256(G.read_bytes()).hexdigest(),self.s['completed_raw_model_prediction_sha256'])
  self.assertEqual((self.f['selected_groups'],self.f['selected_original_pairs']),(700,2800))
  self.assertEqual(len({x['case_id'] for x in self.f['case_ids_and_input_hashes_only']}),700)
  for x in self.f['case_ids_and_input_hashes_only']:
   self.assertNotIn('label',x)
   self.assertEqual(len(x['source_data_row_ids']),4)
   self.assertEqual(len(x['original_claim_evidence_pair_hashes']),4)
 def test_independent_original_labels_show_version_sensitivity(self):
  a=self.s['full_original_real_four_way_subset']
  self.assertEqual(a['n_same_claim_with_two_revision_variants'],11988)
  self.assertEqual(a['changed_source_entailment_labels'],11958)
  self.assertEqual(a['label_pair_frequency']['REFUTES+SUPPORTS'],9862)
  self.assertEqual(a['label_pair_frequency']['NOT ENOUGH INFO+SUPPORTS'],2082)
  self.assertLess(a['version_blind_one_label_max_accuracy_from_gold'],0.502)
 def test_actual_generic_model_errors_preserved_unrepaired(self):
  x=self.s['frozen_separate_nli_sample']['model']
  self.assertEqual((x['n_model_source_pairs'],x['model_correct']),(2800,1136))
  self.assertEqual((x['human_gold_label_flips_across_versions'],x['correctly_detected_label_flips'],x['missed_human_gold_label_flips']),(1396,529,867))
  self.assertEqual((x['both_source_versions_label_correct'],x['input_truncated_count']),(282,4))
 def test_revisions_are_not_independent_claims_or_dates(self):
  self.assertEqual(self.c['independent_wikipedia_revision_transition_families_in_original_subset'],5098)
  self.assertEqual(self.c['original_real_four_way_groups'],5994)
  self.assertEqual(self.c['frozen_sample_distinct_revision_families'],680)
  self.assertEqual(self.c['human_gold_original_score_sha256'],hashlib.sha256(S.read_bytes()).hexdigest())
 def test_posthoc_specialist_uses_same_frozen_case_ids_when_completed(self):
  if not SPECIALIST.exists():self.skipTest('Specialist comparison not scored until all 2,800 source predictions complete')
  result=json.loads(SPECIALIST.read_text())
  self.assertEqual(result['same_original_pairs'],2800)
  self.assertEqual(result['frozen_case_subset_sha256'],hashlib.sha256(F.read_bytes()).hexdigest())
  self.assertEqual(result['specialist_predictions_sha256'],hashlib.sha256(SPECIAL_RAW.read_bytes()).hexdigest())
  self.assertEqual(result['generic_predictions_sha256'],hashlib.sha256(G.read_bytes()).hexdigest())
  self.assertEqual(sum(c['count'] for c in result['vitaminc_specialist']['confusion']),2800)
if __name__=='__main__':unittest.main()
