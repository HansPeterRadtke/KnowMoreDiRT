"""No model inference: audit external source gold and saved raw failed output."""
import hashlib,json,unittest
from pathlib import Path
HERE=Path(__file__).parent
ROOT=Path('/data/tmp/kmd-scifact-release-20261024/data')
class KmdSourceModelContract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.f=json.loads((HERE/'frozen_model_science_20261024.json').read_text())
  cls.a=json.loads((HERE/'labelled_model_comparison_20261024.json').read_text())
  cls.b=json.loads((HERE/'source_coverage_and_extraction_failure_audit_20261024.json').read_text())
  cls.raw=[json.loads(t) for t in (HERE/'results_checkpoint.jsonl').read_text().splitlines()]
 def test_external_original_source_snapshot_frozen(self):
  self.assertEqual(self.f['original_train_sha256'],hashlib.sha256((ROOT/'claims_train.jsonl').read_bytes()).hexdigest())
  self.assertEqual(self.a['pre_inference_frozen_hash'],hashlib.sha256((HERE/'frozen_model_science_20261024.json').read_bytes()).hexdigest())
  self.assertEqual(len(self.raw),len(self.f['cases']))
 def test_do_not_launder_failed_box_extraction_as_valid_inference(self):
  self.assertEqual(self.b['extraction_length_truncations'],8)
  self.assertEqual(self.b['model_generated_frame_invalid'],8)
  self.assertEqual(self.b['box_judge_abstentions_due_to_invalid_structure'],8)
  self.assertTrue(all(x['box_extraction']['finish_reason']=='length' for x in self.raw))
  self.assertTrue(all(x['box_judgment'].get('skipped')=='invalid_extracted_frame' for x in self.raw))
 def test_same_actual_science_claims_and_expert_gold_not_repaired(self):
  m=self.a['metrics'];self.assertEqual(m['sample_count'],8)
  self.assertEqual(m['gold_counts'],{'SUPPORT':4,'CONTRADICT':4})
  self.assertEqual((m['direct_correct'],m['grounded_box_frames']), (4,0))
  self.assertEqual(m['confusion_direct'],{'SUPPORT=>SUPPORT':4,'CONTRADICT=>SUPPORT':4})
  self.assertEqual(self.b['human_rationale_recovered_in_contradiction_cases'],4)
  self.assertEqual(self.b['contradiction_claims_wrongly_called_support'],4)
 def test_source_substring_and_original_annotation_indices(self):
  frozen={x['uid']:x for x in self.f['cases']}
  for x in self.b['rows']:
   self.assertEqual(x['source_sha256'],frozen[x['uid']]['source_sha256'])
   if x['direct_model_quote_exact_input']:
    self.assertIn(x['direct_quote'],frozen[x['uid']]['source_evidence'])
  self.assertEqual(self.b['direct_quotes_exact_source'],7)
if __name__=='__main__':unittest.main()
