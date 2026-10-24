"""Read-only independent original-source integrity, not model inference."""
from pathlib import Path
import hashlib,json,unittest
from collections import Counter
HERE=Path(__file__).parent
SOURCE=Path('/data/tmp/kmd-contractnli-20261024/contract-nli/test.json')
class OriginalContractResultIntegrity(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.f=json.loads((HERE/'frozen_contract_source_scope_20261024.json').read_text())
  cls.o=json.loads((HERE/'human_gold_contract_scope_results_20261024.json').read_text())
  cls.p=[json.loads(x) for x in (HERE/'raw_model_comparison_20261024.jsonl').read_text().splitlines() if x]
 def test_original_gold_and_predicted_exactly_one_each(self):
  self.assertEqual(len(self.f['cases']),len(self.p))
  self.assertEqual(len(self.o['rows']),6)
  self.assertEqual(self.o['original_sha256'],hashlib.sha256(SOURCE.read_bytes()).hexdigest())
  self.assertEqual(self.o['preselected_frozen_sha256'],hashlib.sha256((HERE/'frozen_contract_source_scope_20261024.json').read_bytes()).hexdigest())
  self.assertEqual(self.o['verbatim_model_jsonl_sha256'],hashlib.sha256((HERE/'raw_model_comparison_20261024.jsonl').read_bytes()).hexdigest())
 def test_human_distinct_document_scope_not_flattened(self):
  self.assertEqual(self.o['metrics']['gold_counts'],{'Entailment':2,'Contradiction':2,'NotMentioned':2})
  self.assertEqual(len({x['doc_id'] for x in self.o['rows']}),6)
  self.assertEqual({x['doc_id'] for x in self.o['rows']},{x['id'] for x in self.f['cases']})
 def test_no_perfect_gold_drs_or_deleted_model_errors(self):
  original={x['id']:x for x in self.f['cases']}
  for case in self.o['rows']:
   self.assertEqual(case['source_doc_sha256'],original[case['doc_id']]['doc_text_sha256'])
   self.assertIn(case['label_human'],('Entailment','Contradiction','NotMentioned'))
  self.assertEqual(self.o['metrics']['valid_frames'],sum(bool(r['frame_valid']) for r in self.p))
if __name__=='__main__':unittest.main()
