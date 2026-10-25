"""Read-only evidence/answer integrity for completed 36-case Qwen study.

No inference, no gold repair, no prompt selection, no transfer scoring.
"""
from pathlib import Path
import hashlib,json,unittest
from collections import Counter
HERE=Path(__file__).parent
SOURCE=Path('/data/tmp/kmd-hotpot-original-distractor-20261025.json')
FROZEN=HERE/'frozen_source_only_model_qa_20261025.json'
RAW=HERE/'raw_source_only_qa_qwen_20261025.jsonl'
SCORED=HERE/'source_grounded_model_qa_results_20261025.json'
class OriginalAnswerAndSourceIntegrity(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.f=json.loads(FROZEN.read_text());cls.s=json.loads(SCORED.read_text())
  cls.o=[json.loads(x) for x in RAW.read_text().splitlines() if x]
 def test_original_wikipedia_and_unlabelled_predictions_unchanged(self):
  self.assertEqual(self.f['source_sha256'],hashlib.sha256(SOURCE.read_bytes()).hexdigest())
  self.assertEqual(self.s['original_source_sha256'],self.f['source_sha256'])
  self.assertEqual(self.s['frozen_input_sha256'],hashlib.sha256(FROZEN.read_bytes()).hexdigest())
  self.assertEqual(self.s['verbatim_model_predictions_sha256'],hashlib.sha256(RAW.read_bytes()).hexdigest())
 def test_all_cases_exact_two_methods_and_no_partial_gold(self):
  self.assertEqual(len(self.f['records']),36)
  self.assertEqual(len(self.o),72)
  self.assertEqual(len({(x['uid'],x['method']) for x in self.o}),72)
  self.assertEqual(self.s['buckets']['overall']['n'],36)
  counts=Counter(x['rank_change_group'] for x in self.f['records'])
  self.assertEqual(dict(counts),{'bridge_changed':16,'bridge_same':8,'comparison_changed':8,'comparison_same':4})
 def test_each_output_corresponds_to_frozen_model_input_and_external_source_titles(self):
  lookup={x['uid']:x for x in self.f['records']}
  for x in self.o:
   case=lookup[x['uid']];method=x['method']
   expected=hashlib.sha256(json.dumps({'case':x['uid'],'method':method,'docs':case['source_documents'][method]},sort_keys=True).encode()).hexdigest()
   self.assertEqual(x['input_sha256'],expected)
   self.assertEqual(len(case['source_documents'][method]),2)
 def test_gold_scoring_does_not_treat_copying_text_as_valid_reasoning(self):
  for case in self.s['cases']:
   for m,x in case['methods'].items():
    self.assertTrue(0<=x['f1']<=1)
    self.assertIn(x['em'],(0,1))
    if x['source_quote_exact']:
     self.assertTrue(x['source_title_provided'])
     self.assertTrue(x['source_quote'])
  self.assertIn('linked_minus_fulltext_em_paired',self.s['buckets']['overall'])
if __name__=='__main__':unittest.main()
