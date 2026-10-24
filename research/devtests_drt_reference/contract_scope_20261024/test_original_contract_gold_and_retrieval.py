"""Integrity controls for independent real NDA scope labels and rank output."""
import hashlib,json,unittest
from pathlib import Path
from collections import Counter
HERE=Path(__file__).parent
ORIGINAL=Path('/data/tmp/kmd-contractnli-20261024/contract-nli/test.json')
FROZEN=HERE/'frozen_contract_source_scope_20261024.json'
OUT=HERE/'full_test_lexical_neighborhood_20261024.json'
class OriginalGoldContractControls(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.d=json.loads(ORIGINAL.read_text());cls.r=json.loads(OUT.read_text());cls.f=json.loads(FROZEN.read_text())
 def test_original_external_contract_sha_and_author_annotation(self):
  self.assertEqual(hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),'460267b56052a2dc5aead98eb35eadef9e6734d5723d37b4a9790e410f812387')
  self.assertEqual((len(self.d['documents']),len(self.d['labels'])),(123,17))
  self.assertEqual(self.r['cases'],2091)
 def test_evidence_class_separation_and_span_counts(self):
  n=self.r['label_buckets']
  self.assertEqual((n['Entailment']['queries'],n['Contradiction']['queries'],n['NotMentioned']['queries']),(968,220,903))
  self.assertEqual((n['Entailment']['topk']['3']['ranked_any_gold'],n['Entailment']['topk']['3']['expanded_any_gold']),(435,590))
  self.assertEqual((n['Contradiction']['topk']['3']['ranked_any_gold'],n['Contradiction']['topk']['3']['expanded_any_gold']),(59,109))
  self.assertEqual(n['NotMentioned']['with_annotated_evidence'],0)
 def test_same_identical_hypothesis_across_different_human_labels(self):
  labels=Counter(x['annotation_sets'][0]['annotations']['nda-20']['choice'] for x in self.d['documents'])
  self.assertEqual(dict(labels),{'Entailment':33,'Contradiction':30,'NotMentioned':60})
  self.assertEqual(self.f['hypothesis_id'],'nda-20')
  self.assertEqual(len(self.f['cases']),6)
  # Freeze deliberately contains NO original author gold labels or gold evidence indices.
  for entry in self.f['cases']:
   self.assertNotIn('gold_class_not_shown_to_model',entry)
   self.assertNotIn('annotation_sets',entry)
   self.assertNotIn('gold_evidence_indices',entry)
   self.assertEqual(hashlib.sha256(entry['source_excerpt'].encode()).hexdigest(),entry['source_excerpt_sha256'])
if __name__=='__main__':unittest.main()
