"""Only independently completed original source and frozen parser result checks."""
from pathlib import Path
import hashlib,json,unittest
HERE=Path(__file__).parent
GOLD=Path('/data/tmp/kmd-pmb-eval-20261024/data/pmb-3.0.0/gold')
M=Path('/data/tmp/kmd-pmb-reference-20261024/output/pmb-3.0.0/en')
class OriginalScopedNegationComparison(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.sm=json.loads((HERE/'spacy_dependency_vs_published_drs_gold_results_20261025.json').read_text())
  cls.lg=json.loads((HERE/'spacy_lg_dependency_vs_published_drs_gold_results_20261025.json').read_text())
  cls.fsm=json.loads((HERE/'frozen_spacy_dependency_predictions_20261025.json').read_text())
  cls.flg=json.loads((HERE/'frozen_spacy_lg_dependency_predictions_20261025.json').read_text())
 def test_original_author_gold_and_model_bytes_unchanged(self):
  for split in ('test','dev'):
   for f in [GOLD/(split+'.txt'),GOLD/(split+'.txt.raw'),M/split/'bert_char_1enc/output1.txt']:
    self.assertTrue(f.is_file())
   self.assertEqual(self.fsm['source_sha256'][split],hashlib.sha256((GOLD/(split+'.txt.raw')).read_bytes()).hexdigest())
   self.assertEqual(self.flg['source_sha256'][split],self.fsm['source_sha256'][split])
   self.assertEqual(self.fsm['splits'][split]['n'],self.flg['splits'][split]['n'])
 def test_frozen_source_only_predictions_cannot_use_gold_annotations(self):
  self.assertEqual(self.sm['source_frozen_sha256'],hashlib.sha256((HERE/'frozen_spacy_dependency_predictions_20261025.json').read_bytes()).hexdigest())
  self.assertEqual(self.lg['source_frozen_sha256'],hashlib.sha256((HERE/'frozen_spacy_lg_dependency_predictions_20261025.json').read_bytes()).hexdigest())
  for x in self.flg['splits']['test']['predictions']:
   self.assertNotIn('gold',x)
   self.assertNotIn('human',x)
 def test_original_test_scoped_verb_adj_gold_versus_two_source_parsers(self):
  a=self.sm['results']['test']['strict_on_gold_positive']
  b=self.lg['results']['test']['strict_on_gold_positive']
  self.assertEqual(self.sm['results']['test']['gold_sentences_with_a_negated_verb_or_adjective'],77)
  self.assertEqual([a['neural_DRS']['exact_gold_lemma_set'],a['source_only_dependency']['exact_gold_lemma_set'],b['source_only_dependency']['exact_gold_lemma_set']],[65,56,60])
  self.assertEqual(b['paired_exact_source']['DRS_wins'],11)
  self.assertEqual(b['paired_exact_source']['dependency_wins'],6)
  self.assertLess(b['paired_exact_source']['paired_bootstrap95'][0],0)
  self.assertGreater(b['paired_exact_source']['paired_bootstrap95'][1],0)
 def test_original_dev_heldout_but_not_independent_domain(self):
  a=self.lg['results']['dev']['strict_on_gold_positive']
  self.assertEqual(self.lg['results']['dev']['gold_sentences_with_a_negated_verb_or_adjective'],69)
  self.assertEqual((a['neural_DRS']['exact_gold_lemma_set'],a['source_only_dependency']['exact_gold_lemma_set']),(65,51))
 def test_source_only_correct_when_neural_drs_misrepresents_negation(self):
  x=next(z for z in self.sm['results']['test']['source_only_correct_but_drs_incorrect'] if z['index']==177)
  self.assertEqual(x['gold_verb_adj_negated_lemmas'],['pronounce'])
  self.assertEqual(x['dependency_predicted_lemmas'],['pronounce'])
  self.assertEqual(x['model_predicted_lemmas'],['be'])
if __name__=='__main__':unittest.main()
