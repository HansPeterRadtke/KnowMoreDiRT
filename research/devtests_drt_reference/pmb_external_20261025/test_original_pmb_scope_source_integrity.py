"""External semantic gold/data integrity only; no model calls or gold rewriting."""
import hashlib,json,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=Path('/data/tmp/kmd-pmb-eval-20261024/data/pmb-3.0.0/gold')
MODEL=Path('/data/tmp/kmd-pmb-reference-20261024/output/pmb-3.0.0/en')

class PublishedGoldSemanticContentIntegrity(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.a=json.loads((HERE/'original_pmb_scoped_predicate_content_20261025.json').read_text())
  cls.b=json.loads((HERE/'original_pmb_implication_attribution_scope_audit_20261025.json').read_text())
  cls.c=json.loads((HERE/'pmb_named_referents_in_scoped_events_20261025.json').read_text())
  cls.d=json.loads((HERE/'drs_semantic_necessary_conditions_summary_20261025.json').read_text())
  cls.e=json.loads((HERE/'original_pmb_semantic_scope_error_mechanisms_20261025.json').read_text())
 def test_original_external_gold_neural_parse_and_raw_sentences_preserved(self):
  for split in ('test','dev'):
   source={'gold':BASE/(split+'.txt'),'raw':BASE/(split+'.txt.raw'),'pred':MODEL/split/'bert_char_1enc/output1.txt'}
   for key,p in source.items():
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    self.assertEqual(digest,self.a['results'][split]['input_sha256'][key])
   self.assertEqual(self.a['results'][split]['sentences'],self.b['splits'][split]['sentences'])
 def test_true_negation_scope_content_must_not_be_confused_with_operator_marker(self):
  x=self.d['splits']['test']['negation'];y=self.d['splits']['dev']['negation']
  self.assertEqual([x[k]['correct'] for k in ('detected','same_scoped_predicate_lemmas','same_scoped_predicate_lemmas_and_senses')],[79,65,54])
  self.assertEqual([y[k]['correct'] for k in ('detected','same_scoped_predicate_lemmas','same_scoped_predicate_lemmas_and_senses')],[70,60,41])
  self.assertGreater(x['detected']['fraction'],x['same_scoped_predicate_lemmas']['fraction'])
 def test_conditional_requires_both_premise_and_consequence(self):
  x=self.d['splits']['test']['if_then'];y=self.d['splits']['dev']['if_then']
  self.assertEqual([x[k]['correct'] for k in ('both_condition_and_consequence_operators','exact_predicate_lemmas_on_both_sides','exact_predicate_lemmas_and_senses_on_both_sides')],[25,19,16])
  self.assertEqual([y[k]['correct'] for k in ('both_condition_and_consequence_operators','exact_predicate_lemmas_on_both_sides','exact_predicate_lemmas_and_senses_on_both_sides')],[31,22,14])
 def test_named_roles_indeed_misbind_mary_inside_negated_hit(self):
  cases=self.c['splits']['test']['operators']['NEGATION']['individual_evidence_errors']
  x=next(y for y in cases if y['index']==624)
  self.assertTrue(any(y[0]=='hit' and y[2]=='Destination' and y[3]=='mary' for y in x['gold_named_referent_links']))
  self.assertTrue(any(y[0]=='hit' and y[2]=='Agent' and y[3]=='mary' for y in x['generated_named_referent_links']))
  self.assertNotEqual(x['gold_named_referent_links'],x['generated_named_referent_links'])
  self.assertEqual(self.c['splits']['test']['operators']['NEGATION']['full_named_predicate_role_lemma_exact'],12)
 def test_errors_have_original_scoped_lexical_omissions_and_not_just_wrong_box(self):
  x=self.e['splits']['test']['operators']['NEGATION']
  self.assertEqual(x['gold_source_sentences'],79)
  self.assertEqual(x['nonexclusive_error_cause_counts']['exact_gold_lemma_scope'],65)
  self.assertEqual(x['nonexclusive_error_cause_counts']['at_least_one_gold_lemma_not_extracted_anywhere'],14)
  self.assertEqual(self.e['splits']['test']['operators']['CONDITION']['nonexclusive_error_cause_counts']['gold_lemma_present_but_outside_correct_scoped_box'],6)
 def test_dev_and_test_are_not_represented_as_perfectly_disjoint(self):
  a=set((BASE/'test.txt.raw').read_text().splitlines())
  b=set((BASE/'dev.txt.raw').read_text().splitlines())
  self.assertEqual(a&b,{'Dinosaur remains were found in Asia.'})
if __name__=='__main__':unittest.main()
