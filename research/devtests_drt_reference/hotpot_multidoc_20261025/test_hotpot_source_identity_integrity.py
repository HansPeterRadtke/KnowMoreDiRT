"""Read-only invariants for original human gold and frozen retrieval methods."""
from pathlib import Path
import hashlib,json,unittest
HERE=Path(__file__).parent
ORIGINAL=Path('/data/tmp/kmd-hotpot-original-distractor-20261025.json')
PRED=HERE/'frozen_unlabelled_retrieval_20261025.json'
SCORE=HERE/'hotpot_external_multidoc_gold_results_20261025.json'
AUDIT=HERE/'title_identity_source_link_audit_20261025.json'

class OriginalExternalSourceAndGold(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.p=json.loads(PRED.read_text());cls.s=json.loads(SCORE.read_text());cls.a=json.loads(AUDIT.read_text())
 def test_original_external_source_version_unchanged(self):
  digest=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
  self.assertEqual(digest,'4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea')
  self.assertEqual(digest,self.p['source_sha256'])
  self.assertEqual(digest,self.s['source_sha256'])
  self.assertEqual(self.s['frozen_predictions_sha256'],hashlib.sha256(PRED.read_bytes()).hexdigest())
 def test_complete_human_annotated_two_article_gold(self):
  allrows=self.s['cases']
  self.assertEqual((len(allrows),self.s['scores']['bridge']['n'],self.s['scores']['comparison']['n']),(7405,5918,1487))
  self.assertTrue(all(len(x['gold_doc_titles'])==2 for x in allrows))
  self.assertEqual(self.s['scores']['overall']['ranked_source']['fulltext_bm25']['2']['both_docs'],2729)
  self.assertEqual(self.s['scores']['overall']['ranked_source']['linked_secondhop']['2']['both_docs'],3211)
 def test_dissimilar_task_types_and_measured_link_regressions(self):
  b=self.s['scores']['bridge'];c=self.s['scores']['comparison']
  self.assertEqual((b['ranked_source']['fulltext_bm25']['2']['both_docs'],b['ranked_source']['linked_secondhop']['2']['both_docs']),(2194,2692))
  self.assertEqual((c['ranked_source']['fulltext_bm25']['2']['both_docs'],c['ranked_source']['linked_secondhop']['2']['both_docs']),(535,519))
  self.assertGreater(b['linked_vs_fulltext_pair_at_2']['difference'],0)
  self.assertLess(c['linked_vs_fulltext_pair_at_2']['difference'],0)
  self.assertEqual(self.a['total_differing_top2_doc_choices'],990)
  indexed={(x['type'],x['outcome'],x['evidence_type']):x['count'] for x in self.a['source_mention_outcome_counts']}
  self.assertEqual(indexed['bridge','win','exact_source_title_mention'],605)
  self.assertEqual(indexed['bridge','loss','exact_source_title_mention'],107)
 def test_source_excerpt_sample_offsets_original_and_in_scope(self):
  docs={x['_id']:x for x in json.loads(ORIGINAL.read_text())}
  for case in self.a['sample_link_cases']:
   p={title:sentences for title,sentences in docs[case['uid']]['context']}
   for occurrence in case['source_mention_spans']:
    line=p[occurrence['source_title']][occurrence['source_sentence_index']]
    self.assertEqual(hashlib.sha256(line.encode()).hexdigest(),occurrence['source_sentence_sha256'])
    self.assertIn(occurrence['matched_title'].lower(),line.lower())
 def test_literal_title_link_has_measured_ceiling_not_coreference_success(self):
  x=json.loads((HERE/'literal_identity_reference_ceiling_20261025.json').read_text())
  c=x['counts']
  self.assertEqual((c['first_ranked_article_right'],c['first_ranked_article_wrong']),(5165,753))
  self.assertEqual((c['gold_other_article_name_explicitly_in_first_source'],c['gold_other_article_name_not_literal_in_first_source']),(1080,4085))
  self.assertEqual((c['literal_present_linked_selected_gold'],c['literal_present_linked_missed_gold']),(1008,72))
  self.assertEqual((c['literal_absent_linked_selected_gold'],c['literal_absent_linked_missed_gold']),(1684,2401))
  self.assertAlmostEqual(x['fraction_gold_other_title_not_literal_given_first_gold'],4085/5165,places=5)
 def test_musique_unscored_source_only_snapshot_does_not_claim_performance(self):
  path=HERE/'frozen_untuned_musique_rankings_20261025.json'
  src=Path('/data/tmp/kmd-musique-official-answerable-dev-20261025.jsonl')
  snap=json.loads(path.read_text())
  self.assertEqual(len(snap['predictions']),snap['n'])
  self.assertEqual(snap['n'],2417)
  self.assertEqual(snap['source_original_MuSiQue_sha256'],hashlib.sha256(src.read_bytes()).hexdigest())
  self.assertNotIn('gold',snap)
  self.assertNotIn('results',snap)
  self.assertTrue(all('is_supporting' not in str(x) for x in snap['predictions']))
if __name__=='__main__':unittest.main()
