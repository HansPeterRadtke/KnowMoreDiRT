"""Read-only verification of published DRS output vs independent PMB gold."""
import hashlib,json,unittest
from pathlib import Path

HERE=Path(__file__).parent
G=Path('/data/tmp/kmd-pmb-eval-20261024/data/pmb-3.0.0/gold/test.txt')
P=Path('/data/tmp/kmd-pmb-reference-20261024/output/pmb-3.0.0/en/test/bert_char_1enc/output1.txt')
R=Path('/data/tmp/kmd-pmb-eval-20261024/data/pmb-3.0.0/gold/test.txt.raw')
class ExternalSourceGoldTests(unittest.TestCase):
    def test_original_frozen_external_source_hashes(self):
        x=json.loads((HERE/'pmb_semantic_scope_results_20261024.json').read_text())
        for f in [G,P,R]:self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),x['source_sha256'][f.name])
        self.assertEqual(x['total'],898)
    def test_automatic_negation_presence_measured_without_gold_overwrite(self):
        x=json.loads((HERE/'pmb_semantic_scope_results_20261024.json').read_text())
        m=x['models']['published_neural_DRS'];b=x['models']['lexical_unscoped_negation']
        self.assertEqual(x['gold_negation_count'],79)
        self.assertEqual((m['tp'],m['fp'],m['fn']),(79,1,0))
        self.assertEqual((b['tp'],b['fp'],b['fn']),(73,3,6))
        self.assertGreater(m['f1'],b['f1'])
    def test_posthoc_scope_diagnostic_has_counterexamples(self):
        x=json.loads((HERE/'pmb_operator_ablation_20261024_results.json').read_text())
        self.assertEqual(x['labels']['CONDITION']['gold_cases'],31)
        self.assertEqual(x['labels']['ATTRIBUTION']['gold_cases'],18)
        self.assertEqual(x['labels']['YearOfCentury']['gold_cases'],18)
        self.assertGreater(x['labels']['YearOfCentury']['source_lexical_baseline']['f1'],x['labels']['YearOfCentury']['model_generated_drs']['f1'])
        self.assertTrue(x['no_cross_document_inference'])
if __name__=='__main__':unittest.main()
