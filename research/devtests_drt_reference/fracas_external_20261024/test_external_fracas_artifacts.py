"""Read-only artifact-integrity checks; no LLM calls or rebenchmarking."""
import csv,hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class FraCasResearchArtifacts(unittest.TestCase):
    def test_unmodified_external_frozen_split_bytes(self):
        self.assertEqual(hashlib.sha256((ROOT/'original_dev.csv').read_bytes()).hexdigest(),'754bd62c3d2db209c560ba7d32a360182c9005f5f468b1438eade91262706a1b')
        self.assertEqual(hashlib.sha256((ROOT/'original_train.csv').read_bytes()).hexdigest(),'6194f7455a084e2bad8e4df4d82e24d206b938485f70365b780e153fa15e3c9d')
        self.assertTrue((ROOT/'LICENSE').exists())

    def test_external_label_distribution_is_retained(self):
        with (ROOT/'original_dev.csv').open(newline='',encoding='utf-8') as source:
            rows=list(csv.DictReader(source))
        self.assertEqual(len(rows),70)
        from collections import Counter
        self.assertEqual(dict(Counter(x['label'] for x in rows)),{'entailment':34,'neutral':28,'contradiction':8})
        self.assertEqual(sum(not x['hypothesis'] for x in rows),1)

    def test_finished_lexical_evaluation_is_not_mutated(self):
        r=json.loads((ROOT/'lexical_baseline_results.json').read_text())
        self.assertEqual((r['records'],r['exact_full_premise_correct'],r['all_neutral_correct']),(70,32,28))
        self.assertEqual(r['source_sha256'],hashlib.sha256((ROOT/'original_dev.csv').read_bytes()).hexdigest())

    def test_production_schema_pilot_failure_is_preserved(self):
        a=json.loads((ROOT/'kmd_production_drs_cpu_train_pilot_results.json').read_text())
        b=json.loads((ROOT/'kmd_production_drs_cpu_train_compat_results.json').read_text())
        self.assertFalse(a['accepted']);self.assertFalse(b['accepted'])
        self.assertEqual(a['errors'][0]['type'],'AssertionError')
        self.assertEqual(b.get('finish_reason'),'length')
        self.assertEqual(b.get('valid_json'),False)
        self.assertEqual(a['train_sha256'],b['train_sha256'])

    def test_secondary_direct_nli_is_an_invalid_probe_not_model_accuracy(self):
        r=json.loads((ROOT/'secondary_cpu_nli_results.json').read_text())
        self.assertEqual((r['total'],r['invalid_outputs']),(24,24))
        self.assertTrue(all(not c['raw_model_output'] for c in r['results']))
        self.assertEqual(r['fixed_sample']['dataset_sha256'],hashlib.sha256((ROOT/'original_dev.csv').read_bytes()).hexdigest())

if __name__=='__main__':unittest.main()
