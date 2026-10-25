"""Read-only audit of ORIGINAL KMD source parsing, PMB text and raw model outputs.

No model generation, no semantic repairs, no gold fixture changes. Compares
original JSON-schema validity separately from canonical KMD semantic validity.
"""
from pathlib import Path
import hashlib,json,unittest
import jsonschema
from knowmoredirt.model_planner import build_chunk_drs_prompt,chunk_drs_json_schema,_validate_chunk_drs_payload
HERE=Path(__file__).resolve().parent
SRC=Path('/data/tmp/kmd-pmb-eval-20261024/data/pmb-3.0.0/gold/test.txt.raw')
FREEZE=HERE/'frozen_real_pmb_for_kmd_v5_20261025.json'
RAW=HERE/'actual_kmd_v5_qwen_model_raw_20261025.jsonl'
SCORED=HERE/'original_kmd_schema_model_validity_20261025.json'
class OriginalKmdRealSourceFeasibility(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.f=json.loads(FREEZE.read_text());cls.a=[json.loads(z) for z in RAW.read_text().splitlines() if z];cls.s=json.loads(SCORED.read_text())
 def test_original_human_written_sources_and_selection_frozen(self):
  self.assertEqual(hashlib.sha256(FREEZE.read_bytes()).hexdigest(),self.s['frozen_input_sha256'])
  self.assertEqual(hashlib.sha256(SRC.read_bytes()).hexdigest(),self.f['source_test_text_sha256'])
  original=SRC.read_text().splitlines()
  self.assertEqual(len(self.f['cases']),7)
  self.assertEqual({x['id'] for x in self.f['cases']},{x['source_id'] for x in self.a})
  for x in self.f['cases']:
   self.assertEqual(original[x['sentence_index']],x['sentence'])
   self.assertEqual(hashlib.sha256(x['sentence'].encode()).hexdigest(),x['sentence_sha256'])
 def test_raw_model_once_and_original_validator_catches_all(self):
  self.assertEqual(self.s['original_predictions_sha256'],hashlib.sha256(RAW.read_bytes()).hexdigest())
  by={x['source_id']:x for x in self.a}
  for x in self.f['cases']:
   rec=by[x['id']];raw=json.loads(rec['raw_model_output']);src=x['sentence']
   schema=chunk_drs_json_schema(source_id=x['id'],include_auxiliary_fields=False,constrain_stable_ids=True)
   self.assertEqual(rec['original_production_schema_sha256'],hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest())
   self.assertEqual(rec['production_prompt_sha256'],hashlib.sha256(build_chunk_drs_prompt(src,rel_path=x['id']).encode()).hexdigest())
   self.assertEqual(list(jsonschema.Draft202012Validator(schema).iter_errors(raw)),[])
   v=_validate_chunk_drs_payload(raw,src)
   self.assertFalse(v['schema_valid'])
   self.assertFalse(rec['accepted_source_grounded_drs'])
   self.assertTrue(v['errors'])
   self.assertEqual(rec['finish_reason'],'stop')
 def test_canonical_json_schema_is_not_a_fact_grounding_certificate(self):
  s=self.s['summary']
  self.assertEqual((s['case_count'],s['complete_JSON'],s['canonical_json_schema_valid'],s['KMD_validated_and_grounded']),(7,7,7,0))
  self.assertEqual((s['root_asserted_box_only'],s['non_asserted_scope_box_present'],s['has_source_grounding_failure'],s['has_kmd_schema_errors']),(7,0,5,7))
  self.assertEqual(s['failed_or_truncated_generation'],0)
  self.assertEqual(round(s['total_cpu_seconds'],2),427.17)
 def test_time_grounding_failure_persisted_as_negative_evidence(self):
  claims=self.s['per_source_cases']
  self.assertEqual(len([x for x in claims if x['kmd_source_grounding_failures']]),5)
  self.assertTrue(all(any('2023-01-01' in fail for fail in x['kmd_source_grounding_failures']) for x in claims if x['kmd_source_grounding_failures']))
  self.assertTrue(all(x['root_asserted_only'] and not x['non_asserted_subordinate_scope_box_present'] for x in claims))
if __name__=='__main__':unittest.main()
