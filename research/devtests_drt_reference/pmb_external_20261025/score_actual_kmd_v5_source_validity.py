"""One-time audit of actual production KMD v5 generated JSON and source semantics.

Requires all seven original CPU predictions complete; no model call or repair.
Uses (i) canonical original JSON Schema, (ii) unmodified KMD validation, and
(iii) original PMB independently annotated source DRS operators as descriptive
semantic controls. They are not one-to-one KMD-vs-PMB box gold annotations.
"""
from pathlib import Path
from collections import Counter
import hashlib,json
import jsonschema
from knowmoredirt.model_planner import chunk_drs_json_schema,_validate_chunk_drs_payload
from operator_scope_predicate_audit import GROOT,SHA,blocks,clauses
HERE=Path(__file__).parent
F=HERE/'frozen_real_pmb_for_kmd_v5_20261025.json'
P=HERE/'actual_kmd_v5_qwen_model_raw_20261025.jsonl'
OUT=HERE/'original_kmd_schema_model_validity_20261025.json'
F_SHA='4f500add47c30aa0b0463772a33b747bf909e1332fe9290a312d9698cb178d1b'

def main():
 if OUT.exists():raise RuntimeError('Original v5 source analysis exists; do not rerun or overwrite')
 assert hashlib.sha256(F.read_bytes()).hexdigest()==F_SHA
 frozen=json.loads(F.read_text());cases=frozen['cases'];lines=P.read_text().splitlines()
 assert len(lines)==len(cases)==7, 'NO_PARTIAL_SCORE_UNTIL_ALL_MODELS_HAVE_FINISHED'
 raw=[json.loads(x) for x in lines];by={x['source_id']:x for x in raw};assert len(by)==7
 assert hashlib.sha256((GROOT/'test.txt.raw').read_bytes()).hexdigest()==SHA['test']['raw']
 assert hashlib.sha256((GROOT/'test.txt').read_bytes()).hexdigest()==SHA['test']['gold']
 gold=blocks(GROOT/'test.txt')
 records=[]
 for item in cases:
  x=by[item['id']]
  assert item['sentence_sha256']==x['source_sentence_sha256']
  text=item['sentence']
  orig=clauses(gold[item['sentence_index']]);ops=Counter(c[1] for c in orig if len(c)==3 and c[1] in ('NEGATION','CONDITION','CONSEQUENCE','ATTRIBUTION','POSSIBILITY','NECESSITY'))
  schema=chunk_drs_json_schema(source_id=item['id'],include_auxiliary_fields=False,constrain_stable_ids=True)
  assert x['original_production_schema_sha256']==hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest()
  assert hashlib.sha256(text.encode()).hexdigest()==item['sentence_sha256']
  parse=None;canon_errors=[];canonical_json_valid=False
  if x.get('complete_valid_json'):
   parse=json.loads(x['raw_model_output'])
   canonical_json_valid=True
   validator=jsonschema.Draft202012Validator(schema)
   canon_errors=[str(e.message)[:260] for e in validator.iter_errors(parse)]
   v=_validate_chunk_drs_payload(parse,text)
   assert v.get('schema_valid')==x['validation'].get('schema_valid')
  else: v={'schema_valid':False,'errors':['model_json_incomplete'], 'grounding_failures':[]}
  boxes=parse.get('drs',{}).get('boxes',[]) if isinstance(parse,dict) else []
  conds=parse.get('drs',{}).get('conditions',[]) if isinstance(parse,dict) else []
  ids=[c.get('box_id') for c in conds if isinstance(c,dict)]
  scoped=len(boxes)>1 and any(b.get('kind')!='asserted' for b in boxes if isinstance(b,dict))
  source_scope_needed=item['category_from_source_cue'] in ('negation','dream','reported_speech','conditional')
  records.append({'id':item['id'],'source_category_by_original_text_not_human_annotated':item['category_from_source_cue'],
    'source_sentence_index':item['sentence_index'],'source_hash':item['sentence_sha256'],
    'independent_original_pmb_gold_operator_count':dict(ops),
    'model_output_json_complete':bool(x.get('complete_valid_json')),
    'canonical_original_schema_errors':canon_errors[:12],
    'original_json_schema_valid':bool(canonical_json_valid and not canon_errors),
    'production_KMD_source_semantics_valid':bool(x['accepted_source_grounded_drs']),
    'kmd_schema_validation_error_codes':v.get('errors',[])[:25],
    'kmd_source_grounding_failures':v.get('grounding_failures',[])[:12],
    'root_asserted_only':len(boxes)==1 and boxes[0].get('kind')=='asserted' if boxes else False,
    'non_asserted_subordinate_scope_box_present':scoped,
    'root_condition_count':sum(k=='b0' for k in ids),
    'all_condition_count':len(conds),
    'predicates_model_generated':[str(c.get('predicate')) for c in conds if isinstance(c,dict)][:12],
    'model_box_kind':[str(b.get('kind')) for b in boxes if isinstance(b,dict)][:12],
    'finish_reason':x['finish_reason'],'inference_seconds':x['elapsed_seconds'],
    'raw_model_output_sha256':hashlib.sha256(x['raw_model_output'].encode()).hexdigest()})
 summary={'case_count':len(records),'complete_JSON':sum(x['model_output_json_complete'] for x in records),
  'canonical_json_schema_valid':sum(x['original_json_schema_valid'] for x in records),
  'KMD_validated_and_grounded':sum(x['production_KMD_source_semantics_valid'] for x in records),
  'root_asserted_box_only':sum(x['root_asserted_only'] for x in records),
  'non_asserted_scope_box_present':sum(x['non_asserted_subordinate_scope_box_present'] for x in records),
  'has_source_grounding_failure':sum(bool(x['kmd_source_grounding_failures']) for x in records),
  'has_kmd_schema_errors':sum(bool(x['kmd_schema_validation_error_codes']) for x in records),
  'failed_or_truncated_generation':sum(x['finish_reason']!='stop' for x in records),
  'total_cpu_seconds':round(sum(x['inference_seconds'] for x in records),2)}
 data={'source_gold_repo':'RikVN/DRS_parsing@190838f9adea2af616eb76156d2e66c8565a6c6e',
  'KMD_schema_and_prompt':'actual KMD chunk-drs-v5 schema and prompt, untouched as of origin/master:321cc018, validation _validate_chunk_drs_payload',
  'cpu_model':'Qwen2.5-1.5B-Instruct GGUF Q4_K_M, two CPU threads, n_ctx=3072, max completion=850, temp=0, local grammar compatibility (?: to ( only)',
  'frozen_input_sha256':F_SHA,'original_predictions_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),
  'evaluation':'Each original generation tested against original KMD JSON Schema via jsonschema Draft202012Validator and exact unchanged KMD source/DRS payload validator. Independent published PMB operator categories are descriptive only; no human semantic tag provided during inference.',
  'summary':summary,'per_source_cases':records,
  'limitations':['This local CPU small Qwen is NOT the intended large production KMD reasoning worker model; no shared inference server was used','Grammar conversion of noncapturing regex to capturing regex changes only grammar compiler compatibility and accepted string language not substantive semantics, but production server may use different sampling/backend','The seven original PMB natural utterances were selected by lexical category and frozen without gold structure, but sources are not pristine against prior research or pretraining','The PMB author DRS operator inventory and the project custom KMD box schema do NOT have identical clause structures; cannot calculate PMB truth-equivalence from KMD box names alone','Valid canonical JSON schema says types/IDs match regex; referent/entity grounding/box semantics must be checked separately, and false world assumptions are not independently validated just by a citation','No paired production KMD DRS answer accuracy vs provenance-neighbor baseline or new scientific discovery is measured']}
 OUT.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
 print('KMD_V5_VERIFIED_MODEL_FEASIBILITY',summary)
 for row in records:
  print(row['id'],row['source_category_by_original_text_not_human_annotated'],'JSON',row['model_output_json_complete'],'canonical_schema',row['original_json_schema_valid'],'KMD_accepted',row['production_KMD_source_semantics_valid'],
    'has_scoped_box',row['non_asserted_subordinate_scope_box_present'],
    'original_gold_operators',row['independent_original_pmb_gold_operator_count'],
    'errors',row['kmd_schema_validation_error_codes'][:4], 'grounding',row['kmd_source_grounding_failures'][:2])
if __name__=='__main__':main()
