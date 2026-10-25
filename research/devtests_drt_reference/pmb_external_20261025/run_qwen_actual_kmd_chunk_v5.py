"""Bounded model extraction through real KMD v5 chunk prompt/schema/validator.

This is an independent real-source production-SCHEMA feasibility study on an
isolated small CPU model. It is NOT actual production KMD inference server or a
full scientific reasoning demonstration. No original human DRS gold is given
to the model, no repairs to semantics, no retries of finished predictions.
"""
import hashlib,json,time,resource
from pathlib import Path
from llama_cpp import Llama,LlamaGrammar
from knowmoredirt.model_planner import build_chunk_drs_prompt,chunk_drs_json_schema,_validate_chunk_drs_payload
HERE=Path(__file__).parent
FREEZE=HERE/'frozen_real_pmb_for_kmd_v5_20261025.json'
OUT=HERE/'actual_kmd_v5_qwen_model_raw_20261025.jsonl'
MODEL=Path('/data/tmp/kmd-qwen-small-isolated-20261024/qwen2.5-1.5b-instruct-q4_k_m.gguf')
F_SHA='4f500add47c30aa0b0463772a33b747bf909e1332fe9290a312d9698cb178d1b'
MAX_TOKENS=850
N_CTX=3072

def compat(o):
 # Installed llama-cpp-python grammar compiler does not support (?:) group
 # syntax in Python regular expressions. Rewrite ONLY noncapturing to capturing;
 # the accepted underlying string language is the same. KMD schema itself,
 # original parser and production validation remain byte-identical.
 if isinstance(o,dict):return {k:(v.replace('(?:','(') if k=='pattern' and isinstance(v,str) else compat(v)) for k,v in o.items()}
 if isinstance(o,list):return [compat(x) for x in o]
 return o

def one(llm,case):
 source=case['sentence'];rel_path=case['id']
 prompt=build_chunk_drs_prompt(source,rel_path=rel_path)
 schema=chunk_drs_json_schema(source_id=rel_path,include_auxiliary_fields=False,constrain_stable_ids=True)
 grammar_source=compat(schema)
 result={'source_id':rel_path,'source_sentence_sha256':case['sentence_sha256'],
   'source_id_category':case['category_from_source_cue'],
   'production_model_planner_unchanged':True,'production_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
   'original_production_schema_sha256':hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest(),
   'grammar_compat_sha256':hashlib.sha256(json.dumps(grammar_source,sort_keys=True).encode()).hexdigest(),
   'json_grammar_compatibility_change':'(?: -> ( only inside schema pattern regex; no other structural edits',
   'finish_reason':None,'source_only_category_not_human_gold':case['category_from_source_cue'],
   'raw_model_output':'','accepted_source_grounded_drs':False,'validation':None,'errors':[]}
 started=time.monotonic()
 try:
  grammar=LlamaGrammar.from_json_schema(json.dumps(grammar_source),verbose=False)
  res=llm(prompt,max_tokens=MAX_TOKENS,temperature=0,top_p=1,echo=False,grammar=grammar)
  raw=res['choices'][0]['text'];result['raw_model_output']=raw
  result['finish_reason']=res['choices'][0].get('finish_reason')
  result['tokens']=res.get('usage',{})
  try: parsed=json.loads(raw);result['complete_valid_json']=True
  except Exception as e:
   result['complete_valid_json']=False
   result['errors'].append({'stage':'json','type':type(e).__name__,'description':str(e)[:250]})
   parsed=None
  if isinstance(parsed,dict):
   v=_validate_chunk_drs_payload(parsed,source)
   result['validation']=v
   result['accepted_source_grounded_drs']=bool(v.get('schema_valid') and not v.get('grounding_failures') and isinstance(parsed.get('drs'),dict) and parsed['drs'].get('conditions'))
   drs=parsed.get('drs',{});result['extracted_drs_summary']={k:len(drs.get(k,[])) for k in ('boxes','referents','conditions','identity_hypotheses','temporal_records')}
   result['box_kinds']=[str(x.get('kind')) for x in drs.get('boxes',[]) if isinstance(x,dict)]
   result['condition_predicates']=[str(x.get('predicate')) for x in drs.get('conditions',[]) if isinstance(x,dict)]
 except Exception as e:
  result['errors'].append({'stage':'driver','type':type(e).__name__,'description':str(e)[:500]})
 result['elapsed_seconds']=round(time.monotonic()-started,2)
 return result

def main():
 assert hashlib.sha256(FREEZE.read_bytes()).hexdigest()==F_SHA
 frozen=json.loads(FREEZE.read_text());cases=frozen['cases'];assert len(cases)==7
 done=[json.loads(x) for x in OUT.read_text().splitlines() if x] if OUT.exists() else []
 completed={x['source_id'] for x in done};assert len(completed)==len(done)
 print('EXISTING_CASES',len(completed),'REMAIN',len(cases)-len(completed),flush=True)
 if len(completed)==len(cases):return
 model=Llama(model_path=str(MODEL),n_ctx=N_CTX,n_threads=2,n_batch=64,n_gpu_layers=0,chat_format='chatml',seed=42,verbose=False)
 for case in cases:
  if case['id'] in completed:continue
  assert hashlib.sha256(case['sentence'].encode()).hexdigest()==case['sentence_sha256']
  result=one(model,case)
  result['model']='Qwen/Qwen2.5-1.5B-Instruct-GGUF@91cad51170dc346986eccefdc2dd33a9da36ead9 q4_k_m; CPU 2 threads n_ctx='+str(N_CTX)+' max_completion_tokens='+str(MAX_TOKENS)
  with OUT.open('a') as fp:
   fp.write(json.dumps(result,ensure_ascii=False,default=str)+'\n');fp.flush()
  completed.add(case['id'])
  print('CASE',len(completed),'of',len(cases),case['category_from_source_cue'],'DONE',case['id'],
    'accepted',result['accepted_source_grounded_drs'],'json',result.get('complete_valid_json'),
    'finish',result['finish_reason'],'condition_count',(result.get('extracted_drs_summary') or {}).get('conditions'),
    'box_kind',result.get('box_kinds'),'error',result.get('errors')[:1],'sec',result['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
