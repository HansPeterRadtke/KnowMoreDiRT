"""Bounded 8-case real-article Qwen 1.5B provenance vs boxed-semantics comparison.

A. Direct source-aware claim assessment with source quote and title.
B. Source-only model-extracted DRS-inspired attributed conditions, then an
   independent second model call consumes ONLY the generated representation.
No gold used as model input. Boxes are a SMALL DRS-INSPIRED semantic frame, NOT
the full production KMD `chunk-drs-v5` schema. No tuning, retries or gold edits.
Uses idle Raspi CPU at low thread count, never shared server or GPU. Each fully
finished case is checkpointed immediately; never repeats completed inference.
"""
from __future__ import annotations
import hashlib,json,re,time,sys
from pathlib import Path
from llama_cpp import Llama
ROOT=Path(__file__).parent
FROZEN=ROOT/'frozen_model_science_20261024.json'
MODEL=Path('/data/tmp/kmd-qwen-small-isolated-20261024/qwen2.5-1.5b-instruct-q4_k_m.gguf')
CHK=ROOT/'results_checkpoint.jsonl'
SHA='d707abc529f2d6be734b6d570069077c1518e38809b908e5821a8465919ca0ba'
VERSION='science-source-direct-vs-source-only-boxes-v1'

def call(llm,system,user,max_tokens):
 start=time.monotonic()
 try:
  z=llm.create_chat_completion(messages=[{'role':'system','content':system},{'role':'user','content':user}],max_tokens=max_tokens,temperature=0,top_p=1,response_format={'type':'json_object'})
  raw=z['choices'][0]['message']['content'] or ''
  try: payload=json.loads(raw)
  except Exception:payload=None
  return {'raw':raw,'parsed':payload,'elapsed_seconds':round(time.monotonic()-start,2),'usage':z.get('usage',{}),'finish_reason':z['choices'][0].get('finish_reason')}
 except Exception as e:
  return {'raw':'','parsed':None,'error':str(e)[:250],'elapsed_seconds':round(time.monotonic()-start,2),'usage':{},'finish_reason':'failed'}

def direct(llm,case):
 system=('You evaluate a claim ONLY from the supplied source text and its identified article. '
 'Output one JSON object exactly with keys "verdict" (SUPPORT, CONTRADICT, UNKNOWN) and "evidence_quote". '
 'Treat absence of proof as UNKNOWN; do not import general facts. Quote a CONTIGUOUS source substring only or empty if unknown. '
 'Preserve negation, author attribution, subjects, and conditional/hypothetical statements.')
 user=f"SOURCE PAPER: {case['source_title']}\nSOURCE SENTENCES:\n{case['source_evidence']}\nCLAIM: {case['claim']}\nJSON:"
 return call(llm,system,user,85)

def boxes(llm,case):
 system=('Convert only this quoted scientific source into a compact DRS-INSPIRED JSON discourse frame. '
 'Output keys: "source", "boxes", "conditions". '
 'source is the supplied article title. "boxes" is an array of at most 3 objects with keys "kind" '
 '(asserted/report/conditional/quoted/hypothetical), "time_scope" (source-supported string or empty). '
 '"conditions" is at most 3 objects with keys "subject", "predicate", "object", '
 '"polarity" (positive/negative/unknown), "box_index" (0-based index in boxes), '
 '"evidence_quote" (short exact contiguous substring from the original source). '
 'Represent facts attributed to the study rather than global proven truths. '
 'Do not see or answer any scientific user claim. Never invent measurements or source text. JSON only.')
 user=f"SOURCE PAPER: {case['source_title']}\nSOURCE TEXT:\n{case['source_evidence']}\nOUTPUT JSON:"
 return call(llm,system,user,195)

def frame_ok(obj,case):
 if not isinstance(obj,dict):return False,'not_json_object'
 if not isinstance(obj.get('boxes'),list) or not isinstance(obj.get('conditions'),list):return False,'missing_box_or_conditions'
 if not (0<len(obj['boxes'])<=3 and 0<len(obj['conditions'])<=3):return False,'wrong_box_condition_count'
 for box in obj['boxes']:
  if not isinstance(box,dict) or box.get('kind') not in ('asserted','report','conditional','quoted','hypothetical'):return False,'malformed_box_kind'
 for cond in obj['conditions']:
  if not isinstance(cond,dict):return False,'malformed_condition'
  if cond.get('polarity') not in ('positive','negative','unknown'):return False,'malformed_polarity'
  idx=cond.get('box_index')
  if type(idx) is not int or idx<0 or idx>=len(obj['boxes']):return False,'unbound_box'
  q=cond.get('evidence_quote')
  if not isinstance(q,str) or not q or q not in case['source_evidence']:return False,'quote_not_exact_source_substring'
 return True,'grounded_box_structure'

def judge(llm,case,frame):
 system=('Using only the PRE-EXTRACTED attributed propositions and their stated scope, judge whether the claim is '
 'supported, contradicted or not established. The source was not passed directly: missing information is UNKNOWN. '
 'Output one JSON object with key "verdict" (SUPPORT/CONTRADICT/UNKNOWN) and "premise_quote" '
 '(a short exact substring from one evidence_quote in the box frames). Preserve negation and same-entity scope.')
 return call(llm,system,f"SOURCE FRAMES: {json.dumps(frame,ensure_ascii=False)}\nCLAIM: {case['claim']}\nJSON:",65)

def main():
 if hashlib.sha256(FROZEN.read_bytes()).hexdigest()!=SHA:raise RuntimeError('source freeze changed')
 freeze=json.loads(FROZEN.read_text());cases=freeze['cases'];assert len(cases)==8
 assert MODEL.exists() and MODEL.stat().st_size==1117320736
 completed=[]
 if CHK.exists():
  completed=[json.loads(x) for x in CHK.read_text().splitlines() if x]
 done={x['uid'] for x in completed}
 assert len(done)==len(completed)
 print('PREVIOUS_COMPLETE_CASES',len(done),'REMAIN',len(cases)-len(done),flush=True)
 if len(done)==len(cases):print('ALL_ALREADY_COMPLETED_NO_REINFERENCE',flush=True);return
 start=time.monotonic()
 llm=Llama(model_path=str(MODEL),n_ctx=1900,n_threads=2,n_batch=64,n_gpu_layers=0,chat_format='chatml',seed=42,verbose=False)
 for case in cases:
  if case['uid'] in done:continue
  if hashlib.sha256(case['source_evidence'].encode()).hexdigest()!=case['source_sha256']:raise RuntimeError('source corrupted')
  a=direct(llm,case)
  b=boxes(llm,case)
  valid,reason=frame_ok(b.get('parsed'),case)
  c=judge(llm,case,b['parsed']) if valid else {'raw':'','parsed':{'verdict':'UNKNOWN'},'skipped':'invalid_extracted_frame','elapsed_seconds':0,'usage':{},'finish_reason':'skipped'}
  rec={'uid':case['uid'],'source_sha256':case['source_sha256'],'model_revision':'Qwen/Qwen2.5-1.5B-Instruct-GGUF@91cad51170dc346986eccefdc2dd33a9da36ead9',
       'runtime':'llama-cpp-python 0.3.9 CPU nthreads2 nctx1900 seed42 temperature0 JSON grammar','method_version':VERSION,
       'direct':a,'box_extraction':b,'box_accepted':valid,'box_reason':reason,'box_judgment':c}
  with CHK.open('a',encoding='utf-8') as f:
   f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush()
  done.add(case['uid'])
  print('COMPLETED',len(done),'/',len(cases),case['uid'],'source_chars',len(case['source_evidence']),
        'direct',a.get('parsed',{}).get('verdict') if isinstance(a.get('parsed'),dict) else 'INVALID',
        'box',reason,'box_outcome',c.get('parsed',{}).get('verdict') if isinstance(c.get('parsed'),dict) else 'INVALID',
        'case_seconds',round(sum(x.get('elapsed_seconds',0) for x in (a,b,c)),1),flush=True)
 print('ALL_CASES_DONE',len(done),'ELAPSED_SECONDS',round(time.monotonic()-start,1),flush=True)
if __name__=='__main__':main()
