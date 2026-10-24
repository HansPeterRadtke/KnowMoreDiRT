"""Automatic, locally heldout source scopes vs direct neighbor text, CPU only.

Model only sees published legal source text, source ID, and shared hypothesis.
Gold labels stay absent from inference. Real model-extracted typed proposition
is a SCOPE FRAME, not a full/KMD-compliant discourse representation structure.
"""
from pathlib import Path
import hashlib,json,time
from llama_cpp import Llama
HERE=Path(__file__).parent
SOURCE=HERE/'frozen_contract_source_scope_20261024.json'
FROZEN_SHA='0f7aa329b52413a21d0362a211856edf96297133f9c72001ad45ff14c2cd4d90'
MODEL=Path('/data/tmp/kmd-qwen-small-isolated-20261024/qwen2.5-1.5b-instruct-q4_k_m.gguf')
OUT=HERE/'raw_model_comparison_20261024.jsonl'

def ask(llm,system,prompt,tokens):
 t=time.monotonic()
 try:
  x=llm.create_chat_completion(messages=[{'role':'system','content':system},{'role':'user','content':prompt}],max_tokens=tokens,temperature=0,top_p=1,response_format={'type':'json_object'})
  raw=x['choices'][0]['message'].get('content') or ''
  try:obj=json.loads(raw)
  except Exception:obj=None
  return {'raw':raw,'parsed':obj,'finish_reason':x['choices'][0].get('finish_reason'),'usage':x.get('usage'), 'elapsed_seconds':round(time.monotonic()-t,2)}
 except Exception as e:return {'raw':'','parsed':None,'error':str(e)[:300], 'elapsed_seconds':round(time.monotonic()-t,2),'finish_reason':'failed'}

def direct(llm,case,claim):
 sys=('Only this particular PUBLIC CONTRACT governs the answer. Different contracts can grant different rights. '
  'Assess the hypothesis as Entailment, Contradiction or NotMentioned, where no document evidence is NOT negation. '
  'Exceptions, parties, time after agreement termination, permission versus obligation all matter. '
  'Return only compact JSON keys "verdict" and "quote". Quote 4-12 exact words from supplied source if possible; otherwise empty quote.')
 return ask(llm,sys,f"Contract ID: {case['id']}\nSource excerpts:\n{case['source_excerpt']}\nHypothesis: {claim}\nJSON:",85)

def extract(llm,case):
 sys=('From this contract excerpt, extract only ONE legally relevant rule about keeping, returning or destroying confidential data '
   'and whether it applies after termination. Never guess extra rules. '
   'Return compact JSON with keys: "holder" (one party), "modality" (permit/forbid/require/conditional/unknown), '
   '"action" (short), "time" (short), "exception" (short), "quote" (copy 5-15 words verbatim from source). '
   'All fields source-grounded. If no rule exists set modality unknown and quote empty. ONE object ONLY.')
 return ask(llm,sys,f"Contract ID: {case['id']}\nSource excerpts:\n{case['source_excerpt']}\nJSON:",145)

def grounded(x,case):
 if not isinstance(x,dict):return False,'invalid_json'
 if any(k not in x for k in ('holder','modality','action','time','exception','quote')):return False,'incomplete_structure'
 if x['modality'] not in ('permit','forbid','require','conditional','unknown'):return False,'bad_modality'
 q=x['quote'];source=case['source_excerpt']
 if x['modality']!='unknown' and (not isinstance(q,str) or len(q)<5 or q not in source):return False,'quote_not_grounded'
 return True,'valid_structured_source_extraction'

def reason(llm,obj,claim):
 sys=('Decide entailment, contradiction or not mentioned using ONLY the extracted contract rule. '
      'An exception can change scope; do not assume other unlisted contract provisions. '
      'Output JSON only with "verdict": "Entailment", "Contradiction", or "NotMentioned" and "reason": concise wording. '
      'If the extracted rule cannot settle the hypothesis output NotMentioned.')
 return ask(llm,sys,'Extracted source rule: '+json.dumps(obj)+'\nHypothesis: '+claim+'\nJSON:',70)

def main():
 assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==FROZEN_SHA,'Frozen input changed'
 freeze=json.loads(SOURCE.read_text());assert len(freeze['cases'])==6
 done=[json.loads(x) for x in OUT.read_text().splitlines() if x] if OUT.exists() else []
 finished={x['id'] for x in done};assert len(done)==len(finished)
 if len(done)==6:print('ALREADY_FINISHED_NO_RETRY');return
 llm=Llama(model_path=str(MODEL),n_ctx=2100,n_threads=2,n_batch=64,n_gpu_layers=0,seed=42,chat_format='chatml',verbose=False)
 for case in freeze['cases']:
  if case['id'] in finished:continue
  assert hashlib.sha256(case['source_excerpt'].encode()).hexdigest()==case['source_excerpt_sha256']
  a=direct(llm,case,freeze['hypothesis']);b=extract(llm,case);good,reasoncode=grounded(b.get('parsed'),case)
  c=reason(llm,b['parsed'],freeze['hypothesis']) if good else {'raw':'','parsed':{'verdict':'NotMentioned'},'skipped':'invalid_scope_frame','elapsed_seconds':0,'finish_reason':'skipped'}
  rec={'id':case['id'],'source_excerpt_sha256':case['source_excerpt_sha256'],
    'model':'Qwen/Qwen2.5-1.5B-Instruct-GGUF@91cad51170dc346986eccefdc2dd33a9da36ead9','cpu_threads':2,'ctx':2100,
    'direct':a,'structured_extraction':b,'frame_valid':good,'frame_reason':reasoncode,'structured_verdict':c}
  with OUT.open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush()
  finished.add(case['id'])
  print('FINISHED',len(finished),'/',len(freeze['cases']),'doc',case['id'],
   'direct',a.get('parsed',{}).get('verdict') if isinstance(a.get('parsed'),dict) else 'BAD',
   'structured',reasoncode,'result',c.get('parsed',{}).get('verdict') if isinstance(c.get('parsed'),dict) else 'BAD',
   'runtime',round(sum(z.get('elapsed_seconds',0) for z in (a,b,c)),1),flush=True)
if __name__=='__main__':main()
