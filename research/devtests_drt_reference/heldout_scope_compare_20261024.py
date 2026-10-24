"""Pre-frozen original-recording span retrieval/scope ablation; TinyLlama CPU.

Model output uses simplified DRS-like world, modality and polarity slots. It
is NOT KMD's production schema/parser; invalid model frames remain failures.
No model tuning or label-specific retries after looking at test outcomes.
"""
import argparse, hashlib, json, re, time, resource
from collections import Counter
from pathlib import Path
HERE=Path(__file__).parent
GOLD=HERE/'heldout_recordings_scope_gold_20261024.json'
MODEL='/data/models/tinyllama-1.1b-1t-openorca.Q4_0.gguf'

def neighbor_metadata(text,target):
    # Scope terms in the provenance-preserving excerpt + target, not bare match.
    s=text.lower();t=target.lower()
    if 'dream' in s and ('timmy' in s or 'dream' in t): return 'dream'
    if 'if the question' in s or 'under this condition' in s:return 'conditional'
    if 'could be considered' in s:return 'speculative'
    if 'for example' in s and ('the date a law' in s or 'law can' in s):return 'hypothetical'
    if 'we could' in s or 'we maybe' in s or 'we can ask' in s or 'we might' in s or 'maybe need' in s:return 'proposal'
    return 'asserted'

def model_scope(excerpt,anchor,llm):
    # Prompt and schema written once; no tuning against the heldout examples.
    prompt=("You annotate one SOURCE TARGET as structured discourse representation.\n"
      "Use only the excerpt. Do not assess real-world truth.\n"
      "Pick discourse scope from EXACTLY: asserted, dream, hypothetical, conditional, proposal, speculative.\n"
      "Give one JSON object with keys world, modality, polarity.\n"
      "world is real or dream or hypothetical; modality is asserted or conditional or proposal or speculative.\n"
      "A dream claim is world dream. An example is world hypothetical. An if clause is modality conditional.\n"
      "An agent suggestion is modality proposal. A philosophical possible axiom is speculative.\n"
      "Everything else is asserted in this source (not necessarily true).\n"
      "EXCERPT: "+excerpt+"\nTARGET: "+anchor+"\nJSON: ")
    result=llm(prompt,max_tokens=56,temperature=0,stop=['\n\n','EXCERPT:'],echo=False)
    raw=result['choices'][0]['text'].strip();start=raw.find('{');end=raw.rfind('}')
    parsed=None
    if start>=0 and end>start:
      try:parsed=json.loads(raw[start:end+1])
      except json.JSONDecodeError:pass
    label='invalid'
    if isinstance(parsed,dict):
      w=parsed.get('world');m=parsed.get('modality')
      if w=='dream':label='dream'
      elif w=='hypothetical':label='hypothetical'
      elif m in ('asserted','conditional','proposal','speculative'):label=m
    return label,raw

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',action='store_true');args=ap.parse_args()
    payload=GOLD.read_bytes(); rows=json.loads(payload)['rows'];assert len(rows)==10
    from hashlib import sha256
    source_root=Path('/data/infra/docs/guidlines/original-recordings')
    for r in rows:
      b=(source_root/r['source']).read_bytes();assert sha256(b).hexdigest()==r['source_sha256']
      t=b.decode('utf-8');assert t[r['source_char_start']:r['source_char_end']]==r['excerpt']
    llm=None
    if args.model:
      from llama_cpp import Llama
      llm=Llama(model_path=MODEL,n_ctx=1024,n_threads=2,n_batch=64,verbose=False)
    out=[]; started=time.monotonic()
    for i,r in enumerate(rows):
      baseline=neighbor_metadata(r['excerpt'],r['anchor'])
      if llm is not None:prediction,raw=model_scope(r['excerpt'],r['anchor'],llm)
      else: prediction,raw=None,None
      out.append({'id':r['id'],'gold':r['gold_scope'],'baseline':baseline,'baseline_correct':baseline==r['gold_scope'],'model_extracted_scope':prediction,'model_correct':prediction==r['gold_scope'] if llm else None,'model_raw':raw,'source':r['source'],'source_char_start':r['source_char_start'],'source_char_end':r['source_char_end']})
      print(f"{i+1}/{len(rows)} gold={r['gold_scope']} neighbor={baseline} model={prediction} elapsed={round(time.monotonic()-started,1)}s",flush=True)
    result={'dataset_sha256':sha256(payload).hexdigest(),'input_count':len(rows),'model':MODEL if llm else None,'model_sampling':'temperature=0,max_tokens=56,n_ctx=1024,2 CPU threads' if llm else None,'seconds':time.monotonic()-started,'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'baseline_correct':sum(r['baseline_correct'] for r in out),'model_correct':sum(r['model_correct'] for r in out) if llm else None,'model_invalid':sum(r['model_extracted_scope']=='invalid' for r in out) if llm else None,'cases':out,'limitations':['Manually scoped natural recordings, not independent double annotations','Heuristic baseline is not strongest retriever','TinyLlama direct DRS-like JSON is not deployed KMD schema or pipeline','No real-world truth inference assessed']}
    file=HERE/('heldout_recordings_model_vs_neighbor_20261024_results.json' if llm else 'heldout_recordings_neighbor_20261024_results.json');file.write_text(json.dumps(result,indent=2)+'\n');print('RESULT',result['baseline_correct'],result['model_correct'],'invalid',result['model_invalid'],flush=True)
if __name__=='__main__':main()
