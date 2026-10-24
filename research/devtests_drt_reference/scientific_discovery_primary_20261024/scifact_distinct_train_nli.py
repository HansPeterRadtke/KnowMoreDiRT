"""Independent SciFact TRAIN-scoped small CPU NLI evaluation.

External peer-annotated scientific claim/evidence pairs; source selection
frozen by hash of claim and document ID before inference, never by label.
Disjoint from previous dev model attempt. Cannot be called final production
model or DRS evaluation: NLI gives opinions, not independent experiments.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent
DATA=Path('/data/tmp/kmd-scifact-release-20261024/data')
MODEL=Path('/data/tmp/kmd-scientific-nli-20261024')
MODEL_SHA_EXPECTED=''
CLAIMS_SHA='f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0'
CORPUS_SHA='b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62'
LABELS=['CONTRADICT','SUPPORT','NEUTRAL']

def parse_lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
def pick(claim,abstract):
 import re
 words=lambda t:set(re.findall(r'(?u)\b\w\w+\b',t.casefold()))
 q=words(claim)
 return [j for _,j in sorted([(len(q & words(s))/(len(q|words(s)) or 1),i) for i,s in enumerate(abstract)], key=lambda z:(-z[0],z[1]))[:2]]

def main():
 assert hashlib.sha256((DATA/'claims_train.jsonl').read_bytes()).hexdigest()==CLAIMS_SHA
 assert hashlib.sha256((DATA/'corpus.jsonl').read_bytes()).hexdigest()==CORPUS_SHA
 claims=parse_lines(DATA/'claims_train.jsonl'); docs={str(d['doc_id']):d for d in parse_lines(DATA/'corpus.jsonl')}
 cases=[]
 for c in claims:
  for did,evidence_sets in c['evidence'].items():
   cases.append({'claim_id':c['id'],'doc_id':str(did),'claim':c['claim'],'annotated_sets':evidence_sets})
 cases.sort(key=lambda x:hashlib.sha256(f"{x['claim_id']}|{x['doc_id']}".encode()).digest())
 cases=cases[:96]; assert len(cases)==96
 frozen={'protocol':'fixed first 96 hash-sorted claim-document source pairs with ≥1 expert evidence annotation, all labels excluded from sorting','sha256_train':CLAIMS_SHA,'sha256_corpus':CORPUS_SHA,'pair_keys':[[x['claim_id'],x['doc_id']] for x in cases],'model':'cross-encoder/nli-MiniLM2-L6-H768 revision b95119ce93d3e065de6214e38cd4a97b0f2f2c6d','maximum_context_tokens':192,'top_source_sentences':2,'decision':'max nonneutral minus neutral margin across top2, require positive margin'}
 freeze=HERE/'scifact_distinct_train_nli_frozen_cases_20261024.json'
 encoded=json.dumps(frozen,indent=2)+'\n'
 if freeze.exists():assert freeze.read_text()==encoded
 else:freeze.write_text(encoded)
 inputs=[];groups=[]
 for c in cases:
  d=docs[c['doc_id']];ix=pick(c['claim'],d['abstract']);groups.append(ix)
  for i in ix:inputs.append((d['abstract'][i],c['claim']))
 onnx=MODEL/'onnx/model_qint8_arm64.onnx';assert onnx.stat().st_size==82823065
 start=time.perf_counter();tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
 opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
 session=ort.InferenceSession(str(onnx),sess_options=opts,providers=['CPUExecutionProvider'])
 probabilities=[]
 for i in range(0,len(inputs),24):
  batch=inputs[i:i+24]
  features=tok([x[0] for x in batch],[x[1] for x in batch],max_length=192,truncation=True,padding=True,return_tensors='np')
  raw=session.run(None,{x.name:features[x.name].astype('int64') for x in session.get_inputs()})[0]
  for y in raw:
   z=np.exp(y-np.max(y));probabilities.append([round(float(a),6) for a in z/z.sum()])
  print('INFERENCE',min(i+24,len(inputs)),'/',len(inputs),'SECONDS',round(time.perf_counter()-start,1),flush=True)
 assert len(probabilities)==len(inputs)
 out=[];pos=0
 for c,indices in zip(cases,groups):
  scored=[]
  for sid in indices:
   p=probabilities[pos];pos+=1
   margin=max(p[:2])-p[2]
   scored.append({'sentence_idx':sid,'probabilities':p,'predicted_sentence':LABELS[int(np.argmax(p))],'nonneutral_margin':round(margin,6)})
  winner=max(scored,key=lambda x:x['nonneutral_margin']) if scored else None
  label=winner['predicted_sentence'] if winner and winner['nonneutral_margin']>0 else 'NEUTRAL'
  gold=set(x['label'] for x in c['annotated_sets'])
  ref=(next(iter(gold)) if len(gold)==1 else 'MIXED')
  out.append({'claim_id':c['claim_id'],'doc_id':c['doc_id'],'label_predicted':label,'human_annotated_label':ref,'correct':label==ref,'source_sentence_predictions':scored,'annotated_evidence_sentence_ids':[x['sentences'] for x in c['annotated_sets']],
      'claim_sha256':hashlib.sha256(c['claim'].encode()).hexdigest(),'article_title_sha256':hashlib.sha256(docs[c['doc_id']]['title'].encode()).hexdigest()})
 evaluated=[x for x in out if x['human_annotated_label']!='MIXED']
 from collections import Counter
 result={'model_repo_revision':'b95119ce93d3e065de6214e38cd4a97b0f2f2c6d','cpu_model_onnx_sha256':hashlib.sha256(onnx.read_bytes()).hexdigest(), 'provenance':frozen,
 'total_fixed_pairs':len(out),'evaluable_unambiguous_gold_pairs':len(evaluated),'correct':sum(x['correct'] for x in evaluated),'neutral_count':sum(x['label_predicted']=='NEUTRAL' for x in evaluated),
 'gold_counts':dict(Counter(x['human_annotated_label'] for x in evaluated)),'prediction_counts':dict(Counter(x['label_predicted'] for x in evaluated)),'confusion':[{'gold':k[0],'predicted':k[1],'count':v} for k,v in sorted(Counter((x['human_annotated_label'],x['label_predicted']) for x in evaluated).items())],
 'source_claim_count':len(set(x['claim_id'] for x in out)),'elapsed_seconds':round(time.perf_counter()-start,2),'source_tokens_extracted':len(inputs),'results':out,
 'limitations':['Published SciFact TRAIN subset is disjoint from earlier DEV retrieval/model attempt, labels generated by original external annotators','Source documents deliberately oracle-selected through published evidence links; not end-to-end retrieval','Science label correctness is annotated text-evidence consistency, not world truth','The model is generic natural language inference, not KMD DRS, and cannot prove scientific falsification','Selection uses only IDs and presence of annotated gold, not gold label; gold not used to tune prompt, sentence selection or thresholds']}
 target=HERE/'scifact_distinct_train_nli_results_20261024.json'
 if target.exists():raise RuntimeError('DO_NOT_REPEAT_ALREADY_EXECUTED_EXPERIMENT')
 target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print('DISTINCT_EXTERNAL_GOLD_NLI',result['correct'],'/',result['evaluable_unambiguous_gold_pairs'],'CONFUSION',result['confusion'])
if __name__=='__main__':main()
