"""Untrained external NLI model evaluates REAL paired Wikipedia revisions.

Human evidence support/refute/unknown labels are deliberately NOT read.
Original source claim+evidence of publisher VitaminC test (2021), frozen 700
four-way revision groups, 2,800 paired model NLI predictions; original version
IDs and evidence hashes retained. No model ensemble, no DRS oracle or gold.
"""
from pathlib import Path
import json,hashlib,time,resource
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
HERE=Path(__file__).parent
SOURCE=Path('/data/tmp/kmd-vitaminc-original-test-20261025.jsonl')
FROZEN=HERE/'frozen_real_revision_source_only_20261025.json'
OUT=HERE/'raw_source_version_nli_predictions_20261025.jsonl'
SRC_SHA='7ad1808dbc30c62e0a1427a53022d0dfaff668a1fde3c4b612a2d266edd753ad'
F_SHA='a6c6fca2dd5de8674606c8496c7e6fb1918321ee16cf815cfe5d7ec2fdd9ed7c'
MODEL=Path('/data/tmp/kmd-scientific-nli-20261024')
ONNX_SHA='449024a8711a3e655cad410cb2192158f9897fc2a32b65dbb817922351150db7'
LABELS=('REFUTES','SUPPORTS','NOT ENOUGH INFO')
BATCH=24;MAX_LENGTH=256

def main():
 start=time.monotonic()
 assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SRC_SHA
 assert hashlib.sha256(FROZEN.read_bytes()).hexdigest()==F_SHA
 assert hashlib.sha256((MODEL/'onnx/model_qint8_arm64.onnx').read_bytes()).hexdigest()==ONNX_SHA
 selected=json.loads(FROZEN.read_text())['case_ids_and_input_hashes_only'];assert len(selected)==700
 selected_map={c['case_id']:c for c in selected};source_rows={}
 for line in SOURCE.read_text().splitlines():
  if not line:continue
  r=json.loads(line)
  if r['case_id'] in selected_map:source_rows[r['unique_id']]=r
 cases=[]
 for c in selected:
  assert len(c['source_data_row_ids'])==len(c['original_claim_evidence_pair_hashes'])==4
  for unique,expected in zip(c['source_data_row_ids'],c['original_claim_evidence_pair_hashes']):
   row=source_rows[unique]
   assert hashlib.sha256((row['claim']+'\n'+row['evidence']).encode()).hexdigest()==expected
   assert row['case_id']==c['case_id']
   cases.append({'uid':unique,'case_id':c['case_id'],'hash':expected,'claim':row['claim'],'evidence':row['evidence']})
 assert len(cases)==2800 and len({x['uid'] for x in cases})==len(cases)
 existing=[json.loads(x) for x in OUT.read_text().splitlines() if x] if OUT.exists() else []
 done={x['uid'] for x in existing};assert len(done)==len(existing)
 assert done<={x['uid'] for x in cases}
 if len(done)==len(cases):print('ALREADY_FINISHED_INFERENCE_NO_REPEAT');return
 configs=json.loads((MODEL/'config.json').read_text());assert [configs['id2label'][str(i)] for i in range(3)]==['contradiction','entailment','neutral']
 tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
 so=ort.SessionOptions();so.intra_op_num_threads=2;so.inter_op_num_threads=1
 sess=ort.InferenceSession(str(MODEL/'onnx/model_qint8_arm64.onnx'),sess_options=so,providers=['CPUExecutionProvider'])
 fresh=[x for x in cases if x['uid'] not in done]
 print('MODEL_CPU_STARTED','already',len(done),'remaining',len(fresh),'original_test_hash',SRC_SHA,'model_sha',ONNX_SHA,flush=True)
 for i in range(0,len(fresh),BATCH):
  batch=fresh[i:i+BATCH]
  premises=[x['evidence'] for x in batch];hypotheses=[x['claim'] for x in batch]
  original_tokens=tok(premises,hypotheses,truncation=False,padding=False)
  counts=[len(z) for z in original_tokens['input_ids']]
  enc=tok(premises,hypotheses,truncation=True,max_length=MAX_LENGTH,padding=True,return_tensors='np')
  logits=sess.run(None,{x.name:enc[x.name].astype('int64') for x in sess.get_inputs()})[0]
  with OUT.open('a',encoding='utf-8') as dest:
   for case,score,token_count in zip(batch,logits,counts):
    z=np.exp(score-np.max(score)); probs=z/z.sum();best=int(np.argmax(probs))
    record={'uid':case['uid'],'case_id':case['case_id'],'source_pair_sha256':case['hash'],
      'model':'cross-encoder/nli-MiniLM2-L6-H768@b95119ce93d3e065de6214e38cd4a97b0f2f2c6d',
      'model_binary_sha256':ONNX_SHA,'runtime':'onnxruntime CPUExecutionProvider, 2 threads, inference-only, no retraining',
      'tokenizer_max_length':MAX_LENGTH,'untruncated_token_length':token_count,'token_input_truncated':token_count>MAX_LENGTH,
      'predicted_label':LABELS[best], 'probabilities_label_order_refutes_supports_nei':[round(float(v),6) for v in probs]}
    dest.write(json.dumps(record)+'\n')
   dest.flush()
  if (i//BATCH)%8==0 or i+BATCH>=len(fresh):
   print('CHECKPOINTED',len(done)+min(i+BATCH,len(fresh)),'/',len(cases),'seconds',round(time.monotonic()-start,1),'peak_rss_kib',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
 print('FINISHED',len(cases),'seconds',round(time.monotonic()-start,2),flush=True)
if __name__=='__main__':main()
