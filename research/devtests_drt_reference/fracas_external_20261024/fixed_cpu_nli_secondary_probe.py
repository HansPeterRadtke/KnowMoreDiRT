"""Secondary, fixed-prompt CPU NLI check on a preexisting external split.

This is a DIRECT TinyLlama language judgment, NOT the KMD DRS executor.
The FraCaS dev split was already consumed by the lexical baseline; it must
not be represented as a pristine final holdout after this experiment.
"""
import csv,hashlib,json,re,time,resource
from pathlib import Path
from llama_cpp import Llama
HERE=Path(__file__).parent
SRC=HERE/'original_dev.csv'
MODEL='/data/models/tinyllama-1.1b-1t-openorca.Q4_0.gguf'
PICK=24
ROW_COUNT=70
rows=list(csv.DictReader(SRC.open(newline='',encoding='utf-8')))
assert len(rows)==ROW_COUNT
indices=sorted(range(len(rows)),key=lambda i:hashlib.sha256(f'{i}|{rows[i]["premise"]}|{rows[i]["hypothesis"]}'.encode()).digest())[:PICK]
idx_file=HERE/'secondary_fixed_indices.json'
freeze={'dataset_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'selection':'first 24 sorted by sha256 of index/premise/hypothesis without gold','indices':indices,'prompt_version':'nli-open-world-three-way-v1','not_pristine_new_holdout':True}
if idx_file.exists():assert json.loads(idx_file.read_text())==freeze
else:idx_file.write_text(json.dumps(freeze,indent=2)+'\n')
start=time.monotonic();model=Llama(model_path=MODEL,n_ctx=1536,n_threads=2,n_batch=64,verbose=False,seed=47)
result=[]
for i in indices:
 p,h=rows[i]['premise'],rows[i]['hypothesis']
 prompt=("Determine whether the hypothesis MUST be true, MUST be false, or is undetermined given ONLY the premises. "
         "The lack of evidence is not evidence for the opposite. "
         "Return ONLY one of the words: entailment, contradiction, neutral.\n"
         f"Premises: {p}\nHypothesis: {h}\nAnswer: ")
 response=model(prompt,max_tokens=12,temperature=0,stop=['\n','Premises:','Hypothesis:'],echo=False)
 raw=response['choices'][0]['text'].strip()
 found=re.match(r'^\s*(entailment|contradiction|neutral)\b',raw,re.I)
 pred=found.group(1).lower() if found else 'invalid'
 result.append({'row_index':i,'prediction':pred,'raw_model_output':raw,'finish_reason':response['choices'][0]['finish_reason']})
 print('CASE',len(result),'of',len(indices),'pred',pred,'elapsed',round(time.monotonic()-start,1),flush=True)
# Score only after every prediction has been frozen in memory.
for item in result:
 expected=rows[item['row_index']]['label'];item['expected']=expected;item['correct']=item['prediction']==expected
 from lexical_full_premise_baseline import predict
 item['lexical_baseline_prediction']=predict(rows[item['row_index']]['premise'],rows[item['row_index']]['hypothesis'])
 item['lexical_baseline_correct']=item['lexical_baseline_prediction']==expected
obj={'fixed_sample':freeze,'model':MODEL,'runtime':'llama-cpp-python 0.3.9','n_ctx':1536,'n_threads':2,'max_tokens':12,'seed':47,'temperature':0,'elapsed_seconds':round(time.monotonic()-start,2),'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'total':len(result),'model_correct':sum(x['correct'] for x in result),'baseline_correct':sum(x['lexical_baseline_correct'] for x in result),'invalid_outputs':sum(x['prediction']=='invalid' for x in result),'results':result,'limitations':['Nonpristine dev split after lexical baseline inspection','TinyLlama direct one-label judgment is not DRS model parsing, symbolic inference or scientific falsification','Fixed small sample, not statistical generalization','No prompt search or retries']}
(HERE/'secondary_cpu_nli_results.json').write_text(json.dumps(obj,indent=2)+'\n');print('FINISHED',obj['model_correct'],obj['baseline_correct'],'invalid',obj['invalid_outputs'],flush=True)
