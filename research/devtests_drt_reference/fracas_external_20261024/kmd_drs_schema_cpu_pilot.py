"""One bounded TRAIN-only smoke check of actual KMD chunk DRS schema/validator.

External FraCaS *development* labels are never inspected, used in prompts or
scored here. This is NOT full KMD retrieval, not Qwen, and not a benchmark win.
"""
import csv,hashlib,json,sys,time,traceback
from pathlib import Path
from llama_cpp import Llama,LlamaGrammar
from knowmoredirt.model_planner import build_chunk_drs_prompt,chunk_drs_json_schema,_repair_chunk_drs_payload,_validate_chunk_drs_payload
HERE=Path(__file__).parent
TRAIN=HERE/'original_train.csv'
MODEL=Path('/data/models/tinyllama-1.1b-1t-openorca.Q4_0.gguf')
ROWS=list(csv.DictReader(TRAIN.open(newline='',encoding='utf-8')))
INDEX=1
SOURCE=ROWS[INDEX]['premise']
REL_PATH='fracas-external/train-row-1'
OUTPUT=HERE/'kmd_production_drs_cpu_train_pilot_results.json'
started=time.monotonic()
result={'source':'external FraCaS train index 1','premise':SOURCE,'train_sha256':hashlib.sha256(TRAIN.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),'model':'TinyLlama-1.1B OpenOrca Q4_0; no shared model slots','schema':'production chunk_drs_json_schema; prompt build_chunk_drs_prompt','inference_configuration':{'max_tokens':320,'n_ctx':2048,'n_threads':2,'temperature':0,'seed':42},'accepted':False,'errors':[]}
try:
    prompt=build_chunk_drs_prompt(SOURCE,rel_path=REL_PATH)
    schema=chunk_drs_json_schema(source_id=REL_PATH,constrain_stable_ids=True)
    result['prompt_sha256']=hashlib.sha256(prompt.encode()).hexdigest()
    result['json_schema_sha256']=hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest()
    print('PILOT_PREPARED length',len(prompt),'schema_chars',len(json.dumps(schema)),flush=True)
    grammar=LlamaGrammar.from_json_schema(json.dumps(schema),verbose=False)
    llm=Llama(model_path=str(MODEL),n_ctx=2048,n_threads=2,n_batch=64,verbose=False,seed=42)
    print('PILOT_LOADED_AFTER_SECONDS',round(time.monotonic()-started,1),flush=True)
    response=llm(prompt,max_tokens=320,temperature=0,echo=False,grammar=grammar)
    raw=response['choices'][0]['text'];result['raw_output']=raw
    result['finish_reason']=response['choices'][0]['finish_reason']
    result['usage']=response.get('usage')
    print('PILOT_GENERATION_FINISHED',result['finish_reason'],'chars',len(raw),flush=True)
    try:
        parsed=json.loads(raw);result['valid_json']=True
    except json.JSONDecodeError as exc:
        result['valid_json']=False;result['errors'].append({'stage':'json','type':type(exc).__name__,'message':str(exc)})
    if result['valid_json']:
        repaired=_repair_chunk_drs_payload(parsed,SOURCE)
        validation=_validate_chunk_drs_payload(repaired,SOURCE)
        result['validation']=validation
        result['accepted']=bool(validation.get('schema_valid') and not validation.get('grounding_failure_count'))
        result['drs_condition_count']=len(repaired.get('drs',{}).get('conditions',[]))
        result['drs_box_count']=len(repaired.get('drs',{}).get('boxes',[]))
        print('KMD_VALIDATION',result['accepted'],'conditions',result['drs_condition_count'],flush=True)
except Exception as exc:
    result['errors'].append({'stage':'runner','type':type(exc).__name__,'message':str(exc)[:900]})
finally:
    result['elapsed_seconds']=round(time.monotonic()-started,2)
    OUTPUT.write_text(json.dumps(result,indent=2,default=str)+'\n',encoding='utf-8')
    print('RESULT_WRITTEN',str(OUTPUT),'seconds',result['elapsed_seconds'],flush=True)
