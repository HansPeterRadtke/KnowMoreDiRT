"""Direct multi-document Qwen answer experiment; same prompt/budget per retrieval method.

Original gold answer and supporting facts are never read, and there is no
follow-up repair. Each result is checkpointed; completed results never re-run.
This is not DRS and Qwen model may have seen HotpotQA in pretraining.
"""
import hashlib,json,time
from pathlib import Path
from llama_cpp import Llama
HERE=Path(__file__).parent
F=HERE/'frozen_source_only_model_qa_20261025.json'
OUT=HERE/'raw_source_only_qa_qwen_20261025.jsonl'
MODEL=Path('/data/tmp/kmd-qwen-small-isolated-20261024/qwen2.5-1.5b-instruct-q4_k_m.gguf')
SHA='8c5af13eb3db9b89923a37a2811082113a7e0d28e6c81921512c479d4977ccc2'
METHODS=('fulltext_bm25','linked_secondhop')

def answer(llm,case,method):
    system=('Answer questions using ONLY the two quoted source articles, without pretrained outside knowledge. '
       'Articles have different subjects; respect proper names, dates, negation and reference identity. '
       'If the supplied source facts cannot establish the answer, answer "unknown". '
       'Output one JSON object with keys "answer", "source_title", "source_quote". '
       'source_title must exactly match one shown document title, source_quote copy 5-25 contiguous source words. '
       'If there is no evidence, leave title and quote empty. No explanation beyond this JSON.')
    sources='\n'.join(f'ARTICLE: {x["title"]}\n{x["text"]}' for x in case['source_documents'][method])
    user=f'SOURCE DOCUMENTS:\n{sources}\nQUESTION: {case["question"]}\nJSON:'
    start=time.monotonic()
    try:
       result=llm.create_chat_completion(messages=[{'role':'system','content':system},{'role':'user','content':user}],max_tokens=125,temperature=0,top_p=1,response_format={'type':'json_object'})
       raw=result['choices'][0]['message'].get('content') or ''
       try:parsed=json.loads(raw)
       except Exception:parsed=None
       return {'raw':raw,'parsed':parsed,'stop_reason':result['choices'][0].get('finish_reason'),'usage':result.get('usage'),'seconds':round(time.monotonic()-start,2)}
    except Exception as e:
       return {'raw':None,'parsed':None,'stop_reason':'error','error':str(e)[:300],'seconds':round(time.monotonic()-start,2)}

def main():
    if hashlib.sha256(F.read_bytes()).hexdigest()!=SHA:raise RuntimeError('frozen inputs changed')
    freeze=json.loads(F.read_text());cases=freeze['records'];assert len(cases)==36
    assert MODEL.stat().st_size==1117320736
    old=[json.loads(x) for x in OUT.read_text().splitlines() if x] if OUT.exists() else []
    done={(x['uid'],x['method']) for x in old};assert len(done)==len(old)
    print('PREVIOUS_COMPLETIONS',len(done),'REMAINING',2*len(cases)-len(done),flush=True)
    if len(done)==2*len(cases):return
    llm=Llama(model_path=str(MODEL),n_ctx=2304,n_threads=2,n_batch=64,chat_format='chatml',n_gpu_layers=0,seed=42,verbose=False)
    started=time.monotonic()
    for i,case in enumerate(cases):
       order=METHODS if int(case['selection_hash'][-1],16)%2==0 else tuple(reversed(METHODS))
       for method in order:
        if (case['uid'],method) in done:continue
        r=answer(llm,case,method)
        rec={'uid':case['uid'],'method':method,'input_sha256':hashlib.sha256(json.dumps({'case':case['uid'],'method':method,'docs':case['source_documents'][method]},sort_keys=True).encode()).hexdigest(),
          'model':'Qwen/Qwen2.5-1.5B-Instruct-GGUF@91cad51170dc346986eccefdc2dd33a9da36ead9','runtime':'llama_cpp Python CPU 2 threads, nice 15, temperature 0, n_ctx=2304, completion max=125','output':r}
        with OUT.open('a',encoding='utf-8') as fp:
          fp.write(json.dumps(rec,ensure_ascii=False)+'\n');fp.flush()
        done.add((case['uid'],method))
        print('COMPLETED',len(done),'/',len(cases)*2,'type',case['rank_change_group'],'method',method,
              'answer',str(r.get('parsed',{}).get('answer'))[:45] if isinstance(r.get('parsed'),dict) else 'INVALID',
              'seconds',r['seconds'],'total_elapsed',round(time.monotonic()-started,1),flush=True)
    print('DONE_ALL',len(done),'ELAPSED',round(time.monotonic()-started,1),flush=True)
if __name__=='__main__':main()
