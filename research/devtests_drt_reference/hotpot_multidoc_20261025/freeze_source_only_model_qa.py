"""Gold-BLIND original multi-document natural QA, fixed 36-case comparative panel.

Selection by type and source-rank-change ONLY, no gold answers/source supports.
Use ORIGINAL 2018 contexts with two non-oracle ranked documents. No DRS.
"""
from pathlib import Path
from collections import defaultdict
import hashlib,json,re
HERE=Path(__file__).parent
SRC=Path('/data/tmp/kmd-hotpot-original-distractor-20261025.json')
PRED=HERE/'frozen_unlabelled_retrieval_20261025.json'
OUT=HERE/'frozen_source_only_model_qa_20261025.json'
S_SHA='4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea'
P_SHA='3b458f9dca38f0fd6a00896928d7df98c6269297c0da4a65fd4b22dfbd76cdf8'
QUOTAS={'bridge_changed':16,'bridge_same':8,'comparison_changed':8,'comparison_same':4}
STOP=set('a an is are of the to who what where when how that this which in from did does for with by and as on at a'.split())

def tok(text):return set(z for z in re.findall(r'(?u)\b\w\w+\b',text.lower()) if z not in STOP)

def extract(question,sentences):
    if not sentences:return {'indices':[],'text':''}
    q=tok(question)
    sc=[]
    for i,s in enumerate(sentences):
        b=tok(s);sc.append((len(q&b)/(len(q|b) or 1),i))
    top=[i for _,i in sorted(sc,key=lambda x:(-x[0],x[1]))[:2]]
    picks=sorted({0}|{k for i in top for k in range(max(0,i-1),min(len(sentences),i+2))})
    text=[];selected=[]
    for i in picks:
        chunk=f'[{i}] {sentences[i].strip()}'
        if selected and sum(len(t) for t in text)+len(chunk)>1300:continue
        if len(chunk)>1300:chunk=chunk[:1280]+' [SOURCE-SENTENCE-TRUNCATED]'
        text.append(chunk);selected.append(i)
    return {'indices':selected,'text':'\n'.join(text)}

def main():
 if OUT.exists():raise RuntimeError('Already frozen, never reselect')
 assert hashlib.sha256(SRC.read_bytes()).hexdigest()==S_SHA
 assert hashlib.sha256(PRED.read_bytes()).hexdigest()==P_SHA
 original=json.loads(SRC.read_text());pred=json.loads(PRED.read_text())['predictions'];assert len(original)==len(pred)==7405
 pools=defaultdict(list)
 for a,p in zip(original,pred):
  changed=p['ranks']['fulltext_bm25'][:2]!=p['ranks']['linked_secondhop'][:2]
  bucket=a['type']+'_'+('changed' if changed else 'same')
  key=hashlib.sha256(('hotpotqwen1.5-20261025-'+a['_id']).encode()).hexdigest()
  pools[bucket].append((key,a,p))
 records=[]
 for b,quota in QUOTAS.items():
  pool=sorted(pools[b],key=lambda x:x[0]);assert len(pool)>=quota
  for key,a,p in pool[:quota]:
   ctx={t:ss for t,ss in a['context']}
   methods={}
   for m in ['fulltext_bm25','linked_secondhop']:
    docs=p['ranks'][m][:2]
    methods[m]=[{'title':title,**extract(a['question'],ctx[title])} for title in docs]
    assert len(methods[m])==2
   records.append({'uid':a['_id'],'type_from_original_question_metadata':a['type'],'rank_change_group':b,
      'hash_source_text':hashlib.sha256(json.dumps(a['context'],ensure_ascii=False).encode()).hexdigest(),
      'question':a['question'],'question_sha256':p['question_sha256'],'source_documents':methods,
      'selection_hash':key})
 records.sort(key=lambda x:x['selection_hash'])
 out={'source_sha256':S_SHA,'unlabelled_retrieval_sha256':P_SHA,'model_case_count':len(records),
   'selection':'SHA256 id sorted within original question type × predicted ranked doc difference, quotas fixed 16 bridge_changed 8 bridge_same 8 comparison_changed 4 comparison_same, without reference to gold article titles, supporting sentence IDs, or answer.',
   'input':'Question plus top2 retrieved article source titles and automatically selected source sentences (same algorithm per method); no answer, gold support or DRS annotations.',
   'methods':['fulltext_bm25','linked_secondhop'],'limits':['A selected subset after corpus-level source labels already analyzed; independent individual answers not inspected or used to choose item IDs','Pretrained Qwen may have encountered HotpotQA answer text during original training; cannot guarantee unseen model evaluation','Original Wikipedia articles are two out of up to ten question-specific candidates, not internet-wide search','Model answer semantic correctness and exact source provenance assessed only after predictions are frozen'],
   'records':records}
 OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print('FROZEN',len(records),'BUCKETS',QUOTAS,'SHA256',hashlib.sha256(OUT.read_bytes()).hexdigest(),'RANK_CHANGED',sum(x['source_documents']['fulltext_bm25']!=x['source_documents']['linked_secondhop'] for x in records))
 print('PROMPT_CHAR_RANGE',min(sum(len(y['text']) for y in x['source_documents']['fulltext_bm25']) for x in records),max(sum(len(y['text']) for y in x['source_documents']['fulltext_bm25']) for x in records))
if __name__=='__main__':main()
