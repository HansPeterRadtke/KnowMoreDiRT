"""Frozen lexical source/neighbor retrieval against real ContractNLI gold.

Automatically retrieve ranked document spans from all 123 legal test contracts
for all 17 independent human hypotheses. Human gold *never used* to rank.
Same document is an oracle doc, source document itself is supplied; never
misreport source recovery as end-to-end corpus QA or DRS correctness.
"""
import hashlib,json,re,time
from collections import Counter,defaultdict
from pathlib import Path
SOURCE=Path('/data/tmp/kmd-contractnli-20261024/contract-nli/test.json')
HERE=Path(__file__).parent
OUT=HERE/'full_test_lexical_neighborhood_20261024.json'
SHA='460267b56052a2dc5aead98eb35eadef9e6734d5723d37b4a9790e410f812387'
TOP=(1,3,5,10)

def tok(x):return set(re.findall(r'(?u)\b\w\w+\b',x.casefold()))

def main():
 if OUT.exists():raise RuntimeError('DO_NOT_REPEAT_COMPLETED_EXPERIMENT')
 assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
 dataset=json.loads(SOURCE.read_text());start=time.monotonic();res=[]
 for document in dataset['documents']:
  full=document['text'];spans=document['spans'];texts=[full[a:b] for a,b in spans];span_words=[tok(x) for x in texts]
  for key,meta in dataset['labels'].items():
   ann=document['annotation_sets'][0]['annotations'][key]; gold=set(ann['spans']);q=tok(meta['hypothesis'])
   indices=sorted(range(len(spans)),key=lambda i:(-(len(q&span_words[i])/(len(q|span_words[i]) or 1)),i))
   ranks={}
   for k in TOP:
    sel=set(indices[:k]);neigh={j for i in sel for j in range(max(0,i-1),min(len(spans),i+2))}
    ranks[str(k)]={'ranked_gold_found':len(sel & gold),'ranked_all_gold':bool(gold and gold<=sel),
      'neighbors_gold_found':len(neigh & gold),'neighbors_all_gold':bool(gold and gold<=neigh),
      'source_window_count':len(neigh)}
   res.append({'document_id':document['id'],'hypothesis_id':key,'human_label':ann['choice'],'gold_span_count':len(gold),'gold_span_sha256':hashlib.sha256(json.dumps(sorted(gold)).encode()).hexdigest(),'ranks':ranks})
 byclass=defaultdict(list)
 for r in res:byclass[r['human_label']].append(r)
 summary={}
 for cls,items in byclass.items():
  positives=[x for x in items if x['gold_span_count']>0]
  o={'queries':len(items),'with_annotated_evidence':len(positives),'gold_evidence_spans':sum(x['gold_span_count'] for x in positives),'topk':{}}
  for k in TOP:
   d=str(k)
   o['topk'][d]={'ranked_any_gold':sum(x['ranks'][d]['ranked_gold_found']>0 for x in positives),
    'expanded_any_gold':sum(x['ranks'][d]['neighbors_gold_found']>0 for x in positives),
    'ranked_all_gold':sum(x['ranks'][d]['ranked_all_gold'] for x in positives),
    'expanded_all_gold':sum(x['ranks'][d]['neighbors_all_gold'] for x in positives),
    'ranked_gold_span_hits':sum(x['ranks'][d]['ranked_gold_found'] for x in positives),
    'expanded_gold_span_hits':sum(x['ranks'][d]['neighbors_gold_found'] for x in positives),
    'mean_window_spans':round(sum(x['ranks'][d]['source_window_count'] for x in items)/len(items),3)}
   assert o['topk'][d]['expanded_any_gold']>=o['topk'][d]['ranked_any_gold']
  summary[cls]=o
 out={'source':'ContractNLI original CC BY 4.0, Koreeda and Manning 2021, https://stanfordnlp.github.io/contract-nli/',
    'source_sha256':SHA,'cases':len(res),'distinct_contracts':len(dataset['documents']),'distinct_hypotheses':len(dataset['labels']),
    'method':'Token-set Jaccard top-k over entire original contract human-annotated spans, ±1 neighboring spans; ranker never sees gold. Evidence span indices from original gold used only after ranking.',
    'label_buckets':summary,'runtime_seconds':round(time.monotonic()-start,2),
    'limitations':['Oracle source DOCUMENT selected in advance; not entire-document discovery, no end-to-end claim verification','No actual DRS inference: this is a robust context retrieval control to compare later against a full DRS system',
     'NotMentioned has no annotated positive spans, so success is not measurable from evidence recall alone',
     'Expanding context costs tokens and may introduce distraction; this analysis measures only gold evidence coverage, not downstream answer quality'],
    'case_results':res}
 OUT.write_text(json.dumps(out,indent=2)+'\n');print('EXTERNAL_123_DOC_CONTRACT_RETRIEVAL',out['cases'],out['runtime_seconds'])
 for cls,s in summary.items():
  print(cls,'positive_docs',s['with_annotated_evidence'],'span_total',s['gold_evidence_spans'])
  for k,x in s['topk'].items():print(' TOP',k,'rank_any',x['ranked_any_gold'],'neigh_any',x['expanded_any_gold'],'rank_all',x['ranked_all_gold'],'neigh_all',x['expanded_all_gold'])
if __name__=='__main__':main()
