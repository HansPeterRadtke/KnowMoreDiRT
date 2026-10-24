"""Independent SciFact evidence-retrieval benchmark against published human labels.

NO gold evidence used in retrieval, fitting, rank thresholds or feature selection.
Train data unused. The fixed previously released development split is spent
once by this comparison; further changes would need new untouched data.

Methods are genuine automatic full-corpus retrieval of 5,183 scientific abstracts:
(a) title lexical tfidf, (b) title+abstract tfidf, (c) abstract BM25.
Top-k retrieved document ids are matched against human-evidence doc IDs.
A separate ORACLE-DOC rationale extractor examines nearest exact sentence and
neighbors, explicitly not an automatic end-to-end fact verifier.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,re,time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

SOURCE_SHA={
 'corpus.jsonl':'b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62',
 'claims_dev.jsonl':'86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217',
}
K=(1,5,10,20)
DOC_K1=1.2;DOC_B=0.75

def read_jsonl(p):
 return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line]

def ranked(scores,n):
 return np.argsort(-scores,kind='stable')[:n].tolist()

def sci_gold(claim):
 return {str(i) for i in claim['evidence'].keys()}

def bm25_docs(doc_texts,claim_texts):
 vectorizer=CountVectorizer(lowercase=True,stop_words='english',token_pattern=r'(?u)\b\w\w+\b')
 matrix=vectorizer.fit_transform(doc_texts).tocsr()
 n, _=matrix.shape
 lengths=np.asarray(matrix.sum(axis=1)).ravel()
 avg=float(np.mean(lengths))
 dfs=np.asarray((matrix>0).sum(axis=0)).ravel()
 idfs=np.log(1+(n-dfs+0.5)/(dfs+0.5))
 queries=vectorizer.transform(claim_texts).tocsr()
 r=[]
 for idx in range(len(claim_texts)):
  terms=queries[idx].indices
  scores=np.zeros(n,dtype='float64')
  for term in terms:
   col=matrix.getcol(int(term)).tocoo();tf=col.data.astype('float64');di=col.row
   val=idfs[term]*(tf*(DOC_K1+1))/(tf+DOC_K1*(1-DOC_B+DOC_B*lengths[di]/avg))
   np.add.at(scores,di,val)
  r.append(ranked(scores,max(K)))
 return r

def lexical_rationale(claim,abstract,sets):
 # uses claim+abstract alone; annotated rationale is inspected only in scoring.
 term_re=re.compile(r'(?u)\b\w\w+\b')
 q=set(term_re.findall(claim.lower()))
 scored=[]
 for i,sentence in enumerate(abstract):
  tok=set(term_re.findall(sentence.lower()))
  union=len(tok|q);score=len(tok&q)/union if union else 0
  scored.append((score,i))
 top=[i for _,i in sorted(scored,key=lambda x:(-x[0],x[1]))[:2]]
 one=set(top)
 neighborhood={j for i in one for j in range(max(0,i-1),min(len(abstract),i+2))}
 rationale_union=set().union(*sets) if sets else set()
 return {'top_sentences':sorted(one),'neighbor_sentences':sorted(neighborhood),'rationale_union':sorted(rationale_union),
         'hit_top2':bool(one&rationale_union),'hit_with_neighbors':bool(neighborhood&rationale_union),
         'top2_size':len(one),'window_size':len(neighborhood)}

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--data',type=Path,default=Path('/data/tmp/kmd-scifact-release-20261024/data'))
 args=pa.parse_args();start=time.perf_counter()
 for f,sha in SOURCE_SHA.items():
  assert hashlib.sha256((args.data/f).read_bytes()).hexdigest()==sha, f+' source changed'
 docs=read_jsonl(args.data/'corpus.jsonl');claims=read_jsonl(args.data/'claims_dev.jsonl')
 assert (len(docs),len(claims))==(5183,300)
 id_to_index={str(d['doc_id']):i for i,d in enumerate(docs)}
 assert len(id_to_index)==len(docs)
 title=[d['title'] for d in docs];full=[d['title']+' '+' '.join(d['abstract']) for d in docs]
 questions=[c['claim'] for c in claims]
 predictions={}
 for name,texts in [('title_tfidf',title),('title_abstract_tfidf',full)]:
  v=TfidfVectorizer(stop_words='english',ngram_range=(1,2),sublinear_tf=True,lowercase=True)
  docmatrix=v.fit_transform(texts);qmat=v.transform(questions)
  rank=[]
  for j in range(len(claims)):
   scores=(docmatrix@qmat[j].transpose()).toarray().ravel()
   rank.append(ranked(scores,max(K)))
  predictions[name]=rank
 predictions['abstract_bm25']=bm25_docs(full,questions)
 doc_ids=[str(d['doc_id']) for d in docs]
 results=[];method_stats={k:{f'one_relevant_in_top_{n}':0 for n in K} | {f'all_relevant_in_top_{n}':0 for n in K} for k in predictions}
 for idx,c in enumerate(claims):
  gold=sci_gold(c)
  assert all(d in id_to_index for d in gold)
  x={'id':c['id'],'claim':c['claim'],'gold_doc_ids':sorted(gold),'cited_doc_ids':c['cited_doc_ids'],'evidence_status':'labeled' if gold else 'no_labeled_evidence','methods':{}}
  for name,allrank in predictions.items():
   top=[doc_ids[i] for i in allrank[idx]]
   m={f'top_{n}':top[:n] for n in K}
   x['methods'][name]=m
   if gold:
    for n in K:
     got=set(top[:n]);method_stats[name][f'one_relevant_in_top_{n}']+=int(bool(got&gold))
     method_stats[name][f'all_relevant_in_top_{n}']+=int(gold<=got)
  results.append(x)
 evidence_n=sum(bool(sci_gold(c)) for c in claims)
 # Oracle evidence document for sentence rationale audit: NOT actual retrieval result.
 rat=[]
 for c in claims:
  for strid,evidences in c['evidence'].items():
   d=docs[id_to_index[strid]]
   sets=[set(e['sentences']) for e in evidences]
   x=lexical_rationale(c['claim'],d['abstract'],sets)
   rat.append({'claim_id':c['id'],'doc_id':strid,'gold_labels':sorted({e['label'] for e in evidences}),**x})
 summary={'method':'Predeclared title/full TFIDF and abstract BM25, no DRS. All gold claim/doc matches from SciFact expert annotations.','source_sha256':SOURCE_SHA,'original_source_repository':'https://github.com/allenai/scifact','corpus_docs':len(docs),'claims':len(claims),'claims_with_labeled_evidence':evidence_n,'claims_without_labeled_evidence':len(claims)-evidence_n,'multiple_gold_doc_claims':sum(len(sci_gold(c))>1 for c in claims),'methods':method_stats,
 'rationale_oracle_gold_doc':{'claim_document_pairs':len(rat),'hit_top2':sum(x['hit_top2'] for x in rat),'hit_top2_plus_neighbors':sum(x['hit_with_neighbors'] for x in rat),'context_window_mean_size':sum(x['window_size'] for x in rat)/len(rat)},
 'execution_seconds':round(time.perf_counter()-start,2),'caveats':['No source-label prediction or scientific truth verification was performed','Gold evidence document IDs are not used as search features; all document evidence retrieval ranks are computed across 5183 abstracts','Rationale recall uses oracle gold documents; not automatic end-to-end','A no-annotated-evidence claim is NOT a proven false scientific claim','Cited_doc_ids may include publications without human labeled SUPPORT/CONTRADICT evidence','This development split is now consumed for this protocol; do not tune and re-label as unseen test'], 'results':results,'rationale_gold_doc_cases':rat}
 target=Path(__file__).with_name('scifact_fixed_retrieval_results_20261024.json')
 if target.exists(): raise RuntimeError('Refuse overwrite: previously consumed experiment')
 target.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print('CLAIMS',len(claims),'LABELLED',evidence_n,'MULTIDOC',summary['multiple_gold_doc_claims'],'SECONDS',summary['execution_seconds'])
 for method,counts in method_stats.items(): print(method,counts)
 print('RATIONALE',summary['rationale_oracle_gold_doc'])
if __name__=='__main__':main()
