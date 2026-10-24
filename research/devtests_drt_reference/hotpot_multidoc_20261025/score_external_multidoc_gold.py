"""One-time independent HotpotQA human-gold scoring of FROZEN source rankings.

Original supporting fact article and sentence labels accessed only here;
retriever and inferred title links never saw gold. Score source-level AND
sentence-level evidence. Paired bootstrap CIs and McNemar exact test.
"""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,random,math
from scipy.stats import binomtest
HERE=Path(__file__).parent
SRC=Path('/data/tmp/kmd-hotpot-original-distractor-20261025.json')
PRED=HERE/'frozen_unlabelled_retrieval_20261025.json'
OUT=HERE/'hotpot_external_multidoc_gold_results_20261025.json'
D_SHA='4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea'
K=(2,3,4,6)
METHODS=('title_bm25','fulltext_bm25','title_first2_bm25','linked_secondhop')

def paired_ci(a,b,seed=20261025,nboot=1800):
 rng=random.Random(seed);n=len(a)
 vals=[(int(x)-int(y)) for x,y in zip(a,b)]
 samples=[]
 for _ in range(nboot):
  samples.append(sum(vals[rng.randrange(n)] for j in range(n))/n)
 samples.sort()
 return {'difference':round(sum(vals)/n,6),'bootstrap_95CI':[round(samples[int(0.025*nboot)],6),round(samples[int(0.975*nboot)],6)],
    'wins':sum(x>0 for x in vals),'losses':sum(x<0 for x in vals),'ties':sum(x==0 for x in vals),
    'mcnemar_exact_two_sided_p':binomtest(sum(x>0 for x in vals),sum(x!=0 for x in vals),0.5).pvalue if any(x!=0 for x in vals) else 1.0}

def main():
 if OUT.exists(): raise RuntimeError('Frozen score already done; no repeat or overwrite')
 assert hashlib.sha256(SRC.read_bytes()).hexdigest()==D_SHA
 source=json.loads(SRC.read_text());outputs=json.loads(PRED.read_text());preds=outputs['predictions']
 assert len(source)==len(preds)==7405
 buckets=defaultdict(list)
 allrows=[]
 for doc,pred in zip(source,preds):
  assert doc['_id']==pred['uid']
  assert hashlib.sha256(doc['question'].encode()).hexdigest()==pred['question_sha256']
  gold_titles={title for title,sent_index in doc['supporting_facts']}
  assert len(gold_titles)==2
  contexts={title:sents for title,sents in doc['context']}
  assert gold_titles<=set(contexts)
  rows={'uid':doc['_id'],'type':doc['type'],'difficulty':doc['level'],'answer_sha256':hashlib.sha256(doc['answer'].encode()).hexdigest(),
        'question_sha256':pred['question_sha256'],'supporting_doc_count':len(gold_titles),
        'gold_doc_titles':sorted(gold_titles), 'methods':{}}
  for name in METHODS:
   docs=pred['ranks'][name];assert len(docs)==min(6,len(doc['context']))
   ranked={str(k):{'both_docs':gold_titles<=set(docs[:k]),'at_least_one_doc':bool(gold_titles & set(docs[:k]))} for k in K}
   rows['methods'][name]=ranked
  # Original annotated sentence titles and 0-based positions, scored only in this phase.
  true_pairs={(title,int(index)) for title,index in doc['supporting_facts']}
  sent_summary={}
  for setting in ('top3','top3_neighborhood'):
   found={(title,i) for title,parts in pred['sentence_predictions'].items() for i in parts[setting]}
   known={pair for pair in found if pair in true_pairs}
   hit_docs={t for t,i in known}
   sent_summary[setting]={'gold_sentence_recall':len(known)/len(true_pairs),
    'all_gold_docs_with_one_sentence':len(hit_docs)==len(gold_titles),
    'any_gold_sentence':bool(known),
    'gold_sentence_count':len(known), 'gold_sentence_total':len(true_pairs)}
  rows['sentence_summary_for_top4_fulltext_docs']=sent_summary
  buckets[doc['type']].append(rows);buckets['overall'].append(rows);allrows.append(rows)
 def summarize(arr):
  out={'n':len(arr),'ranked_source':{}}
  for m in METHODS:
   out['ranked_source'][m]={str(k):{'both_docs':sum(x['methods'][m][str(k)]['both_docs'] for x in arr),
      'at_least_one_doc':sum(x['methods'][m][str(k)]['at_least_one_doc'] for x in arr)} for k in K}
  out['sentence_evidence_top4_fulltext_docs']={k:{'any_sentence':sum(x['sentence_summary_for_top4_fulltext_docs'][k]['any_gold_sentence'] for x in arr),
    'both_documents_at_least_one_sentence':sum(x['sentence_summary_for_top4_fulltext_docs'][k]['all_gold_docs_with_one_sentence'] for x in arr),
    'sentence_recall_total':sum(x['sentence_summary_for_top4_fulltext_docs'][k]['gold_sentence_count'] for x in arr),
    'sentence_gold_total':sum(x['sentence_summary_for_top4_fulltext_docs'][k]['gold_sentence_total'] for x in arr)} for k in ('top3','top3_neighborhood')}
  arr_a=[x['methods']['linked_secondhop']['2']['both_docs'] for x in arr]
  arr_b=[x['methods']['fulltext_bm25']['2']['both_docs'] for x in arr]
  out['linked_vs_fulltext_pair_at_2']=paired_ci(arr_a,arr_b)
  return out
 data={'external_gold':'Yang et al. 2018, HotpotQA distractor development, human-evidence doc IDs/sentence IDs, CC BY-SA 4.0',
       'source_sha256':D_SHA,'frozen_predictions_sha256':hashlib.sha256(PRED.read_bytes()).hexdigest(),
       'protocol':'Report exact both-human-annotated-gold-source accuracy at rank k, sentence evidence recall, and paired title-reference hop vs direct fulltext BM25 accuracy; no model answers, no unlicensed source replacement, no DRS.',
       'scores':{k:summarize(v) for k,v in buckets.items()},
       'interpretation_limits':['Source documents already among up to ten question-specific distractor paragraphs (oracle candidate document collection; 60 examples have fewer than ten), not full Wikipedia document retrieval','Human answer gold exists but is NOT evaluated in this lexical retrieval-only experiment','Pair bootstrap descriptive and exact McNemar on two methods fixed before looking at labels; no thresholds tuned','No published PMB DRS parser applied to this Wikipedia corpus; cannot compare actual DRS against same-condition retrieval','Two gold documents per question are human supporting fact titles, not independent experimental truth/medical science','Historical Wikipedia snapshot and answers may reflect benchmark popularity and annotation mistakes, not fully new discoveries'],
       'cases':allrows}
 OUT.write_text(json.dumps(data,ensure_ascii=False)+'\n')
 print('HOT_POT_GOLD',len(allrows),'types',[(k,len(v)) for k,v in buckets.items()])
 for typ,rec in data['scores'].items():
  print('\nTYPE',typ,'n',rec['n'])
  for m,d in rec['ranked_source'].items():print(m,'both@2',d['2']['both_docs'],'both@4',d['4']['both_docs'],'any@2',d['2']['at_least_one_doc'])
  print('paired linked vs fulltext@2',rec['linked_vs_fulltext_pair_at_2'])
  print('sentence evidence',rec['sentence_evidence_top4_fulltext_docs'])
if __name__=='__main__':main()
