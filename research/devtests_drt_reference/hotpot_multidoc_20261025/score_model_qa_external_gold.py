"""Score COMPLETE model outputs vs untouched official HotpotQA development gold.

Model neither saw gold answers nor gold supporting article/sentence labels. No
prompt/ranking threshold retuning, and no rerunning expensive completed cases.
"""
from pathlib import Path
import hashlib,json,re,string,random
from collections import Counter,defaultdict
from scipy.stats import binomtest
HERE=Path(__file__).parent
DATA=Path('/data/tmp/kmd-hotpot-original-distractor-20261025.json')
FROZEN=HERE/'frozen_source_only_model_qa_20261025.json'
RAW=HERE/'raw_source_only_qa_qwen_20261025.jsonl'
OUT=HERE/'source_grounded_model_qa_results_20261025.json'
N=36

def normalize(s):
 s=s.lower().strip()
 s=''.join(c for c in s if c not in set(string.punctuation))
 s=re.sub(r'\b(a|an|the)\b',' ',s)
 return ' '.join(s.split())

def f1(pred,gold):
 a,b=normalize(pred).split(),normalize(gold).split()
 if a in (['yes'],['no'],['noanswer']) or b in (['yes'],['no'],['noanswer']):return float(a==b)
 common=sum((Counter(a)&Counter(b)).values())
 if not a and not b:return 1.0
 return 2*common/(len(a)+len(b)) if a and b else 0

def paired(a,b):
 dif=[int(x)-int(y) for x,y in zip(a,b)]
 wins=sum(x>0 for x in dif);loss=sum(x<0 for x in dif)
 from random import Random
 rng=Random(20261025);boot=[];n=len(dif)
 for j in range(2000):boot.append(sum(dif[rng.randrange(n)] for k in range(n))/n)
 boot.sort()
 return {'wins':wins,'losses':loss,'ties':n-wins-loss,'difference':round(sum(dif)/n,4),
   'paired_bootstrap_95CI':[round(boot[50],4),round(boot[1950],4)],
   'exact_mcnemar_p':binomtest(wins,wins+loss,0.5).pvalue if wins+loss else 1.0}

def parse_output(raw):
 obj=raw.get('output',{}).get('parsed')
 if not isinstance(obj,dict) or not isinstance(obj.get('answer'),str):return ('INVALID',None,None)
 return obj['answer'],obj.get('source_title'),obj.get('source_quote')

def main():
 if OUT.exists():raise RuntimeError('completed gold scored output exists; refusing rerun')
 f=json.loads(FROZEN.read_text());cases=f['records'];assert len(cases)==N
 a=[json.loads(l) for l in RAW.read_text().splitlines() if l];assert len(a)==N*2,'incomplete inference; do not score partial'
 by={(x['uid'],x['method']):x for x in a};assert len(by)==len(a)
 assert f['source_sha256']==hashlib.sha256(DATA.read_bytes()).hexdigest()
 allgold={x['_id']:x for x in json.loads(DATA.read_text())}
 rows=[]
 for case in cases:
  source=allgold[case['uid']];assert case['question']==source['question']
  gold=source['answer'];sg={t for t,k in source['supporting_facts']}
  gsp={(t,int(k)) for t,k in source['supporting_facts']}
  rec={'uid':case['uid'],'type':source['type'],'rank_change_group':case['rank_change_group'],'answer_gold':gold,
       'original_gold_article_titles':sorted(sg),'methods':{}}
  for method in f['methods']:
   output=by[(case['uid'],method)];pred,title,quote=parse_output(output)
   supplied=case['source_documents'][method];selected_titles={x['title'] for x in supplied}
   sent={(x['title'],i) for x in supplied for i in x['indices']}
   source_by_title={x['title']:x['text'] for x in supplied}
   grounded_title=isinstance(title,str) and title in selected_titles
   grounded_quote=bool(grounded_title and isinstance(quote,str) and quote.strip() and quote.strip() in source_by_title[title])
   case_out={'answer':pred,'normalized_answer':normalize(pred),'em':int(normalize(pred)==normalize(gold)),
      'f1':round(f1(pred,gold),5),'source_title':title,'source_quote':quote,
      'source_title_provided':grounded_title,'source_quote_exact':grounded_quote,
      'both_gold_articles_in_context':sg<=selected_titles,'one_gold_article_in_context':bool(sg & selected_titles),
      'gold_sentence_count_in_context':len(sent & gsp), 'all_gold_sentences_in_context':gsp<=sent,
      'any_gold_sentence_in_context':bool(gsp&sent),
      'source_documents_provided':[x['title'] for x in supplied],
      'seconds':output['output'].get('seconds'),'model_finish_reason':output['output'].get('stop_reason'),
      'raw_output_sha256':hashlib.sha256(str(output['output'].get('raw')).encode()).hexdigest()}
   rec['methods'][method]=case_out
  rows.append(rec)
 def summ(subset):
  z={'n':len(subset),'metrics':{}}
  for m in f['methods']:
   x=[r['methods'][m] for r in subset]
   z['metrics'][m]={'em_n':sum(v['em'] for v in x),'f1_avg':round(sum(v['f1'] for v in x)/len(x),5),
       'both_gold_docs':sum(v['both_gold_articles_in_context'] for v in x),
       'any_gold_sentence':sum(v['any_gold_sentence_in_context'] for v in x),
       'source_quote_exact':sum(v['source_quote_exact'] for v in x),
       'answer_correct_with_any_source_quote':sum(v['em'] and v['source_quote_exact'] for v in x),
       'invalid_outputs':sum(v['answer']=='INVALID' for v in x),
       'total_seconds':round(sum(v['seconds'] or 0 for v in x),2)}
  z['linked_minus_fulltext_em_paired']=paired([r['methods']['linked_secondhop']['em'] for r in subset],[r['methods']['fulltext_bm25']['em'] for r in subset])
  return z
 # Negative/sham control: both methods supplied EXACTLY THE SAME two source
 # articles/excerpts. Any answer divergence here cannot be attributed to
 # adding a title link, and must remain reported as runtime instability.
 by_uid={x['uid']:x for x in cases}
 unchanged=[x for x in rows if x['rank_change_group'].endswith('_same')]
 sham={'n_identical_source_cases':len(unchanged),
    'identical_original_input_docs':sum(by_uid[x['uid']]['source_documents']['fulltext_bm25']==by_uid[x['uid']]['source_documents']['linked_secondhop'] for x in unchanged),
    'identical_answer_strings':sum(x['methods']['fulltext_bm25']['answer']==x['methods']['linked_secondhop']['answer'] for x in unchanged),
    'identical_citation_quotes':sum(x['methods']['fulltext_bm25']['source_quote']==x['methods']['linked_secondhop']['source_quote'] for x in unchanged),
    'warning':'Both model methods see the same source and prompt in this group: differences are not link-induced retrieval effects.'}
 assert sham['n_identical_source_cases']==12
 buckets=defaultdict(list)
 for r in rows:
  buckets['overall'].append(r);buckets[r['type']].append(r);buckets[r['rank_change_group']].append(r)
 payload={'original_source_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
   'frozen_input_sha256':hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
   'verbatim_model_predictions_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),
   'preselected_sample_size':N,'methods':['fulltext_bm25','linked_secondhop'],
   'external_gold':'HotpotQA distractor development 2018, author answer and supporting article/sentence human annotations, CC BY-SA 4.0',
   'analysis_scope':'Genuine two-document historical Wikipedia source-grounded answer by same local Qwen CPU, same fixed prompt/settings and two automatic source snippets; compare retrieval selection, NOT DRS',
   'sham_control_equal_source_inputs':sham,
   'buckets':{k:summ(v) for k,v in buckets.items()},
   'limitations':['Subset selected after earlier 7405-case retrieval-level aggregate gold was examined, but exact model-case IDs were chosen by predeclared per-id hash and rank-change group, with individual answer gold unseen until scoring',
     'Pretrained Qwen can have seen HotpotQA gold text in training, despite user instruction to rely only on sources',
     'Model-generated source quote validity does not prove semantic implication or correct cross-document reasoning',
     'Both methods have oracle ten-candidate-document context and source-specific title; no full-web discovery',
     'Not a source-vs-DRS comparison, only direct source retrieval method ablation; no scientific novelty or falsification'],
   'cases':rows}
 OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 for k,z in payload['buckets'].items():
  print(k,'N',z['n'],'full',z['metrics']['fulltext_bm25'],'linked',z['metrics']['linked_secondhop'],'paired',z['linked_minus_fulltext_em_paired'])
if __name__=='__main__':main()
