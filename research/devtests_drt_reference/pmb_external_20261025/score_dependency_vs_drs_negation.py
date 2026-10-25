"""Gold scoring of FROZEN source-only dependency semantic baseline vs neural DRS.

Human gold is consulted only here, after predictions were persisted and hashed.
The task is IDENTICAL for both: source sentence negated in-scope VERB/ADJECTIVE
lexical lemma multiset, not negation cue alone. Includes exact hit, micro F1,
false-positive predictions on other sentences and paired 95% intervals.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,random
from scipy.stats import binomtest
from operator_scope_predicate_audit import GROOT,PROOT,SHA,blocks,clauses,operator_concepts
HERE=Path(__file__).parent
F=HERE/'frozen_spacy_dependency_predictions_20261025.json'
OUT=HERE/'spacy_dependency_vs_published_drs_gold_results_20261025.json'
F_SHA='fee8800b45fe36cff3c332262bfdcb34e6231ed7149697cc837c2115eebea1c3'

def formal_va(g):
 c,_=operator_concepts(clauses(g),'NEGATION')
 result=Counter()
 for (lemma,sense),n in c.items():
  if sense.startswith(('"v.','"a.')):result[lemma]+=n
 return result

def f1(a,b):
 hit=sum((a&b).values());n=sum(a.values())+sum(b.values());return 2*hit/n if n else 1.0

def confidence(paired):
 n=len(paired);dif=[int(x[0])-int(x[1]) for x in paired]
 rng=random.Random(20261025);boots=[]
 for k in range(2500):boots.append(sum(dif[rng.randrange(n)] for _ in range(n))/n)
 boots.sort();win=sum(d>0 for d in dif);lose=sum(d<0 for d in dif)
 return {'n':n,'DRS_minus_dependency_exact_difference':round(sum(dif)/n,5),'paired_bootstrap95':[round(boots[62],5),round(boots[2437],5)],
    'DRS_wins':win,'dependency_wins':lose,'ties':n-win-lose,
    'mcnemar_two_sided_p':float(binomtest(win,win+lose,0.5).pvalue) if win+lose else 1.0}

def micro(rows,m):
 p=sum(x[m+'_pred_terms'] for x in rows)
 g=sum(x['gold_terms'] for x in rows)
 t=sum(x[m+'_matches'] for x in rows)
 return {'source_sentences':len(rows),'predicted_terms':p,'gold_terms':g,'matched_gold_terms':t,
    'precision':round(t/p,5) if p else None,'recall':round(t/g,5) if g else None,
    'micro_f1':round(2*t/(p+g),5) if p+g else None,
    'exact_gold_lemma_set':sum(x[m+'_exact'] for x in rows),
    'per_source_f1_mean':round(sum(x[m+'_f1'] for x in rows)/len(rows),5) if rows else None}

def main():
 if OUT.exists():raise RuntimeError('Already assessed frozen source-only baseline; no repeat')
 assert hashlib.sha256(F.read_bytes()).hexdigest()==F_SHA
 f=json.loads(F.read_text());result={'source_frozen_sha256':F_SHA,
     'source_only_dependency_model':f['model']+'@'+f['spacy_model_version'],
     'published_semantic_model':'RikVN/Neural_DRS@81bcb26dfcc08f0da999cc7beaa2952f8820860c, bert_char_1enc/output1.txt',
     'expert_gold':'RikVN/DRS_parsing@190838f9adea2af616eb76156d2e66c8565a6c6e PMB 3.0',
     'identical_task_for_both':'VERB and ADJ lemma multiset inside human-corrected negated scope. Each method returns candidate negated verb/adjective lemmas from input text. Source-only dependency raw model never receives DRS annotations.',
     'results':{},
     'limitations':['Source-only dependency baseline is a syntactic structured parse, not a full provenance-neighbor retrieval system, so performance differences here cannot isolate DRS versus the full KMD memory architecture.',
        'All PMB splits have been publicly published and have been previously used in this chat as operator diagnostics; this is a stricter fixed comparison, not pristine new heldout end-to-end benchmark.',
        'Target V/ADJ subset excludes many true nominal negations, presuppositions and modal scope, and exact lexical matching can misrepresent equivalent paraphrases and sense errors.',
        'One source cue can generate more than one predicted term, scores retain false-positive candidates in original non-gold-negated source text.',
        'Neither parser output proves empirical scientific truth, relevance to an author question or new falsifiable insight.']}
 for split in ('test','dev'):
  gp=GROOT/(split+'.txt');pp=PROOT/split/'bert_char_1enc/output1.txt';sent=GROOT/(split+'.txt.raw')
  for key,path in (('gold',gp),('pred',pp),('raw',sent)):
   assert hashlib.sha256(path.read_bytes()).hexdigest()==SHA[split][key]
  gs,ps,ss=blocks(gp),blocks(pp),sent.read_text().splitlines();d=f['splits'][split]['predictions']
  assert len(gs)==len(ps)==len(ss)==len(d)
  rows=[]
  for i,(gold_block,parser_block,source,dep) in enumerate(zip(gs,ps,ss,d)):
   assert i==dep['sentence_index']
   assert hashlib.sha256(source.encode()).hexdigest()==dep['source_sha256']
   gold=formal_va(gold_block);model=formal_va(parser_block);base=Counter(dep['predicted_negated_va_lemmas'])
   x={'index':i,'sentence':source,'sentence_sha256':hashlib.sha256(source.encode()).hexdigest(),
      'gold_terms':sum(gold.values()),'gold_verb_adj_negated_lemmas':sorted(gold.elements()),
      'model_predicted_lemmas':sorted(model.elements()),'dependency_predicted_lemmas':sorted(base.elements()),
      'source_only_dependency_trace':dep['source_cues_and_dependency_trace']}
   for typ,c in (('drs',model),('dependency',base)):
    x[typ+'_pred_terms']=sum(c.values());x[typ+'_matches']=sum((gold&c).values());x[typ+'_exact']=(c==gold)
    x[typ+'_f1']=round(f1(gold,c),5)
   rows.append(x)
  pos=[x for x in rows if x['gold_terms']>0]
  neg=[x for x in rows if x['gold_terms']==0]
  out={'original_source_sentence_count':len(rows),'gold_sentences_with_a_negated_verb_or_adjective':len(pos),
    'gold_empty_target_source_sentences':len(neg),
    'strict_on_gold_positive':{'neural_DRS':micro(pos,'drs'),'source_only_dependency':micro(pos,'dependency'),
       'paired_exact_source':confidence([(x['drs_exact'],x['dependency_exact']) for x in pos])},
    'false_positive_on_gold_empty_target':{'neural_DRS':sum(x['drs_pred_terms']>0 for x in neg),
        'source_only_dependency':sum(x['dependency_pred_terms']>0 for x in neg)},
    'all_sentences':{'neural_DRS':micro(rows,'drs'),'source_only_dependency':micro(rows,'dependency')},
    'original_source_misses_retrieved':[x for x in pos if (not x['dependency_exact'] or not x['drs_exact'])][:55],
    'source_only_correct_but_drs_incorrect':[x for x in pos if x['dependency_exact'] and not x['drs_exact']][:30],
    'drs_correct_but_source_only_incorrect':[x for x in pos if x['drs_exact'] and not x['dependency_exact']][:30]}
  result['results'][split]=out
  print('SPLIT',split,'TOTAL',len(rows),'GOLD_POSITIVE',len(pos),'GOLD_NEGATED_VA_CONCEPTS',sum(x['gold_terms'] for x in pos))
  for name,z in out['strict_on_gold_positive'].items():print('  ',name,z)
  print('FALSE_POSITIVE_GOLD_EMPTY DRS=',out['false_positive_on_gold_empty_target']['neural_DRS'],'DEPENDENCY=',out['false_positive_on_gold_empty_target']['source_only_dependency'])
 OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print('GOLD_SCORE_SOURCE_SHA256',hashlib.sha256(OUT.read_bytes()).hexdigest())
if __name__=='__main__':main()
