"""Source-grounded POST-HOC diagnostic of literal title-reference ceiling.

Human gold consulted only after original ranking frozen and scored; this is a
forensic path-availability audit, not a new tuned retrieval benchmark or a
proof that an automatically inferred alias/coreference would work.
"""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib
HERE=Path(__file__).parent
SRC=Path('/data/tmp/kmd-hotpot-original-distractor-20261025.json')
PRED=HERE/'frozen_unlabelled_retrieval_20261025.json'
SCORE=HERE/'hotpot_external_multidoc_gold_results_20261025.json'
OUT=HERE/'literal_identity_reference_ceiling_20261025.json'

def run():
 if OUT.exists():raise RuntimeError('previous output exists')
 s=json.loads(SRC.read_text());p=json.loads(PRED.read_text())['predictions'];gold=json.loads(SCORE.read_text())
 assert len(s)==len(p)==len(gold['cases'])==7405
 by=Counter();examples=defaultdict(list)
 for row,r,g in zip(s,p,gold['cases']):
  if row['type']!='bridge':continue
  assert row['_id']==r['uid']==g['uid']
  golds=set(g['gold_doc_titles']);first=r['ranks']['fulltext_bm25'][0]
  if first not in golds:
   by['first_ranked_article_wrong']+=1
   key='first_wrong'
  else:
   by['first_ranked_article_right']+=1
   other=next(iter(golds-{first}));source={d[0]:d[1] for d in row['context']}
   mention=any((' '+other.lower()+' ') in (' '+sentence.lower()+' ') for sentence in source[first])
   recovered=(r['ranks']['linked_secondhop'][1]==other)
   key=('literal_present_' if mention else 'literal_absent_')+('linked_selected_gold' if recovered else 'linked_missed_gold')
   by[key]+=1
   # Character-level evidence link, no external semantic interpretation.
   if mention:by['gold_other_article_name_explicitly_in_first_source']+=1
   else:by['gold_other_article_name_not_literal_in_first_source']+=1
  if len(examples[key])<20:
   examples[key].append({'uid':row['_id'],'question':row['question'],
     'human_gold_source_titles':sorted(golds),'top_first_title':first,'top_second_base':r['ranks']['fulltext_bm25'][1],
     'top_second_link':r['ranks']['linked_secondhop'][1],
     'first_source_support_excerpt':[source for title,i in row['supporting_facts'] if title==first for source in ([dict(row['context'])[title][i][:170]] if i<len(dict(row['context'])[title]) else [])][:2]})
 assert by['first_ranked_article_wrong']+by['first_ranked_article_right']==5918
 assert by['gold_other_article_name_not_literal_in_first_source']+by['gold_other_article_name_explicitly_in_first_source']==by['first_ranked_article_right']
 out={'original_source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),
  'original_frozen_predictions_sha256':hashlib.sha256(PRED.read_bytes()).hexdigest(),
  'posthoc_gold_score_sha256':hashlib.sha256(SCORE.read_bytes()).hexdigest(),
  'note':'After-gold retrieval ceiling analysis; source actual first chosen from model-free fulltext BM25; gold other Wikipedia title exact string present/absent in first article body. Identity aliases may still be present; do not label all absent titles as semantically unlinked.',
  'bridge_cases':5918,'counts':dict(by),
  'fraction_gold_other_title_not_literal_given_first_gold':round(by['gold_other_article_name_not_literal_in_first_source']/by['first_ranked_article_right'],6),
  'case_examples':{k:sorted(v,key=lambda x:hashlib.sha256(x['uid'].encode()).hexdigest())[:5] for k,v in examples.items()},
  'limitations':['Posthoc descriptive diagnostic, not a fresh heldout after seeing labels','An absent full proper title can still be linked via pronoun, shortened surname, alias, indirect relationship, titles and nicknames','No automatic semantic/DRS coreference detection was measured or validated on these 4085 cases','The first source is chosen from ten already offered sources, not a global corpus','Source text is 2018 version and gold labels reflect benchmark, not new verified scientific facts']}
 OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print('GOLD_BRIDGE',out['bridge_cases'],'COUNTS',out['counts'],'MISSING_LITERAL_RATE',out['fraction_gold_other_title_not_literal_given_first_gold'])
 for k,v in out['case_examples'].items():
  print('EXAMPLE',k,v[0]['question'][:130] if v else '-')
if __name__=='__main__':run()
