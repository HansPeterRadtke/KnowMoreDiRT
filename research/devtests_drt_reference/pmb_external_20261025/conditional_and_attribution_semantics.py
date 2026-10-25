"""Check published automatically-generated DRS truth-conditional *content*.

Original annotator gold untouched. This is a post-hoc model semantic diagnostic
with a predefined exact both-premises/consequence criterion, not a fabricated
truth value, learned classifier, or official end-to-end logical entailment.
Original zero-intervention source semantics must already have been extracted.
"""
import hashlib,json,re
from collections import Counter
from pathlib import Path
from operator_scope_predicate_audit import GROOT,PROOT,SHA,blocks,clauses,operator_concepts,all_concepts
HERE=Path(__file__).parent
OUT=HERE/'original_pmb_implication_attribution_scope_audit_20261025.json'
OPS=('CONDITION','CONSEQUENCE','ATTRIBUTION','NEGATION')

def lemma_count(c):
 result=Counter()
 for (lemma,sense),v in c.items():result[lemma]+=v
 return result

def semantic_sig(a,b,mode):
 if mode=='lemma':return lemma_count(a)==lemma_count(b)
 return a==b

def main():
 if OUT.exists():raise RuntimeError('Original inference-scope audit already performed; no overwrite')
 result={'method':'For every gold CONDITION/CONSEQUENCE pair, both human-gold boxes and both independently generated neural-parser boxes must contain the same MULTISET of sense-labelled concepts for exact sense consistency; separately compare concept lemma only, and operator presence. Same standard on separate test and dev splits, original input hashes checked.',
     'original_gold_git':'RikVN/DRS_parsing@190838f9adea2af616eb76156d2e66c8565a6c6e',
     'published_parser_git':'RikVN/Neural_DRS@81bcb26dfcc08f0da999cc7beaa2952f8820860c',
     'splits':{},
     'limitations':['Exact scoped lemmas and senses are NECESSARY but not SUFFICIENT for sound formal derivation: the clause graph, operator variable links, referents and quantifiers may still be wrong.',
       'Gold DRS semantic annotations are expert linguistic interpretations, not scientific reality; this is NOT formal automated proof of genuine novel empirical facts.',
       'This study does not test cross-document identity or dream and quote-world projection.','PMB dev/test source semantic gold was published with the neural parser original outputs; cannot guarantee complete absence of training/test exposure beyond authors release protocol.',
       'This comparison follows a previous operator benchmark on the same data; it is a stricter diagnostic, not a new heldout model evaluation.']}
 for split in ('test','dev'):
  gpath=GROOT/(split+'.txt');ppath=PROOT/split/'bert_char_1enc/output1.txt';rawpath=GROOT/(split+'.txt.raw')
  for key,p in (('gold',gpath),('pred',ppath),('raw',rawpath)):
   assert hashlib.sha256(p.read_bytes()).hexdigest()==SHA[split][key]
  g,p,rs=blocks(gpath),blocks(ppath),rawpath.read_text().splitlines();assert len(g)==len(p)==len(rs)
  rows=[];neg=[];attr=[]
  for i,(gblock,pblock,source) in enumerate(zip(g,p,rs)):
   gold=clauses(gblock);pred=clauses(pblock)
   gg={op:operator_concepts(gold,op) for op in OPS}
   pp={op:operator_concepts(pred,op) for op in OPS}
   if gg['CONDITION'][1] and gg['CONSEQUENCE'][1]:
    good=all(pp[o][1] for o in ('CONDITION','CONSEQUENCE'))
    exact_lemma=good and all(semantic_sig(gg[o][0],pp[o][0],'lemma') for o in ('CONDITION','CONSEQUENCE'))
    exact_sense=good and all(semantic_sig(gg[o][0],pp[o][0],'sense') for o in ('CONDITION','CONSEQUENCE'))
    # Whether whole model source sentence at least contains the gold semantic
    # lemmas anywhere. If true, wrong bound conditions indicate scope errors.
    anywhere=lemma_count(all_concepts(pred))
    gold_in_anywhere=all(len((lemma_count(gg[o][0]) - anywhere))==0 for o in ('CONDITION','CONSEQUENCE'))
    rows.append({'index':i,'sentence':source,'sha256':hashlib.sha256(source.encode()).hexdigest(),
     'gold_antecedent_terms':sorted(f'{lem}:{sense}' for (lem,sense) in gg['CONDITION'][0].elements()),
     'gold_consequent_terms':sorted(f'{lem}:{sense}' for (lem,sense) in gg['CONSEQUENCE'][0].elements()),
     'pred_antecedent_terms':sorted(f'{lem}:{sense}' for (lem,sense) in pp['CONDITION'][0].elements()),
     'pred_consequent_terms':sorted(f'{lem}:{sense}' for (lem,sense) in pp['CONSEQUENCE'][0].elements()),
     'both_operators_produced':good,'both_scopes_exact_lemma':exact_lemma,'both_scopes_exact_lemma_and_sense':exact_sense,
     'gold_scoped_lemmas_all_present_somewhere_in_generated_sentence':gold_in_anywhere})
   for op,container in [('NEGATION',neg),('ATTRIBUTION',attr)]:
    if gg[op][1]:
     container.append({'index':i,'sentence':source,'gold_operator':True,'model_operator':bool(pp[op][1]),
      'exact_lemma_target':bool(pp[op][1]) and semantic_sig(gg[op][0],pp[op][0],'lemma'),
      'exact_sense_target':bool(pp[op][1]) and semantic_sig(gg[op][0],pp[op][0],'sense')})
  def summ(x):
   return {'gold_cases':len(x),'produced_operator':sum(r['model_operator'] for r in x),
     'exact_lemma_semantic_target':sum(r['exact_lemma_target'] for r in x),
     'exact_lemma_and_sense_target':sum(r['exact_sense_target'] for r in x)}
  val={'sentences':len(g),'gold_antecedent_consequent_sentence_pairs':len(rows),
    'parser_has_both_conditional_boxes':sum(x['both_operators_produced'] for x in rows),
    'both_antecedent_consequent_scopes_exact_lemmas':sum(x['both_scopes_exact_lemma'] for x in rows),
    'both_antecedent_consequent_scopes_exact_lemmas_and_senses':sum(x['both_scopes_exact_lemma_and_sense'] for x in rows),
    'all_gold_scoped_concepts_present_anywhere_in_auto_generated_sentence':sum(x['gold_scoped_lemmas_all_present_somewhere_in_generated_sentence'] for x in rows),
    'original_attribution':summ(attr),'original_negation':summ(neg),
    'all_conditional_pair_rows':rows,
    'examples_of_incorrect_premise_or_consequent':[z for z in rows if not z['both_scopes_exact_lemma']][:12]}
  result['splits'][split]=val
  print('SPLIT',split,'N',len(g),'GOLD_CONDITIONAL_PAIRS',len(rows),'PARSER_BOTH_OPERATOR',val['parser_has_both_conditional_boxes'],
    'BOTH_SCOPED_LEMMAS_EXACT',val['both_antecedent_consequent_scopes_exact_lemmas'],
    'BOTH_SCOPED_SENSES_EXACT',val['both_antecedent_consequent_scopes_exact_lemmas_and_senses'],
    'ANYWHERE_GOLD_LEMMAS',val['all_gold_scoped_concepts_present_anywhere_in_auto_generated_sentence'])
  print('ATTR',val['original_attribution'],'NEG',val['original_negation'])
 OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print('RESULT_SHA256',hashlib.sha256(OUT.read_bytes()).hexdigest())
if __name__=='__main__':main()
