"""Independent integrity-verified summary of ORIGINAL PMB parsed operator CONTENT.

Reports Wilson intervals for gold-exact scoped content, unlike operator-only
F1, and probes whether predicate-role-name triples remain bound correctly.
No new semantic model calls, gold edits or test parameter optimization.
"""
import hashlib,json
from pathlib import Path
from scipy.stats import binomtest
HERE=Path(__file__).parent
SNAP=HERE/'original_pmb_scoped_predicate_content_20261025.json'
PAIR=HERE/'original_pmb_implication_attribution_scope_audit_20261025.json'
ROLES=HERE/'pmb_named_referents_in_scoped_events_20261025.json'
OUT=HERE/'drs_semantic_necessary_conditions_summary_20261025.json'

def ci(k,n):
 assert 0<=k<=n
 r=binomtest(k,n).proportion_ci(confidence_level=0.95,method='wilson')
 return {'correct':k,'gold_cases':n,'fraction':round(k/n,6),'Wilson95':[round(r.low,6),round(r.high,6)]}

def main():
 if OUT.exists():raise RuntimeError('Do not overwrite original semantic summary')
 a=json.loads(SNAP.read_text());b=json.loads(PAIR.read_text());c=json.loads(ROLES.read_text())
 res={'source_gold_and_published_model_sha256':a['results']['test']['input_sha256'],
      'source_score_sha256':{z.name:hashlib.sha256(z.read_bytes()).hexdigest() for z in (SNAP,PAIR,ROLES)},
      'model_published_ref':'https://github.com/RikVN/Neural_DRS/tree/81bcb26dfcc08f0da999cc7beaa2952f8820860c',
      'gold_pmb_ref':'https://github.com/RikVN/DRS_parsing/tree/190838f9adea2af616eb76156d2e66c8565a6c6e',
      'scope':'Model **generated** DR clauses vs human-corrected original DRS semantics. Does NOT establish full truth-conditions or scientific discovery.',
      'splits':{},
      'caveats':['Paired dev and test are from the same PMB source collection, not independent historical corpora or producer institutions',
       'Exact predicate content is a necessary condition for some kinds of formal inference, not a sufficient condition, so figures must not be interpreted as reasoner accuracy.',
       'Word-sense differences inflate exact lexical-semantic error counts compared with coarser meaning-equivalence metrics; the lemma-only result is separately reported.',
       'Referent-role errors are checked only when a named entity and an in-scope event role are found by the restrictive parser, NOT all coreference.',
       'Wilson intervals assume these benchmark sentences behave as independent Bernoulli trials; shared annotation/model biases are not covered.']}
 for split in ('test','dev'):
  p=b['splits'][split];neg=p['original_negation'];att=p['original_attribution'];name=c['splits'][split]['operators']
  condition={'both_condition_and_consequence_operators':ci(p['parser_has_both_conditional_boxes'],p['gold_antecedent_consequent_sentence_pairs']),
    'exact_predicate_lemmas_on_both_sides':ci(p['both_antecedent_consequent_scopes_exact_lemmas'],p['gold_antecedent_consequent_sentence_pairs']),
    'exact_predicate_lemmas_and_senses_on_both_sides':ci(p['both_antecedent_consequent_scopes_exact_lemmas_and_senses'],p['gold_antecedent_consequent_sentence_pairs'])}
  out={'original_gold_sentences':p['sentences'],
    'negation':{'detected':ci(neg['produced_operator'],neg['gold_cases']),
      'same_scoped_predicate_lemmas':ci(neg['exact_lemma_semantic_target'],neg['gold_cases']),
      'same_scoped_predicate_lemmas_and_senses':ci(neg['exact_lemma_and_sense_target'],neg['gold_cases'])},
    'if_then':condition,
    'attributed_scope':{'detected':ci(att['produced_operator'],att['gold_cases']),
      'same_attributed_predicate_lemmas':ci(att['exact_lemma_semantic_target'],att['gold_cases']),
      'same_attributed_predicate_lemmas_and_senses':ci(att['exact_lemma_and_sense_target'],att['gold_cases'])},
    'named_actor_or_location_in_negated_event':{'cases':name['NEGATION']['gold_source_sentences_with_named_scoped_event_roles'],
      'correct_event_predicate_role_and_name':ci(name['NEGATION']['full_named_predicate_role_lemma_exact'],name['NEGATION']['gold_source_sentences_with_named_scoped_event_roles']),
      'correct_event_predicate_role_name_and_sense':ci(name['NEGATION']['full_named_predicate_role_with_sense_exact'],name['NEGATION']['gold_source_sentences_with_named_scoped_event_roles'])}}
  res['splits'][split]=out
 OUT.write_text(json.dumps(res,indent=2)+'\n')
 for split,out in res['splits'].items():
  print('SPLIT',split,'SENT',out['original_gold_sentences'])
  for typ,group in out.items():
   if not isinstance(group,dict):continue
   for metric,val in group.items():
    if isinstance(val,dict) and 'fraction' in val:print('  ',typ,metric,str(val['correct'])+'/'+str(val['gold_cases']),'CI',val['Wilson95'])
 print('SOURCE_HASHES_VERIFIED_OUTPUT',hashlib.sha256(OUT.read_bytes()).hexdigest())
if __name__=='__main__':main()
