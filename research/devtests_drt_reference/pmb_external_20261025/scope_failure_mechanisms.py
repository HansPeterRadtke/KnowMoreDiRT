"""Read-only categorization of PMB automatic parse error causes.

For gold operator-scoped lemma inventories, separate source lexical omission
from a correct lexical item appearing elsewhere but outside governing DRS
scope, as well as spurious model-scoped lemmas. Categories can overlap.
Original independent human gold/model DRS source SHA asserted, no retuning.
"""
from pathlib import Path
from collections import Counter
import hashlib,json
from operator_scope_predicate_audit import GROOT,PROOT,SHA,blocks,clauses,operator_concepts,all_concepts
HERE=Path(__file__).parent
OUT=HERE/'original_pmb_semantic_scope_error_mechanisms_20261025.json'
OPS=('NEGATION','CONDITION','CONSEQUENCE','ATTRIBUTION')

def bylemma(counter):
 out=Counter()
 for (word,sense),count in counter.items():out[word]+=count
 return out

def main():
 if OUT.exists():raise RuntimeError('Do not repeat/overwrite existing source error taxonomy')
 result={'method':'Original human-gold DRS operator target concept inventory vs original neural DRS target and any model sentence lexical concept. Gold-only and model-only mismatch categories MAY overlap.',
 'original_sources':'RikVN/DRS_parsing@190838f9adea2af616eb76156d2e66c8565a6c6e and RikVN/Neural_DRS@81bcb26dfcc08f0da999cc7beaa2952f8820860c','splits':{},
 'limitations':['A gold scoped lemma absent from auto DRS can be lexical concept extraction error; a matching word outside the operator box suggests scope placement but does not establish full graph equivalence.',
  'Categories overlap and should never be summed as an error rate. These are linguistic gold, not empirical world-truth controls.',
  'No automatically generated model inference or new model test is done in this follow-up diagnostic.']}
 for split in ('test','dev'):
  gp=GROOT/(split+'.txt');pp=PROOT/split/'bert_char_1enc/output1.txt';rp=GROOT/(split+'.txt.raw')
  for key,p in (('gold',gp),('pred',pp),('raw',rp)):
   assert hashlib.sha256(p.read_bytes()).hexdigest()==SHA[split][key]
  golds,preds,sents=blocks(gp),blocks(pp),rp.read_text().splitlines()
  assert len(golds)==len(preds)==len(sents)
  scores={}
  for op in OPS:
   errors=[];tot=Counter()
   for i,(g,p,s) in enumerate(zip(golds,preds,sents)):
    g_orig,_gt=operator_concepts(clauses(g),op)
    gold_boxes=operator_concepts(clauses(g),op)[1]
    if not gold_boxes:continue
    pred_orig,pr_boxes=operator_concepts(clauses(p),op)
    gold,pred=bylemma(g_orig),bylemma(pred_orig)
    all_lemma=bylemma(all_concepts(clauses(p)))
    omitted=gold-all_lemma
    missing_scoped=gold-pred
    misplaced=(missing_scoped & (all_lemma-pred))
    extra=pred-gold
    exact=gold==pred
    types={'exact_gold_lemma_scope':exact,
      'no_operator':not bool(pr_boxes),
      'at_least_one_gold_lemma_not_extracted_anywhere':bool(omitted),
      'gold_lemma_present_but_outside_correct_scoped_box':bool(misplaced),
      'spurious_lemma_inside_scoped_box':bool(extra)}
    tot.update(k for k,v in types.items() if v)
    if not exact:
     errors.append({'index':i,'sentence':s,'classification':types,'gold_lemmas':list(gold.elements()),'model_scoped_lemmas':list(pred.elements()),
        'missing_from_any_model_box':list(omitted.elements()),'possibly_misscoped':list(misplaced.elements()),'model_spurious_in_scope':list(extra.elements())})
   scores[op]={'gold_source_sentences':sum(1 for i,(g,p,s) in enumerate(zip(golds,preds,sents)) if operator_concepts(clauses(g),op)[1]),
       'nonexclusive_error_cause_counts':dict(tot), 'source_linked_mismatch_examples':errors[:18]}
  result['splits'][split]={'input_sha256':SHA[split],'operators':scores}
  print('SPLIT',split)
  for op,v in scores.items():print(op,v['gold_source_sentences'],v['nonexclusive_error_cause_counts'])
 OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print('RESULT_HASH',hashlib.sha256(OUT.read_bytes()).hexdigest())
if __name__=='__main__':main()
