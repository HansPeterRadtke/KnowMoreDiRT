"""True source gold vs trained-published neural DRS scoped PREDICATE CONTENT.

Unlike the earlier 'NEGATION present' presence test, trace operator's argument
box(es), descend through truth-functional/modal subboxes and compare lexical
concept predicates *inside operator scope*, not just whether a negation marker
was produced. Original PMB 3.0 author-corrected gold and model outputs are
untouched, no oracle predicate tags, no model re-generation or gold tuning.

This deliberately only scores lexical sense-labelled unary concepts, not all
DRS roles, variable identities, operator nesting, time/writer identity or
world truth. A high score is NOT full correct truth conditions, and a low score
can include lexical-sense errors unrelated to scope. Dev/test are separate.
"""
import hashlib,json,re
from pathlib import Path
from collections import Counter
HERE=Path(__file__).parent
GROOT=Path('/data/tmp/kmd-pmb-eval-20261024/data/pmb-3.0.0/gold')
PROOT=Path('/data/tmp/kmd-pmb-reference-20261024/output/pmb-3.0.0/en')
OUT=HERE/'original_pmb_scoped_predicate_content_20261025.json'
OPERATORS=('NEGATION','CONDITION','CONSEQUENCE','POSSIBILITY','ATTRIBUTION')
CHILD_OPS={'NEGATION','CONDITION','CONSEQUENCE','POSSIBILITY','NECESSITY','ATTRIBUTION','DISJUNCTION','EXPLANATION','RESULT','OR','IMP','POS','NEC'}
SHA={
 'test':{'gold':'aa5668fb150d3332f42fdee828457a89e3acdf2a92846b8b6e8f027654a032e5','pred':'7d56cceda381c66c9d890c5078161f22dedbaf3df002b04306a960e7bd8ed5a9','raw':'4bd557c2e6655d9b9c32269d4f6aa32415a314777bf16b517d7ae3da280f6263'},
 'dev':{'gold':'10575176eeffaa5b7fa50e2db2c73b526620543f62452a0a470ce845a7e29ad6','pred':'f9bbfa86e64d05d19a2416e20721dedbf8aba68daa477f77a4e67081bca76521','raw':'84372c8a91a107e6cc808eb57c2cca699ae8aec1df2e355d1ccb145d5e5fc2c3'}}

def blocks(f):return [x for x in re.split(r'\n\s*\n',f.read_text().strip()) if x.strip()]

def clauses(block):
 out=[]
 for line in block.splitlines():
  if not line.strip() or line.lstrip().startswith('%'):continue
  p=line.split('%',1)[0].strip().split()
  if len(p)>=3 and re.fullmatch(r'b\d+',p[0]):out.append(p)
 return out

def concept(line):
 # The authors' clausal convention: b1 live "v.01" x2 ; semantic sense in
 # quoted third field. Ignore unsensed proper names/roles and type unification.
 return (line[1].casefold(),line[2].lower()) if (len(line)>=4 and re.fullmatch(r'"[a-z]\.\d\d"',line[2]) and re.fullmatch(r'[A-Za-z][\w-]*',line[1])) else None

def concepts_in_box(clauses_,target):
 by={};nested={}
 for p in clauses_:
  by.setdefault(p[0],[]).append(p)
  if len(p)==3 and p[1] in CHILD_OPS and re.fullmatch(r'b\d+',p[2]):
   nested.setdefault(p[0],set()).add(p[2])
 # Do not follow PRESUPPOSITION edges; they often point out of negation scope.
 seen=set();pending=[target]
 while pending:
  b=pending.pop()
  if b in seen:continue
  seen.add(b)
  pending.extend(sorted(nested.get(b,set())-seen))
 result=Counter()
 for b in seen:
  for p in by.get(b,[]):
   c=concept(p)
   if c:result[c]+=1
 return result,seen

def operator_concepts(clauses_,op):
 scoped=Counter();targets=[]
 for p in clauses_:
  if len(p)>=3 and p[1]==op and re.fullmatch(r'b\d+',p[2]):
   target=p[2];c,visited=concepts_in_box(clauses_,target)
   scoped.update(c);targets.append({'box':target,'concepts':sorted([f'{a[0]}:{a[1]}' for a in c.elements()]),'descendants':sorted(visited)})
 return scoped,targets

def all_concepts(clauses_):
 return Counter(c for p in clauses_ if (c:=concept(p)))

def compare_count(pred,gold):
 common=sum((pred & gold).values());p=sum(pred.values());g=sum(gold.values())
 return {'gold_terms':g,'pred_terms':p,'matched':common,'missing':sum((gold-pred).values()),'spurious':sum((pred-gold).values()),'exact_match':pred==gold,
   'f1':(2*common/(p+g) if p+g else 1.0), 'recall':(common/g if g else None)}

def main():
 if OUT.exists():raise RuntimeError('Already measured scoped original benchmark; never overwrite')
 response={'source':'Original PMB 3.0 human-corrected author gold and published real automatically generated neural DRS, no regeneration',
           'source_git_revisions':{'gold':'RikVN/DRS_parsing@190838f9adea2af616eb76156d2e66c8565a6c6e','model_outputs':'RikVN/Neural_DRS@81bcb26dfcc08f0da999cc7beaa2952f8820860c'},
           'method':'For each operator in original gold and original neural model parse, traverse its target subordinate box and recursively nested box operators (excluding outward PRESUPPOSITION); compare sense-bearing lexical concepts attached to the governed scope, preserving multiplicity; exact scope/concept signature vs gold. No human scope tags supplied to model.',
           'results':{},'warnings':['Operator target box lexical concept comparison does NOT validate full DRS clause graph/negation truth conditions, argument role, temporal conditions or model prediction on research files.',
             'A wrong lemma/sense counts as a scope-content error even if the model found the correct scope; the complementary lemma-only comparison distinguishes some lexical-sense mistakes.',
             'Flattening lexical terms across multiple same-type boxes within a sentence may mask operator nesting and target assignment errors; both splits are source snapshots, not independently rerun model generations.',
             'PMB data are largely short standalone sentences, not scientific documents, multi-document identity or dream narratives; source annotations are linguistic meaning gold not empirically proven facts.',
             'These original dev/test splits are author-labelled, but authors and models have historically released test outputs; cannot infer completely unexposed model train partitions beyond the original dataset protocol.']}
 for split in ('test','dev'):
  g=GROOT/(split+'.txt');p=PROOT/split/'bert_char_1enc/output1.txt';raw=GROOT/(split+'.txt.raw')
  for key,f in (('gold',g),('pred',p),('raw',raw)):
   assert hashlib.sha256(f.read_bytes()).hexdigest()==SHA[split][key]
  gs,ps,rs=blocks(g),blocks(p),raw.read_text().splitlines()
  assert len(gs)==len(ps)==len(rs)
  metric={}
  for op in OPERATORS:
   rows=[]
   for i,(gb,pb,source) in enumerate(zip(gs,ps,rs)):
    gc,pc=clauses(gb),clauses(pb)
    gold,gt=operator_concepts(gc,op);pred,pt=operator_concepts(pc,op)
    if not gt and not pt:continue
    mc=compare_count(pred,gold)
    allpred=all_concepts(pc)
    lemma=lambda ctr:Counter({k[0]:sum(vv for kk,vv in ctr.items() if kk[0]==k[0]) for k in ctr})
    lc=compare_count(lemma(pred),lemma(gold))
    # Gold lexical term anywhere in model parse but not in governed scope is
    # evidence of a potential SCOPE-placement failure, not just missing lemma.
    leaked=sum((gold & (allpred-pred)).values())
    row={'index':i,'source_sentence':source,'source_sentence_sha256':hashlib.sha256(source.encode()).hexdigest(),
     'gold_box_targets':gt,'pred_box_targets':pt,'gold_operator':bool(gt),'model_operator':bool(pt),
     'sense_predicate_comparison':mc,'lemma_predicate_comparison':lc,
     'gold_lexical_content_present_elsewhere_but_outside_operator_scope':leaked}
    rows.append(row)
   aligned=[x for x in rows if x['gold_operator']]
   both=[x for x in rows if x['gold_operator'] and x['model_operator']]
   sem=[x for x in both if x['sense_predicate_comparison']['gold_terms']>0]
   def agg(subset,key):
    matched=sum(z[key]['matched'] for z in subset);p_total=sum(z[key]['pred_terms'] for z in subset);g_total=sum(z[key]['gold_terms'] for z in subset)
    return {'documents':len(subset),'gold_terms':g_total,'model_terms':p_total,'matched_terms':matched,
      'micro_precision':round(matched/p_total,6) if p_total else None,
      'micro_recall':round(matched/g_total,6) if g_total else None,
      'micro_f1':round(2*matched/(p_total+g_total),6) if p_total+g_total else None,
      'exact_sentence_scoped_inventory':sum(z[key]['exact_match'] for z in subset),
      'mean_per_sentence_f1':round(sum(z[key]['f1'] for z in subset)/len(subset),6) if subset else None}
   metric[op]={'original_gold_operator_instances':sum(len(x['gold_box_targets']) for x in rows),
     'original_model_operator_instances':sum(len(x['pred_box_targets']) for x in rows),
     'gold_sentences':len(aligned),'both_have_operator':len(both),
     'both_with_gold_lexical_content':len(sem),
     'scope_sense_all_gold_sentences':agg(aligned,'sense_predicate_comparison'),
     'scope_sense_both_with_gold_lexical_content':agg(sem,'sense_predicate_comparison'),
     'scope_lemma_both_with_gold_lexical_content':agg(sem,'lemma_predicate_comparison'),
     'gold_scoped_lexical_items_seen_elsewhere_in_model_but_not_in_matched_scope':sum(x['gold_lexical_content_present_elsewhere_but_outside_operator_scope'] for x in both),
     'examples_of_scope_or_lexical_error':[x for x in sem if not x['lemma_predicate_comparison']['exact_match']][:25],
     'examples_of_exact_match':[x for x in sem if x['sense_predicate_comparison']['exact_match']][:8]}
  response['results'][split]={'sentences':len(gs),'input_sha256':SHA[split],'operator_results':metric}
 out=json.dumps(response,indent=2,ensure_ascii=False)+'\n'
 OUT.write_text(out)
 print('VERIFIED_SNAPSHOT',split,'SENTENCES',len(gs))
 for op,o in metric.items():
  s=o['scope_sense_both_with_gold_lexical_content']; l=o['scope_lemma_both_with_gold_lexical_content']
  print('OPERATOR',op,'gold_sentences',o['gold_sentences'],'both',o['both_have_operator'],'with_content',o['both_with_gold_lexical_content'],
    'scoped_lemma_micro_f1',l['micro_f1'],'exact_lemma',l['exact_sentence_scoped_inventory'],'scoped_sense_f1',s['micro_f1'],
    'exact_sense',s['exact_sentence_scoped_inventory'],'out_of_scope_lexical_items',o['gold_scoped_lexical_items_seen_elsewhere_in_model_but_not_in_matched_scope'])
 print('OUTPUT_SHA256',hashlib.sha256(OUT.read_bytes()).hexdigest())
if __name__=='__main__':main()
