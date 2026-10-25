"""Measure named referent -> predicate role inside source-scoped DRS operators.

Unlike matching scoped lemma alone, this checks event-to-entity *identity* via
roles and a named participant, despite alpha-renamed DRS variable IDs.
PMB annotated 'Name' and semantic roles are human-corrected external gold;
trained neural DRS outputs are original published bytes, not oracle structures.
Named proper referents are collected from full sentence (including accessible
presuppositions) and linked to predicates *inside* a negation/attribution or
conditional scope. Partial necessary evidence of actor-binding correctness.
"""
import hashlib,json,re
from collections import Counter
from pathlib import Path
from operator_scope_predicate_audit import GROOT,PROOT,SHA,blocks,clauses,concepts_in_box
HERE=Path(__file__).parent
OUT=HERE/'pmb_named_referents_in_scoped_events_20261025.json'
OPS=('NEGATION','CONDITION','CONSEQUENCE','ATTRIBUTION','POSSIBILITY')
# Avoid labeling variable unification, time reference, or other non-role links
# as human agent/action bindings.
ROLE_NAMES={'Agent','Patient','Theme','Experiencer','Stimulus','Source','Location','Destination','Recipient','Beneficiary','Owner','Pivot','User','Co-Theme','Co-Agent','Creator','Causer','Participant','Bearer','Goal'}

def names_and_lexical_event_signatures(c):
 names={}
 for a in c:
  if len(a)>=4 and a[1]=='Name' and re.fullmatch(r'x\d+',a[2]):
   key=a[3].strip('"').strip().lower()
   if not key or key=='?' or len(key)<2:continue
   names.setdefault(a[2],set()).add(key)
 return names

def scoped_links(c,op):
 names=names_and_lexical_event_signatures(c)
 out=Counter();targets=[]
 for line in c:
  if len(line)>=3 and line[1]==op and re.fullmatch(r'b\d+',line[2]):
   _,visited=concepts_in_box(c,line[2]);targets.append(line[2])
   verbs={}
   for x in c:
    if x[0] in visited and len(x)>=4 and re.fullmatch(r'"[a-z]\.\d\d"',x[2]):
     verbs.setdefault(x[3],set()).add((x[1].casefold(),x[2].lower()))
   for x in c:
    if len(x)>=4 and x[0] in visited and x[1] in ROLE_NAMES and x[2] in verbs and x[3] in names:
     for event,ws in ((x[2],verbs[x[2]]),):
      for lemma,sense in ws:
       for name in names[x[3]]:
        # DRS can place names outside subordinate box; referent variable still
        # links the scope event to named entity in source sentence.
        out[(lemma,sense,x[1],name)]+=1
 return out,len(targets)

def no_sense(c):
 x=Counter()
 for (lemma,sense,role,name),v in c.items():x[(lemma,role,name)]+=v
 return x

def main():
 if OUT.exists():raise RuntimeError('already evaluated named-referent scope')
 result={'method':'Join published DRS scope box -> scoped event lexical predicate + semantic role -> original Name referent in same sentence; normalize variable names by proper name and compare role/predicate/name multiset against independently human-corrected source gold.',
  'source_revisions':{'published_neural':'RikVN/Neural_DRS@81bcb26dfcc08f0da999cc7beaa2952f8820860c','expert_gold':'RikVN/DRS_parsing@190838f9adea2af616eb76156d2e66c8565a6c6e'},'splits':{},
  'limitations':['Named referent gold covers a small subset of all sentences and excludes unnamed pronouns, anaphoric equality resolution, names assigned to multiple entities, event referent unification, and all cross-document identity.','Sense-labeled concepts can include event/type noun rather than only action, and role mapping can require semantic generalization; this is a diagnostic subset, NOT whole truth conditions.',
    'A NAME clause outside a negated box does not imply that the named entity is being negated; only relation to the scoped lexical event is scored.',
    'PMB short sentences are not new scientific documents, dream-law contexts or verification of truth. Model output is a published real automatic parse, not KMD production.']}
 for split in ('test','dev'):
  g=GROOT/(split+'.txt');p=PROOT/split/'bert_char_1enc/output1.txt';raw=GROOT/(split+'.txt.raw')
  for name,path in (('gold',g),('pred',p),('raw',raw)):
   assert hashlib.sha256(path.read_bytes()).hexdigest()==SHA[split][name]
  gs,ps,ss=blocks(g),blocks(p),raw.read_text().splitlines();assert len(gs)==len(ps)==len(ss)
  scopes={}
  for op in OPS:
   rows=[]
   for i,(a,b,text) in enumerate(zip(gs,ps,ss)):
    gold,ng=scoped_links(clauses(a),op);pred,np=scoped_links(clauses(b),op)
    if not gold:continue
    gsimple,p_simple=no_sense(gold),no_sense(pred)
    golditems=sum(gold.values());matched=sum((gold&pred).values())
    glemitems=sum(gsimple.values());lemmatched=sum((gsimple&p_simple).values())
    rows.append({'index':i,'source_sentence':text,'source_sha256':hashlib.sha256(text.encode()).hexdigest(),
       'gold_operator_boxes':ng,'generated_operator_boxes':np,
       'gold_named_referent_links':[list(z) for z in gold.elements()],
       'generated_named_referent_links':[list(z) for z in pred.elements()],
       'correct_links_with_sense':matched,'gold_links_with_sense':golditems,'pred_links_with_sense':sum(pred.values()),
       'correct_links_lemma_only':lemmatched,'gold_links_lemma_only':glemitems,'pred_links_lemma_only':sum(p_simple.values()),
       'all_gold_named_role_predicates_exact_lemma':gsimple==p_simple,
       'all_gold_named_role_predicates_exact_sense':gold==pred})
   gt=sum(r['gold_links_lemma_only'] for r in rows);pt=sum(r['pred_links_lemma_only'] for r in rows);mt=sum(r['correct_links_lemma_only'] for r in rows)
   gst=sum(r['gold_links_with_sense'] for r in rows);pst=sum(r['pred_links_with_sense'] for r in rows);mst=sum(r['correct_links_with_sense'] for r in rows)
   scopes[op]={'gold_source_sentences_with_named_scoped_event_roles':len(rows),
      'gold_named_source_event_role_links':gt,
      'original_auto_scope_operator_detected':sum(r['generated_operator_boxes']>0 for r in rows),
      'full_named_predicate_role_lemma_exact':sum(r['all_gold_named_role_predicates_exact_lemma'] for r in rows),
      'full_named_predicate_role_with_sense_exact':sum(r['all_gold_named_role_predicates_exact_sense'] for r in rows),
      'micro_f1_lemma_name_role':round(2*mt/(gt+pt),6) if gt+pt else None,
      'micro_f1_sense_name_role':round(2*mst/(gst+pst),6) if gst+pst else None,
      'individual_evidence_errors':[r for r in rows if not r['all_gold_named_role_predicates_exact_lemma']][:15],
      'examples_of_exact_matches':[r for r in rows if r['all_gold_named_role_predicates_exact_lemma']][:4]}
  result['splits'][split]={'original_source_sentence_count':len(gs),'source_sha256':SHA[split], 'operators':scopes}
  print('SPLIT',split,'N',len(gs))
  for op,x in scopes.items():print(op,'gold_cases',x['gold_source_sentences_with_named_scoped_event_roles'],'gold_named_links',x['gold_named_source_event_role_links'],'exact_role_lemma',x['full_named_predicate_role_lemma_exact'],'exact_sense',x['full_named_predicate_role_with_sense_exact'],'lemma_role_f1',x['micro_f1_lemma_name_role'])
 OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print('FILE_SHA',hashlib.sha256(OUT.read_bytes()).hexdigest())
if __name__=='__main__':main()
