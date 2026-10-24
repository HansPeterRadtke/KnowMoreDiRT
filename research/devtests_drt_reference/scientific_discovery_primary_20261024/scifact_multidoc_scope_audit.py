"""Human-annotated SciFact cross-paper evidence: scope != independent truth.

No model and no new annotation, only exact published evidence sentence IDs. All
source-to-claim labels are external SciFact gold, so this is a *descriptive*
audit on an already-consumed development split, NOT a heldout model test.
"""
import json,hashlib,re
from pathlib import Path
from collections import Counter
SRC=Path('/data/tmp/kmd-scifact-release-20261024/data')
HERE=Path(__file__).parent
OUT=HERE/'scifact_multidoc_scope_audit_results_20261024.json'
C_SHA='b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62'
D_SHA='86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217'
QUALIFIERS={
 'premenopausal':r'\bpremenopausal\b','postmenopausal':r'\bpostmenopausal\b',
 'primary_disease':r'\bprimary\b','secondary_disease':r'\bsecondary\b',
 'animal_model':r'\b(pig|pigs|murine|mouse|mice|rat|rats)\b','randomised_study':r'\b(randomi[sz]ed|randomly assigned)\b',
 'mortality':r'\bmortality\b|\bdying\b','incidence':r'\bincidence\b',
 'association_not_causation':r'\bassociated\b|\bcorrelation\b|\bassociat(e|ion)\b',
 'possible_hedge':r'\b(suggest|may|might|appears)\b',
}

def load(name):return [json.loads(x) for x in (SRC/name).read_text().splitlines() if x]

def main():
 if OUT.exists():raise RuntimeError('Existing audit; no repetition')
 assert hashlib.sha256((SRC/'corpus.jsonl').read_bytes()).hexdigest()==C_SHA
 assert hashlib.sha256((SRC/'claims_dev.jsonl').read_bytes()).hexdigest()==D_SHA
 claims=load('claims_dev.jsonl');docs={str(x['doc_id']):x for x in load('corpus.jsonl')}
 multi=[x for x in claims if len(x['evidence'])>1]
 assert len(multi)==10
 results=[]
 for c in multi:
  docs_out=[]
  all_labels=set()
  for did,evs in c['evidence'].items():
   d=docs[did];found=set();source_spans=[]
   for e in evs:
    all_labels.add(e['label'])
    for i in e['sentences']:
     assert 0<=i<len(d['abstract'])
     raw=d['abstract'][i]
     found.update(k for k,p in QUALIFIERS.items() if re.search(p,raw,re.I))
     source_spans.append({'sentence_index':i,'excerpt_sha256':hashlib.sha256(raw.encode()).hexdigest(),'short_evidence':raw[:185]})
   docs_out.append({'doc_id':str(did),'title':d['title'],'annotated_labels':sorted({x['label'] for x in evs}),'detected_scope_cues':sorted(found),'source_spans':source_spans})
  results.append({'claim_id':c['id'],'claim':c['claim'],'external_gold_label_types':sorted(all_labels),'cross_source_label_disagreement':len(all_labels)>1,'distinct_evidence_papers':len(docs_out),'papers':docs_out})
 out={'corpus_sha256':C_SHA,'dev_sha256':D_SHA,'original_source':'https://github.com/allenai/scifact','multi_doc_claims':len(results),'support_only_claims':sum(x['external_gold_label_types']==['SUPPORT'] for x in results),
 'contradict_only_claims':sum(x['external_gold_label_types']==['CONTRADICT'] for x in results),'cross_source_disagreeing_label_claims':sum(x['cross_source_label_disagreement'] for x in results),'evidence_papers':sum(len(x['papers']) for x in results),
 'claim_ids_with_premenopausal_postmenopausal_source':[x['claim_id'] for x in results if any('premenopausal' in p['detected_scope_cues'] or 'postmenopausal' in p['detected_scope_cues'] for p in x['papers'])],
 'claim_ids_with_mortality_or_incidence_source':[x['claim_id'] for x in results if any('mortality' in p['detected_scope_cues'] or 'incidence' in p['detected_scope_cues'] for p in x['papers'])],
 'limitations':['Externally annotated multi-document evidence group is only ten claims, no demonstrated opposite-label paper disagreement in this subset','Same exact gold label across papers does not establish publisher independence, independent samples, independent intervention or ultimate truth','Sentence-level lexical qualifier extraction is automatic and fallible, not full discourse scope parsing','More than one paper may synthesize overlapping underlying clinical trials; deduplicate evidence families before Bayesian combination','The labels describe text-grounded support/refute for an authored claim, not outcomes of independent new experiments'],
 'results':results}
 OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
 print('MULTIDOC',out['multi_doc_claims'],'PAPERS',out['evidence_papers'],'CONSISTENT_SUPPORT',out['support_only_claims'],'CONSISTENT_REFUTE',out['contradict_only_claims'],'MIXED_LABEL_CLAIMS',out['cross_source_disagreeing_label_claims'])
 print('SUBGROUP_QUALIFIER_CLAIMS',out['claim_ids_with_premenopausal_postmenopausal_source'],'INCIDENCE_OR_MORTALITY',out['claim_ids_with_mortality_or_incidence_source'])
if __name__=='__main__':main()
