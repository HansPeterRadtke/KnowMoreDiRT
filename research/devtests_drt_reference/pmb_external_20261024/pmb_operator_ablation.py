"""Exploratory category-wise gold-PMB scope extraction vs published model DRS.

These diagnostics are NOT an independent new heldout (earlier negation analysis
already examined same test data), not cross-document inference, and not a
production KMD comparison. Fixed lexical patterns do not use model output.
"""
import json,re,hashlib
from pathlib import Path
HERE=Path(__file__).parent
G=Path('/data/tmp/kmd-pmb-eval-20261024/data/pmb-3.0.0/gold/test.txt')
P=Path('/data/tmp/kmd-pmb-reference-20261024/output/pmb-3.0.0/en/test/bert_char_1enc/output1.txt')
RAW=Path('/data/tmp/kmd-pmb-eval-20261024/data/pmb-3.0.0/gold/test.txt.raw')
OUT=HERE/'pmb_operator_ablation_20261024_results.json'
PATTERNS={
 'NEGATION':r"\b(no|not|never|nothing|nobody|neither|none|without|can't|cannot|won't|didn't|doesn't|don't|isn't|aren't|wasn't|weren't|shouldn't|couldn't|wouldn't|mustn't|hasn't|haven't|hadn't)\b|n['’]t\b",
 'CONDITION':r'\b(if|unless|provided|whenever|assuming)\b',
 'CONSEQUENCE':r'\b(then|therefore|otherwise)\b',
 'POSSIBILITY':r'\b(might|maybe|may|perhaps|could|possible|possibly)\b',
 'ATTRIBUTION':r'\b(said|says|told|claims|reported|according|asserted|believes|thought)\b',
 'YearOfCentury':r'\b(?:18|19|20)\d\d\b',
}

def groups(t):return [x for x in re.split(r'\n\s*\n',t.strip()) if x.strip()]
def contains(g,label):
 return any((len(a:=l.strip().split())>=3 and a[1]==label) for l in g.splitlines() if l.strip() and not l.startswith('%'))
def metrics(rows,key):
 tp=sum(v['gold'] and v[key] for v in rows);fp=sum(not v['gold'] and v[key] for v in rows)
 fn=sum(v['gold'] and not v[key] for v in rows);tn=sum(not v['gold'] and not v[key] for v in rows)
 return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':round(tp/(tp+fp),4) if tp+fp else None,'recall':round(tp/(tp+fn),4) if tp+fn else None,'f1':round(2*tp/(2*tp+fn+fp),4) if (2*tp+fn+fp) else None}

def main():
 if OUT.exists():raise RuntimeError('existing output; no overwrite')
 gold,pred=groups(G.read_text()),groups(P.read_text());sents=RAW.read_text().splitlines()
 assert len(sents)==len(gold)==len(pred)==898
 scores={}
 for cls,pattern in PATTERNS.items():
  row=[{'gold':contains(g,cls),'model':contains(p,cls),'baseline':bool(re.search(pattern,s,re.I)),'index':i,'source':s} for i,(s,g,p) in enumerate(zip(sents,gold,pred))]
  scores[cls]={'gold_cases':sum(x['gold'] for x in row),'model_generated_drs':metrics(row,'model'),'source_lexical_baseline':metrics(row,'baseline'),
    'model_false_negatives':[{'idx':x['index'],'source':x['source']} for x in row if x['gold'] and not x['model']][:12],
    'baseline_false_negatives':[{'idx':x['index'],'source':x['source']} for x in row if x['gold'] and not x['baseline']][:12]}
 data={'note':'Post-hoc exploratory scope-category comparison on the SAME 898 PMB source sentences; not a new blinded generalization experiment.',
 'source_revisions':{'reference_DRS':'190838f9adea2af616eb76156d2e66c8565a6c6e','published_neural_DRS':'81bcb26dfcc08f0da999cc7beaa2952f8820860c'},
 'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (G,P,RAW)},
 'labels':scores,'no_cross_document_inference':True,'limits':['Only presence of operator, not full DRS semantic predicate/box correctness or evidence citation','Lexical patterns are simple post-hoc plausible controls, not matched-budget provenance+neighbor retrieval','No original train/model queries were rerun; precomputed published neural parses vs human-corrected PMB gold','No independent physical-world claim truth or science discovery addressed']}
 OUT.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
 for key,x in scores.items():print(key,'gold=',x['gold_cases'],'DRS',x['model_generated_drs'],'regex',x['source_lexical_baseline'])
if __name__=='__main__':main()
