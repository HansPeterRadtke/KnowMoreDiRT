"""Score only AFTER all 8 frozen model predictions were durably recorded.

Original externally annotated SciFact TRAIN labels are read solely for this
post-inference analysis. No model prompts, test selection or gold changed.
"""
from collections import Counter
from pathlib import Path
import json,hashlib
HERE=Path(__file__).parent
ROOT=Path('/data/tmp/kmd-scifact-release-20261024/data')
CHK=HERE/'results_checkpoint.jsonl'
FROZEN=HERE/'frozen_model_science_20261024.json'
OUT=HERE/'labelled_model_comparison_20261024.json'

def read_jsonl(p):return [json.loads(x) for x in p.read_text().splitlines() if x]
def get_verdict(x):
 if not isinstance(x,dict):return 'INVALID'
 v=x.get('verdict')
 return v if v in ('SUPPORT','CONTRADICT','UNKNOWN') else 'INVALID'

def score():
 if OUT.exists():raise RuntimeError('Completed results exist; do not overwrite')
 freeze=json.loads(FROZEN.read_text());items=freeze['cases']
 completed=read_jsonl(CHK)
 assert len(completed)==len(items)==8,'No partial scoring: predictions must all exist'
 byid={x['uid']:x for x in completed};assert len(byid)==8
 claims={str(c['id']):c for c in read_jsonl(ROOT/'claims_train.jsonl')}
 r=[]
 for case in items:
  got=byid[case['uid']]
  assert got['source_sha256']==case['source_sha256']
  _,id,doc=case['uid'].split('-')
  labels={v['label'] for v in claims[id]['evidence'][doc]}
  assert len(labels)==1
  gold=next(iter(labels))
  a=get_verdict(got['direct'].get('parsed'))
  b=get_verdict(got['box_judgment'].get('parsed'))
  qa=got['direct'].get('parsed');quote=(qa.get('evidence_quote') if isinstance(qa,dict) else None)
  trace_valid=bool(quote and isinstance(quote,str) and quote in case['source_evidence'])
  r.append({'uid':case['uid'],'gold':gold,'direct':a,'boxed':b,'direct_correct':a==gold,'boxed_correct':b==gold,
    'direct_quote_exact_source':trace_valid,'boxed_grounded_frames':got['box_accepted'],'boxed_validation_reason':got['box_reason'],
    'direct_elapsed':got['direct'].get('elapsed_seconds'),'boxed_elapsed':got['box_extraction'].get('elapsed_seconds',0)+got['box_judgment'].get('elapsed_seconds',0),
    'direct_tokens':got['direct'].get('usage'),'boxed_tokens':[got['box_extraction'].get('usage'),got['box_judgment'].get('usage')],
    'model_raw_file':'results_checkpoint.jsonl','source_doc':doc})
 s={'sample_count':len(r),'gold_counts':dict(Counter(x['gold'] for x in r)),
   'direct_correct':sum(x['direct_correct'] for x in r),'boxed_correct':sum(x['boxed_correct'] for x in r),
   'invalid_direct':sum(x['direct']=='INVALID' for x in r),'invalid_boxed':sum(x['boxed']=='INVALID' for x in r),
   'grounded_box_frames':sum(x['boxed_grounded_frames'] for x in r),'direct_quotes_exact_source':sum(x['direct_quote_exact_source'] for x in r),
   'direct_total_seconds':round(sum(x['direct_elapsed'] or 0 for x in r),2),'boxed_total_seconds':round(sum(x['boxed_elapsed'] or 0 for x in r),2),
   'confusion_direct':dict(Counter(x['gold']+'=>'+x['direct'] for x in r)),
   'confusion_boxed':dict(Counter(x['gold']+'=>'+x['boxed'] for x in r))}
 out={'pre_inference_frozen_hash':hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
      'model_raw_checkpoint_sha256':hashlib.sha256(CHK.read_bytes()).hexdigest(),
      'source_data_sha256':hashlib.sha256((ROOT/'claims_train.jsonl').read_bytes()).hexdigest(),
      'purpose':'A bounded real-source comparison with human-source-document truth labels. NOT formal DRS, production KMD, multi-document knowledge discovery or field discovery.',
      'scientific_limitations':['N=8 stratified class balance before inference, high statistical uncertainty','SciFact TRAIN subset never used in this experiment before freezing; pretrained Qwen exposure to dataset cannot be ruled out',
          'No actual full KMD chunk DRS; this is a lightweight automatic DRS-inspired source-only box extractor plus later claim judge, 2 model calls vs 1 direct',
          'Human labels measure source support/contradiction, NOT empirical truth or scientific novelty',
          'Gold evidence document supplied to both methods; source sentence selection automated, not gold annotated rationales','No dream/fiction, quotation or cross-paper coreference benchmark on these cases','Invalid source quotes or boxes never silently repaired'],
      'metrics':s,'cases':r}
 OUT.write_text(json.dumps(out,indent=2)+'\n')
 print('EXTERNAL_GOLD',s)
if __name__=='__main__':score()
