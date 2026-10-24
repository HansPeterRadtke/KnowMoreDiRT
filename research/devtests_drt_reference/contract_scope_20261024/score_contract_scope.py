"""Independent labels scored only after all original model outputs committed.

Human expert evidence spans from ContractNLI are used for audit, NEVER input.
"""
from pathlib import Path
from collections import Counter
import json,hashlib
HERE=Path(__file__).parent
TEST=Path('/data/tmp/kmd-contractnli-20261024/contract-nli/test.json')
FROZEN=HERE/'frozen_contract_source_scope_20261024.json'
RAW=HERE/'raw_model_comparison_20261024.jsonl'
OUT=HERE/'human_gold_contract_scope_results_20261024.json'

def verdict(obj):
 if not isinstance(obj,dict):return 'INVALID'
 v=obj.get('verdict')
 return v if v in ('Entailment','Contradiction','NotMentioned') else 'INVALID'

def main():
 if OUT.exists():raise RuntimeError('Scoring already done; no rewriting')
 d=json.loads(TEST.read_text());f=json.loads(FROZEN.read_text());cases=f['cases']
 assert f['original_test_sha256']==hashlib.sha256(TEST.read_bytes()).hexdigest()
 a=[json.loads(s) for s in RAW.read_text().splitlines() if s]
 assert len(a)==len(cases)==6,'No partial result: wait for inference to finish'
 originals={x['id']:x for x in d['documents']}
 recs={x['id']:x for x in a};assert len(recs)==6
 rows=[]
 for item in cases:
  src=originals[item['id']];rec=recs[item['id']]
  assert hashlib.sha256(src['text'].encode()).hexdigest()==item['doc_text_sha256']
  assert rec['source_excerpt_sha256']==item['source_excerpt_sha256']
  gold=src['annotation_sets'][0]['annotations'][f['hypothesis_id']]
  a_gold=[src['text'][src['spans'][k][0]:src['spans'][k][1]].strip() for k in gold['spans']]
  # Counts strictly *intact*, independently human annotated exact spans.
  matches=[k for k,s in zip(gold['spans'],a_gold) if s and s in item['source_excerpt']]
  a_pred=verdict(rec['direct'].get('parsed'))
  b_pred=verdict(rec['structured_verdict'].get('parsed'))
  q=rec['direct'].get('parsed',{}).get('quote') if isinstance(rec['direct'].get('parsed'),dict) else ''
  rows.append({'doc_id':item['id'],'label_human':gold['choice'],'direct_prediction':a_pred,'structured_prediction':b_pred,
    'direct_correct':a_pred==gold['choice'],'structured_correct':b_pred==gold['choice'],
    'human_gold_spans':list(gold['spans']),'human_gold_spans_in_exact_source_excerpt':matches,
    'original_annotated_evidence_found_in_excerpt':bool(matches),
    'retrieved_source_span_indices':item['source_span_indices'],
    'direct_quote_exact_source':bool(q and isinstance(q,str) and q in item['source_excerpt']),
    'frame_valid':rec['frame_valid'],'frame_reason':rec['frame_reason'],
    'direct_seconds':rec['direct'].get('elapsed_seconds',0),
    'structured_seconds':rec['structured_extraction'].get('elapsed_seconds',0)+rec['structured_verdict'].get('elapsed_seconds',0),
    'raw_output_key':'raw_model_comparison_20261024.jsonl','source_url':item['source_url'],
    'source_doc_sha256':item['doc_text_sha256']})
 metric={'n':len(rows),'gold_counts':dict(Counter(x['label_human'] for x in rows)),
  'direct_correct':sum(x['direct_correct'] for x in rows), 'structured_correct':sum(x['structured_correct'] for x in rows),
  'direct_confusion':dict(Counter(x['label_human']+'=>'+x['direct_prediction'] for x in rows)),
  'structured_confusion':dict(Counter(x['label_human']+'=>'+x['structured_prediction'] for x in rows)),
  'full_gold_spans_in_context':sum(x['original_annotated_evidence_found_in_excerpt'] for x in rows),
  'source_quote_grounded':sum(x['direct_quote_exact_source'] for x in rows),
  'valid_frames':sum(x['frame_valid'] for x in rows),
  'total_direct_seconds':round(sum(x['direct_seconds'] for x in rows),2),
  'total_structured_seconds':round(sum(x['structured_seconds'] for x in rows),2)}
 out={'original_dataset':'ContractNLI authors Koreeda and Manning 2021, CC BY 4.0, Stanford/Hitachi',
   'original_sha256':hashlib.sha256(TEST.read_bytes()).hexdigest(),
   'preselected_frozen_sha256':hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
   'verbatim_model_jsonl_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),
   'comparison_methods':'Direct source-aware Qwen verdict vs source-only compact attributed rule plus Qwen verdict, same retrieved evidence but unequal output/call budgets',
   'methods_limitations':['N=6 one fixed hypothesis NDA-20 across 6 real public NDA documents and 3 independent human classes; labels stratify selection but never passed to Qwen',
    'Source excerpts selected by lexical overlap and local context, not human evidence; missing gold evidence makes factual accuracy impossible to attribute solely to model',
    'Compact attribution frame is not actual full KMD DRS, not cross-document multi-hop deduction, nor dream or fictional world inference',
    'Two-stage route has more model calls than direct route, so these scores are a bounded exploratory comparison not fair matched-budget superiority',
    'External legal source judgments do not establish scientific truth or new facts','Qwen may have training exposure to original dataset; not demonstrably held out of pretraining'],
   'metrics':metric,'rows':rows}
 OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
 print('CONTRACT_SCOPE_EXTERNAL_HUMAN_GOLD',metric)
if __name__=='__main__':main()
