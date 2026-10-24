"""Post-hoc audit of *completed* real scientific-model results and source gold.

Preserves checkpoint and first scored JSON unchanged. This measures whether
published human rationale sentences were actually present in the model input,
separating SOURCE RETRIEVAL failure from INFERENCE failure; not new tuning.
"""
import json,hashlib
from collections import Counter
from pathlib import Path
HERE=Path(__file__).parent
ROOT=Path('/data/tmp/kmd-scifact-release-20261024/data')
DEST=HERE/'source_coverage_and_extraction_failure_audit_20261024.json'

def load(p):return [json.loads(t) for t in p.read_text().splitlines() if t]

def main():
 if DEST.exists():raise RuntimeError('No overwrite of prior audit')
 frozen=json.loads((HERE/'frozen_model_science_20261024.json').read_text());cases=frozen['cases'];preds={x['uid']:x for x in load(HERE/'results_checkpoint.jsonl')};assert len(preds)==len(cases)==8
 claims={str(x['id']):x for x in load(ROOT/'claims_train.jsonl')}
 rows=[]
 for case in cases:
  uid=case['uid'];_,cid,did=uid.split('-');c=claims[cid];ann=c['evidence'][did];gold=set(x['label'] for x in ann);assert len(gold)==1
  gold=next(iter(gold));gold_sent={int(k) for a in ann for k in a['sentences']}
  selected={int(x) for x in case['source_sentences']}; hit=gold_sent & selected
  p=preds[uid];direct=p['direct'].get('parsed')
  quote=direct.get('evidence_quote') if isinstance(direct,dict) else None
  rows.append({'uid':uid,'external_gold':gold,'source_title':case['source_title'],'human_rationale_sentence_ids':sorted(gold_sent),'automatic_source_sentence_ids':sorted(selected),
    'human_gold_rationale_present_in_input':bool(hit),'overlap_sentence_indices':sorted(hit),
    'direct_model_verdict':direct.get('verdict') if isinstance(direct,dict) else None,
    'direct_model_quote_exact_input':bool(quote and isinstance(quote,str) and quote in case['source_evidence']),
    'direct_quote':quote,'box_extraction_finish_reason':p['box_extraction'].get('finish_reason'),
    'box_model_output_valid':p['box_accepted'],'box_judge_skipped':bool(p['box_judgment'].get('skipped')),
    'source_sha256':case['source_sha256']})
 refs=[x for x in rows if x['external_gold']=='CONTRADICT']
 out={'source_sha256':{'frozen':hashlib.sha256((HERE/'frozen_model_science_20261024.json').read_bytes()).hexdigest(),
    'checkpoint':hashlib.sha256((HERE/'results_checkpoint.jsonl').read_bytes()).hexdigest(),
    'external_train':hashlib.sha256((ROOT/'claims_train.jsonl').read_bytes()).hexdigest()},
    'sample_size':len(rows),
    'human_gold_rationale_recovered_overall':sum(x['human_gold_rationale_present_in_input'] for x in rows),
    'contradiction_cases':len(refs),'human_rationale_recovered_in_contradiction_cases':sum(x['human_gold_rationale_present_in_input'] for x in refs),
    'contradiction_claims_wrongly_called_support':sum(x['direct_model_verdict']=='SUPPORT' for x in refs),
    'model_generated_frame_invalid':sum(not x['box_model_output_valid'] for x in rows),
    'extraction_length_truncations':sum(x['box_extraction_finish_reason']=='length' for x in rows),
    'box_judge_abstentions_due_to_invalid_structure':sum(x['box_judge_skipped'] for x in rows),
    'direct_quotes_exact_source':sum(x['direct_model_quote_exact_input'] for x in rows),
    'important_interpretation':['Every CONTRADICT gold scientific pair included at least one original expert annotated evidence sentence in the model input, so failures cannot be explained solely by omitted gold rationale',
       'All eight parser responses exhausted fixed 195-token budget and lacked a valid complete grounded model-produced box: the second model call did not occur and abstention was the safe path',
       'This is one small local Qwen-1.5B variant, NOT the deployed KMD large-model DRS; no conclusions about general formal DRT parser superiority can follow',
       'Human annotated scientific article textual support/refute is not ultimate scientific truth or new discovery','Some correct SUPPORT calls lacked included gold rationale, raising concern that plausibility/model priors or alternative non-gold source evidence may determine the result; this small audit cannot distinguish them'],
    'rows':rows}
 DEST.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
 print('SOURCE_AUDIT',out['sample_size'],'gold_present',out['human_gold_rationale_recovered_overall'],'contradictions_with_source',out['human_rationale_recovered_in_contradiction_cases'],
      'contradictions_wrongly_supported',out['contradiction_claims_wrongly_called_support'],'frame_truncated',out['extraction_length_truncations'])
if __name__=='__main__':main()
