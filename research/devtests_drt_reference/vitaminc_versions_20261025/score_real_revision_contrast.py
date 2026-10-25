"""Human-gold evaluation of FROZEN real Wikipedia VERSION contrast predictions.

Requires all 2,800 model probability outputs. One-time scoring: original
VitaminC test writer labels are not read during selection/model inference.
Separate version-scoped text entailment from real-world truth, and analyze
necessary failure of a source-version-BLIND knowledge assertion. Every
published source record, score, and original model output remains intact.
"""
from pathlib import Path
from collections import Counter,defaultdict
from scipy.stats import binomtest
import hashlib,json,random
HERE=Path(__file__).parent
SRC=Path('/data/tmp/kmd-vitaminc-original-test-20261025.jsonl')
F=HERE/'frozen_real_revision_source_only_20261025.json'
P=HERE/'raw_source_version_nli_predictions_20261025.jsonl'
OUT=HERE/'human_gold_revision_contrast_results_20261025.json'
SRC_SHA='7ad1808dbc30c62e0a1427a53022d0dfaff668a1fde3c4b612a2d266edd753ad'
F_SHA='a6c6fca2dd5de8674606c8496c7e6fb1918321ee16cf815cfe5d7ec2fdd9ed7c'
LABELS={'SUPPORTS','REFUTES','NOT ENOUGH INFO'}

def fullgroups(records):
 groups=defaultdict(list)
 for row in records:
  if row['revision_type']=='real':groups[row['case_id']].append(row)
 selected={}
 for cid,rows in groups.items():
  if len(rows)!=4:continue
  cs=Counter(x['claim'] for x in rows);ev=Counter(x['evidence'] for x in rows)
  if sorted(cs.values())!=[2,2] or sorted(ev.values())!=[2,2]:continue
  if len({x.get('wiki_revision_id') for x in rows})!=1:continue
  if any(len({x['evidence'] for x in rows if x['claim']==claim})!=2 for claim in cs):continue
  selected[cid]=rows
 return selected

def pairs_by_claim(rows,pred=None):
 by=defaultdict(list)
 for x in rows:
  by[x['claim']].append(x)
 assert len(by)==2 and all(len(z)==2 for z in by.values())
 result=[]
 for claim,variants in by.items():
  labels=[x['label'] for x in variants]
  assert set(labels)<=LABELS
  values={'source_label_changed_across_versions':labels[0]!=labels[1],
    'source_label_combo':'+'.join(sorted(labels)),
    'same_claim_sha256':hashlib.sha256(claim.encode()).hexdigest(),
    'unique_source_record_ids':[x['unique_id'] for x in variants],
    'original_source_hashes':[hashlib.sha256(x['evidence'].encode()).hexdigest() for x in variants]}
  if pred:
   outputs=[pred[x['unique_id']] for x in variants]
   plabels=[x['predicted_label'] for x in outputs]
   values.update({'model_label_changed_across_versions':plabels[0]!=plabels[1],
    'human_labels_in_variant_order':labels,'model_labels_in_variant_order':plabels,
    'both_source_variants_classified_correctly':plabels==labels,
    'one_or_more_wrong':plabels!=labels,
    'source_input_token_truncation_any':any(x['token_input_truncated'] for x in outputs)})
  result.append(values)
 return result

def paired_bootstrap(groups,metric,iterations=1900):
 data=[sum(int(x[metric]) for x in g)/len(g) for g in groups];rng=random.Random(20261025)
 val=sorted(sum(data[rng.randrange(len(data))] for _ in data)/len(data) for _ in range(iterations))
 return {'n_groups':len(data),'estimate':round(sum(data)/len(data),6),'cluster_bootstrap_95CI':[round(val[47],6),round(val[1852],6)]}

def main():
 if OUT.exists():raise RuntimeError('Scoring already completed; do not rewrite or repeat')
 assert hashlib.sha256(SRC.read_bytes()).hexdigest()==SRC_SHA
 assert hashlib.sha256(F.read_bytes()).hexdigest()==F_SHA
 original=[json.loads(t) for t in SRC.read_text().splitlines() if t]
 full=fullgroups(original);assert len(full)==5994
 cases=json.loads(F.read_text())['case_ids_and_input_hashes_only']; assert len(cases)==700
 raw=[json.loads(t) for t in P.read_text().splitlines() if t];assert len(raw)==2800,'Do not score partial model results'
 preds={x['uid']:x for x in raw};assert len(preds)==len(raw)
 all_gold_pairs=[];all_cases=[]
 for cid,rows in full.items():
  a=pairs_by_claim(rows)
  all_gold_pairs.extend(a)
  if any(x['source_label_changed_across_versions'] for x in a):pass
 sampled=[];preds_all=[]
 for c in cases:
  rows=full[c['case_id']]
  assert {x['unique_id'] for x in rows}==set(c['source_data_row_ids'])
  for id,digest in zip(c['source_data_row_ids'],c['original_claim_evidence_pair_hashes']):
   original_row=next(x for x in rows if x['unique_id']==id)
   assert hashlib.sha256((original_row['claim']+'\n'+original_row['evidence']).encode()).hexdigest()==digest
   assert preds[id]['source_pair_sha256']==digest
  pairs=pairs_by_claim(rows,preds)
  sampled.append({'case_id':c['case_id'],'source_page':c['source_page'],'wiki_revision_transition_id':c['wiki_revision_transition_id'],
    'original_revision_source_type':'real','claim_version_pairs':pairs})
  preds_all.extend(pairs)
 def summary(arr):
  comb=Counter(x['source_label_combo'] for x in arr)
  return {'n_same_claim_with_two_revision_variants':len(arr),
   'changed_source_entailment_labels':sum(x['source_label_changed_across_versions'] for x in arr),
   'label_pair_frequency':dict(comb),
   'version_blind_one_label_max_accuracy_from_gold':round((2*len(arr)-sum(x['source_label_changed_across_versions'] for x in arr))/(2*len(arr)),6),
   'why_upper_bound_only':'Version-unaware strategy must say same truth label for both evidence versions: when independent source gold differs it cannot be correct on both. This is a combinatorial oracle upper bound, NOT a measured retriever or an operational oracle classifier.'}
 fulls=summary(all_gold_pairs)
 sampled_score=summary(preds_all)
 flat=[];model_errors=[]
 for c in sampled:
  for pair in c['claim_version_pairs']:
   ids=pair['unique_source_record_ids']; a,b=(preds[uid] for uid in ids)
   flat.extend((a,b))
   if pair['human_labels_in_variant_order']!=pair['model_labels_in_variant_order']:
    model_errors.append({'case_id':c['case_id'],'page':c['source_page'],**pair})
 assert len(flat)==2800
 gold_rows={x['unique_id']:x for rows in full.values() for x in rows if x['unique_id'] in preds}
 assert len(gold_rows)==len(flat)
 confusion=Counter((gold_rows[x['uid']]['label'],x['predicted_label']) for x in flat)
 pred_flips=sum(p['model_label_changed_across_versions'] for p in preds_all)
 actual_flips=sum(p['source_label_changed_across_versions'] for p in preds_all)
 both_flips=sum(p['source_label_changed_across_versions'] and p['model_label_changed_across_versions'] for p in preds_all)
 false_flips=sum(not p['source_label_changed_across_versions'] and p['model_label_changed_across_versions'] for p in preds_all)
 exact_pairs=sum(p['both_source_variants_classified_correctly'] for p in preds_all)
 correct=sum(gold_rows[x['uid']]['label']==x['predicted_label'] for x in flat)
 model_summary={'n_model_source_pairs':len(flat),'original_label_frequency':dict(Counter(gold_rows[x['uid']]['label'] for x in flat)),
  'model_label_frequency':dict(Counter(x['predicted_label'] for x in flat)),
  'model_accuracy_on_original_source_scoped_labels':round(correct/len(flat),6),'model_correct':correct,
  'confusion':[{'human_source_label':g,'model_predicted_label':p,'count':n} for (g,p),n in sorted(confusion.items())],
  'human_gold_label_flips_across_versions':actual_flips,
  'model_predicted_label_flips_across_versions':pred_flips,
  'correctly_detected_label_flips':both_flips,
  'unwarranted_predicted_label_flips':false_flips,
  'missed_human_gold_label_flips':actual_flips-both_flips,
  'model_label_flip_recall':round(both_flips/actual_flips,6) if actual_flips else None,
  'model_label_flip_precision':round(both_flips/pred_flips,6) if pred_flips else None,
  'both_source_versions_label_correct':exact_pairs,
  'input_truncated_count':sum(x['token_input_truncated'] for x in flat),
  'observed_prediction_probabilities_source':'Unmodified raw_source_version_nli_predictions_20261025.jsonl; model selected its original label on each source variant, never saw human gold'}
 result={'publisher':'Tal Schuster, Adam Fisch, Regina Barzilay, NAACL 2021 VitaminC, https://aclanthology.org/2021.naacl-main.52/',
  'authors_source_revision':'tals/vitaminc@be6febb761b0b2807687e61e0b5282e459df2fa0, test.jsonl, CC BY-SA 3.0',
  'original_test_sha256':SRC_SHA,'frozen_model_input_selection_sha256':F_SHA,'completed_raw_model_prediction_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),
  'scientific_question':'How frequently can an unchanged proposition have different entailment status depending on ORIGINAL SOURCE VERSION, and can an independently pretrained model track rather than merge two source passages?',
  'full_original_real_four_way_subset':fulls,
  'frozen_separate_nli_sample':{'n_case_groups':len(sampled),'n_fixed_claims':len(preds_all),'original_gold':sampled_score,'model':model_summary},
  'clustered_change_frequency_uncertainty':paired_bootstrap([c['claim_version_pairs'] for c in sampled],'source_label_changed_across_versions'),
  'controlled_example_errors':model_errors[:24],
  'actual_source_scoped_prediction_cases':[{'case_id':c['case_id'],'page':c['source_page'],'transition_revision_id':c['wiki_revision_transition_id'],'pairs':c['claim_version_pairs']} for c in sampled[:10]],
  'limitations':['Author VitaminC data label supports/contradicts/insufficient relative to a Wikipedia text version, NOT independently verified universal world truth.',
    'Real Wikipedia revision groups have one revision transition ID but no per-version verified as-of date in downloaded test records, so OLD vs NEW chronological precedence and contemporary correctness cannot be assigned solely from this dataset.',
    'The same claim with different textual versions is not strict same-world/same-time contradiction. Unscoped merging makes a false contradiction.',
    'Original test has been public since 2021 and independently pretrained NLI may be exposed indirectly; not a training-exposure guarantee.',
    'Actual sample is exactly 700 structurally selected groups, selected by source metadata only before labels; original human labels are read only after all 2,800 model outputs are saved.',
    'Original Wikipedia texts can be corrected, revised, disputed or vandalized; a revision is a documented statement transition, not independently validated scientific discovery.',
    'Only source-claim fact-verification text NLI; no full KMD DRS parsing, no dream/fiction/quote world semantics or cross-paper causal experimental falsification.',
    'Model is off-the-shelf SNLI/MNLI ONNX CPU and not the intended KMD production Qwen worker; source quote is not a mathematical proof or experiment.']}
 OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print('ALL_REAL_WIKIPEDIA_4WAY',fulls)
 print('FROZEN_SUBSET',sampled_score)
 print('MODEL_INFERENCE',model_summary)
 print('CLUSTER_INTERVAL',result['clustered_change_frequency_uncertainty'])
 print('JSON_SHA256',hashlib.sha256(OUT.read_bytes()).hexdigest())
if __name__=='__main__':main()
