"""Freeze truly authored Wikipedia-REVISION claim/evidence contrasts, no labels.

Original VitaminC 2021 author test split, SHA pinned. Selection only uses
original metadata and the STRUCTURE (two distinct claim texts and two evidence
texts repeated cross-product), not human classification or model outcomes.
"""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json
HERE=Path(__file__).parent
SOURCE=Path('/data/tmp/kmd-vitaminc-original-test-20261025.jsonl')
OUT=HERE/'frozen_real_revision_source_only_20261025.json'
SHA='7ad1808dbc30c62e0a1427a53022d0dfaff668a1fde3c4b612a2d266edd753ad'
SAMPLE_SIZE=700

def load():
 assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
 return [json.loads(x) for x in SOURCE.read_text().splitlines() if x]

def structural_candidates(records):
 by=defaultdict(list)
 for row in records:
  if row['revision_type']=='real':by[row['case_id']].append(row)
 elig=[]
 for cid,rows in by.items():
  if len(rows)!=4:continue
  cs=Counter(r['claim'] for r in rows); es=Counter(r['evidence'] for r in rows)
  if sorted(cs.values())!=[2,2] or sorted(es.values())!=[2,2]:continue
  if len({r['wiki_revision_id'] for r in rows})!=1:continue
  if any(len({r['evidence'] for r in rows if r['claim']==claim})!=2 for claim in cs):continue
  elig.append((hashlib.sha256(('vitaminc-kmd-revision-20261025-'+cid).encode()).hexdigest(),cid,rows))
 return sorted(elig)

def main():
 if OUT.exists():raise RuntimeError('Frozen original cases already exist; do not reselect')
 original=load();possible=structural_candidates(original)
 assert len(possible)==5994
 cases=[]
 for rank,(score,cid,items) in enumerate(possible[:SAMPLE_SIZE]):
  cases.append({'case_id':cid,'source_page':items[0]['page'],
    'wiki_revision_transition_id':items[0]['wiki_revision_id'],
    'source_data_row_ids':[x['unique_id'] for x in items],
    'original_claim_evidence_pair_hashes':[hashlib.sha256((x['claim']+'\n'+x['evidence']).encode()).hexdigest() for x in items],
    'sha256_selection_score':score})
 output={'original_publisher':'Schuster, Fisch, Barzilay 2021, VitaminC real Wikipedia revisions, NAACL, https://aclanthology.org/2021.naacl-main.52/',
  'original_source':'https://huggingface.co/datasets/tals/vitaminc/tree/be6febb761b0b2807687e61e0b5282e459df2fa0',
  'original_license':'CC BY-SA 3.0; raw original text retained externally, no wholesale dataset copy to project Git',
  'source_sha256':SHA,'structurally_eligible_real_four_way_source_revision_groups':len(possible),
  'selected_groups':len(cases),'selected_original_pairs':4*len(cases),
  'selection':'Hash sorted real case_id groups having exactly two distinct claim strings, two distinct evidence strings crossed 2 x 2, original revision transition ID; no GOLD label read, no source revision treated as timeless fact',
  'model_method_predeclared':'Off-the-shelf SNLI/MNLI cross-encoder/nli-MiniLM2-L6-H768 quantized ONNX CPU, source evidence versus exactly same claim for both variant passages, no model prompt/label tuning, one inference per unique claim-evidence.',
  'original_author_test_gold_not_sealed_from_pretraining':'Original test released 2021; source corpus public, model pretraining exposures cannot be ruled out; source-side predictions are frozen before original human label use.',
  'temporal_scope_warning':'wiki_revision_id identifies a Wikipedia transition; dataset does not include an unambiguous as-of-time timestamp for each version pair. Refer to evidence variants A and B, not exact publication dates/current world truth.',
  'case_ids_and_input_hashes_only':cases}
 OUT.write_text(json.dumps(output,indent=2)+'\n')
 print('FROZEN_VITAMINC',len(cases),'OF',len(possible),'REAL_REVISION_GROUPS','ORIGINAL_PAIRS',4*len(cases),'FROZEN_SHA',hashlib.sha256(OUT.read_bytes()).hexdigest())
 print('PAGES_EXAMPLES',[x['source_page'] for x in cases[:12]])
if __name__=='__main__':main()
