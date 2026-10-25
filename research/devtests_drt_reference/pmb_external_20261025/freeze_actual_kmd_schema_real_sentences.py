"""Freeze human-authored PMB source text for actual KMD chunk_drs-v5 smoke trial.

Select by prespecified lexical cues and hash, NOT by correct DRS gold,
not by intended result and not by model outcome. This tests feasibility of
model extracting deployed-schema DRS without shared GPU, NOT full KMD answer.
"""
from pathlib import Path
import hashlib,json,re
HERE=Path(__file__).parent
SRC=Path('/data/tmp/kmd-pmb-eval-20261024/data/pmb-3.0.0/gold/test.txt.raw')
OUT=HERE/'frozen_real_pmb_for_kmd_v5_20261025.json'
SHA='4bd557c2e6655d9b9c32269d4f6aa32415a314777bf16b517d7ae3da280f6263'
GROUPS={
  'negation':(r"\bnot\b|n['’]t\b|\bnever\b|\bneither\b",2),
  'conditional':(r'\bif\b|\bunless\b|\bwhenever\b',2),
  'dream':(r'\bdream\w*\b',1),
  'reported_speech':(r'\bsaid\b|according to',2),
}

def main():
 if OUT.exists():raise RuntimeError('FROZEN ORIGINALS ALREADY EXIST DO NOT RETUNE')
 assert hashlib.sha256(SRC.read_bytes()).hexdigest()==SHA
 lines=SRC.read_text().splitlines();used=set();rows=[]
 for category,(pattern,count) in GROUPS.items():
  available=[]
  for i,s in enumerate(lines):
   if i in used or len(s)>=145 or not re.search(pattern,s,re.I):continue
   available.append((hashlib.sha256(f'KMD-v5-real-20261025-{category}-{i}'.encode()).hexdigest(),i,s))
  assert len(available)>=count,category
  for key,i,s in sorted(available)[:count]:
   rows.append({'id':f'pmb3-test-source-{i}','category_from_source_cue':category,'sentence_index':i,
     'sentence':s,'sentence_sha256':hashlib.sha256(s.encode()).hexdigest(),'selection_hash':key})
   used.add(i)
 rows.sort(key=lambda x:x['selection_hash'])
 assert len(rows)==7
 output={'source_gold_repo':'RikVN/DRS_parsing@190838f9adea2af616eb76156d2e66c8565a6c6e',
   'source_test_text_sha256':SHA,'selection':'Prespecified regex groups, max 144 characters, SHA sorted before new model calls; NO human gold structures inspected for eligibility','n':len(rows),
   'method':'Actual installed KMD production build_chunk_drs_prompt and chunk_drs_json_schema with source ID; local Qwen2.5 1.5B gguf CPU JSON grammar with necessary local regex (?: to () syntax compatibility only; validator unchanged; never repair model semantics).',
   'caveats':['The user established PMB and source examples have been studied in prior turns; cannot claim pristine heldout source relative to research or LLM training','Selection predefines lexical scope groups, which is not independent human gold for dreams/quotations/conditions','Tiny CPU model differs from intended production KMD Qwen worker model, and an offline JSON grammar compatibility adapter differs from deployed server inference','Feasibility/semantic smoke test on seven independently written natural sentences, not statistically powered general model benchmark or cross-document reasoning'],
   'cases':rows}
 OUT.write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
 print('REAL_SOURCE_FROZEN',len(rows),'SHA256',hashlib.sha256(OUT.read_bytes()).hexdigest())
 for x in rows:print(x['category_from_source_cue'],x['id'],repr(x['sentence']))
if __name__=='__main__':main()
