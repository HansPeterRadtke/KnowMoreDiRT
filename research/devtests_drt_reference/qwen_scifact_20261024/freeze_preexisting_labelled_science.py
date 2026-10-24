"""Immutable source-grounded SciFact cases, stratified by external human label.

External SciFact TRAIN annotations, excluding 96 pairs already used in prior
October 24 research. Only gold labels select class balance, never prompts,
frame schemas or thresholds. Frozen before new Qwen inference.
"""
import hashlib,json,re
from pathlib import Path
HERE=Path(__file__).parent
ROOT=Path('/data/tmp/kmd-scifact-release-20261024/data')
OLD=Path('research/devtests_drt_reference/scientific_discovery_primary_20261024/scifact_distinct_train_nli_frozen_cases_20261024.json')
OUT=HERE/'frozen_model_science_20261024.json'

def load(path):return [json.loads(line) for line in path.read_text().splitlines() if line]

def pick(q,sentences):
 def tok(s):return set(re.findall(r'(?u)\b\w\w+\b',s.lower()))
 qt=tok(q)
 ranked=sorted(range(len(sentences)),key=lambda i:(-(len(qt&tok(sentences[i]))/(len(qt|tok(sentences[i])) or 1)),i))[:2]
 return sorted({k for i in ranked for k in range(max(0,i-1),min(len(sentences),i+2))})

def main():
 if OUT.exists():raise RuntimeError('frozen cases already present; no regeneration')
 claims=load(ROOT/'claims_train.jsonl');docs={str(x['doc_id']):x for x in load(ROOT/'corpus.jsonl')}
 prev=json.loads(OLD.read_text())
 avoid={tuple(map(str,x)) for x in prev['pair_keys']}
 candidates={'SUPPORT':[],'CONTRADICT':[]}
 for c in claims:
  for d,evs in c['evidence'].items():
   if (str(c['id']),str(d)) in avoid:continue
   labels={e['label'] for e in evs}
   if len(labels)!=1:continue
   label=next(iter(labels));
   if label not in candidates:continue
   digest=hashlib.sha256(f"{c['id']}-{d}-heldout-kmd-20261024".encode()).hexdigest()
   ix=pick(c['claim'],docs[str(d)]['abstract']);
   if not ix:continue
   evidence='\n'.join(f'Sentence {i}: {docs[str(d)]["abstract"][i]}' for i in ix)
   # No annotated rationale IDs used for selecting input text, only lexical overlap.
   candidates[label].append({'uid':f"sci-{c['id']}-{d}",'source_sha256':hashlib.sha256(evidence.encode()).hexdigest(),
     'source_doc_id':str(d),'source_title':docs[str(d)]['title'],'claim':c['claim'],'source_sentences':ix,
     'source_evidence':evidence,'selection_sha256':digest})
 for label in candidates:candidates[label].sort(key=lambda x:x['selection_sha256'])
 picked=[r for cls in ('SUPPORT','CONTRADICT') for r in candidates[cls][:4]]
 picked.sort(key=lambda x:x['selection_sha256'])
 assert len(picked)==8
 frozen={'source_repository':'https://github.com/allenai/scifact','original_train_sha256':hashlib.sha256((ROOT/'claims_train.jsonl').read_bytes()).hexdigest(),
   'original_corpus_sha256':hashlib.sha256((ROOT/'corpus.jsonl').read_bytes()).hexdigest(),
   'protocol':'4 original annotator SUPPORT, 4 CONTRADICT, independently externally labeled. Excludes earlier 96 article pairs; class stratification before inference only, labels hidden in new model prompts. Exactly automatic Jaccard top two source sentences plus adjacent context.',
   'cases':picked,
   'important_limits':['Selected from existing SciFact TRAIN source cases; training data possible exposure to pretrained Qwen and prior studies cannot be ruled out','Cases are not cross-paper or dream narratives; support/refute is human assessment of a source, not ultimate physical truth','Source spans are automatically selected, not gold rationales','No human labels passed to model or used to tune prompts']}
 OUT.write_text(json.dumps(frozen,indent=2,ensure_ascii=False)+'\n')
 print('FROZEN',len(picked),'HASH',hashlib.sha256(OUT.read_bytes()).hexdigest(),'CHAR_LENGTH_RANGE',[min(len(x['source_evidence']) for x in picked),max(len(x['source_evidence']) for x in picked)])
 print('IDS',[(x['uid'],len(x['source_evidence']),x['source_sentences']) for x in picked])
if __name__=='__main__':main()
