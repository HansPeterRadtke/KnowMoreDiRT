"""Frozen external source-only spaCy LG baseline, same rules as prior SM run.

Post-hoc stronger general-purpose parser capacity check; NOT a new heldout
model search or tuned algorithm. Gold remains unread in this prediction stage.
"""
import hashlib,json
from pathlib import Path
import spacy
from operator_scope_predicate_audit import GROOT,SHA
from spacy_dependency_negation_baseline import source_negation_candidates
HERE=Path(__file__).parent
DEST=HERE/'frozen_spacy_lg_dependency_predictions_20261025.json'

def run():
 if DEST.exists():raise RuntimeError('Frozen LG baseline exists; do not rerun')
 nlp=spacy.load('en_core_web_lg',disable=['ner'])
 out={'model':'en_core_web_lg','spacy_version':spacy.__version__,'spacy_model_version':nlp.meta.get('version'),
  'same_fixed_source_only_rule':'source_negation_candidates from spacy_dependency_negation_baseline.py unchanged after original SM test score',
  'source_git':'RikVN/DRS_parsing@190838f9adea2af616eb76156d2e66c8565a6c6e',
  'source_sha256':{},'splits':{},
  'limit':'Post-hoc robustness: stronger off-the-shelf dependency model, same heuristics, same already-consumed original PMB test/dev source utterances. NO gold annotations provided to parser.'}
 for split in ('test','dev'):
  src=GROOT/(split+'.txt.raw');assert hashlib.sha256(src.read_bytes()).hexdigest()==SHA[split]['raw']
  sents=src.read_text().splitlines();rows=[]
  for i,doc in enumerate(nlp.pipe(sents,batch_size=48,n_process=1)):
   lemmas,trace=source_negation_candidates(doc)
   rows.append({'sentence_index':i,'source_sha256':hashlib.sha256(sents[i].encode()).hexdigest(),
     'predicted_negated_va_lemmas':list(lemmas.elements()),'source_cues_and_dependency_trace':trace})
  out['source_sha256'][split]=SHA[split]['raw'];out['splits'][split]={'n':len(rows),'predictions':rows}
  print('LG_SOURCE_ONLY_FROZEN',split,len(rows),'ANY_NEG_VA',sum(bool(x['predicted_negated_va_lemmas']) for x in rows),flush=True)
 DEST.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
 print('LG_SHA256',hashlib.sha256(DEST.read_bytes()).hexdigest())
if __name__=='__main__':run()
