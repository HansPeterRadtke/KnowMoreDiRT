"""Exploratory post-hoc chunk-boundary ablation on frozen source passages.

Not a separate blind holdout, and not a corpus-scale retrieval comparison.
Fixed 80-character slices, lexical overlap, ±2 neighbors. Never edits gold.
"""
import json,re,hashlib
from pathlib import Path
from heldout_scope_compare_20261024 import neighbor_metadata
HERE=Path(__file__).parent
raw=(HERE/'heldout_recordings_scope_gold_20261024.json').read_bytes();rows=json.loads(raw)['rows'];out=[]
for r in rows:
    t=r['excerpt']; cs=[(i,t[i:i+80]) for i in range(0,len(t),80)]
    tokens=set(re.findall(r'[a-zA-Z]{2,}',r['anchor'].lower()))
    def score(chunk):
        words=set(re.findall(r'[a-zA-Z]{2,}',chunk.lower()))
        return len(tokens&words)
    best=max(range(len(cs)),key=lambda i:(score(cs[i][1]),-i))
    single=cs[best][1]
    full=''.join(s for _,s in cs[max(0,best-2):min(len(cs),best+3)])
    single_pred=neighbor_metadata(single,r['anchor']);expand_pred=neighbor_metadata(full,r['anchor'])
    exact_anchor= r['anchor'].lower() in full.lower()
    out.append({'id':r['id'],'gold':r['gold_scope'],'best_chunk_char_start':cs[best][0],'best_chunk_token_overlap':score(single),'anchor_recovered_after_expansion':exact_anchor,'chunk_only_prediction':single_pred,'expanded_prediction':expand_pred,'chunk_only_correct':single_pred==r['gold_scope'],'expanded_correct':expand_pred==r['gold_scope']})
result={'gold_sha256':hashlib.sha256(raw).hexdigest(),'chunk_chars':80,'neighbor_window':2,'method':'Lexical query-anchor retrieval over only the case excerpt (oracle document); not corpus-scale','posthoc_exploratory':True,'total':len(out),'anchor_recall':sum(r['anchor_recovered_after_expansion'] for r in out),'chunk_only_correct':sum(r['chunk_only_correct'] for r in out),'neighbor_expanded_correct':sum(r['expanded_correct'] for r in out),'rows':out}
(HERE/'heldout_chunk_retrieval_20261024_results.json').write_text(json.dumps(result,indent=2)+'\n')
print('POSTHOC_CHUNK',result['chunk_only_correct'],result['neighbor_expanded_correct'],'anchor recall',result['anchor_recall'],'/',len(out))
