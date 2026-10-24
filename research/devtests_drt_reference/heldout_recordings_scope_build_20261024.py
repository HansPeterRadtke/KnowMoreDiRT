"""Freeze source-offset narrative probes from original user recordings.

Gold labels were manually fixed from original utterance context BEFORE model
calls; they're annotation judgments, not real-world truth. No other KMD test
fixture and no previously benchmarked synthetic sentence is sampled.
"""
from pathlib import Path
import hashlib,json
ROOT=Path('/data/infra/docs/guidlines/original-recordings')
ROWS=[
 ('kmd/Recording 2026-08-05 17-22-26-192.txt','everything is called condition','asserted'),
 ('kmd/Recording 2026-08-05 17-22-26-192.txt','date a law can be changed','hypothetical'),
 ('kmd/Recording 2026-08-05 17-22-26-192.txt','I definitely have to fucking create a whole fucking file system database first','asserted'),
 ('kmd/Recording 2026-08-06 12-09-51-499.txt','Timmy writing about his dream','dream'),
 ('kmd/Recording 2026-08-06 12-09-51-499.txt','official evidence','conditional'),
 ('kmd/Recording 2026-08-06 17-30-53-705.txt','ten previous chunks','proposal'),
 ('agents/Recording 2026-10-17 23-00-29-387.txt','what questions about this topic does this text not answer','proposal'),
 ('agents/Recording 2026-10-18 18-38-21-960.txt','there is truth','speculative'),
 ('agents/Recording 2026-10-18 18-38-21-960.txt','a statement not being true','asserted'),
 ('agents/Recording 2026-10-18 18-38-21-960.txt','threshold that defines on and off','conditional'),
]
output=[]
for i,(file,anchor,label) in enumerate(ROWS):
    t=(ROOT/file).read_text(encoding='utf-8')
    pos=t.lower().find(anchor.lower());assert pos>=0,(file,anchor)
    assert t.lower().count(anchor.lower())==1,(file,anchor)
    start=max(0,pos-110);end=min(len(t),pos+len(anchor)+210)
    excerpt=t[start:end]
    assert excerpt[pos-start:pos-start+len(anchor)].lower()==anchor.lower()
    output.append({'id':f'original-recording-{i:02d}','source':file,'source_sha256':hashlib.sha256((ROOT/file).read_bytes()).hexdigest(),'source_char_start':start,'source_char_end':end,'anchor':anchor,'anchor_start_in_excerpt':pos-start,'excerpt':excerpt,'gold_scope':label})
dest=Path(__file__).with_name('heldout_recordings_scope_gold_20261024.json')
encoded=json.dumps({'protocol':'Gold frozen before inference; exact original character spans, manually categorized in context. Held out from KMD fixtures and tuned classifiers; not independent dual-human annotation.','rows':output},indent=2)+'\n'
if dest.exists():
    if dest.read_text()!=encoded:raise SystemExit('FROZEN_GOLD_DIFFERS: refusing to overwrite')
else:dest.write_text(encoded)
print('FROZEN',len(output),'GOLD_COUNTS',{x:sum(r['gold_scope']==x for r in output) for x in sorted({r['gold_scope'] for r in output})})
