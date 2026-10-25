"""External transfer of PREVIOUS Hotpot lexical identity-link method to MuSiQue.

No new ranking method, boost, stopwords or thresholds: import the earlier
already-gold-scored fixed Hotpot implementations without alteration.
Only question + 20 original paragraphs; no MuSiQue support/is_supporting flags,
answer, aliases, or question decomposition seen at prediction time.
"""
from pathlib import Path
import hashlib,json,time
from predict_unlabelled_retrieval import bm25,rank
SRC=Path('/data/tmp/kmd-musique-official-answerable-dev-20261025.jsonl')
OUT=Path(__file__).with_name('frozen_untuned_musique_rankings_20261025.json')
SHA='15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b'

def main():
 if OUT.exists():raise RuntimeError('Transfer rankings already frozen; never overwrite')
 assert hashlib.sha256(SRC.read_bytes()).hexdigest()==SHA
 cases=[json.loads(x) for x in SRC.read_text().splitlines() if x];assert len(cases)==2417
 preds=[];start=time.monotonic()
 for j,item in enumerate(cases,1):
  question=item['question'];p=item['paragraphs']
  titles=[x['title'] for x in p];texts=[x['paragraph_text'] for x in p]
  idxs=[x['idx'] for x in p];assert len(set(idxs))==len(p)
  s_title=bm25(question,titles)
  s_full=bm25(question,[title+' '+body for title,body in zip(titles,texts)])
  s_first=bm25(question,[title+' '+body[:400] for title,body in zip(titles,texts)])
  methods={'title_bm25':rank(s_title),'fulltext_bm25':rank(s_full),'title_first2_approx_bm25':rank(s_first)}
  first=methods['fulltext_bm25'][0];full=' '+texts[first].lower()+' '
  boost=max(s_full)*0.65 if max(s_full)>0 else 0
  others=[i for i in range(len(titles)) if i!=first]
  linked=sorted(others,key=lambda i:(-(s_full[i]+(boost if len(titles[i])>=5 and (' '+titles[i].lower()+' ') in full else 0)),-s_title[i],i))
  methods['linked_secondhop']=[first]+linked
  preds.append({'uid':item['id'],'question_sha256':hashlib.sha256(question.encode()).hexdigest(),
      'paragraph_identifier_hash':hashlib.sha256(json.dumps([(x['idx'],x['title']) for x in p],ensure_ascii=False).encode()).hexdigest(),
      'ranked_indices':{key:[idxs[i] for i in v[:8]] for key,v in methods.items()}})
  if j%750==0:print('RANKED',j,'/2417 ELAPSED',round(time.monotonic()-start,2),flush=True)
 out={'source_original_MuSiQue_sha256':SHA,'hotpot_scoring_pred_sha256':'3b458f9dca38f0fd6a00896928d7df98c6269297c0da4a65fd4b22dfbd76cdf8',
      'protocol':'Unchanged BM25 tokenization, retrieval and exact title mention boost transferred after Hotpot test scores were seen, before any MuSiQue support labels are accessed.',
      'source':'StonyBrookNLP MuSiQue answerable dev original CC BY 4.0, https://github.com/stonybrooknlp/musique, mirror dgslibisey/MuSiQue matching other mirrors SHA256',
      'n':len(preds),'runtime_seconds':round(time.monotonic()-start,3),
      'limits':['New dataset has 20 original provided candidate Wikipedia paragraphs and gold 2–4-hop chains, unlike Hotpot ten/2-hop','Title-first approximate method uses first 400 text characters because MuSiQue has paragraph text string not sentence list; title/full/linked main methods unchanged','No semantic parsing or DRS model; comparison only shows cross-dataset transfer of untrained source-link retrieval'],
      'predictions':preds}
 OUT.write_text(json.dumps(out,ensure_ascii=False)+'\n')
 print('FROZEN_TRANSFER',len(preds),'SHA',hashlib.sha256(OUT.read_bytes()).hexdigest(),'SECONDS',out['runtime_seconds'])
if __name__=='__main__':main()
