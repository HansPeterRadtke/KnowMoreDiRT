"""Full 7405 real HotpotQA multi-document retrieval, NEVER read labels in ranking.

Predictions of title-only BM25, title+text BM25, linked-hop expansion, and
within-document sentence evidence are frozen BEFORE gold scoring. No DRS.
As-of 2018 original Wikipedia paragraph snapshot, 2 annotated source articles.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,math,re,time
INPUT=Path('/data/tmp/kmd-hotpot-original-distractor-20261025.json')
OUT=Path(__file__).with_name('frozen_unlabelled_retrieval_20261025.json')
DATA_SHA='4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea'
STOP=set('the a an of in on at to for from by with and or is are was were be been being it its that this these those who which what when where how if not same both did do does had has have as their his her its whose would could should can into between many much what one two name named'.split())

def tok(t):return [w for w in re.findall(r'[\w]+',t.lower()) if len(w)>1 and w not in STOP]

def bm25(query,documents):
 q=set(tok(query));texts=[tok(s) for s in documents];n=len(texts)
 lens=[len(s) for s in texts];avg=sum(lens)/n or 1
 df=Counter(w for x in texts for w in set(x))
 result=[]
 for words in texts:
  counts=Counter(words);length=len(words);score=0.0
  for w in q:
   f=counts[w]
   if f:
    idf=math.log1p((n-df[w]+0.5)/(df[w]+0.5))
    score+=idf*(f*2.2)/(f+1.2*(0.25+0.75*length/avg))
  result.append(score)
 return result

def rank(values):return sorted(range(len(values)),key=lambda i:(-values[i],i))

def sentences_for_query(question,paragraph):
 sentences=paragraph
 if not sentences:return []
 scores=bm25(question,sentences)
 top=rank(scores)[:3]
 expanded=sorted({j for i in top for j in range(max(0,i-1),min(len(sentences),i+2))})
 return {'top3':top,'top3_neighborhood':expanded}

def main():
 if OUT.exists():raise RuntimeError('Frozen predictions already exist; do not overwrite')
 assert hashlib.sha256(INPUT.read_bytes()).hexdigest()==DATA_SHA
 items=json.loads(INPUT.read_text());assert len(items)==7405
 result=[];start=time.monotonic()
 for count,item in enumerate(items,1):
  # Only question, context, and stable source id are accessible here.
  question=item['question'];context=item['context'];titles=[x[0] for x in context];texts=[' '.join(x[1]) for x in context]
  scores_title=bm25(question,titles)
  scores_full=bm25(question,[title+' '+body for title,body in zip(titles,texts)])
  scores_title_first=bm25(question,[title+' '+(' '.join(context[i][1][:2])) for i,title in enumerate(titles)])
  methods={'title_bm25':rank(scores_title),'fulltext_bm25':rank(scores_full),'title_first2_bm25':rank(scores_title_first)}
  # Gold-blind title-reference second hop: first highest fulltext doc; rank other
  # titles additionally by whether first article explicitly names their title.
  first=methods['fulltext_bm25'][0];body=(' '+texts[first].lower()+' ')
  boost=max(scores_full) *0.65 if max(scores_full)>0 else 0
  remaining=[i for i in range(len(titles)) if i!=first]
  second=sorted(remaining,key=lambda i:(-(scores_full[i]+(boost if len(titles[i])>=5 and (' '+titles[i].lower()+' ') in body else 0)),-scores_title[i],i))
  methods['linked_secondhop']=[first]+second
  pred={'uid':item['_id'],'question_sha256':hashlib.sha256(question.encode()).hexdigest(),
    'source_document_ids_sha256':hashlib.sha256(json.dumps(titles,ensure_ascii=False).encode()).hexdigest(),
    'sources':titles,'ranks':{name:[titles[i] for i in order[:6]] for name,order in methods.items()},
    'sentence_predictions':{titles[i]:sentences_for_query(question,context[i][1]) for i in methods['fulltext_bm25'][:4]}}
  result.append(pred)
  if count%1500==0: print('RANKED',count,'/7405 SECONDS',round(time.monotonic()-start,1),flush=True)
 out={'protocol':'Fixed untuned BM25 lexical multi-document retrieval over 10 provided distractor paragraphs, exact title identity optional second hop + source-neighborhood retrieval; no gold accessed until separately scored.',
      'source_sha256':DATA_SHA,'source':'Yang et al., HotpotQA EMNLP 2018 official distractor development split, CC BY-SA 4.0, https://hotpotqa.github.io/',
      'n_cases':len(result),'run_seconds':round(time.monotonic()-start,2),
      'limitations':['Original text and gold are 2018 Wikipedia snapshots, not guaranteed current factual truth or new scientific discovery','Oracle ten Wikipedia documents are already retrieved by HotpotQA; not full Wikipedia retrieval','Linked second hop uses exact title mention, not learned coreference or DRS','No article answer or supporting fact used in prediction','Source-neighbor within doc is scored by gold only after freezing predictions'],
      'predictions':result}
 OUT.write_text(json.dumps(out,ensure_ascii=False)+'\n')
 print('FROZEN_PREDICTIONS',len(result),'SHA256',hashlib.sha256(OUT.read_bytes()).hexdigest(),'SECONDS',out['run_seconds'])
if __name__=='__main__':main()
