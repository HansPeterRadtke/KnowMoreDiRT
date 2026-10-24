"""Frozen previously-unused real ContractNLI test documents, single same claim.

Original Stanford/Hitachi human annotations; automatic retrieved source spans
WITHOUT human gold evidence index; class balancing only before inference.
"""
import hashlib,json,re
from pathlib import Path
ROOT=Path('/data/tmp/kmd-contractnli-20261024/contract-nli')
OUT=Path(__file__).parent/'frozen_contract_source_scope_20261024.json'
HYP='nda-20'

def toks(t):return set(re.findall(r'(?u)\b\w\w+\b',t.lower()))

def get_context(d,q):
    source=d['text'];spans=d['spans'];qt=toks(q)
    ranked=[]
    for i,(a,b) in enumerate(spans):
        seg=source[a:b];st=toks(seg)
        ranked.append((len(qt&st)/(len(qt|st) or 1),i))
    ranked.sort(key=lambda x:(-x[0],x[1]))
    top=[i for _,i in ranked[:3]]
    index=sorted({j for i in top for j in range(max(0,i-1),min(len(spans),i+2))})
    evidence='\n'.join(f'Span {i}: {source[spans[i][0]:spans[i][1]].strip()}' for i in index)
    return evidence[:2300],index,top

def main():
    if OUT.exists():raise RuntimeError('FROZEN EXISTS NO REPLACE')
    raw=(ROOT/'test.json').read_bytes();d=json.loads(raw);q=d['labels'][HYP]['hypothesis']
    rows=[]
    for doc in d['documents']:
        gold=doc['annotation_sets'][0]['annotations'][HYP]
        assert gold['choice'] in ('Entailment','Contradiction','NotMentioned')
        key=hashlib.sha256(f"contractnli-20261024-{doc['id']}-{HYP}".encode()).hexdigest()
        evidence,span_indices,top=get_context(doc,q)
        if len(evidence)<140:continue
        rows.append({'id':doc['id'],'file_name':doc['file_name'],'source_url':doc.get('url'),
              'doc_text_sha256':hashlib.sha256(doc['text'].encode()).hexdigest(),'source_excerpt_sha256':hashlib.sha256(evidence.encode()).hexdigest(),
              'source_excerpt':evidence,'source_span_indices':span_indices,'source_top3_indices':top,
              'sort_hash':key,'gold_class_not_shown_to_model':gold['choice']})
    chosen=[]
    for label in ('Entailment','Contradiction','NotMentioned'):
        arr=sorted((x for x in rows if x['gold_class_not_shown_to_model']==label),key=lambda x:x['sort_hash'])
        assert len(arr)>=2
        chosen+=arr[:2]
    chosen.sort(key=lambda x:x['sort_hash'])
    for x in chosen:
        # No oracle evidence source indices included in chosen model input.
        del x['gold_class_not_shown_to_model']
    frozen={'original_test_sha256':hashlib.sha256(raw).hexdigest(),'source_project':'Stanford/Hitachi ContractNLI, CC BY 4.0 original human-annotated test dataset, https://stanfordnlp.github.io/contract-nli/',
        'hypothesis_id':HYP,'hypothesis':q,'selection':'same fixed NDA-20 hypothesis across six distinct public contracts; two of each human gold class selected by predeclared sha256(document id), no human evidence spans in model retrieval; labels hidden in model prompts and separate scoring',
        'source_retrieval':'top3 automatic hypothesis-to-document sentence span Jaccard + local span neighbors; excerpt capped 2300 chars, original source and full gold retained in tmp',
        'cases':chosen,'limitations':['Real legal documents and independent human labels, but single fixed proposition; NOT cross-paper chaining or dream semantics','Linguistic scope differs by contract author, exceptions and obligations; no direct evaluation of full KMD DRS','External test can have pretraining exposure to released corpus, not verified unseen to Qwen']}
    OUT.write_text(json.dumps(frozen,indent=2,ensure_ascii=False)+'\n')
    print('FROZEN',len(chosen),'DOC IDS',[x['id'] for x in chosen],'SHA256',hashlib.sha256(OUT.read_bytes()).hexdigest(),'LENGTHS',[len(x['source_excerpt']) for x in chosen])
if __name__=='__main__':main()
