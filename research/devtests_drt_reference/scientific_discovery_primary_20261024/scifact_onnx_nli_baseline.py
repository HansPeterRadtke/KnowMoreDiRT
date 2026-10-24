"""Predeclared OFF-THE-SHELF scientific evidence NLI probe, CPU-only.

One generic SNLI+MultiNLI model, zero fine-tuning and zero label tuning.
Uses actual source sentences and claim text; does NOT consult SciFact evidence
labels to choose sentences or make predictions. Oracle-document experiment is
separate from fully retrieved BM25 top document. Dev was already used for
retrieval baseline, so these are diagnostic results, NOT pristine final holdout.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent
MODEL=Path('/data/tmp/kmd-scientific-nli-20261024')
DATA=Path('/data/tmp/kmd-scifact-release-20261024/data')
EXPECTED_MODEL_BYTES=82823065
LABELS=['CONTRADICT','SUPPORT','NEUTRAL']

def jsonl(f):return [json.loads(z) for z in f.read_text().splitlines() if z]

def pick_sentences(question,sentences):
    # fixed, no gold sentence access. Mirrors prior lexical rationale selection.
    import re
    t=set(re.findall(r'(?u)\b\w\w+\b',question.casefold()))
    scores=[]
    for i,s in enumerate(sentences):
        u=set(re.findall(r'(?u)\b\w\w+\b',s.casefold())); scores.append((len(t&u)/(len(t|u) or 1),i))
    return [i for _,i in sorted(scores,key=lambda a:(-a[0],a[1]))[:2]]

def collect_tasks(claims,docs,retrieved):
    questions=[];seen={};tasks=[]
    for c,record in zip(claims,retrieved['results']):
        assert c['id']==record['id']
        selected={'retrieved_top1':[record['methods']['abstract_bm25']['top_1'][0]],'oracle_evidence_document':list(c['evidence'].keys())}
        for setting,ids in selected.items():
            for docid in ids:
                doc=docs[str(docid)];sids=pick_sentences(c['claim'],doc['abstract'])
                for sid in sids:
                    pair=(c['claim'],doc['abstract'][sid])
                    key=hashlib.sha256(json.dumps(pair,ensure_ascii=False).encode()).hexdigest()
                    if key not in seen:
                        seen[key]=len(questions);questions.append(pair)
                    tasks.append({'claim_id':c['id'],'setting':setting,'doc_id':str(docid),'sentence_index':sid,'pair_index':seen[key],'sentence_sha256':hashlib.sha256(doc['abstract'][sid].encode()).hexdigest()})
    return questions,tasks

def classify(scores):
    # fixed evidence selection: largest (max nonneutral - neutral) margin.
    # No probability or threshold calibration on dev labels.
    vals=[]
    for sc in scores:
        scores=sc['probabilities'];confidence=max(scores[:2])-scores[2]
        vals.append((confidence,int(np.argmax(scores[:2])),sc))
    if not vals:return 'NEUTRAL',None,None
    best=max(vals,key=lambda x:x[0]);cls=LABELS[best[1]] if best[0]>0 else 'NEUTRAL'
    return cls,best[2]['sentence_index'],round(best[0],5)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-claims',type=int,default=300)
    args=parser.parse_args();assert args.max_claims==300, 'protocol fixed at 300; do not select after results'
    onnx=MODEL/'onnx/model_qint8_arm64.onnx';assert onnx.stat().st_size==EXPECTED_MODEL_BYTES
    rpath=HERE/'scifact_fixed_retrieval_results_20261024.json';retrieved=json.loads(rpath.read_text())
    claims=jsonl(DATA/'claims_dev.jsonl');docs={str(d['doc_id']):d for d in jsonl(DATA/'corpus.jsonl')}
    assert len(claims)==len(retrieved['results'])==300
    questions,tasks=collect_tasks(claims,docs,retrieved)
    start=time.perf_counter();tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    so=ort.SessionOptions();so.intra_op_num_threads=2;so.inter_op_num_threads=1
    session=ort.InferenceSession(str(onnx),sess_options=so,providers=['CPUExecutionProvider'])
    labels=[]
    for offset in range(0,len(questions),24):
        pair=questions[offset:offset+24]
        enc=tok([x[1] for x in pair],[x[0] for x in pair],max_length=192,truncation=True,padding=True,return_tensors='np')
        inputs={x.name:enc[x.name].astype('int64') for x in session.get_inputs()}
        logits=session.run(None,inputs)[0]
        for l in logits:
            z=l-np.max(l);probs=np.exp(z);probs/=probs.sum();labels.append([round(float(x),6) for x in probs])
        if offset%240==0:print('INFERRED',min(offset+24,len(questions)),'/',len(questions),'elapsed',round(time.perf_counter()-start,1),flush=True)
    assert len(labels)==len(questions)
    bykey={}
    for task in tasks:
        idx=task['pair_index'];t={**task,'probabilities':labels[idx]}
        bykey.setdefault((task['claim_id'],task['setting'],task['doc_id']),[]).append(t)
    predictions=[]
    for c,row in zip(claims,retrieved['results']):
        all_preds={}
        for setting in ('retrieved_top1','oracle_evidence_document'):
            idset=([str(row['methods']['abstract_bm25']['top_1'][0])] if setting=='retrieved_top1' else list(c['evidence'].keys()))
            for docid in idset:
                lab,sid,margin=classify(bykey.get((c['id'],setting,docid),[]))
                gold=set(x['label'] for x in c['evidence'].get(docid,[]))
                entry={'doc_id':docid,'predicted':lab,'selected_sentence_index':sid,'margin':margin,'gold_label':next(iter(gold)) if len(gold)==1 else ('MIXED' if gold else 'NO_ANNOTATED_EVIDENCE'),
                       'true_evidence_doc':docid in c['evidence']}
                all_preds.setdefault(setting,[]).append(entry)
        predictions.append({'claim_id':c['id'],'claim_sha256':hashlib.sha256(c['claim'].encode()).hexdigest(),'predictions':all_preds})
    gold_doc=[x for y in predictions for x in y['predictions']['oracle_evidence_document'] if x['gold_label']!='MIXED']
    retrieved_labeled=[x['predictions']['retrieved_top1'][0] for x in predictions if x['predictions']['retrieved_top1'][0]['true_evidence_doc']]
    retrieved_unlabeled=[x['predictions']['retrieved_top1'][0] for x in predictions if not x['predictions']['retrieved_top1'][0]['true_evidence_doc']]
    metrics={'oracle_gold_source_labelled_pairs':len(gold_doc),'oracle_gold_source_correct':sum(x['predicted']==x['gold_label'] for x in gold_doc),'oracle_gold_source_neutral':sum(x['predicted']=='NEUTRAL' for x in gold_doc),'top1_labeled_retrieved_pairs':len(retrieved_labeled),'top1_retrieved_correct_label':sum(x['predicted']==x['gold_label'] for x in retrieved_labeled),'top1_unlabeled_retrieved':len(retrieved_unlabeled),'top1_unlabeled_model_asserted_non_neutral':sum(x['predicted']!='NEUTRAL' for x in retrieved_unlabeled),'model_onnx_sha256':hashlib.sha256(onnx.read_bytes()).hexdigest(),'model_repo_revision':'b95119ce93d3e065de6214e38cd4a97b0f2f2c6d','seconds':round(time.perf_counter()-start,2),'pair_count':len(questions),'raw_sentence_predictions':len(labels),'claim_count':len(claims),
    'limitations':['Uses pre-consumed SciFact dev, no final fresh heldout after method selection','Oracle gold-source label accuracy is conditional on human-selected relevant document, not automatic document retrieval','Unlabeled retrieved document is absence of annotation, not necessarily false claim; the non-neutral count measures ungrounded assertion relative to dataset annotation','Model trained on SNLI/MultiNLI, not specialized science, so domain transfer can fail','Does not implement KMD DRS or scientific causal discovery','Automatic lexical top-two sentence selection may miss gold rationales']}
    output={'metrics':metrics,'predictions':predictions,'tasks':tasks,'label_order':LABELS,'nli_config':{'model':'cross-encoder/nli-MiniLM2-L6-H768','optimized':'onnx/model_qint8_arm64.onnx','n_threads':2,'max_token_length':192,'batch_size':24,'sentence_picker':'token Jaccard top2 with no gold','decision':'max nonneutral-minus-neutral across top 2, require positive margin'}}
    target=HERE/'scifact_onnx_nli_results_20261024.json'
    if target.exists():raise RuntimeError('Refuse to overwrite already performed inference')
    target.write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    print('FINISHED_METRICS',metrics,flush=True)
if __name__=='__main__':main()
