"""Frozen zero-training full-premise lexical baseline for third-party FraCaS dev.

Reads ORIGINAL split unchanged, evaluates after predictions computed. This
selects no parameters and never looks at the train labels.
"""
import csv,hashlib,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
DEV=HERE/'original_dev.csv'

def predict(premise, hypothesis):
    # No external rules/WordNet/LLM, no hidden meaning or paraphrase.
    p=re.sub(r'[^\w]+',' ',premise.casefold()).strip()
    h=re.sub(r'[^\w]+',' ',hypothesis.casefold()).strip()
    return 'entailment' if h and (' '+h+' ') in (' '+p+' ') else 'neutral'

def main():
    rows=list(csv.DictReader(DEV.open(newline='',encoding='utf-8')))
    outputs=[{'index':i,'prediction':predict(r['premise'],r['hypothesis'])} for i,r in enumerate(rows)]
    # Gold is only loaded here after predictions have been frozen in memory.
    gold=[r['label'] for r in rows]
    from collections import Counter
    for out,actual in zip(outputs,gold):out['expected']=actual;out['correct']=out['prediction']==actual
    counts=Counter(x['expected'] for x in outputs)
    conf=Counter((x['expected'],x['prediction']) for x in outputs)
    obj={'source_sha256':hashlib.sha256(DEV.read_bytes()).hexdigest(),'source_split':'fracas/dev','method':'casefolded alphanumeric normalized exact hypothesis substring of the full premise; abstain neutral otherwise; fixed zero-training control','records':len(outputs),'gold_counts':dict(counts),'exact_full_premise_correct':sum(x['correct'] for x in outputs),'all_neutral_correct':counts.get('neutral',0),'confusion':[{'expected':e,'predicted':p,'count':n} for (e,p),n in sorted(conf.items())],'rows':outputs}
    (HERE/'lexical_baseline_results.json').write_text(json.dumps(obj,indent=2)+'\n')
    print('FIXED_FRAcas_BASELINE',obj['exact_full_premise_correct'],'/',obj['records'],'all_neutral',obj['all_neutral_correct'],'confusion',obj['confusion'])
if __name__=='__main__':main()
