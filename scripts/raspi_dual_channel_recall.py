#!/usr/bin/env python3
"""Read-only original-SHA-attested native two-channel known-source recall.

Requires frozen gold and independently verifies each known HTML publisher
original and derived text. Reports explicit source-label recall on BOTH
independent channels; never updates or trains ranking/model/embeddings.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from file_system_catalog.retrieval_regression import validated_gold,attest_html_original_gold
from file_system_catalog.dual_channel_recall import evaluate_verified_dual_case,aggregate_verified_dual_cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('gold','dual-cli','source-map','original-root','derived-root','completed-ledger','output'):
        p.add_argument('--'+arg,required=True,type=Path)
    p.add_argument('--gold-sha256',required=True)
    p.add_argument('--top',type=int,default=10)
    p.add_argument('--case-id',action='append',help='Do not compare this subtotal to full evaluation')
    args=p.parse_args()
    if type(args.top) is not int or not 1<=args.top<=20:
        raise ValueError('invalid bounded top limit')
    if (args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir()
            or args.output.parent.is_symlink()):
        raise ValueError('private evaluation output must be a new file inside a trusted directory')
    if args.gold.is_symlink() or not args.gold.is_file() or args.gold.stat().st_size>200*1024:
        raise ValueError('untrusted frozen gold source')
    gold_raw=args.gold.read_bytes()
    if hashlib.sha256(gold_raw).hexdigest()!=args.gold_sha256:
        raise ValueError('frozen gold hash changed before reading')
    document=json.loads(gold_raw)
    if (not isinstance(document,dict) or document.get('schema')!='kmd.frozen_multirelevant_original_gold.v1'
            or document.get('frozen_before_any_new_queries') is not True):
        raise ValueError('incorrect predeclared publisher source holdout schema')
    full=validated_gold(document['cases'])
    selected=full
    if args.case_id:
        requested=set(args.case_id)
        if len(requested)!=len(args.case_id) or not requested.issubset({z['id'] for z in full}):
            raise ValueError('invalid, duplicate or unknown case identifier')
        selected=[z for z in full if z['id'] in requested]
    for path in (args.dual_cli,args.source_map,args.completed_ledger):
        if path.is_symlink() or not path.is_file():
            raise ValueError('missing or symlinked canonical source proof dependency')
    with sqlite3.connect(args.source_map.resolve().as_uri()+'?mode=ro',uri=True) as source_map:
        source_map.execute('PRAGMA query_only=ON')
        attested=attest_html_original_gold(
            document['cases'],{z['id'] for z in selected},
            original_root=args.original_root,derivative_root=args.derived_root,
            source_map=source_map,
        )
        # The dual reader validates original hashes with the current approval
        # ledger. Independently check all selected gold publisher files above.
        cases=[]
        for index,case in enumerate(selected,1):
            start=time.monotonic()
            proc=subprocess.run([
                sys.executable,str(args.dual_cli),case['query'],'--top',str(args.top),
                '--source-map',str(args.source_map),
                '--original-root',str(args.original_root),
                '--derivative-root',str(args.derived_root),
                '--download-ledger',str(args.completed_ledger),
            ],capture_output=True,text=True,check=True,timeout=180)
            payload=json.loads(proc.stdout)
            result=evaluate_verified_dual_case(case,payload,top=args.top)
            result['publisher_gold_attested']=attested[case['id']]
            result['elapsed_seconds']=round(time.monotonic()-start,3)
            cases.append(result)
            print('CHECK',index,'/',len(selected),result['id'],
                  'raw',result['raw_topk_hits'],'summary',result['summary_topk_hits'],
                  'either',result['either_channel_topk_hits'],flush=True)
    report={
        'schema':'kmd.live_attested_two_channel_source_recall.v1',
        'captured_utc':datetime.now(timezone.utc).isoformat(),
        'frozen_gold_sha256':hashlib.sha256(gold_raw).hexdigest(),
        'original_source_gold_sha_rechecked':True,
        'complete_fixed_population':len(selected)==len(full),
        'tested_dual_cli':str(args.dual_cli),
        'same_originals_and_source_map_for_dual_search':True,
        'summary':aggregate_verified_dual_cases(cases,top=args.top),
        'cases':cases,
        'warning':'This is a source-candidate identity benchmark, NOT independent answer truth; no model, relevance scoring or database was changed.',
    }
    with args.output.open('x',encoding='utf8') as output:
        json.dump(report,output,indent=2,ensure_ascii=False)
        output.write('\n')
    args.output.chmod(0o600)
    print('TOTAL',json.dumps(report['summary'],ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
