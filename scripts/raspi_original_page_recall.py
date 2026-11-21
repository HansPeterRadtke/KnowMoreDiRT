#!/usr/bin/env python3
"""Read-only source-original-verified physical PDF page recall benchmark.

Verifies a pre-frozen original physical PDF quote/page before native search.
Scores only which original PDF pages are returned, not answer truth or global
retrieval accuracy. The tool writes only a NEW owner-private report, and the
existing KMD source reader may also append to its own normal private audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from file_system_catalog.original_page_recall import validate_publisher_page_gold,score_source_evidence_pages


def approved_receipts(path:Path) -> dict[str,str]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size>64*1024*1024:
        raise ValueError('official source acquisition ledger unavailable or too large')
    doc=json.loads(path.read_bytes())
    if not isinstance(doc,dict) or not isinstance(doc.get('items'),list):
        raise ValueError('invalid approved publisher acquisitions')
    receipts={}
    for row in doc['items']:
        if not isinstance(row,dict):
            raise ValueError('invalid publisher acquisition receipt entry')
        if row.get('status')=='complete':
            name=row.get('path');sha=row.get('sha256')
            if (not isinstance(name,str) or name in receipts or not isinstance(sha,str)
                    or not re.fullmatch('[0-9a-f]{64}',sha)):
                raise ValueError('ambiguous or unhashed approved original')
            receipts[name]=sha
    return receipts


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ('gold','source-map','original-root','derived-root','completed-ledger','source-cli','output'):
        ap.add_argument('--'+arg,required=True,type=Path)
    ap.add_argument('--gold-sha256',required=True)
    ap.add_argument('--top',type=int,default=5)
    ap.add_argument('--case-id',action='append',help='Selected subtotal is not the whole population')
    args=ap.parse_args()
    if type(args.top) is not int or not 1<=args.top<=10:
        raise ValueError('physical PDF page recall top cutoff must be 1..10')
    if (args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir()
            or args.output.parent.is_symlink()):
        raise ValueError('new private physical page report path required')
    for p in (args.gold,args.source_map,args.source_cli):
        if not p.is_file() or p.is_symlink():
            raise ValueError('source or pinned holdout missing or symlinked')
    if args.gold.stat().st_size>128*1024:
        raise ValueError('physical page gold record too large')
    content=args.gold.read_bytes()
    if hashlib.sha256(content).hexdigest()!=args.gold_sha256:
        raise ValueError('physical page gold SHA differs from frozen original')
    frozen=json.loads(content)
    receipts=approved_receipts(args.completed_ledger)
    with sqlite3.connect(args.source_map.resolve().as_uri()+'?mode=ro',uri=True) as sm:
        sm.execute('PRAGMA query_only=ON')
        verified=validate_publisher_page_gold(frozen=frozen,source_map=sm,
            original_root=args.original_root,derivative_root=args.derived_root,
            completed_receipts=receipts)
    if args.case_id:
        requested=set(args.case_id)
        if len(requested)!=len(args.case_id) or not requested.issubset({x['id'] for x in verified}):
            raise ValueError('unknown or repeated requested gold case')
        selected=[x for x in verified if x['id'] in requested]
    else:
        selected=verified
    if not selected:
        raise ValueError('no selected publisher PDF page cases')
    rows=[]
    for ix,case in enumerate(selected,1):
        response=subprocess.run([
            sys.executable,str(args.source_cli),'source-evidence',case['query'],
            '--original-path',case['source_original_path'],'--top',str(args.top),
            '--snippet-characters','1300',
        ],capture_output=True,text=True,timeout=90,check=True)
        result=json.loads(response.stdout)
        row=score_source_evidence_pages(gold=case,result=result,top=args.top)
        rows.append(row)
        print('CHECK',ix,'/',len(selected),row['id'],'gold_page',row['expected_independently_verified_physical_pdf_page'],
              'rank',row['first_verified_gold_page_rank_at_top'],flush=True)
    report={
        'schema':'kmd.native_original_physical_page_recall.v1',
        'captured_utc':datetime.now(timezone.utc).isoformat(),
        'independent_frozen_original_gold_sha256':args.gold_sha256,
        'original_pdf_and_derived_source_full_sha_and_gold_page_reverified':True,
        'complete_frozen_case_population':len(selected)==len(verified),
        'selected_cases':len(selected),'cutoff':args.top,
        'independent_verified_gold_page_hits_at_top':sum(x['gold_page_hit'] for x in rows),
        'gold_page_rank_one_hits':sum(x['first_verified_gold_page_rank_at_top']==1 for x in rows),
        'scope':'Source-restricted exact original PDF physical page recall. Not answer factuality, source selection correctness, or exhaustive relevance.',
        'source_files_or_native_filedb_changed':False,'cases':rows,
    }
    fd=os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o600)
    with os.fdopen(fd,'w',encoding='utf8') as out:
        json.dump(report,out,indent=2,ensure_ascii=False)
        out.write('\n');out.flush();os.fsync(out.fileno())
    print('TOTAL',report['independent_verified_gold_page_hits_at_top'],'/',len(rows),
          'TOP1',report['gold_page_rank_one_hits'],flush=True)


if __name__=='__main__':
    main()
