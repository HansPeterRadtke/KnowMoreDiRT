#!/usr/bin/env python3
"""Create private, physical-page-correct PDF source derivatives for future review.

STAGING ONLY. Never writes a live original, derivative, source-map, native
KnowMoreDiRT FileDB, vector index or summary. Never interrupts running models
or workers. In particular, this tool cannot publish/promote staged text.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'src'))
sys.path.insert(0,'/data/infra/libexec')
from file_system_catalog.source_page_repair_stage import (
    MAX_ITEMS, MAX_REVIEWED_EXTENDED_PAGES,
    stage_verified_original, extended_original_physical_pdf_pages,
    validate_separately_attested_long_pdf,
)

AUDIT_ROOT=Path('/data/var/knowmoredirt/knowledge-library/search-audit')
SOURCE_MAP=Path('/data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3')
NATIVE_DB=Path('/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3')
ORIGINAL_ROOT=Path('/data/var/Knowledge Library')
DERIVED_ROOT=Path('/data/var/knowmoredirt/knowledge-library/semantic/text')
APPROVED=Path('/data/infra/etc/knowledge-library-sources.json')
LEDGER=Path('/data/var/Knowledge Library/_management/status.json')


def exact_pdfinfo_pages(path:Path)->int:
    try:
        r=subprocess.run(['/usr/bin/pdfinfo',str(path)],capture_output=True,text=True,
                         timeout=10,check=False,env={**os.environ,'LC_ALL':'C'})
    except (OSError,subprocess.TimeoutExpired) as exc:
        raise ValueError('publisher physical PDF count unavailable') from exc
    if r.returncode or len(r.stdout)>32768:
        raise ValueError('publisher original physical PDFinfo failed or excessive output')
    matches=re.findall(r'(?m)^Pages:\s+([1-9][0-9]{0,5})\s*$',r.stdout)
    if len(matches)!=1:return -1
    return int(matches[0])


def read_bounded_json(path:Path,max_bytes:int)->dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size>max_bytes:
        raise ValueError('missing, symlinked or oversized private manifest')
    data=json.loads(path.read_bytes())
    if not isinstance(data,dict):raise ValueError('invalid private JSON manifest')
    return data


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repair-plan',type=Path,required=True)
    ap.add_argument('--offset',type=int,default=0)
    ap.add_argument('--limit',type=int,required=True)
    ap.add_argument('--stage-root',type=Path,required=True,
                    help='NEW directory under the private Raspi search-audit root')
    ap.add_argument('--output',type=Path,required=True,help='NEW private JSON receipt')
    ap.add_argument('--extended-shadow-proof',type=Path,default=None,
                    help='Explicit separately verified four-long-PDF original physical-page proof; one source only')
    args=ap.parse_args()
    if os.uname().nodename.split('.')[0]!='raspi':
        raise SystemExit('Raspi-only private source page staging')
    if (type(args.offset) is not int or not 0<=args.offset<MAX_ITEMS
            or type(args.limit) is not int or not 1<=args.limit<=32):
        raise ValueError('one to thirty-two selected private candidate originals per batch')
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('receipt exists: never overwrite prior source-stage evidence')
    base=AUDIT_ROOT.resolve(strict=True)
    if not args.output.parent.resolve(strict=True).is_relative_to(base):
        raise ValueError('private stage receipt must be within the established Raspi audit root')
    parent=args.stage_root.parent.resolve(strict=True)
    if not parent.is_relative_to(base):
        raise ValueError('stage-root must be within the existing private Raspi audit tree')
    if args.stage_root.exists() or args.stage_root.is_symlink():
        raise FileExistsError('use a new private stage-root for each immutable source batch')
    plan=read_bounded_json(args.repair_plan,512*1024)
    pinned_repair_plan_sha256=hashlib.sha256(args.repair_plan.read_bytes()).hexdigest()
    if plan.get('schema')!='kmd.readonly_source_page_provenance_remediation_manifest_v1':
        raise ValueError('unknown or unchecked source repair proposal schema')
    items=plan.get('records')
    if not isinstance(items,list) or len(items)!=327:
        raise ValueError('expected full independently reviewed 327-source proposal')
    paths=[x.get('source_original_path') for x in items if isinstance(x,dict)]
    if len(set(paths))!=len(items):raise ValueError('duplicate repair candidate path')
    selected=items[args.offset:args.offset+args.limit]
    if not selected:raise ValueError('empty private repair batch')
    extended=None
    extended_sha=None
    if args.extended_shadow_proof is not None:
        if args.limit!=1 or len(selected)!=1:
            raise ValueError('reviewed above-normal PDF page fallback permits exactly one original at a time')
        if not args.extended_shadow_proof.resolve(strict=True).is_relative_to(base):
            raise ValueError('extended physical-page evidence must be from the private audit tree')
        extended=read_bounded_json(args.extended_shadow_proof,1024*1024)
        validate_separately_attested_long_pdf(selected[0],extended)
        extended_sha=hashlib.sha256(args.extended_shadow_proof.read_bytes()).hexdigest()
    raw_manifest=read_bounded_json(APPROVED,32*1024*1024)
    raw_ledger=read_bounded_json(LEDGER,64*1024*1024)
    pinned_approved_source_manifest_sha256=hashlib.sha256(APPROVED.read_bytes()).hexdigest()
    pinned_original_ledger_sha256=hashlib.sha256(LEDGER.read_bytes()).hexdigest()
    approved=raw_manifest.get('sources')
    completed=raw_ledger.get('items')
    if not isinstance(approved,list) or not isinstance(completed,list):
        raise ValueError('malformed approved original source registry')
    by_source={}
    for row in approved:
        path=row.get('path') if isinstance(row,dict) else None
        if not isinstance(path,str) or path in by_source:raise ValueError('duplicate/invalid approved source')
        by_source[path]=row
    receipts={}
    for row in completed:
        if not isinstance(row,dict):raise ValueError('malformed completed original ledger entry')
        if row.get('status')=='complete':
            path=row.get('path');sha=row.get('sha256')
            if (not isinstance(path,str) or path in receipts
                    or not isinstance(sha,str) or not re.fullmatch('[0-9a-f]{64}',sha)):
                raise ValueError('ambiguous original complete acquisition receipt')
            receipts[path]=sha
    args.stage_root.mkdir(mode=0o700,parents=False)
    os.chmod(args.stage_root,0o700)
    # Import AFTER the output has been proven a private, isolated new path.
    from knowledge_library_semantic import _large_pdf_pages
    staged=[]
    with sqlite3.connect(f'{SOURCE_MAP.resolve().as_uri()}?mode=ro',uri=True) as sm,\
         sqlite3.connect(f'{NATIVE_DB.resolve().as_uri()}?mode=ro',uri=True) as native:
        # Resident summarizers can hold short SQLite schema/WAL locks.
        # Wait for readers to become available instead of interrupting them;
        # a bounded timeout preserves the old data on persistent contention.
        sm.execute('PRAGMA busy_timeout=45000')
        native.execute('PRAGMA busy_timeout=45000')
        sm.execute('PRAGMA query_only=ON')
        native.execute('PRAGMA query_only=ON')
        for n,item in enumerate(selected,1):
            rel=item.get('source_original_path','') if isinstance(item,dict) else ''
            try:
                kind=item.get('candidate_existing_extractor_repair_method')
                if extended is not None:
                    validate_separately_attested_long_pdf(item,extended)
                    page_reader=lambda source,scratch:extended_original_physical_pdf_pages(
                        source,scratch,pages=item['publisher_physical_pdf_pages'])
                    max_pages=MAX_REVIEWED_EXTENDED_PAGES
                elif kind=='current_pagewise_fallback_in_private_staging':
                    page_reader=_large_pdf_pages
                    max_pages=60
                else:
                    raise ValueError('source above existing automatic fallback requires an independently approved long-source audit')
                res=stage_verified_original(
                    item=item,source_entry=by_source.get(rel,{}),
                    source_map=sm,native_db=native,completed_receipts=receipts,
                    original_root=ORIGINAL_ROOT,derived_root=DERIVED_ROOT,
                    stage_dir=args.stage_root,
                    extract_page_units=page_reader,actual_pdf_count=exact_pdfinfo_pages,
                    max_physical_pages=max_pages)
                staged.append(res)
                print('STAGED_VALIDATED',args.offset+n,'/',len(items),
                      'PHYSICAL_PAGES',res['current_physical_pdf_pages'],
                      'NEW_SHA',res['staged_sha256'][:16],flush=True)
            except (OSError,ValueError,sqlite3.DatabaseError,RuntimeError) as err:
                staged.append({'status':'staging_rejected','source_path':str(rel)[:1600],
                               'reason':f'{type(err).__name__}: {str(err)[:180]}'})
                print('STAGING_REJECTED',args.offset+n,'/',len(items),
                      str(rel).rsplit('/',1)[-1][:68],f'{type(err).__name__}: {str(err)[:120]}',flush=True)
    if (extended_sha is not None
            and hashlib.sha256(args.extended_shadow_proof.read_bytes()).hexdigest()!=extended_sha):
        raise ValueError('independent long PDF physical-page proof changed during staging')
    if (hashlib.sha256(args.repair_plan.read_bytes()).hexdigest()!=pinned_repair_plan_sha256
            or hashlib.sha256(APPROVED.read_bytes()).hexdigest()!=pinned_approved_source_manifest_sha256
            or hashlib.sha256(LEDGER.read_bytes()).hexdigest()!=pinned_original_ledger_sha256):
        raise ValueError('approved plan or source authorization ledger changed during private staging')
    receipt={'schema':'kmd.private_original_pdf_repair_staging_batch.v1',
             'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'stage_mode':'private SHA-revision-checked physical page derivative only',
             'repair_plan_sha256':pinned_repair_plan_sha256,
             'approved_curated_source_manifest_sha256':pinned_approved_source_manifest_sha256,
             'approved_complete_original_ledger_sha256':pinned_original_ledger_sha256,
             'separately_verified_long_original_page_shadow_sha256':extended_sha,
             'offset':args.offset,'limit':args.limit,'source_records_attempted':len(selected),
             'staged_validated':sum(x['status']=='staged_private_only' for x in staged),
             'staging_rejected':sum(x['status']=='staging_rejected' for x in staged),
             'no_live_original_pdf_old_derived_source_map_native_vectors_summary_worker_writes_by_stager':True,
             'staged_original_page_excerpt_physical_verification':'Not yet requested; individual publisher page excerpt attestation remains necessary before promotion.',
             'can_promote_without_separately_coordinated_transactions':False,
             'staged':staged}
    fd=os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w',encoding='utf8') as out:
        json.dump(receipt,out,ensure_ascii=False,indent=2)
        out.write('\n')
        out.flush();os.fsync(out.fileno())
    print('PRIVATE_STAGE_RECEIPT',args.output,'VALIDATED',receipt['staged_validated'],
          'REJECTED',receipt['staging_rejected'],'SHA',hashlib.sha256(args.output.read_bytes()).hexdigest(),flush=True)
    return 1 if receipt['staging_rejected'] else 0


if __name__=='__main__':
    raise SystemExit(main())
