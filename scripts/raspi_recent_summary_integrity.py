#!/usr/bin/env python3
"""Audit newly indexed native recursive summary sources, read-only and bounded.

Reads the real two KMD worker completion journals. Independently rehashes
approved publisher originals, derivatives and every current source-linked
native summary, child/parent span and vector byte digest in recent complete
originals. No index writes, model inference, source download or new services.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from file_system_catalog.recent_source_tree_audit import audit_current_source_tree

UNITS=('knowledge-library-hierarchical-summary.service',
       'knowledge-library-hierarchical-summary-q4.service')


def completed_paths_from_lines(lines: list[str], *, limit: int) -> list[str]:
    """Take recent distinct real worker-completion records; no source content."""
    if type(limit) is not int or not 1<=limit<=25:
        raise ValueError('invalid bounded recent source count')
    sources=[]
    marker='KMD_HIERARCHICAL_DOCUMENT_COMPLETE '
    for line in lines:
        if not isinstance(line,str) or len(line)>4096 or not line.startswith(marker):
            continue
        message=line[len(marker):]
        parts=message.split(' ',1)
        if len(parts)!=2:
            continue
        source=parts[1].strip()
        if not source or len(source)>1600 or '\n' in source or '\r' in source:
            continue
        # Journal can replay the same source if the process restarts.
        if source in sources:
            sources.remove(source)
        sources.append(source)
    return sources[-limit:]


def ledger_receipts(path: Path) -> tuple[dict[str,str],str]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size>64*1024*1024:
        raise ValueError('missing or oversized official completed original receipt ledger')
    raw=path.read_bytes()
    data=json.loads(raw)
    if not isinstance(data,dict) or not isinstance(data.get('items'),list):
        raise ValueError('invalid official approval ledger schema')
    receipts={}
    for record in data['items']:
        if not isinstance(record,dict):
            raise ValueError('invalid original receipt entry')
        if record.get('status')=='complete':
            original=record.get('path')
            checksum=record.get('sha256')
            if (not isinstance(original,str) or original in receipts
                    or not isinstance(checksum,str) or not re.fullmatch('[0-9a-f]{64}',checksum)):
                raise ValueError('duplicate or unverifiable approved publisher source')
            receipts[original]=checksum
    return receipts,hashlib.sha256(raw).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ('source-map','native-db','original-root','derived-root','completed-ledger','output'):
        ap.add_argument('--'+arg,required=True,type=Path)
    ap.add_argument('--since',default='2026-11-21 00:00:00',
                    help='Local systemd journal date; use exact YYYY-MM-DD HH:MM:SS')
    ap.add_argument('--last',type=int,default=15)
    args=ap.parse_args()
    if (type(args.last) is not int or not 1<=args.last<=25
            or not isinstance(args.since,str) or len(args.since)!=19
            or args.since[4]!='-' or args.since[7]!='-' or args.since[10]!=' '
            or args.since[13]!=':' or args.since[16]!=':'):
        raise ValueError('require bounded last and explicit local timestamp')
    try:
        datetime.strptime(args.since,'%Y-%m-%d %H:%M:%S')
    except ValueError as error:
        raise ValueError('invalid strict local journal timestamp') from error
    if (args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir()
            or args.output.parent.is_symlink()):
        raise ValueError('private audit output already exists or directory untrusted')
    for name in (args.source_map,args.native_db):
        if not name.is_file() or name.is_symlink():
            raise ValueError('canonical native source-map/SQLite missing or symlinked')
    log=subprocess.run(['journalctl',*(x for unit in UNITS for x in ('-u',unit)),
                        '--since',args.since,'--no-pager','-o','cat', '-n','6500'],
                       capture_output=True,text=True,timeout=20,check=True)
    paths=completed_paths_from_lines(log.stdout.splitlines(),limit=args.last)
    if not paths:
        raise ValueError('worker journal has no verified completed source events for selected time')
    receipts,ledger_sha=ledger_receipts(args.completed_ledger)
    checks=[]
    with sqlite3.connect(args.source_map.resolve().as_uri()+'?mode=ro',uri=True) as sm, \
         sqlite3.connect(args.native_db.resolve().as_uri()+'?mode=ro',uri=True) as db:
        sm.execute('PRAGMA query_only=ON');db.execute('PRAGMA query_only=ON')
        for source in paths:
            try:
                record=audit_current_source_tree(
                    original_path=source,source_map=sm,native_db=db,
                    original_root=args.original_root,derivative_root=args.derived_root,
                    completed_receipts=receipts)
                checks.append({'status':'verified',**record})
                print('VERIFIED',source.rsplit('/',1)[-1][:90],
                      record['current_native_summary_nodes_fully_verified'],flush=True)
            except Exception as error:
                issue=f'{type(error).__name__}: {str(error)[:220]}'
                state='unverified_oversized' if 'exceeds independent byte audit limit' in str(error) else 'failed_provenance'
                checks.append({'status':state,'original_path':source,'reason':issue})
                print('NOT_VERIFIED',source.rsplit('/',1)[-1][:90],state,flush=True)
    passed=[row for row in checks if row['status']=='verified']
    failed=[row for row in checks if row['status']=='failed_provenance']
    report={
        'schema':'kmd.recent_source_full_original_and_recursive_native_graph_audit.v1',
        'completed_utc':datetime.now(timezone.utc).isoformat(),
        'worker_journal_units':list(UNITS),'source_journal_since_local':args.since,
        'completed_original_receipt_ledger_sha256':ledger_sha,
        'approved_originals_hashed_and_native_graph_verified':len(passed),
        'verified_native_summary_nodes':sum(z['current_native_summary_nodes_fully_verified'] for z in passed),
        'known_source_audit_failures':len(failed),
        'oversized_sources_explicitly_unverified':len(checks)-len(passed)-len(failed),
        'journal_completed_origins_selected':len(paths),
        'queries_and_originals_accessed_read_only':True,
        'no_source_or_filedb_or_model_mutation':True,
        'statement_scope':'Publisher original+derivative SHA, native exact span, model-produced summary and embedding digest/shape, complete source graph; not factual accuracy.',
        'sources':checks,
    }
    fd=os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o600)
    with os.fdopen(fd,'w',encoding='utf8') as handle:
        json.dump(report,handle,indent=2,ensure_ascii=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    print('AUDIT_TOTALS',json.dumps({k:v for k,v in report.items() if k in
        ('approved_originals_hashed_and_native_graph_verified','verified_native_summary_nodes',
         'known_source_audit_failures','oversized_sources_explicitly_unverified')},ensure_ascii=False),flush=True)
    raise SystemExit(2 if failed else 0)


if __name__=='__main__':
    main()
