#!/usr/bin/env python3
"""Read-only, fail-closed dependency inventory before any future PDF repair cutover.

Never stops workers, acquires write leases, rewrites original/derived text,
updates source-map, manipulates native FileDB, deletes or publishes anything.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from file_system_catalog.source_page_cutover_readiness import inspect_staged_cutover

PRIVATE_ROOT=Path('/data/var/knowmoredirt/knowledge-library/search-audit')
SOURCE_MAP=Path('/data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3')
NATIVE=Path('/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3')
ORIGINALS=Path('/data/var/Knowledge Library')
DERIVED=Path('/data/var/knowmoredirt/knowledge-library/semantic/text')
SUMMARY=Path('/data/var/knowmoredirt/knowledge-library/summaries')
RECEIPTS=ORIGINALS/'_management/status.json'
SERVICE_UNITS={
 'kmd-summary-worker-coder':'knowledge-library-hierarchical-summary.service',
 'kmd-summary-worker-q4':'knowledge-library-hierarchical-summary-q4.service',
}


def bounded_private_json(path:Path,max_bytes:int=10*1024*1024):
    if (not path.is_file() or path.is_symlink() or path.stat().st_size>max_bytes
            or not path.resolve().is_relative_to(PRIVATE_ROOT.resolve())):
        raise ValueError('private staged source receipt missing, untrusted or oversized')
    return json.loads(path.read_bytes())


def observe_service_states() -> dict[str,str]:
    found={}
    for label,unit in SERVICE_UNITS.items():
        try:
            result=subprocess.run(['systemctl','show',unit,'--property=ActiveState','--value'],
                     capture_output=True,text=True,timeout=5,check=False)
            value=result.stdout.strip()
            found[label]=value if result.returncode==0 and value in ('active','inactive','failed') else 'unknown'
        except (OSError,subprocess.TimeoutExpired):found[label]='unknown'
    return found


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage-preflight',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    if os.uname().nodename.split('.')[0]!='raspi':
        raise SystemExit('Read-only current native source-map inventory is Raspi-local')
    output=args.output
    if (not output.is_absolute() or not output.parent.is_dir() or output.parent.is_symlink()
            or not output.parent.resolve().is_relative_to(PRIVATE_ROOT.resolve())
            or output.exists() or output.is_symlink()):
        raise ValueError('output must be new in the private search-audit tree')
    receipt=bounded_private_json(args.stage_preflight)
    if (receipt.get('schema')!='kmd.all_327_approved_original_pdf_private_stage_revision_and_physical_page_preflight.v1'
            or receipt.get('source_pdf_originals_verified_and_staged')!=327
            or receipt.get('all_original_old_derived_and_new_staged_full_sha_reverified_after_staging') is not True
            or receipt.get('old_live_source_cache_native_filedb_embeddings_summary_nodes_changed_by_stage') is not False):
        raise ValueError('private full original-revision staging evidence invalid')
    items=receipt.get('original_sources_and_corrected_private_derivative_candidates')
    if not isinstance(items,list) or len(items)!=327:
        raise ValueError('missing 327 independent publisher source staged paths')
    if RECEIPTS.is_symlink() or not RECEIPTS.is_file() or RECEIPTS.stat().st_size>64*1024*1024:
        raise ValueError('approved original acquisition receipt invalid')
    ledger=json.loads(RECEIPTS.read_bytes())
    completed={}
    for item in ledger['items']:
        if item.get('status')!='complete':continue
        p=item['path']
        if p in completed:raise ValueError('ambiguous completed approved original identity')
        completed[p]=item['sha256']
    states=observe_service_states()
    with sqlite3.connect(SOURCE_MAP.resolve().as_uri()+'?mode=ro',uri=True,timeout=25) as sm,\
            sqlite3.connect(NATIVE.resolve().as_uri()+'?mode=ro',uri=True,timeout=25) as native:
        sm.execute('PRAGMA query_only=ON');native.execute('PRAGMA query_only=ON')
        status=inspect_staged_cutover(items,source_map=sm,native=native,
                     original_root=ORIGINALS,derived_root=DERIVED,
                     stage_root=PRIVATE_ROOT,completed_receipts=completed,
                     service_states=states,summary_root=SUMMARY)
    status['checked_utc']=datetime.now(timezone.utc).isoformat()
    status['source_preflight_sha256']=hashlib.sha256(args.stage_preflight.read_bytes()).hexdigest()
    status['no_live_migration_was_attempted']=True
    serialized=(json.dumps(status,ensure_ascii=False,indent=2)+'\n').encode()
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as writer:
        writer.write(serialized);writer.flush();os.fsync(writer.fileno())
    print(json.dumps({'status':'read_only_inventory_complete','sources':status['originals_reverified'],
          'workers_not_quiesced':status['workers_not_quiesced'],
          'safe_to_cutover':status['safe_to_cutover'],
          'native_dependencies':status['native_dependencies'],'report_sha256':hashlib.sha256(serialized).hexdigest(),
          'output':str(output)},sort_keys=True))

if __name__=='__main__':main()
