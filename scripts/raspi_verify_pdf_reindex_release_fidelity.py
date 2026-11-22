#!/usr/bin/env python3
"""Fail-closed post-rebuild release fidelity gate on private original-PDF copies.

The page-corrected source-map can be internally valid but still have no usable
new source embeddings or recursive summaries. Check all required source,
source-span, native embedding and complete tree revisions in an immutable
private clone; NEVER promote a live index or halt resident workers. The
output deliberately cannot authorize a multi-file SQLite/filesystem release.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from file_system_catalog.staged_page_native_bridge import pinned_private_corrected_sources
from file_system_catalog.source_page_reindex_release_gate import assess_corrected_native_release
from file_system_catalog.source_page_shadow_cohort import verify_private_shadow_bundle

PRIVATE=Path('/data/var/knowmoredirt/knowledge-library/search-audit')
ACTIVE=Path('/data/var/knowmoredirt/knowledge-library')


def inside_private(file:Path,*,file_kind:str):
    if not file.is_absolute() or file.is_symlink() or not file.resolve().is_relative_to(PRIVATE.resolve()):
        raise ValueError('release fidelity reads only private audited sources and isolated snapshots')
    if file_kind=='file' and (not file.is_file() or file.stat().st_size>128*1024*1024):
        raise ValueError('expected bounded private source SQLite or receipt file')
    if file_kind=='directory' and not file.is_dir():
        raise ValueError('expected existing private source/summary directory')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--staged-preflight',type=Path,required=True)
    p.add_argument('--private-candidate-db',type=Path,required=True)
    p.add_argument('--historical-private-bundle',type=Path,required=True)
    p.add_argument('--private-derived-root',type=Path,required=True)
    p.add_argument('--private-summary-root',type=Path,required=True)
    p.add_argument('--new-private-report',type=Path,required=True)
    args=p.parse_args()
    for name in ('staged_preflight','private_candidate_db'):
        inside_private(getattr(args,name),file_kind='file')
    for name in ('private_derived_root','private_summary_root','historical_private_bundle'):
        inside_private(getattr(args,name),file_kind='directory')
    verified_archive=verify_private_shadow_bundle(bundle_root=args.historical_private_bundle)
    if verified_archive['native_rows_verified']['originals']!=327:
        raise ValueError('private historical original/native source baseline incomplete')
    historical_db=args.historical_private_bundle/'original_native_and_source_map_subset.sqlite3'
    inside_private(historical_db,file_kind='file')
    destination=args.new_private_report
    if (not destination.is_absolute() or destination.exists() or destination.is_symlink()
            or not destination.parent.is_dir()
            or not destination.parent.resolve().is_relative_to(PRIVATE.resolve())
            or destination.parent.resolve().is_relative_to(args.private_derived_root.resolve())
            or destination.parent.resolve().is_relative_to(args.private_summary_root.resolve())
            or destination.parent.resolve().is_relative_to(args.historical_private_bundle.resolve())
            or destination.parent.resolve().is_relative_to(args.staged_preflight.parent.resolve())):
        raise ValueError('release receipt must be a NEW separate private audit report file')
    # Even accidentally supplying the original 3.0 GB live FileDB cannot
    # result in a write: this script only opens a <=128MB private clone.
    if args.private_candidate_db.resolve().is_relative_to(ACTIVE.resolve()) and not args.private_candidate_db.resolve().is_relative_to(PRIVATE.resolve()):
        raise ValueError('live original and native database are not candidate clones')
    pinned=pinned_private_corrected_sources(stage_preflight=args.staged_preflight,
                                             private_root=PRIVATE)
    if len(pinned)!=327:
        raise ValueError('reindex release fidelity demands all corrected publisher sources')
    with sqlite3.connect(args.private_candidate_db.resolve().as_uri()+'?mode=ro',
                         uri=True,timeout=18) as conn, \
         sqlite3.connect(historical_db.resolve().as_uri()+'?mode=ro',
                         uri=True,timeout=18) as history:
        conn.execute('PRAGMA query_only=ON')
        history.execute('PRAGMA query_only=ON')
        result=assess_corrected_native_release(records=list(pinned.values()),
                 source_map=conn,native=conn,historical=history,
                 derived_root=args.private_derived_root,
                 summary_root=args.private_summary_root,expected_count=327)
    result['checked_utc']=datetime.now(timezone.utc).isoformat()
    result['private_candidate_db_file_sha256']=hashlib.sha256(args.private_candidate_db.read_bytes()).hexdigest()
    result['private_source_receipt_sha256']=hashlib.sha256(args.staged_preflight.read_bytes()).hexdigest()
    result['historical_full_cohort_source_and_native_archive_reverified']=True
    result['historical_bundle_old_source_sha256']=hashlib.sha256(historical_db.read_bytes()).hexdigest()
    result['not_live_original_source_map_or_native_db_production_rebuild']=True
    result['writer_and_reader_generation_not_coordinated']=True
    blob=(json.dumps(result,indent=2,ensure_ascii=False)+'\n').encode()
    fd=os.open(destination,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as out:out.write(blob);out.flush();os.fsync(out.fileno())
    print(json.dumps({'checked_originals':result['cohort_expected_originals'],
         'current_derivatives_sha_verified':result['originals_with_rechecked_source_derivative_sha'],
         'new_native_complete_originals':result['originals_with_native_full_source_roots_and_vectors'],
         'full_new_search_index_ready':result['full_corpus_new_revision_search_ready'],
         'live_release_permitted':False,
         'failure_reasons':result['failures_by_reason'],
         'private_report':str(destination)},sort_keys=True))
if __name__=='__main__':main()
