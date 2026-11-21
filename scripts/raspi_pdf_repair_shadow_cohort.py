#!/usr/bin/env python3
"""Build a private 327-original selective recovery bundle + SQLite rollback rehearsal.

This is NOT a real PDF repair cutover and NEVER pauses services or modifies
live originals, native FileDB, source-map, vectors, caches, or summaries.
A writer lease, newer snapshot, cross-store recovery and original-page proofs
are still mandatory before any live promotion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from file_system_catalog.source_page_cutover_readiness import inspect_staged_cutover
from file_system_catalog.source_page_shadow_cohort import export_cohort_snapshot,rehearse_isolated_sqlite_rollback
from file_system_catalog.source_review_audit import safe_file,sha_file
from file_system_catalog.summary_coverage import revision_identity

AUDIT_ROOT=Path('/data/var/knowmoredirt/knowledge-library/search-audit')
APPROVED_ROOT=Path('/data/var/Knowledge Library')
OLD_DERIVED=Path('/data/var/knowmoredirt/knowledge-library/semantic/text')
NATIVE=Path('/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3')
SOURCE_MAP=Path('/data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3')
SUMMARIES=Path('/data/var/knowmoredirt/knowledge-library/summaries')
SERVICES={'kmd-summary-worker-coder':'knowledge-library-hierarchical-summary.service',
          'kmd-summary-worker-q4':'knowledge-library-hierarchical-summary-q4.service'}


def private_staging_records(path:Path):
    if (not path.is_file() or path.is_symlink() or path.stat().st_size>8*1024*1024
            or not path.resolve().is_relative_to(AUDIT_ROOT.resolve())):
        raise ValueError('original source staged preflight receipt unavailable')
    data=json.loads(path.read_bytes())
    if (data.get('schema')!='kmd.all_327_approved_original_pdf_private_stage_revision_and_physical_page_preflight.v1'
            or data.get('source_pdf_originals_verified_and_staged')!=327
            or data.get('all_original_old_derived_and_new_staged_full_sha_reverified_after_staging') is not True):
        raise ValueError('staged publisher original cohort not independently approved')
    records=data.get('original_sources_and_corrected_private_derivative_candidates')
    if not isinstance(records,list) or len(records)!=327:raise ValueError('incomplete repair cohort')
    return records


def approved_complete_receipts():
    source=APPROVED_ROOT/'_management/status.json'
    if source.is_symlink() or not source.is_file() or source.stat().st_size>64*1024*1024:
        raise ValueError('approved source acquisition receipt missing')
    completed={}
    for item in json.loads(source.read_bytes())['items']:
        if item.get('status')!='complete':continue
        key=item['path']
        if key in completed:raise ValueError('duplicate approved original acquisition identity')
        completed[key]=item['sha256']
    return completed


def service_states():
    import subprocess
    states={}
    for label,unit in SERVICES.items():
        try:
            result=subprocess.run(['systemctl','show',unit,'--property=ActiveState','--value'],
                                   capture_output=True,text=True,timeout=5)
            state=result.stdout.strip()
            states[label]=state if result.returncode==0 and state in ('active','inactive','failed') else 'unknown'
        except (OSError,subprocess.TimeoutExpired):states[label]='unknown'
    return states


def copy_old_provenance(source:Path,relative:str,target_root:Path,expected_sha:str):
    source=safe_file(source,relative)
    dest=target_root/relative
    if dest.exists() or dest.is_symlink():raise ValueError('refusing to replace recovery bundle source')
    dest.parent.mkdir(parents=True,exist_ok=True)
    os.chmod(dest.parent,0o700)
    fd=os.open(dest,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as output,source.open('rb') as inp:
        shutil.copyfileobj(inp,output,length=1024*1024)
        output.flush();os.fsync(output.fileno())
    if sha_file(source)!=expected_sha or sha_file(dest)!=expected_sha:
        raise ValueError('private saved old source revision changed or copy differs')
    return dest.stat().st_size


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-preflight',required=True,type=Path)
    parser.add_argument('--new-private-bundle-dir',required=True,type=Path)
    args=parser.parse_args()
    if os.uname().nodename.split('.')[0]!='raspi':
        raise SystemExit('original recovery rehearsal must run on native Raspi read-only metadata')
    folder=args.new_private_bundle_dir
    if (not folder.is_absolute() or not folder.parent.is_dir() or folder.exists()
            or folder.is_symlink() or not folder.parent.resolve().is_relative_to(AUDIT_ROOT.resolve())):
        raise ValueError('private recovery bundle path must be new and within private audit storage')
    records=private_staging_records(args.stage_preflight)
    receipts=approved_complete_receipts()
    observed=service_states()
    # Validate all live original + old derivative + new staged SHA revisions and
    # native summary-tree/source-map dependencies BEFORE any private export.
    with sqlite3.connect(SOURCE_MAP.resolve().as_uri()+'?mode=ro',uri=True,timeout=20) as sm,\
         sqlite3.connect(NATIVE.resolve().as_uri()+'?mode=ro',uri=True,timeout=20) as native:
        sm.execute('PRAGMA query_only=ON');native.execute('PRAGMA query_only=ON')
        verified=inspect_staged_cutover(records,source_map=sm,native=native,
               original_root=APPROVED_ROOT,derived_root=OLD_DERIVED,
               stage_root=AUDIT_ROOT,completed_receipts=receipts,
               service_states=observed,summary_root=SUMMARIES)
    if verified['originals_reverified']!=327 or verified['safe_to_cutover'] is not False:
        raise ValueError('unexpected source revision verification result')
    folder.mkdir(mode=0o700)
    print('SELECTIVE_SNAPSHOT_START',folder,flush=True)
    files=[];size=0
    # No publisher files are modified. Preserve old live text and currently
    # revision-pinned summary tree + progress files, for future recovery.
    for i,z in enumerate(verified['proofs'],1):
        rel=z['original_path']+'.txt'
        saved=copy_old_provenance(OLD_DERIVED,rel,folder/'old_derived',z['old_derived_sha256'])
        size+=saved;files.append({'saved_relative_path':'old_derived/'+rel,'sha256':z['old_derived_sha256'],'bytes':saved})
        art=z['private_summary_artifacts']
        if art['old_summary_tree_exists']:
            prefix=re.sub(r'[^A-Za-z0-9_.-]+','-',Path(z['original_path']).name[:64])
            name=prefix+'.'+revision_identity(z['original_path'],z['original_sha256'],z['old_derived_sha256'])[:24]+'.summary-tree.json'
            for label,suffix in (('old_summary_tree',''),('old_progress_journal','.progress.json')):
                subfolder='progress/' if suffix else ''
                rel=subfolder+name+suffix
                saved=copy_old_provenance(SUMMARIES,rel,folder/'old_summaries',art[label+'_sha256'])
                size+=saved;files.append({'saved_relative_path':'old_summaries/'+rel,'sha256':art[label+'_sha256'],'bytes':saved})
        if i%100==0:print('COPIED_SAFE_OLD_SOURCE_REVISIONS',i,flush=True)
    out=folder/'original_native_and_source_map_subset.sqlite3'
    with sqlite3.connect(SOURCE_MAP.resolve().as_uri()+'?mode=ro',uri=True,timeout=20) as sm,\
         sqlite3.connect(NATIVE.resolve().as_uri()+'?mode=ro',uri=True,timeout=20) as native:
        sm.execute('PRAGMA query_only=ON');native.execute('PRAGMA query_only=ON')
        exported=export_cohort_snapshot(original_receipts=verified['proofs'],
               source_map=sm,native=native,dest=out,expected_count=327)
    # This only updates the new PRIVATE SQLite copy; the forced delete of
    # old vector and summary rows is always ROLLED BACK before returning.
    rollback=rehearse_isolated_sqlite_rollback(out,batch_size=327)
    if (not rollback['tables_identical_after_forced_rollback']
            or rollback['originals_exercised']!=327):
        raise ValueError('shadow DB rollback rehearsal failed')
    with sqlite3.connect(out.resolve().as_uri()+'?mode=ro',uri=True) as check:
        if check.execute('pragma quick_check').fetchone()[0]!='ok' or check.execute('pragma foreign_key_check').fetchall():
            raise ValueError('private recovery snapshot lost integrity after simulated rollback')
    filehash=sha_file(out)
    report={'schema':'kmd.private_cohort_original_pdf_rollback_shadowsnapshot.v1',
       'created_utc':datetime.now(timezone.utc).isoformat(),
       'originals':327,'independently_reverified_publisher_old_and_new_source_sha':True,
       'newest_live_state_during_capture':'summary_services_currently_active_never_coordinated_for_live_cutover',
       'service_states_at_preflight':observed,
       'preserved_old_derived_and_generated_source_artifacts':len(files),
       'private_saved_old_content_total_bytes':size,
       'archived_file_sha_manifest':files,
       'native_shadow_snapshot':{'file':'original_native_and_source_map_subset.sqlite3',
                                 'sha256':filehash,'bytes':out.stat().st_size,**exported},
       'isolated_delete_then_forced_rollback':rollback,
       'no_live_original_cache_source_map_vector_summary_or_worker_changes':True,
       'not_an_authorized_multi_file_migration':True,
       'not_safe_as_a_future_recovery_point_without_fresh_quiesced_source_revision_check':True}
    digest=folder/'sealed_read_only_recovery_rehearsal.json'
    blob=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf8')
    fd=os.open(digest,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as fp:fp.write(blob);fp.flush();os.fsync(fp.fileno())
    print(json.dumps({'status':'private_shadow_originals_rollback_rehearsed',
       'files_preserved':len(files),'bytes':size,'native_counts':exported['counts'],
       'rollback':rollback,'service_states':observed,'not_cutover_ready':True,
       'receipt_sha256':hashlib.sha256(blob).hexdigest(),'receipt':str(digest)},sort_keys=True))

if __name__=='__main__':main()
