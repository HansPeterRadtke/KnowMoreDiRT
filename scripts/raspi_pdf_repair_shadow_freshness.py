#!/usr/bin/env python3
"""Prove if a sealed private 327-original PDF rollback bundle has become stale.

Full physical archive SHA verification + exact read-only current live source
identity and all original-linked vector/summary bytes comparison. Fail closed
for a changed source, extra summary, missing historic tree or journal. Never
promote sources, hold writer locks or authorize a live multifile cutover.
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
from file_system_catalog.source_page_shadow_cohort import (
    compare_historical_shadow_with_current_live,verify_private_shadow_bundle,
)
AUDIT=Path('/data/var/knowmoredirt/knowledge-library/search-audit')
SM=Path('/data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3')
NATIVE=Path('/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3')
SUMMARY=Path('/data/var/knowmoredirt/knowledge-library/summaries')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--historical-private-bundle',type=Path,required=True)
    ap.add_argument('--new-private-output',type=Path,required=True)
    args=ap.parse_args()
    bundle=args.historical_private_bundle
    output=args.new_private_output
    if (not bundle.is_absolute() or not output.is_absolute()
            or not bundle.resolve().is_relative_to(AUDIT.resolve())
            or not output.parent.resolve().is_relative_to(AUDIT.resolve())
            or output.exists() or output.is_symlink()):
        raise ValueError('must verify one existing protected historical publisher bundle to new private output')
    verified=verify_private_shadow_bundle(bundle_root=bundle)
    if verified['native_rows_verified']['originals']!=327:
        raise ValueError('private historical source cohort is not complete')
    with sqlite3.connect(SM.resolve().as_uri()+'?mode=ro',uri=True,timeout=120) as sm,\
         sqlite3.connect(NATIVE.resolve().as_uri()+'?mode=ro',uri=True,timeout=120) as db:
        sm.execute('PRAGMA query_only=ON');db.execute('PRAGMA query_only=ON')
        report=compare_historical_shadow_with_current_live(
            historical_db=bundle/'original_native_and_source_map_subset.sqlite3',
            source_map=sm,native=db,expected_originals=327,
            summary_root=SUMMARY,archive_root=bundle)
    report.update({'checked_utc':datetime.now(timezone.utc).isoformat(),
        'historical_snapshot_self_consistent_full_archive_sha_verification':True,
        'historical_bundle_schema':verified['native_rows_verified'],
        'not_a_live_recovery_snapshot_without_new_coordinated_exclusive_writer_lease':True,
        'no_live_files_or_sqlite_databases_mutated':True})
    blob=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf8')
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as writer:
        writer.write(blob);writer.flush();os.fsync(writer.fileno())
    print(json.dumps({
        'status':'historical_snapshot_not_cutover_safe_and_provenance_checked',
        'originals_rechecked':report['historical_originals'],
        'live_summary_rows':report['current_native_summary_representations'],
        'historical_summary_rows':report['historical_native_summary_representations'],
        'live_summary_trees':report['summary_tree_and_progress_revision_counts']['current_source_trees'],
        'historical_summary_trees':report['summary_tree_and_progress_revision_counts']['historical_source_trees'],
        'changed_since_snapshot':report['historical_snapshot_content_is_different_from_current_live'],
        'live_cutover_allowed':False,'receipt_sha256':hashlib.sha256(blob).hexdigest(),
        'report_path':str(output)},sort_keys=True))
if __name__=='__main__':main()
