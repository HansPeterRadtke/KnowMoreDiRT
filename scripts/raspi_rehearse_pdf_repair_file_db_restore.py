#!/usr/bin/env python3
"""Simulate text swap and native DB updates at two crash points in private copies.

Mandatory full historical archive validation, exact original source SHA and
rollback simulation. Live source-map, original books, index and services remain
untouched; this is NOT a cutover or proof of multi-store live atomicity.
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
from file_system_catalog.source_page_shadow_cohort import verify_private_shadow_bundle,rehearse_private_file_database_restore
AUDITS=Path('/data/var/knowmoredirt/knowledge-library/search-audit')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-bundle',type=Path,required=True)
    parser.add_argument('--staged-preflight',type=Path,required=True)
    parser.add_argument('--new-private-rehearsal-dir',type=Path,required=True)
    args=parser.parse_args()
    for path in (args.private_bundle,args.staged_preflight,args.new_private_rehearsal_dir):
        if not path.is_absolute() or not path.parent.resolve().is_relative_to(AUDITS.resolve()):
            raise ValueError('only the private original PDF audit tree can be used for rehearsal')
    if args.new_private_rehearsal_dir.exists():
        raise ValueError('private rehearsal directory already in use')
    verified=verify_private_shadow_bundle(bundle_root=args.private_bundle)
    if not verified['native_original_and_summary_logical_fingerprints_equal']:
        raise ValueError('historical private source rollback proof failed')
    if args.staged_preflight.is_symlink() or args.staged_preflight.stat().st_size>8*1024*1024:
        raise ValueError('invalid publisher stage preflight')
    records=json.loads(args.staged_preflight.read_bytes())
    if (records.get('schema')!='kmd.all_327_approved_original_pdf_private_stage_revision_and_physical_page_preflight.v1'
            or records.get('source_pdf_originals_verified_and_staged')!=327):
        raise ValueError('untrusted staged source cohort')
    record_by_path={x['original_path']:x for x in records['original_sources_and_corrected_private_derivative_candidates']}
    if len(record_by_path)!=327:
        raise ValueError('duplicate staged original identity')
    db=args.private_bundle/'original_native_and_source_map_subset.sqlite3'
    with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as source:
        # A real original with summary rows also exercises FK cascades.
        row=source.execute('''SELECT f.relative_path_display FROM filesystem_entries f
                    JOIN content_chunks c ON c.filesystem_entry_id=f.id
                    JOIN content_representations r ON r.chunk_id=c.chunk_id
                    GROUP BY f.id ORDER BY COUNT(r.representation_id) DESC,f.relative_path_display
                    LIMIT 1''').fetchone()
        if not row:raise ValueError('private original with dependent native summaries unavailable')
        path=row[0]
        record=record_by_path.get(path)
        if not record:raise ValueError('snapshot original missing from source stage evidence')
        staged=Path(record['staged_private_file'])
    report=rehearse_private_file_database_restore(bundle_root=args.private_bundle,
                        staged_new_text=staged,original_path=path,private_root=AUDITS,
                        scenarios_root=args.new_private_rehearsal_dir)
    report['schema']='kmd.isolated_old_source_text_summary_and_native_db_crash_recovery_rehearsal.v1'
    report['checked_utc']=datetime.now(timezone.utc).isoformat()
    report['archive_independently_reverified_before_test']=True
    report['source_originals_in_archive']=327
    output=args.new_private_rehearsal_dir/'private_cross_file_and_sqlite_forced_restore_evidence.json'
    serialized=(json.dumps(report,indent=2,ensure_ascii=False)+'\n').encode()
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as writer:writer.write(serialized);writer.flush();os.fsync(writer.fileno())
    print(json.dumps({'status':'two_private_cross_file_fault_points_recovered',
         'original_native_chunk_rows':report['original_native_chunk_rows'],
         'original_summary_rows':report['original_native_summary_rows'],
         'scenarios_restored':len(report['failure_injection_scenarios']),
         'no_live_file_db_or_worker_mutations':True,'not_a_real_cutover':True,
         'receipt_sha256':hashlib.sha256(serialized).hexdigest(),
         'receipt_path':str(output)},sort_keys=True))

if __name__=='__main__':main()
