#!/usr/bin/env python3
"""Private-only 327 corrected-PDF/source-map/full native dependency dry-run.

Creates new isolated page-correct file and database candidate generations,
plus separately selectable injected failure/recovery cases. This cannot run
a live cutover and never stops or writes the resident indexing workers.
All private publisher content stays off Git and outside user reports.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from file_system_catalog.staged_page_native_bridge import pinned_private_corrected_sources
from file_system_catalog.source_page_shadow_cohort import verify_private_shadow_bundle
from file_system_catalog.source_page_forward_rehearsal import (
  InjectedPrivateForwardFailure,private_forward_generation,private_restore_generation)

PRIVATE=Path('/data/var/knowmoredirt/knowledge-library/search-audit')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-private-bundle',type=Path,required=True)
    parser.add_argument('--staged-preflight',type=Path,required=True)
    parser.add_argument('--new-private-generation-dir',type=Path,required=True)
    parser.add_argument('--new-private-result',type=Path,required=True)
    parser.add_argument('--mode',required=True,choices=('forward','fail_after_164_files',
                  'fail_after_database_commit','restore_old'))
    args=parser.parse_args()
    if not all(isinstance(p,Path) and p.is_absolute() for p in (
            args.historical_private_bundle,args.staged_preflight,args.new_private_generation_dir,args.new_private_result)):
        raise ValueError('all publisher archive/source and private output paths must be absolute')
    for original in (args.historical_private_bundle,args.staged_preflight):
        if original.is_symlink() or not original.resolve().is_relative_to(PRIVATE.resolve()):
            raise ValueError('source is not in the existing protected publisher audit root')
    if args.new_private_result.parent.resolve()!=args.new_private_generation_dir.parent.resolve():
        raise ValueError('private forward and evidence receipt must use one separate exclusive audit parent')
    for new in (args.new_private_generation_dir,args.new_private_result):
        parent=new.parent.resolve()
        if (new.exists() or new.is_symlink()
              or not parent.is_relative_to(PRIVATE.resolve())
              or parent.is_relative_to(args.historical_private_bundle.resolve())
              or parent.is_relative_to(args.staged_preflight.parent.resolve())):
            raise ValueError('new exclusive private outputs must remain outside historical backup and staged source tree')
    archive=verify_private_shadow_bundle(bundle_root=args.historical_private_bundle)
    if archive['native_rows_verified']['originals']!=327:
        raise ValueError('historical original-page recovery archive is incomplete')
    staged=pinned_private_corrected_sources(stage_preflight=args.staged_preflight,
                                              private_root=PRIVATE)
    outcome='PRIVATE_SUCCESS_NOT_SEARCH_READY'
    if args.mode=='restore_old':
        result=private_restore_generation(bundle_root=args.historical_private_bundle,
                        private_root=PRIVATE,output=args.new_private_generation_dir)
        outcome='PRIVATE_OLD_ARCHIVE_RECOVERED_NOT_CURRENT_LIVE_STATE'
    else:
        try:
            result=private_forward_generation(
              historical_db=args.historical_private_bundle/'original_native_and_source_map_subset.sqlite3',
              records=list(staged.values()),stage_root=PRIVATE,private_root=PRIVATE,
              output=args.new_private_generation_dir,fail_after_copies=(164 if args.mode=='fail_after_164_files' else None),
              fail_after_commit=(args.mode=='fail_after_database_commit'))
            if args.mode!='forward':raise RuntimeError('expected private failure injection did not trigger')
        except InjectedPrivateForwardFailure as exc:
            if args.mode=='forward':raise
            outcome='EXPECTED_PRIVATE_ARTIFICIAL_FAILURE_NO_LIVE_EFFECT'
            result={'artificially_injected_failure':str(exc),
                    'no_live_mutation':True,
                    'recovery_from_original_sealed_private_bundle_must_be_rehearsed_separately':True}
    # Verify historical archive still seals even after the isolated exercise.
    verified_after=verify_private_shadow_bundle(bundle_root=args.historical_private_bundle)
    if verified_after['native_rows_verified']!=archive['native_rows_verified']:
        raise ValueError('original private recovery archive changed during simulation')
    receipt={'schema':'kmd.private_pdf_forward_generation_full_cohort_rehearsal.v1',
      'completed_utc':datetime.now(timezone.utc).isoformat(),
      'mode':args.mode,'outcome':outcome,
      'historical_archive_independently_verified_before_and_after':True,
      'historical_archive_is_not_the_current_live_writer_snapshot':True,
      'private_generated_directory':str(args.new_private_generation_dir),
      'original_library_source_map_native_vectors_and_summary_trees_not_modified':True,
      'safe_to_use_this_rehearsal_as_live_cutover':False,
      'no_resident_service_stop_restart_or_dependency_change':True,
      'result':result}
    blob=(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode()
    fd=os.open(args.new_private_result,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
    print(json.dumps({'mode':args.mode,'outcome':outcome,'originals':327,
         'historical_archive_unchanged':True,
         'private_result_sha256':hashlib.sha256(blob).hexdigest(),
         'private_receipt_location':str(args.new_private_result),
         'live_source_map_or_vectors_changed':False,'live_cutover_permitted':False},sort_keys=True))
if __name__=='__main__':main()
