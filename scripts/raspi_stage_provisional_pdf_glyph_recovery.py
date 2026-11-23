#!/usr/bin/env python3
"""Stage one *unapproved* physical-PDF glyph preview for owner-private review.

Use only approved original and private previously verified 327-page staged
source. Never modify source originals, live text, source-map, native vectors
or summaries; never stop resident workers or claim a release can proceed.
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
from file_system_catalog.alternate_pdf_page_text import extract_alternate_original_pdf_pages
from file_system_catalog.alternate_pdf_page_search import exact_skipped_staged_physical_pages
from file_system_catalog.private_provisional_pdf_recovery import assemble_unverified_original_pdf_glyph_preview
from raspi_staged_pdf_alternate_glyph_search import current_revision
from raspi_staged_pdf_original_page_search import private_receipt,completed_acquisitions,AUDITS,SOURCES


def exclusive_write(path:Path,body:bytes):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(body);f.flush();os.fsync(f.fileno())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-preflight',type=Path,required=True)
    parser.add_argument('--original-path',required=True)
    parser.add_argument('--private-new-preview-dir',type=Path,required=True,
      help='ONE entirely new directory, under the existing protected Raspi search-audit tree')
    args=parser.parse_args()
    out=args.private_new_preview_dir
    if (not isinstance(args.original_path,str) or not 1<=len(args.original_path)<=1600
          or args.original_path.startswith('/') or '\\' in args.original_path
          or not args.original_path.endswith('.pdf')
          or any(p in ('','.','..') for p in args.original_path.split('/'))):
        raise ValueError('only one exact approved original PDF source identity is allowed')
    if (not out.is_absolute() or out.exists() or out.is_symlink()
            or not out.parent.is_dir() or out.parent.is_symlink()
            or not out.parent.resolve().is_relative_to(AUDITS.resolve())):
        raise ValueError('output must be a new protected owner-private source review directory')
    pinned_preflight_sha=hashlib.sha256(args.stage_preflight.read_bytes()).hexdigest()
    rows=[x for x in private_receipt(args.stage_preflight) if x['original_path']==args.original_path]
    if len(rows)!=1:raise ValueError('original not in approved 327 previously staged physical-PDF correction cohort')
    item=rows[0]
    ledger=SOURCES/'_management/status.json'
    pinned_ledger_sha=hashlib.sha256(ledger.read_bytes()).hexdigest()
    pinned=current_revision(item,completed_acquisitions())
    old_text=pinned['new_staged_source'].read_text('utf8','strict')
    n=item['original_pdf_physical_pages']
    omitted=exact_skipped_staged_physical_pages(old_text,n,item['new_max_staged_page_label'])
    if not 1<=len(omitted)<=32:
        raise ValueError('source does not have one to thirty-two eligible damaged original pages for a bounded review')
    alternate=extract_alternate_original_pdf_pages(pinned['original_pdf'],
                pinned['original_sha256'],pages=omitted,expected_physical_page_count=n)
    result=assemble_unverified_original_pdf_glyph_preview(old_text,
                 physical_page_count=n,alternate_pages=alternate['pages'])
    final=current_revision(item,completed_acquisitions())
    if (final!=pinned or hashlib.sha256(args.stage_preflight.read_bytes()).hexdigest()!=pinned_preflight_sha
            or hashlib.sha256(ledger.read_bytes()).hexdigest()!=pinned_ledger_sha):
        raise ValueError('source, approved receipt, live old or staged revision changed during private preparation')
    payload=result.pop('preview_text').encode('utf8')
    if hashlib.sha256(payload).hexdigest()!=result['unreleased_private_preview_sha256']:
        raise ValueError('private preview bytes unexpectedly changed')
    manifest={**result,'status':'PRIVATE_UNAPPROVED_MACHINE_PDF_TEXT_PREVIEW_NOT_FOR_LIVE_CUTOVER',
      'created_utc':datetime.now(timezone.utc).isoformat(),
      'original_source_path':args.original_path,
      'publisher_original_sha256':pinned['original_sha256'],
      'live_old_derived_sha256':pinned['old_derived_sha256'],
      'prior_private_staged_sha256':pinned['staged_derived_sha256'],
      'native_original_filedb_id':pinned['native_id'],
      'source_preflight_receipt_sha256':pinned_preflight_sha,
      'approved_original_acquisition_ledger_sha256':pinned_ledger_sha,
      'all_original_page_source_and_approved_revisions_rechecked_before_and_after':True,
      'first_original_pdf_page_count_and_sha_verified_before_and_after_extraction':True,
      'alternate_machine_extractors_need_original_visual_quality_and_layout_review':True,
      'no_native_source_map_filedb_vector_or_summary_cutover_allowed':True,
      'no_source_original_or_live_derived_files_modified':True,
      'no_external_network_OCR_or_services':True,
      'private_preview_file':'provisional-unapproved-physical-original-PDF-page-text.txt',
      'alternate_page_diagnostics_without_original_book_text':[
         {k:v for k,v in p.items() if k!='text'} for p in alternate['pages']]}
    blob=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf8')
    out.mkdir(mode=0o700,parents=False)
    os.chmod(out,0o700)
    preview=out/'provisional-unapproved-physical-original-PDF-page-text.txt'
    meta=out/'provisional-preview-private-manifest.json'
    exclusive_write(preview,payload)
    exclusive_write(meta,blob)
    if (hashlib.sha256(preview.read_bytes()).hexdigest()!=result['unreleased_private_preview_sha256']
            or hashlib.sha256(meta.read_bytes()).digest()!=hashlib.sha256(blob).digest()):
        raise ValueError('private recovery preview or manifest failed independent write verification')
    print(json.dumps({'status':'PRIVATE_PREVIEW_CREATED_NOT_AUTHORIZED_FOR_LIVE_OR_CITATION',
     'original_physical_pages':n,'formerly_skipped':len(omitted),
     'newly_machine_extracted_pages':result['provisional_alt_recovery_count'],
     'still_unreadable_pages':result['unrecovered_original_pages'],
     'previously_searchable_pages':result['prior_temporary_searchable_original_physical_pages'],
     'private_preview_now_candidate_searchable_pages':result['new_temporary_searchable_original_physical_pages'],
     'preview_sha256':hashlib.sha256(payload).hexdigest(),
     'private_manifest_sha256':hashlib.sha256(blob).hexdigest(),
     'private_output_root':str(out),'no_live_files_changed':True},sort_keys=True))

if __name__=='__main__':main()
