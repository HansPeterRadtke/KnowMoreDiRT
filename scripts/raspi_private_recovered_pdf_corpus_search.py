#!/usr/bin/env python3
"""Opt-in ORIGINAL-PDF-bound search of strictly private recovered page fragments.

Searches only pages previously omitted by the existing correct-page
source preview, and supplements rather than replaces the canonical native
FileDB and source index. Every result is either an independently exact
same-physical-original-PDF corroborated *fragment*, or merely a candidate
source/page locator without an asserted quotation. No owner book text is
published, no external model/network/OCR, no live corpus writes permitted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from file_system_catalog.private_recovered_pdf_corpus_search import PrivateRecoveredOriginalPageRAMIndex
from file_system_catalog.alternate_pdf_page_search import (
 require_current_approved_staged_pdf_source,
 corroborate_specific_alternate_source_fragments,
)
from file_system_catalog.source_review_audit import safe_file,sha_file
from file_system_catalog.publisher_pdf_page import verify_pdf_page_excerpts
from raspi_staged_pdf_original_page_search import private_receipt,completed_acquisitions,AUDITS,SOURCES,SM,NATIVE,DERIVED


def safe_private(path:Path,max_bytes:int):
 if (not path.is_absolute() or path.is_symlink() or not path.is_file()
        or path.stat().st_size>max_bytes or not path.resolve().is_relative_to(AUDITS.resolve())):
    raise ValueError('private original PDF source cohort receipt missing, untrusted or oversized')
 return json.loads(path.read_bytes())


def current_original_revision(item,receipts):
 with sqlite3.connect(SM.resolve().as_uri()+'?mode=ro',uri=True,timeout=30) as sm,\
      sqlite3.connect(NATIVE.resolve().as_uri()+'?mode=ro',uri=True,timeout=30) as db:
  sm.execute('PRAGMA query_only=ON');db.execute('PRAGMA query_only=ON')
  return require_current_approved_staged_pdf_source(item,original_root=SOURCES,
     derived_root=DERIVED,stage_root=AUDITS,source_map=sm,native=db,
     completed_acquisitions=receipts)


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('query')
 ap.add_argument('--stage-preflight',required=True,type=Path)
 ap.add_argument('--private-170-rollup',required=True,type=Path)
 ap.add_argument('--top',type=int,default=3)
 ap.add_argument('--candidates',type=int,default=30)
 ap.add_argument('--original-path',default=None,help='Optional exact source in the private recovered corpus')
 args=ap.parse_args()
 if (type(args.top) is not int or type(args.candidates) is not int
         or not 1<=args.top<=args.candidates<=60):
  raise ValueError('invalid bounded returned/candidate source page count')
 if args.original_path is not None and (not isinstance(args.original_path,str) or
        args.original_path.startswith('/') or '\\' in args.original_path or
        any(p in ('','.','..') for p in args.original_path.split('/'))):
  raise ValueError('source must be exact original publisher relative PDF path')
 # Source list and reconstructed preview cohort are both independently
 # attested owner-private; this opt-in reader never accepts arbitrary paths.
 preflight_bytes=args.stage_preflight.read_bytes();preflight_sha=hashlib.sha256(preflight_bytes).hexdigest()
 approved=private_receipt(args.stage_preflight)
 by_original={x['original_path']:x for x in approved}
 assert len(by_original)==len(approved)==327
 rollup=safe_private(args.private_170_rollup,512*1024)
 if (rollup.get('schema')!='kmd.170_approved_original_pdf_unapproved_machine_glyph_preview_batch.v1'
       or rollup.get('status')!='ALL_PREVIEWS_ARE_PRIVATE_UNAPPROVED_AND_NOT_ELIGIBLE_FOR_NATIVE_RELEASE'
       or rollup.get('counts',{}).get('verified_private_previews')!=170
       or rollup.get('approved_327_original_stage_preflight_sha256')!=preflight_sha
       or rollup.get('counts',{}).get('pages_machine_recovered_provisionally')!=411
       or rollup.get('unresolved_source_specific_private_staging_failures')
       or not rollup.get('no_original_live_derived_source_map_native_vector_summary_or_running_worker_changes')):
  raise ValueError('private 170 approved review-only original PDF cohort is incomplete or untrusted')
 roll_sha=sha_file(args.private_170_rollup)
 entries=rollup['complete_originals']
 if len(entries)!=170 or len({x['original_path'] for x in entries})!=170:
  raise ValueError('ambiguous original publisher identity in machine-recovery review cohort')
 if args.original_path and args.original_path not in {x['original_path'] for x in entries}:
  raise ValueError('exact requested source not in published private recovery preview')
 receipts=completed_acquisitions()
 with PrivateRecoveredOriginalPageRAMIndex(max_originals=170) as index:
  for report in entries:
   path=report['original_path']
   it=by_original.get(path)
   if it is None:raise ValueError('private machine preview is not tied to an approved source PDF stage')
   src=current_original_revision(it,receipts)
   root=Path(report['private_preview_dir'])
   if (not root.is_dir() or root.is_symlink()
           or not root.resolve().is_relative_to(AUDITS.resolve())):
    raise ValueError('private original PDF preview path moved or is not a restricted directory')
   manifest=root/'provisional-preview-private-manifest.json'
   if manifest.is_symlink() or not manifest.is_file():
    raise ValueError('private original recovery manifest is not an ordinary file')
   if (sha_file(manifest)!=report['private_manifest_sha256']):
    raise ValueError('private recovered original PDF page manifest changed')
   m=json.loads(manifest.read_bytes())
   if (m.get('status')!='PRIVATE_UNAPPROVED_MACHINE_PDF_TEXT_PREVIEW_NOT_FOR_LIVE_CUTOVER'
          or m.get('original_source_path')!=path
          or m.get('publisher_original_sha256')!=src['original_sha256']
          or m.get('prior_private_staged_sha256')!=src['staged_derived_sha256']
          or m.get('live_old_derived_sha256')!=src['old_derived_sha256']
          or m.get('native_original_filedb_id')!=src['native_id']
          or m.get('unreleased_private_preview_sha256')!=report['private_preview_sha256']
          or m.get('original_physical_page_count')!=it['original_pdf_physical_pages']
          or not m.get('unchanged_original_physical_page_bodies_exactly_preserved')
          or not m.get('no_native_source_map_filedb_vector_or_summary_cutover_allowed')):
    raise ValueError('source identity or explicit private non-release restrictions changed')
   preview=root/m['private_preview_file']
   if (not preview.is_file() or preview.is_symlink() or preview.parent!=root
          or sha_file(preview)!=report['private_preview_sha256']):
    raise ValueError('original PDF private recovered page text bytes changed')
   index.add_approved_source_preview(original_path=path,
       original_sha256=src['original_sha256'],
       private_preview_text=preview.read_bytes().decode('utf8','strict'),
       physical_page_count=it['original_pdf_physical_pages'],
       source_specific_page_change_receipts=m['replaced_page_proof_without_book_text'],
       current_old_live_derived_sha256=src['old_derived_sha256'],
       approved_first_corrected_page_stage_sha256=src['staged_derived_sha256'],
       unapproved_preview_sha256=report['private_preview_sha256'],
       native_original_file_id=src['native_id'])
  index.commit()
  hits=index.ranked(args.query,limit=args.candidates,original_path=args.original_path)
  valid=[];excluded=0
  for hit in hits:
   it=by_original[hit['original_path']]
   post=current_original_revision(it,completed_acquisitions())
   r=next(v for v in entries if v['original_path']==hit['original_path'])
   dest=Path(r['private_preview_dir']);man=dest/'provisional-preview-private-manifest.json'
   manifest=json.loads(man.read_bytes())
   newfile=dest/manifest['private_preview_file']
   if man.is_symlink() or not man.is_file() or newfile.is_symlink():
     raise ValueError('source selection private proof changed into symlink')
   if (hit['original_sha256']!=post['original_sha256']
      or hit['current_old_live_derived_sha256']!=post['old_derived_sha256']
      or hit['approved_first_corrected_page_stage_sha256']!=post['staged_derived_sha256']
      or hit['native_original_file_id']!=post['native_id']
      or hit['unapproved_preview_sha256']!=sha_file(newfile)
      or sha_file(man)!=r['private_manifest_sha256']):
     raise ValueError('current original, native source identity or private recovery preview changed before original page proof')
   complete_unapproved_physical_page=index.provisional_page_for_independent_quotation_check(
     original_path=hit['original_path'],original_sha256=hit['original_sha256'],
     physical_pdf_page=hit['physical_pdf_page'],source_body_sha256=hit['source_body_sha256'])
   fragments=corroborate_specific_alternate_source_fragments(
     physical_page=hit['physical_pdf_page'],
     provisional_text=complete_unapproved_physical_page,query=args.query,
     independent_verifier=verify_pdf_page_excerpts,
     original_pdf=post['original_pdf'],original_sha256=post['original_sha256'])
   latest=current_original_revision(it,completed_acquisitions())
   if latest!=post or sha_file(newfile)!=hit['unapproved_preview_sha256'] or sha_file(man)!=r['private_manifest_sha256']:
     raise ValueError('original source or review preview changed during original physical-page fragment proof')
   fragment=fragments['exact_same_page_corroborated_fragment']
   # NEVER emit the uncorroborated machine preview excerpt on this default
   # whole-cohort search path. Only the exact source-page-corroborated phrase
   # may be quoted, while an unverified source/path still acts as navigation.
   valid.append({'original_source_path':hit['original_path'],
     'publisher_original_sha256':post['original_sha256'],
     'physical_original_pdf_page':hit['physical_pdf_page'],
     'covered_distinct_query_terms':hit['covered_distinct_query_terms'],
     'exact_source_page_independently_two_extractor_corroborated_fragment':fragment,
     'status':('publisher_original_two_machine_exact_phrase_verified_only'
               if fragment is not None else 'provisional_original_source_page_locator_no_quote'),
     'machine_only_excerpt_excluded_by_default':True,
     'native_source_map_and_complete_source_and_private_revision_rechecked_after_original_PDF_proof':True,
     'whole_pdf_page_visual_text_accuracy_proven':False,
     'real_world_factual_answer_verified':False})
   if len(valid)>=args.top:break
  if (hashlib.sha256(args.stage_preflight.read_bytes()).hexdigest()!=preflight_sha
        or sha_file(args.private_170_rollup)!=roll_sha):
    raise ValueError('private original PDF source release corpus changed during read-only query')
  print(json.dumps({'schema':'kmd.private_170_provisional_recovery_original_pdf_global_search.v1',
       'query':args.query,'scope':args.original_path or 'all_170_private_unapproved_originals',
       'current_native_originals_proven_and_temporarily_indexed':len(index.seen),
       'private_machine_only_recovered_original_pdf_pages_ranked_in_RAM':len(index.entries),
       'verified_source_and_private_revision_binding_required':True,
       'original_source_page_results':valid,
       'no_unverified_machine_quotes_emitted':True,
       'no_corpus_native_index_source_map_vectors_or_summaries_changed':True,
       'no_external_download_or_model':True,
       'results_are_research_evidence_navigation_not_factual_answers':True},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
