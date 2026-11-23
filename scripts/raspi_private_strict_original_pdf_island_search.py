#!/usr/bin/env python3
"""Opt-in source- and page-bound search of otherwise rejected original PDF pages.

The default 411-page private preview continues to reject entire NUL-corrupt
pages. This separate read-only fallback uses ONLY clean contiguous glyph runs
from those exact 12 strictly omitted physical publisher pages, then verifies
any proposed short quote against independent Poppler extraction of the same
SHA-verified publisher ORIGINAL physical page. No OCR, model, network, write
to source-map/native FileDB, source cache, summaries, or live worker.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import sqlite3
import stat
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from file_system_catalog.private_strict_pdf_island_search import OriginalGlyphIslandRAMIndex
from file_system_catalog.alternate_pdf_page_text import original_pdf_gs_page,quality_of_alternate_page
from file_system_catalog.alternate_pdf_page_search import (
 require_current_approved_staged_pdf_source,corroborate_specific_alternate_source_fragments,
)
from file_system_catalog.publisher_pdf_page import _poppler,verify_pdf_page_excerpts
from file_system_catalog.source_review_audit import sha_file
from raspi_staged_pdf_original_page_search import private_receipt,completed_acquisitions,AUDITS,SOURCES,SM,NATIVE,DERIVED


def bounded_rollup(path:Path):
 if (not isinstance(path,Path) or not path.is_absolute() or path.is_symlink()
         or not path.is_file() or path.stat().st_size>512*1024
         or not path.resolve().is_relative_to(AUDITS.resolve())):
  raise ValueError('untrusted or oversized owner-private original PDF glyph review rollup')
 return json.loads(path.read_bytes())


def pinned_original(item,ledger):
 with sqlite3.connect(SM.resolve().as_uri()+'?mode=ro',uri=True,timeout=40) as sm,\
      sqlite3.connect(NATIVE.resolve().as_uri()+'?mode=ro',uri=True,timeout=40) as db:
  sm.execute('PRAGMA query_only=ON');db.execute('PRAGMA query_only=ON')
  return require_current_approved_staged_pdf_source(item,
     original_root=SOURCES,derived_root=DERIVED,stage_root=AUDITS,
     source_map=sm,native=db,completed_acquisitions=ledger)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('query',help='Short source-text search question, not a model prompt to answer')
 p.add_argument('--stage-preflight',required=True,type=Path)
 p.add_argument('--private-170-rollup',required=True,type=Path)
 p.add_argument('--top',type=int,default=3)
 p.add_argument('--candidates',type=int,default=35)
 p.add_argument('--original-path',default=None,help='Optional exact approved relative original PDF path')
 args=p.parse_args()
 if (type(args.top) is not int or type(args.candidates) is not int
         or not 1<=args.top<=args.candidates<=60):
  raise ValueError('invalid bounded source page candidate count')
 if args.original_path is not None and (not isinstance(args.original_path,str)
          or not 1<=len(args.original_path)<=1600 or not args.original_path.endswith('.pdf')
          or args.original_path.startswith('/') or '\\' in args.original_path
          or any(x in ('','.','..') for x in args.original_path.split('/'))):
  raise ValueError('invalid exact approved original publisher source path')
 preflight_sha=sha_file(args.stage_preflight)
 approved=private_receipt(args.stage_preflight)
 by_path={x['original_path']:x for x in approved}
 if len(by_path)!=327 or len(approved)!=327:
  raise ValueError('malformed prior publisher original source cohort')
 roll=bounded_rollup(args.private_170_rollup)
 roll_sha=sha_file(args.private_170_rollup)
 if (roll.get('schema')!='kmd.170_approved_original_pdf_unapproved_machine_glyph_preview_batch.v1'
        or roll.get('status')!='ALL_PREVIEWS_ARE_PRIVATE_UNAPPROVED_AND_NOT_ELIGIBLE_FOR_NATIVE_RELEASE'
        or roll.get('approved_327_original_stage_preflight_sha256')!=preflight_sha
        or roll.get('counts',{}).get('verified_private_previews')!=170
        or roll.get('counts',{}).get('remaining_unreadable_original_pages')!=12
        or roll.get('counts',{}).get('pages_machine_recovered_provisionally')!=411
        or roll.get('unresolved_source_specific_private_staging_failures')
        or not roll.get('no_original_live_derived_source_map_native_vector_summary_or_running_worker_changes')):
  raise ValueError('unapproved old-source page cohort proof changed or incomplete')
 entries=roll['complete_originals']
 if len(entries)!=170 or len({x['original_path'] for x in entries})!=170:
  raise ValueError('duplicate or omitted private machine PDF source identity')
 limited=[]
 for report in entries:
  rel=report['original_path']
  if rel not in by_path or report['original_pdf_sha256']!=by_path[rel]['original_sha256']:
   raise ValueError('unapproved revision is not the originally attested publisher PDF')
  for page in report['still_unreadable_original_physical_pages']:
   limited.append((report,page))
 if len(limited)!=12 or len({(a['original_path'],p) for a,p in limited})!=12:
  raise ValueError('malformed strict original PDF unreadable physical-page inventory')
 if args.original_path is not None and args.original_path not in {a['original_path'] for a,p in limited}:
  raise ValueError('requested original has no strict unavailable glyph page')
 ledger=completed_acquisitions()
 scoped=limited if args.original_path is None else [(a,p) for a,p in limited if a['original_path']==args.original_path]
 with OriginalGlyphIslandRAMIndex() as index:
  for report,page in scoped:
   path=report['original_path'];item=by_path[path]
   original=pinned_original(item,ledger)
   if (original['original_sha256']!=report['original_pdf_sha256'] or
      original['staged_derived_sha256']!=report['approved_corrected_stage_sha256']):
    raise ValueError('old or corrected source revision changed')
   # Attest the private staged preview source and its exact *unrecovered* pages.
   directory=Path(report['private_preview_dir'])
   if (not directory.is_dir() or directory.is_symlink()
         or not directory.resolve().is_relative_to(AUDITS.resolve())):
    raise ValueError('untrusted owner-private rejected PDF page inventory directory')
   manifest=directory/'provisional-preview-private-manifest.json'
   if (manifest.is_symlink() or not manifest.is_file()
         or sha_file(manifest)!=report['private_manifest_sha256']):
    raise ValueError('unapproved private PDF document/page receipt changed')
   m=json.loads(manifest.read_bytes())
   if (m.get('publisher_original_sha256')!=original['original_sha256']
         or m.get('original_source_path')!=path
         or m.get('native_original_filedb_id')!=original['native_id']
         or m.get('prior_private_staged_sha256')!=original['staged_derived_sha256']
         or m.get('live_old_derived_sha256')!=original['old_derived_sha256']
         or m.get('unrecovered_original_pages')!=report['still_unreadable_original_physical_pages']
         or not m.get('no_native_source_map_filedb_vector_or_summary_cutover_allowed')):
    raise ValueError('private original physical PDF unreadable-page proof is stale')
   preview=directory/m['private_preview_file']
   if (not preview.is_file() or preview.is_symlink()
         or sha_file(preview)!=report['private_preview_sha256']):
    raise ValueError('protected unreleased publisher-source preview changed')
   # The same publisher original descriptor proves PDFinfo page numbers, GS
   # machine glyph bytes, and exact original SHA before + after; no new PDF.
   with os.fdopen(os.open(original['original_pdf'],os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW),'rb') as pub:
    fd=pub.fileno();before=os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or hashlib.file_digest(pub,'sha256').hexdigest()!=original['original_sha256']:
     raise ValueError('publisher original source file SHA changed before island scan')
    limit=time.monotonic()+40
    info=_poppler(['/usr/bin/pdfinfo',f'/proc/self/fd/{fd}'],fd,limit)
    pages=re.findall(rb'(?m)^Pages:[ \t]+([1-9][0-9]{0,5})[ \t]*$',info or b'')
    if len(pages)!=1 or int(pages[0])!=item['original_pdf_physical_pages'] or not 1<=page<=int(pages[0]):
     raise ValueError('original PDF actual physical page boundaries changed')
    gs=original_pdf_gs_page(fd,page,limit)
    if quality_of_alternate_page(gs)['status']!='unavailable':
     raise ValueError('previously strict unavailable page now has a fully readable candidate')
    index.add_pinned_page(path=path,original_sha256=original['original_sha256'],physical_page=page,
     raw_machine_text=gs,original_page_count=item['original_pdf_physical_pages'],
     native_original_file_id=original['native_id'],current_old_derived_sha256=original['old_derived_sha256'],
     corrected_staged_derived_sha256=original['staged_derived_sha256'])
    after=original['original_pdf'].stat(follow_symlinks=False)
    if (not os.path.samestat(before,after) or before.st_size!=after.st_size
       or before.st_mtime_ns!=after.st_mtime_ns or before.st_ctime_ns!=after.st_ctime_ns):
     raise ValueError('original publisher PDF changed during glyph page scan')
    pub.seek(0)
    if hashlib.file_digest(pub,'sha256').hexdigest()!=original['original_sha256']:
     raise ValueError('original publisher PDF bytes changed during glyph page scan')
   if pinned_original(item,ledger)!=original or sha_file(preview)!=report['private_preview_sha256']:
    raise ValueError('original publisher PDF/native identity changed after glyph scan')
  index.commit()
  hits=index.ranked(args.query,limit=args.candidates,original_path=args.original_path)
  # One corrupt publisher page can contain multiple physically disconnected
  # clean islands. An unverified first-ranked island is NOT evidence that the
  # next island lacks an original-page corroborated passage. Check bounded
  # distinct islands before returning any page as source-only navigation.
  chosen={};attempts={}
  for hit in hits:
   ident=(hit['path'],hit['physical_pdf_page'])
   if ident in chosen and chosen[ident]['same_original_PDF_physical_page_two_extractor_exact_excerpt'] is not None:
    continue
   if attempts.get(ident,0)>=5:continue
   attempts[ident]=attempts.get(ident,0)+1
   source=pinned_original(by_path[hit['path']],ledger)
   if (hit['original_sha256']!=source['original_sha256']
        or hit['native_original_file_id']!=source['native_id']
        or hit['corrected_staged_derived_sha256']!=source['staged_derived_sha256']
        or hit['current_old_derived_sha256']!=source['old_derived_sha256']):
    raise ValueError('original current source identity changed before independent quote proof')
   text=index.private_fragment_for_second_extractor(row_id=hit['row_id'],path=hit['path'],
      original_sha256=hit['original_sha256'],physical_page=hit['physical_pdf_page'],body_sha256=hit['body_sha256'])
   proof=corroborate_specific_alternate_source_fragments(physical_page=hit['physical_pdf_page'],
       provisional_text=text,query=args.query,original_pdf=source['original_pdf'],
       original_sha256=source['original_sha256'],independent_verifier=verify_pdf_page_excerpts)
   if pinned_original(by_path[hit['path']],ledger)!=source:
    raise ValueError('publisher source-native revision changed after independent same-page phrase proof')
   frag=proof['exact_same_page_corroborated_fragment']
   selected={'original_source_path':hit['path'],'original_pdf_sha256':source['original_sha256'],
     'original_physical_PDF_page':hit['physical_pdf_page'],'query_terms_located':hit['query_terms_in_machine_only_locator'],
     'same_original_PDF_physical_page_two_extractor_exact_excerpt':frag,
     'status':('exact_short_original_page_phrase_corrob_only' if frag else 'private_page_locator_without_verifiable_quotation'),
     'full_original_page_text_figure_layout_and_factual_accuracy_not_proven':True,
     'source_map_and_native_revision_rechecked_after_proof':True,
     'unverified_machine_only_text_never_returned':True}
   if ident not in chosen or frag is not None:chosen[ident]=selected
  # Verified same-page publisher excerpts precede source-only locators.
  # Never treat an unverified first island as a reason to discard all other
  # bounded search hits for the same publisher original physical page.
  results=sorted(chosen.values(),key=lambda x:x['same_original_PDF_physical_page_two_extractor_exact_excerpt'] is None)[:args.top]
  if sha_file(args.stage_preflight)!=preflight_sha or sha_file(args.private_170_rollup)!=roll_sha:
   raise ValueError('original private glyph proof receipt changed while source search was running')
  print(json.dumps({'schema':'kmd.12_strict_rejected_original_pdf_physical_page_clean_island_search.v1',
    'query':args.query,'scope':args.original_path or 'all_12_strict_unavailable_original_pdf_pages',
    'total_unrecovered_pages_in_sealed_original_cohort':12,
    'scoped_original_pdf_pages_checked_with_source_sha':len(index.keys),
    'private_clean_island_text_spans_ranked_in_RAM':len(index.entries),
    'results':results,'no_unverified_machine_quotes_emitted':True,
    'no_source_map_filedb_source_derivative_native_vector_summary_or_worker_mutation':True,
    'no_ocr_download_model_or_network':True,'this_is_navigational_source_evidence_not_a_factual_answer':True},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
