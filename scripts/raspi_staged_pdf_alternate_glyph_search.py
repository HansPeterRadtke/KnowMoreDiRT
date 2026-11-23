#!/usr/bin/env python3
"""Read-only provisional original-PDF page text search for NUL-glyph damage.

This explicit opt-in path works only with exactly one approved, staged,
revision-attested original PDF and its previously *omitted* temporary page
units. It neither changes the native FileDB nor silently promotes an
independent Ghostscript machine extraction into an authoritative citation.
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
from file_system_catalog.alternate_pdf_page_text import extract_alternate_original_pdf_pages
from file_system_catalog.alternate_pdf_page_search import (
 exact_skipped_staged_physical_pages,provisional_alternate_query_hits,
 require_current_approved_staged_pdf_source,
)
from raspi_staged_pdf_original_page_search import (
 private_receipt, completed_acquisitions, AUDITS,SOURCES,SM,NATIVE,DERIVED,
)


def current_revision(item,receipts):
    # Always create fresh read-only connections rather than inadvertently
    # retaining a long-lived SQLite WAL snapshot before/after extraction.
    with sqlite3.connect(SM.resolve().as_uri()+'?mode=ro',uri=True,timeout=25) as sm,\
         sqlite3.connect(NATIVE.resolve().as_uri()+'?mode=ro',uri=True,timeout=25) as native:
        sm.execute('PRAGMA query_only=ON');native.execute('PRAGMA query_only=ON')
        return require_current_approved_staged_pdf_source(item,
            original_root=SOURCES,derived_root=DERIVED,stage_root=AUDITS,
            source_map=sm,native=native,completed_acquisitions=receipts)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('query',help='Terms to find within damaged original-PDF physical pages')
    ap.add_argument('--stage-preflight',required=True,type=Path,
          help='Previously approved private full cohort stage receipt, not a raw PDF path')
    ap.add_argument('--original-path',required=True,
          help='Exact original relative path in completed approved publisher catalog')
    ap.add_argument('--page',type=int,default=None,
          help='Optional one physical PDF page omitted by normal staged search')
    ap.add_argument('--top',type=int,default=3)
    ap.add_argument('--no-secondary-text-extractor-check',action='store_true',
          help='Skip Poppler phrase corroboration; ALL snippets remain expressly provisional')
    args=ap.parse_args()
    if (not isinstance(args.original_path,str) or not 1<=len(args.original_path)<=1600
            or args.original_path.startswith('/') or '\\' in args.original_path
            or any(p in ('','.','..') for p in args.original_path.split('/'))
            or not args.original_path.endswith('.pdf')):
        raise ValueError('only an exact approved original relative PDF path is allowed')
    if type(args.top) is not int or not 1<=args.top<=8:raise ValueError('invalid bounded top count')
    # The private receipt validates the entire 327-source cohort but only the
    # named original's byte and native revisions are processed by this path.
    initial_sha=hashlib.sha256(args.stage_preflight.read_bytes()).hexdigest()
    cohort=private_receipt(args.stage_preflight)
    items=[row for row in cohort if row['original_path']==args.original_path]
    if len(items)!=1:raise ValueError('original missing from approved private 327-PDF repair cohort')
    item=items[0]
    receipts=completed_acquisitions()
    pinned=current_revision(item,receipts)
    staged_text=pinned['new_staged_source'].read_text('utf8','strict')
    physical_count=item['original_pdf_physical_pages']
    omitted=exact_skipped_staged_physical_pages(staged_text,physical_count,
                                        item['new_max_staged_page_label'])
    if args.page is not None:
        if type(args.page) is not int or args.page not in omitted:
            raise ValueError('selected exact physical PDF page was not omitted by the normal staged search')
        chosen=[args.page]
    else:
        if len(omitted)>32:
            raise ValueError('more than 32 damaged pages: choose an explicit physical --page')
        chosen=omitted
    if not chosen:
        raise ValueError('approved original has no skipped glyph-damaged temporary source pages')
    alternate=extract_alternate_original_pdf_pages(pinned['original_pdf'],
           pinned['original_sha256'],pages=chosen,expected_physical_page_count=physical_count)
    search=provisional_alternate_query_hits(args.query,alternate['pages'],top=args.top,
           original_pdf=pinned['original_pdf'],
           expected_original_sha256=pinned['original_sha256'],
           corroborate_with_poppler=not args.no_secondary_text_extractor_check)
    # Enforce a distinct post-extraction SQLite snapshot and complete SHA
    # recheck, including the private stage and the original approval ledger.
    final=current_revision(item,completed_acquisitions())
    if final!=pinned:
        raise ValueError('original or current source/native identity changed during alternate search')
    if hashlib.sha256(args.stage_preflight.read_bytes()).hexdigest()!=initial_sha:
        raise ValueError('private approved stage receipt changed during alternate search')
    source={'schema':'kmd.optional_readonly_original_pdf_omitted_glyph_pages_candidate_search.v1',
      'query':args.query,'exact_approved_original_path':args.original_path,
      'approved_original_pdf_sha256':pinned['original_sha256'],
      'native_original_filedb_id':pinned['native_id'],
      'original_pdf_physical_page_count':physical_count,
      'normal_temporary_publisher_page_search_omitted_physical_pages':len(omitted),
      'selected_omitted_physical_pages':chosen,
      'provisional_ghostscript_pages_with_text':sum(p['status']=='source_bounded_provisional_machine_text' for p in alternate['pages']),
      'provisional_machine_text_page_diagnostics':[
        {k:v for k,v in p.items() if k not in ('text',)} for p in alternate['pages']],
      'ranked_temporary_source_page_candidates':search,
      'original_and_current_old_live_and_private_staged_sha_and_native_id_checked_before_and_after':True,
      'source_original_PDF_and_private_source_files_never_modified':True,
      'alternate_machine_text_page_numbers_are_original_PDF_pages_but_words_need_visual_review':True,
      'no_download_no_OCR_no_vector_rebuild_no_native_index_or_summary_mutation':True,
      'original_page_text_not_a_factual_answer':True}
    print(json.dumps(source,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
