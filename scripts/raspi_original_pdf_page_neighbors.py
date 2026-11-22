#!/usr/bin/env python3
"""Optional read-only bounded adjacent original physical PDF page evidence.

Requires an exact completed original catalog path and explicit physical page.
Uses no network/cloud models or OCR. Does not modify source map, native FileDB,
originals, running workers or the default search behavior.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from file_system_catalog.verified_pdf_neighbor_context import verified_page_neighbors
from file_system_catalog.source_triage_provenance import verify_selected_source,require_unchanged_final_source_map
from file_system_catalog.source_review_audit import safe_file

ROOT=Path('/data/var/Knowledge Library')
DERIVED=Path('/data/var/knowmoredirt/knowledge-library/semantic/text')
SOURCE_MAP=Path('/data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3')
CATALOG=Path('/data/infra/etc/knowledge-library-sources.json')
LEDGER=ROOT/'_management/status.json'


def unique_rows(rows):
    table={}
    for row in rows:
        path=row.get('path') if isinstance(row,dict) else None
        if not isinstance(path,str) or path in table:
            raise ValueError('ambiguous original source identity in private catalog/ledger')
        table[path]=row
    return table


def original_pdf_context(original_path:str,page:int,*,neighbor_radius:int=1,
                         required_center_excerpt:str|None=None)->dict:
    if (not isinstance(original_path,str) or not 1<=len(original_path)<=1600
         or not original_path.lower().endswith('.pdf') or original_path.startswith('/')
         or '\\' in original_path or any(s in ('','.','..') for s in original_path.split('/'))):
        raise ValueError('invalid exact approved original PDF path')
    ledger=unique_rows(json.loads(LEDGER.read_bytes())['items'])
    catalog=unique_rows(json.loads(CATALOG.read_bytes())['sources'])
    receipt=ledger.get(original_path)
    if not receipt or receipt.get('status')!='complete' or original_path not in catalog:
        raise ValueError('requested publisher PDF lacks unique completed download approval')
    with sqlite3.connect(SOURCE_MAP.resolve().as_uri()+'?mode=ro',uri=True,timeout=20) as sm:
        sm.execute('PRAGMA query_only=ON')
        versions=sm.execute('SELECT source_sha256,derived_sha256,text_characters FROM originals WHERE source_path=? AND status=?',(original_path,'complete')).fetchall()
        if len(versions)!=1:raise ValueError('original lacks exactly one current extracted-source revision')
        source=verify_selected_source(original_path,source_versions={original_path:versions[0]},
            source_map=sm,approval_ledger={original_path:receipt},catalog={original_path:catalog[original_path]},
            original_root=ROOT,derived_root=DERIVED)
    publisher=safe_file(ROOT,original_path)
    context=verified_page_neighbors(publisher,source['original_sha256'],requested_page=page,
                      neighbor_radius=neighbor_radius,required_center_excerpt=required_center_excerpt)
    # A fresh source-map read after page extraction detects a concurrent
    # source-derived revision change that the first WAL read could not see.
    with sqlite3.connect(SOURCE_MAP.resolve().as_uri()+'?mode=ro',uri=True,timeout=20) as latest:
        latest.execute('PRAGMA query_only=ON')
        require_unchanged_final_source_map({original_path:source},newest_source_map=latest)
    return {'schema':'kmd.read_only_current_original_pdf_physical_neighbor_evidence.v1',
          'source':source,'physical_pages':context,
          'source_map_revision_reverified_after_physical_page_extraction':True,
          'source_locator_is_explicit_exact_catalog_path':True,
          'source_is_not_a_factual_answer':True}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--original-path',required=True)
    ap.add_argument('--page',type=int,required=True)
    ap.add_argument('--neighbor-radius',type=int,default=1,choices=(0,1))
    ap.add_argument('--required-center-excerpt',default=None,help='Exact short phrase present on requested original physical page; optional')
    args=ap.parse_args()
    print(json.dumps(original_pdf_context(args.original_path,args.page,neighbor_radius=args.neighbor_radius,
                        required_center_excerpt=args.required_center_excerpt),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
