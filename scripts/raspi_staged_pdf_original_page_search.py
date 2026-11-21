#!/usr/bin/env python3
"""Read-only lexical publisher-PDF page search within 327 privately staged repairs.

Builds an ephemeral in-memory SQLite FTS5 index; leaves original KnowMoreDiRT
FileDB, source-map, live text and summary workers completely unmodified.
Only independently original-page-verified passages are returned. Full
publisher text and quoted excerpts remain strictly local; do not publish.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from file_system_catalog.staged_physical_page_search import (
    StagedPhysicalPageRAMIndex,verify_staged_page_result,
)
from file_system_catalog.hybrid_search import query_terms

AUDITS=Path('/data/var/knowmoredirt/knowledge-library/search-audit')
SOURCES=Path('/data/var/Knowledge Library')
APPROVED=SOURCES/'_management/status.json'
SM=Path('/data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3')
NATIVE=Path('/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3')
DERIVED=Path('/data/var/knowmoredirt/knowledge-library/semantic/text')


def private_receipt(path:Path):
    if (path.is_symlink() or not path.is_file() or path.stat().st_size>8*1024*1024
            or not path.resolve().is_relative_to(AUDITS.resolve())):
        raise ValueError('private approved original PDF staged receipt unavailable')
    x=json.loads(path.read_bytes())
    if (x.get('schema')!='kmd.all_327_approved_original_pdf_private_stage_revision_and_physical_page_preflight.v1'
            or x.get('source_pdf_originals_verified_and_staged')!=327
            or x.get('all_original_old_derived_and_new_staged_full_sha_reverified_after_staging') is not True
            or x.get('status')!='ALL_PRIVATE_STAGE_CANDIDATES_VERIFIED_NOT_PUBLISHED_TO_LIVE_SEARCH'):
        raise ValueError('staged publisher original evidence receipt not in correct accepted state')
    items=x.get('original_sources_and_corrected_private_derivative_candidates')
    if not isinstance(items,list) or len(items)!=327:
        raise ValueError('staged publisher original corpus is incomplete')
    return items


def completed_acquisitions():
    if APPROVED.is_symlink() or APPROVED.stat().st_size>64*1024*1024:
        raise ValueError('approved private acquisition ledger invalid')
    completed={}
    for item in json.loads(APPROVED.read_bytes())['items']:
        if item.get('status')!='complete':continue
        path=item['path']
        if path in completed:raise ValueError('approved original acquisition path ambiguity')
        completed[path]=item['sha256']
    return completed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query',help='Research question or search terms; source text is not an answer')
    parser.add_argument('--stage-preflight',required=True,type=Path)
    parser.add_argument('--original-path',help='Optional exact original-relative source path, otherwise search all 327')
    parser.add_argument('--top',type=int,default=3,help='Maximum original PDF verified excerpts (1-8)')
    parser.add_argument('--candidates',type=int,default=30,help='Maximum ranked page candidates to verify (1-100)')
    args=parser.parse_args()
    if not 1<=args.top<=8 or not args.top<=args.candidates<=100:
        parser.error('invalid bounded original-physical-page verification count')
    items=private_receipt(args.stage_preflight)
    if args.original_path:
        if len(args.original_path)>1600 or args.original_path.startswith('/') or '..' in Path(args.original_path).parts:
            raise ValueError('invalid original source identity')
        items=[x for x in items if x['original_path']==args.original_path]
        if len(items)!=1:raise ValueError('original not present in independently approved repaired cohort')
    receipts=completed_acquisitions()
    evidence=[];failed=[]
    with StagedPhysicalPageRAMIndex(max_records=len(items)) as index:
        for item in items:index.add(item,stage_root=AUDITS)
        index.commit()
        with sqlite3.connect(SM.resolve().as_uri()+'?mode=ro',uri=True,timeout=25) as sm,\
                sqlite3.connect(NATIVE.resolve().as_uri()+'?mode=ro',uri=True,timeout=25) as native:
            sm.execute('PRAGMA query_only=ON');native.execute('PRAGMA query_only=ON')
            for hit in index.ranked(args.query,limit=args.candidates):
                check=verify_staged_page_result(hit,query=args.query,original_root=SOURCES,
                        derived_root=DERIVED,stage_root=AUDITS,completed_receipts=receipts,
                        source_map=sm,native_db=native)
                if check['status']=='publisher_original_physical_pdf_page_verified':
                    evidence.append(check)
                    if len(evidence)>=args.top:break
                else:failed.append(check)
        result={'schema':'kmd.temporary_private_physical_pdf_page_verified_evidence_search.v1',
                'query':args.query,'scope':args.original_path or 'all_327_approved_private_repair_originals',
                'temporary_indexed_originals':len(index.loaded_originals),
                'temporary_indexed_physical_page_units':len(index.rows),
                'private_staged_text_bytes_indexed':index.bytes_indexed,
                'verified_original_pdf_physical_page_passages':evidence,
                'unverified_ranked_candidates_excluded':len(failed),
                'rejected_for_weak_excerpt_query_coverage':sum(z['status']=='insufficient_source_query_term_coverage' for z in failed),
                'lexical_minimum_distinct_query_terms_per_verified_excerpt':min(2,len(query_terms(args.query))),
                'source_original_semantic_answer_not_generated':True,
                'not_a_second_persistent_database':True,
                'no_original_source_filedb_source_map_summary_cache_or_worker_writes':True}
        print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
