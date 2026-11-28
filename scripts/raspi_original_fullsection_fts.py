#!/usr/bin/env python3
"""Opt-in read-only original PDF *complete source unit* FTS5 by literal folder.

Unlike first-unit lookup this scans all CURRENT, SHA-checked extracted text
sections in one bounded source folder into a disposable RAM index.
Only selected originals receive a full approved publisher original SHA,
derived SHA, source-map revision, unique native FileDB identity and final
independent second readback. No model, download or persistent index writes.

Derived PDF section labels are not proof of actual publisher physical pages.
A verified source passage is a candidate reference, NOT answer correctness.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from file_system_catalog.original_fullsection_fts import (
 OriginalSectionRAMIndex,add_current_scope,verify_current_source_window,
 validate_literal_source_directory_prefix,MAX_FULLSOURCE_WINDOWS,
)
from file_system_catalog.publisher_firstunit_discovery import (
 compiled_required_original_source_phrase,
)
from raspi_publisher_firstunit_discovery import approved_original_receipts,input_queries
from file_system_catalog.pdf_visual_link import safe_private_result_path

SOURCE_MAP=Path('/data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3')
NATIVE_DB=Path('/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3')
SOURCE_ROOT=Path('/data/var/Knowledge Library')
DERIVED_ROOT=Path('/data/var/knowmoredirt/knowledge-library/semantic/text')
RECEIPTS=SOURCE_ROOT/'_management/status.json'


def readonly(path:Path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('current original source-map/native database missing or symlinked')
    db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=45)
    db.execute('PRAGMA query_only=ON')
    return db


def search(*,prefix:str,questions:list[dict],top:int,source_map:Path,
           native_db:Path,original_root:Path,derived_root:Path,
           completed_ledger:Path)->dict:
    prefix=validate_literal_source_directory_prefix(prefix)
    if (prefix is None or type(top) is not int or not 1<=top<=10
            or not isinstance(questions,list) or not 1<=len(questions)<=8):
        raise ValueError('must specify a bounded exact original source PDF subfolder and query batch')
    for q in questions:
        if not isinstance(q,dict) or not isinstance(q.get('query'),str):
            raise ValueError('invalid current original PDF query')
    receipts=approved_original_receipts(completed_ledger)
    start=time.monotonic()
    collected=[]
    proof_rows=[]
    selected_docs={}
    # Build this bounded temporary index exactly ONCE, then query the
    # original source sections with the same current read-only source-map.
    with closing(readonly(source_map)) as sm, OriginalSectionRAMIndex() as index:
        coverage=add_current_scope(source_map=sm,derived_root=derived_root,
                                   prefix=prefix,index=index)
        index_sec=round(time.monotonic()-start,3)
        with closing(readonly(native_db)) as native:
            for q in questions:
                phrase=q.get('require_source_phrase')
                pattern=(compiled_required_original_source_phrase(phrase)
                         if phrase is not None else None)
                channels={}
                for channel in ('literal','porter'):
                    hits=[];rejects=[]
                    visited=set()
                    for entry,rowid,rank in index.ranked(q['query'],channel=channel,limit=500):
                        if len(hits)>=top:break
                        if entry.original_path in visited:continue
                        if pattern is not None and not pattern.search(entry.original_body):
                            continue
                        visited.add(entry.original_path)
                        try:
                            proof=verify_current_source_window(
                                window=entry,query=q['query'],phrase=phrase,
                                original_root=original_root,derived_root=derived_root,
                                source_map=sm,native_db=native,
                                completed_receipts=receipts)
                        except (ValueError,OSError,UnicodeError) as error:
                            rejects.append({'original_path':entry.original_path,
                                            'reason':type(error).__name__+': '+str(error)[:160]})
                            continue
                        proof_rows.append((entry,q['query'],phrase,proof))
                        selected_docs[entry.original_path]=True
                        hits.append({'source_rank':len(hits)+1,
                                     'rank_channel':'complete_original_pdf_section_'+channel,
                                     'fts_bm25_not_a_probability':round(rank,6),
                                     'source_map_title_metadata_not_citation':entry.title,
                                     **proof})
                    channels[channel]={
                        'source_originals':hits,'returned_current_verified_originals':len(hits),
                        'stale_or_unverified_candidates_omitted':rejects,
                    }
                collected.append({
                    'id':q['id'],'query':q['query'],
                    'query_sha256':hashlib.sha256(q['query'].encode()).hexdigest(),
                    **({'required_phrase_only_in_source_text':phrase}
                       if phrase is not None else {}),
                    'independent_rank_channels':channels,
                    'rank_fusion_applied':False,
                })
    # Fresh independent read-only DB handles and publisher SHA readback AFTER
    # all searches. Existing original native writer commits may still occur
    # after this point; no unsupported atomic generation promise is made.
    with closing(readonly(source_map)) as sm,closing(readonly(native_db)) as native:
        for window,query,phrase,first in proof_rows:
            later=verify_current_source_window(
                window=window,query=query,phrase=phrase,original_root=original_root,
                derived_root=derived_root,source_map=sm,native_db=native,
                completed_receipts=approved_original_receipts(completed_ledger))
            if first!=later:
                raise ValueError('source original revision changed after later source search; abort output')
    return {
        'schema':'kmd.original_complete_pdf_section_source_scoped_lexical_v1',
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'literal_original_source_directory_prefix':prefix,
        'source_scan':coverage,
        'questions':collected,'query_count':len(collected),
        'unique_returned_originals':len(selected_docs),
        'initial_scan_seconds':index_sec,
        'total_seconds':round(time.monotonic()-start,3),
        'complete_approved_original_and_derived_sha_rechecked_after_all_queries':True,
        'current_original_native_filedb_identity_rechecked_after_all_queries':True,
        'every_returned_pdf_excerpt_has_current_derived_offsets_and_sha':True,
        'extracted_page_unit_labels_not_original_physical_page_citations':True,
        'source_publisher_actual_PDF_physical_pages_unverified':True,
        'source_excerpt_not_literal_original_PDF_bytes':True,
        'factual_answers_not_assessed':True,
        'all_index_input_derivative_sha_verified_but_not_all_original_pdfs_fully_hashed':True,
        'no_immutable_cross_store_generation_pinned':True,
        'no_new_source_acquisition_or_source_modification':True,
        'no_native_database_or_index_mutation':True,
        'no_second_persistent_index_or_model':True,
        'ranking_channels_independent_no_fusion':True,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    q=p.add_mutually_exclusive_group(required=True)
    q.add_argument('--query')
    q.add_argument('--queries-file',type=Path)
    p.add_argument('--path-prefix',required=True)
    p.add_argument('--top',type=int,default=5)
    p.add_argument('--source-map',type=Path,default=SOURCE_MAP)
    p.add_argument('--native-db',type=Path,default=NATIVE_DB)
    p.add_argument('--original-root',type=Path,default=SOURCE_ROOT)
    p.add_argument('--derivative-root',type=Path,default=DERIVED_ROOT)
    p.add_argument('--completed-ledger',type=Path,default=RECEIPTS)
    p.add_argument('--output',type=Path,required=True,help='New private output under existing KMD source-audit tree')
    args=p.parse_args()
    out=safe_private_result_path(args.output,source_root=args.original_root,
                                 derived_root=args.derivative_root,native_filedb=args.native_db,
                                 source_map_db=args.source_map)
    questions=input_queries(query=args.query,query_file=args.queries_file)
    record=search(prefix=args.path_prefix,questions=questions,top=args.top,
                  source_map=args.source_map,native_db=args.native_db,
                  original_root=args.original_root,derived_root=args.derivative_root,
                  completed_ledger=args.completed_ledger)
    raw=(json.dumps(record,indent=2,ensure_ascii=False)+'\n').encode()
    fd=os.open(out,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_CLOEXEC,0o600)
    with os.fdopen(fd,'wb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    print('COMPLETE_PDF_ORIGINAL_SOURCE_WINDOWS',record['source_scan']['original_source_windows_indexed'])
    print('VERIFIED_ORIGINAL_PUBLISHER_FILES',record['unique_returned_originals'])
    print('RETURNED_SOURCE_CANDIDATES_NOT_FACTUAL_ANSWERS',True)
    print('PRIVATE_EVIDENCE',str(out),'SHA256',hashlib.sha256(raw).hexdigest())


if __name__=='__main__':main()
