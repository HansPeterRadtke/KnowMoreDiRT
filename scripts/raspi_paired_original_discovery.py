#!/usr/bin/env python3
"""Source-verified two-channel original discovery, opt-in and read-only.

Return native original-passage semantic hits AND original-verified lexical
native-summary candidates side by side. Union is discovery coverage only; no
retrained rank, aggregate confidence, factual answer or unsafe index cutover.
Recheck current publisher and derivative bytes once after both channels run.
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
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))

from file_system_catalog.source_review_audit import safe_file,sha_file
from scripts.raspi_low_memory_native_original_search import native_low_memory_search
from scripts.raspi_source_verified_summary_discovery import (
    verified_original_discovery,DEFAULT_ORIGINAL_ROOT,DEFAULT_DERIVED_ROOT,
    DEFAULT_SOURCE_MAP,DEFAULT_NATIVE_DB,_PRIVATE_ROOT,
)
from scripts.raspi_summary_claim_evidence import read_approved_revision

MAX_PER_CHANNEL=5
MAX_UNION=10
# Publisher originals are SHA-streamed; extracted text is not. Keep their
# independent memory/read constraints separate, as in the native source audit.
MAX_SINGLE_ORIGINAL_BYTES=128*1024*1024
MAX_SINGLE_DERIVED_BYTES=64*1024*1024
MAX_TOTAL_SOURCE_BYTES=256*1024*1024


def independent_original_union(native:dict, lexical:dict)->list[dict]:
    """No score fusion. Keep native order, then append new lexical candidates."""
    if (not isinstance(native,dict) or not isinstance(lexical,dict)
            or native.get('count')!=len(native.get('results',[]))
            or native.get('count',0)>MAX_PER_CHANNEL
            or lexical.get('originals_physically_verified_and_returned')!=len(lexical.get('candidates',[]))
            or len(lexical['candidates'])>MAX_PER_CHANNEL
            or not isinstance(native.get('final_current_original_source_revision_check'),dict)):
        raise ValueError('missing complete independent source-revision search receipts')
    guard=native['final_current_original_source_revision_check']
    if (guard.get('full_publisher_original_and_current_derivative_SHA_rechecked') is not True
            or guard.get('safe_to_cutover_live') is not False):
        raise ValueError('native search lacks confirmed original current-byte revision proof')
    if (lexical.get('not_a_factual_answer_or_source_passage_quote') is not True
            or lexical.get('cross_store_immutable_generation_pinned') is not False):
        raise ValueError('lexical discovery lacks original/source caution')
    selected:dict[str,dict]={}
    for rank,item in enumerate(native['results'],1):
        path=item.get('original_path')
        sha=item.get('original_source_sha256')
        if (not isinstance(path,str) or not isinstance(sha,str)
                or len(sha)!=64 or item.get('exact_passage_hash_verified') is not True
                or not item.get('source_passage_sha256') or path in selected):
            raise ValueError('invalid or duplicate native original identity/provenance')
        selected[path]={'original_path':path,'original_source_sha256':sha,
                        'native_source_passage_rank':rank,
                        'native_original_passage_sha256':item['source_passage_sha256'],
                        'lexical_summary_candidate_rank':None,
                        'both_independently_returned':False}
    for rank,item in enumerate(lexical['candidates'],1):
        path=item.get('original_path')
        sha=item.get('original_sha256_reverified')
        if (not isinstance(path,str) or not isinstance(sha,str)
                or len(sha)!=64
                or item.get('source_original_and_derived_bytes_verified') is not True
                or item.get('source_citable_passage_verified_by_this_discovery') is not False):
            raise ValueError('lexical candidate lacks verified original/document-only scope')
        if path in selected:
            if selected[path]['original_source_sha256']!=sha:
                raise ValueError('same original has contradictory current source revisions')
            if selected[path]['lexical_summary_candidate_rank'] is not None:
                raise ValueError('duplicate lexical path')
            selected[path]['lexical_summary_candidate_rank']=rank
            selected[path]['both_independently_returned']=True
        else:
            selected[path]={'original_path':path,'original_source_sha256':sha,
                            'native_source_passage_rank':None,
                            'native_original_passage_sha256':None,
                            'lexical_summary_candidate_rank':rank,
                            'both_independently_returned':False}
    if not 0<=len(selected)<=MAX_UNION:
        raise ValueError('unbounded source discovery union')
    return list(selected.values())


def final_original_union_revision_guard(
    *,records:list[dict],source_map:sqlite3.Connection,native_db:sqlite3.Connection,
    original_root:Path,derivative_root:Path,ledger:Path,
)->dict:
    """Independent physical original+derived current-source gate AFTER both readers."""
    if not isinstance(records,list) or len(records)>MAX_UNION:
        raise ValueError('invalid bounded original set')
    bytes_total=0
    checks=[]
    for item in records:
        path=item['original_path']
        matches=source_map.execute(
            'SELECT source_sha256,derived_sha256,derived_path,original_bytes,status '
            'FROM originals WHERE source_path=?',(path,),
        ).fetchall()
        if len(matches)!=1 or matches[0][4]!='complete':
            raise ValueError('union original lacks unique completed source-map record')
        original_sha,derived_sha,relative_derived,source_bytes,_=matches[0]
        if (original_sha!=item['original_source_sha256']
                or read_approved_revision(ledger,path)!=original_sha):
            raise ValueError('union source is no longer the approved published revision')
        original=safe_file(original_root,path)
        derivative=safe_file(derivative_root,relative_derived)
        if (type(source_bytes) is not int or source_bytes != original.stat().st_size
                or original.stat().st_size>MAX_SINGLE_ORIGINAL_BYTES
                or derivative.stat().st_size>MAX_SINGLE_DERIVED_BYTES):
            raise ValueError('union original size mismatch or protected byte limit exceeded')
        bytes_total+=original.stat().st_size+derivative.stat().st_size
        if bytes_total>MAX_TOTAL_SOURCE_BYTES:
            raise ValueError('union original physical read budget exceeded')
        if sha_file(original)!=original_sha or sha_file(derivative)!=derived_sha:
            raise ValueError('union original or derived publisher-source bytes changed after ranking')
        native_files=native_db.execute(
            "SELECT id,size_bytes,content_sha256,hash_status FROM filesystem_entries "
            "WHERE relative_path_display=? AND entry_type='file'",(path,),
        ).fetchall()
        if len(native_files)!=1 or native_files[0][1]!=source_bytes:
            raise ValueError('native original file identity no longer agrees with verified bytes')
        _native_id,_native_size,stored_native_sha,hash_status=native_files[0]
        if stored_native_sha is None:
            # Large source files are intentionally not hashed by the native
            # FileDB scanner. Their *complete* original SHA is still
            # independently verified above against the current source map
            # and approved acquisition ledger, with exact native ID/size.
            if hash_status!='skipped_large':
                raise ValueError('native hash missing without documented large-file policy')
        elif stored_native_sha!=original_sha:
            raise ValueError('native original SHA contradicts independently verified bytes')
        if read_approved_revision(ledger,path)!=original_sha:
            raise ValueError('approved original receipt changed during final paired check')
        latest=source_map.execute(
            'SELECT source_sha256,derived_sha256,derived_path,original_bytes,status '
            'FROM originals WHERE source_path=?',(path,),
        ).fetchall()
        if latest!=matches:
            raise ValueError('source-map revision moved during final original byte proof')
        checks.append({'original_path':path,'original_sha256':original_sha,
                       'derived_sha256':derived_sha,
                       'native_original_file_entry_id':_native_id,
                       'native_scanner_hash_status':hash_status,
                       'native_stored_original_sha_reverified_if_present':stored_native_sha is not None,
                       'publisher_original_sha_independently_rehashed_even_when_native_skipped_large':True,
                       'original_and_derivative_byte_shas_rehashed_after_both_searches':True})
    return {'schema':'kmd.paired_original_discovery_final_physical_revision_guard.v1',
            'originals_reverified_after_both_channels':len(checks),
            'total_source_bytes_physically_rehashed':bytes_total,
            'verification_records':checks,
            'source_map_original_and_native_file_identity_checked':True,
            'cross_store_immutable_generation_pinned':False,
            'native_generated_embeddings_unchanged':True,
            'is_original_passage_relevance_or_truth_proof':False}


def paired_verified_discovery(
    *,query:str,top:int=5,
    original_root:Path=DEFAULT_ORIGINAL_ROOT,
    derived_root:Path=DEFAULT_DERIVED_ROOT,
    source_map_path:Path=DEFAULT_SOURCE_MAP,
    native_db_path:Path=DEFAULT_NATIVE_DB,
    ledger:Path=DEFAULT_ORIGINAL_ROOT/'_management/status.json',
)->dict:
    if (not isinstance(query,str) or not 1<=len(query)<=600 or not query.strip()
            or type(top) is not int or not 1<=top<=MAX_PER_CHANNEL):
        raise ValueError('invalid bounded paired source discovery request')
    started=time.monotonic()
    # In DELETE journal mode the active summarizers periodically hold a
    # SQLite write lock. Only a known SQLITE_BUSY/LOCKED *read-only* query is
    # retried, once, after a short bounded backoff. Do not retry unknown
    # errors or any operation with external write effects.
    import time as _clock
    for attempt in range(2):
        try:
            native=native_low_memory_search(query,top=top,mode='any')
            break
        except sqlite3.OperationalError as exc:
            if attempt or getattr(exc,'sqlite_errorcode',None) not in (
                    sqlite3.SQLITE_BUSY,sqlite3.SQLITE_LOCKED):
                raise
            _clock.sleep(0.35)
    native_seconds=time.monotonic()-started
    with sqlite3.connect(source_map_path.resolve().as_uri()+'?mode=ro',uri=True,timeout=25) as sm,\
         sqlite3.connect(native_db_path.resolve().as_uri()+'?mode=ro',uri=True,timeout=25) as db:
        sm.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA query_only=ON')
        lexical=verified_original_discovery(
            query=query,source_map=sm,native_db=db,
            original_root=original_root,derived_root=derived_root,
            ledger=ledger,limit=top,porter=True,
        )
        union=independent_original_union(native,lexical)
        final=final_original_union_revision_guard(
            records=union,source_map=sm,native_db=db,
            original_root=original_root,derivative_root=derived_root,ledger=ledger,
        )
    return {
        'schema':'kmd.opt_in_paired_native_passage_plus_summary_lexical_original_discovery.v1',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'query_sha256':hashlib.sha256(query.encode('utf8')).hexdigest(),
        'requested_per_channel':top,
        'native_source_search_seconds':round(native_seconds,3),
        'native_read_sqlite_busy_retry_limit':1,
        'elapsed_seconds':round(time.monotonic()-started,3),
        'native_original_passage_candidates':len(native['results']),
        'source_verified_summary_lexical_original_candidates':len(lexical['candidates']),
        'unique_original_candidates':len(union),
        'candidates_returned_by_both':sum(int(x['both_independently_returned']) for x in union),
        'candidate_union_native_order_then_lexical_additions_not_combined_rank':union,
        'final_current_publisher_source_byte_and_native_file_identity_check':final,
        'native_source_verified_passage_results':native,
        'separate_source_verified_summary_lexical_result':lexical,
        'no_aggregate_confidence_or_factual_answer':True,
        'no_single_interleaved_top_five_quality_claim':True,
        'no_edition_scope_supported_by_paired_native_channel':True,
        'source_revision_validation_is_point_in_time_not_atomic_multistore_generation':True,
        'side_effects':{
          'originals_downloaded':0,'native_db_written':0,
          'source_map_written':0,'original_files_changed':0,
          'service_restarts':0,'external_model_requests':0,
          'local_e5_query_encoder_used':True},
    }


def main()->None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('query')
    ap.add_argument('--top',type=int,default=5)
    ap.add_argument('--private-output',type=Path,required=True)
    args=ap.parse_args()
    dest=args.private_output
    if (not dest.is_absolute() or dest.is_symlink() or dest.exists()
            or dest.suffix!='.json' or not dest.parent.is_dir() or dest.parent.is_symlink()
            or not dest.resolve().is_relative_to(_PRIVATE_ROOT.resolve())):
        raise ValueError('destination must be a fresh restricted private search-audit JSON file')
    report=paired_verified_discovery(query=args.query,top=args.top)
    fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w',encoding='utf8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
        f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({
       'status':'paired_source_verified_document_discovery',
       'native_original_evidence_candidates':report['native_original_passage_candidates'],
       'lexical_original_candidates':report['source_verified_summary_lexical_original_candidates'],
       'distinct_originals':report['unique_original_candidates'],
       'originals_final_physically_rehashed':report[
           'final_current_publisher_source_byte_and_native_file_identity_check'][
           'originals_reverified_after_both_channels'],
       'factual_answer_proven':False,
       'private_receipt':str(dest),
    }))


if __name__=='__main__':
    main()
