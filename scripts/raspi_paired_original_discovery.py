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
import re
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))

from file_system_catalog.source_review_audit import safe_file,sha_file
from file_system_catalog.publisher_firstunit_discovery import (
    FirstUnitRAMIndex,load_complete_source_previews,verify_index_candidate,
)
from scripts.raspi_publisher_firstunit_discovery import approved_original_receipts
from scripts.raspi_low_memory_native_original_search import native_low_memory_search
from scripts.raspi_source_verified_summary_discovery import (
    verified_original_discovery,DEFAULT_ORIGINAL_ROOT,DEFAULT_DERIVED_ROOT,
    DEFAULT_SOURCE_MAP,DEFAULT_NATIVE_DB,_PRIVATE_ROOT,
)
from scripts.raspi_summary_claim_evidence import read_approved_revision

MAX_PER_CHANNEL=5
# The native passage and source-firstunit defaults remain independently
# bounded to five. Only an explicit deeper *generated-summary* navigation
# channel may use up to ten source-verified original documents.
MAX_SUMMARY_LEXICAL_PER_CHANNEL=10
MAX_UNION=20
MAX_PAIRED_UNION=15
# Publisher originals are SHA-streamed; extracted text is not. Keep their
# independent memory/read constraints separate, as in the native source audit.
MAX_SINGLE_ORIGINAL_BYTES=128*1024*1024
MAX_SINGLE_DERIVED_BYTES=64*1024*1024
MAX_TOTAL_SOURCE_BYTES=256*1024*1024


def independent_original_union(
    native:dict, lexical:dict, *, lexical_top:int=MAX_PER_CHANNEL,
)->list[dict]:
    """No score fusion. Keep native order, then append verified lexical sources.

    A default five plus five candidate cap stays unchanged. The opt-in
    separate summary lexical channel can extend to ten originals, never
    making one interleaved ranking or copying unverifiable source content.
    """
    if (type(lexical_top) is not int
            or not 1<=lexical_top<=MAX_SUMMARY_LEXICAL_PER_CHANNEL
            or not isinstance(native,dict) or not isinstance(lexical,dict)
            or native.get('count')!=len(native.get('results',[]))
            or native.get('count',0)>MAX_PER_CHANNEL
            or lexical.get('originals_physically_verified_and_returned')!=len(lexical.get('candidates',[]))
            or len(lexical['candidates'])>lexical_top
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
    if not 0<=len(selected)<=min(MAX_PAIRED_UNION,MAX_PER_CHANNEL+lexical_top):
        raise ValueError('unbounded source discovery union')
    return list(selected.values())


def verified_publisher_firstunit_candidates(
    *,query:str,top:int,source_map:sqlite3.Connection,native_db:sqlite3.Connection,
    original_root:Path,derivative_root:Path,ledger:Path,
    verify_pdf_previews:int=0,
)->dict:
    """Read existing current-source previews once; SHA-verify each selected hit.

    A source-unit preview is an exact current derivative excerpt, not a
    publisher-PDF-byte quotation or a physically attested page. This
    optional channel also covers originals without completed summaries.
    """
    if not isinstance(query,str) or not 1<=len(query)<=600 or (
            type(top) is not int or not 1<=top<=MAX_PER_CHANNEL
            or type(verify_pdf_previews) is not int or not 0<=verify_pdf_previews<=3):
        raise ValueError('invalid bounded original first-unit search')
    receipts=approved_original_receipts(ledger)
    with FirstUnitRAMIndex() as idx:
        scan=load_complete_source_previews(
            source_map=source_map,derived_root=derivative_root,index=idx,
        )
        ranked=idx.ranked(query,channel='porter',limit=max(top*6,30))
        verified=[]
        omitted=[]
        attempted_original_pdf_page_checks=0
        for entry,identity,score in ranked:
            if len(verified)>=top:
                break
            try:
                physically_check_pdf=(
                    entry.original_path.casefold().endswith('.pdf')
                    and attempted_original_pdf_page_checks<verify_pdf_previews
                )
                candidate=verify_index_candidate(
                    entry=entry,original_root=original_root,
                    derived_root=derivative_root,
                    completed_receipts=receipts,
                    native_db=native_db,source_map=source_map,
                    query=query,verify_pdf_page=physically_check_pdf,
                    max_original_bytes=MAX_SINGLE_ORIGINAL_BYTES,
                )
                if physically_check_pdf:
                    attempted_original_pdf_page_checks+=1
                if (candidate['current_approved_publisher_and_extracted_bytes_rehashed'] is not True
                        or candidate['quoted_text_is_original_publisher_bytes'] is not False
                        or candidate['factual_answer_verified'] is not False
                        or candidate['source_unit_label_scope']!='derived_source_text_only_not_physical_page_proof'):
                    raise ValueError('publisher-source first-unit provenance is ambiguous')
                verified.append({
                    'source_rank':len(verified)+1,
                    'native_firstunit_candidate_id':identity,
                    'porter_bm25_sort_only_not_probability':float(score),
                    **candidate,
                })
            except (ValueError,UnicodeError,OSError,sqlite3.Error) as exc:
                omitted.append({
                    'source_path':entry.original_path,
                    'lexical_rank_before_source_proof':len(verified)+len(omitted)+1,
                    'reason_type':type(exc).__name__,
                    'candidate_not_returned':True,
                })
    return {
        'schema':'kmd.opt_in_verified_publisher_firstunit_source_discovery.v1',
        'source_scan':scan,
        'source_firstunit_index_is_ephemeral_in_memory':True,
        'source_originals_including_unsummarized_are_eligible':True,
        'returned':len(verified),
        'candidate_originals':verified,
        'original_pdf_page_checks_requested_max':verify_pdf_previews,
        'original_pdf_page_checks_attempted':attempted_original_pdf_page_checks,
        'original_pdf_pages_independently_verified':sum(
            int((x.get('physical_pdf_page') or {}).get('physical_pdf_page_verified') is True)
            for x in verified),
        'unverified_source_candidates_excluded':omitted,
        'original_and_full_derivative_sha_reverified_per_returned_candidate':True,
        'source_unit_text_is_derived_original_not_physical_pdf_page_verification':True,
        'factual_answer_verified':False,
    }


def augment_original_union_with_firstunit(base:list[dict], firstunit:dict)->list[dict]:
    """Add at most five independently verified original-source-unit identities.

    Preserve native/summary ranks and append only *new* documents in verified
    Porter order. The final full publisher/derived source SHA must be checked
    AGAIN for all selected originals by final_original_union_revision_guard.
    """
    if (not isinstance(base,list) or len(base)>MAX_PAIRED_UNION
            or not isinstance(firstunit,dict)
            or firstunit.get('returned')!=len(firstunit.get('candidate_originals',[]))
            or firstunit.get('returned',0)>MAX_PER_CHANNEL
            or firstunit.get('source_unit_text_is_derived_original_not_physical_pdf_page_verification') is not True
            or firstunit.get('factual_answer_verified') is not False):
        raise ValueError('untrusted original firstunit candidate receipt')
    selected={}
    for item in base:
        path=item.get('original_path')
        if not isinstance(path,str) or path in selected:
            raise ValueError('duplicate or invalid existing source identity')
        selected[path]={**item,'publisher_firstunit_porter_rank':None,
                        'publisher_firstunit_excerpt_sha256':None}
    for rank,hit in enumerate(firstunit['candidate_originals'],1):
        path=hit.get('original_path')
        original_sha=hit.get('publisher_original_sha256')
        derivative_sha=hit.get('current_derivative_sha256')
        preview=hit.get('publisher_first_unit_preview')
        preview_sha=hit.get('source_preview_sha256')
        if (not isinstance(path,str) or not isinstance(original_sha,str)
                or not re.fullmatch(r'[0-9a-f]{64}',original_sha)
                or not isinstance(derivative_sha,str)
                or not re.fullmatch(r'[0-9a-f]{64}',derivative_sha)
                or type(hit.get('source_rank')) is not int
                or hit['source_rank']!=rank
                or not isinstance(preview,str)
                or not isinstance(preview_sha,str)
                or hashlib.sha256(preview.encode('utf8')).hexdigest()!=preview_sha
                or type(hit.get('source_preview_start_char')) is not int
                or type(hit.get('source_preview_end_char')) is not int
                or not 0<=hit['source_preview_start_char']<hit['source_preview_end_char']
                or hit['source_preview_end_char']-hit['source_preview_start_char']!=len(preview)
                or hit.get('current_approved_publisher_and_extracted_bytes_rehashed') is not True
                or hit.get('quoted_text_is_original_publisher_bytes') is not False
                or hit.get('factual_answer_verified') is not False
                or hit.get('source_unit_label_scope')!='derived_source_text_only_not_physical_page_proof'
                or type(hit.get('native_original_filedb_entry_id')) is not int):
            raise ValueError('first-unit extra source lacks independent current-original proof')
        if path in selected:
            row=selected[path]
            if row['original_source_sha256']!=original_sha:
                raise ValueError('inconsistent original SHA between source retrieval channels')
            if row['publisher_firstunit_porter_rank'] is not None:
                raise ValueError('duplicate firstunit verified original')
        else:
            row={'original_path':path,'original_source_sha256':original_sha,
                 'native_source_passage_rank':None,
                 'native_original_passage_sha256':None,
                 'lexical_summary_candidate_rank':None,
                 'both_independently_returned':False,
                 'publisher_firstunit_porter_rank':None,
                 'publisher_firstunit_excerpt_sha256':None}
            selected[path]=row
        row['publisher_firstunit_porter_rank']=rank
        row['publisher_firstunit_excerpt_sha256']=preview_sha
        row['publisher_firstunit_excerpt_start_char']=hit['source_preview_start_char']
        row['publisher_firstunit_excerpt_end_char']=hit['source_preview_end_char']
        row['publisher_firstunit_derived_sha256']=derivative_sha
        row['publisher_firstunit_native_original_id']=hit['native_original_filedb_entry_id']
    if len(selected)>MAX_UNION:
        raise ValueError('verified three-channel original set exceeds bounded union')
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
        if 'publisher_firstunit_derived_sha256' in item:
            if (item['publisher_firstunit_derived_sha256']!=derived_sha
                    or item.get('publisher_firstunit_native_original_id')!=_native_id):
                raise ValueError('first-unit passage identity moved between independent source checks')
            # One more source-linked excerpt integrity check after all channel
            # ranks were combined, not merely the source-map SHA comparison.
            source_text=derivative.read_text(encoding='utf8',errors='strict')
            start=item['publisher_firstunit_excerpt_start_char']
            end=item['publisher_firstunit_excerpt_end_char']
            if (type(start) is not int or type(end) is not int
                    or not 0<=start<end<=len(source_text)
                    or hashlib.sha256(source_text[start:end].encode()).hexdigest()
                        !=item['publisher_firstunit_excerpt_sha256']):
                raise ValueError('first-unit excerpt no longer occupies the exact current original-source interval')
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
    include_source_firstunit:bool=False,
    verify_firstunit_pdf_pages:int=0,
    summary_lexical_top:int|None=None,
)->dict:
    lexical_top=top if summary_lexical_top is None else summary_lexical_top
    if (not isinstance(query,str) or not 1<=len(query)<=600 or not query.strip()
            or type(top) is not int or not 1<=top<=MAX_PER_CHANNEL
            or type(lexical_top) is not int or not top<=lexical_top<=MAX_SUMMARY_LEXICAL_PER_CHANNEL
            or type(include_source_firstunit) is not bool
            or type(verify_firstunit_pdf_pages) is not int
            or not 0<=verify_firstunit_pdf_pages<=3
            or (not include_source_firstunit and verify_firstunit_pdf_pages)):

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
            ledger=ledger,limit=lexical_top,porter=True,
        )
        union=independent_original_union(native,lexical,lexical_top=lexical_top)
        firstunit=None
        if include_source_firstunit:
            firstunit=verified_publisher_firstunit_candidates(
                query=query,top=top,source_map=sm,native_db=db,
                original_root=original_root,derivative_root=derived_root,
                ledger=ledger,verify_pdf_previews=verify_firstunit_pdf_pages,
            )
            union=augment_original_union_with_firstunit(union,firstunit)
        final=final_original_union_revision_guard(
            records=union,source_map=sm,native_db=db,
            original_root=original_root,derivative_root=derived_root,ledger=ledger,
        )
    return {
        'schema':('kmd.opt_in_verified_three_independent_original_channels.v1'
                  if firstunit is not None else
                  'kmd.opt_in_paired_native_passage_plus_summary_lexical_original_discovery.v1'),
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'query_sha256':hashlib.sha256(query.encode('utf8')).hexdigest(),
        'requested_per_channel':top,
        **({'opt_in_independent_summary_lexical_top':lexical_top,
            'independent_verified_summary_navigation_rank_not_native_similarity':True}
           if summary_lexical_top is not None else {}),
        'native_source_search_seconds':round(native_seconds,3),
        'native_read_sqlite_busy_retry_limit':1,
        'elapsed_seconds':round(time.monotonic()-started,3),
        'native_original_passage_candidates':len(native['results']),
        'source_verified_summary_lexical_original_candidates':len(lexical['candidates']),
        **({'publisher_firstunit_verified_original_candidates':len(firstunit['candidate_originals']),
            'publisher_firstunit_scan':firstunit['source_scan'],
            'publisher_firstunit_source_verified_candidates':firstunit['candidate_originals'],
            'publisher_firstunit_original_pdf_page_checks_requested_max':firstunit['original_pdf_page_checks_requested_max'],
            'publisher_firstunit_original_pdf_page_checks_attempted':firstunit['original_pdf_page_checks_attempted'],
            'publisher_firstunit_original_pdf_pages_independently_verified':firstunit['original_pdf_pages_independently_verified'],
            'publisher_firstunit_candidates_dropped':firstunit['unverified_source_candidates_excluded'],
            'source_firstunit_is_original_derived_text_not_physical_pdf_quote':True,
            'three_independent_channel_ranks_not_fused':True,
            'up_to_fifteen_source_verified_candidate_documents_not_a_new_top_five':(
            lexical_top<=MAX_PER_CHANNEL),
        **({'up_to_twenty_originals_in_explicitly_broadened_independent_navigation_not_fused':True}
           if lexical_top>MAX_PER_CHANNEL else {})}
           if firstunit is not None else {}),
        'unique_original_candidates':len(union),
        'candidates_returned_by_both':sum(int(x['both_independently_returned']) for x in union),
        **({'candidates_found_in_at_least_two_of_three_source_channels':sum(
            sum(r.get(k) is not None for k in (
                'native_source_passage_rank','lexical_summary_candidate_rank',
                'publisher_firstunit_porter_rank'))>=2 for r in union),
            'candidates_found_exclusively_by_original_firstunit':sum(
                r['publisher_firstunit_porter_rank'] is not None
                and r['native_source_passage_rank'] is None
                and r['lexical_summary_candidate_rank'] is None for r in union),
        } if firstunit is not None else {}),
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
    ap.add_argument('--include-source-firstunit',action='store_true',
                    help='Opt-in original first-unit search across all approved originals; extra RAM-only 17k-source index, slow cold start; no score fusion')
    ap.add_argument('--summary-lexical-top',type=int,
                    help='Opt-in up to ten separately source-verified generated-summary lexical originals. Default remains the requested native top (maximum five); no fusion, increased original read/verification cost.')
    ap.add_argument('--verify-firstunit-pdf-pages',type=int,default=0,
                    help='With --include-source-firstunit, independently check original physical PDF page for at most 0-3 ranked PDF candidates; no answer truth claim')
    ap.add_argument('--private-output',type=Path,required=True)
    args=ap.parse_args()
    dest=args.private_output
    if (not dest.is_absolute() or dest.is_symlink() or dest.exists()
            or dest.suffix!='.json' or not dest.parent.is_dir() or dest.parent.is_symlink()
            or not dest.resolve().is_relative_to(_PRIVATE_ROOT.resolve())):
        raise ValueError('destination must be a fresh restricted private search-audit JSON file')
    report=paired_verified_discovery(
        query=args.query,top=args.top,
        include_source_firstunit=args.include_source_firstunit,
        verify_firstunit_pdf_pages=args.verify_firstunit_pdf_pages,
        summary_lexical_top=args.summary_lexical_top,
    )
    fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w',encoding='utf8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
        f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({
       'status':'paired_source_verified_document_discovery',
       'native_original_evidence_candidates':report['native_original_passage_candidates'],
       'lexical_original_candidates':report['source_verified_summary_lexical_original_candidates'],
       **({'source_firstunit_original_candidates':report['publisher_firstunit_verified_original_candidates'],
            'original_pdf_pages_independently_checked':report['publisher_firstunit_original_pdf_pages_independently_verified']}
          if args.include_source_firstunit else {}),
       'distinct_originals':report['unique_original_candidates'],
       'originals_final_physically_rehashed':report[
           'final_current_publisher_source_byte_and_native_file_identity_check'][
           'originals_reverified_after_both_channels'],
       'factual_answer_proven':False,
       'private_receipt':str(dest),
    }))


if __name__=='__main__':
    main()
