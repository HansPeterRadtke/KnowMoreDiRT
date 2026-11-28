#!/usr/bin/env python3
"""Optional original-SHA-checked RAM-only publisher-first-unit lexical lookup.

Cold-start FTS index build reads ~16 KiB of each currently completed original
DERIVATIVE. Search candidates are not quoted or promoted until a full original
and derivative byte SHA, completed acquisition ledger and unique native FileDB
original have been independently verified. No persistent new database, model,
source acquisition, or existing KMD source/index mutation is performed.
Batch several queries to amortize cold-start indexing. The two FTS rankings
remain independent and never imply calibrated answer truth.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from file_system_catalog.publisher_firstunit_discovery import (
    FirstUnitRAMIndex,load_complete_source_previews,verify_index_candidate,
    validate_literal_source_directory_prefix,
    validate_required_original_source_phrase,required_firstunit_source_phrase_proof,
    compiled_required_original_source_phrase,
    verify_attested_firstunit_phrase_on_physical_pdf_page,
)


def approved_original_receipts(path:Path) -> dict[str,str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size>64*1024*1024:
        raise ValueError('official original acquisition ledger missing/too large')
    doc=json.loads(path.read_bytes())
    if not isinstance(doc,dict) or not isinstance(doc.get('items'),list):
        raise ValueError('invalid complete publisher source acquisition ledger')
    receipts={}
    for row in doc['items']:
        if not isinstance(row,dict):
            raise ValueError('invalid approved original receipt row')
        if row.get('status')=='complete':
            original=row.get('path');digest=row.get('sha256')
            if (not isinstance(original,str) or original in receipts
                    or not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest)):
                raise ValueError('duplicate or invalid approved original source digest')
            receipts[original]=digest
    return receipts


def input_queries(*,query:str|None,query_file:Path|None) -> list[dict]:
    if bool(query is not None)==bool(query_file is not None):
        raise ValueError('provide exactly one text question or existing private query file')
    if query_file is not None:
        if (query_file.is_symlink() or not query_file.is_file()
                or query_file.stat().st_size>100*1024):
            raise ValueError('untrusted bounded batch question file')
        payload=json.loads(query_file.read_bytes())
        if not isinstance(payload,dict) or set(payload)!={'queries'}:
            raise ValueError('batch input must contain exactly queries array')
        items=payload['queries']
    else:
        items=[{'id':'single','query':query}]
    if not isinstance(items,list) or not 1<=len(items)<=32:
        raise ValueError('bounded one-to-thirty-two query batch required')
    seen=set()
    for row in items:
        if (not isinstance(row,dict)
                or set(row) not in ({'id','query'},
                                    {'id','query','require_source_phrase'})):
            raise ValueError('invalid batch question object')
        if 'require_source_phrase' in row:
            value=row['require_source_phrase']
            if value is None:
                raise ValueError('explicit per-question phrase must not be null')
            validate_required_original_source_phrase(value)
        label=row['id'];q=row['query']
        if (not isinstance(label,str) or not re.fullmatch('[A-Za-z0-9_.-]{1,80}',label)
                or label in seen or not isinstance(q,str) or not 3<=len(q)<=600
                or not q.strip()):
            raise ValueError('invalid, repeated or unbounded user question')
        seen.add(label)
    return items


def _sqlite_busy_or_locked(error:sqlite3.OperationalError)->bool:
    """Only actual sqlite BUSY/LOCKED errors are retryable read conflicts."""
    code=getattr(error,'sqlite_errorcode',None)
    return (isinstance(code,int) and
            (code&255) in (sqlite3.SQLITE_BUSY,sqlite3.SQLITE_LOCKED))


def verify_original_candidate_with_bounded_busy_retry(**kwargs)->dict:
    """At most one retry of an idempotent, source-read-only verification.

    Do NOT label an indexer's temporary SQLite lock as an unapproved or
    irrelevant publisher document. A persistent writer conflict aborts the
    complete command, leaving its fresh output file unwritten.
    """
    for attempt in range(2):
        try:
            return verify_index_candidate(**kwargs)
        except sqlite3.OperationalError as exc:
            if not _sqlite_busy_or_locked(exc):
                raise
            if attempt==1:
                raise RuntimeError(
                    'current original verification cannot proceed under active '
                    'SQLite read lock; query aborted without a partial no-match result'
                ) from exc
            time.sleep(0.4)
    raise AssertionError('unreachable bounded SQLite read verification state')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    choose=ap.add_mutually_exclusive_group(required=True)
    choose.add_argument('--query')
    choose.add_argument('--queries-file',type=Path)
    ap.add_argument('--top',type=int,default=5)
    ap.add_argument('--path-prefix',help='Opt-in exact original subfolder ending in /. Literal percent and underscore are not wildcards; only that folder is scanned.')
    ap.add_argument('--verify-pdf-previews',type=int,default=0,
                    help='Independently verify physical original page for at most this many results per channel/query (0..3)')
    ap.add_argument('--require-source-phrase',
                    help='Optional global literal phrase inside SHA-verified FIRST publisher-source unit (not title/summary). For batches use each question\'s optional require_source_phrase field instead; absence is NOT proof later pages lack it.')
    ap.add_argument('--source-map',type=Path,default=Path('/data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3'))
    ap.add_argument('--native-db',type=Path,default=Path('/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3'))
    ap.add_argument('--original-root',type=Path,default=Path('/data/var/Knowledge Library'))
    ap.add_argument('--derivative-root',type=Path,default=Path('/data/var/knowmoredirt/knowledge-library/semantic/text'))
    ap.add_argument('--completed-ledger',type=Path,default=Path('/data/var/Knowledge Library/_management/status.json'))
    ap.add_argument('--output',type=Path,help='New private JSON output file; stdout otherwise')
    args=ap.parse_args()
    args.path_prefix=validate_literal_source_directory_prefix(args.path_prefix)
    if (type(args.top) is not int or not 1<=args.top<=20
            or type(args.verify_pdf_previews) is not int or not 0<=args.verify_pdf_previews<=3):
        raise ValueError('invalid bounded number of sources or optional physical-page checks')
    if (args.output is not None and (args.output.exists() or args.output.is_symlink()
            or not args.output.parent.is_dir() or args.output.parent.is_symlink())):
        raise ValueError('private publisher lexical output must be a new path inside a trusted directory')
    for db in (args.source_map,args.native_db):
        if db.is_symlink() or not db.is_file():
            raise ValueError('canonical source-map or native FileDB unavailable')
    queries=input_queries(query=args.query,query_file=args.queries_file)
    global_required_phrase=validate_required_original_source_phrase(
        args.require_source_phrase
    )
    per_query_phrase_count=sum('require_source_phrase' in row for row in queries)
    if global_required_phrase is not None and per_query_phrase_count:
        raise ValueError('global and per-question source phrase requirements conflict')
    any_source_phrase=global_required_phrase is not None or bool(per_query_phrase_count)
    receipts=approved_original_receipts(args.completed_ledger)
    result_rows=[]
    start=time.monotonic()
    with sqlite3.connect(args.source_map.resolve().as_uri()+'?mode=ro',uri=True,timeout=12) as sm,\
         sqlite3.connect(args.native_db.resolve().as_uri()+'?mode=ro',uri=True,timeout=12) as native,\
         FirstUnitRAMIndex() as idx:
        sm.execute('PRAGMA query_only=ON');native.execute('PRAGMA query_only=ON')
        index_receipt=load_complete_source_previews(source_map=sm,derived_root=args.derivative_root,
                                                    index=idx,path_prefix=args.path_prefix)
        indexing_seconds=round(time.monotonic()-start,3)
        # A nonzero explicit omission count is a coverage limitation, not
        # proof that the omitted original is absent or irrelevant.
        for index,question in enumerate(queries,1):
            required_phrase=question.get('require_source_phrase',global_required_phrase)
            required_pattern=(compiled_required_original_source_phrase(required_phrase)
                              if required_phrase is not None else None)
            channels={}
            for channel in ('literal','porter'):
                returned=[]
                omitted=[]
                source_phrase_excluded=[]
                # More candidates than displayed to make failed-current-source
                # checks visible without silently substituting a stale original.
                oversampled=idx.ranked(question['query'],channel=channel,
                                       limit=min(100,max(args.top*6,30)))
                for entry,ident,score in oversampled:
                    if len(returned)>=args.top:break
                    if args.path_prefix is not None and not entry.original_path.startswith(args.path_prefix):
                        raise ValueError('source candidate crossed requested literal directory boundary')
                    try:
                        body=verify_original_candidate_with_bounded_busy_retry(
                            entry=entry,original_root=args.original_root,
                            derived_root=args.derivative_root,
                            completed_receipts=receipts,native_db=native,
                            source_map=sm,query=question['query'],
                            # Avoid optional physical page checks on untrusted
                            # preliminary candidates with no required phrase.
                            # Their original SHA is still rechecked below.
                            verify_pdf_page=(
                                len(returned)<args.verify_pdf_previews
                                and required_pattern is None
                            ))
                    except (ValueError,OSError,UnicodeError) as err:
                        omitted.append({'original_path':entry.original_path,
                                        'reason':f'{type(err).__name__}: {str(err)[:180]}'})
                        continue
                    if required_phrase is not None:
                        attestation=required_firstunit_source_phrase_proof(
                            entry=entry,verified_original_candidate=body,
                            required_phrase=required_phrase,
                        )
                        if attestation is None:
                            source_phrase_excluded.append({
                                'original_path':entry.original_path,
                                'source_original_sha256':body['publisher_original_sha256'],
                                'source_derived_sha256':body['current_derivative_sha256'],
                                'source_unit_label':body['source_unit_label'],
                                'current_original_and_derived_SHA_verified':True,
                                'reason':'required_literal_phrase_not_in_first_verified_source_unit',
                                'not_a_claim_about_other_document_pages_or_figure_content':True,
                                'not_an_unanswerability_verdict':True,
                            })
                            continue
                        body['explicit_required_source_phrase_proof']=attestation
                        if len(returned)<args.verify_pdf_previews:
                            # Prove the required PHRASE and surrounding source
                            # context on the physical PDF page, not an
                            # unrelated query-centered original excerpt.
                            body['required_source_phrase_original_pdf_page_proof']=(
                                verify_attested_firstunit_phrase_on_physical_pdf_page(
                                    entry=entry,verified_original_candidate=body,
                                    phrase_attestation=attestation,
                                    original_root=args.original_root,
                                )
                            )
                    returned.append({'source_rank':len(returned)+1,'native_firstunit_candidate_id':ident,
                                     'lexical_bm25_not_probability':score,
                                     'rank_channel':'literal_title_firstpage_bm25' if channel=='literal'
                                       else 'porter_title_firstpage_bm25',**body})
                channels[channel]={'returned_current_originals':len(returned),
                                   'source_originals':returned,
                                   'stale_or_unverified_candidates_omitted':omitted,
                                   **({'original_pdf_phrase_page_checks_requested_max':args.verify_pdf_previews,
                                       'original_pdf_phrase_contexts_independently_verified':sum(
                                           item.get('required_source_phrase_original_pdf_page_proof',{}).get(
                                               'physical_original_page_verified_for_required_phrase_context') is True
                                           for item in returned),
                                       'required_phrase_pdf_checks_use_phrase_centered_context_not_query_excerpt':True}
                                      if required_phrase is not None and args.verify_pdf_previews else {}),
                                   **({'verified_source_unit_candidates_missing_requested_phrase':
                                       source_phrase_excluded,
                                       'required_phrase_checks_are_only_on_bounded_ranked_candidates':True}
                                      if required_phrase is not None else {})}
            result_rows.append({'id':question['id'],'query':question['query'],
                                'query_sha256':hashlib.sha256(question['query'].encode()).hexdigest(),
                                **({'explicit_required_first_source_unit_phrase':required_phrase,
                                    'source_phrase_rule_applied_to_this_question':True,
                                    'source_phrase_rule_not_a_full_document_search':True}
                                   if required_phrase is not None else {}),
                                'independent_rank_channels':channels,
                                'rank_fusion_applied':False})
            print('SOURCE_LOOKUP',index,'/',len(queries),question['id'],
                  'literal',channels['literal']['returned_current_originals'],
                  'porter',channels['porter']['returned_current_originals'],
                  file=sys.stderr,flush=True)
    output={
      'schema':('kmd.optional_attested_first_source_unit_phrase_gated_lexical.v1'
                if any_source_phrase else
                'kmd.readonly_publisher_first_source_unit_dual_lexical_discovery.v1'),
      **({'global_required_first_source_unit_phrase':global_required_phrase,
          **({'explicit_required_first_source_unit_phrase':global_required_phrase}
             if global_required_phrase is not None else {}),
          'per_query_source_phrase_count':per_query_phrase_count,
          'phrase_acceptance_never_from_catalog_title_or_generated_summary':True,
          'negative_or_empty_results_do_not_prove_absence_in_other_units':True,
          'ranked_candidate_scan_bounded_and_not_exhaustive_full_corpus_phrase_search':True,
          'no_default_rank_or_index_modification':True,
          'optional_original_physical_pdf_verification_attests_required_phrase_context_not_separate_query_excerpt':True}
         if any_source_phrase else {}),
      'generated_utc':datetime.now(timezone.utc).isoformat(),
      **({'literal_source_directory_prefix':args.path_prefix,
          'all_returned_current_originals_within_literal_directory':all(
              candidate['original_path'].startswith(args.path_prefix)
              for row in result_rows
              for ranked in row['independent_rank_channels'].values()
              for candidate in ranked['source_originals']),
          'folder_scope_does_not_prove_absence_outside_requested_directory':True}
         if args.path_prefix is not None else {}),
      'query_count':len(queries),'source_scan':index_receipt,
      'index_build_seconds':indexing_seconds,
      'total_seconds':round(time.monotonic()-start,3),
      'no_persistent_second_database_created':True,
      'native_filedb_and_source_map_read_only':True,
      'persistent_sqlite_busy_causes_command_failure_not_silent_missing_document':True,
      'unexpected_database_errors_are_not_classified_as_stale_originals':True,
      'all_displayed_candidates_original_and_full_derivative_sha_reverified':True,
      'originals_downloaded_or_modified':0,
      'independent_rank_channels_no_fusion':True,
      'purpose':'Extra read-only first-document-source-unit lexical discovery for not-yet-summarized source originals, never a replacement for native KMD semantic search.',
      'warnings':['A source keyword or BM25 match does NOT prove relevance or answer correctness.',
                  'Publisher quotes are verified against current extracted source; literal publisher PDF physical page only where the separate verifier says verified.',
                  'Only the first bounded source unit of each extracted original is indexed. Distant facts and unextracted images/tables cannot be found by this candidate method.',
                  'The two independent ranked lists are NOT fused, and source preview text is untrusted original data.'],
      'queries':result_rows,
    }
    body=(json.dumps(output,indent=2,ensure_ascii=False)+'\n').encode('utf8')
    if args.output is None:
        sys.stdout.buffer.write(body)
    else:
        fd=os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o600)
        with os.fdopen(fd,'wb') as stream:
            stream.write(body);stream.flush();os.fsync(stream.fileno())
        print('OUTPUT',args.output,'SHA256',hashlib.sha256(body).hexdigest(),file=sys.stderr,flush=True)


if __name__=='__main__':
    main()
