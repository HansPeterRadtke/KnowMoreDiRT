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
        if not isinstance(row,dict) or set(row)!={'id','query'}:
            raise ValueError('invalid batch question object')
        label=row['id'];q=row['query']
        if (not isinstance(label,str) or not re.fullmatch('[A-Za-z0-9_.-]{1,80}',label)
                or label in seen or not isinstance(q,str) or not 3<=len(q)<=600
                or not q.strip()):
            raise ValueError('invalid, repeated or unbounded user question')
        seen.add(label)
    return items


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    choose=ap.add_mutually_exclusive_group(required=True)
    choose.add_argument('--query')
    choose.add_argument('--queries-file',type=Path)
    ap.add_argument('--top',type=int,default=5)
    ap.add_argument('--verify-pdf-previews',type=int,default=0,
                    help='Independently verify physical original page for at most this many results per channel/query (0..3)')
    ap.add_argument('--source-map',type=Path,default=Path('/data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3'))
    ap.add_argument('--native-db',type=Path,default=Path('/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3'))
    ap.add_argument('--original-root',type=Path,default=Path('/data/var/Knowledge Library'))
    ap.add_argument('--derivative-root',type=Path,default=Path('/data/var/knowmoredirt/knowledge-library/semantic/text'))
    ap.add_argument('--completed-ledger',type=Path,default=Path('/data/var/Knowledge Library/_management/status.json'))
    ap.add_argument('--output',type=Path,help='New private JSON output file; stdout otherwise')
    args=ap.parse_args()
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
    receipts=approved_original_receipts(args.completed_ledger)
    result_rows=[]
    start=time.monotonic()
    with sqlite3.connect(args.source_map.resolve().as_uri()+'?mode=ro',uri=True) as sm,\
         sqlite3.connect(args.native_db.resolve().as_uri()+'?mode=ro',uri=True) as native,\
         FirstUnitRAMIndex() as idx:
        sm.execute('PRAGMA query_only=ON');native.execute('PRAGMA query_only=ON')
        index_receipt=load_complete_source_previews(source_map=sm,derived_root=args.derivative_root,
                                                    index=idx)
        indexing_seconds=round(time.monotonic()-start,3)
        # A nonzero explicit omission count is a coverage limitation, not
        # proof that the omitted original is absent or irrelevant.
        for index,question in enumerate(queries,1):
            channels={}
            for channel in ('literal','porter'):
                returned=[]
                omitted=[]
                # More candidates than displayed to make failed-current-source
                # checks visible without silently substituting a stale original.
                oversampled=idx.ranked(question['query'],channel=channel,
                                       limit=min(100,max(args.top*6,30)))
                for entry,ident,score in oversampled:
                    if len(returned)>=args.top:break
                    try:
                        body=verify_index_candidate(
                            entry=entry,original_root=args.original_root,
                            derived_root=args.derivative_root,
                            completed_receipts=receipts,native_db=native,
                            source_map=sm,query=question['query'],
                            verify_pdf_page=len(returned)<args.verify_pdf_previews)
                    except (ValueError,OSError,UnicodeError,sqlite3.DatabaseError) as err:
                        omitted.append({'original_path':entry.original_path,
                                        'reason':f'{type(err).__name__}: {str(err)[:180]}'})
                        continue
                    returned.append({'source_rank':len(returned)+1,'native_firstunit_candidate_id':ident,
                                     'lexical_bm25_not_probability':score,
                                     'rank_channel':'literal_title_firstpage_bm25' if channel=='literal'
                                       else 'porter_title_firstpage_bm25',**body})
                channels[channel]={'returned_current_originals':len(returned),
                                   'source_originals':returned,
                                   'stale_or_unverified_candidates_omitted':omitted}
            result_rows.append({'id':question['id'],'query':question['query'],
                                'query_sha256':hashlib.sha256(question['query'].encode()).hexdigest(),
                                'independent_rank_channels':channels,
                                'rank_fusion_applied':False})
            print('SOURCE_LOOKUP',index,'/',len(queries),question['id'],
                  'literal',channels['literal']['returned_current_originals'],
                  'porter',channels['porter']['returned_current_originals'],
                  file=sys.stderr,flush=True)
    output={
      'schema':'kmd.readonly_publisher_first_source_unit_dual_lexical_discovery.v1',
      'generated_utc':datetime.now(timezone.utc).isoformat(),
      'query_count':len(queries),'source_scan':index_receipt,
      'index_build_seconds':indexing_seconds,
      'total_seconds':round(time.monotonic()-start,3),
      'no_persistent_second_database_created':True,
      'native_filedb_and_source_map_read_only':True,
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
