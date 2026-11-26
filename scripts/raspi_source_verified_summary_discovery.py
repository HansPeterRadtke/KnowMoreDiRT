#!/usr/bin/env python3
"""Optional verified-original discovery from native generated-summary words.

One-shot, RAM-only FTS over current native summary metadata. Summaries identify
candidate original documents; they NEVER become publisher evidence. The current
publisher bytes, derived bytes, native source graph, hashes and approved receipt
are independently verified for every returned candidate. The original FileDB,
corpus, portal, index workers and installed native semantic reader are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from file_system_catalog.hybrid_search import query_terms
from file_system_catalog.publisher_html_article_discovery import PublisherScope
from file_system_catalog.recent_source_tree_audit import (
    audit_current_source_tree,MAX_SOURCE_BYTES,MAX_DERIVED_BYTES,PROMPT,
)
from scripts.raspi_summary_claim_evidence import read_approved_revision

DEFAULT_ORIGINAL_ROOT=Path('/data/var/Knowledge Library')
DEFAULT_DERIVED_ROOT=Path('/data/var/knowmoredirt/knowledge-library/semantic/text')
DEFAULT_SOURCE_MAP=Path('/data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3')
DEFAULT_NATIVE_DB=Path('/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3')
_PRIVATE_ROOT=Path('/data/var/knowmoredirt/knowledge-library/search-audit')
_SCHEMA='kmd.source_verified_native_summary_lexical_discovery.v1'


def validate_publisher_url_prefix(value: str | None) -> str | None:
    """Reuse the installed publisher-archive URL identity contract.

    This only compares existing approved source-map metadata. It makes no
    network request and never guesses or upgrades a missing publisher version.
    """
    if value is None:
        return None
    return PublisherScope.parse(value).url


def _url_in_scope(url: str | None, *, publisher_url_prefix: str) -> bool:
    """Untrusted source-map URL may be malformed: never match it by accident."""
    if not isinstance(url,str):
        return False
    try:
        return PublisherScope.parse(publisher_url_prefix).contains(url)
    except ValueError:
        return False


def current_summary_roots(source_map:sqlite3.Connection,native_db:sqlite3.Connection,
                          *,maximum_roots:int=20000,maximum_total_characters:int=50_000_000,
                          publisher_url_prefix:str|None=None):
    """Use current revision metadata only, fail closed on ambiguous summaries.

    This initial collection does NOT physically validate each original. Only
    result candidates receive full original and summary-graph verification.
    """
    if (type(maximum_roots) is not int or not 1<=maximum_roots<=25000
            or type(maximum_total_characters) is not int
            or not 500000<=maximum_total_characters<=100_000_000):
        raise ValueError('invalid bounded current-summary scan')
    publisher_url_prefix=validate_publisher_url_prefix(publisher_url_prefix)
    if publisher_url_prefix is None:
        revisions={
            path:(srcsha,dersha) for path,srcsha,dersha in source_map.execute(
                "SELECT source_path,source_sha256,derived_sha256 "
                "FROM originals WHERE status='complete'"
            )
        }
    else:
        revisions={
            path:(srcsha,dersha) for path,srcsha,dersha,url in source_map.execute(
                "SELECT source_path,source_sha256,derived_sha256,original_url "
                "FROM originals WHERE status='complete'"
            ) if _url_in_scope(url,publisher_url_prefix=publisher_url_prefix)
        }
    native_rows=native_db.execute("""
        SELECT f.relative_path_display,c.start_char,c.end_char,c.content_sha256,
               r.representation_text,r.representation_text_sha256,r.generation_json
        FROM filesystem_entries f
        JOIN content_chunks c ON c.filesystem_entry_id=f.id
        JOIN content_representations r ON r.chunk_id=c.chunk_id
        WHERE f.entry_type='file' AND c.collection_id LIKE '%:summary'
          AND c.start_char=0 AND r.representation_kind='summary_long'
          AND r.prompt_version=? AND r.analysis_status='complete'
    """,(PROMPT,)).fetchall()
    candidates={}
    excluded={'historical_or_malformed':0,'oversized_root':0,'ambiguous_current_roots':0,
              'outside_requested_publisher_scope':0}
    for path,start,end,chunksha,summary,textsha,rawgen in native_rows:
        expected=revisions.get(path)
        if not expected:
            excluded[('outside_requested_publisher_scope'
                      if publisher_url_prefix is not None
                      else 'historical_or_malformed')]+=1
            continue
        try:meta=json.loads(rawgen)
        except (ValueError,TypeError):
            excluded['historical_or_malformed']+=1
            continue
        if (not isinstance(meta,dict) or not isinstance(summary,str)
                or len(summary)>15000 or not isinstance(end,int) or end<=0):
            excluded['oversized_root']+=1
            continue
        if (meta.get('original_sha256')!=expected[0]
                or meta.get('derived_sha256')!=expected[1]
                or chunksha!=expected[0]
                or meta.get('source_start_char')!=0
                or meta.get('source_end_char')!=end
                or meta.get('summary_sha256')!=textsha
                or hashlib.sha256(summary.encode('utf8')).hexdigest()!=textsha):
            excluded['historical_or_malformed']+=1
            continue
        candidates.setdefault(path,[]).append((path,summary,expected[0],expected[1],textsha))
    rows=[]
    total_chars=0
    for path,entries in sorted(candidates.items()):
        if len(entries)!=1:
            excluded['ambiguous_current_roots']+=1
            continue
        record=entries[0]
        total_chars+=len(path)+len(record[1])
        if len(rows)>=maximum_roots or total_chars>maximum_total_characters:
            raise ValueError('current generated-summary corpus exceeds ephemeral safety bound')
        rows.append(record)
    if not rows and publisher_url_prefix is None:
        raise ValueError('no current summary roots eligible for candidate discovery')
    return rows,{'native_root_rows_scanned':len(native_rows),
                 'publisher_url_prefix_scope':publisher_url_prefix,
                 'completed_originals_in_requested_url_scope':len(revisions),
                 'empty_requested_url_scope':bool(publisher_url_prefix and not rows),
                 'source_map_complete_entries':source_map.execute(
                     "SELECT COUNT(*) FROM originals WHERE status='complete'"
                 ).fetchone()[0],
                 'metadata_revision_matched_roots':len(rows),
                 'indexed_summary_characters':total_chars,
                 'excluded':excluded,
                 'physical_original_bytes_rehashed_for_every_root':False}


def lexical_ranked_summary_candidates(rows,query,*,porter:bool=True,limit:int=80):
    """FTS5 on transient generated summary/title text, never original evidence."""
    if (not isinstance(query,str) or not 1<=len(query)<=600 or not query.strip()
            or type(porter) is not bool or type(limit) is not int or not 1<=limit<=100):
        raise ValueError('invalid bounded lexical discovery query')
    if (not isinstance(rows,(list,tuple)) or len(rows)>25000
            or sum(len(row[0])+len(row[1]) for row in rows)>50_000_000):
        raise ValueError('ephemeral generated-summary input exceeds safety bound')
    terms=query_terms(query)
    if not terms: return []
    tokenizer=('porter unicode61 remove_diacritics 2'
               if porter else 'unicode61 remove_diacritics 2')
    db=sqlite3.connect(':memory:')
    try:
        # Tokenizer comes from exactly two internal literal choices, not user SQL.
        db.execute("CREATE VIRTUAL TABLE summaries USING fts5("
                   "title,summary,tokenize='"+tokenizer+"')")
        db.executemany('INSERT INTO summaries(rowid,title,summary) VALUES(?,?,?)',(
            (idx+1,record[0].rsplit('/',1)[-1].replace('-',' '),record[1])
            for idx,record in enumerate(rows)
        ))
        db.commit()
        query_expr=' OR '.join('"'+t.replace('"','""')+'"' for t in terms)
        ranked=db.execute(
            'SELECT rowid,bm25(summaries,4.0,1.0) AS score FROM summaries '
            'WHERE summaries MATCH ? ORDER BY score ASC,rowid ASC LIMIT ?',
            (query_expr,limit),
        ).fetchall()
        return [
            {'path':rows[docid-1][0],
             'original_sha256_from_metadata':rows[docid-1][2],
             'derived_sha256_from_metadata':rows[docid-1][3],
             'native_root_summary_sha256':rows[docid-1][4],
             'lexical_rank_among_current_summary_roots':i,
             'bm25_sort_only_not_probability':score}
            for i,(docid,score) in enumerate(ranked,1)
        ]
    finally:
        db.close()


def verified_original_discovery(
    *,query:str,source_map:sqlite3.Connection,native_db:sqlite3.Connection,
    original_root:Path,derived_root:Path,ledger:Path,limit:int=5,
    porter:bool=True,max_scan:int=80,
    publisher_url_prefix:str|None=None,
):
    if (type(limit) is not int or not 1<=limit<=5
            or type(max_scan) is not int or not limit<=max_scan<=100):
        raise ValueError('invalid bounded current-original result count')
    publisher_url_prefix=validate_publisher_url_prefix(publisher_url_prefix)
    rows,coverage=current_summary_roots(
        source_map,native_db,publisher_url_prefix=publisher_url_prefix,
    )
    ranked=lexical_ranked_summary_candidates(rows,query,porter=porter,limit=max_scan)
    accepted=[]
    rejected=[]
    for candidate in ranked:
        try:
            path=candidate['path']
            approved=read_approved_revision(ledger,path)
            if approved!=candidate['original_sha256_from_metadata']:
                raise ValueError('approved source hash differs from native summary')
            graph=audit_current_source_tree(
                original_path=path,source_map=source_map,native_db=native_db,
                original_root=original_root,derivative_root=derived_root,
                completed_receipts={path:approved},
                max_original_bytes=MAX_SOURCE_BYTES,
                max_derived_bytes=MAX_DERIVED_BYTES,
            )
            if (graph['publisher_original_sha256_reverified']!=approved
                    or graph['derived_original_sha256_reverified']!=candidate['derived_sha256_from_metadata']
                    or graph['current_native_summary_nodes_fully_verified']<1):
                raise ValueError('original or native summary revision not verified')
            if read_approved_revision(ledger,path)!=approved:
                raise ValueError('approved acquisition receipt changed during physical source check')
            url_records=source_map.execute(
                "SELECT original_url,status,source_sha256,derived_sha256 FROM originals "
                "WHERE source_path=?",(path,),
            ).fetchall()
            if (len(url_records)!=1 or url_records[0][1]!='complete'
                    or url_records[0][2]!=approved
                    or url_records[0][3]!=candidate['derived_sha256_from_metadata']
                    or (publisher_url_prefix is not None and
                        not _url_in_scope(url_records[0][0],
                                          publisher_url_prefix=publisher_url_prefix))):
                raise ValueError('publisher URL scope or current source revision changed during original check')
            # A valid source identity and summary graph are NOT evidence that
            # this source actually supports the user's proposition.
            accepted.append({
                'original_path':path,
                'source_map_publisher_url_rechecked':url_records[0][0],
                'requested_publisher_url_prefix_matched': (
                    publisher_url_prefix is None or
                    _url_in_scope(url_records[0][0],
                                  publisher_url_prefix=publisher_url_prefix)
                ),
                'original_sha256_reverified':approved,
                'derived_sha256_reverified':graph['derived_original_sha256_reverified'],
                'native_root_summary_sha256_metadata':candidate['native_root_summary_sha256'],
                'lexical_rank_among_current_summary_roots':candidate['lexical_rank_among_current_summary_roots'],
                'bm25_sort_only_not_probability':candidate['bm25_sort_only_not_probability'],
                'all_current_native_summary_nodes_verified':graph['current_native_summary_nodes_fully_verified'],
                'original_bytes_physically_hashed':graph['original_bytes_rehashed'],
                'source_original_and_derived_bytes_verified':True,
                'source_citable_passage_verified_by_this_discovery':False,
                'summary_words_only_locate_original_not_factual_proof':True,
            })
            if len(accepted)>=limit:break
        except (ValueError,sqlite3.Error,OSError,KeyError) as exc:
            rejected.append({
                'lexical_rank':candidate['lexical_rank_among_current_summary_roots'],
                'reason_type':type(exc).__name__,
                'candidate_dropped_without_original_proof':True,
            })
    return {
        'schema':_SCHEMA,
        'query_sha256':hashlib.sha256(query.encode('utf8')).hexdigest(),
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'lexical_tokenizer':('porter' if porter else 'unicode61'),
        'requested_publisher_url_prefix':publisher_url_prefix,
        'scope_matched_only_before_ranking_and_again_after_original_proof':True,
        'top_requested':limit,
        'search_metadata_coverage':coverage,
        'candidate_ranks_fetched':len(ranked),
        'candidate_ranks_actually_examined':len(accepted)+len(rejected),
        'originals_physically_verified_and_returned':len(accepted),
        'candidates':[dict(c,returned_candidate_rank=i)
                      for i,c in enumerate(accepted,1)],
        'unverified_candidates_dropped':rejected,
        'not_a_factual_answer_or_source_passage_quote':True,
        'summary_based_retrieval_does_not_replace_original_or_semantic_search':True,
        'originals_and_native_index_can_change_after_this_point_in_time_check':True,
        'cross_store_immutable_generation_pinned':False,
        'side_effects':{
            'originals_or_sources_downloaded':0,
            'model_calls':0,'native_database_writes':0,
            'originals_or_derived_files_modified':0,
            'running_worker_services_modified':0,
        },
    }


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('query')
    p.add_argument('--top',type=int,default=5)
    p.add_argument('--unicode61',action='store_true',
                   help='Use Unicode-only FTS5 instead of default Porter stemming')
    p.add_argument('--publisher-url-prefix',
                   help='Exact HTTPS archived publisher version directory; never fall back to another version')
    p.add_argument('--private-output',type=Path,required=True)
    p.add_argument('--original-root',type=Path,default=DEFAULT_ORIGINAL_ROOT)
    p.add_argument('--derived-root',type=Path,default=DEFAULT_DERIVED_ROOT)
    p.add_argument('--source-map',type=Path,default=DEFAULT_SOURCE_MAP)
    p.add_argument('--native-db',type=Path,default=DEFAULT_NATIVE_DB)
    p.add_argument('--completed-ledger',type=Path,
                   default=DEFAULT_ORIGINAL_ROOT/'_management/status.json')
    args=p.parse_args()
    dest=args.private_output
    if (not dest.is_absolute() or dest.exists() or dest.is_symlink()
            or dest.suffix!='.json' or not dest.parent.is_dir()
            or dest.parent.is_symlink()
            or not dest.resolve().is_relative_to(_PRIVATE_ROOT.resolve())):
        raise ValueError('output must be a fresh private JSON receipt in existing search-audit root')
    for database in (args.native_db,args.source_map):
        if not database.is_file() or database.is_symlink():
            raise ValueError('canonical native/source-map database unavailable')
    with sqlite3.connect(args.source_map.resolve().as_uri()+'?mode=ro',uri=True) as sm,\
         sqlite3.connect(args.native_db.resolve().as_uri()+'?mode=ro',uri=True) as db:
        sm.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA query_only=ON')
        result=verified_original_discovery(
            query=args.query,source_map=sm,native_db=db,
            original_root=args.original_root,derived_root=args.derived_root,
            ledger=args.completed_ledger,limit=args.top,porter=not args.unicode61,
            publisher_url_prefix=args.publisher_url_prefix,
        )
    fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w',encoding='utf8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({
        'status':'verified_original_candidates_from_ephemeral_generated_summary_index',
        'returned':result['originals_physically_verified_and_returned'],
        'summary_indexed_originals':result['search_metadata_coverage']['metadata_revision_matched_roots'],
        'read_only':True,
        'textual_source_fact_proof':False,
        'publisher_url_scope':result['requested_publisher_url_prefix'],
        'scope_empty':result['search_metadata_coverage']['empty_requested_url_scope'],
        'receipt':str(dest),
    }))


if __name__=='__main__':main()
