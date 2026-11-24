#!/usr/bin/env python3
"""Opt-in ONE-SHOT source-verified native search with lower Python memory.

The original installed Infra semantic search performs all publisher-original,
derived source, native-ID, passage SHA and final revision checks. ONLY its
native candidate scanner is replaced, inside one short-lived CLI process,
by an exact-result streaming implementation. The installed portal, workers,
SQLite databases and source files are never patched or rewritten. Measured
on 17,715 originals: 72% lower peak Python scanner allocations, but NOT a
verified speed improvement. Writes only a new private local evidence receipt.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,'/data/infra/libexec')
from file_system_catalog.native_streamed_semantic import stream_exact_native_semantic_entries

_PRIVATE_ROOT=Path('/data/var/knowmoredirt/knowledge-library/search-audit')
_SCAN_LOCK=threading.Lock()


@contextmanager
def _isolated_one_shot_streamed_native_scanner(native_module):
    """Replace one process-local scanner; ALWAYS restore on success/error.

    Do not import this helper into the persistent authenticated web portal.
    It is designed for a one-shot, single-Python-thread command only; other
    OS processes and the imported module's on-disk code are untouched.
    """
    if threading.active_count()!=1:
        raise RuntimeError('low-memory native scan only permitted in a one-shot single-threaded CLI')
    if not _SCAN_LOCK.acquire(blocking=False):
        raise RuntimeError('parallel scanner override is not authorized')
    missing=object()
    original=missing
    try:
        original=getattr(native_module,'search_semantic_entries',missing)
        if original is missing:
            raise ValueError('installed native reader has no recognized existing scanner')
        if original is stream_exact_native_semantic_entries:
            raise RuntimeError('already substituted low-memory scanner; refusing nested override')
        native_module.search_semantic_entries=stream_exact_native_semantic_entries
        yield
    finally:
        if original is not missing:
            native_module.search_semantic_entries=original
        _SCAN_LOCK.release()


def native_low_memory_search(query:str,*,top:int=5,mode:str='any',
                             original_path:str|None=None,native_module=None):
    """Run the existing source-checking reader unchanged except scanning.

    Includes the existing FULL original/derived physical SHA and native
    generation checks, no caching or copy of the source metadata. Results
    are source candidate excerpts, NOT an independently confirmed answer.
    """
    if (not isinstance(query,str) or not 1<=len(query)<=600 or not query.strip()
            or type(top) is not int or not 1<=top<=5
            or mode not in ('any','full','overview')
            or original_path is not None and
               (not isinstance(original_path,str) or not original_path
                or original_path.startswith('/') or '\\' in original_path
                or any(part in ('','.','..') for part in original_path.split('/')))):
        raise ValueError('invalid bounded current original native source query')
    if native_module is None:
        import knowledge_library_semantic_query as native_module
    with _isolated_one_shot_streamed_native_scanner(native_module):
        result=native_module.search(query,top=top,mode=mode,original_path=original_path)
    if (not isinstance(result,dict) or result.get('count')!=len(result.get('results',[]))
            or result.get('count',0)>top
            or not isinstance(result.get('final_current_original_source_revision_check'),dict)):
        raise ValueError('installed native search did not provide a bounded verified result')
    guard=result['final_current_original_source_revision_check']
    if (guard.get('safe_to_cutover_live') is not False
            or guard.get('cross_store_immutable_generation_pinned') is not False):
        raise ValueError('unexpected installed source generation safety contract')
    for record in result['results']:
        if not record.get('exact_passage_hash_verified') or not record.get('source_passage_sha256'):
            raise ValueError('native scanned source excerpt lacks exact publisher-derived passage SHA proof')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query',help='Bounded natural-language question; already archived originals only')
    parser.add_argument('--top',type=int,default=5)
    parser.add_argument('--mode',choices=('any','overview','full'),default='any')
    parser.add_argument('--original-path',help='Optional exact, already catalogued original path')
    parser.add_argument('--private-output',type=Path,required=True,
            help='Create a new private JSON receipt under existing source audit directory')
    args=parser.parse_args()
    path=args.private_output
    if (not path.is_absolute() or path.is_symlink() or path.exists()
            or not path.parent.is_dir() or path.parent.is_symlink()
            or not path.resolve().is_relative_to(_PRIVATE_ROOT.resolve())):
        raise ValueError('refusing an unsafe or previously occupied private source receipt path')
    start=time.monotonic()
    result=native_low_memory_search(args.query,top=args.top,mode=args.mode,
                                    original_path=args.original_path)
    report={
        'schema':'kmd.private_optin_one_shot_low_memory_native_original_search.v1',
        'query_sha256':hashlib.sha256(args.query.encode('utf8')).hexdigest(),
        'search_scope':'existing Infra native full source and current source-map revision verification',
        'model':result['results'][0].get('model') if result['results'] else None,
        'model_revision_note':'current installed Infra LocalE5, not modified',
        'top':args.top,'retrieval_mode':args.mode,
        'original_path_filter':args.original_path,
        'wall_seconds':round(time.monotonic()-start,3),
        'total_results':result['count'],
        'full_installed_native_original_source_evidence':result,
        'only_native_scanner_process_local_temporarily_substituted':True,
        'source_revision_checks_preserved':True,
        'publisher_paper_or_HTML_answer_factual_truth_claimed':False,
        'live_portal_or_Infra_code_modified':False,
        'native_DB_source_map_vectors_originals_modified':False,
        'no_network_or_new_source_acquisition':True,
        'no_automated_ANN_index_or_persistent_vector_cache':True,
        'benchmarked_speed_improvement_claimed':False,
    }
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as file:
        json.dump(report,file,indent=2,ensure_ascii=False)
        file.flush();os.fsync(file.fileno())
    print('EXACT_SOURCE_VERIFIED_NATIVE_RESULTS',result['count'])
    print('CURRENT_ORIGINAL_SOURCE_REVISION_FENCE',bool(result['final_current_original_source_revision_check']))
    print('ONE_SHOT_LOW_MEMORY_DOES_NOT_CHANGE_INSTALLED_SEARCH',True)
    print('NO_VERIFIED_LATENCY_IMPROVEMENT_CLAIM',True)
    print('PRIVATE_SOURCE_RECEIPT',path)

if __name__=='__main__':main()
