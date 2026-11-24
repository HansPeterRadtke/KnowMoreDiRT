#!/usr/bin/env python3
"""Batch 2-6 offline source searches in ONE warm, low-memory native process.

The original installed Infra native search still reopens the original catalog,
completed receipt and source-map on EVERY query, verifies publisher originals,
derived passages and current native IDs, and does its final source guard.
The only process-local changes are the exactly equivalent low-memory vector
scanner and reuse of an identical pinned CPU E5 model session. After ALL
queries finish, a combined fresh final source SHA/native revision recheck
rejects stale earlier results. Neither scanner nor model substitution can
escape this single-threaded, short-lived process. No source downloads.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
import os
import resource
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,'/data/infra/libexec')
from raspi_low_memory_native_original_search import _isolated_one_shot_streamed_native_scanner
from file_system_catalog.native_pinned_batch_snapshot import NativeOneBatchSnapshot

PRIVATE=Path('/data/var/knowmoredirt/knowledge-library/search-audit')
_SNAPSHOT_OVERRIDE_LOCK=threading.Lock()


@contextmanager
def _one_shot_pinned_E5_model(native_module):
    """Warm exactly the installed model; restore original class on all paths."""
    if threading.active_count()!=1:raise RuntimeError('batch source lookup requires a single Python thread')
    model_class=native_module.LocalE5
    model=model_class(threads=2,batch_size=4)
    if (not isinstance(getattr(model,'model',None),str)
            or not isinstance(getattr(model,'revision',None),str)
            or not callable(getattr(model,'embed_queries',None))):
        raise ValueError('installed native model has no verified revision/embedding interface')
    def cached_E5(*,threads:int,batch_size:int):
        if threads!=2 or batch_size!=4:
            raise ValueError('attempted model configuration changed within a pinned one-shot source batch')
        return model
    try:
        native_module.LocalE5=cached_E5
        yield model
    finally:
        native_module.LocalE5=model_class


@contextmanager
def _one_shot_current_native_vector_snapshot_scanner(native_module,snapshot):
    """Read exact candidate vectors ONCE, rechecking SQLite version each query."""
    if threading.active_count()!=1 or not isinstance(snapshot,NativeOneBatchSnapshot):
        raise ValueError('native original snapshot must be a real one-process, single-threaded KMD snapshot')
    if not _SNAPSHOT_OVERRIDE_LOCK.acquire(blocking=False):
        raise RuntimeError('nested source snapshot scanner patch is not permitted')
    original=getattr(native_module,'search_semantic_entries',None)
    try:
        if not callable(original):
            raise ValueError('no installed native scanner to restore after the query batch')
        def exact_from_snapshot(adapter,query_vector):
            return snapshot.rank(adapter,query_vector)
        native_module.search_semantic_entries=exact_from_snapshot
        yield
    finally:
        if original is not None:
            native_module.search_semantic_entries=original
        _SNAPSHOT_OVERRIDE_LOCK.release()


def search_verified_source_batch(queries:list[str],*,top:int=3,mode:str='any',
                                 original_path:str|None=None,native_module=None,
                                 prepared_snapshot:NativeOneBatchSnapshot|None=None):
    """Search already archived sources with final full-batch source recheck.

    The batch is bounded to max 6 queries x max 4 hits; global final guard
    accepts <=30 records and rehashes originals and current native content.
    This only prevents *observed* mid-query revision changes; independent
    writers are not yet admitted to one immutable cross-database generation.
    """
    if (not isinstance(queries,list) or not 2<=len(queries)<=6
            or not all(isinstance(x,str) and x.strip() and 1<=len(x)<=600 for x in queries)
            or len(queries)!=len(set(queries))
            or type(top) is not int or not 1<=top<=4
            or mode not in ('any','overview','full')
            or original_path is not None and
                (not isinstance(original_path,str) or not original_path
                 or len(original_path)>500 or original_path.startswith('/')
                 or '\\' in original_path or any(z in ('','.','..') for z in original_path.split('/')))):
        raise ValueError('invalid bounded offline source search batch')
    if native_module is None:
        import knowledge_library_semantic_query as native_module
    if prepared_snapshot is not None and (mode!='any' or original_path is not None):
        raise ValueError('native shared vector snapshot only supports the exact unfiltered any-mode corpus')
    if prepared_snapshot is not None and len(queries)>4:
        raise ValueError('native shared vector snapshot permits at most four queries per RAM-admitted batch')
    scanner_context=(_isolated_one_shot_streamed_native_scanner(native_module)
                     if prepared_snapshot is None else
                     _one_shot_current_native_vector_snapshot_scanner(native_module,prepared_snapshot))
    results=[];timings=[]
    with scanner_context:
        with _one_shot_pinned_E5_model(native_module) as model:
            for question in queries:
                start=time.monotonic()
                result=native_module.search(question,top=top,mode=mode,original_path=original_path)
                duration=round(time.monotonic()-start,3)
                if (not isinstance(result,dict) or type(result.get('count')) is not int
                        or result['count']!=len(result.get('results',[]))
                        or not 0<=result['count']<=top):
                    raise ValueError('installed native search did not produce bounded source candidates')
                native_guard=result.get('final_current_original_source_revision_check')
                if (not isinstance(native_guard,dict)
                        or native_guard.get('safe_to_cutover_live') is not False
                        or native_guard.get('cross_store_immutable_generation_pinned') is not False):
                    raise ValueError('installed source result lacks honest revision guard')
                for item in result['results']:
                    candidate=item.get('source_passage')
                    sha=item.get('source_passage_sha256')
                    if (not item.get('exact_passage_hash_verified')
                            or not isinstance(candidate,str) or not candidate
                            or not isinstance(sha,str)
                            or hashlib.sha256(candidate.encode('utf8')).hexdigest()!=sha):
                        raise ValueError('native source candidate excerpt SHA does not match original')
                if prepared_snapshot is not None:
                    prepared_snapshot.require_current()
                results.append(result)
                timings.append(duration)
            all_candidates=[hit for result in results for hit in result['results']]
            if len(all_candidates)>24:
                raise ValueError('combined batch native source proof exceeds bounded candidate budget')
            # Critically, earlier results can become stale while we fetch a
            # later question. A single fresh current-source guard for ALL
            # results must pass before ANY source passage is returned.
            current_originals=native_module._originals()
            current_ledger=native_module._completed_original_ledger_hashes()
            current_source_map=native_module._current_derived_versions()
            combined=native_module._final_returned_source_generation_check(
                all_candidates,original_map=current_originals,
                source_hashes=current_ledger,derived_versions=current_source_map)
            if (not isinstance(combined,dict)
                    or combined.get('cross_store_immutable_generation_pinned') is not False
                    or combined.get('safe_to_cutover_live') is not False):
                raise ValueError('final source batch proof did not preserve immutable-generation limit')
            if prepared_snapshot is not None:
                prepared_snapshot.require_current()
    return {'schema':'kmd.private_one_shot_multi_query_native_approved_original_source_batch.v1',
      'query_count':len(queries),'queries_sha256':[hashlib.sha256(x.encode('utf8')).hexdigest() for x in queries],
      'original_scope_path':original_path,'mode':mode,'top_per_query':top,
      'current_source_originals_returned':len(all_candidates),
      'per_query_readonly_search_seconds':timings,
      'shared_local_CPU_model':model.model,'shared_pinned_model_revision':model.revision,
      'full_independent_final_native_publisher_and_derivative_source_proof':combined,
      'full_installed_source_verified_query_results':results,
      'one_shared_model_session_only':True,
      'one_process_local_only_exact_streaming_native_scanner':prepared_snapshot is None,
      'ephemeral_single_batch_read_only_current_vector_snapshot_used':prepared_snapshot is not None,
      'current_native_vector_snapshot_candidate_count':(len(prepared_snapshot.candidates) if prepared_snapshot is not None else None),
      'current_native_vector_snapshot_originals':(prepared_snapshot.complete_originals if prepared_snapshot is not None else None),
      'final_all_earlier_result_revisions_rechecked_after_last_query':True,
      'non_atomic_cross_SQLite_source_generation_pinned':False,
      'source_passages_not_factually_verified_answers':True,
      'installed_default_Infra_reader_modified':False,
      'native_SQLite_or_source_files_changed':False,
      'no_new_external_source_acquisition':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query',action='append',required=True,
        help='Repeat 2-6 times; all queries run through same pinned CPU source model')
    parser.add_argument('--top',type=int,default=3)
    parser.add_argument('--mode',choices=('any','full','overview'),default='any')
    parser.add_argument('--original-path',help='Optional one exact already archived original common to all queries')
    parser.add_argument('--reuse-current-native-vectors',action='store_true',
        help='Opt in to one bounded in-memory READ ONLY candidate snapshot for two to four questions. Requires at least 2 GiB available RAM, refuses any native DB modification. Only any mode, no path filter.')
    parser.add_argument('--private-output',type=Path,required=True)
    opts=parser.parse_args()
    path=opts.private_output
    if (not path.is_absolute() or path.is_symlink() or path.exists()
            or not path.parent.is_dir() or path.parent.is_symlink()
            or not path.resolve().is_relative_to(PRIVATE.resolve())):
        raise ValueError('batch source results require a new existing-directory owner-private output file')
    start=time.monotonic()
    snapshot_build_seconds=None
    if opts.reuse_current_native_vectors:
        if opts.mode!='any' or opts.original_path is not None:
            raise ValueError('shared native vector snapshot supports only full original corpus mode')
        if len(opts.query)>4:
            raise ValueError('shared native vector snapshot admits only two to four distinct questions to protect RAM')
        import knowledge_library_semantic_query as installed
        with NativeOneBatchSnapshot(installed.ORIGINAL) as snapshot:
            result=search_verified_source_batch(opts.query,top=opts.top,mode=opts.mode,
                  original_path=opts.original_path,native_module=installed,
                  prepared_snapshot=snapshot)
            snapshot_build_seconds=snapshot.build_seconds
    else:
        result=search_verified_source_batch(opts.query,top=opts.top,mode=opts.mode,
                                             original_path=opts.original_path)
    result['finished_utc']=datetime.now(timezone.utc).isoformat()
    result['full_process_batch_elapsed_seconds']=round(time.monotonic()-start,3)
    result['ephemeral_native_snapshot_build_seconds']=snapshot_build_seconds
    # Linux resource.ru_maxrss is KiB, not the size of the stored vector array.
    result['peak_entire_process_RSS_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['RSS_includes_local_model_SQLite_pages_Python_and_native_allocations']=True
    raw=(json.dumps(result,indent=2,ensure_ascii=False)+'\n').encode()
    fd=os.open(path,os.O_CREAT|os.O_WRONLY|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as out:
        out.write(raw);out.flush();os.fsync(out.fileno())
    print('BATCH_VERIFIED_ORIGINAL_SOURCE_QUERIES',result['query_count'])
    print('SOURCE_PASSAGE_RESULTS_AFTER_COLLECTIVE_LATE_REVISION_PROOF',result['current_source_originals_returned'])
    print('PINNED_LOCAL_CPU_MODEL_ONE_SESSION',True)
    print('NO_LIVE_KMD_NATIVE_SOURCE_FILES_OR_INDEX_CHANGES',True)
    print('PRIVATE_RECEIPT',path)

if __name__=='__main__':main()
