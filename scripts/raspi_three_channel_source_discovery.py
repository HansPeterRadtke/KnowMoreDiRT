#!/usr/bin/env python3
"""Opt-in original-linked, separately ranked semantic and lexical discovery.

This does not replace the installed primary dual search or provide an answer.
It never writes to the native FileDB/source-map/originals, never uses cloud
models, and verifies full publisher original/extracted bytes for each returned
candidate. A mismatched source makes the entire result fail closed.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sqlite3
import sys
import time

THIS=Path(__file__).resolve().parent
sys.path.insert(0,str(THIS))
from raspi_frozen_same_snapshot_three_channel_eval import (
    CORPUS,DERIVED,SOURCE_MAP,LocalE5,capture_source_channels,
    title_summary_fts,query_channels,checked_sqlite,
)
from file_system_catalog.source_triage_provenance import verify_selected_source


def read_only_source_discovery(query:str,*,top:int=8)->dict:
    if not isinstance(query,str) or not 3<=len(query.strip())<=600 or type(top) is not int or not 1<=top<=10:
        raise ValueError('optional source-discovery question/limit is not bounded')
    model=LocalE5(threads=2,batch_size=4)
    start=time.monotonic()
    snapshot=capture_source_channels(model)
    lexical,_=title_summary_fts(snapshot)
    ranks=query_channels(model,snapshot,lexical,query)
    visible={name:paths[:top] for name,paths in ranks.items()}
    paths={path for group in visible.values() for path in group}
    verified={}
    with checked_sqlite(SOURCE_MAP) as ro:
        ro.execute('BEGIN')
        for path in sorted(paths):
            verified[path]=verify_selected_source(path,source_versions=snapshot['source_revisions'],
              source_map=ro,approval_ledger=snapshot['receipt'],catalog=snapshot['catalog'],
              original_root=CORPUS,derived_root=DERIVED)
        ro.rollback()
    return {
       'schema':'kmd.opt_in_three_channel_source_discovery_with_real_original_sha.v1',
       'checked_utc':datetime.now(timezone.utc).isoformat(),
       'query':query,'limit_per_channel':top,
       'candidate_population':snapshot['load_counts']['complete_original_paths'],
       'same_read_only_native_source_snapshot_for_all_channels':True,
       'source_map_and_native_sqlite_are_independently_read_and_not_atomic_cross_file':True,
       'original_passages_include_only_native_source_chunk_vectors_not_facets':True,
       'separate_original_channels_no_rank_fusion_or_source_suppression':True,
       'lexical_method_unchanged_historical_title5_summary1_porter_or':True,
       'external_index_or_original_write_performed':False,
       'service_stop_restart_or_gpu_model_eviction_performed':False,
       'source_navigation_only_not_a_generated_answer':True,
       'all_returned_originals_physically_sha_verified':len(verified),
       'elapsed_seconds':round(time.monotonic()-start,3),
       'channels':{name:[{'rank':i,'source':verified[path]}
             for i,path in enumerate(records,1)] for name,records in visible.items()},
       'caution':'Ranked document identities only. The actual original source section, publisher PDF physical page, factual accuracy, context and conditional or negative claims are not established by semantic or lexical similarity.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('query')
    ap.add_argument('--top',type=int,default=8)
    args=ap.parse_args()
    result=read_only_source_discovery(args.query,top=args.top)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
