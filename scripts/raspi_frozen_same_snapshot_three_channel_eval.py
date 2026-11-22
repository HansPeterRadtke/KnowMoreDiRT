#!/usr/bin/env python3
"""One-pass source-SHA-attested frozen discovery, no live index mutation.

Original chunk vectors, current recursive-summary vectors and FTS title/root
summaries are sampled into one stable in-memory source cohort. After the DB
read handles are closed, each question uses only these captured rows. This
is NOT the installed legacy raw reader: native analysis facets are excluded
because only exact original-passage chunks and hierarchical summary vectors
are compared. Labels are source-known relevant, not independent answer truth.
"""
from __future__ import annotations
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,'/data/infra/libexec')
from file_system_catalog.frozen_same_snapshot_rank import (
  frozen_lexical_terms,rank_max_similarity,reciprocal_rank_fusion,label_coverage,
  verified_native_vector,
)
from file_system_catalog.summary_coverage import current_completed_originals,PROMPT_VERSION
from file_system_catalog.source_review_audit import safe_file,sha_file
from knowledge_library_semantic_model import LocalE5

PRIVATE=Path('/data/var/knowmoredirt/knowledge-library/search-audit')
CORPUS=Path('/data/var/Knowledge Library')
DERIVED=Path('/data/var/knowmoredirt/knowledge-library/semantic/text')
SOURCE_MAP=Path('/data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3')
NATIVE=Path('/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3')
CATALOG=Path('/data/infra/etc/knowledge-library-sources.json')
CASE_FILES={
 'historical':PRIVATE/'hard-retrieval-benchmark-20261107/predeclared-gold.json',
 'independent':PRIVATE/'retrieval-hybrid-corpuswide-20261120/independent_thirteen_originals_holdout_frozen.json',
 'multiple':PRIVATE/'multirelevant-originals-20261121/original_multirelevant_gold_frozen.json',
 'heldout':PRIVATE/'unseen-rerank-holdout-20261107/predeclared-holdout-gold.json',
}
OLD=PRIVATE/'native-overview-retrieval-20261121/predeclared_native_overview_shadow_results.json'
DIM=384


def unique_map(rows,key):
    mapping={}
    for row in rows:
        k=row[key]
        if not isinstance(k,str) or not k or k in mapping:
            raise ValueError('nonunique or invalid frozen publisher/source-map key')
        mapping[k]=row
    return mapping


def checked_sqlite(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing original native source-map or FileDB')
    con=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=30)
    con.execute('PRAGMA query_only=ON')
    return con


def load_frozen_gold():
    prev_raw=OLD.read_bytes();prev=json.loads(prev_raw)
    previous=unique_map(prev['cases'],'id')
    golds={}
    for name,p in CASE_FILES.items():
        raw=json.loads(p.read_bytes());rows=raw if isinstance(raw,list) else raw['cases']
        golds[name]=unique_map(rows,'id')
    frozen=[]
    for id,old in previous.items():
        src= ('historical' if old['family']=='historical_28' else
              'independent' if old['family']=='independent_research_13' else
              'multiple')
        record=golds[src][id]
        known=record.get('relevant_paths') or [record.get('expected_path')]
        if (record['query'] is None or not isinstance(record['query'],str)
                or hashlib.sha256(record['query'].encode()).hexdigest()!=old['query_sha256']
                or known!=old['known_originals']):
            raise ValueError('previous question/known original label was changed after freezing')
        frozen.append({'id':id,'group':'previous_46','query':record['query'],
                       'known':known,'family':old['family']})
    for id,record in golds['heldout'].items():
        if id in previous:raise ValueError('independent holdout reused a prior frozen question')
        frozen.append({'id':id,'group':'independent_27','query':record['query'],
                       'known':[record['expected_path']],'family':'unseen_independent_27'})
    if len(frozen)!=73 or sum(len(z['known']) for z in frozen)!=79:
        raise ValueError('frozen question population or source identities changed')
    return frozen,{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in CASE_FILES.items()},hashlib.sha256(prev_raw).hexdigest()


def capture_source_channels(model):
    """Snapshot original and derived revision metadata, then native vectors.

    DB SELECT statements each provide one consistent snapshot; source-map and
    native DB are distinct stores, so each frozen known-original SHA is later
    rechecked. Writer scheduling is NOT stopped or assumed to be quiesced.
    """
    receipt=json.loads((CORPUS/'_management/status.json').read_bytes())
    ledger=unique_map([z for z in receipt['items'] if z.get('status')=='complete'],'path')
    catalog=unique_map(json.loads(CATALOG.read_bytes())['sources'],'path')
    with checked_sqlite(SOURCE_MAP) as sm:
        sm.execute('BEGIN')
        source_map=unique_map([dict(zip(('path','original_sha','derived_sha','derived_path','chars','status','bytes'),r))
           for r in sm.execute('SELECT source_path,source_sha256,derived_sha256,derived_path,text_characters,status,original_bytes FROM originals')],'path')
        sm.rollback()
    source_revisions={}
    for path,z in source_map.items():
        downloaded=ledger.get(path);src=catalog.get(path)
        if (not downloaded or not src or z['status']!='complete'
               or z['original_sha']!=downloaded['sha256']
               or z['bytes']!=downloaded.get('bytes_received')
               or not isinstance(z['chars'],int) or z['chars']<=100
               or z['derived_path']!=path+'.txt'):
            continue
        source_revisions[path]=(z['original_sha'],z['derived_sha'],z['chars'])
    native_read_start=time.monotonic()
    with checked_sqlite(NATIVE) as db:
        db.execute('BEGIN')
        completed=current_completed_originals(db,source_revisions)
        all_files={path:(file_id,file_size) for file_id,path,file_size in db.execute('SELECT id,relative_path_display,size_bytes FROM filesystem_entries WHERE entry_type=?',('file',))}
        paths=sorted(path for path in completed if path in all_files and all_files[path][1]==source_map[path]['bytes'])
        indexes={path:idx for idx,path in enumerate(paths)}
        if len(paths)<5000:raise RuntimeError('insufficient current complete native summary source roots')
        ids={all_files[path][0]:indexes[path] for path in paths}
        raw_vectors=[];raw_ids=[];raw_seen=0;raw_rejected=0
        query_raw='''SELECT f.id, c.content_sha256, c.embedding_model,c.embedding_model_revision,
         c.embedding_dimension,c.embedding_dtype,c.embedding_blob,c.embedding_sha256
         FROM content_chunks c JOIN filesystem_entries f ON f.id=c.filesystem_entry_id
         WHERE c.collection_id NOT LIKE '%:summary' AND c.embedding_blob IS NOT NULL AND f.entry_type='file' '''
        for id,source_sha,model_name,revision,dim,dtype,blob,digest in db.execute(query_raw):
            if id not in ids:continue
            path=paths[ids[id]]
            raw_seen+=1
            if source_sha!=source_revisions[path][0]:raw_rejected+=1;continue
            array=verified_native_vector(blob,digest,model_name,revision,dim,dtype,
                expected_model=model.model,expected_revision=model.revision)
            if array is None:raw_rejected+=1;continue
            raw_vectors.append(array);raw_ids.append(ids[id])
        sums=[];summary_ids=[];summary_rejected=0;summary_seen=0;root_text={}
        query_summary='''SELECT f.id,c.content_sha256,c.start_char,c.end_char,
         r.representation_kind,r.representation_text,r.representation_text_sha256,
         r.generation_json,r.embedding_model,r.embedding_model_revision,
         r.embedding_dimension,r.embedding_dtype,r.embedding_blob,r.embedding_sha256
         FROM content_representations r JOIN content_chunks c ON c.chunk_id=r.chunk_id
         JOIN filesystem_entries f ON f.id=c.filesystem_entry_id
         WHERE c.collection_id LIKE '%:summary' AND r.prompt_version=?
           AND r.representation_kind IN ('summary','summary_long')
           AND r.analysis_status='complete' AND r.embedding_blob IS NOT NULL AND f.entry_type='file' '''
        for (id,original_sha,source_start,source_end,kind,plain,plain_sha,gen,model_name,revision,dim,dtype,blob,digest) in db.execute(query_summary,(PROMPT_VERSION,)):
            if id not in ids:continue
            path=paths[ids[id]];old_sha,derived_sha,_=source_revisions[path]
            summary_seen+=1
            if (old_sha!=original_sha or not isinstance(plain,str)
                    or hashlib.sha256(plain.encode()).hexdigest()!=plain_sha):
                summary_rejected+=1;continue
            try:metadata=json.loads(gen)
            except (TypeError,ValueError):summary_rejected+=1;continue
            if (metadata.get('original_sha256')!=old_sha or metadata.get('derived_sha256')!=derived_sha
                or metadata.get('source_start_char')!=source_start or metadata.get('source_end_char')!=source_end
                or metadata.get('summary_sha256')!=plain_sha):
                summary_rejected+=1;continue
            arr=verified_native_vector(blob,digest,model_name,revision,dim,dtype,
                expected_model=model.model,expected_revision=model.revision)
            if arr is None:summary_rejected+=1;continue
            sums.append(arr);summary_ids.append(ids[id])
            if (kind=='summary_long' and source_start==0 and path not in root_text
                   and metadata.get('source_tree_sha256')==derived_sha):
                root_text[path]=plain
        db.rollback()
    capture_elapsed=time.monotonic()-native_read_start
    if not raw_vectors or not sums or len(root_text)<5000:
        raise RuntimeError('snapshot incomplete native raw vectors or valid source-linked root summaries')
    return {'paths':paths,'source_revisions':source_revisions,'metadata':source_map,'catalog':catalog,
      'receipt':ledger,'raw':np.ascontiguousarray(np.stack(raw_vectors).astype('<f4')),
      'raw_docids':np.array(raw_ids,dtype='<i4'),
      'summary':np.ascontiguousarray(np.stack(sums).astype('<f4')),
      'summary_docids':np.array(summary_ids,dtype='<i4'),
      'root_text':root_text,
      'load_counts':{'complete_original_paths':len(paths),'all_current_roots':len(completed),
          'raw_vectors_accepted':len(raw_vectors),'raw_seen':raw_seen,'raw_rejected':raw_rejected,
          'summary_vectors_accepted':len(sums),'summary_seen':summary_seen,
          'summary_rejected':summary_rejected,'root_texts_source_checked':len(root_text),
          'native_db_read_duration_seconds':round(capture_elapsed,3)}}


def title_summary_fts(snapshot):
    db=sqlite3.connect(':memory:')
    db.execute("CREATE VIRTUAL TABLE doc USING fts5(title, summary, original_path UNINDEXED, tokenize='porter unicode61 remove_diacritics 2')")
    rows=[]
    for path in snapshot['paths']:
        root=snapshot['root_text'].get(path)
        if not root:continue
        title=snapshot['catalog'][path].get('title') or Path(path).name
        rows.append((title,root,path))
    db.executemany('INSERT INTO doc VALUES (?,?,?)',rows)
    db.commit()
    return db,len(rows)


def source_evidence_for_gold(frozen,snapshot):
    paths=sorted({path for item in frozen for path in item['known']})
    successes=[]
    for path in paths:
        if path not in snapshot['source_revisions']:raise ValueError('frozen original lost its source-map identity: '+path)
        record=snapshot['metadata'][path]
        original=safe_file(CORPUS,path)
        derived=safe_file(DERIVED,record['derived_path'])
        old_sha,derived_sha,_=snapshot['source_revisions'][path]
        success=(sha_file(original)==old_sha==snapshot['receipt'][path]['sha256']
                 and sha_file(derived)==derived_sha
                 and original.stat().st_size==snapshot['receipt'][path]['bytes_received'])
        if not success:raise ValueError('known original content changed since frozen snapshot: '+path)
        successes.append(path)
    return {'distinct_originals_physically_rehashed':len(successes),
            'all_frozen_known_original_revisions_match_current_source_map_and_download_ledger':True,
            'candidate_documents_outside_gold_not_physically_rehashed':True,
            'exact_original_page_passages_not_proven_by_sha':True}


def query_channels(model,snapshot,fts,question):
    queryvec=model.embed_queries([question])[0]
    ranks={}
    ranks['original_passage']=rank_max_similarity(snapshot['raw'],snapshot['raw_docids'],queryvec,snapshot['paths'],top=30)
    ranks['summary']=rank_max_similarity(snapshot['summary'],snapshot['summary_docids'],queryvec,snapshot['paths'],top=30)
    terms=frozen_lexical_terms(question)
    expr=' OR '.join('"'+token+'"' for token in terms)
    ranks['lexical']=[row[0] for row in fts.execute('SELECT original_path FROM doc WHERE doc MATCH ? ORDER BY bm25(doc,5.0,1.0),original_path LIMIT 30',(expr,))]
    return ranks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new-private-report',type=Path,required=True)
    args=p.parse_args()
    output=args.new_private_report
    if (not output.is_absolute() or output.exists() or output.is_symlink()
         or not output.parent.is_dir() or output.parent.is_symlink()
         or not output.resolve().is_relative_to(PRIVATE.resolve())):
        raise ValueError('output must be a new private source-evaluation report')
    frozen,casehashes,previoushash=load_frozen_gold()
    # CPU constrained to two real threads, do not take shared GPU models.
    model=LocalE5(threads=2,batch_size=4)
    snapshot=capture_source_channels(model)
    fts,nlex=title_summary_fts(snapshot)
    proofs=source_evidence_for_gold(frozen,snapshot)
    allcases=[]
    t0=time.monotonic()
    for index,case in enumerate(frozen,1):
        ranks=query_channels(model,snapshot,fts,case['query'])
        ranks['rrf_two']=reciprocal_rank_fusion({k:ranks[k] for k in ('original_passage','summary')},top=30)
        ranks['rrf_three']=reciprocal_rank_fusion({k:ranks[k] for k in ('original_passage','summary','lexical')},top=30)
        metrics={str(t):label_coverage(case['known'],ranks,top=t) for t in (1,5,10,20,30)}
        record={'id':case['id'],'group':case['group'],'family':case['family'],
           'query_sha256':hashlib.sha256(case['query'].encode()).hexdigest(),
           'known_originals':case['known'],
           'label_ranks':{name:[r.index(path)+1 if path in r else None for path in case['known']] for name,r in ranks.items()},
           'metrics':metrics,
           'original_top5':{name:paths[:5] for name,paths in ranks.items()},
           'source_revisions':{name:{'original_sha256':snapshot['source_revisions'][name][0],
                         'derived_sha256':snapshot['source_revisions'][name][1]} for name in case['known']}}
        allcases.append(record)
        if index==1 or index%6==0 or index==len(frozen):
            print('CHECKED_FROZEN',index,'OF',len(frozen),'ID',case['id'],
                  flush=True)
    elapsed=time.monotonic()-t0
    aggregates={}
    for group in ('previous_46','independent_27'):
        selected=[x for x in allcases if x['group']==group]
        by_top={}
        for top in ('1','5','10','20','30'):
            by_top[top]={'queries':len(selected),'known_source_labels':sum(x['metrics'][top]['known_source_count'] for x in selected)}
            for key in ('original_passage','summary','lexical','rrf_two','rrf_three'):
                by_top[top][key+'_known_source_hits']=sum(x['metrics'][top][key+'_hits'] for x in selected)
            for name,parts in (('original_and_summary_union',('original_passage','summary')),
                              ('all_three_source_union',('original_passage','summary','lexical'))):
                by_top[top][name+'_known_source_hits']=sum(
                  sum(any(x['label_ranks'][p][i] is not None and x['label_ranks'][p][i]<=int(top)
                              for p in parts) for i in range(len(x['known_originals']))) for x in selected)
        aggregates[group]=by_top
    doc={'schema':'kmd.actual_current_same_read_snapshot_triple_channel_predeclared_gold_recall.v1',
         'captured_utc':datetime.now(timezone.utc).isoformat(),
         'source_native_and_source_map_are_separate_ro_transaction_snapshots':True,
         'all_search_channels_and_questions_use_one_captured_native_in_memory_embedding_set':True,
         'nonproduction_original_chunk_ranking_excludes_original_analysis_facet_vectors':True,
         'not_the_installed_legacy_full_original_passage_search_implementation':True,
         'query_encoder_cpu_only':model.model,'query_encoder_revision':model.revision,
         'native_snapshot':snapshot['load_counts'],'fts5_current_native_root_titles':nlex,
         'lexical_policy':'EXACT historical November22 title5/summary1 Porter OR and unsplit long stopword literal; bug intentionally kept for reproducibility; no holdout tuning.',
         'rrf_policy':'predeclared equal-vote reciprocal rank fusion k=60, at most 30 documents per channel',
         'frozen_gold_file_sha256':casehashes,'previous_46_frozen_rank_digest':previoushash,
         'gold_original_bytes':proofs,
         'capture_time_seconds':round(elapsed,3),
         'results':aggregates,'cases':allcases,
         'no_source_publisher_original_content_or_native_or_source_map_written':True,
         'top_source_candidates_not_all_physically_rehashed':True,
         'no_factual_answers_or_passage_page_truth_proven':True,
         'not_authorization_to_change_production_ranking_without_separate_context_adversarial_evaluation':True}
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w',encoding='utf8') as out:
        json.dump(doc,out,indent=2,ensure_ascii=False)
        out.write('\n');out.flush();os.fsync(out.fileno())
    print('SAME_SNAPSHOT_METRICS',json.dumps(aggregates,ensure_ascii=False),flush=True)
    print('PRIVATE_SHA',hashlib.sha256(output.read_bytes()).hexdigest(),flush=True)
    print('REPORT',output,flush=True)

if __name__=='__main__':main()
