"""Read-only PDF repair dependency inventory. NEVER a cutover operator.

The staged repaired derivatives may be correct, yet the live original FileDB
has old original-content vectors and generated summaries referring to old
source-map derivative revisions. This tool only inventories exact identities,
existing native dependents and current worker activity. It cannot authorize
or execute destructive/live changes, or grant a service-release lease.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .summary_coverage import revision_identity

from .source_review_audit import safe_file, sha_file

SHA = re.compile(r"[a-f0-9]{64}\Z")


def inspect_staged_cutover(records:list[dict], *, source_map:sqlite3.Connection,
                            native:sqlite3.Connection, original_root:Path,
                            derived_root:Path, stage_root:Path,
                            completed_receipts:dict[str,str],
                            service_states:dict[str,str],
                            summary_root:Path|None=None,
                            expected_count:int=327) -> dict:
    """Produce an independent, fail-closed inventory, with NO mutation.

    `service_states` are actually observed, not inferred from a claimed model
    completion. Even when all observed services are inactive, an exact writer
    lease/transaction/rollback plan is still required; this function NEVER
    returns permission to promote or remove legacy vectors/summaries.
    """
    if (type(expected_count) is not int or not 1<=expected_count<=327 or
            not isinstance(records,list) or len(records)!=expected_count):
        raise ValueError('incomplete or unbounded source cohort')
    required={'kmd-summary-worker-coder','kmd-summary-worker-q4'}
    if not isinstance(service_states,dict) or set(service_states)!=required:
        raise ValueError('missing actually observed summary service state')
    if any(state not in ('active','inactive','unknown','failed') for state in service_states.values()):
        raise ValueError('untrusted service state')
    if source_map.execute('PRAGMA query_only').fetchone()[0]!=1 or native.execute('PRAGMA query_only').fetchone()[0]!=1:
        raise ValueError('both metadata connections must be read-only')
    if not (stage_root.is_dir() and not stage_root.is_symlink()):
        raise ValueError('private staging root not a safe directory')
    if summary_root is not None and (not summary_root.is_dir() or summary_root.is_symlink()):
        raise ValueError('summary dependency root unavailable or unsafe')
    indexed_ids=set();paths=set();counts={'original_vectors':0,'summary_nodes':0,
        'summary_vectors':0,'affected_sources_with_summary_nodes':0,'affected_sources_with_original_vectors':0,
        'old_revision_summary_trees':0,'old_revision_progress_journals':0,
        'unexpected_new_revision_summary_trees':0}
    listing=[]
    for item in records:
        if not isinstance(item,dict):raise ValueError('invalid staged receipt')
        path=item.get('original_path');o=item.get('original_sha256');old=item.get('old_derived_sha256');new=item.get('new_staged_derived_sha256')
        native_id=item.get('native_original_filedb_id');stage=item.get('staged_private_file')
        if (not isinstance(path,str) or path in paths or not path.endswith('.pdf') or
                not all(isinstance(v,str) and SHA.fullmatch(v) for v in (o,old,new)) or
                type(native_id) is not int or native_id in indexed_ids or
                not isinstance(stage,str) or not Path(stage).is_absolute()):
            raise ValueError('ambiguous approved original or staged revision')
        paths.add(path);indexed_ids.add(native_id)
        row=source_map.execute('SELECT source_sha256,derived_sha256,derived_path,status FROM originals WHERE source_path=?',(path,)).fetchall()
        if len(row)!=1 or row[0]!=(o,old,path+'.txt','complete'):
            raise ValueError('live source-map no longer has the attested old source revision')
        if completed_receipts.get(path)!=o:
            raise ValueError('approved acquisition original revision changed')
        n=native.execute('SELECT id FROM filesystem_entries WHERE relative_path_display=? AND entry_type=?',(path,'file')).fetchall()
        if len(n)!=1 or n[0][0]!=native_id:raise ValueError('native original identity differs from staging receipt')
        orig=safe_file(original_root,path);der=safe_file(derived_root,path+'.txt')
        candidate=Path(stage)
        if not candidate.is_relative_to(stage_root):raise ValueError('staged path outside private audited tree')
        candidate=safe_file(stage_root,str(candidate.relative_to(stage_root)))
        if sha_file(orig)!=o or sha_file(der)!=old or sha_file(candidate)!=new:
            raise ValueError('original, old derivative or private repaired derivative SHA changed')
        if old==new:raise ValueError('candidate did not change the previous invalid derivative')
        chunk=native.execute('SELECT COUNT(*),SUM(CASE WHEN embedding_blob IS NOT NULL THEN 1 ELSE 0 END) '
              'FROM content_chunks WHERE filesystem_entry_id=?',(native_id,)).fetchone()
        reps=native.execute('SELECT COUNT(*),SUM(CASE WHEN r.embedding_blob IS NOT NULL THEN 1 ELSE 0 END) '
              'FROM content_representations r JOIN content_chunks c ON r.chunk_id=c.chunk_id '
              'WHERE c.filesystem_entry_id=?',(native_id,)).fetchone()
        chunk_count,vecs=int(chunk[0]),int(chunk[1] or 0)
        summary_count,summary_vectors=int(reps[0]),int(reps[1] or 0)
        counts['original_vectors']+=vecs;counts['summary_nodes']+=summary_count;counts['summary_vectors']+=summary_vectors
        counts['affected_sources_with_summary_nodes']+=summary_count>0
        counts['affected_sources_with_original_vectors']+=vecs>0
        artifacts={}
        if summary_root is not None:
            basename=re.sub(r'[^A-Za-z0-9_.-]+','-',Path(path).name[:64])
            old_name=basename+'.'+revision_identity(path,o,old)[:24]+'.summary-tree.json'
            newer=basename+'.'+revision_identity(path,o,new)[:24]+'.summary-tree.json'
            old_tree=summary_root/old_name
            old_progress=summary_root/'progress'/(old_name+'.progress.json')
            new_tree=summary_root/newer
            for label,file,limit in (('old_summary_tree',old_tree,50*1024*1024),
                                      ('old_progress_journal',old_progress,4*1024*1024),
                                      ('new_revision_summary_tree',new_tree,50*1024*1024)):
                exists=file.exists() or file.is_symlink()
                if exists and (file.is_symlink() or not file.is_file() or file.stat().st_size>limit):
                    raise ValueError('summary artifact unsafe or oversized')
                artifacts[label+'_exists']=exists
                if exists:artifacts[label+'_sha256']=sha_file(file)
            if artifacts['old_summary_tree_exists']:
                payload=json.loads(old_tree.read_bytes())
                if (payload.get('original_path')!=path or payload.get('original_sha256')!=o
                        or payload.get('derived_sha256')!=old):
                    raise ValueError('historical summary tree different source revision')
            if summary_count>0 and not artifacts['old_summary_tree_exists']:
                raise ValueError('native summary representations have no revision-matched original tree')
            counts['old_revision_summary_trees']+=artifacts['old_summary_tree_exists']
            counts['old_revision_progress_journals']+=artifacts['old_progress_journal_exists']
            counts['unexpected_new_revision_summary_trees']+=artifacts['new_revision_summary_tree_exists']
        listing.append({'original_path':path,'original_sha256':o,'old_derived_sha256':old,
             'new_staged_derived_sha256':new,'native_original_file_id':native_id,
             'existing_native_chunks':chunk_count,'existing_native_original_vectors':vecs,
             'existing_native_summary_representations':summary_count,
             'existing_native_summary_vectors':summary_vectors,
             'staged_sha_checked':True,'live_old_original_sha_checked':True,
             'private_summary_artifacts':artifacts})
    return {'schema':'kmd.staged_pdf_repair_readonly_cutover_dependency_inventory.v1',
            'originals_reverified':len(listing),'native_dependencies':counts,
            'service_states':service_states,'workers_not_quiesced':any(s!='inactive' for s in service_states.values()),
            'safe_to_cutover':False,
            'blockers':['live_original_summarization_services_not_quiesced'] if any(s!='inactive' for s in service_states.values()) else [],
            'still_required':['coordinate_other_consumers_and_lease_release',
              'versioned_rollback_of_old_derived_source_map_native_vectors_and_summaries',
              'transactional_source_revision_cutover_operator_not_implemented',
              'independent_post_migration_original_page_proof_and_stale_summary_invalidation'],
            'no_source_cache_source_map_filedb_vector_or_summary_writes':True,
            'proofs':listing}
