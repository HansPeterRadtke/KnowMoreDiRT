"""ISOLATED publisher PDF text/source-map and native dependent-row migration.

Strictly private copied historical sources, never an operator for production
files or multi-database cross-store commits. Simulate the newly corrected
physical PDF page text entering the source map, with stale original content
vectors/summary representations removed. The resulting isolated corpus is
explicitly NOT ready for semantic retrieval until new index work completes.
"""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3

from .source_review_audit import safe_file,sha_file
from .source_revision_audit import _verified_derived_body_count
from .staged_physical_page_search import MARKER,source_units_from_corrected_text
from .source_page_shadow_cohort import table_fingerprint,_TABLES


class InjectedPrivateForwardFailure(RuntimeError):
    """Expected artificial failure in an exclusive private test directory."""


def _private_paths(*,baseline_db:Path,private_root:Path,output:Path):
    if not all(isinstance(p,Path) and p.is_absolute() for p in (baseline_db,private_root,output)):
        raise ValueError('private rehearsal requires absolute paths')
    if not private_root.is_dir() or private_root.is_symlink() or not baseline_db.is_file() or baseline_db.is_symlink():
        raise ValueError('private snapshot root or historical SQLite baseline unsafe')
    baseline_parent=baseline_db.parent.resolve()
    output_parent=output.parent.resolve()
    if (not baseline_db.resolve().is_relative_to(private_root.resolve())
            or not output_parent.is_relative_to(private_root.resolve())
            or output_parent.is_relative_to(baseline_parent)
            or output.exists() or output.is_symlink()):
        # Never create a new output INSIDE an immutable recovery bundle,
        # even as an exclusive new filename. Future archive verification
        # must not depend on accidental child entries or mutable receipts.
        raise ValueError('new exclusive private output required outside sealed baseline')


def _stage_rows(*,historical_db:Path,records:list[dict],stage_root:Path,
                expected_count:int) -> tuple[list[dict],dict]:
    if (type(expected_count) is not int or not 1<=expected_count<=327
            or not isinstance(records,list) or len(records)!=expected_count
            or not stage_root.is_dir() or stage_root.is_symlink()):
        raise ValueError('incomplete original PDF source cohort or stage root')
    with sqlite3.connect(historical_db.resolve().as_uri()+'?mode=ro',uri=True,timeout=12) as db:
        db.execute('PRAGMA query_only=ON')
        original={r[0]:r[1:] for r in db.execute('''SELECT original_path,original_sha256,
             old_derived_sha256,new_staged_derived_sha256,native_file_id
             FROM snapshot_revisions''')}
        baseline={t:table_fingerprint(db,t) for t in _TABLES}
        if len(original)!=expected_count or db.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('incomplete or inconsistent private historical original baseline')
        paths=set();items=[]
        for record in sorted(records,key=lambda r:r.get('original_path','')):
            if not isinstance(record,dict):raise ValueError('invalid original stage receipt row')
            path=record.get('original_path')
            if (not isinstance(path,str) or path in paths or path not in original
                    or not path.endswith('.pdf') or path.startswith('/') or '\\' in path
                    or any(x in ('','.','..') for x in path.split('/'))):
                raise ValueError('unapproved, duplicate or unsafe publisher original')
            paths.add(path)
            sha=original[path]
            if (record.get('original_sha256'),record.get('old_derived_sha256'),
                record.get('new_staged_derived_sha256'),record.get('native_original_filedb_id'))!=sha:
                raise ValueError('staged candidate differs from historical original/revision/ID')
            meta=db.execute('SELECT source_sha256,derived_sha256,derived_path,original_kind,status '
                   'FROM originals WHERE source_path=?',(path,)).fetchall()
            entry=db.execute('SELECT relative_path_display,entry_type FROM filesystem_entries '
                   'WHERE id=?',(sha[3],)).fetchall()
            if len(meta)!=1 or meta[0]!=(sha[0],sha[1],path+'.txt','pdf','complete') or entry!=[(path,'file')]:
                raise ValueError('historical source-map or native source identity differs')
            stage=Path(record.get('staged_private_file',''))
            if (not stage.is_absolute() or not stage.is_relative_to(stage_root)
                    or record.get('original_pdf_physical_pages',0)<1):
                raise ValueError('staged candidate outside approved private tree')
            stage=safe_file(stage_root,str(stage.relative_to(stage_root)))
            if stage.stat().st_size>32*1024*1024 or sha_file(stage)!=sha[2]:
                raise ValueError('staged corrected publisher text SHA or bounded size mismatch')
            text=stage.read_text('utf8','strict')
            pages=record['original_pdf_physical_pages']
            markers=list(MARKER.finditer(text))
            if (not markers or len(markers)!=record.get('new_staged_original_page_unit_count')
                    or record.get('new_max_staged_page_label')!=pages
                    or set(int(x[2]) for x in markers)!=set(range(1,pages+1))):
                raise ValueError('corrected publisher PDF unit count or real physical page coverage mismatch')
            publishable=list(source_units_from_corrected_text(text,pages,pages))
            if not publishable:
                raise ValueError('corrected physical PDF pages have no verifiable text')
            # KMD source-map character accounting MUST exclude its generated
            # original-provenance header; using len(text) corrupts progress.
            body_chars=_verified_derived_body_count(text,original_path=path,
                  original_sha=sha[0],source_kind='pdf')
            if body_chars<=0:
                raise ValueError('empty verified physical PDF source body')
            items.append({'original_path':path,'native_file_id':sha[3],
                 'original_sha256':sha[0],'old_derived_sha256':sha[1],
                 'new_staged_derived_sha256':sha[2],
                 'source_file':stage,'source_bytes':stage.stat().st_size,
                 'physical_pdf_pages':pages,'source_units':len(markers),
                 'quoteable_page_units':len(publishable),'body_characters':body_chars})
        if len(paths)!=expected_count:raise ValueError('not all approved original pages present')
    return items,baseline


def _copy_exact_new(source:Path,target:Path,expected_sha:str):
    if target.exists() or target.is_symlink():raise ValueError('private output file already exists')
    target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    fd=os.open(target,os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_WRONLY,0o600)
    try:
        with os.fdopen(fd,'wb') as f,source.open('rb') as inp:
            shutil.copyfileobj(inp,f,1024*1024);f.flush();os.fsync(f.fileno())
    except BaseException:
        raise
    if sha_file(target)!=expected_sha:raise ValueError('copied original publisher source candidate SHA differs')


def private_forward_generation(*,historical_db:Path,records:list[dict],
                stage_root:Path,private_root:Path,output:Path,expected_count:int=327,
                fail_after_copies:int|None=None,fail_after_commit:bool=False) -> dict:
    """Copy and advance a PRIVATE selective original cohort; never live data.

    Failpoints explicitly raise after incomplete staged files or after a
    PRIVATE database commit. Recovery from the old sealed bundle remains
    separate and must be verified; no automatic live promotion is possible.
    """
    _private_paths(baseline_db=historical_db,private_root=private_root,output=output)
    if output.parent.resolve().is_relative_to(stage_root.resolve()):
        raise ValueError('private candidate output must not live inside the staged source corpus')
    if (fail_after_copies is not None and (type(fail_after_copies) is not int
          or not 1<=fail_after_copies<expected_count)):
        raise ValueError('private partial-copy failure must occur strictly within the cohort')
    if type(fail_after_commit) is not bool:
        raise ValueError('invalid private failure injection flag')
    items,base=_stage_rows(historical_db=historical_db,records=records,
                            stage_root=stage_root,expected_count=expected_count)
    output.mkdir(mode=0o700)
    candidate=output/'corrected_source_text'
    candidate.mkdir(mode=0o700)
    for n,item in enumerate(items,1):
        _copy_exact_new(item['source_file'],candidate/(item['original_path']+'.txt'),
                   item['new_staged_derived_sha256'])
        if fail_after_copies==n:
            raise InjectedPrivateForwardFailure(f'after private corrected-file copy {n} of {expected_count}')
    clone=output/'private_after_transition_not_search_ready.sqlite3'
    _copy_exact_new(historical_db,clone,sha_file(historical_db))
    counts={}
    now=datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(clone,timeout=15) as db:
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('BEGIN IMMEDIATE')
        for item in items:
            cur=db.execute('''UPDATE originals SET derived_sha256=?, extracted_pages=?,
                   nonempty_pages=?, text_characters=?,status=?,error=NULL,updated_utc=?
                   WHERE source_path=? AND source_sha256=? AND derived_sha256=? AND status='complete' ''',
                 (item['new_staged_derived_sha256'],item['physical_pdf_pages'],
                  item['physical_pdf_pages'],item['body_characters'],
                  'staged_needs_reindex',now,item['original_path'],
                  item['original_sha256'],item['old_derived_sha256']))
            if cur.rowcount!=1:raise ValueError('private original source-map revision changed during simulation')
            db.execute('DELETE FROM content_chunks WHERE filesystem_entry_id=?',(item['native_file_id'],))
        counts={t:db.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] for t in _TABLES}
        if (counts['originals']!=expected_count or counts['filesystem_entries']!=expected_count
                or counts['content_chunks'] or counts['content_representations']
                or db.execute('PRAGMA foreign_key_check').fetchall()
                or db.execute('PRAGMA quick_check').fetchone()[0]!='ok'):
            raise ValueError('isolated new source map still contains obsolete native index rows')
        db.commit()  # private clone ONLY; no live multi-store transaction.
    if fail_after_commit:
        raise InjectedPrivateForwardFailure('after isolated private candidate database commit')
    with sqlite3.connect(clone.resolve().as_uri()+'?mode=ro',uri=True) as new:
        new.execute('PRAGMA query_only=ON')
        for item in items:
            row=new.execute('SELECT source_sha256,derived_sha256,derived_path,status,extracted_pages,'
                  'nonempty_pages,text_characters FROM originals WHERE source_path=?',
                  (item['original_path'],)).fetchone()
            if row!=(item['original_sha256'],item['new_staged_derived_sha256'],
                  item['original_path']+'.txt','staged_needs_reindex',item['physical_pdf_pages'],
                  item['physical_pdf_pages'],item['body_characters']):
                raise ValueError('new-generation source-map source revision or physical pages inconsistent')
        after={t:table_fingerprint(new,t) for t in _TABLES}
        if after['filesystem_entries']!=base['filesystem_entries']:
            raise ValueError('forward rehearsal modified original native identity/metadata')
    return {'schema':'kmd.private_327_pdf_forward_source_transition_not_search_ready.v1',
       'originals_simulated':len(items),'corrected_page_source_units':sum(z['source_units'] for z in items),
       'lexical_source_page_units_before_original_pdf_quote_verification':sum(z['quoteable_page_units'] for z in items),
       'excluded_broken_or_empty_page_units_from_lexical_candidates':sum(z['source_units']-z['quoteable_page_units'] for z in items),
       'new_page_bodies_characters':sum(z['body_characters'] for z in items),
       'old_content_chunks_invalidated_in_private':base['content_chunks']['rows'],
       'old_summary_representations_invalidated_in_private':base['content_representations']['rows'],
       'private_native_rows_after_transition':counts,
       'old_original_filedb_metadata_unchanged':True,
       'new_corrected_source_sha_and_all_physical_pages_rechecked':True,
       'new_source_status_requires_reindex_before_native_search':'staged_needs_reindex',
       'actual_corrected_retrieval_and_fresh_summary_reindex_not_yet_built':True,
       'original_corpus_and_active_work_sessions_unmodified':True,
       'historical_backup_may_be_stale_and_not_current_rollback':True,
       'no_live_cross_database_or_filesystem_transaction':True,
       'safe_to_cutover_live':False,
       'old_source_snapshot_digest':base,
       'private_candidate_digest':after}


def private_restore_generation(*,bundle_root:Path,private_root:Path,output:Path,
                              expected_count:int=327):
    """Restore a NEW private generation from protected historical old sources.

    This copies the complete sealed file manifest (including summary trees
    and journals) plus native/source-map subset, then compares their SHA. The
    originals and the sealed historical backup are always unchanged.
    """
    from .source_page_shadow_cohort import verify_private_shadow_bundle
    if (not bundle_root.is_absolute() or not bundle_root.resolve().is_relative_to(private_root.resolve())
            or not bundle_root.is_dir() or bundle_root.is_symlink()):
        raise ValueError('private historical sealed bundle outside approved area')
    baseline=bundle_root/'original_native_and_source_map_subset.sqlite3'
    _private_paths(baseline_db=baseline,private_root=private_root,output=output)
    verified=verify_private_shadow_bundle(bundle_root=bundle_root,expected_count=expected_count)
    sealed=json.loads(safe_file(bundle_root,'sealed_read_only_recovery_rehearsal.json').read_bytes())
    output.mkdir(mode=0o700)
    for item in sealed['archived_file_sha_manifest']:
        old=safe_file(bundle_root,item['saved_relative_path'])
        if old.stat().st_size!=item['bytes'] or sha_file(old)!=item['sha256']:
            raise ValueError('historical source archive changed during independent restore')
        _copy_exact_new(old,output/item['saved_relative_path'],item['sha256'])
    _copy_exact_new(baseline,output/baseline.name,sha_file(baseline))
    with sqlite3.connect((output/baseline.name).resolve().as_uri()+'?mode=ro',uri=True) as old:
        old.execute('PRAGMA query_only=ON')
        if (old.execute('PRAGMA foreign_key_check').fetchall()
             or old.execute('PRAGMA quick_check').fetchone()[0]!='ok'):
            raise ValueError('private recovered old native original subset invalid')
        back={t:table_fingerprint(old,t) for t in _TABLES}
    if back!=sealed['native_shadow_snapshot']['logical_tables']:
        raise ValueError('private old vector and native summary row contents not fully recovered')
    return {'sealed_historical_recovery_files_reproduced':len(sealed['archived_file_sha_manifest']),
       'restored_source_map_originals':expected_count,
       'native_original_chunks_restored':back['content_chunks']['rows'],
       'native_summary_representations_restored':back['content_representations']['rows'],
       'all_archived_source_text_summary_and_journal_file_shas_match':True,
       'all_native_and_original_source_map_row_bytes_match_historical_archive':True,
       'recovered_historical_source_is_not_current_live_source':True,
       'safe_to_replace_live_source_with_this_historical_bundle':False}
