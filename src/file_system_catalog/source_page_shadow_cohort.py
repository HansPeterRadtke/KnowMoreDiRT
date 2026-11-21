"""Read-only selective PDF-original recovery snapshot and isolated SQLite rehearsal.

No function here opens or updates the live publisher corpus, source-map or
native FileDB in writable mode. In particular, rehearsal is intentionally a
single temporary SQLite database, *not* a live multi-file transaction.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import sqlite3

_TABLES=('originals','filesystem_entries','content_chunks','content_representations')
_KEYS={'originals':'source_path','filesystem_entries':'id','content_chunks':'chunk_id',
       'content_representations':'representation_id'}


def _fingerprint_value(hasher, value):
    if value is None:
        hasher.update(b'N;')
    elif isinstance(value, bytes):
        hasher.update(b'B'+str(len(value)).encode()+b':');hasher.update(value)
    else:
        # SQLite storage classes remain stable in the cloned SQL schema.
        data=repr(value).encode('utf8')
        hasher.update(type(value).__name__.encode()+b':'+str(len(data)).encode()+b':'+data)


def table_fingerprint(db:sqlite3.Connection, table:str):
    if table not in _TABLES:raise ValueError('unsupported snapshot relation')
    rows=db.execute(f'SELECT * FROM "{table}" ORDER BY "{_KEYS[table]}"')
    digest=hashlib.sha256();count=0
    for row in rows:
        count+=1
        digest.update(b'ROW[')
        for v in row:_fingerprint_value(digest,v)
        digest.update(b']')
    return {'rows':count,'logical_sha256':digest.hexdigest()}


def _require_read_only(conn):
    if conn.execute('PRAGMA query_only').fetchone()[0]!=1:
        raise ValueError('all live metadata sources must be query-only')


def export_cohort_snapshot(*,original_receipts:list[dict], source_map:sqlite3.Connection,
                           native:sqlite3.Connection, dest:Path, expected_count:int=327):
    """Clone exact current approved-cohort rows without copying the 3GB FileDB.

    The two live SQLite inputs are independently read-only: they CANNOT form
    a real two-database, cross-file cutover transaction. The result is a
    recoverability baseline pending a later frozen writer lease and re-audit.
    """
    _require_read_only(native);_require_read_only(source_map)
    if (type(expected_count) is not int or not 1<=expected_count<=327
            or not isinstance(original_receipts,list) or len(original_receipts)!=expected_count):
        raise ValueError('invalid original source cohort')
    if not dest.is_absolute() or dest.is_symlink() or dest.exists() or not dest.parent.is_dir():
        raise ValueError('snapshot must be a new private absolute sqlite path')
    paths=[z.get('original_path') for z in original_receipts]
    ids=[z.get('native_original_file_id',z.get('native_original_filedb_id')) for z in original_receipts]
    if (len(set(paths))!=expected_count or len(set(ids))!=expected_count
            or any(type(p) is not str or not p or p.startswith('/') or '\x00' in p or '\\' in p
                   or '..' in p.split('/') for p in paths)
            or any(type(i) is not int or i<1 for i in ids)):
        raise ValueError('duplicate/invalid original publisher path or native identity')
    for z in original_receipts:
        if any(not isinstance(z.get(k),str) or len(z[k])!=64 or
               any(x not in '0123456789abcdef' for x in z[k])
               for k in ('original_sha256','old_derived_sha256','new_staged_derived_sha256')):
            raise ValueError('unattested PDF original/old/new SHA revision')
    # O_EXCL and 0600 keep private per-original records away from the repo.
    handle=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    os.close(handle)
    target=sqlite3.connect(dest,timeout=10)
    try:
        target.execute('PRAGMA foreign_keys=ON')
        target.execute('PRAGMA journal_mode=DELETE')
        # Begin separate read-only consistent snapshots. Do not hold these
        # transactions while copying the publisher documents to disk.
        source_map.execute('BEGIN');native.execute('BEGIN')
        try:
            for table,src in (('filesystem_entries',native),('content_chunks',native),
                              ('content_representations',native),('originals',source_map)):
                schema=src.execute('SELECT sql FROM sqlite_master WHERE type=? AND name=?',('table',table)).fetchone()
                if not schema or not schema[0] or 'CREATE TABLE' not in schema[0].upper():
                    raise ValueError('source table schema not available')
                target.execute(schema[0])
            # The production indexes are intentionally not imported, but the
            # PRIVATE shadow needs FK lookup indexes: otherwise repeated
            # ON DELETE CASCADE checks scale quadratically in 12K summaries.
            target.execute('CREATE INDEX shadow_chunks_parent_id ON content_chunks(filesystem_entry_id)')
            target.execute('CREATE INDEX shadow_representations_parent_id ON content_representations(chunk_id)')
            target.execute('''CREATE TABLE snapshot_revisions (
                original_path TEXT PRIMARY KEY NOT NULL,
                original_sha256 TEXT NOT NULL,old_derived_sha256 TEXT NOT NULL,
                new_staged_derived_sha256 TEXT NOT NULL,native_file_id INTEGER NOT NULL UNIQUE
            )''')
            target.executemany('INSERT INTO snapshot_revisions VALUES(?,?,?,?,?)',
               ((z['original_path'],z['original_sha256'],z['old_derived_sha256'],
                 z['new_staged_derived_sha256'],i) for z,i in zip(original_receipts,ids)))
            markers=','.join('?' for _ in paths)
            imarkers=','.join('?' for _ in ids)
            sources={
                'originals':(source_map,'SELECT * FROM originals WHERE source_path IN ('+markers+') ORDER BY source_path',paths),
                'filesystem_entries':(native,'SELECT * FROM filesystem_entries WHERE id IN ('+imarkers+') ORDER BY id',ids),
                'content_chunks':(native,'SELECT * FROM content_chunks WHERE filesystem_entry_id IN ('+imarkers+') ORDER BY chunk_id',ids),
                'content_representations':(native,'SELECT r.* FROM content_representations r JOIN content_chunks c ON c.chunk_id=r.chunk_id WHERE c.filesystem_entry_id IN ('+imarkers+') ORDER BY r.representation_id',ids),
            }
            counts={}
            for table in ('originals','filesystem_entries','content_chunks','content_representations'):
                db,sql,args=sources[table]
                rows=db.execute(sql,args)
                placeholders=','.join('?' for _ in rows.description)
                insert='INSERT INTO "'+table+'" VALUES ('+placeholders+')'
                count=0
                while batch:=rows.fetchmany(128):
                    count+=len(batch)
                    if count>100_000:raise ValueError('unexpectedly large original subset')
                    target.executemany(insert,batch)
                counts[table]=count
            if counts['originals']!=expected_count or counts['filesystem_entries']!=expected_count:
                raise ValueError('incomplete source-map or native FileDB publisher original cohort')
            for z,i in zip(original_receipts,ids):
                row=target.execute('SELECT source_sha256,derived_sha256,derived_path,status FROM originals WHERE source_path=?',(z['original_path'],)).fetchone()
                entry=target.execute('SELECT relative_path_display FROM filesystem_entries WHERE id=?',(i,)).fetchone()
                if (row!=(z['original_sha256'],z['old_derived_sha256'],z['original_path']+'.txt','complete')
                        or entry!=(z['original_path'],)):
                    raise ValueError('publisher original revision and native entry inconsistent')
            if target.execute('PRAGMA foreign_key_check').fetchall():
                raise ValueError('foreign key dependent rows inconsistent')
            if target.execute('PRAGMA quick_check').fetchone()[0]!='ok':
                raise ValueError('selective cohort SQLite integrity failed')
            target.commit()
        finally:
            # Non-writer BEGIN release, never commit to the live SQLite input.
            native.rollback();source_map.rollback()
        return {'counts':counts,'logical_tables':{t:table_fingerprint(target,t) for t in _TABLES},
                'new_snapshot_only':True,'live_source_map_or_native_writes':False,
                'multi_file_live_cutover_approved':False}
    finally:
        target.close()


def rehearse_isolated_sqlite_rollback(snapshot_db:Path, *, batch_size:int=327):
    """In a *private* SQLite clone only, force full-cohort mutation then ROLLBACK.

    This does not restore or authorize any change in the live distinct source
    cache, source-map, native database, or independently generated summary tree.
    """
    if type(batch_size) is not int or not 1<=batch_size<=327:
        raise ValueError('invalid private SQLite rehearsal size')
    with sqlite3.connect(snapshot_db) as db:
        db.execute('PRAGMA foreign_keys=ON')
        original={name:table_fingerprint(db,name) for name in _TABLES}
        originals=db.execute('SELECT original_path,new_staged_derived_sha256,native_file_id FROM snapshot_revisions ORDER BY original_path LIMIT ?',
                            (batch_size,)).fetchall()
        if len(originals)!=batch_size:raise ValueError('incomplete private snapshot cohort')
        old_chunks=db.execute('SELECT COUNT(*) FROM content_chunks').fetchone()[0]
        old_reps=db.execute('SELECT COUNT(*) FROM content_representations').fetchone()[0]
        db.execute('BEGIN IMMEDIATE')
        try:
            for path,new_hash,ident in originals:
                db.execute('UPDATE originals SET derived_sha256=? WHERE source_path=?',(new_hash,path))
                db.execute('DELETE FROM content_chunks WHERE filesystem_entry_id=?',(ident,))
            if db.execute('PRAGMA foreign_key_check').fetchall():
                raise ValueError('rehearsal deleted dependent source indexes inconsistently')
            mutated_chunks=db.execute('SELECT COUNT(*) FROM content_chunks').fetchone()[0]
            mutated_reps=db.execute('SELECT COUNT(*) FROM content_representations').fetchone()[0]
            if old_chunks and mutated_chunks>=old_chunks:
                raise ValueError('no old native rows were affected by isolated rehearsal')
            if old_reps and mutated_reps>=old_reps and batch_size==327:
                raise ValueError('no native summary dependent rows affected in isolated rehearsal')
            db.rollback()  # deliberate injected failure before commit
        except BaseException:
            db.rollback()
            raise
        restored={name:table_fingerprint(db,name) for name in _TABLES}
        if restored!=original or db.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('rehearsal rollback failed to reproduce complete original native subset')
        return {'tables_identical_after_forced_rollback':True,
                'all_original_native_chunk_and_summary_rows_restored':True,
                'source_revisions_reverted':True,'original_chunk_rows':old_chunks,
                'original_representation_rows':old_reps,
                'simulated_after_delete_chunk_rows':mutated_chunks,
                'simulated_after_delete_representation_rows':mutated_reps,
                'originals_exercised':len(originals),
                'isolated_single_sqlite_database_only':True,
                'does_not_prove_real_cross_file_or_live_cutover_atomicity':True}


def verify_private_shadow_bundle(*,bundle_root:Path,expected_count:int=327):
    """Independently rehash *sealed* old source files plus native rows.

    This is a read-only proof of a historical private snapshot, never proof
    that current live workers or source-map have not advanced since capture.
    """
    import json
    import re
    from .summary_coverage import revision_identity
    from .source_review_audit import safe_file,sha_file
    if (not bundle_root.is_absolute() or not bundle_root.is_dir()
            or bundle_root.is_symlink() or type(expected_count) is not int
            or not 1<=expected_count<=327):
        raise ValueError('invalid private recovery bundle directory')
    sealed=safe_file(bundle_root,'sealed_read_only_recovery_rehearsal.json')
    if sealed.stat().st_size>2*1024*1024:
        raise ValueError('oversized private snapshot receipt')
    receipt=json.loads(sealed.read_bytes())
    if (receipt.get('schema')!='kmd.private_cohort_original_pdf_rollback_shadowsnapshot.v1'
            or receipt.get('originals')!=expected_count
            or receipt.get('not_safe_as_a_future_recovery_point_without_fresh_quiesced_source_revision_check') is not True
            or receipt.get('no_live_original_cache_source_map_vector_summary_or_worker_changes') is not True):
        raise ValueError('incomplete or falsely certified historical private recovery receipt')
    originals=receipt['archived_file_sha_manifest']
    if not isinstance(originals,list) or len(originals)>expected_count*3 or len(originals)<expected_count:
        raise ValueError('invalid bounded old original and summary recovery listing')
    seen=set();total_bytes=0
    for row in originals:
        if not isinstance(row,dict):raise ValueError('untrusted private recovery row')
        relative=row['saved_relative_path']
        if (not isinstance(relative,str) or relative in seen
                or not relative.startswith(('old_derived/','old_summaries/'))
                or relative.startswith('/') or '..' in relative.split('/')
                or type(row['bytes']) is not int or not 0<=row['bytes']<=32*1024*1024):
            raise ValueError('duplicate or unsafe archived source name')
        seen.add(relative)
        original=safe_file(bundle_root,relative)
        if original.stat().st_size!=row['bytes'] or sha_file(original)!=row['sha256']:
            raise ValueError('recovery content SHA/length mismatch')
        total_bytes+=row['bytes']
    if total_bytes!=receipt['private_saved_old_content_total_bytes']:
        raise ValueError('private saved old publisher content total mismatch')
    native=receipt['native_shadow_snapshot']
    db=safe_file(bundle_root,native['file'])
    if sha_file(db)!=native['sha256'] or db.stat().st_size!=native['bytes']:
        raise ValueError('native selective recovery snapshot SHA/length mismatch')
    with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True,timeout=8) as conn:
        conn.execute('PRAGMA query_only=ON')
        if conn.execute('PRAGMA quick_check').fetchone()[0]!='ok' or conn.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('private recovery relational integrity check failed')
        expected=native['logical_tables']
        actual={name:table_fingerprint(conn,name) for name in _TABLES}
        if expected!=actual:
            raise ValueError('private native original and summary row fingerprints differ')
        if (actual['originals']['rows']!=expected_count
                or actual['filesystem_entries']['rows']!=expected_count
                or conn.execute('SELECT COUNT(*) FROM snapshot_revisions').fetchone()[0]!=expected_count):
            raise ValueError('private cohort row coverage incomplete')
        archived={z['saved_relative_path']:z for z in originals}
        sm={z[0]:z[1:] for z in conn.execute('SELECT source_path,source_sha256,derived_sha256,derived_path,status FROM originals')}
        native={z[0]:z[1] for z in conn.execute('SELECT id,relative_path_display FROM filesystem_entries')}
        # Keep exactly the source-linked files needed for recovery. An archive
        # can be self-consistently hashed yet omit a necessary old derivative
        # or contain the wrong original's stale summary tree.
        dependents=dict(conn.execute('''SELECT c.filesystem_entry_id, COUNT(r.representation_id)
                   FROM content_chunks c LEFT JOIN content_representations r
                   ON r.chunk_id=c.chunk_id GROUP BY c.filesystem_entry_id'''))
        required=set();summarized=0
        for path,source_sha,old_sha,new_sha,identity in conn.execute(
                  'SELECT original_path,original_sha256,old_derived_sha256,new_staged_derived_sha256,native_file_id FROM snapshot_revisions'):
            if (sm.get(path)!=(source_sha,old_sha,path+'.txt','complete')
                    or native.get(identity)!=path):
                raise ValueError('historical source revision/source-map/native identity mismatch')
            rel='old_derived/'+path+'.txt'
            if rel not in archived or archived[rel]['sha256']!=old_sha:
                raise ValueError('old cached publisher text missing or wrong original revision')
            required.add(rel)
            if dependents.get(identity,0):
                summarized+=1
                prefix=re.sub(r'[^A-Za-z0-9_.-]+','-',Path(path).name[:64])
                name=prefix+'.'+revision_identity(path,source_sha,old_sha)[:24]+'.summary-tree.json'
                tree='old_summaries/'+name
                progress='old_summaries/progress/'+name+'.progress.json'
                if tree not in archived or progress not in archived:
                    raise ValueError('native summary source revision lacks historical tree/progress backup')
                required.update((tree,progress))
                summary=json.loads(safe_file(bundle_root,tree).read_bytes())
                if (summary.get('original_path')!=path or summary.get('original_sha256')!=source_sha
                        or summary.get('derived_sha256')!=old_sha):
                    raise ValueError('archived old summary tree references a different original revision')
        if required!=set(archived):
            raise ValueError('private backup has unexplained source or stale summary files')
    return {'verified_old_source_and_summary_files':len(originals),
            'verified_old_source_and_summary_bytes':total_bytes,
            'native_rows_verified':{name:actual[name]['rows'] for name in _TABLES},
            'native_original_and_summary_logical_fingerprints_equal':True,
            'summary_revisions_with_tree_and_progress_verified':summarized,
            'all_saved_source_paths_match_exact_old_revision_map':True,
            'snapshot_file_full_sha256_verified':True,
            'snapshot_is_historical_and_not_an_authorized_live_cutover':True,
            'all_verification_reads_from_private_sealed_bundle':True}


def rehearse_private_file_database_restore(*,bundle_root:Path,staged_new_text:Path,
                    original_path:str,private_root:Path,scenarios_root:Path):
    """Inject two failure points into a PRIVATE text/summary/DB cutover clone.

    This method NEVER touches live cached source text, source-map, native
    vectors or summary trees. Restores old file bytes and the exact DB cohort
    from the independently archived historical recovery bundle.
    """
    import shutil
    import re
    from .source_review_audit import safe_file,sha_file
    from .summary_coverage import revision_identity
    if (not isinstance(original_path,str) or not original_path.endswith('.pdf')
            or original_path.startswith('/') or '\\' in original_path or
            any(p in ('','.','..') for p in original_path.split('/'))):
        raise ValueError('unsafe original publisher identity')
    for directory in (bundle_root,private_root):
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError('private original source recovery directory unsafe')
    if (not scenarios_root.is_absolute() or scenarios_root.exists() or scenarios_root.is_symlink()
            or not scenarios_root.parent.resolve().is_relative_to(private_root.resolve())
            or not staged_new_text.resolve().is_relative_to(private_root.resolve())
            or not bundle_root.resolve().is_relative_to(private_root.resolve())):
        raise ValueError('simulation must remain within new exclusive private audit area')
    bundle_db=safe_file(bundle_root,'original_native_and_source_map_subset.sqlite3')
    old_text=safe_file(bundle_root,'old_derived/'+original_path+'.txt')
    with sqlite3.connect(bundle_db.resolve().as_uri()+'?mode=ro',uri=True) as source_db:
        source_db.execute('PRAGMA query_only=ON')
        revision=source_db.execute('''SELECT original_sha256,old_derived_sha256,
                 new_staged_derived_sha256,native_file_id FROM snapshot_revisions
                 WHERE original_path=?''',(original_path,)).fetchall()
        if len(revision)!=1:raise ValueError('original not in historically frozen cohort')
        original_sha,old_sha,new_sha,identity=revision[0]
        native_chunks=source_db.execute('SELECT COUNT(*) FROM content_chunks WHERE filesystem_entry_id=?',(identity,)).fetchone()[0]
        summary_rows=source_db.execute('SELECT COUNT(*) FROM content_representations WHERE chunk_id IN (SELECT chunk_id FROM content_chunks WHERE filesystem_entry_id=?)',(identity,)).fetchone()[0]
        if not native_chunks or not summary_rows:raise ValueError('isolated failure injection requires source with real native chunks and summaries')
        base_fingerprints={name:table_fingerprint(source_db,name) for name in _TABLES}
        if source_db.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('sealed native recovery subset is not relationally consistent')
    if sha_file(old_text)!=old_sha or sha_file(staged_new_text)!=new_sha:
        raise ValueError('old archive or new private publisher revision bytes changed')
    prefix=re.sub(r'[^A-Za-z0-9_.-]+','-',Path(original_path).name[:64])
    tree_name=prefix+'.'+revision_identity(original_path,original_sha,old_sha)[:24]+'.summary-tree.json'
    old_tree=safe_file(bundle_root,'old_summaries/'+tree_name)
    old_progress=safe_file(bundle_root,'old_summaries/progress/'+tree_name+'.progress.json')
    expected_tree=sha_file(old_tree);expected_progress=sha_file(old_progress)
    scenarios_root.mkdir(mode=0o700)
    output=[]
    for scenario in ('fail_after_file_swap','fail_after_database_commit'):
        root=scenarios_root/scenario
        root.mkdir(mode=0o700)
        files=root/'files';files.mkdir(mode=0o700)
        active=files/'source_current.txt'
        incoming=files/'source_staged_new.txt'
        current_tree=files/'source_current_summary_tree.json'
        current_progress=files/'source_current_summary_progress.json'
        parked_tree=files/'old_revision_tree_parked.json'
        parked_progress=files/'old_revision_progress_parked.json'
        shadow_db=root/'private_isolated_cohort.sqlite3'
        for source,dest in ((old_text,active),(staged_new_text,incoming),(old_tree,current_tree),
                            (old_progress,current_progress),(bundle_db,shadow_db)):
            if dest.exists():raise ValueError('private rehearsal paths are not isolated')
            shutil.copyfile(source,dest)
            os.chmod(dest,0o600)
        if (sha_file(active)!=old_sha or sha_file(incoming)!=new_sha
                or sha_file(current_tree)!=expected_tree or sha_file(current_progress)!=expected_progress):
            raise ValueError('private injected-failure starting source bytes mismatch')
        os.replace(incoming,active)  # PRIVATE ONLY: simulate atomic per-file replacement
        os.replace(current_tree,parked_tree)
        os.replace(current_progress,parked_progress)
        if sha_file(active)!=new_sha:raise ValueError('private source file switch failed')
        mutated=None
        if scenario=='fail_after_database_commit':
            with sqlite3.connect(shadow_db) as db:
                db.execute('PRAGMA foreign_keys=ON')
                db.execute('BEGIN IMMEDIATE')
                db.execute('UPDATE originals SET derived_sha256=? WHERE source_path=?',(new_sha,original_path))
                db.execute('DELETE FROM content_chunks WHERE filesystem_entry_id=?',(identity,))
                if db.execute('PRAGMA foreign_key_check').fetchall():
                    raise ValueError('private simulated native index removal inconsistent')
                remains=db.execute('SELECT COUNT(*) FROM content_chunks WHERE filesystem_entry_id=?',(identity,)).fetchone()[0]
                reps=db.execute('SELECT COUNT(*) FROM content_representations WHERE chunk_id IN (SELECT chunk_id FROM content_chunks WHERE filesystem_entry_id=?)',(identity,)).fetchone()[0]
                new_map=db.execute('SELECT derived_sha256 FROM originals WHERE source_path=?',(original_path,)).fetchone()
                if remains or reps or new_map!=(new_sha,):
                    raise ValueError('private committed replacement not coherent')
                db.commit()
                mutated={'deleted_old_chunks':native_chunks,'deleted_old_representations':summary_rows,
                         'source_map_new_revision_committed_in_scratch':True}
        # Crash simulated at a hazardous boundary. RESTORE using immutable
        # archival bytes, not an unsafe inverse operation on the live FileDB.
        for source,target in ((old_text,active),(old_tree,current_tree),(old_progress,current_progress),
                              (bundle_db,shadow_db)):
            staged=target.with_name(target.name+'.recovery.part')
            if staged.exists():raise ValueError('private recovery target already exists')
            shutil.copyfile(source,staged)
            os.chmod(staged,0o600)
            os.replace(staged,target)
        if (sha_file(active)!=old_sha or sha_file(current_tree)!=expected_tree
                or sha_file(current_progress)!=expected_progress or sha_file(shadow_db)!=sha_file(bundle_db)):
            raise ValueError('original cached PDF text/tree/progress or native snapshot not exactly restored')
        with sqlite3.connect(shadow_db.resolve().as_uri()+'?mode=ro',uri=True) as db:
            if db.execute('PRAGMA foreign_key_check').fetchall():
                raise ValueError('restored original native backup has orphaned summary rows')
            recovered={name:table_fingerprint(db,name) for name in _TABLES}
            if recovered!=base_fingerprints:
                raise ValueError('private failure recovery did not restore all native original/source summary rows')
            row=db.execute('SELECT derived_sha256 FROM originals WHERE source_path=?',(original_path,)).fetchone()
            if row!=(old_sha,):raise ValueError('source-map old publisher derived revision not restored')
        output.append({'injected_failure_point':scenario,
                       'simulated_mutation':mutated,
                       'restored_publisher_old_derivative_sha256':old_sha,
                       'old_source_summary_and_progress_shas_restored':True,
                       'native_original_rows_and_all_summary_rows_logically_restored':True,
                       'private_snapshot_sqlite_binary_restored':True,
                       'no_live_writes_or_service_changes':True})
    return {'original_path':original_path,'historical_original_sha256':original_sha,
            'old_derived_sha256':old_sha,'new_staged_derived_sha256':new_sha,
            'native_file_id':identity,'original_native_chunk_rows':native_chunks,
            'original_native_summary_rows':summary_rows,
            'failure_injection_scenarios':output,
            'actual_live_multifile_atomicity_not_proven':True,
            'only_private_isolated_copies_changed_and_restored':True}
