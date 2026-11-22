"""Atomic, *private-only* source-generation pointer and immutable reader pins.

Separate SQLite transactions and file renames cannot be a single live atomic
cutover. This proves a proposed new-readers-only immutable generation pattern
in an isolated directory, with a single pointer switch and pinned old views.
It has NO production entrypoint, cannot touch live KMD paths, cannot stop
workers, and explicitly does not authorize a real deployment.
"""
from __future__ import annotations
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
from typing import Callable

from .source_review_audit import safe_file, sha_file
from .source_page_reindex_release_gate import _summary_tree
from .summary_coverage import current_completed_originals,revision_identity

SCHEMA='kmd.private_original_pdf_generation.v1'
POINTER_SCHEMA='kmd.private_original_pdf_generation_pointer.v1'
GATE_SCHEMA='kmd.corrected_pdf_native_source_revision_release_fidelity_gate.v1'
SHA=re.compile(r'^[a-f0-9]{64}$')
ROLES=frozenset(('native_sqlite','source_map_sqlite','corrected_derived_text','recursive_summary_tree'))
MANDATORY=frozenset(('native_sqlite','source_map_sqlite','corrected_derived_text','recursive_summary_tree'))
MAX_MANIFEST_BYTES=1024*1024
MAX_ASSETS=2048
LIVE_KMD_ROOT=Path('/data/var/knowmoredirt/knowledge-library')
PRIVATE_AUDIT_ROOT=LIVE_KMD_ROOT/'search-audit'
OTHER_ORIGINALS_ROOT=Path('/data/var/Knowledge Library')


def _sha(blob:bytes)->str:
    return hashlib.sha256(blob).hexdigest()


def _generation_path(base:Path,generation:int)->Path:
    if type(generation) is not int or not 1<=generation<=1_000_000:
        raise ValueError('invalid immutable private generation number')
    return base/'generations'/f'gen-{generation:08d}'


def _validate_private_root(base:Path,allowed_root:Path)->None:
    if (not isinstance(base,Path) or not isinstance(allowed_root,Path)
            or not base.is_absolute() or not allowed_root.is_absolute()
            or base.is_symlink() or allowed_root.is_symlink()
            or not base.is_dir() or not allowed_root.is_dir()
            or not base.resolve(strict=True).is_relative_to(allowed_root.resolve(strict=True))
            or base.resolve(strict=True)==allowed_root.resolve(strict=True)):
        raise ValueError('private atomic source-generation root must be under isolated audit storage')
    resolved=base.resolve(strict=True)
    if ((resolved.is_relative_to(LIVE_KMD_ROOT)
            and not resolved.is_relative_to(PRIVATE_AUDIT_ROOT))
            or resolved.is_relative_to(OTHER_ORIGINALS_ROOT)
            or (str(resolved).startswith('/data/') and
                not resolved.is_relative_to(PRIVATE_AUDIT_ROOT))):
        raise ValueError('private generation switch refuses every production/original-data root')


def _read_json_regular(path:Path,limit:int):
    if path.is_symlink() or not path.is_file() or path.stat().st_size>limit:
        raise ValueError('untrusted or unbounded private generation metadata')
    blob=path.read_bytes()
    if len(blob)>limit:
        raise ValueError('unbounded private generation metadata bytes')
    result=json.loads(blob.decode('utf8','strict'))
    if not isinstance(result,dict):
        raise ValueError('invalid private JSON object')
    return result,_sha(blob)


def _sql_readonly_check(path:Path):
    if path.with_name(path.name+'-wal').exists() or path.with_name(path.name+'-shm').exists():
        raise ValueError('private SQLite has mutable WAL sidecars')
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=5) as db:
        db.execute('PRAGMA query_only=ON')
        if db.execute('PRAGMA integrity_check').fetchone() != ('ok',):
            raise ValueError('private generation SQLite fails integrity_check')


def verify_generation(base:Path,allowed_root:Path,generation:int):
    """Re-hash *all* assets every time before a version is used as evidence."""
    _validate_private_root(base,allowed_root)
    folder=_generation_path(base,generation)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError('immutable private generation directory is missing or symlinked')
    manifest,manifest_sha=_read_json_regular(folder/'manifest.json',MAX_MANIFEST_BYTES)
    if (set(manifest)!={'schema','generation','cohort_originals','native_gate','assets'}
            or manifest['schema']!=SCHEMA or manifest['generation']!=generation
            or type(manifest['cohort_originals']) is not int
            or not 1<=manifest['cohort_originals']<=327
            or not isinstance(manifest['assets'],list)
            or not 4<=len(manifest['assets'])<=MAX_ASSETS):
        raise ValueError('unrecognized or incomplete immutable source-generation manifest')
    paths=set();kinds={kind:[] for kind in ROLES}
    for item in manifest['assets']:
        if not isinstance(item,dict) or set(item)!={'kind','path','sha256','bytes'}:
            raise ValueError('untrusted private generation asset shape')
        role=item['kind'];relative=item['path'];digest=item['sha256'];size=item['bytes']
        if (role not in ROLES or not isinstance(relative,str) or relative in paths
                or not isinstance(digest,str) or SHA.fullmatch(digest) is None
                or type(size) is not int or not 0<size<=4*1024*1024*1024):
            raise ValueError('invalid or duplicate private generation asset')
        if role in ('native_sqlite','source_map_sqlite') and not relative.endswith('.sqlite3'):
            raise ValueError('private native/source-map roles must be SQLite files')
        if role=='corrected_derived_text' and not relative.endswith('.txt'):
            raise ValueError('private original derivative must be UTF-8 text')
        if role=='recursive_summary_tree' and not relative.endswith('.summary-tree.json'):
            raise ValueError('private recursive summaries must be tree JSON')
        path=safe_file(folder,relative)
        if path.stat().st_size!=size or sha_file(path)!=digest:
            raise ValueError('immutable private source-generation asset bytes changed')
        if role in ('native_sqlite','source_map_sqlite'):_sql_readonly_check(path)
        paths.add(relative);kinds[role].append(path)
    if (set(kinds)!=MANDATORY or len(kinds['native_sqlite'])!=1
           or len(kinds['source_map_sqlite'])!=1
           or len(kinds['corrected_derived_text'])!=manifest['cohort_originals']
           or len(kinds['recursive_summary_tree'])!=manifest['cohort_originals']):
        raise ValueError('incomplete all-source native, source map, derivative or summary file cohort')
    qa=manifest['native_gate']
    if not isinstance(qa,dict) or set(qa)!={'path','sha256'} or not isinstance(qa['sha256'],str):
        raise ValueError('missing exact private all-source native QA evidence')
    qa_file=safe_file(folder,qa['path'])
    if sha_file(qa_file)!=qa['sha256']:
        raise ValueError('private native reindex QA receipt changed')
    receipt,_=_read_json_regular(qa_file,MAX_MANIFEST_BYTES)
    if (receipt.get('schema')!=GATE_SCHEMA
            or receipt.get('cohort_expected_originals')!=manifest['cohort_originals']
            or receipt.get('originals_with_native_full_source_roots_and_vectors')!=manifest['cohort_originals']
            or receipt.get('all_corrected_revision_search_evidence_complete') is not True
            or receipt.get('safe_to_cutover_live') is not False):
        raise ValueError('new private original-revision index is not completely rebuilt and source-verified')
    # IMPORTANT: a positive historical QA file is not enough: explicitly
    # bind it to the TWO SQLite bytes and exact candidate inventory.
    expected={'native_sqlite_sha256':sha_file(kinds['native_sqlite'][0]),
              'source_map_sqlite_sha256':sha_file(kinds['source_map_sqlite'][0])}
    if any(receipt.get(k)!=v for k,v in expected.items()):
        raise ValueError('native QA receipt is not bound to the exact pair of isolated SQLite files')
    source_map=kinds['source_map_sqlite'][0]
    native_db=kinds['native_sqlite'][0]
    with sqlite3.connect(source_map.resolve().as_uri()+'?mode=ro',uri=True) as sm, \
         sqlite3.connect(native_db.resolve().as_uri()+'?mode=ro',uri=True) as native:
        sm.execute('PRAGMA query_only=ON');native.execute('PRAGMA query_only=ON')
        records=sm.execute('''SELECT source_path,source_sha256,derived_path,
               derived_sha256,original_kind,status,text_characters
               FROM originals ORDER BY source_path''').fetchall()
        if len(records)!=manifest['cohort_originals']:
            raise ValueError('isolated source map contains incomplete or unexpected originals')
        sources={};all_derived=set();all_trees=set()
        tree_root=folder/'summaries'
        if tree_root.is_symlink() or not tree_root.is_dir():
            raise ValueError('missing private recursive summary generation folder')
        for path,original_sha,derivative_path,derived_sha,kind,status,characters in records:
            if (not isinstance(path,str) or path.startswith('/') or '\\' in path
                    or any(z in ('','.','..') for z in path.split('/'))
                    or not isinstance(original_sha,str) or SHA.fullmatch(original_sha) is None
                    or not isinstance(derived_sha,str) or SHA.fullmatch(derived_sha) is None
                    or derivative_path!=path+'.txt' or kind!='pdf' or status!='complete'
                    or type(characters) is not int or characters<1):
                raise ValueError('unverified corrected source-map PDF identity or status')
            derived_relative='text/'+derivative_path
            source_text=safe_file(folder,derived_relative).read_text('utf8','strict')
            if not source_text or _sha(source_text.encode())!=derived_sha:
                raise ValueError('publisher derivative content differs from current source map')
            all_derived.add(derived_relative)
            original_name=Path(path).name[:64]
            import re as _re
            original_name=_re.sub(r'[^A-Za-z0-9_.-]+','-',original_name)
            tree_rel='summaries/'+original_name+'.'+revision_identity(path,original_sha,derived_sha)[:24]+'.summary-tree.json'
            all_trees.add(tree_rel)
            _summary_tree(root=tree_root,path=path,original_sha=original_sha,
                          derived_sha=derived_sha,expected_text_len=len(source_text))
            sources[path]=(original_sha,derived_sha,characters)
        if (set(a['path'] for a in manifest['assets'] if a['kind']=='corrected_derived_text')!=all_derived
                or set(a['path'] for a in manifest['assets'] if a['kind']=='recursive_summary_tree')!=all_trees):
            raise ValueError('generation source derivatives or tree files are not exactly the pinned catalog originals')
        completed=current_completed_originals(native,sources)
        if len(completed)!=manifest['cohort_originals']:
            raise ValueError('current native FileDB summary roots do not cover every corrected original')
    return {'generation':generation,'manifest_sha256':manifest_sha,
            'assets':tuple((v['kind'],v['path'],v['sha256']) for v in manifest['assets']),
            'candidate_files_byte_verified':len(manifest['assets']),
            'private_generation_complete':True,
            'safe_to_cutover_live':False}


def _read_pointer(base:Path):
    path=base/'active-private-generation.json'
    if not path.exists():
        if path.is_symlink():raise ValueError('private generation marker is a symlink')
        return None
    pointer,_=_read_json_regular(path,2048)
    if (set(pointer)!={'schema','generation','manifest_sha256'}
            or pointer['schema']!=POINTER_SCHEMA
            or type(pointer['generation']) is not int
            or not isinstance(pointer['manifest_sha256'],str)
            or SHA.fullmatch(pointer['manifest_sha256']) is None):
        raise ValueError('untrusted private generation marker')
    return pointer


@dataclass(frozen=True)
class PinnedPrivateGeneration:
    generation:int
    manifest_sha256:str
    original_root:Path
    assets:tuple

    def verify_bytes(self,allowed_root:Path):
        proof=verify_generation(self.original_root,allowed_root,self.generation)
        if proof['manifest_sha256']!=self.manifest_sha256:
            raise ValueError('pinned source generation was modified after reader selection')
        return proof


def _open_owner_only_pointer_lock(base:Path):
    lock=base/'.private-generation-pointer.lock'
    fd=os.open(lock,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
    try:
        st=os.fstat(fd)
        if (not stat.S_ISREG(st.st_mode) or st.st_uid!=os.geteuid()
                or stat.S_IMODE(st.st_mode)&0o077):
            raise ValueError('private generation pointer lock must be owned and inaccessible to other users')
        return os.fdopen(fd,'r+')
    except BaseException:
        os.close(fd)
        raise


def read_pinned(base:Path,allowed_root:Path):
    _validate_private_root(base,allowed_root)
    with _open_owner_only_pointer_lock(base) as fd:
        fcntl.flock(fd.fileno(),fcntl.LOCK_SH)
        pointer=_read_pointer(base)
        if pointer is None:raise ValueError('no private source generation has been published')
        proof=verify_generation(base,allowed_root,pointer['generation'])
        if proof['manifest_sha256']!=pointer['manifest_sha256']:
            raise ValueError('private marker points to a modified source-generation manifest')
    return PinnedPrivateGeneration(pointer['generation'],pointer['manifest_sha256'],
                                   base,proof['assets'])


def publish_private_only(base:Path,allowed_root:Path,generation:int,*,
                         before_pointer_replace:Callable[[],None]|None=None,
                         after_pointer_replace:Callable[[],None]|None=None):
    """Privately switch one pointer. The complete previous generation survives.

    An optional callback exists only for injecting a crash into *isolated*
    tests; this never supports a live original/source-map/native migration.
    """
    _validate_private_root(base,allowed_root)
    proof=verify_generation(base,allowed_root,generation)
    with _open_owner_only_pointer_lock(base) as fd:
        fcntl.flock(fd.fileno(),fcntl.LOCK_EX)
        previous=_read_pointer(base)
        if previous and generation!=previous['generation']+1:
            raise ValueError('private generation must increment exactly one to prevent replay')
        if previous:
            old=verify_generation(base,allowed_root,previous['generation'])
            if old['manifest_sha256']!=previous['manifest_sha256']:
                raise ValueError('old privately published original generation was modified')
        elif generation!=1:
            raise ValueError('first private generation must have number one')
        # Reverify new generation while holding exclusive marker lock.
        repeated=verify_generation(base,allowed_root,generation)
        if repeated['manifest_sha256']!=proof['manifest_sha256']:
            raise ValueError('new privately staged generation changed during publication')
        blob=(json.dumps({'schema':POINTER_SCHEMA,'generation':generation,
              'manifest_sha256':proof['manifest_sha256']},sort_keys=True,separators=(',',':'))+'\n').encode()
        marker=base/'active-private-generation.json'
        temp=base/('.private-generation-pointer-new-'+str(os.getpid()))
        flags=os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW
        temp_fd=os.open(temp,flags,0o600)
        try:
            with os.fdopen(temp_fd,'wb') as out:
                out.write(blob);out.flush();os.fsync(out.fileno())
            if before_pointer_replace:
                before_pointer_replace()
            final_preflight=verify_generation(base,allowed_root,generation)
            if final_preflight['manifest_sha256']!=proof['manifest_sha256']:
                raise ValueError('new private generation changed while preparing atomic pointer')
            os.replace(temp,marker)
            if after_pointer_replace:
                after_pointer_replace()
            dirfd=os.open(base,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(dirfd)
            finally:os.close(dirfd)
        finally:
            temp.unlink(missing_ok=True)
    return {'private_pointer_generation':generation,
            'private_manifest_sha256':proof['manifest_sha256'],
            'previous_private_generation':previous['generation'] if previous else None,
            'older_generations_preserved_without_deletion':True,
            'readers_required_to_pin_generation_before_opening_assets':True,
            'current_live_readers_are_not_generation_aware':True,
            'safe_to_cutover_live':False}
