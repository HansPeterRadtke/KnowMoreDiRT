"""PRIVATE-ONLY cooperative legacy reader exclusion for immutable generation trials.

This is NOT a production authorization or lease. Unknown/unregistered clients,
live KMD writers, existing summary workers and legacy readers are NOT covered.
The only permitted root is an isolated private source-generation fixture.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Callable

from .private_source_generation import (
    _open_owner_only_pointer_lock, _read_pointer, _validate_private_root,
    publish_private_only, verify_generation,
)

SCHEMA='kmd.private_cooperative_legacy_read_admission_seal.v1'
LOCK_NAME='.private-legacy-reader-admission.lock'
SEAL_NAME='private-legacy-admission-sealed.json'
MAX_SEAL_BYTES=1024


def _owner_only_file_descriptor(path:Path,flags:int):
    mode=0o600
    descriptor=os.open(path,flags|os.O_CLOEXEC|os.O_NOFOLLOW,mode)
    try:
        info=os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid!=os.geteuid()
                or stat.S_IMODE(info.st_mode)&0o077):
            raise ValueError('private cooperative admission controls must be owner-only regular files')
        return os.fdopen(descriptor,'r+b' if flags&os.O_RDWR else 'rb',buffering=0)
    except BaseException:
        os.close(descriptor)
        raise


def _open_admission_lock(base:Path):
    return _owner_only_file_descriptor(base/LOCK_NAME,os.O_RDWR|os.O_CREAT)


@contextmanager
def _private_directory_admission(base:Path,*,exclusive:bool):
    """Pin the private generation-root directory inode during admission.

    A single lock filename could be replaced while a client is holding the
    old inode. A second flock on this same directory prevents that split-lock
    bypass for cooperating Linux clients, without touching live source data.
    """
    fd=os.open(base,os.O_RDONLY|os.O_NOFOLLOW|os.O_DIRECTORY|os.O_CLOEXEC)
    locked=False
    try:
        before=os.fstat(fd)
        current=os.stat(base,follow_symlinks=False)
        if (not stat.S_ISDIR(before.st_mode) or before.st_uid!=os.geteuid()
                or not os.path.samestat(before,current)):
            raise ValueError('private directory admission root is replaced or not owner-controlled')
        mode=fcntl.LOCK_EX|fcntl.LOCK_NB if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(fd,mode)
            locked=True
        except BlockingIOError as exc:
            raise ValueError('cooperating legacy reader or writer remains active; private directory admission denied') from exc
        if not os.path.samestat(before,os.stat(base,follow_symlinks=False)):
            raise ValueError('private directory admission root changed while obtaining its lock')
        yield
    finally:
        if locked:fcntl.flock(fd,fcntl.LOCK_UN)
        os.close(fd)


def read_private_legacy_seal(base:Path,allowed_root:Path):
    """Inspect the durable private legacy-admission tombstone, never live state."""
    _validate_private_root(base,allowed_root)
    path=base/SEAL_NAME
    if not path.exists():
        if path.is_symlink():raise ValueError('private admission seal is symlinked')
        return None
    with _owner_only_file_descriptor(path,os.O_RDONLY) as src:
        before=os.fstat(src.fileno())
        if not 40<=before.st_size<=MAX_SEAL_BYTES:
            raise ValueError('private legacy admission seal has invalid bounded size')
        raw=src.read(MAX_SEAL_BYTES+1)
        after_fd=os.fstat(src.fileno())
    after=path.stat(follow_symlinks=False)
    if (len(raw)!=before.st_size or not os.path.samestat(before,after_fd)
            or not os.path.samestat(before,after)
            or before.st_mtime_ns!=after.st_mtime_ns
            or before.st_ctime_ns!=after.st_ctime_ns):
        raise ValueError('private legacy admission seal changed during read')
    try:info=json.loads(raw.decode('utf8','strict'))
    except (ValueError,UnicodeError) as exc:
        raise ValueError('private legacy admission seal is not valid JSON') from exc
    if (not isinstance(info,dict)
            or set(info)!={'schema','sealed_for_generation','manifest_sha256'}
            or info['schema']!=SCHEMA
            or type(info['sealed_for_generation']) is not int
            or not 1<=info['sealed_for_generation']<=1_000_000
            or not isinstance(info['manifest_sha256'],str)
            or len(info['manifest_sha256'])!=64
            or any(x not in '0123456789abcdef' for x in info['manifest_sha256'])):
        raise ValueError('private legacy admission seal is invalid or ambiguous')
    return info


@dataclass(frozen=True)
class AdmittedPrivateLegacyClient:
    """Cooperating old reader OR writer holds a shared private admission lock."""
    kind:str
    safe_to_cutover_live:bool=False
    unregistered_clients_excluded:bool=False


@contextmanager
def _admit_private_legacy_client(base:Path,allowed_root:Path,kind:str):
    if kind not in ('reader','writer'):
        raise ValueError('private legacy source client kind must be reader or writer')
    _validate_private_root(base,allowed_root)
    # Lock directory first, then lock file, then pointer. Publisher uses the
    # same order, avoiding lock inversion while protecting lock-file swaps.
    with _private_directory_admission(base,exclusive=False):
        with _open_admission_lock(base) as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_SH)
            try:
                if read_private_legacy_seal(base,allowed_root) is not None:
                    raise ValueError('private legacy reader admission is sealed')
                # A removed tombstone must not reopen older clients after a
                # published pointer. This excludes ONLY cooperating fixtures.
                with _open_owner_only_pointer_lock(base) as pointer_lock:
                    fcntl.flock(pointer_lock.fileno(),fcntl.LOCK_SH)
                    if _read_pointer(base) is not None:
                        raise ValueError('private legacy reader admission seal missing after published generation')
                yield AdmittedPrivateLegacyClient(kind=kind)
            finally:
                fcntl.flock(lock.fileno(),fcntl.LOCK_UN)


@contextmanager
def admit_private_legacy_reader(base:Path,allowed_root:Path):
    """An enrolled old-style PRIVATE reader holds SH for its entire operation."""
    with _admit_private_legacy_client(base,allowed_root,'reader') as client:
        yield client


@contextmanager
def admit_private_legacy_writer(base:Path,allowed_root:Path):
    """An enrolled old-style PRIVATE writer holds SH for its entire operation.

    This does not cover any production extractor, native worker or manual SQL
    client. These fixtures never write to the live Knowledge Library.
    """
    with _admit_private_legacy_client(base,allowed_root,'writer') as client:
        yield client


def _durably_seal(base:Path,generation:int,manifest_sha256:str):
    blob=(json.dumps({'schema':SCHEMA,'sealed_for_generation':generation,
          'manifest_sha256':manifest_sha256},sort_keys=True,separators=(',',':'))+'\n').encode()
    path=base/SEAL_NAME
    # Use an exclusively created unpredictable temporary name. A simulated
    # process kill before rename can leave another worker's older temporary
    # file behind; PID reuse must not block a later private-only recovery.
    fd,temp_name=tempfile.mkstemp(prefix='.private-legacy-seal-new-',dir=base)
    temp=Path(temp_name)
    try:
        os.fchmod(fd,0o600)
        with os.fdopen(fd,'wb') as stream:
            stream.write(blob);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
        directory=os.open(base,os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        temp.unlink(missing_ok=True)


def publish_private_after_legacy_drain(base:Path,allowed_root:Path,generation:int,*,
                                        after_seal:Callable[[],None]|None=None):
    """Fail fast with active cooperative OLD readers; close admissions forever.

    A crash after sealing but before pointer replacement intentionally leaves
    legacy reads DENIED and the old pointer unchanged. Only a later successful
    private-only generation publish can advance the pointer. No live reader,
    other service, or unregistered writer is enrolled by this function.
    """
    _validate_private_root(base,allowed_root)
    # Invalid staged source revisions must not unnecessarily seal admissions.
    verify_generation(base,allowed_root,generation)
    with _private_directory_admission(base,exclusive=True):
        with _open_admission_lock(base) as lock:
            try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('cooperating legacy reader or writer remains active; private generation switch denied') from exc
            try:
                # Validate next ordinal under the pointer's SH lock. The existing
                # private publisher rechecks with EX before its own atomic rename.
                with _open_owner_only_pointer_lock(base) as pointer_lock:
                    fcntl.flock(pointer_lock.fileno(),fcntl.LOCK_SH)
                    current=_read_pointer(base)
                    expected=1 if current is None else current['generation']+1
                    if generation!=expected:
                        raise ValueError('private generation must increment exactly one, including first private generation')
                seal=read_private_legacy_seal(base,allowed_root)
                if current is not None and seal is None:
                    raise ValueError('private legacy reader admission seal missing after prior publication')
                if seal is not None:
                    if generation<seal['sealed_for_generation']:
                        raise ValueError('private legacy admission seal forbids generation replay')
                    sealed_proof=verify_generation(base,allowed_root,seal['sealed_for_generation'])
                    if sealed_proof['manifest_sha256']!=seal['manifest_sha256']:
                        raise ValueError('private legacy admission seal manifest no longer matches a verified generation')
                # The entire private manifest and native source reindex QA are
                # verified while the cooperative old-reader gate is exclusive.
                proof=verify_generation(base,allowed_root,generation)
                if seal is None:
                    _durably_seal(base,generation,proof['manifest_sha256'])
                if after_seal:after_seal()  # test-only simulated crash; seal stays.
                result=publish_private_only(base,allowed_root,generation)
                return {**result,
                    'cooperating_legacy_reader_admissions_sealed':True,
                    'unregistered_clients_excluded':False,
                    'all_live_KMD_readers_admitted':False,
                    'safe_to_cutover_live':False}
            finally:
                fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
