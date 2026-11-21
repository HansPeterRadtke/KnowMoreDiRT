#!/usr/bin/env python3
"""Read-only or explicitly applied two-phase source-disjoint worker assignment.

This program NEVER stops a process/service, restarts or evicts a model, edits
systemd or modifies original books, FileDB or source-map. Both workers must
already have voluntarily restarted into the tested runtime protocol before
ANY manifest write is permitted. The default is dry-run.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from file_system_catalog.source_shard_handoff_operator import compute_step

ROOT=Path('/data/var/knowmoredirt/knowledge-library')
MANIFEST=ROOT/'hierarchical-shard-handoff.json'
STATUS={'coder':ROOT/'hierarchical-worker-status.json',
        'q4':ROOT/'hierarchical-worker-q4-status.json'}
LOCK=ROOT/'.hierarchical-shard-handoff-operator.lock'
JOURNAL=ROOT/'hierarchical-shard-handoff-operator.jsonl'
TARGET=None  # Must be selected explicitly by the invoking owner operator.


def live_pid(unit: str) -> int:
    response=subprocess.run(['systemctl','show',unit,'-p','MainPID','--value'],
                            capture_output=True,text=True,check=True,timeout=5)
    try:
        pid=int(response.stdout.strip())
    except ValueError as error:
        raise ValueError('invalid installed source worker PID') from error
    if pid<=1:
        raise ValueError('installed source worker not running')
    return pid


def proc_command(pid: int) -> bytes:
    if type(pid) is not int or not 1<pid<2**31:
        raise ValueError('invalid live source worker process identity')
    p=Path('/proc')/str(pid)/'cmdline'
    with p.open('rb') as stream:
        return stream.read(8192)


def read_states() -> dict:
    out={}
    for label,path in STATUS.items():
        if path.is_symlink() or not path.is_file() or path.stat().st_size>64*1024:
            raise ValueError('missing, symlinked or oversized worker source status')
        out[label]=json.loads(path.read_text('utf8'))
    return out


def verify() -> dict:
    return compute_step(manifest_path=MANIFEST,worker_statuses=read_states(),
                        target=TARGET,live_pid=live_pid,proc_command=proc_command)


def safe_apply() -> dict:
    if not ROOT.is_dir() or ROOT.is_symlink():
        raise ValueError('expected private source runtime directory missing')
    descriptor=os.open(LOCK,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
    try:
        old=os.fstat(descriptor)
        if (not stat.S_ISREG(old.st_mode) or old.st_uid!=os.geteuid()
                or stat.S_IMODE(old.st_mode)&0o077):
            raise ValueError('unsafe source-ownership handoff lock')
        fcntl.flock(descriptor,fcntl.LOCK_EX)
        # All live worker/source checks are REPEATED inside the exclusive
        # single-operator lock before any atomic runtime file replacement.
        plan=verify()
        if not plan['requires_change']:
            return {**plan,'action':'no_change'}
        content=plan.pop('manifest_bytes_to_write_after_second_preflight')
        if not isinstance(content,bytes):
            raise ValueError('missing staged allocation content')
        digest=hashlib.sha256(content).hexdigest()
        if plan['source_next_sha256']!=digest:
            raise ValueError('staged source allocation digest mismatch')
        existing=MANIFEST.read_bytes() if MANIFEST.is_file() and not MANIFEST.is_symlink() else None
        if (hashlib.sha256(existing).hexdigest() if existing is not None else None)!=plan['source_current_sha256']:
            raise ValueError('source allocation manifest changed after verification')
        fd,tmp=tempfile.mkstemp(prefix='.new-shard-handoff-',dir=ROOT)
        try:
            os.fchmod(fd,0o600)
            with os.fdopen(fd,'wb') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp,MANIFEST)
            directory_fd=os.open(ROOT,os.O_RDONLY|os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            Path(tmp).unlink(missing_ok=True)
        # Owner-only, append-only audit of this material allocation change.
        # The installed services are not sent any signal by the operator.
        event={
            'schema':'kmd.source_shard_allocation_change_event.v1',
            'at_utc':datetime.now(timezone.utc).isoformat(),
            'old_stage':plan['source_current_stage'],
            'old_generation':plan['source_current_generation'],
            'old_manifest_sha256':plan['source_current_sha256'],
            'new_stage':plan['source_next_stage'],
            'new_generation':plan['source_next_generation'],
            'new_manifest_sha256':digest,
            'both_live_acknowledgements':plan['worker_current_proofs'],
            'any_service_signalled':False,
        }
        event_fd=os.open(JOURNAL,os.O_WRONLY|os.O_APPEND|os.O_CREAT|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
        try:
            event_stat=os.fstat(event_fd)
            if (not stat.S_ISREG(event_stat.st_mode) or event_stat.st_uid!=os.geteuid()
                    or stat.S_IMODE(event_stat.st_mode)&0o077):
                raise ValueError('unsafe source allocation event journal')
            raw=(json.dumps(event,sort_keys=True,ensure_ascii=False)+'\n').encode()
            if os.write(event_fd,raw)!=len(raw):
                raise OSError('short source allocation journal write')
            os.fsync(event_fd)
        finally:
            os.close(event_fd)
        return {**plan,'action':'owner_only_manifest_stage_persisted'}
    finally:
        os.close(descriptor)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',required=True,choices=('balanced','baseline'))
    parser.add_argument('--apply',action='store_true',help='Only after BOTH live workers acknowledge current source-boundary allocation; never stops services')
    args=parser.parse_args()
    global TARGET
    TARGET=args.target
    if args.apply:
        result=safe_apply()
    else:
        result=verify()
        result.pop('manifest_bytes_to_write_after_second_preflight',None)
        result['action']='dry_run_proposal_no_files_changed'
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
