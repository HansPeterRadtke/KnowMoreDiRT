"""Opt-in document-boundary publisher-revision hold for both KMD workers.

No marker => exact preexisting worker behavior. An owner-only, monotonic
hold-generation marker makes each *upgraded* worker finish its existing
original and stop starting new originals until a strictly later release
marker. This does NOT stop a systemd service, resident model, other manual
writers, or arbitrary native readers and does NOT permit a live cutover.
"""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from typing import NamedTuple

SCHEMA='kmd.publisher_pdf_document_boundary_quiescence.v1'
SCOPE='all_native_originals_for_327_pdf_reindex'
MAX_CONTROL_BYTES=2048

class SourceBoundaryQuiescence(NamedTuple):
    generation:int
    action:str
    digest:str|None


def marker_bytes(action:str,generation:int)->bytes:
    if (action not in ('hold','release') or type(generation) is not int
          or not 1<=generation<=1_000_000_000):
        raise ValueError('invalid bounded publisher source-boundary control generation')
    return (json.dumps({'schema':SCHEMA,'scope':SCOPE,'generation':generation,
                        'action':action},sort_keys=True,separators=(',',':'))+'\n').encode('utf8')


def read_control(path:Path,*,prior_generation:int=0,prior_digest:str|None=None):
    """Read only, reject replacement/replay/deletion after observed generation."""
    if (not isinstance(path,Path) or not path.is_absolute()
            or type(prior_generation) is not int or prior_generation<0
            or (prior_generation==0)!=(prior_digest is None)
            or prior_digest is not None and (not isinstance(prior_digest,str) or len(prior_digest)!=64)):
        raise ValueError('invalid prior source-boundary control state')
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError('unsafe source-boundary parent directory')
    if not path.exists():
        if prior_generation or path.is_symlink():
            raise ValueError('previous publisher hold manifest removed or symlinked')
        return SourceBoundaryQuiescence(0,'release',None)
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        before=os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or not 30<=before.st_size<=MAX_CONTROL_BYTES
                or stat.S_IMODE(before.st_mode)&0o077 or before.st_uid!=os.geteuid()):
            raise ValueError('publisher control must be regular and owner-only')
        raw=os.read(fd,MAX_CONTROL_BYTES+1)
        after=os.fstat(fd)
        if (len(raw)!=before.st_size or not os.path.samestat(before,after)
                or before.st_mtime_ns!=after.st_mtime_ns
                or before.st_ctime_ns!=after.st_ctime_ns):
            raise ValueError('publisher control modified during read')
    finally:
        os.close(fd)
    digest=hashlib.sha256(raw).hexdigest()
    try:
        data=json.loads(raw.decode('utf8'))
    except (ValueError,UnicodeError) as error:
        raise ValueError('publisher control is not strict UTF-8 JSON') from error
    if (not isinstance(data,dict)
            or set(data)!={'schema','scope','generation','action'}
            or data['schema']!=SCHEMA or data['scope']!=SCOPE
            or data['action'] not in ('hold','release')
            or type(data['generation']) is not int
            or not 1<=data['generation']<=1_000_000_000):
        raise ValueError('unexpected publisher generation or control schema')
    if (data['generation']<prior_generation
            or data['generation']==prior_generation
              and prior_digest is not None and digest!=prior_digest):
        raise ValueError('publisher hold release generation replay or same-generation rewrite')
    return SourceBoundaryQuiescence(data['generation'],data['action'],digest)


def hold_at_source_boundary(*,marker:Path,worker_label:str,worker_pid:int,
               prior_generation:int,prior_digest:str|None,update_status,
               should_stop,wait=time.sleep,heartbeat_seconds:float=5.0):
    """No new source begins until a later generation authorizes release.

    Caller invokes ONLY at an original boundary after the old import
    subprocess has exited. An actual hold acknowledgement is written ONLY
    by the resident worker, never by an operator or separate report script.
    """
    if (worker_label not in ('coder','q4') or type(worker_pid) is not int
            or worker_pid<1 or not callable(update_status) or not callable(should_stop)
            or not callable(wait) or not 0<heartbeat_seconds<=30):
        raise ValueError('untrusted source boundary worker identity or callbacks')
    control=read_control(marker,prior_generation=prior_generation,prior_digest=prior_digest)
    while control.action=='hold':
        update_status(phase='publisher_revision_quiesced',current_original=None,
             quiesce_protocol=SCHEMA,quiesce_scope=SCOPE,
             quiesce_generation=control.generation,
             quiesce_manifest_sha256=control.digest,
             quiesce_worker_pid=worker_pid,
             quiesce_ack_utc=datetime.now(timezone.utc).isoformat())
        if should_stop():
            return control
        wait(heartbeat_seconds)
        if should_stop():
            # A stop requested during the heartbeat should not re-ack or
            # re-read and must never start a new original.
            return control
        control=read_control(marker,prior_generation=control.generation,
                              prior_digest=control.digest)
    if control.generation:
        update_status(phase='publisher_revision_released',current_original=None,
             quiesce_protocol=SCHEMA,quiesce_scope=SCOPE,
             quiesce_generation=control.generation,
             quiesce_manifest_sha256=control.digest,
             quiesce_worker_pid=worker_pid,
             quiesce_ack_utc=datetime.now(timezone.utc).isoformat())
    return control


def read_only_worker_acknowledgements(*,control:SourceBoundaryQuiescence,
              states:dict[str,dict],current_pid,current_cmdline,
              now_utc:datetime|None=None):
    """Operator validation, not release of source-map/native writer locks."""
    if control.action!='hold' or not control.generation or not control.digest:
        raise ValueError('publisher revision hold is not active')
    if not isinstance(states,dict) or set(states)!={'coder','q4'}:
        raise ValueError('both independent source worker statuses required')
    now_utc=now_utc or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        raise ValueError('must use UTC aware time for live acknowledgement')
    units={'coder':'knowledge-library-hierarchical-summary.service',
           'q4':'knowledge-library-hierarchical-summary-q4.service'}
    verified={}
    for worker,state in states.items():
        if not isinstance(state,dict):raise ValueError('missing live worker status')
        pid=current_pid(units[worker])
        if type(pid) is not int or pid<=1:
            raise ValueError('unverified source worker main PID')
        cmd=current_cmdline(pid)
        if (state.get('worker_pid')!=pid
                or state.get('quiesce_worker_pid')!=pid
                or b'raspi_hierarchical_worker.py' not in cmd
                or b'--shard-count\x004\x00' not in cmd
                or (worker=='q4')!=(b'--worker-label\x00q4\x00' in cmd)):
            raise ValueError('unverified source worker process identity')
        if (state.get('quiesce_protocol')!=SCHEMA
                or state.get('quiesce_scope')!=SCOPE
                or state.get('quiesce_generation')!=control.generation
                or state.get('quiesce_manifest_sha256')!=control.digest
                or state.get('phase')!='publisher_revision_quiesced'
                or state.get('current_original') is not None
                or state.get('quiesce_runtime_started_utc')!=state.get('started_utc')):
            raise ValueError('source worker has not acknowledged a quiesced original boundary')
        try:
            heartbeat=datetime.fromisoformat(state['updated_utc'])
            recorded=datetime.fromisoformat(state['quiesce_ack_utc'])
        except (KeyError,TypeError,ValueError) as exc:
            raise ValueError('missing independent original-boundary worker heartbeat') from exc
        if (heartbeat.tzinfo is None or recorded.tzinfo is None
                or heartbeat>now_utc or recorded>now_utc
                or (now_utc-heartbeat).total_seconds()>15
                or (now_utc-recorded).total_seconds()>15):
            raise ValueError('stale or future source-boundary pause acknowledgement')
        verified[worker]={'pid':pid,'generation':control.generation,
                          'last_heartbeat_utc':state['updated_utc'],
                          'at_document_boundary':True}
    return {'schema':'kmd.read_only_publisher_327_source_quiescence_ack.v1',
         'workers':verified,'both_resident_summary_workers_paused_at_source_boundaries':True,
         'other_source_map_and_native_writers_and_readers_not_locked':True,
         'not_an_authorized_native_pdf_cutover':True,
         'safe_to_cutover_live':False}
