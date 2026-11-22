#!/usr/bin/env python3
"""READ ONLY: verify both resident KMD workers acknowledged a document-boundary hold.

Never creates a marker, signals a process, stops a model, acquires a live
writer lease, or claims native source-map/PDF-repair cutover is safe. Optional
control marker is managed ONLY by a separately authorized owner/supervisor
procedure after both workers have loaded the new source protocol.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from file_system_catalog.source_page_quiescence import (
    SCHEMA,SCOPE,read_control,read_only_worker_acknowledgements,
)

RUNROOT=Path('/data/var/knowmoredirt/knowledge-library')
MARKER=RUNROOT/'hierarchical-publisher-revision-quiesce.json'
WORKERS={'coder':RUNROOT/'hierarchical-worker-status.json',
         'q4':RUNROOT/'hierarchical-worker-q4-status.json'}


def read_private_worker_status(path:Path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('original summary worker status absent or symlinked')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        before=os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or not 50<=before.st_size<=65536
                or stat.S_IMODE(before.st_mode)&0o077 or before.st_uid!=os.geteuid()):
            raise ValueError('untrusted source worker status file')
        data=os.read(fd,65537)
        after=os.fstat(fd)
        if (len(data)!=before.st_size or not os.path.samestat(before,after)
                or before.st_mtime_ns!=after.st_mtime_ns):
            raise ValueError('worker status changed during exact read')
    finally:
        os.close(fd)
    result=json.loads(data.decode('utf8'))
    if not isinstance(result,dict):
        raise ValueError('invalid worker source-boundary status JSON')
    return result


def current_main_pid(unit:str):
    if unit not in ('knowledge-library-hierarchical-summary.service',
                   'knowledge-library-hierarchical-summary-q4.service'):
        raise ValueError('unknown installed local owner summary worker')
    p=subprocess.run(['systemctl','show',unit,'-p','MainPID','--value'],
                     capture_output=True,text=True,timeout=5,check=True)
    if not p.stdout.strip().isdigit() or int(p.stdout.strip())<=1:
        raise ValueError('worker systemd has no verified active main process')
    return int(p.stdout.strip())


def actual_process_cmdline(pid:int):
    if type(pid) is not int or pid<=1:
        raise ValueError('invalid live worker process id')
    p=Path('/proc')/str(pid)/'cmdline'
    with p.open('rb') as stream:
        return stream.read(8192)


def main():
    # The default read-only command is deliberately never an operator:
    # no --apply or --new-control options exist.
    if len(sys.argv)!=1:
        raise ValueError('publisher pause readiness has no action flags or live mutations')
    control=read_control(MARKER)
    states={label:read_private_worker_status(path) for label,path in WORKERS.items()}
    pid={label:current_main_pid('knowledge-library-hierarchical-summary'+
           ('-q4' if label=='q4' else '')+'.service') for label in ('coder','q4')}
    process_started={label:(state.get('quiesce_runtime_started_utc')==state.get('started_utc')
           and state.get('quiesce_protocol')==SCHEMA
           and state.get('quiesce_scope')==SCOPE
           and type(state.get('worker_pid')) is int
           and state['worker_pid']==pid[label]) for label,state in states.items()}
    status={'schema':'kmd.publisher_native_full_source_document_boundary_readiness.v1',
         'current_marker_action':control.action,'current_marker_generation':control.generation,
         'existing_live_workers_running_new_pause_protocol':process_started,
         'both_new_code_workers_confirmed':all(process_started.values()),
         'both_workers_quiesced_at_complete_original_boundary':False,
         'other_native_writers_and_retrieval_readers_not_locked':True,
         'safe_to_cutover_live':False,
         'does_not_create_manifest_stop_restart_or_signal_services':True}
    if control.action=='hold':
        try:
            verified=read_only_worker_acknowledgements(control=control,states=states,
                   current_pid=current_main_pid,current_cmdline=actual_process_cmdline)
        except (ValueError,KeyError,TypeError,OSError,subprocess.SubprocessError) as error:
            status['hold_acknowledgements_refused']=f'{type(error).__name__}: {str(error)[:180]}'
        else:
            # An external operator might advance a release while this
            # precondition check is in flight. Re-read the exact generation;
            # never report a paused generation that has already been released.
            fresh=read_control(MARKER,prior_generation=control.generation,
                               prior_digest=control.digest)
            if fresh!=control:
                status['hold_acknowledgements_refused']='publisher hold control advanced during read-only verification'
            else:
                status['both_workers_quiesced_at_complete_original_boundary']=verified[
                    'both_resident_summary_workers_paused_at_source_boundaries']
    elif control.generation==0:
        status['hold_acknowledgements_refused']='no owner-only publisher hold manifest requested'
    else:
        status['hold_acknowledgements_refused']='source generation is released, not held'
    print(json.dumps(status,sort_keys=True))

if __name__=='__main__':main()
