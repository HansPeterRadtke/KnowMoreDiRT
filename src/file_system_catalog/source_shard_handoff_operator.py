"""Fail-closed *operator preparation* for a staged native source-shard handoff.

This code never stops a service or interacts with a resident model. An operator
must first confirm BOTH existing workers have naturally started the
source_shard_runtime protocol. Then bootstrap the known baseline and let both
workers explicitly acknowledge each allocation generation at a source boundary.
Each subsequent step can only release a bucket, wait for acknowledgement, and
then hand it to the second worker. It is not a stand-alone authority to restart
or kill shared services.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Callable

from .source_shard_runtime import (
    SCHEMA, KNOWN_STAGES, current_assignment, write_manifest_bytes,
)

UNIT = {
    'coder': 'knowledge-library-hierarchical-summary.service',
    'q4': 'knowledge-library-hierarchical-summary-q4.service',
}


def _fresh_status(data: Any, label: str, *, expected_shards: frozenset[int],
                  expected_generation: int, expected_stage: str | None,
                  expected_sha: str | None, live_pid: Callable[[str], int],
                  proc_command: Callable[[int], bytes],
                  now_utc: datetime) -> dict:
    if not isinstance(data, dict):
        raise ValueError('missing current worker acknowledgement')
    if (data.get('allocation_protocol') != SCHEMA
            or data.get('allocation_runtime_started_utc') != data.get('started_utc')
            or not isinstance(data.get('started_utc'), str)):
        raise ValueError('worker is not running the source-boundary handoff protocol')
    worker_pid = data.get('worker_pid')
    if (type(worker_pid) is not int or worker_pid <= 1
            or worker_pid != live_pid(UNIT[label])):
        raise ValueError('worker acknowledgement process is not the installed live main PID')
    proc = proc_command(worker_pid)
    if (b'raspi_hierarchical_worker.py' not in proc
            or b'--shard-count\x004\x00' not in proc
            or label=='q4' and b'--worker-label\x00q4\x00' not in proc):
        raise ValueError('unexpected live source worker program or identity')
    if label=='coder' and b'--worker-label\x00q4\x00' in proc:
        raise ValueError('Coder source worker and Q4 labels are swapped')
    if (data.get('allocation_generation') != expected_generation
            or data.get('allocation_stage') != expected_stage
            or data.get('allocation_manifest_sha256') != expected_sha
            or data.get('assigned_shards') != sorted(expected_shards)):
        raise ValueError('worker has not yet acknowledged the prior allocation at a source boundary')
    try:
        started=datetime.fromisoformat(data['started_utc'])
        updated=datetime.fromisoformat(data['updated_utc'])
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError('worker status lacks real timestamp provenance') from exc
    if (started.tzinfo is None or updated.tzinfo is None
            or started > now_utc or updated > now_utc
            or (now_utc-updated).total_seconds()>1800):
        raise ValueError('worker handoff acknowledgement is stale or invalid')
    source=data.get('current_original')
    if source is not None:
        if not isinstance(source,str):
            raise ValueError('invalid active source identity')
        import hashlib as _hashlib
        bucket=int.from_bytes(_hashlib.sha256(source.encode()).digest()[:8],'big')%4
        if bucket not in expected_shards:
            raise ValueError('worker is still processing a source from a released bucket')
    if data.get('phase') in ('stopped','error_recovering'):
        raise ValueError('source worker has not reported a healthy running state')
    return {'worker':label,'pid':worker_pid,'started_utc':data['started_utc'],
            'updated_utc':data['updated_utc'],'ack_generation':expected_generation,
            'ack_stage':expected_stage,'assigned_shards':sorted(expected_shards)}


def verify_releasing_worker_acknowledged(*, coder_state: dict,
                                         requested_generation: int,
                                         live_pid: Callable[[str], int],
                                         proc_command: Callable[[int], bytes],
                                         now_utc: datetime | None = None) -> dict:
    """Q4 can acquire shard three ONLY after Coder's live source-boundary ACK.

    This is checked independently in the worker at the acquisition boundary,
    not only in the administrative script which generated the manifest. A
    forged/bypassed balanced-stage manifest never overrides the live state.
    """
    if type(requested_generation) is not int or requested_generation < 2:
        raise ValueError('invalid source acquisition generation')
    if not isinstance(coder_state, dict):
        raise ValueError('missing actual releasing worker state')
    stage = coder_state.get('allocation_stage')
    generation = coder_state.get('allocation_generation')
    if not ((stage == 'release_three' and generation == requested_generation-1)
            or (stage == 'balanced' and generation == requested_generation)):
        raise ValueError('Coder has not acknowledged release of shard three')
    digest = coder_state.get('allocation_manifest_sha256')
    if not isinstance(digest,str) or len(digest)!=64:
        raise ValueError('missing accepted previous source allocation revision')
    return _fresh_status(
        coder_state,'coder',expected_shards=frozenset((1,2)),
        expected_generation=generation,expected_stage=stage,expected_sha=digest,
        live_pid=live_pid,proc_command=proc_command,
        now_utc=now_utc or datetime.now(timezone.utc),
    )


def verify_releasing_q4_acknowledged(*, q4_state: dict,
                                     requested_generation: int,
                                     live_pid: Callable[[str], int],
                                     proc_command: Callable[[int], bytes],
                                     now_utc: datetime | None = None) -> dict:
    """Coder must never reacquire bucket three before live Q4 releases it.

    This independently gates the rollback/acquisition branch of the reader
    so a direct, bypassed baseline manifest cannot create overlapping work.
    """
    if type(requested_generation) is not int or requested_generation < 2:
        raise ValueError('invalid rollback acquisition generation')
    if not isinstance(q4_state, dict):
        raise ValueError('missing actual Q4 releasing worker state')
    stage = q4_state.get('allocation_stage')
    generation = q4_state.get('allocation_generation')
    if not ((stage == 'release_three' and generation == requested_generation-1)
            or (stage == 'baseline' and generation == requested_generation)):
        raise ValueError('Q4 has not acknowledged release of shard three')
    digest = q4_state.get('allocation_manifest_sha256')
    if not isinstance(digest,str) or len(digest)!=64:
        raise ValueError('missing accepted previous Q4 allocation revision')
    return _fresh_status(
        q4_state,'q4',expected_shards=frozenset((0,)),
        expected_generation=generation,expected_stage=stage,expected_sha=digest,
        live_pid=live_pid,proc_command=proc_command,
        now_utc=now_utc or datetime.now(timezone.utc),
    )


def compute_step(*, manifest_path: Path,
                 worker_statuses: dict[str,dict],
                 target: str,
                 live_pid: Callable[[str],int],
                 proc_command: Callable[[int],bytes],
                 now_utc: datetime | None = None) -> dict:
    """Read-only, preconditions-only stage proposal. No source/units touched."""
    if target not in ('balanced','baseline'):
        raise ValueError('unsupported safe target allocation')
    now_utc=now_utc or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        raise ValueError('require an offset-aware current timestamp')
    if set(worker_statuses) != {'coder','q4'}:
        raise ValueError('missing current worker owner status')
    if manifest_path.is_symlink():
        raise ValueError('live ownership manifest is symlinked')
    if manifest_path.is_file():
        result=current_assignment(manifest_path,worker='coder',
                                  default=KNOWN_STAGES['baseline']['coder'],shard_count=4)
        stage=result.stage
        generation=result.generation
        digest=result.digest
    elif manifest_path.exists():
        raise ValueError('existing live source allocation path is not a file')
    else:
        stage=None
        generation=0
        digest=None
    assignments=KNOWN_STAGES[stage or 'baseline']
    acknowledged={label:_fresh_status(worker_statuses[label],label,
        expected_shards=assignments[label],expected_generation=generation,
        expected_stage=stage,expected_sha=digest,
        live_pid=live_pid,proc_command=proc_command,now_utc=now_utc)
        for label in ('coder','q4')}
    if stage is None:
        next_stage='baseline'
    elif stage == target:
        next_stage=stage
    elif stage=='release_three':
        next_stage=target
    elif stage in ('baseline','balanced'):
        next_stage='release_three'
    else:
        raise ValueError('unexpected current source assignment stage')
    noop=stage==next_stage
    proposed_bytes=None if noop else write_manifest_bytes(next_stage,generation+1)
    return {
        'schema':'kmd.two_worker_verified_handoff_operator_plan.v1',
        'source_current_stage':stage or 'absent',
        'source_current_generation':generation,
        'source_current_sha256':digest,
        'source_next_stage':next_stage,
        'source_next_generation':generation if noop else generation+1,
        'source_next_sha256':digest if noop else hashlib.sha256(proposed_bytes).hexdigest(),
        'target':target,'requires_change':not noop,
        'worker_current_proofs':acknowledged,
        'source_current_assignments':{k:sorted(v) for k,v in assignments.items()},
        'source_next_assignments':{k:sorted(v) for k,v in KNOWN_STAGES[next_stage].items()},
        'operator_must_wait_for_both_acknowledgements_between_steps':True,
        'does_not_signal_services_or_models':True,
        'manifest_bytes_to_write_after_second_preflight':proposed_bytes,
    }
