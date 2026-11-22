#!/usr/bin/env python3
"""Single persistent, low-resource Raspberry Pi recursive-summary worker.

Iterate ALL verified extracted original documents. For each original, generate
all recursive summary levels with the local model and append one vector for
every generated summary node to the same canonical native KMD FileDB.
The worker is resumable after restart and never replaces existing originals.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from file_system_catalog.summary_coverage import (
    current_completed_originals, pending_failures, revision_identity, summary_progress_metrics,
)
from file_system_catalog.summary_storage_policy import DEFAULT_DATABASE_BUDGET_BYTES
from file_system_catalog.source_shard_runtime import current_assignment
from file_system_catalog.source_page_quiescence import (
    SCHEMA as PUBLISHER_QUIESCE_SCHEMA,
    SCOPE as PUBLISHER_QUIESCE_SCOPE,
    read_control as read_publisher_quiesce_control,
    hold_at_source_boundary,
)
from file_system_catalog.source_shard_handoff_operator import (
    verify_releasing_worker_acknowledged, verify_releasing_q4_acknowledged,
)

FILEDB = Path("/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3")
SOURCE_MAP = Path("/data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3")
RUNROOT = Path("/data/var/knowmoredirt/knowledge-library")
STATE = RUNROOT / "hierarchical-worker-status.json"
LOCK = RUNROOT / ".hierarchical-worker.lock"
CLI = Path(__file__).with_name("raspi_hierarchical_library.py")
SOURCE_MANIFEST = Path("/data/infra/etc/knowledge-library-sources.json")
# Optional, owner-only *runtime* allocation file. Absent by default, so an
# installed systemd unit retains its exact existing --assigned-shards value.
# Only after both workers run this code may the supervisor stage changes.
ALLOCATION_MANIFEST = RUNROOT / "hierarchical-shard-handoff.json"
# Optional, no file by default. Only a future separately coordinated
# maintenance operator may request a source-boundary hold; NEVER stop
# resident workers or replace source-map/native files from this reader.
PUBLISHER_QUIESCE_MANIFEST = RUNROOT / "hierarchical-publisher-revision-quiesce.json"
stop = False

def signal_stop(_signal, _frame):
    global stop
    stop = True

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def state_update(**fields):
    old = {}
    try:
        old = json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    old.update(fields)
    old["updated_utc"] = now()
    RUNROOT.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_name(STATE.name + ".tmp." + str(os.getpid()))
    try:
        with tmp.open("x", encoding="utf-8") as output:
            json.dump(old, output, indent=2, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, STATE)
    finally:
        tmp.unlink(missing_ok=True)

def scope():
    with sqlite3.connect("file:" + str(SOURCE_MAP) + "?mode=ro", uri=True) as db:
        available = db.execute(
            "SELECT source_path, source_sha256,derived_sha256,text_characters "
            "FROM originals WHERE status='complete' AND text_characters>100"
        ).fetchall()
    current_versions = {
        path: (original_sha, derived_sha, int(characters))
        for path, original_sha, derived_sha, characters in available
    }
    with sqlite3.connect("file:" + str(FILEDB) + "?mode=ro", uri=True) as db:
        processed = current_completed_originals(db, current_versions)
    return available, processed

def shard_for_original(original_path: str, shard_count: int) -> int:
    """Stable original-file assignment; independent of revisions and queue order.

    Concurrent workers may write to the same native FileDB, but no two
    partitions are ever allowed to claim the same original document.
    """
    if not isinstance(original_path,str) or not original_path:
        raise ValueError("invalid original file path for stable shard assignment")
    if type(shard_count) is not int or not 1<=shard_count<=32:
        raise ValueError("invalid total worker count")
    h=hashlib.sha256(original_path.encode("utf-8")).digest()
    return int.from_bytes(h[:8],"big")%shard_count


def parse_assigned_shards(shard_count: int, shard_index: int,
                          assigned_shards: str | None) -> frozenset[int]:
    """A weighted split: Coder may own three of four stable hash buckets.

    A deterministic partition avoids duplicate summaries and allows a
    throughput-balanced split based on the two models' measured rates.
    """
    if type(shard_count) is not int or not 1<=shard_count<=32:
        raise ValueError("invalid number of worker shards")
    if type(shard_index) is not int or not 0<=shard_index<shard_count:
        raise ValueError("invalid shard index")
    if assigned_shards is None:
        return frozenset((shard_index,))
    if not isinstance(assigned_shards,str) or len(assigned_shards)>120:
        raise ValueError("invalid assigned worker shard list")
    parts=assigned_shards.split(",")
    if not parts or any(not x.isascii() or not x.isdecimal() for x in parts):
        raise ValueError("worker shard list must be positive decimal indexes")
    result=frozenset(int(x) for x in parts)
    if len(result)!=len(parts) or any(not 0<=x<shard_count for x in result):
        raise ValueError("worker shard list contains duplicates or out-of-range indexes")
    return result


def balanced_backlog(records: list, completed_count: int):
    """Mix brief sources with substantial chapter and book material.

    Every eighth *scheduled* source attempts a medium-size document
    (at least 60k extracted chars); every fortieth attempts a large
    document (at least 200k chars). Each bucket remains shortest-first,
    all sources are eventually visited, and no extra model concurrency
    is introduced. The mutable index already records successful originals
    across resumptions, so a restarted run resumes at its global position.
    """
    if type(completed_count) is not int or completed_count < 0:
        raise ValueError("invalid processed-original count")
    buckets = {"short": [], "medium": [], "large": []}
    for row in records:
        if len(row) != 4 or type(row[3]) is not int or row[3] < 0:
            raise ValueError("invalid original source size in backlog")
        if row[3] >= 200000:
            bucket = "large"
        elif row[3] >= 60000:
            bucket = "medium"
        else:
            bucket = "short"
        buckets[bucket].append(row)
    for label in buckets:
        buckets[label] = deque(sorted(
            buckets[label], key=lambda row: (row[3], row[0])
        ))
    for offset in range(len(records)):
        ordinal = completed_count + offset + 1
        preferred = (
            "large" if ordinal % 40 == 0
            else "medium" if ordinal % 8 == 0
            else "short"
        )
        for label in dict.fromkeys((preferred, "short", "medium", "large")):
            if buckets[label]:
                yield buckets[label].popleft()
                break


def verify_source_shard_acquisition(candidate, prior_shards, runtime_label, *, startup=False):
    """Fail closed if either worker acquires bucket three before release ACK.

    Startup must NOT trust the installed CLI baseline as proof that a
    restarted worker already owns bucket three. A legitimate rollback may
    have released it to Q4, while the fixed systemd command still contains
    --assigned-shards 1,2,3. Cross-check live release ACKs at startup too.
    """
    if 3 not in candidate.shards:
        return
    if candidate.generation == 0 and candidate.stage is None and 3 in prior_shards:
        return  # No manifest: precisely the existing legacy CLI assignment.
    must_check = startup or 3 not in prior_shards
    if runtime_label=='coder' and candidate.stage=='baseline' and candidate.generation <= 1:
        must_check = False  # Initial verified baseline is the installed unit's existing assignment.
    if not must_check:
        return
    if runtime_label=='q4' and candidate.stage!='balanced':
        raise ValueError('Q4 cannot acquire shard three outside balanced stage')
    if runtime_label=='coder' and candidate.stage!='baseline':
        raise ValueError('Coder cannot reacquire shard three outside baseline stage')
    # The administrative operator checks two real live workers before writing
    # any new generation. Independently verify this also in the Q4 worker:
    # a manually altered/forged manifest must not cause dual source ownership.
    import subprocess as _subprocess
    def active_pid(unit):
        result=_subprocess.run(['systemctl','show',unit,'-p','MainPID','--value'],
                               capture_output=True,text=True,timeout=5,check=True)
        return int(result.stdout.strip())
    def current_command(pid):
        with (Path('/proc')/str(pid)/'cmdline').open('rb') as stream:
            return stream.read(8192)
    other_label='coder' if runtime_label=='q4' else 'q4'
    state_path=(RUNROOT/'hierarchical-worker-status.json' if other_label=='coder'
                else RUNROOT/'hierarchical-worker-q4-status.json')
    if state_path.is_symlink() or not state_path.is_file() or state_path.stat().st_size>64*1024:
        raise ValueError('missing reliable releasing worker source-boundary handoff state')
    state=json.loads(state_path.read_text('utf8'))
    if other_label=='coder':
        verify_releasing_worker_acknowledged(
            coder_state=state,requested_generation=candidate.generation,
            live_pid=active_pid,proc_command=current_command,
        )
    else:
        verify_releasing_q4_acknowledged(
            q4_state=state,requested_generation=candidate.generation,
            live_pid=active_pid,proc_command=current_command,
        )


def run_command(parts, label, path):
    print("KMD_HIERARCHICAL", now(), label, path, flush=True)
    result = subprocess.run(
        [sys.executable, "-u", str(CLI), *parts],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        timeout=14400,  # extremely large books still get bounded retries
        check=False,
    )
    tail = result.stdout[-7000:]
    for line in tail.splitlines():
        if "ORIGINAL" in line or "COMPLETE_TREE" in line or "PLAN" in line or "CANONICAL_KMD_UPDATED" in line:
            print("KMD_HIERARCHICAL_RESULT", line[:1400], flush=True)
    if result.returncode:
        raise RuntimeError(label + " exited " + str(result.returncode) + ": " + tail[-800:])
    return tail

def run(
    *, sleep_seconds: int, max_calls: int, once: bool, dry_run: bool,
    model_url: str = "http://10.8.0.7:14829",
    model_name: str = "qwen38-q4-default",
    output_tokens: int = 1600,
    context_cap_tokens: int | None = None,
    shared_slot_idle_seconds: float = 0.0,
    compact_native_facets: bool = False,
    shard_count: int = 1,
    shard_index: int = 0,
    assigned_shards: str | None = None,
    worker_label: str = "",
):
    if os.uname().nodename.split(".")[0] != "raspi":
        raise RuntimeError("KnowMoreDiRT Knowledge Library exists only on Raspi")
    shard_set=parse_assigned_shards(shard_count,shard_index,assigned_shards)
    if (not isinstance(worker_label,str)
            or len(worker_label)>32
            or (worker_label and not all(c.isascii() and (c.islower() or c.isdigit() or c in "_-")
                                         for c in worker_label))):
        raise ValueError("worker label must be a simple safe suffix")
    global LOCK, STATE
    if worker_label:
        LOCK = RUNROOT / f".hierarchical-worker-{worker_label}.lock"
        STATE = RUNROOT / f"hierarchical-worker-{worker_label}-status.json"
    if (not model_url.startswith(("http://10.","http://127.0.0.1:","http://localhost:"))
            or not model_name.strip()
            or not 700<=output_tokens<=8000
            or (context_cap_tokens is not None and not 1024<=context_cap_tokens<=262144)
            or not 0<=shared_slot_idle_seconds<=300):
        raise ValueError("invalid bounded local model configuration")
    if not CLI.is_file() or not SOURCE_MAP.is_file() or not FILEDB.is_file():
        raise FileNotFoundError("missing canonical KnowMoreDiRT summary program/database")
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, signal_stop)
    RUNROOT.mkdir(parents=True, exist_ok=True)
    lock = LOCK.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("another KnowMoreDiRT hierarchical worker already owns this library")
    try:
        failed = {}
        prior = {}
        try:
            prior = json.loads(STATE.read_text())
            if isinstance(prior.get("failed"), dict):
                failed = prior["failed"]
        except (OSError, ValueError):
            pass
        runtime_label = "coder" if not worker_label else worker_label
        runtime_enabled = shard_count == 4 and runtime_label in ("coder", "q4")
        if not runtime_enabled and ALLOCATION_MANIFEST.exists():
            raise ValueError("unrecognized worker cannot join active source ownership handoff")
        current_generation = 0
        current_digest = None
        current_stage = None
        if runtime_enabled:
            # Persisted generation prevents a restart from silently falling
            # back to the old CLI buckets after a live balanced allocation
            # manifest was removed or rolled back by mistake.
            current_generation = prior.get("allocation_generation", 0)
            current_digest = prior.get("allocation_manifest_sha256")
            assignment = current_assignment(
                ALLOCATION_MANIFEST,worker=runtime_label,default=shard_set,
                shard_count=shard_count,previous_generation=current_generation,
                previous_digest=current_digest)
            verify_source_shard_acquisition(assignment,shard_set,runtime_label,startup=True)
            shard_set = assignment.shards
            current_generation = assignment.generation
            current_digest = assignment.digest
            current_stage = assignment.stage
        quiesce_enabled = runtime_enabled
        quiesce_generation = 0
        quiesce_digest = None
        if quiesce_enabled:
            # A restarted worker must never silently forget a prior hold
            # or release generation. A missing manifest after opt-in fails
            # closed even if the old process has already exited.
            quiesce_generation = prior.get("quiesce_generation", 0)
            quiesce_digest = prior.get("quiesce_manifest_sha256")
            prior_control = read_publisher_quiesce_control(
                PUBLISHER_QUIESCE_MANIFEST,
                prior_generation=quiesce_generation,
                prior_digest=quiesce_digest,
            )
            quiesce_generation = prior_control.generation
            quiesce_digest = prior_control.digest
        start_utc=now()
        state_update(phase="starting", worker_pid=os.getpid(), started_utc=start_utc,
                     quiesce_runtime_started_utc=start_utc if quiesce_enabled else None,
                     quiesce_protocol=PUBLISHER_QUIESCE_SCHEMA if quiesce_enabled else None,
                     quiesce_scope=PUBLISHER_QUIESCE_SCOPE if quiesce_enabled else None,
                     quiesce_generation=quiesce_generation,
                     quiesce_manifest_sha256=quiesce_digest,
                     quiesce_worker_pid=os.getpid() if quiesce_enabled else None,
                     quiesce_ack_utc=None,
                     allocation_runtime_started_utc=start_utc if runtime_enabled else None,
                     summaries_per_original="all levels", source_of_truth=str(FILEDB),
                     model=model_name, model_url=model_url,
                     quality_source_group_context_cap_tokens=context_cap_tokens,
                     shared_voice_slot_idle_seconds=shared_slot_idle_seconds,
                     output_tokens=output_tokens,
                     compact_native_facets=compact_native_facets,
                     shard_index=shard_index,shard_count=shard_count,
                     assigned_shards=sorted(shard_set),
                     allocation_protocol="kmd.two_worker_shard_handoff.v1" if runtime_enabled else None,
                     allocation_generation=current_generation,
                     allocation_manifest_sha256=current_digest,
                     allocation_stage=current_stage,
                     shard_algorithm="sha256_path_first64_be_modulo",
                     worker_label=worker_label or "original_primary",
                     concurrent_llm_requests=1,
                     storage_budget_bytes=DEFAULT_DATABASE_BUDGET_BYTES,
                     native_facet_policy="summary_and_keywords_single_constrained_model_call")
        def enforce_original_boundary_hold():
            nonlocal quiesce_generation,quiesce_digest
            if not quiesce_enabled:
                return
            def record_control(**fields):
                nonlocal quiesce_generation,quiesce_digest
                # CRITICAL: remember the observed hold generation AS SOON AS
                # it is acknowledged. If a marker is deleted/replayed while
                # held, a raised reader error must not return our local
                # generation to zero on the next outer retry iteration.
                generation=fields.get('quiesce_generation')
                if type(generation) is int and generation>0:
                    quiesce_generation=generation
                    quiesce_digest=fields['quiesce_manifest_sha256']
                state_update(**fields)
            observed = hold_at_source_boundary(
                marker=PUBLISHER_QUIESCE_MANIFEST,
                worker_label=runtime_label, worker_pid=os.getpid(),
                prior_generation=quiesce_generation, prior_digest=quiesce_digest,
                update_status=record_control, should_stop=lambda:stop,
            )
            quiesce_generation=observed.generation
            quiesce_digest=observed.digest

        while not stop:
            try:
                # Do not enumerate or begin a new original while the hold is
                # active. If it arrived during the previous long build/import,
                # this acknowledgement occurs AFTER that subprocess ended.
                enforce_original_boundary_hold()
                if stop:
                    break
                if runtime_enabled:
                    assignment = current_assignment(
                        ALLOCATION_MANIFEST,worker=runtime_label,default=shard_set,
                        shard_count=shard_count,previous_generation=current_generation,
                        previous_digest=current_digest)
                    if (assignment.shards != shard_set or assignment.generation != current_generation
                            or assignment.digest != current_digest):
                        verify_source_shard_acquisition(assignment,shard_set,runtime_label)
                        shard_set = assignment.shards
                        current_generation = assignment.generation
                        current_digest = assignment.digest
                        current_stage = assignment.stage
                        state_update(phase="allocation_applied",current_original=None,
                                     assigned_shards=sorted(shard_set),
                                     allocation_generation=current_generation,
                                     allocation_manifest_sha256=current_digest,
                                     allocation_stage=current_stage)
                        print("KMD_HIERARCHICAL_ALLOCATION_APPLIED",now(),runtime_label,
                              current_generation,sorted(shard_set),flush=True)
                available, processed = scope()
                # A failure from before sharding may belong to the OTHER
                # worker. Keep only this worker's assigned source failures,
                # so retries/statuses remain source-disjoint.
                assigned_available = [
                    row for row in available
                    if shard_for_original(row[0],shard_count) in shard_set
                ]
                failed = pending_failures(failed, assigned_available, processed)
                records = [
                    (path, original_sha, derived_sha, int(size))
                    for path, original_sha, derived_sha, size in available
                    if path not in processed
                    and shard_for_original(path,shard_count) in shard_set
                ]
                shard_eligible_count=sum(
                    shard_for_original(row[0],shard_count) in shard_set
                    for row in available
                )
                shard_completed_count=sum(
                    shard_for_original(path,shard_count) in shard_set
                    for path in processed
                )
                assigned_pending=len(records)
                # Periodically exercise chapter-level and larger recursive
                # hierarchies, rather than delaying all substantial books until
                # thousands of very short documents have been processed.
                # Ordering never introduces a second concurrent model request.
                records = balanced_backlog(records, shard_completed_count)
                model_flags=[
                    "--model-url",model_url,"--model",model_name,
                    "--output-tokens",str(output_tokens),
                    "--safety-tokens","384",
                    "--shared-slot-idle-seconds",str(shared_slot_idle_seconds),
                    "--native-facets",
                ]
                if context_cap_tokens is not None:
                    model_flags.extend(["--context-cap-tokens",str(context_cap_tokens)])
                if compact_native_facets:
                    model_flags.append("--compact-native-facets")
                versions = {p: (orig, deriv, int(chars)) for p, orig, deriv, chars in available}
                metrics = summary_progress_metrics(versions, processed)
                state_update(
                    phase="running", **metrics, failed=failed,
                    backlog_policy="interleave_short_medium_every8_large_every40",
                    assigned_eligible_originals=shard_eligible_count,
                    assigned_complete_originals=shard_completed_count,
                    assigned_pending_originals=assigned_pending,
                )
                if assigned_pending==0:
                    state_update(phase="caught_up", current_original=None)
                    print("KMD_HIERARCHICAL_CAUGHT_UP", now(), len(processed), flush=True)
                    if once: return
                    time.sleep(360)
                    continue
                progress = False
                completed_count = len(processed)
                completed_chars = metrics["indexed_text_characters"]
                eligible_chars = metrics["eligible_text_characters"]
                completed_in_pass = 0
                allocation_changed = False
                for path, original_sha, derived_sha, text_chars in records:
                    if stop: break
                    # Verify again at every source boundary; a hold posted
                    # after scope() but before this candidate cannot admit
                    # any additional source. Never pause inside build/import.
                    enforce_original_boundary_hold()
                    if stop: break
                    if runtime_enabled:
                        # Source *boundary* only: never change ownership while
                        # building/importing the current source. A releasing
                        # worker publishes this acknowledgement only after
                        # its previous subprocess finished completely.
                        candidate = current_assignment(
                            ALLOCATION_MANIFEST,worker=runtime_label,default=shard_set,
                            shard_count=shard_count,previous_generation=current_generation,
                            previous_digest=current_digest)
                        if (candidate.shards != shard_set or candidate.generation != current_generation
                                or candidate.digest != current_digest):
                            verify_source_shard_acquisition(candidate,shard_set,runtime_label)
                            shard_set = candidate.shards
                            current_generation = candidate.generation
                            current_digest = candidate.digest
                            current_stage = candidate.stage
                            state_update(phase="allocation_applied",current_original=None,
                                         assigned_shards=sorted(shard_set),
                                         allocation_generation=current_generation,
                                         allocation_manifest_sha256=current_digest,
                                         allocation_stage=current_stage)
                            print("KMD_HIERARCHICAL_ALLOCATION_APPLIED",now(),runtime_label,
                                  current_generation,sorted(shard_set),flush=True)
                            allocation_changed = True
                            break  # Rebuild a source-disjoint backlog.
                    identity = revision_identity(path, original_sha, derived_sha)
                    status = failed.get(path, {})
                    if status.get("identity") != identity:
                        status = {}
                    if status.get("retry_after_epoch", 0) > time.time():
                        continue
                    state_update(phase="summarizing", current_original=path,
                                 current_characters=text_chars, current_started_utc=now())
                    try:
                        if dry_run:
                            run_command(["build", "--original-path", path,
                                         "--max-calls", str(max_calls),
                                         *model_flags,
                                         "--plan-only"], "PLAN", path)
                        else:
                            from raspi_hierarchical_library import payload_path
                            completed_tree = payload_path(path, original_sha, derived_sha)
                            if not completed_tree.is_file():
                                run_command(
                                    ["build", "--original-path", path,
                                     "--max-calls", str(max_calls),
                                     *model_flags],
                                    "BUILD", path,
                                )
                            state_update(phase="embedding", current_original=path)
                            run_command(
                                ["import", "--original-path", path, "--write"],
                                "EMBED_AND_STORE", path,
                            )
                        failed.pop(path, None)
                        progress = True
                        completed_count += 1
                        completed_chars += text_chars
                        completed_in_pass += 1
                        shard_completed_count += 1
                        assigned_pending -= 1
                        state_update(
                            complete_originals=completed_count,
                            assigned_complete_originals=shard_completed_count,
                            assigned_pending_originals=max(0,assigned_pending),
                            remaining_originals=max(0, len(available)-completed_count),
                            percentage=round(100.0*completed_count/max(1,len(available)), 6),
                            eligible_text_characters=eligible_chars,
                            indexed_text_characters=completed_chars,
                            text_percentage=round(100.0*completed_chars/max(1,eligible_chars), 6),
                            last_completed_original=path,
                            last_completed_utc=now(),
                            failed=failed,
                        )
                        print("KMD_HIERARCHICAL_DOCUMENT_COMPLETE", now(), path, flush=True)
                    except Exception as error:
                        num = min(100, int(status.get("attempts", 0)) + 1)
                        retry_delay = min(6*3600, 60 * 2**min(num-1,8))
                        failed[path] = {
                            "attempts": num, "identity": identity,
                            "last_error": str(error)[-500:],
                            "retry_after_epoch": round(time.time() + retry_delay),
                            "last_failed_utc": now(),
                        }
                        state_update(phase="retrying_others", failed=failed,
                                     last_failed_original=path, last_failed_utc=now())
                        print("KMD_HIERARCHICAL_RETRY_SCHEDULED", path,
                              type(error).__name__, str(error)[-400:],
                              "retry_seconds", retry_delay, flush=True)
                    if once:
                        return
                    if not stop and sleep_seconds:
                        time.sleep(sleep_seconds)
                    # Refresh authoritative completion counts and newly acquired
                    # originals regularly; do not wait for the 17k-item pass.
                    if completed_in_pass >= 10:
                        break
                if stop: break
                if allocation_changed:
                    continue  # Release/acquisition takes effect immediately at boundary.
                if not progress:
                    earliest = min((v.get("retry_after_epoch", time.time()+60)
                                    for v in failed.values()), default=time.time()+120)
                    rest = max(30, min(600, earliest-time.time()))
                    state_update(phase="waiting_for_retry", next_retry_after_seconds=int(rest))
                    time.sleep(rest)
            except Exception as error:
                state_update(phase="error_recovering", last_error=str(error)[-700:])
                print("KMD_HIERARCHICAL_WORKER_ERROR", repr(error), flush=True)
                if once:
                    raise
                time.sleep(60)
        state_update(phase="stopped", current_original=None, stopped_utc=now())
    finally:
        lock.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sleep-seconds", type=int, default=4)
    p.add_argument("--max-calls", type=int, default=3000)
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--model-url",default="http://10.8.0.7:14829")
    p.add_argument("--model",default="qwen38-q4-default")
    p.add_argument("--output-tokens",type=int,default=1600)
    p.add_argument("--context-cap-tokens",type=int,default=None)
    p.add_argument("--shared-slot-idle-seconds",type=float,default=0.0)
    p.add_argument("--compact-native-facets",action="store_true")
    p.add_argument("--shard-count",type=int,default=1)
    p.add_argument("--shard-index",type=int,default=0)
    p.add_argument("--assigned-shards",default=None)
    p.add_argument("--worker-label",default="")
    args = p.parse_args()
    if not 0 <= args.sleep_seconds <= 3600 or not 1 <= args.max_calls <= 50000:
        raise ValueError("invalid bounded worker settings")
    run(sleep_seconds=args.sleep_seconds, max_calls=args.max_calls,
        once=args.once, dry_run=args.dry_run,
        model_url=args.model_url,model_name=args.model,
        output_tokens=args.output_tokens,
        context_cap_tokens=args.context_cap_tokens,
        shared_slot_idle_seconds=args.shared_slot_idle_seconds,
        compact_native_facets=args.compact_native_facets,
        shard_count=args.shard_count,shard_index=args.shard_index,
        assigned_shards=args.assigned_shards,worker_label=args.worker_label)
