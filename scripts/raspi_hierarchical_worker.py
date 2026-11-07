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
    current_completed_originals, summary_progress_metrics,
)

FILEDB = Path("/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3")
SOURCE_MAP = Path("/data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3")
RUNROOT = Path("/data/var/knowmoredirt/knowledge-library")
STATE = RUNROOT / "hierarchical-worker-status.json"
LOCK = RUNROOT / ".hierarchical-worker.lock"
CLI = Path(__file__).with_name("raspi_hierarchical_library.py")
SOURCE_MANIFEST = Path("/data/infra/etc/knowledge-library-sources.json")
stop = False

def signal_stop(_signal, _frame):
    global stop
    stop = True

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

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

def pending_failures(failed: dict, available: list, processed: set) -> dict:
    """Do not report already indexed or retired revisions as active failures."""
    valid = {row[0] for row in available}
    return {
        path: entry for path, entry in failed.items()
        if path in valid and path not in processed
    }


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

def run(*, sleep_seconds: int, max_calls: int, once: bool, dry_run: bool):
    if os.uname().nodename.split(".")[0] != "raspi":
        raise RuntimeError("KnowMoreDiRT Knowledge Library exists only on Raspi")
    if not CLI.is_file() or not SOURCE_MAP.is_file() or not FILEDB.is_file():
        raise FileNotFoundError("missing canonical KnowMoreDiRT summary program/database")
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, signal_stop)
    RUNROOT.mkdir(parents=True, exist_ok=True)
    lock = LOCK.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another KnowMoreDiRT hierarchical worker already owns this library")
    failed = {}
    try:
        prior = json.loads(STATE.read_text())
        if isinstance(prior.get("failed"), dict):
            failed = prior["failed"]
    except (OSError, ValueError):
        pass
    state_update(phase="starting", worker_pid=os.getpid(), started_utc=now(),
                 summaries_per_original="all levels", source_of_truth=str(FILEDB),
                 model="qwen38-q4-default", concurrent_llm_requests=1,
                 storage_budget_bytes=4 * 1024**3,
                 native_facet_policy="summary_and_keywords_single_constrained_model_call")
    while not stop:
        try:
            available, processed = scope()
            failed = pending_failures(failed, available, processed)
            records = [
                (path, original_sha, derived_sha, int(size))
                for path, original_sha, derived_sha, size in available
                if path not in processed
            ]
            # Periodically exercise chapter-level and larger recursive
            # hierarchies, rather than delaying all substantial books until
            # thousands of very short documents have been processed.
            # Ordering never introduces a second concurrent model request.
            records = balanced_backlog(records, len(processed))
            versions = {p: (orig, deriv, int(chars)) for p, orig, deriv, chars in available}
            metrics = summary_progress_metrics(versions, processed)
            state_update(
                phase="running", **metrics, failed=failed,
                backlog_policy="interleave_short_medium_every8_large_every40",
            )
            if not any(p not in processed for p in versions):
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
            for path, original_sha, derived_sha, text_chars in records:
                if stop: break
                identity = sha((path+"\0"+original_sha+"\0"+derived_sha).encode())
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
                                     "--plan-only"], "PLAN", path)
                    else:
                        from raspi_hierarchical_library import payload_path
                        completed_tree = payload_path(path, original_sha, derived_sha)
                        if not completed_tree.is_file():
                            run_command(
                                ["build", "--original-path", path,
                                 "--max-calls", str(max_calls),
                                 "--output-tokens", "1600",
                                 "--safety-tokens", "384",
                                 "--native-facets"],
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
                    state_update(
                        complete_originals=completed_count,
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

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sleep-seconds", type=int, default=4)
    p.add_argument("--max-calls", type=int, default=3000)
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not 0 <= args.sleep_seconds <= 3600 or not 1 <= args.max_calls <= 50000:
        raise ValueError("invalid bounded worker settings")
    run(sleep_seconds=args.sleep_seconds, max_calls=args.max_calls,
        once=args.once, dry_run=args.dry_run)
