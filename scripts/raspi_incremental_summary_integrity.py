#!/usr/bin/env python3
"""Incrementally attest newly completed current original-source summary graphs.

The existing canonical source-map and native KMD FileDB are opened only in
query-only mode. No summary generation, model inference, source download,
database mutation or persistent new service. A journal cursor is advanced
ONLY when every selected event's current original and full recursive summary
graph can be physically reverified against its approved publisher receipt.

Run under a resource-broker admission for potentially large source hashes.
An ordinary unverified source blocks cursor advancement; retain the receipt,
investigate it and never silently skip that original to mark a later page
complete. This is a resumable operator command, NOT an automatic scheduler.
"""
from __future__ import annotations

import argparse
from contextlib import closing,contextmanager
import fcntl
import stat
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from typing import Iterator

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))

from file_system_catalog.incremental_source_completion import (
    SOURCE_UNITS,MAX_EVENTS_PER_BATCH,first_page_in_journal_order,
    latest_source_events,parse_completion_entry,validate_previous_cursor_state,
)
from file_system_catalog.recent_source_tree_audit import (
    audit_current_source_tree,MAX_SOURCE_BYTES,MAX_EXTENDED_SOURCE_BYTES,
)
from raspi_recent_summary_integrity import ledger_receipts

PROJECT='owner-20261101-knowledge-library-growth-newchat'
CURSOR_SCHEMA='kmd.private_read_only_completed_original_journal_cursor.v1'
REPORT_SCHEMA='kmd.private_incremental_current_source_summary_provenance_cursor_batch.v1'
DATE=re.compile(r'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\Z')
MAX_LOG_TIME_SECONDS=45


def ensure_private_new_file(path:Path) -> Path:
    path=Path(path)
    if path.exists() or path.is_symlink() or not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError('refuse existing/unsafe incremental source audit receipt')
    if (path.parent.stat().st_mode & 0o077)!=0:
        raise ValueError('incremental source audit parent must be owner-private')
    return path


def write_exclusive_json(path:Path,record:dict) -> str:
    raw=(json.dumps(record,indent=2,ensure_ascii=False)+'\n').encode('utf8')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC|os.O_NOFOLLOW,0o600)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        # Keep any uncertain, partial receipt rather than overwrite it.
        raise
    return hashlib.sha256(raw).hexdigest()


def save_cursor_atomically(path:Path,state:dict) -> None:
    raw=(json.dumps(state,indent=2,ensure_ascii=False)+'\n').encode('utf8')
    tmp=path.with_name('.'+path.name+'.pending-owned')
    if tmp.exists() or tmp.is_symlink():
        raise ValueError('existing uncertain source cursor journal update; refuse')
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC|os.O_NOFOLLOW,0o600)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp,path)
        # Best-effort persist the parent directory entry; do not fsync
        # either live database or any unowned original publisher content.
        dir_fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(dir_fd)
        finally:os.close(dir_fd)
    except Exception:
        if tmp.exists() and not tmp.is_symlink():
            tmp.unlink()
        raise


def validated_cursor(path:Path, *, initial_since:str|None)->dict|None:
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError('source cursor path must not be symlink')
    if path.is_file():
        if initial_since is not None:
            raise ValueError('existing source cursor cannot restart at initial timestamp')
        if (stat.S_IMODE(path.stat().st_mode)!=0o600
                or not 1<=path.stat().st_size<=65536):
            raise ValueError('previous owner-private source cursor is unsafe or oversized')
        state=validate_previous_cursor_state(json.loads(path.read_bytes()))
        prior=Path(state['last_successful_audit_file'])
        if (prior.parent!=path.parent or prior.is_symlink() or not prior.is_file()
                or stat.S_IMODE(prior.stat().st_mode)!=0o600
                or not 1<=prior.stat().st_size<=4*1024*1024):
            raise ValueError('previous private original source audit receipt unavailable or oversized')
        raw=prior.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=state['last_successful_original_receipt_sha256']:
            raise ValueError('previous original source audit receipt digest changed; refuse cursor')
        batch=json.loads(raw)
        if (batch.get('schema')!=REPORT_SCHEMA
                or batch.get('last_event_cursor')!=state['last_successful_event_cursor']
                or batch.get('cursor_advanced') is not True
                or batch.get('all_selected_originals_verified') is not True):
            raise ValueError('source cursor previous receipt inconsistent')
        return state
    if path.exists():
        raise ValueError('source cursor path must be a regular existing file or absent')
    if not isinstance(initial_since,str) or not DATE.fullmatch(initial_since):
        raise ValueError('initial source journal UTC since must be strict')
    try:datetime.strptime(initial_since,'%Y-%m-%d %H:%M:%S')
    except ValueError as err:raise ValueError('invalid source-journal UTC start date') from err
    return None


def confirm_retained_journal_cursor(previous:dict) -> None:
    """Fail closed if journald vacuumed or lost our last audited event.

    A cursor token in our protected private receipt is not enough to
    prove the source journal still retains that event. journalctl
    --after-cursor for a missing token may return status 1 with no
    messages, indistinguishable from a genuinely empty new page.
    Reopen the inclusive --cursor first and demand its VERY FIRST
    completion event has exactly the previous verified cursor.
    Stream only one JSON record, not all later journal messages.
    """
    key=previous['last_successful_event_cursor']
    cmd=['/usr/bin/timeout','--signal=TERM','12',
         'journalctl',
         *(word for unit in SOURCE_UNITS for word in ('-u',unit)),
         '--no-pager','--output=json',
         '--grep=^KMD_HIERARCHICAL_DOCUMENT_COMPLETE ',
         '--cursor='+key]
    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        text=True,encoding='utf8',errors='strict',env={**os.environ,'TZ':'UTC'})
    try:
        assert process.stdout is not None
        line=process.stdout.readline()
        if not line:
            raise ValueError('prior audited source-completion journal cursor unavailable; no safe resume')
        actual=parse_completion_entry(line)
        if actual.cursor!=key:
            raise ValueError('previous source-completion cursor not retained; refusing to skip or replay')
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=4)
        if process.stdout:process.stdout.close()
        if process.stderr:process.stderr.close()


def journal_page(*, previous:dict|None, initial_since:str|None,
                 limit:int) -> list:
    """Fetch oldest N from both real worker units; never 'journalctl -n' tail.

    A small bounded stdout stream avoids retaining the entire historic
    journal. When the requested N is reached, stopping the CLI intentionally
    leaves later entries to the verified cursor on the NEXT call.
    """
    if type(limit) is not int or not 1<=limit<=MAX_EVENTS_PER_BATCH:
        raise ValueError('invalid bounded original source journal page')
    if previous:
        confirm_retained_journal_cursor(previous)
    cmd=['/usr/bin/timeout','--signal=TERM',str(MAX_LOG_TIME_SECONDS),
         'journalctl',
         *(word for unit in SOURCE_UNITS for word in ('-u',unit)),
         '--no-pager','--output=json',
         '--grep=^KMD_HIERARCHICAL_DOCUMENT_COMPLETE ']
    if previous:
        cmd.extend(['--after-cursor='+previous['last_successful_event_cursor']])
    else:
        cmd.extend(['--since='+initial_since])
    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                             text=True,encoding='utf8',errors='strict',
                             env={**os.environ,'TZ':'UTC'},bufsize=1)
    try:
        def selected_lines()->Iterator[str]:
            # journalctl without --follow terminates normally. With a
            # timeout wrapper, never wait indefinitely for a bad journal.
            assert process.stdout is not None
            while True:
                line=process.stdout.readline()
                if not line:
                    break
                yield line
        entries=first_page_in_journal_order(selected_lines(),limit=limit)
        if len(entries)==limit and process.poll() is None:
            process.terminate()
        try:
            code=process.wait(timeout=6)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=6)
            raise ValueError('source journal query did not exit after bounded paging')
        # Explicitly stopped by us after reaching the requested count:
        # negative SIGTERM is expected; the timeout wrapper may report 143.
        # journalctl returns status 1 with empty stdout/stderr when a
        # previously verified cursor has no matching *new* completion.
        # This means no observable new event, NOT a failure or a source
        # indexed count. Never advance the cursor for that observation.
        if code==1 and not entries:
            err=process.stderr.read(700) if process.stderr else ''
            if not err.strip():
                return []
            raise ValueError('source journal cursor unavailable; '+err[:180])
        if code not in (0,-15,143) or (len(entries)<limit and code!=0):
            err=process.stderr.read(700) if process.stderr else ''
            raise ValueError('source journal unavailable or timed out; '+err[:180])
        if any(entries[i].journal_timestamp_us>entries[i+1].journal_timestamp_us
               for i in range(len(entries)-1)):
            raise ValueError('source completion journal sequence out of order')
        return entries
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if process.stdout:process.stdout.close()
        if process.stderr:process.stderr.close()


@contextmanager
def exclusive_private_cursor_auditor(path:Path):
    """Prevent two owner turns from publishing a regressed cursor.

    The lock is a 0600 owner-private file, not a new database/service or a
    producer lock. It protects only our optional audit cursor transaction.
    A concurrent auditor fails immediately; never wait behind it or
    overwrite its valid source journal advance.
    """
    lock=path.with_name('.'+path.name+'.audit-lock')
    fd=os.open(lock,os.O_RDWR|os.O_CREAT|os.O_CLOEXEC|os.O_NOFOLLOW,0o600)
    acquired=False
    try:
        current=os.fstat(fd)
        if not stat.S_ISREG(current.st_mode) or stat.S_IMODE(current.st_mode)!=0o600:
            raise ValueError('owner-private original cursor audit lock is unsafe')
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('another original-source cursor audit is active; no overlapping work') from exc
        acquired=True
        yield
    finally:
        if acquired:
            fcntl.flock(fd,fcntl.LOCK_UN)
        os.close(fd)


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('source-map','native-db','original-root','derived-root',
                'completed-ledger','output','cursor-file'):
        parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--initial-since',default=None,
                        help='Exact initial UTC start: YYYY-MM-DD HH:MM:SS, only if no cursor')
    parser.add_argument('--batch-size',type=int,default=10)
    parser.add_argument('--max-original-mib',type=int,
                        default=MAX_SOURCE_BYTES//(1024**2))
    a=parser.parse_args()
    if (type(a.batch_size) is not int or not 1<=a.batch_size<=MAX_EVENTS_PER_BATCH
            or type(a.max_original_mib) is not int
            or not 1<=a.max_original_mib<=MAX_EXTENDED_SOURCE_BYTES//(1024**2)):
        raise ValueError('bounded original source count and file SHA limit required')
    output=ensure_private_new_file(a.output)
    if (a.cursor_file.parent!=output.parent or a.cursor_file==output
            or a.cursor_file.is_symlink()):
        raise ValueError('cursor and receipt must be separate files in same owner-private directory')
    with exclusive_private_cursor_auditor(a.cursor_file):
        run_incremental_batch(a,output)


def run_incremental_batch(a,arg_output:Path) -> None:
    output=arg_output
    previous=validated_cursor(a.cursor_file,initial_since=a.initial_since)
    for p in (a.source_map,a.native_db):
        if not p.is_file() or p.is_symlink():
            raise ValueError('canonical source map and KMD FileDB must be regular files')
    entries=journal_page(previous=previous,initial_since=a.initial_since,limit=a.batch_size)
    if not entries:
        # This is a valid no-change source observation, NOT a completed
        # audit or a cursor advanced past nonexistent original files.
        payload={'schema':REPORT_SCHEMA,'project':PROJECT,
                 'checked_utc':datetime.now(timezone.utc).isoformat(),
                 'previous_cursor':previous['last_successful_event_cursor'] if previous else None,
                 'first_event_cursor':None,'last_event_cursor':None,
                 'selected_completion_events':0,
                 'selected_unique_source_paths':0,
                 'new_complete_originals_physically_verified':0,
                 'native_summary_nodes_physically_verified':0,
                 'all_selected_originals_verified':False,'cursor_advanced':False,
                 'no_new_original_completions_observed':True,
                 'canonical_source_map_and_native_index_queried_read_only':True}
        write_exclusive_json(output,payload)
        print('SOURCE_COMPLETION_NO_NEW_JOURNAL_EVENTS_CURSOR_UNCHANGED',flush=True)
        return
    receipt_sha,_=ledger_receipts(a.completed_ledger)
    # This is an approved receipt mapping, not one composite database
    # generation. Validate original bytes against each current source-map row.
    selected=latest_source_events(entries)
    checked=[]
    with closing(sqlite3.connect(a.source_map.resolve().as_uri()+'?mode=ro',
                                 uri=True,timeout=20)) as sm,\
         closing(sqlite3.connect(a.native_db.resolve().as_uri()+'?mode=ro',
                                 uri=True,timeout=120)) as native:
        sm.execute('PRAGMA query_only=ON')
        native.execute('PRAGMA query_only=ON')
        for event in selected:
            try:
                row=audit_current_source_tree(
                    original_path=event.source_path,source_map=sm,native_db=native,
                    original_root=a.original_root,derivative_root=a.derived_root,
                    completed_receipts=receipt_sha,
                    max_original_bytes=a.max_original_mib*1024*1024)
                checked.append({'status':'verified',**row,'source_event_cursor':event.cursor})
                print('INCREMENTAL_CURRENT_ORIGINAL_VERIFIED',
                      event.source_path.rsplit('/',1)[-1][:75],
                      row['current_native_summary_nodes_fully_verified'],flush=True)
            except Exception as exc:
                # A current source may legitimately have been revised again;
                # the former completion is NOT proof of a current tree.
                issue=f'{type(exc).__name__}: {str(exc)[:250]}'
                checked.append({'status':'not_currently_verified',
                                'source_path':event.source_path,
                                'source_event_cursor':event.cursor,
                                'reason':issue,
                                'must_not_advance_cursor':True})
                print('INCREMENTAL_ORIGINAL_UNVERIFIED_KEEP_CURSOR',
                      event.source_path.rsplit('/',1)[-1][:75],issue[:95],flush=True)
    all_good=len(checked)==len(selected) and all(z['status']=='verified' for z in checked)
    total=(previous['total_events_audited'] if previous else 0)
    payload={
        'schema':REPORT_SCHEMA,'project':PROJECT,
        'checked_utc':datetime.now(timezone.utc).isoformat(),
        'journal_sources':list(SOURCE_UNITS),
        'initial_since_utc':a.initial_since if previous is None else None,
        'previous_cursor':previous['last_successful_event_cursor'] if previous else None,
        'first_event_cursor':entries[0].cursor,
        'last_event_cursor':entries[-1].cursor,
        'selected_completion_events':len(entries),
        'selected_unique_source_paths':len(selected),
        'source_paths_duplicated_within_page':len(entries)-len(selected),
        'new_complete_originals_physically_verified':sum(z['status']=='verified' for z in checked),
        'native_summary_nodes_physically_verified':sum(z.get('current_native_summary_nodes_fully_verified',0) for z in checked),
        'source_selection_oldest_first_not_journal_tail':True,
        'originals_acquisition_receipts_rechecked_against_current_source_map':True,
        'all_selected_originals_verified':all_good,
        'cursor_advanced':all_good,
        'source_map_native_and_publisher_file_views_not_atomic':True,
        'publisher_original_files_and_derived_files_unchanged':True,
        'no_new_download_model_query_or_native_writer':True,
        'checks':checked,
    }
    report_sha=write_exclusive_json(output,payload)
    if all_good:
        state={'schema':CURSOR_SCHEMA,'project':PROJECT,
            'last_successful_event_cursor':entries[-1].cursor,
            'last_successful_original_receipt_sha256':report_sha,
            'last_successful_audit_file':str(output),
            'total_events_audited':total+len(entries),
            'updated_utc':datetime.now(timezone.utc).isoformat()}
        save_cursor_atomically(a.cursor_file,state)
        check=validate_previous_cursor_state(json.loads(a.cursor_file.read_bytes()))
        assert check['last_successful_original_receipt_sha256']==report_sha
    print('INCREMENTAL_PROVENANCE_PAGE_RECEIPT',{
        'events':len(entries),'unique_sources':len(selected),
        'verified_origins':payload['new_complete_originals_physically_verified'],
        'verified_native_nodes':payload['native_summary_nodes_physically_verified'],
        'cursor_advanced':all_good,
        'private_receipt_sha256':report_sha},flush=True)
    if not all_good:
        raise SystemExit(2)


if __name__=='__main__':
    main()
