"""Bounded, lossless-cursor *selection* of original-source completion events.

The journal is used only to identify candidates; each selected original
still requires independent publisher original SHA, extracted text SHA,
approved receipt and full native summary-graph verification. Journal
events are not evidence that an original is currently indexed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime,timezone
import json
from pathlib import PurePosixPath
import re
from typing import Iterable

PREFIX='KMD_HIERARCHICAL_DOCUMENT_COMPLETE '
CURSOR=re.compile(r'[A-Za-z0-9:;=+._-]{8,700}\Z')
UTC_STAMP=re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.+_-]{12,40}\Z')
MAX_EVENT_BYTES=8192
MAX_EVENTS_PER_BATCH=25
MAX_PATH_CHARACTERS=1600
SOURCE_UNITS=(
 'knowledge-library-hierarchical-summary.service',
 'knowledge-library-hierarchical-summary-q4.service',
)

@dataclass(frozen=True)
class CompletionEvent:
    cursor: str
    recorded_utc: str
    journal_timestamp_us: int
    source_path: str

    def as_receipt(self) -> dict:
        return {'cursor':self.cursor,'recorded_utc':self.recorded_utc,
                'journal_timestamp_us':self.journal_timestamp_us,
                'source_path':self.source_path}


def validate_cursor(cursor: str) -> str:
    if not isinstance(cursor,str) or not CURSOR.fullmatch(cursor):
        raise ValueError('source-completion journal cursor missing or malformed')
    return cursor


def parse_completion_entry(line: str) -> CompletionEvent:
    """Accept only a real per-unit JSON completion with a journal cursor."""
    if not isinstance(line,str) or not line or len(line.encode('utf8'))>MAX_EVENT_BYTES:
        raise ValueError('unbounded or invalid original-completion journal record')
    try:
        payload=json.loads(line)
    except (json.JSONDecodeError,TypeError) as exc:
        raise ValueError('original completion log is not valid journald JSON') from exc
    if not isinstance(payload,dict):
        raise ValueError('source completion journal record is not an object')
    cursor=validate_cursor(payload.get('__CURSOR'))
    stamp=payload.get('__REALTIME_TIMESTAMP')
    if not isinstance(stamp,str) or not stamp.isdecimal() or not 1<=len(stamp)<=21:
        raise ValueError('missing trusted journald event timestamp')
    t=int(stamp)
    if t<=0:
        raise ValueError('invalid completion event journal time')
    msg=payload.get('MESSAGE')
    if not isinstance(msg,str) or not msg.startswith(PREFIX):
        raise ValueError('unrecognized original source worker completion message')
    suffix=msg[len(PREFIX):]
    when,sep,path=suffix.partition(' ')
    if not sep or not UTC_STAMP.fullmatch(when) or not path:
        raise ValueError('missing or invalid original source completion metadata')
    try:
        date=datetime.fromisoformat(when)
    except ValueError as exc:
        raise ValueError('unparseable source completion timestamp') from exc
    if date.tzinfo is None:
        raise ValueError('source completion timestamp requires timezone')
    if (len(path)>MAX_PATH_CHARACTERS or path.startswith('/') or
            '\\' in path or '\x00' in path or '\n' in path or '\r' in path or
            any(part in ('','.','..') for part in path.split('/'))):
        raise ValueError('original completion contains an unsafe publisher relative path')
    if len(PurePosixPath(path).parts)<2:
        raise ValueError('original completion must name a publisher original subfolder')
    return CompletionEvent(cursor=cursor,
        recorded_utc=date.astimezone(timezone.utc).isoformat(),
        journal_timestamp_us=t,source_path=path)


def first_page_in_journal_order(lines: Iterable[str], *, limit: int) -> list[CompletionEvent]:
    """Choose oldest pending events, never the last N (which skips backlog).

    This function does not deduplicate source paths: a second completion
    for a revised original is a separate event and must not disappear.
    """
    if type(limit) is not int or not 1<=limit<=MAX_EVENTS_PER_BATCH:
        raise ValueError('incremental source auditor requires bounded batch size')
    out=[];seen=set()
    for line in lines:
        if not isinstance(line,str) or not line.strip():
            continue
        item=parse_completion_entry(line)
        if item.cursor in seen:
            raise ValueError('duplicate journald completion cursor in source page')
        seen.add(item.cursor)
        out.append(item)
        if len(out)==limit:
            break
    return out


def latest_source_events(events: list[CompletionEvent]) -> list[CompletionEvent]:
    """Deduplicate exact source files *within* an audited event page only."""
    if not isinstance(events,list) or any(not isinstance(x,CompletionEvent) for x in events):
        raise ValueError('invalid incremental source event batch')
    unique={}
    for event in events:
        # Re-inserting moves repeated original to newest event in this page.
        if event.source_path in unique:
            del unique[event.source_path]
        unique[event.source_path]=event
    return list(unique.values())


def validate_previous_cursor_state(state: dict) -> dict:
    if (not isinstance(state,dict)
            or state.get('schema')!='kmd.private_read_only_completed_original_journal_cursor.v1'
            or state.get('project')!='owner-20261101-knowledge-library-growth-newchat'):
        raise ValueError('source completion cursor state missing required scope')
    validate_cursor(state.get('last_successful_event_cursor'))
    if (type(state.get('total_events_audited')) is not int
            or state['total_events_audited']<1):
        raise ValueError('source completion cursor lacks successful event count')
    if (not isinstance(state.get('last_successful_original_receipt_sha256'),str)
            or not re.fullmatch('[0-9a-f]{64}',state['last_successful_original_receipt_sha256'])):
        raise ValueError('source completion cursor lacks verifiable prior audit identity')
    return state
