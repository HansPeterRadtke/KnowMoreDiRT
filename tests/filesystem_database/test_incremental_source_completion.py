"""Incremental original-summary journal cursor safety and no-lost-event tests.

Unit tests use synthetic original completion messages; production source-map,
FileDB, worker journals and physical publisher originals are untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from file_system_catalog.incremental_source_completion import (
    CompletionEvent,parse_completion_entry,first_page_in_journal_order,
    latest_source_events,validate_cursor,validate_previous_cursor_state,
)


def line(index:int, path:str='graphics/blender-5-2-physics/approved.html',
         *, message:str|None=None, cursor:str|None=None):
    if message is None:
        message=f'KMD_HIERARCHICAL_DOCUMENT_COMPLETE 2026-11-28T10:44:{index%60:02d}+00:00 {path}'
    item={'__CURSOR':cursor or 's=9a0f0b;'+f'i={index:04x};'+f't={index:06x}',
        '__REALTIME_TIMESTAMP':str(1795862600000000+index*1000000),
        'MESSAGE':message}
    return json.dumps(item,separators=(',',':'))


class CurrentOriginalSourceJournalCompletionTests(unittest.TestCase):
    def test_strict_realistic_both_worker_completed_original_message(self):
        for i,path in enumerate((
            'graphics/blender-5-2/ies-source.html',
            'ai-agents/2026-graph-agent-research.pdf',
            'science/journal/math-study.pdf'
        ),1):
            e=parse_completion_entry(line(i,path))
            self.assertIsInstance(e,CompletionEvent)
            self.assertEqual(e.source_path,path)
            self.assertTrue(e.cursor.startswith('s=9a0f0b;'))
            self.assertTrue(e.recorded_utc.endswith('+00:00'))
            self.assertTrue(e.journal_timestamp_us>0)
            self.assertEqual(e.as_receipt()['source_path'],path)

    def test_oldest_page_not_tail_and_cursor_excludes_previous_events(self):
        lines=[line(i,'papers/number-%02d.pdf'%i) for i in range(1,31)]
        first=first_page_in_journal_order(lines,limit=7)
        self.assertEqual([f'papers/number-{i:02d}.pdf' for i in range(1,8)],
                         [x.source_path for x in first])
        self.assertEqual(7,len({z.cursor for z in first}))
        # A resumed journalctl --after-cursor must start AFTER the
        # last successfully proven event, never from the last N.
        second=first_page_in_journal_order(lines[7:],limit=7)
        self.assertEqual('papers/number-08.pdf',second[0].source_path)
        self.assertNotIn(first[-1].cursor,{x.cursor for x in second})

    def test_page_reader_does_not_consume_newer_events_after_bound(self):
        checked=[]
        def stream():
            for i in range(1,90):
                checked.append(i)
                yield line(i,'papers/november-%03d.pdf'%i)
        first=first_page_in_journal_order(stream(),limit=5)
        self.assertEqual([1,2,3,4,5],checked)
        self.assertEqual(5,len(first))
        self.assertEqual('papers/november-005.pdf',first[-1].source_path)

    def test_duplicate_source_revisions_are_preserved_as_events(self):
        same='graphics/manuals/approved-derivative-revised.html'
        raw=[line(1,'ai-agents/a.pdf'),line(2,same),line(3,'ai-agents/b.pdf'),
             line(4,same),line(5,'ai-agents/c.pdf')]
        page=first_page_in_journal_order(raw,limit=5)
        self.assertEqual(5,len(page))
        unique=latest_source_events(page)
        self.assertEqual(4,len(unique))
        self.assertEqual(page[3].cursor,unique[2].cursor)
        self.assertEqual(page[-1].cursor,unique[-1].cursor)
        self.assertIn(page[3].cursor,{x.cursor for x in unique})
        self.assertNotIn(page[1].cursor,{x.cursor for x in unique})

    def test_unsafe_or_truncated_source_events_fail_closed(self):
        for source in ('','/etc/passwd','../parent.pdf','graphics/../wrong.pdf',
                       'a.pdf','bad\\directory/file.pdf','graphics//duplicate.pdf',
                       'graphics/foo\nforge.pdf','graphics/foo\rforge.pdf'):
            with self.subTest(source=repr(source)),self.assertRaises(ValueError):
                parse_completion_entry(line(1,source))
        for forged in ('DEBUG KMD_HIERARCHICAL_DOCUMENT_COMPLETE 2026-11-28T10:44:01+00:00 papers/ok.pdf',
                       'KMD_HIERARCHICAL_DOCUMENT_COMPLETE papers/ok.pdf',
                       'KMD_HIERARCHICAL_DOCUMENT_COMPLETE 2026-11-28T10:44:01 papers/ok.pdf'):
            with self.subTest(forged=forged),self.assertRaises(ValueError):
                parse_completion_entry(line(1,message=forged))
        with self.assertRaises(ValueError):
            parse_completion_entry('bad_json')
        with self.assertRaises(ValueError):
            parse_completion_entry('{}')
        with self.assertRaises(ValueError):
            parse_completion_entry(line(1)+' '*9000)

    def test_invalid_or_duplicate_cursors_and_invalid_batch_limits(self):
        for cursor in ('',None,'../bad/path','new\nline','s='+'1'*900):
            with self.subTest(cursor=repr(cursor)),self.assertRaises(ValueError):
                validate_cursor(cursor)
        for limit in (0,-1,26,True,1.5):
            with self.subTest(limit=limit),self.assertRaises(ValueError):
                first_page_in_journal_order([line(1)],limit=limit)
        duplicate=line(1)
        with self.assertRaisesRegex(ValueError,'duplicate journald'):
            first_page_in_journal_order([duplicate,duplicate],limit=3)

    def test_previous_cursor_requires_same_project_verified_immutable_receipt(self):
        state={'schema':'kmd.private_read_only_completed_original_journal_cursor.v1',
               'project':'owner-20261101-knowledge-library-growth-newchat',
               'last_successful_event_cursor':'s=abc123def456;i=001',
               'total_events_audited':10,
               'last_successful_original_receipt_sha256':hashlib.sha256(b'verified').hexdigest(),
               'last_successful_audit_file':'/private/previous.json'}
        self.assertEqual(state,validate_previous_cursor_state(dict(state)))
        for bad in (
            {**state,'project':'other'},
            {**state,'last_successful_event_cursor':'bad'},
            {**state,'total_events_audited':0},
            {**state,'total_events_audited':True},
            {**state,'last_successful_original_receipt_sha256':'0'*63},
            {**state,'schema':'other'},
        ):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                validate_previous_cursor_state(bad)


if __name__=='__main__':
    unittest.main()
