"""Native source-identity reader must survive ordinary serialized FileDB writes.

All files and locks are disposable test fixtures; production summary workers,
source-map, original publisher bytes and SQLite are never opened here.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from test_summary_search_audit import load_cli


class NativeSummaryReaderLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db=Path(self.tmp.name)/'fake-native.sqlite3'
        with sqlite3.connect(self.db) as c:
            c.execute('CREATE TABLE original_anchor(id INTEGER PRIMARY KEY,original_path TEXT NOT NULL)')
            c.execute('INSERT INTO original_anchor VALUES(1,?)',('original/unchanged.pdf',))
        self.cli=load_cli()

    def test_retries_real_read_only_source_query_behind_an_exclusive_writer(self):
        def delayed_reader():
            with closing(self.cli.open_native_summary_read_connection(
                         self.db,wait_seconds=3)) as c:
                a=c.execute('SELECT original_path FROM original_anchor').fetchone()[0]
                return a,c.execute('PRAGMA query_only').fetchone()[0]
        writer=sqlite3.connect(self.db,isolation_level=None)
        self.addCleanup(writer.close)
        self.assertEqual('delete',writer.execute('PRAGMA journal_mode').fetchone()[0])
        writer.execute('BEGIN EXCLUSIVE')
        with ThreadPoolExecutor(max_workers=1) as pool:
            f=pool.submit(delayed_reader)
            try:
                time.sleep(.3)
                self.assertFalse(f.done())
            finally:
                writer.execute('ROLLBACK')
            path,readonly=f.result(timeout=3)
        self.assertEqual('original/unchanged.pdf',path)
        self.assertEqual(1,readonly)
        self.assertEqual('delete',writer.execute('PRAGMA journal_mode').fetchone()[0])
        self.assertEqual('original/unchanged.pdf',writer.execute('SELECT original_path FROM original_anchor').fetchone()[0])

    def test_default_budget_is_bounded_and_no_persistent_writes_are_possible(self):
        self.assertEqual(120.0,self.cli.NATIVE_READER_WAIT_SECONDS)
        with closing(self.cli.open_native_summary_read_connection(self.db)) as c:
            self.assertEqual(1,c.execute('PRAGMA query_only').fetchone()[0])
            with self.assertRaises(sqlite3.OperationalError):
                c.execute('INSERT INTO original_anchor VALUES(2,?)',('unwanted',))
        with sqlite3.connect(self.db) as c:
            self.assertEqual(1,c.execute('SELECT count(*) FROM original_anchor').fetchone()[0])

    def test_invalid_waits_and_wrong_paths_fail_closed(self):
        for invalid in (0,-1,301,float('inf'),float('nan'),True,'forever'):
            with self.subTest(wait=invalid),self.assertRaises(ValueError):
                self.cli.open_native_summary_read_connection(self.db,wait_seconds=invalid)
        absent=Path(self.tmp.name)/'absent.sqlite3'
        with self.assertRaises(ValueError):
            self.cli.open_native_summary_read_connection(absent)
        link=Path(self.tmp.name)/'link.sqlite3'
        link.symlink_to(self.db)
        with self.assertRaises(ValueError):
            self.cli.open_native_summary_read_connection(link)

    def test_bounded_wait_refuses_indefinite_exclusive_lock(self):
        writer=sqlite3.connect(self.db,isolation_level=None)
        self.addCleanup(writer.close)
        writer.execute('BEGIN EXCLUSIVE')
        try:
            begin=time.monotonic()
            with self.assertRaises(sqlite3.OperationalError):
                with closing(self.cli.open_native_summary_read_connection(
                        self.db,wait_seconds=1)) as c:
                    c.execute('SELECT original_path FROM original_anchor').fetchall()
            elapsed=time.monotonic()-begin
            self.assertGreaterEqual(elapsed,.8)
            self.assertLess(elapsed,2.5)
        finally:
            writer.execute('ROLLBACK')


if __name__=='__main__':
    unittest.main()
