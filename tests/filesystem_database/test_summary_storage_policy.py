"""No silent unbounded native KMD growth or loss of host disk reserve."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from file_system_catalog.summary_storage_policy import (
    DEFAULT_DATABASE_BUDGET_BYTES, MAX_DATABASE_BUDGET_BYTES,
    MIN_FREE_DISK_RESERVE_BYTES, SQLITE_WRITE_HEADROOM_BYTES,
    storage_budget_snapshot, require_storage_admission,
)

GIB = 1024 ** 3


class NativeSummaryStorageCapacityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name) / 'original-kmd.sqlite3'
        self.file.write_bytes(b'valid test database placeholder')

    def sampled_free(self, free_bytes):
        # os.statvfs.f_bavail (not f_bfree) respects an ordinary user's disk quota.
        return patch('file_system_catalog.summary_storage_policy.os.statvfs',
                     return_value=SimpleNamespace(f_frsize=4096, f_bavail=free_bytes // 4096))

    def test_large_bounded_ceiling_does_not_preallocate_space(self):
        self.assertEqual(DEFAULT_DATABASE_BUDGET_BYTES, 32 * GIB)
        self.assertEqual(MAX_DATABASE_BUDGET_BYTES, 32 * GIB)
        with self.sampled_free(120 * GIB):
            report = require_storage_admission(self.file)
        self.assertTrue(report['admission_safe'])
        self.assertEqual(self.file.stat().st_size, report['database_bytes'])
        self.assertEqual(120 * GIB, report['unprivileged_filesystem_available_bytes'])
        self.assertGreater(report['database_remaining_budget_bytes'], 31 * GIB)

    def test_disk_pressure_fails_closed_with_no_file_change(self):
        before = self.file.read_bytes()
        reserve = MIN_FREE_DISK_RESERVE_BYTES
        headroom = SQLITE_WRITE_HEADROOM_BYTES
        for available in (reserve - 4096, reserve, reserve + headroom - 4096):
            with self.subTest(available=available), self.sampled_free(available):
                self.assertFalse(storage_budget_snapshot(self.file)['admission_safe'])
                with self.assertRaisesRegex(RuntimeError, 'disk reserve'):
                    require_storage_admission(self.file)
        self.assertEqual(before, self.file.read_bytes())
        with self.sampled_free(reserve + headroom):
            self.assertTrue(require_storage_admission(self.file)['admission_safe'])

    def test_rejects_unsafe_database_path_and_invalid_bounds(self):
        link = Path(self.tmp.name) / 'link.sqlite3'
        link.symlink_to(self.file)
        with self.assertRaisesRegex(RuntimeError, 'unsafe'):
            require_storage_admission(link)
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            require_storage_admission(Path(self.tmp.name) / 'missing')
        for value in (0, 512 * 1024, 33 * GIB, '32'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                storage_budget_snapshot(self.file, budget_bytes=value)
        for value in (-1, 0, 256, 129 * GIB):
            with self.subTest(value=value), self.assertRaises(ValueError):
                storage_budget_snapshot(self.file, reserve_bytes=value)

    def test_budget_uses_actual_file_size_and_rejects_capacity_exhaustion(self):
        with self.file.open('r+b') as data:
            data.truncate(1 * GIB)  # sparse local test fixture, no disk allocation
        with self.sampled_free(120 * GIB):
            report = require_storage_admission(self.file, budget_bytes=2 * GIB)
            self.assertEqual(1 * GIB, report['database_remaining_budget_bytes'])
        with self.sampled_free(120 * GIB):
            with self.assertRaisesRegex(RuntimeError, 'budget is exhausted'):
                require_storage_admission(self.file, budget_bytes=1 * GIB)


if __name__ == '__main__':
    unittest.main()
