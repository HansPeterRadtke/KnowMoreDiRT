"""Bounded native KMD interprocess writer lock safety.

All tests use temporary lock files. Never touch the original FileDB
or the global production content lock.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.util
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock


def cli():
    path=Path(__file__).resolve().parents[2]/"scripts"/"raspi_hierarchical_library.py"
    spec=importlib.util.spec_from_file_location("kmd_content_lock_test_module",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Clock:
    def __init__(self):
        self.now=0.
        self.slept=[]
    def monotonic(self):
        return self.now
    def sleep(self,n):
        self.slept.append(n)
        self.now+=n


class NativeWriterAdmissionTests(unittest.TestCase):
    def test_retries_temporarily_busy_original_writer_lock(self):
        module=cli()
        count=[]
        clock=Clock()
        def flock(fd,flags):
            count.append(flags)
            if len(count)<=4:
                raise BlockingIOError("other source writer owns content lock")
        with mock.patch("fcntl.flock",side_effect=flock), \
             mock.patch.object(module.time,"monotonic",side_effect=clock.monotonic), \
             mock.patch.object(module.time,"sleep",side_effect=clock.sleep):
            module.acquire_native_filedb_content_lock(6,wait_seconds=2,poll_seconds=.25)
        self.assertEqual(5,len(count))
        self.assertEqual([.25]*4,clock.slept)
        self.assertTrue(all(x & fcntl.LOCK_EX for x in count))
        self.assertTrue(all(x & fcntl.LOCK_NB for x in count))

    def test_refuses_to_wait_indefinitely_if_writer_never_releases(self):
        module=cli()
        clock=Clock()
        with mock.patch("fcntl.flock",side_effect=BlockingIOError("busy")), \
             mock.patch.object(module.time,"monotonic",side_effect=clock.monotonic), \
             mock.patch.object(module.time,"sleep",side_effect=clock.sleep):
            with self.assertRaisesRegex(TimeoutError,"beyond bounded wait"):
                module.acquire_native_filedb_content_lock(
                    3,wait_seconds=1,poll_seconds=.25
                )
        self.assertGreaterEqual(clock.now,1)

    def test_real_two_open_descriptors_serialize_and_preserve_lock(self):
        module=cli()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"original-fakedb-write.lock"
            with path.open("a+") as first, path.open("a+") as second:
                fcntl.flock(first.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                begin=time.monotonic()
                with ThreadPoolExecutor(max_workers=1) as pool:
                    wait=pool.submit(module.acquire_native_filedb_content_lock,
                                     second.fileno(),wait_seconds=4,poll_seconds=.05)
                    time.sleep(.2)
                    self.assertFalse(wait.done())
                    fcntl.flock(first.fileno(),fcntl.LOCK_UN)
                    wait.result(timeout=3)
                    self.assertGreaterEqual(time.monotonic()-begin,.18)
                fcntl.flock(second.fileno(),fcntl.LOCK_UN)

    def test_invalid_configuration_fails_closed(self):
        module=cli()
        for wait,poll in [(0,.2),(301,.2),(10,0),(10,11)]:
            with self.subTest(wait=wait,poll=poll),self.assertRaises(ValueError):
                module.acquire_native_filedb_content_lock(7,wait_seconds=wait,poll_seconds=poll)


if __name__=="__main__":
    unittest.main()
