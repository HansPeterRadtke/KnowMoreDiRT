"""Cooperative legacy-client exclusion for PRIVATE source generation rehearsal only.

No live source-map/FileDB, no real KMD service, and no unregistered clients
are considered enrolled by these tests. A private lock is not production proof.
"""
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import json
import os
import unittest
from unittest import mock

import test_private_source_generation as source_fixtures
from file_system_catalog.private_source_generation import open_pinned_private_read_view, read_pinned
from file_system_catalog.private_source_admission import (
    admit_private_legacy_reader,
    admit_private_legacy_writer,
    publish_private_after_legacy_drain,
    read_private_legacy_seal,
)


def _separate_process_private_legacy_hold(base:str,allowed:str,entered,release,results):
    """Hold a cooperative private reader lock in an independent Linux process."""
    try:
        with admit_private_legacy_reader(Path(base),Path(allowed)):
            entered.set()
            if not release.wait(timeout=7):
                raise RuntimeError('private test-only reader release timed out')
        results.put('clean')
    except BaseException as exc:
        results.put(type(exc).__name__+':'+str(exc)[:170])


class PrivateCooperativeAdmissionTests(unittest.TestCase):
    def setUp(self):
        fixture=source_fixtures.PrivateSourceGenerationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture=fixture
        self.base=fixture.switch
        self.allowed=fixture.private_root

    def stage(self,generation,which):
        return self.fixture.stage(generation,which)

    def test_active_legacy_reader_refuses_coordinated_private_generation_switch(self):
        self.stage(1,0)
        with admit_private_legacy_reader(self.base,self.allowed) as old:
            self.assertFalse(old.safe_to_cutover_live)
            with self.assertRaisesRegex(ValueError,'legacy reader.*active'):
                publish_private_after_legacy_drain(self.base,self.allowed,1)
            self.assertIsNone(read_private_legacy_seal(self.base,self.allowed))
            self.assertFalse((self.base/'active-private-generation.json').exists())
        result=publish_private_after_legacy_drain(self.base,self.allowed,1)
        self.assertTrue(result['cooperating_legacy_reader_admissions_sealed'])
        self.assertFalse(result['unregistered_clients_excluded'])
        self.assertFalse(result['safe_to_cutover_live'])
        self.assertEqual(1,read_pinned(self.base,self.allowed).generation)
        with self.assertRaisesRegex(ValueError,'legacy reader admission.*sealed'):
            with admit_private_legacy_reader(self.base,self.allowed):
                pass

    def test_cooperative_legacy_writer_also_blocks_private_publication(self):
        self.stage(1,0)
        with admit_private_legacy_writer(self.base,self.allowed) as writer:
            self.assertEqual('writer',writer.kind)
            self.assertFalse(writer.safe_to_cutover_live)
            with self.assertRaisesRegex(ValueError,'legacy reader.*active'):
                publish_private_after_legacy_drain(self.base,self.allowed,1)
        self.assertIsNone(read_private_legacy_seal(self.base,self.allowed))
        result=publish_private_after_legacy_drain(self.base,self.allowed,1)
        self.assertEqual(1,result['private_pointer_generation'])
        with self.assertRaisesRegex(ValueError,'sealed'):
            with admit_private_legacy_writer(self.base,self.allowed):pass

    def test_replaced_lock_path_cannot_bypass_old_reader_already_holding_admission(self):
        # A file-only flock can be bypassed if the current lock path is
        # renamed and recreated: old reader retains a SH lease on the old
        # inode while a newer publisher takes EX on the new inode.
        # Private generation root must also be a stable shared lock arbiter.
        self.stage(1,0)
        with admit_private_legacy_reader(self.base,self.allowed) as old:
            original=self.base/'.private-legacy-reader-admission.lock'
            moved=self.base/'.test-only-old-private-legacy-admission.lock'
            self.assertFalse(moved.exists())
            original.rename(moved)
            fd=os.open(original,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
            os.close(fd)
            with self.assertRaisesRegex(ValueError,'legacy reader.*active|directory admission'):
                publish_private_after_legacy_drain(self.base,self.allowed,1)
            self.assertIsNone(read_private_legacy_seal(self.base,self.allowed))
            self.assertFalse((self.base/'active-private-generation.json').exists())
        # Both files remain in the private test directory; no other owner's
        # temporary file is deleted or replaced in this regression.
        self.assertTrue(original.exists() and moved.exists())

    def test_cross_process_old_reader_cannot_be_bypassed_by_new_lock_inode(self):
        # A second OS process still holds SH on the original lock file and
        # the stable private root directory while the current pathname is
        # moved and replaced by a different private fixture inode.
        self.stage(1,0)
        ctx=get_context('fork');entered=ctx.Event();release=ctx.Event();messages=ctx.Queue()
        proc=ctx.Process(target=_separate_process_private_legacy_hold,
              args=(str(self.base),str(self.allowed),entered,release,messages))
        proc.start()
        try:
            self.assertTrue(entered.wait(timeout=6),'private old reader never entered')
            old=self.base/'.private-legacy-reader-admission.lock'
            displaced=self.base/'.test-only-displaced-locked-inode'
            self.assertFalse(displaced.exists())
            old.rename(displaced)
            fd=os.open(old,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
            os.close(fd)
            with self.assertRaisesRegex(ValueError,'legacy reader.*active|directory admission'):
                publish_private_after_legacy_drain(self.base,self.allowed,1)
            self.assertIsNone(read_private_legacy_seal(self.base,self.allowed))
            self.assertFalse((self.base/'active-private-generation.json').exists())
        finally:
            release.set();proc.join(timeout=8)
            if proc.is_alive():proc.terminate();proc.join(timeout=3)
        self.assertEqual(0,proc.exitcode)
        self.assertEqual('clean',messages.get(timeout=2))
        self.assertTrue(old.is_file() and displaced.is_file())

    def test_real_separate_process_legacy_reader_blocks_switch_until_it_exits(self):
        self.stage(1,0)
        ctx=get_context('fork');entered=ctx.Event();release=ctx.Event();messages=ctx.Queue()
        proc=ctx.Process(target=_separate_process_private_legacy_hold,
              args=(str(self.base),str(self.allowed),entered,release,messages))
        proc.start()
        try:
            self.assertTrue(entered.wait(timeout=6),'private SH lease never entered')
            with self.assertRaisesRegex(ValueError,'legacy reader.*active'):
                publish_private_after_legacy_drain(self.base,self.allowed,1)
            self.assertFalse((self.base/'active-private-generation.json').exists())
            self.assertIsNone(read_private_legacy_seal(self.base,self.allowed))
        finally:
            release.set();proc.join(timeout=8)
            if proc.is_alive():
                proc.terminate();proc.join(timeout=3)
        self.assertEqual(0,proc.exitcode)
        self.assertEqual('clean',messages.get(timeout=2))
        result=publish_private_after_legacy_drain(self.base,self.allowed,1)
        self.assertEqual(1,result['private_pointer_generation'])
        self.assertFalse(result['all_live_KMD_readers_admitted'])

    def test_old_private_pinned_reader_remains_valid_while_new_generation_publishes(self):
        old=self.stage(1,0)
        self.stage(2,1)
        publish_private_after_legacy_drain(self.base,self.allowed,1)
        with open_pinned_private_read_view(self.base,self.allowed) as pinned:
            original=[x[0] for x in pinned.source_map.execute('select source_path from originals')]
            self.assertEqual(1,pinned.generation)
            with ThreadPoolExecutor(max_workers=1) as pool:
                next_gen=pool.submit(publish_private_after_legacy_drain,
                                     self.base,self.allowed,2).result(timeout=10)
            self.assertEqual(2,next_gen['private_pointer_generation'])
            self.assertEqual(2,read_pinned(self.base,self.allowed).generation)
            self.assertEqual(original,[x[0] for x in pinned.native.execute(
                'select relative_path_display from filesystem_entries')])
            self.assertEqual(original,[x[0] for x in pinned.source_map.execute(
                'select source_path from originals')])
            self.assertTrue(pinned.verify_bytes(self.allowed)['private_generation_complete'])
        with open_pinned_private_read_view(self.base,self.allowed) as newer:
            self.assertEqual(2,newer.generation)
            self.assertNotEqual(original,[x[0] for x in newer.source_map.execute(
                'select source_path from originals')])
        with self.assertRaisesRegex(ValueError,'sealed'):
            with admit_private_legacy_reader(self.base,self.allowed):pass
        self.assertTrue(old.exists())

    def test_crash_after_durable_private_seal_retains_old_pointer_and_denies_new_legacy(self):
        self.stage(1,0)
        def crash():
            seal=read_private_legacy_seal(self.base,self.allowed)
            self.assertEqual(1,seal['sealed_for_generation'])
            self.assertIsNone((self.base/'active-private-generation.json').exists() or None)
            raise RuntimeError('injected crash after private legacy admission sealed')
        with self.assertRaisesRegex(RuntimeError,'injected crash'):
            publish_private_after_legacy_drain(self.base,self.allowed,1,after_seal=crash)
        self.assertFalse((self.base/'active-private-generation.json').exists())
        with self.assertRaisesRegex(ValueError,'sealed'):
            with admit_private_legacy_reader(self.base,self.allowed):pass
        # An operator may complete the same private first commit later;
        # the seal is durable and never automatically reopened.
        outcome=publish_private_after_legacy_drain(self.base,self.allowed,1)
        self.assertEqual(1,outcome['private_pointer_generation'])
        self.assertEqual(1,read_pinned(self.base,self.allowed).generation)

    def test_invalid_private_staged_candidate_or_ordinal_does_not_close_legacy_admission(self):
        folder=self.stage(1,0)
        derivative=next((folder/'text').rglob('*.txt'))
        derivative.write_bytes(derivative.read_bytes()+b'corrupted private staging')
        with self.assertRaises(ValueError):
            publish_private_after_legacy_drain(self.base,self.allowed,1)
        self.assertIsNone(read_private_legacy_seal(self.base,self.allowed))
        with admit_private_legacy_reader(self.base,self.allowed):pass
        # A valid staged generation TWO still cannot be the first pointer.
        self.stage(2,1)
        with self.assertRaisesRegex(ValueError,'increment|first private'):
            publish_private_after_legacy_drain(self.base,self.allowed,2)
        self.assertIsNone(read_private_legacy_seal(self.base,self.allowed))

    def test_symlinked_or_world_readable_private_admission_controls_are_refused(self):
        self.stage(1,0)
        lock=self.base/'.private-legacy-reader-admission.lock'
        lock.symlink_to(self.base/'generations')
        with self.assertRaises(OSError):
            with admit_private_legacy_reader(self.base,self.allowed):pass
        lock.unlink()
        with admit_private_legacy_reader(self.base,self.allowed):pass
        lock.chmod(0o644)
        with self.assertRaisesRegex(ValueError,'owner-only'):
            with admit_private_legacy_reader(self.base,self.allowed):pass
        lock.chmod(0o600)
        publish_private_after_legacy_drain(self.base,self.allowed,1)
        seal=self.base/'private-legacy-admission-sealed.json'
        seal.chmod(0o644)
        with self.assertRaisesRegex(ValueError,'owner-only'):
            read_private_legacy_seal(self.base,self.allowed)

    def test_deleted_private_seal_cannot_reopen_legacy_readers_after_published_pointer(self):
        self.stage(1,0)
        publish_private_after_legacy_drain(self.base,self.allowed,1)
        seal=self.base/'private-legacy-admission-sealed.json'
        seal.unlink()  # malicious private test-only deletion, never the real KMD.
        with self.assertRaisesRegex(ValueError,'legacy reader admission seal missing|legacy reader admission.*sealed'):
            with admit_private_legacy_reader(self.base,self.allowed):pass

    def test_private_admission_cannot_consider_production_source_root(self):
        live=Path('/data/var/knowmoredirt/knowledge-library')
        if live.is_dir():
            with self.assertRaisesRegex(ValueError,'private'):
                with admit_private_legacy_reader(live,Path('/data/var')):pass
            with self.assertRaisesRegex(ValueError,'private'):
                publish_private_after_legacy_drain(live,Path('/data/var'),1)

    def test_substituted_seal_manifest_digest_is_rejected_before_next_pointer(self):
        self.stage(1,0)
        self.stage(2,1)
        publish_private_after_legacy_drain(self.base,self.allowed,1)
        seal_path=self.base/'private-legacy-admission-sealed.json'
        seal=json.loads(seal_path.read_text())
        seal['manifest_sha256']='f'*64
        seal_path.write_text(json.dumps(seal))
        with self.assertRaisesRegex(ValueError,'seal manifest|seal revision'):
            publish_private_after_legacy_drain(self.base,self.allowed,2)
        self.assertEqual(1,read_pinned(self.base,self.allowed).generation)
        with self.assertRaisesRegex(ValueError,'sealed'):
            with admit_private_legacy_reader(self.base,self.allowed):pass

    def test_prior_abandoned_private_seal_temp_is_never_deleted_or_replayed(self):
        self.stage(1,0)
        orphan=self.base/('.private-legacy-seal-new-'+str(os.getpid()))
        orphan.write_bytes(b'foreign-orphaned-private-staging-not-ours')
        before=orphan.read_bytes()
        pub=publish_private_after_legacy_drain(self.base,self.allowed,1)
        self.assertEqual(1,pub['private_pointer_generation'])
        self.assertEqual(before,orphan.read_bytes(),
                         'unknown temporary file must not be overwritten or removed')
        self.assertEqual(1,read_pinned(self.base,self.allowed).generation)
        self.assertEqual(1,read_private_legacy_seal(self.base,self.allowed)['sealed_for_generation'])

    def test_corrupted_or_symlinked_private_seal_does_not_reopen_old_reader(self):
        self.stage(1,0)
        publish_private_after_legacy_drain(self.base,self.allowed,1)
        seal=self.base/'private-legacy-admission-sealed.json'
        seal.write_text('{"unknown": true}')
        with self.assertRaisesRegex(ValueError,'invalid|size'):
            with admit_private_legacy_reader(self.base,self.allowed):pass
        seal.unlink()
        seal.symlink_to(self.base/'generations')
        with self.assertRaises((OSError,ValueError)):
            with admit_private_legacy_reader(self.base,self.allowed):pass
        seal.unlink()
        with self.assertRaisesRegex(ValueError,'seal missing'):
            with admit_private_legacy_reader(self.base,self.allowed):pass
