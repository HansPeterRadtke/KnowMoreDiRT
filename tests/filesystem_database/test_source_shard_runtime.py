"""No model, service or original FileDB effects: strict staged source ownership."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from file_system_catalog.source_shard_runtime import (
    current_assignment, write_manifest_bytes, KNOWN_STAGES, SCHEMA,
)


class ShardSourceRuntimeProtocolTests(unittest.TestCase):
    def setUp(self):
        tmp=TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path=Path(tmp.name)/'owner-source-assignments.json'
        self.default={'coder':frozenset((1,2,3)),'q4':frozenset((0,))}

    def save(self,stage='baseline',generation=1):
        self.path.write_bytes(write_manifest_bytes(stage,generation))
        self.path.chmod(0o600)
        return self.path.read_bytes()

    def load(self,worker='coder',**kwargs):
        return current_assignment(self.path,worker=worker,
                                  default=self.default[worker],shard_count=4,**kwargs)

    def test_absent_manifest_preserves_installed_systemd_worker_assignment(self):
        for worker in ('coder','q4'):
            assignment=self.load(worker)
            self.assertEqual(self.default[worker],assignment.shards)
            self.assertEqual(0,assignment.generation)
            self.assertIsNone(assignment.digest)
            self.assertIsNone(assignment.stage)

    def test_all_three_stages_explicitly_source_disjoint_and_conservative(self):
        for stage in ('baseline','release_three','balanced'):
            with self.subTest(stage=stage):
                data=self.save(stage,7)
                coder=self.load('coder')
                q4=self.load('q4')
                self.assertEqual(KNOWN_STAGES[stage]['coder'],coder.shards)
                self.assertEqual(KNOWN_STAGES[stage]['q4'],q4.shards)
                self.assertEqual(set(),coder.shards & q4.shards)
                self.assertEqual(hashlib.sha256(data).hexdigest(),coder.digest)
                self.assertEqual('release_three'==stage,3 not in coder.shards|q4.shards)
                self.assertEqual(7,q4.generation)

    def test_release_then_expand_and_reject_replayed_or_removed_manifest(self):
        self.save('release_three',11)
        first=self.load()
        self.save('balanced',12)
        newer=self.load(previous_generation=first.generation,previous_digest=first.digest)
        self.assertEqual(frozenset((1,2)),newer.shards)
        self.assertEqual(12,newer.generation)
        self.path.unlink()
        with self.assertRaisesRegex(ValueError,'was removed'):
            self.load(previous_generation=newer.generation,previous_digest=newer.digest)
        self.save('release_three',11)
        with self.assertRaisesRegex(ValueError,'generation'):
            self.load(previous_generation=newer.generation,previous_digest=newer.digest)

    def test_mutating_a_generation_without_bumping_is_refused(self):
        self.save('baseline',7)
        first=self.load()
        self.save('release_three',7)
        with self.assertRaisesRegex(ValueError,'generation replay'):
            self.load(previous_generation=first.generation,previous_digest=first.digest)

    def test_invalid_schema_overlap_and_invalid_generation_fail_closed(self):
        self.save()
        for mutate in (
            lambda x:x['assignments']['q4'].append(3),
            lambda x:x['assignments']['coder'].append(0),
            lambda x:x['assignments']['coder'].append(3),
            lambda x:x.update(stage='unmanaged'),
            lambda x:x.update(generation=True),
            lambda x:x.update(generation=-1),
            lambda x:x.update(schema='unsupported'),
            lambda x:x.update(untrusted='extra'),
        ):
            with self.subTest(mutator=mutate):
                j=json.loads(write_manifest_bytes('baseline',1))
                mutate(j)
                self.path.write_text(json.dumps(j))
                self.path.chmod(0o600)
                with self.assertRaises(ValueError):self.load()

    def test_symlinks_world_readable_or_oversized_manifest_rejected(self):
        valid=self.save()
        self.path.chmod(0o644)
        with self.assertRaisesRegex(ValueError,'untrusted'):
            self.load()
        self.path.chmod(0o600)
        self.assertEqual(1,self.load().generation)
        self.path.unlink()
        target=self.path.with_suffix('.safe')
        target.write_bytes(valid)
        target.chmod(0o600)
        self.path.symlink_to(target)
        with self.assertRaises(OSError):self.load()
        self.path.unlink()
        self.path.write_bytes(b' ' * 4097)
        self.path.chmod(0o600)
        with self.assertRaisesRegex(ValueError,'untrusted'):
            self.load()

    def test_unknown_workers_invalid_count_and_prior_stale_digest_rejected(self):
        self.save()
        for invalid in ('third','',None):
            with self.subTest(worker=invalid),self.assertRaises(ValueError):
                current_assignment(self.path,worker=invalid,default=self.default['q4'],shard_count=4)
        for count in (1,2,True,33):
            with self.subTest(count=count),self.assertRaises(ValueError):
                current_assignment(self.path,worker='coder',default=self.default['coder'],shard_count=count)
        with self.assertRaises(ValueError):self.load(previous_generation=5)
        with self.assertRaises(ValueError):self.load(previous_generation=0,previous_digest='x'*64)
        with self.assertRaises(ValueError):write_manifest_bytes('unsafe',1)
        with self.assertRaises(ValueError):write_manifest_bytes('balanced',0)


if __name__=='__main__':
    unittest.main()
