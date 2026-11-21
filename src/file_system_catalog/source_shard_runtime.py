"""Optional validated source-shard ownership at individual document boundaries.

No manifest exists by default: worker assignments stay exactly as installed.
A separately coordinated two-phase cutover can release a hash bucket, wait
until the releasing worker is observed outside its document, and only then
assign the bucket to the other live worker. No models or services are stopped
by this module, and no native FileDB writes are performed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import NamedTuple

SCHEMA = 'kmd.two_worker_shard_handoff.v1'
MAX_CONFIG_BYTES = 4096
# These are the exact existing four-bucket source assignments.  Stage one
# deliberately leaves bucket three unowned rather than ever double-owned.
KNOWN_STAGES = {
    'baseline': {'coder': frozenset((1, 2, 3)), 'q4': frozenset((0,))},
    'release_three': {'coder': frozenset((1, 2)), 'q4': frozenset((0,))},
    'balanced': {'coder': frozenset((1, 2)), 'q4': frozenset((0, 3))},
}


class Assigned(NamedTuple):
    shards: frozenset[int]
    generation: int
    digest: str | None
    stage: str | None


def current_assignment(path: Path, *, worker: str,
                       default: frozenset[int], shard_count: int,
                       previous_generation: int = 0,
                       previous_digest: str | None = None) -> Assigned:
    """Read an optional atomic, owner-only assignment manifest fail-closed.

    A manifest previously used by the current process cannot be deleted,
    rolled back in generation or rewritten at the same generation. Current
    caller never changes ownership mid-document and must publish application
    state at the next source boundary before an operator advances the stage.
    """
    if (worker not in ('coder', 'q4') or type(shard_count) is not int
            or shard_count != 4 or type(previous_generation) is not int
            or previous_generation < 0 or not isinstance(default, frozenset)
            or not default or any(type(z) is not int or not 0 <= z < 4 for z in default)):
        raise ValueError('invalid source-shard runtime baseline')
    if previous_digest is not None and (not isinstance(previous_digest, str)
                                        or len(previous_digest) != 64):
        raise ValueError('invalid previous manifest digest')
    if previous_generation == 0 and previous_digest is not None:
        raise ValueError('untrusted zero-generation manifest digest')
    if previous_generation > 0 and previous_digest is None:
        raise ValueError('missing previous live manifest digest')
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError('live assignment parent unavailable or symlinked')
    if not path.exists():
        if previous_generation or path.is_symlink():
            raise ValueError('live source allocation manifest was removed or symlinked')
        return Assigned(default, 0, None, None)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_size > MAX_CONFIG_BYTES
                or before.st_size < 30 or stat.S_IMODE(before.st_mode) & 0o077
                or before.st_uid != os.geteuid()):
            raise ValueError('untrusted owner-only source allocation manifest')
        blob = os.read(fd, MAX_CONFIG_BYTES + 1)
        after = os.fstat(fd)
        if (len(blob) != before.st_size or len(blob) > MAX_CONFIG_BYTES
                or not os.path.samestat(before, after)
                or before.st_mtime_ns != after.st_mtime_ns
                or before.st_ctime_ns != after.st_ctime_ns):
            raise ValueError('source allocation manifest mutated during reading')
    finally:
        os.close(fd)
    digest = hashlib.sha256(blob).hexdigest()
    try:
        manifest = json.loads(blob.decode('utf-8'))
    except (UnicodeError, ValueError) as error:
        raise ValueError('source allocation manifest is not strict JSON') from error
    if not isinstance(manifest, dict) or set(manifest) != {'schema', 'generation', 'stage', 'assignments'}:
        raise ValueError('wrong source allocation schema or unknown fields')
    if manifest['schema'] != SCHEMA:
        raise ValueError('unknown source allocation protocol')
    generation = manifest['generation']
    if type(generation) is not int or not 1 <= generation <= 1_000_000_000:
        raise ValueError('invalid source allocation generation')
    if (generation < previous_generation
            or generation == previous_generation and previous_digest is not None and digest != previous_digest):
        raise ValueError('source allocation manifest generation replay/rewrite detected')
    stage = manifest['stage']
    if stage not in KNOWN_STAGES:
        raise ValueError('unknown source allocation handoff stage')
    records = manifest['assignments']
    if not isinstance(records, dict) or set(records) != {'coder', 'q4'}:
        raise ValueError('source allocation requires both known worker labels')
    observed = {}
    for label, indices in records.items():
        if (not isinstance(indices, list) or not indices
                or any(type(z) is not int or not 0 <= z < 4 for z in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('invalid assigned source bucket list')
        observed[label] = frozenset(indices)
    if observed != KNOWN_STAGES[stage]:
        raise ValueError('source allocation differs from tested staged release protocol')
    if observed['coder'] & observed['q4']:
        raise ValueError('duplicate shard ownership forbidden')
    return Assigned(observed[worker], generation, digest, stage)


def write_manifest_bytes(stage: str, generation: int) -> bytes:
    """Pure encoder for an independently gated atomic operator, not a writer."""
    if stage not in KNOWN_STAGES or type(generation) is not int or generation < 1:
        raise ValueError('invalid staged assignment')
    data = {'schema': SCHEMA, 'generation': generation, 'stage': stage,
            'assignments': {k: sorted(v) for k, v in KNOWN_STAGES[stage].items()}}
    return (json.dumps(data, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')
