# Concurrent original-KMD summarization with Qwen3.8 Q4 and Qwen Coder

As of November 8, 2026, the complete Raspberry Pi Knowledge Library has 17,714 verified extracted eligible originals, approximately 2.84 billion extracted text characters. The existing Thor voice-assistant Coder endpoint and Qwen3.8 Q4 endpoint were both already loaded and accessible. No additional model download, model service restart, alternate FileDB, or transcription/voice service change is necessary.

## Verified parallel inference pilot

The existing Coder-backed production indexer remained active on the canonical original KMD FileDB while a separate Qwen3.8 Q4 test summarized the complete W3C WAI Writing for Accessibility source (16,282 verified extracted characters) into the **same native KnowMoreDiRT facet JSON schema** without writing the production FileDB. Both llama.cpp slots reported inference in progress simultaneously. Qwen3.8 finished in 121.55 seconds, produced a complete source-linked root, 28 keywords and 8 keyphrases, with validated native summary provenance. Coder continued indexing its own source queue during the test. Thor memory remained roughly 105–106 GiB used of 122 GiB, services had no unexpected restarts and measured GPU/temperature remained within observed normal load. The additional Q4 test was deliberately read-only.

An even longer full 2.75-million-character HCI textbook already completed Qwen inference, tree verification and native embedding successfully; those previous results remain intact.

## Worker coordination without duplicate sources

The original persistent worker now optionally accepts a static deterministic source partition. The owner-approved original relative path is hashed with SHA-256; the unsigned first 64 bits are reduced modulo the configured shard count. All revisions of the SAME original go to the SAME shard. The two services receive nonoverlapping assigned shard sets and therefore cannot generate the same original simultaneously. Every original is covered exactly once across the assignments. New originals are assigned stably using the same hash without changes to source-map or publisher files.

Based on actual measured source traffic, Coder typically handles about three times as many small documents per hour as Qwen3.8 Q4, though there is substantial variance by source length. The four-way assignment therefore gives the Qwen3.8 worker shard zero and the Coder worker shards one, two, and three. On the current real library manifest, this gives Q4 4,437 originals and 27.06 percent of all extracted characters, and Coder 13,277 originals and 72.94 percent. This is a throughput-motivated division, not a claim that their long-term speeds are fully measured or that the split is permanently optimal.

Each worker retains an independent persistent run lock, status file and retry list. The existing main worker retains the canonical .hierarchical-worker.lock and hierarchical-worker-status.json; the new Q4 worker uses .hierarchical-worker-q4.lock and hierarchical-worker-q4-status.json. Neither introduces another FileDB or shadow semantic index. Both read source-map and the original native KMD FileDB and append source-linked chunks and representation vectors using independent SQLite connections. Actual FileDB writer transactions use BEGIN IMMEDIATE with native foreign keys, complete source IDs, SHA checks and file-size checkpoint. Any short contention or failure is retried through the existing worker retry machinery.

Status reports include global completed-source numbers and the worker's own partition counts, to avoid confusing the main worker's global coverage with each shard's individual workload. They do not count in-progress book branches as complete.

## Test proof and limitations

Five new deterministic sharding tests verify full path coverage without overlap, stable assignments independent of content revisions, invalid/duplicate shard rejection, independent worker locks and statuses, and actual concurrent SQLite native-row inserts from separate connections into the same temporary canonical-shaped original FileDB. The previous semantic, provenance, recursive summary and status tests also pass.

This prevents duplicate processing through deterministic source assignment; it is not a distributed leader-election mechanism and requires **both services to run with the same shard count and disjoint shard ownership**. Never run an unsharded original worker concurrently with a sharded worker. If one service is disabled long-term, either restore the unsharded solo unit safely or reassign its shards to the other worker at a completed-source boundary.

Voice safety remains an important caveat. The Coder server has one llama.cpp inference slot shared with the SWAAG foreground voice assistant. The existing Coder worker waits eight continuously idle seconds before each uncached LLM request. This does NOT preempt a running book-summary call when a voice request arrives. Running a second standard-Qwen model concurrently can also contend for Thor GPU bandwidth even with separate inference slots. Monitor queue waiting time, inference throughput, thermal/DRAM load, native DB errors and source-specific quality. If voice latency deteriorates, stop the secondary Q4 worker and restore an unsharded Coder worker, or reduce the Q4 workload before further tuning.

Use the original source-controlled KMD worker and Infra systemd units rather than untracked live scripts. Preserve all owner-only source and evaluation data outside Git.

## Live original-FileDB dual-worker proof, November 8

The Qwen Coder service was restarted at a successful complete-document
boundary with shards one/two/three and its original unchanged persistent
worker lock and status. Only after verifying the running process and
nonoverlapping path assignment was the secondary Qwen3.8 Q4 systemd
service started with shard zero and its own independent lock/status.
The two processes and both Thor llama.cpp server slots were observed
active concurrently. The original canonical KMD SQLite quick check
reported 'ok' and foreign-key check reported zero violations while
both were active.

The secondary Q4 worker immediately completed and imported its first
real original, the official NVIDIA JetPack releases archive source,
which had previously been a source of retry errors under Coder. It
saved its complete root, original-linked keyword/topic representations
and source SHA metadata in the SAME original KMD FileDB, then started
a second distinct original. The Coder worker was simultaneously
processing a different original. Every observed current source had a
SHA-256 original-path shard assignment within its service's allowed
partition.

Thor showed approximately 113,464 MiB occupied of 125,750 MiB;
temperatures were about 67 degrees Celsius under both model calls,
and both inference model services and the voice gateway remained
running. These observations are a limited live acceptance, not proof
of complete absence of GPU contention or unmeasured human voice
response latency.

Additional source-level tests ensure that a failure receipt inherited
from the previous unsharded worker is not incorrectly reported as an
outstanding error of the newly unassigned shard, and that per-shard
completed-source and pending-source counts are incremented after
each successful import. The operator status command exposes the
two independent worker models/queues while still deriving total
library completion from the authoritative native FileDB.

## Production contention discovered and corrected

The live dual-worker pilot found a race that a fake SQLite writer test
did not initially exercise: the legacy Raspi import CLI uses the native
global `.primary-kmd-content.lock` to serialize final original FileDB
writes. The old one-worker importer acquired this flock in nonblocking
mode. With two independently scheduled models, source import times
can overlap. A real Coder source completed its whole Qwen generation
but temporarily failed its database import with BlockingIOError while
the Q4 worker held that original content writer lock. The worker's
retries preserved the tree and original data, but wasted scheduling
time and produced a false-negative completion status.

The original source-linked import now waits for this **same global
lock** with a bounded ninety-second maximum and a quarter-second
poll, then proceeds with the existing serial native SQLite transaction
and foreign-key/integrity validation. It does not weaken locking,
insert a second database or make concurrent commits possible. If an
unexpected writer holds the original lock longer than the limit, the
explicit timeout causes the existing bounded model/source retry,
rather than waiting forever or silently claiming a completed source.

A regression suite validates actual distinct file descriptors
contending on a disposable lock, releases after the first writer
finishes, retry count, bounded timeout and invalid parameters.
This is a source-controlled CLI fix; its next executed import subprocess
uses the corrected behavior without stopping an in-progress LLM call.

## Live original writer-lock acceptance

The corrected writer-admission routine was tested in the actual
installed Raspi KMD checkout against the *existing* original content
lock. A separate tester deliberately acquired the same original
FileDB flock. The normal authenticated source-checking import of the
already indexed NVIDIA JetPack archive proceeded through model-vector
validation, then waited while the test held the lock. The lock was
released after a short controlled hold; the importer finished with
exit code zero and `new_rows: 0`. No duplicate original chunk or
representation was created, and the previously saved summary and
keywords remained unchanged.

The re-import completed in about eight seconds including local
embedding startup. This verifies that a temporary real writer lock
collision is no longer recorded as a failed original source. If
writers contend for longer than ninety seconds, the fail-closed
timeout remains intact and the normal queued retry runs later.
