# Revision-aware recursive summary completion and real text-volume progress

Recorded on Raspi, 7 November 2026. This isolated KnowMoreDiRT increment
does not replace the original FileDB, source library, extraction service,
recursive language model, embedding model or semantic query ranking.

## Root cause and correctness

The continuous summarization worker originally treated an original as
finished if **any** summary representation existed for its FileDB entry.
That skipped the source after it was re-extracted with a different derived
text revision. It would also skip an interrupted or partial source summary
if such rows were ever present. The query-side revision verifier already
refused old source spans, but the worker would not recreate them, leaving
the document indefinitely missing from summary search.

The authoritative completion check now requires a native summary_long
**root** at source offset zero, with a complete success status, a
nonempty summary SHA-256 that matches its stored text, the current
original source SHA, the current source-map derived-text SHA, and
source_tree_sha256 exactly equal to that derived revision. It verifies
the source_start/end source-span metadata and requires the exact
original relative path. Leaves, stale originals, stale extracted
text, bad source-tree hashes, mismatched summary bytes, partial
source offsets and discarded/retired source rows do **not** count
as completed. It needs no alternate database and only SELECTs
against the canonical KMD tables and source-map.

The source-map text_characters count is text content **excluding**
KMD's provenance header, while native source spans cover the entire
derived UTF-8 file including that header. Comparing raw root
end_char to source-map text_characters falsely rejected every one
of the existing known-complete roots. Instead completeness compares
the full-source SHA-256 recorded for the summary tree to the exact
current derived-file SHA. A read-only check of the existing live
roots verified this source-tree SHA convention.

On source revision change, the native saved summary rows are left
untouched as historical derived records. The worker regenerates
a complete hierarchy with the new hash; the original native FileDB
accepts version-keyed additional rows. Search already enforces the
current original+derived revisions and therefore does not present
stale rows as current. No valuable source files, vectors or prior
summary history were deleted.

## Honest progress measures

The previous progress indicator counted the number of completed
documents, not the amount of source material processed. Because the
worker schedules short documents first, it overstates workload
progress if interpreted as a proportional text-work metric.

The worker and the native operator status now expose **both**:
- completed original documents / eligible originals (document_percent)
- sum of source-map extracted text characters for current-revision
  completed originals / sum of all eligible source text characters
  (text_percentage)

The two percentages are deliberately separate. Text percentage is
still not a prediction of exact inference time: prompt batching,
chapter boundaries, local model response lengths, cache hits,
thermal load, and very large books influence actual processing.
The progress implementation refreshes from the authoritative native
root revisions every ten successes or on worker restart, and updates
both counters after each successful complete import.

At discovery, approximately 220 of 17,714 documents were complete,
but their 473,419 extracted source text characters represented only
about 0.0167 percent of the 2,836,348,592 total extracted characters.
This is expected from small-first ordering, not database corruption.

## Test and deployment conditions

The new summary_coverage module is pure, read-only code usable by the
worker and status CLI. Disposable SQLite tests cover current roots,
stale original revisions, stale derived revisions, missing parent/root
nodes, bad summary SHA-256, bad source-tree SHA, wrong source offset,
source text length metrics, empty indexes and source revisions changing
between runs. The original KMD FileDB remains the sole authority for
original entry IDs and vector storage.

The update should be integrated into the pinned active KMD worktree
only after tests, then the existing owner-authorized systemd worker
restarted at a completed-document boundary. It must immediately
show correct revision-based and text-based counts without losing
previously generated vectors or stopping the LLM indexing task.

## Live reader safety during two concurrent summary writers — 28 November 2026

The full-library summary task now has two source-disjoint background
writers feeding the SAME native FileDB. In its existing DELETE-journal
mode, an authorized summary-vector import may temporarily hold an
exclusive SQLite lock. That is not evidence that a publisher
original was deleted, that its checksum is invalid, or that the
reader should create another database or stop the indexer.

The actual unchanged Unity UI-event reference question failed
during its native-summary vector SELECT with an SQLite
OperationalError (database is locked). Its local native search
entrypoint used a bare read-only sqlite3.connect and SQLite's
five-second default wait, despite an already established
open_native_summary_read_connection helper with a bounded
120-second wait, query-only mode and source file safety checks.
The coverage status command had the same bare five-second
native connection and independently failed in a real live read
while the two workers were active.

The search and status commands now **reuse the existing
bounded native read-only helper** and always close the connection
after reading. Status separately opens the original source-map
as read-only with a bounded twenty-second metadata wait and
closes that read handle as well. This changes neither the
summary-ranking model nor source identity and never takes
a writer lock, switches SQLite journal mode, retries a denied
provider call, creates another index or stops existing workers.

Disposable, real SQLite DELETE-journal tests held
BEGIN EXCLUSIVE beyond five seconds while native **search**
and **status** attempted the same original FileDB. Both
remained pending and returned validated results once the
writer rolled back. A third independent source-map exclusive
writer test confirmed bounded source metadata status recovery.
Two separate persistently held native locks deliberately
exhausted a **one-second** test-only bound and raised
SQLite errors, without inventing source results or waiting
indefinitely. The production native reader cap remains
**120 seconds**; a genuinely longer writer conflict
can still fail closed.

The current real read-only native summary search for the
previously frozen Unity UI-event example returned the same
approved original at rank three with current publisher-original
and extracted derivative SHA and unique native FileDB identity
verified after ranking. The repaired real status returned
**10,751 completed current-revision recursive original roots
of 17,715 eligible originals**, with **6,964** still incomplete,
and **25,399** native summary vectors. The live source-map
eligible count was independently read back and matched the
status denominator. Counts can change as legitimate workers
continue; the status is a dated point-in-time observation,
not an atomic generation spanning separately updated stores.

A separate owner-private audit of the latest fifteen
journal-confirmed completed research originals independently
rehashed their approved publisher files and derivatives and
checked **seventy** native summary-node source spans,
embedding/vector hashes, and parent-child identity, with
no provenance failures. It is a sample, not proof of integrity
or answer correctness of every eligible original.

Operational acceptance: keep both indexing workers running,
query and inspect current original sources, distinguish a
legitimate timed-out lock from corrupted source data,
and continue to validate current SHA/native IDs. Never replace
ordinary folder/grep discovery, raise the configured busy
bound without evidence, or treat a model answer as a
verified original passage.

## Separate source-map revision writer and original evidence readers — 28 November 2026

The native FileDB writer-lock repair does not by itself cover the
**independent extraction/source-map database**. During a legitimate
original-source import or revision update, that SQLite database can
hold a write lock while a KnowMoreDiRT reader is selecting the
current extracted document SHA. The original `_current_versions`
and `checked_original` paths were still opening this source-map
with SQLite's default five-second wait. An ordinary writer commit
could therefore interrupt original-source retrieval or be confused
with missing knowledge.

The shared reader now uses one canonical source-map connection
helper with a **bounded twenty-second wait**, `mode=ro`,
`PRAGMA query_only=ON`, canonical non-symlink file validation
and explicit connection closing. Original-source lookup,
initial and final source-revision readback, and live summary
status use the same helper. A source-map revision committed
while a reader is waiting does **not** authorize an old result:
after the writer finishes, the physical derivative SHA must
still match the current source map and approved original
receipt, otherwise the query fails closed.

Related original evidence readers now also reuse the **existing
bounded hundred-and-twenty-second read-only native FileDB
helper**, including selected original identity, verified PDF
evidence and final source-linked summary checks. Their
connections are closed after SELECTs to avoid preventing the
next legitimate indexing transaction. No source-map or
native FileDB writer behavior, journal mode, original file,
model, summary content, vector ranking or service was changed.

Five isolated SQLite concurrency tests exercised a real
source-map exclusive writer held longer than the previous
five-second read timeout, strict query-only mode, unsafe
path and timeout refusal, persistent writer bounds, and a
**changed derived source SHA committed while an original
reader was blocked**. After release, the current original
was retrieved or the revised derivative failed closed,
never silently replaced by stale text. The broader
original-source, publisher evidence, summary search and
revision-check suite passed **one hundred seventy-three**
tests. These are synthetic writer tests, not a claim that
a live source-map writer was observed holding a lock.

A live read-only verification against the actual archived
Unity UI-event original returned **one** source-linked
passage and the same source at native summary-search
**rank three**. The actual publisher original, derived
text and unique native FileDB entry were independently
SHA/identity-checked after both queries. A separate
incremental audit of **six newly completed real original
documents** verified their full approved source and
derivative bytes and **twenty-eight** source-linked
native summary nodes, including node spans, vectors
and graph relations, with no provenance failure.
A successful source hash check verifies evidence identity
and structure, not a factual answer or the physical
appearance of every publisher figure.

The source-map and native FileDB remain **separate
writer-managed SQLite stores**, so the repeated current
SHA checks are not an atomic shared generation or
a permission for an uncoordinated live index cutover.
If a real writer holds its lock beyond the configured
bounded wait, the read fails and can be investigated;
never start an unapproved second FileDB, discard its
evidence or change the live writer as a workaround.
