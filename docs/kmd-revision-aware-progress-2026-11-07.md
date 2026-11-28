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

## Incremental audited completion journal without dropped original sources

Original recursive-summary workers write real
`KMD_HIERARCHICAL_DOCUMENT_COMPLETE` events to their existing systemd
journals. The earlier one-shot provenance script sampled only the
newest limited group. If more than that group finished between
independent checks, a worker could silently skip older completions
during quality review, despite the original indexed documents
remaining intact. This was a **gap in incremental verification**,
not evidence that a native source was lost.

The new optional `scripts/raspi_incremental_summary_integrity.py`
reads **the oldest pending completion records first**, across both
existing source-summary worker journals, in batches of at most
**twenty-five events**. Each journal event only selects a candidate:
the auditor independently rehashes the approved publisher original
and current derivative, checks the original approved acquisition
receipt and current source-map revision, and validates the **entire**
current native summary graph including source spans, child coverage,
summary text, stored vector bytes and provenance identifiers. A
completion-event message alone is not acceptance.

On its first invocation the operator must specify the **exact UTC
beginning** of the desired review interval. The prior interval is
*not* implicitly covered. Later invocations supply an existing
owner-private cursor file, which is accepted only when its preceding
private immutable audit receipt still matches its SHA and recorded
last event. Before resuming, it additionally demands that the
previous exact source-completion cursor is still present in the
retained systemd journal; if the journal has been vacuumed
or lost that record, the auditor fails closed instead of
mistaking an unavailable prior cursor for an empty new page.
The program never uses a journal `-n` newest tail,
never guesses a missing checkpoint, and never silently advances
past an unverified or oversized original. A failed or stale source
receives a private failure receipt and leaves the cursor unchanged.
An empty journal page also leaves the cursor unchanged.

A private non-blocking cursor lock prevents two overlapping owner
turns from racing to replace the last verified event. The lock
applies only to the optional audit checkpoint, never the two
production indexers or either source database. It safely refuses
concurrent auditors instead of waiting or resetting someone else's
progress. Retain the small cursor file, completed SHA-checked audit
receipts, and lock file in owner-private storage; do not commit
originals, source text or private audit logs to Git.

After securing an appropriate resource-broker CPU/RAM admission,
the first example invocation is:

~~~bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /data/var/knowmoredirt/semantic-venv/bin/python \
  scripts/raspi_incremental_summary_integrity.py \
  --source-map /data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3 \
  --native-db /data/var/knowmoredirt/knowledge-library/filesystem.sqlite3 \
  --original-root '/data/var/Knowledge Library' \
  --derived-root /data/var/knowmoredirt/knowledge-library/semantic/text \
  --completed-ledger '/data/var/Knowledge Library/_management/status.json' \
  --cursor-file /data/var/knowmoredirt/knowledge-library/search-audit/PRIVATE-AUDIT/cursor.json \
  --output /data/var/knowmoredirt/knowledge-library/search-audit/PRIVATE-AUDIT/page01.json \
  --initial-since '2026-11-28 10:40:00' --batch-size 4 --max-original-mib 128
~~~

For a subsequent page use a **new output filename**, the same
cursor path and the same bounds, but **omit `--initial-since`**.
The current approved private evidence sample used two successive
four-event pages and independently verified **eight original files**
and **thirty-four current native summary nodes**, with no duplicate
event cursors or source-graph failures. A third read found no
new completion events and correctly left the cursor unchanged.
When two more completed originals appeared later in the actual
worker journal, the same source cursor continued with those
two records, bringing the observed total to **ten distinct current
originals and forty-six current native summary nodes**. None of
the previous eight events was replayed. This proves resumable
retrieval **from that explicitly selected UTC start while the
systemd journal retains the required records**, not coverage
of the entire historic journal or all seventeen thousand originals.

The audit is strictly read-only toward original library/source-map/
FileDB content: it neither downloads papers nor invokes a model nor
writes summary vectors. Each audit receipt and checkpoint is local,
private and owner-owned. A genuine source revision or missing native
child causes a fail-closed source failure that must be investigated;
it is not authority to delete, refresh or rewrite an original.
The underlying original/derived/source-map and native SQLite
writers are separate and are **not** an atomic shared generation.

## Publisher original source-unit labels are not citation text — 28 November 2026

The dual native search compares model-generated **summary candidates**
with separately ranked **original source-passage candidates**. Original
and derivative SHA verification protects revision identity, but it must
not turn an extractor-written source-unit label into publisher evidence.

A reproducible prior negative control demonstrated a real defect in the
original-passage verification helper: a derivative containing
a marker named PDF page 13 could satisfy an exact publisher-source
query for the words PDF page 13, **even when no publisher page body
contained them**. Later PDF page labels and generated KMD metadata
had similar exposure. Whole-source SHA equality did not fix that
semantic provenance mistake.

The read-only streaming verifier now searches only the **body bytes
after a complete source-unit marker and before the next marker**.
Generated catalog metadata and the extractor's own page labels cannot
establish publisher evidence. Exact phrases may cross file read-block
boundaries within one publisher unit, but never cross into metadata or
two independently extracted PDF pages. The verifier still streams and
hashes the **entire** current derivative even after a phrase matches;
original-source SHA, approved download receipt, native FileDB chunk
identity and exact chunk digest checks remain mandatory. Invalid
or unterminated source-unit labels fail closed, rather than counting
the rest of the file as publisher prose.

When a native excerpt consists of a generated marker or spans page
boundaries, the optional audited-abstention mode may omit that
uncitable passage. The generated summary remains only a navigation
hint. If this reduces recall, inspect the exact original source by
path and select a genuine bounded publisher excerpt; do not loosen
the source boundary rule, fabricate a citation or claim an original
PDF page is physically verified from its extractor page label.

The real approved Blender IES manual and a real archived Asuka
Bench research PDF were checked directly, without new source
downloads, live database writes or model calls. Together five
extractor-written label/control probes were rejected and two
actual source-text passage probes succeeded. Both publisher originals
and current derivatives passed full physical SHA checks, with
acquisition receipts and unique native original IDs rechecked after
the source query. This establishes source identity and genuine
derivative-text location, **not** physical PDF page geometry,
authorship of a generated summary or scientific answer correctness.

A second existing read-only post-hoc source reviewer used
an initial publisher source-unit boundary but could accept
an exact-source SHA-backed review anchor **inside a later
extractor-written PDF page label**. Its current review
anchors now must lie wholly inside one complete publisher
source body, rather than within a later marker or spanning
two separate pages. An unclosed or oversized source-unit
delimiter makes the review abstain. These checks preserve
original/derived SHA verification and leave physical PDF
page confirmation separate; a human review grade is not
automatically relevant or factually true.

The dual-source CLI also reads the canonical source-map using a
bounded **twenty-second query-only SQLite reader** and closes the
connection after verification. This corrects its last direct
five-second default reader under legitimate source metadata commits.
Disposable real SQLite exclusive writer tests held the source-map
lock longer than five seconds: a valid original returned once it
was released, while a source revision committed during the wait
correctly failed closed. The CLI preserves the original independent
source and summary ranks and does not change either index or writer.
