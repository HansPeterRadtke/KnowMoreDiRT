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
