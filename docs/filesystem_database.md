# Filesystem semantic database subsystem

KnowMoreDiRT now includes the filesystem catalog code from the `devtests/file-system` repository as an isolated package named `file_system_catalog`. The devtest copy remains unchanged for independent research. The KMD copy is vendored so the fast semantic database can be initialized and queried without building the slower DRT/DSPG world model.

The subsystem scans arbitrary folder trees into SQLite, preserves lossless filesystem metadata, extracts text, chunks files, creates embeddings, optionally creates LLM-generated semantic representations, supports literal and semantic retrieval, plans bounded searches with a constrained local model, and produces grounded answers with explicit evidence records.

It can be used directly through the vendored package or through the KMD facade:

```python
from knowmoredirt.filesystem import (
    initialize_filesystem_database,
    question_filesystem_database,
)

initialize_filesystem_database(
    "/path/to/raw/folder",
    "/path/to/catalog.sqlite3",
)

result = question_filesystem_database(
    "/path/to/raw/folder",
    "/path/to/catalog.sqlite3",
    "Which report mentions the failed pressure test?",
)
```

The command-line entry points are `kmd-filesystem` for initialization and grounded questions and `kmd-filesystem-search` for direct semantic or literal retrieval.

This subsystem does not initialize DRT, create persistent KMD referents, or perform DRS proof validation. It is the fast retrieval and basic LLM question-answering layer. Future integration will let KMD consume this database for candidate evidence, mention profiles, and referent-resolution candidates while retaining DRT/DSPG as the persistent identity and reasoning layer.

The source was copied from devtests commit `6faaccc9`. Regression tests from that repository are preserved under `tests/filesystem_database/`. Any future synchronization must be explicit and reviewed; changes in either copy do not automatically propagate to the other repository.

## Verified original passage inspection, November 2026

The [source-evidence and abstention research](kmd-source-evidence-and-abstention-2026-11-07.md)
measures false positives from naive similarity and lexical thresholds,
then adds a bounded read-only original-page locator. Every candidate retains
original source offsets and hashes; no score is promoted to factual support.

## Per-node source progress receipts, November 2026

The [recursive summary node-progress audit](kmd-summary-node-progress-2026-11-07.md)
stores private atomic progress after every verified leaf or parent
summary, without copying source text or claiming incomplete trees
are indexed. The existing native status command can inspect progress
for the currently active and revision-matched original source.

## Mixed source-length recursive summary scheduling, November 2026

The [balanced library schedule](kmd-balanced-recursive-library-2026-11-07.md)
interleaves smaller originals with regular medium and larger documents,
allowing actual multi-level summary indexing to be validated earlier
without increasing model concurrency or changing the canonical FileDB.

## Batched native semantic search scalability, November 2026

The [native search scaling report](kmd-summary-search-scaling-2026-11-07.md)
compares bounded batched cosine calculation and deferred source metadata
loading with 10,000/30,000-vector and 27 real query equivalence tests.
Original KMD FileDB authority, source revisions and ranking remain unchanged.

## Version-aware complete source roots and text-volume progress, November 2026

The [revision-aware completion and progress audit](kmd-revision-aware-progress-2026-11-07.md)
requires a current full-source root before skipping any original, and
separately reports document percentage and actual source-text-character
percentage. Source-map text-character counts exclude extraction headers,
so full-source SHA-256 is used instead of comparing incompatible offsets.

## Source-complete concise retry and ranking rejection evidence, November 2026

[The summarization recovery and ranking audit](kmd-summary-retry-and-ranking-audit-2026-11-07.md)
records a source-grounded short-output fallback for real local-model
length exhaustion, stale-failure progress cleanup, and an exploratory
hybrid ranking sweep followed by an unused held-out source test.
A blended ranking that worsened holdout first-place accuracy was *not*
enabled; native summary-first semantic ranking and query logging are preserved.

## Independent real-query summary retrieval QA, November 2026

The [native summary retrieval benchmark](kmd-summary-retrieval-quality-2026-11-07.md)
documents easy and paraphrased real searches, German-language queries,
unsupported-topic false positives, a fair same-document old-excerpt versus
new-summary comparison, and persistent private query/rank JSONL logging.
Search is usable for already summarized originals but has not been shown to
be reliable across the full Knowledge Library.

## Actual Raspi hierarchical summary search pilot, November 2026

The verified two-source native KMD full-document experiment is documented in
[live summary search acceptance](kmd-hierarchical-live-acceptance-2026-11-07.md).
It includes an actual recursive summary-of-summaries and all summary-node
vectors in the original FileDB. Whole-library summarization remains pending.

## Experimental complete-document hierarchical summaries (November 2026)

The source-complete recursive summary engine, source-linked native KMD writer
and real local-model experiments are in the [hierarchical summary report](kmd-hierarchical-summary-2026-11-07.md). These modules are opt-in and not
yet deployed into the live Raspi index: the Infra query resolver must first
recognize summary nodes and protect their source-level provenance.

## Experimental library retrieval extension (November 2026)

A separate KMD-side, in-memory FTS5 and reciprocal-rank-fusion **candidate**
component is documented in [the hybrid Raspi retrieval experiment](kmd-hybrid-raspi-retrieval-2026-11-07.md).
This is tested but not connected to the production FileDB search entry point;
the canonical original-file database and semantic vectors remain unchanged.

## Full Raspi Knowledge Library deployment (October 2026)

The owner also uses this original filesystem subsystem for the complete offline Knowledge Library on host `raspi`; the library contains PDFs, HTML, Markdown, original spreadsheets and ZIP courses, so **do not** call the text-only semantic `kmd-filesystem init` on that raw source root. Its `ContentSemanticPipeline._extract_text` decodes file bytes as text and will otherwise embed meaningless binary PDF/ZIP bytes. The original metadata scanner works on any file kind; semantic text extraction is a separate capability that requires tested format-specific extraction and a reachable local embedding model.

The **Infra-owned** Raspi command is `/data/bin/knowledge-library-filesystem`; full usage and deployment verification steps live in `/data/infra/docs/knowledge-library-knowmoredirt-filesystem.md`. The actual source corpus is `/data/var/Knowledge Library/`, while its private metadata SQLite is `/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3`. The database must stay **outside** the scanned root and all Git repositories. The deployed code lives in a clean, pinned checkout of KnowMoreDiRT; the original local `master` and separate discourse-research worktree were not reset or edited.

Usage on Raspi:

```bash
# Check actual original-file coverage, schema and SQLite integrity:
/data/bin/knowledge-library-filesystem status --verify

# Search KMD native filesystem metadata, no LLM involved:
/data/bin/knowledge-library-filesystem find --name marketing --path money-and-business/ --limit 10
/data/bin/knowledge-library-filesystem find --extension pdf --path money-and-business/ --limit 20
/data/bin/knowledge-library-filesystem find --extension pdf --sort size_desc --limit 10

# Initialize from scratch only if the database does not yet exist:
/data/bin/knowledge-library-filesystem scan --fast-samples --skip-exiftool --max-hash-bytes 65536 --progress-every 1000

# Atomic metadata refresh after library additions. Refuses to erase populated semantic vectors:
/data/bin/knowledge-library-filesystem refresh --fast-samples --skip-exiftool --max-hash-bytes 65536 --progress-every 1000
```

`--fast-samples` and `--skip-exiftool` are **Infra wrapper options**, not an option in upstream `python -m file_system_catalog`. It invokes the unmodified original KMD scanner, but caps the per-file byte-structure histogram at a one-megabyte exact-analysis threshold and three sixty-four-kilobyte samples for larger items. The resulting original metadata indicates `analysis_complete=false` for a sampled file. The deployed wrapper additionally skips ExifTool deep decoding on all files to reduce CPU load, explicitly stores its unavailable status and retains other original KMD PDF/Office/archive parsers. This choice reduces detailed Exif coverage, **not the number of indexed filesystem entries**. The hash cap independently marks files larger than sixty-four kilobytes `skipped_large` with head/tail hashes; it does **not** invent a complete SHA-256. For the original downloaded files, the separately maintained offline Knowledge Library downloader ledger retains their full SHA-256 verification, so the cheaper native KMD metadata scan need not reread the entire multi-gigabyte archive. This distinction remains visible in the status output. All original files and native metadata types are included in the scan, with KMD's unchanged schema and original source paths.

`status --verify` checks that all currently downloaded originals are present in native `filesystem_entries`, the source root is correct, the SQLite schema/integrity/foreign keys are valid, and reports whether native `content_chunks` and `content_representations` have been populated. Metadata alone is not a completed semantic vector index; preserve that limitation for agents and user reports. Full text PDF page search currently also exists in a **separate** Infra-managed, business-only lexical index; it is not part of this native metadata database. All downloaded text, full-file caches, outputs, SQLite, source inventories and privacy-sensitive metadata remain exclusively under private `/data/var`, not in the KMD or Infra Git repository.

**Initial live acceptance (October 30, 2026):** The original KMD filesystem scanner completed on the full Raspi Knowledge Library, producing the three original SQLite tables and 13,798 real filesystem entries (12,764 files and 1,034 directories); all 12,311 downloader-verified original books and documents matched their KMD paths and byte sizes. There were no scan/hash errors, missing originals, or SQLite integrity/foreign-key failures. KMD counted 10,062 PDFs with parsable page metadata. Poppler emitted 165 nonfatal PDF metadata syntax warnings; those affected originals retained PDF page counts and sampled files were independently readable. The bounded mode intentionally skipped deep ExifTool and full KMD hashing of files exceeding sixty-four kibibytes; original download-ledger full SHA-256 values remain separately available. The semantic content tables are empty and must not be reported as a full vectorized knowledge index. The private complete acceptance report and source inventory live outside Git in the Raspi KMD database directory. Always use live `status --verify` for newer data.
