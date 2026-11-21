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

## Concurrent source-disjoint Qwen and Coder indexing, November 2026

The [parallel indexing audit and test plan](kmd-parallel-two-model-indexing-2026-11-08.md)
uses two already loaded Thor models, deterministic disjoint original-file
shards and independent worker status locks, while both write to the same
original native KMD FileDB. It covers throughput evidence, conflict tests,
resource limits, SWAAG voice latency and the safe rollback procedure.

## Voice Coder model benchmarking and safe sharing, November 2026

The [Coder model evaluation](kmd-coder-voice-model-acceleration-2026-11-08.md)
compares the existing voice-assistant model against Qwen3.8 on complete
source/keyword extraction. Model context is read from llama.cpp. A conservative
source-group cap retains detailed retrieval vectors, and optional idle-slot
admission yields the shared model to voice requests between calls.

## Original native KMD facets in recursive summaries, November 2026

The [native facet bridge](kmd-native-facet-integration-2026-11-07.md)
reuses the existing KMD file-facet schema and representation normalization
to generate and embed summary, topic, keyword and keyphrase records at
every recursive level in the existing original FileDB. A real Qwen
summary-plus-keyword call was stored and retrieved successfully.

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

### Native semantic discovery with publisher-verified PDF passages (November 14)

Raspi's summary operator supports
`knowledge-library-summaries find-pdf-evidence '<question>'` for a bounded
combination of native FileDB document discovery and physically checked
publisher PDF excerpts. It returns source identity, native rank, read-only
passage offsets and SHA-256s, and explicit verification/failure status; it
does not replace general text or semantic search, verify an answer's
truth, download new content, or create another FileDB. See
`docs/kmd-source-evidence-and-abstention-2026-11-07.md` for measured
retrieval and the audit/revision contract.

### Cross-format original-source evidence, November 14

Use `knowledge-library-summaries find-source-evidence '<question>'` for a
single original-preserving pass from native KMD semantic discovery to
original-SHA and derived-SHA-verified source passages, including HTML,
Markdown/skill, ZIP and XLSX references as well as physical-page-verified
PDFs. Use `--source-name '<exact catalog title / RFC / DOI / pinned skill>'`
when a named original is intended. A strict single-RFC content question may
also be resolved without a model ranking. Missing and ambiguous named
originals are **not** replaced with a similar publication. Non-PDF extraction
does not imply literal publisher byte equality; only passages with an
explicit `publisher_excerpt` byte proof carry that stronger claim.
The prior `find-pdf-evidence`, original metadata search and local
ordinary text search remain unchanged. Consult
`docs/kmd-source-evidence-and-abstention-2026-11-07.md` for verified
scope, negative controls and reproduction commands.

### Explicit short publisher-title references without unsafe topic routing

The native source-name resolver also recognizes the **exact publisher title
lead before a colon** in explicit source citations, for example:

```bash
/data/var/knowmoredirt/semantic-venv/bin/python \
  /data/src/github/KnowMoreDiRT/scripts/raspi_hierarchical_library.py \
  source-evidence 'According to CiteGuard, what is the false-discovery method?' \
  --top 3
```

A longer full catalog title remains an exact match with higher precedence.
Only an explicitly quoted name, a source-reference construction such as
`According to CiteGuard, ...`, or a source-qualified question such as
`What does the MatLat paper say ...?` can use a title lead. A bare topic,
a mere incidental mention, a comparison or an instruction to exclude a
source cannot silently force a named original. The lead must be an exact
normalized publisher-catalog prefix before a colon, not a substring, fuzzy
match or guessed expansion. Short four-character leads must be distinctive
capitalized/digit-bearing identifiers, not ordinary words. If multiple
originals/editions share the lead (for example two distinct `ToolScope`
papers), the resolver returns **ambiguous** including unsummarized catalog
records; select an exact full title or original path instead. A missing or
unindexed source is never replaced with a similar sounding publication.

A source match selects **only the archived publisher identity**, not an
answer. `source-evidence` then independently checks original file and
extracted-text SHA-256 revisions and requires physically verified original
PDF page text where possible. A query against ordinary summary search may
still return `no_verified_index_match` when that exact original has not yet
been summarized; source-evidence can inspect the completed original without
waiting for a summary. No vector score, embedding, corpus or default general
semantic ranking was changed by this exact-source routing.

Explicitly named *short-title* summary requests fail closed: if a matching
native summary is absent, stale or otherwise unverified, **no unrelated
ranked publication appears as a substitute**. An ambiguous shorthand also
returns no vector-ranked documents, but retains its exact possible catalog
originals for explicit disambiguation. A no-summary read first uses a small
SQL `EXISTS` gate in the one canonical FileDB, skipping even local E5 model
loading. When native summary candidates exist, a parameterized original-path
filter restricts the vector scan to the selected document, followed by all
existing original and derivative SHA checks. This is distinct from a general
unscoped question, which still runs the ordinary cross-corpus ranking.
The first live unsummarized CiteGuard example returned zero substitutions,
correctly directing the caller to original-only evidence; the originally
ranked unrelated `SKILL.md` and other candidates no longer appear in its
source-scoped result. The same real query declined from about three-point-
seven seconds in the old global scan to about one-point-five seconds with
the presence gate (four logged measurements, on a changing live corpus).
The speed observation is a local sample, not a guaranteed service latency.

The private Raspi `search-audit/continued-20261121-wave45/` contains the
publisher catalog/ledger SHA-frozen **twelve-case** named-source benchmark
created before this change. Previously the exact source-name resolver found
none of five explicit short names. The updated parser resolved all five
unique leads, retained ambiguity for both duplicated leads, and preserved
all five non-citation/coordination negative cases. This is deliberately an
**explicit source identity** result, not a claim about general semantic
retrieval recall. Live original-only reads verified `CiteGuard`, `MatLat`
and `P2GS` against current original/derivative digests and physically checked
PDF pages. No new original was acquired, no inference server was changed,
and independently judged answer correctness remains open.

### Bounded source-scope windows in mixed evidence, November 15

The combined original-linked operator supports
`knowledge-library-summaries find-source-evidence '<question>'
--scope-context-characters 320` by default, or zero to disable this
additional read-only context. It returns independently SHA-verified
section-opening, preceding and following *extracted-text* spans to keep
potentially governing quote, dream, conditional and supersession context
visible. Context spans do not borrow content from an adjacent original PDF
page or Markdown section. Their separate status explicitly does not
establish a physical publisher-page location, original-byte match or
real-world factual truth; only a separately checked `source_excerpt`
has its previously available publisher proof. Audit receipts contain
the context span coordinates/hashes **without** the extra quoted text.
See the dated source-evidence-and-abstention report for hard negatives and
the reproduction commands.

### Exact RFC sections through mixed-source retrieval, November 15

`knowledge-library-summaries find-source-evidence 'RFC 9110, Section 9.2.2'`
now selects that verified original RFC section structurally, instead of
searching for incidental copies of the section number in the contents,
references or another section. The body heading, table of contents,
original UTF-8 byte interval and source revision must agree; missing
sections fail closed. The result's `rfc_section_lookup.context.next_cursor`
can be passed to the same command using `--source-name 'RFC 9110'
--continue-section "$cursor" 'Section 9.2.2'`. Each continued window
retains publisher-byte proof and a source-bound cursor. Ordinary
semantic document queries and non-RFC original passages retain their
existing retrieval path, and this structural proof remains evidence,
not automatic correctness of a normative claim.

### Original-linked visual PDF evidence, November 15

The read-only scripts/raspi_visual_pdf_link.py verifies the original
source-map SHA, canonical native FileDB identity, exact PDF physical page,
independent page raster and crop SHA/pixel rectangle, plus an optional
same-page exact extracted-caption-prefix source span. It reports the actual
source-indexed embedding collections and deliberately keeps image-only
OCR failures without fabricated text. All results go to new private
revisioned audit files, not into the existing FileDB. See the dated source
evidence document for original-case checks and limitations.

### Native retrieval-summary length safety, November 19

Qwen-generated file facet JSON now has model-generation-time bounds on
its document summary field (1200 characters initially, 480 on the
strict retry), in addition to the existing strict 400-embedding-token
post-check. This prevents invalid unbounded summaries from being
silently stored or endlessly retried without a constrained alternative.
The original complete source and separate native facets are unchanged;
no generated string is truncated or treated as original publisher proof.
The EVATok source's previous nine failed retries are independently
recorded; actual completion requires a later real worker-run check.

### Full metadata and sampled original-source revision QA — November 20

The read-only scripts/raspi_source_revision_audit.py verifies every curated
original's metadata link across approved catalog, completed download
receipt, source-map and the one canonical FileDB. A deterministic,
source-kind-stratified sample additionally hashes complete original
publisher bytes and complete source-map derivatives. Embedded generated
provenance headers are verified but excluded from the publisher-text
character count. The operator writes only a new private report, cannot
overwrite originals or native database files, and does not create a
second search index. A real snapshot found zero metadata mismatches
across 17,717 source records and zero SHA mismatches among 100 sampled
originals and 98 converted texts. See the separate November 20 project
audit document for reproducibility, citation checks and limitations.

A second independent deterministic replay checked another 100 originals
and 98 derivatives, with ten overlapping originals. Across both audits,
190 unique publisher originals and 188 unique derived texts had direct
byte-level source-revision verification, with no mismatches; the other
library originals remain outside this byte audit and are not declared
verified by sampling.

### Read-only native original retrieval regression guard (November 20)

An original-path-based retrieval audit uses externally predeclared gold
queries, current FileDB and source-map revisions, and the existing native
source-verified semantic search. It reports exact top-one/top-three/
top-five/top-ten original identification and detects when the expected
original is still indexed but has fallen out of the top results. No-gold
historical samples are not assumed unanswerable. The observed old-to-new
recall deterioration on the larger library and rejection of simplistic
root-only ranking are recorded in
`docs/kmd-retrieval-drift-benchmark-2026-11-20.md`. This is a QA tool only;
it neither creates a second index nor modifies production ranking.

### Non-exclusive source gold for retrieval quality, November 20

The canonical read-only native retrieval regression accepts one or many
independently judged relevant original paths per question, measures
first-relevant discovery and fractional known-source recall, and checks
every known relevant source's native original/derivative revision
metadata. Historical baseline comparisons now require unchanged query
and gold identities. A question with one annotated source does NOT
imply all other retrieved sources are irrelevant. A pre-frozen
thirteen-paper comparison demonstrated that the existing KMD hybrid
fusion harmed source retrieval versus the current semantic baseline;
production ranking was deliberately left unchanged. See
docs/kmd-retrieval-drift-benchmark-2026-11-20.md.

### Publisher-attested multi-relevant-original retrieval checks

The read-only original retrieval regression tool optionally accepts
--attest-html-gold --original-root ROOT --derived-root ROOT. It rehashes
the original publisher HTML/text source and its extracted derivative,
checks the current source-map and verifies a frozen supporting phrase
before retrieving. The private gold must identify all supporting sources.
PDF and image evidence requires the separate physical-page verifier.
The new five-question, eleven-source November twenty-first acceptance
is recorded in the dated retrieval benchmark guide. Ordinary search
and scoring remain unchanged.

### Original-source and generated-summary discovery side by side

The optional read-only Raspi dual source discovery operator invokes
scripts/raspi_dual_source_discovery.py with a question and --top limit.
It returns two independent ranked sets from the existing native FileDB:
current source-derived original passage previews and generated
hierarchical summary hints, plus an alphabetic path union. Both exact
publisher original and current derivative file SHA-256 are rechecked,
and raw chunk text SHA is verified again. It does NOT fuse ranks, create
another FileDB or treat an excerpt as a complete, physically validated
PDF source citation. See the November twenty-first evidence and CLI
notes in docs/kmd-retrieval-drift-benchmark-2026-11-20.md.

### Frozen dual-channel candidate recall audit (November 21)

`scripts/raspi_dual_channel_gold_audit.py` independently replays the stored
predeclared original-overview-versus-hierarchical-summary **rank arrays**, not
new embeddings or generated answers. For each fixed source label it measures
whether the source appears within the original-overview top K, summary top K,
either channel's separate top K, or the earlier *experimental* fused top K.
It rejects duplicate frozen questions, repeated expected source paths, malformed
ranks and changed shadow-file checksums. Results retain per-question miss
locations for follow-up publisher-source inspection. This is a candidate
recall diagnostic: no independently verified PDF page citation, exhaustive
relevance set, answer truth, or better default ranking is inferred.

Read-only Raspi reproduction, using the existing private shadow snapshot:

```bash
SHADOW=/data/var/knowmoredirt/knowledge-library/search-audit/native-overview-retrieval-20261121/predeclared_native_overview_shadow_results.json
SHA=$(sha256sum "$SHADOW" | cut -d' ' -f1)
python3 scripts/raspi_dual_channel_gold_audit.py "$SHADOW" --top 10 --expected-sha256 "$SHA"
```

On the frozen 46 questions/52 known source labels, the separate-channel
union found 41 labels, while summary alone found 35, original overviews 31,
and pre-existing experimental fusion 42. The five publisher-attested
multi-source questions recovered all 11 labeled sources through either
separate channel versus nine through fusion. This metric evaluates only
previously frozen candidates and cannot justify replacing the production
ranking. The private replay result and current index-status checkpoint are
under `search-audit/rollover-20261121` on Raspi.

### Optional source-revision-bound physical-PDF preview proof (November 21)

`raspi_dual_source_discovery.py` supports `--verify-pdf-previews N` for one
through five PDF preview snippets, where zero (the default) performs the
previous unmodified dual-channel discovery. With the option, the existing
read-only publisher-page checker independently opens each selected canonical
original PDF without following source symlinks, rehashes its *complete original
bytes* against the completed acquisition receipt, and checks that exactly the
**displayed** extracted-source preview occurs on a specific physical PDF page.
A successful proof reports the original page number, a hash of its extracted
page text, and the basis of the match. Unlocatable excerpts are explicitly
`unverified`; later PDF results above the bounded verification count are
`not_checked_limit`; non-PDF results are `not_pdf`. A changed publisher PDF,
missing source receipt or malformed verifier output fails closed.

For example, on Raspi (one bounded original-PDF proof):

```bash
/data/var/knowmoredirt/semantic-venv/bin/python \
  /data/src/github/KnowMoreDiRT/scripts/raspi_dual_source_discovery.py \
  'How do cited sources support an agent evaluation method?' \
  --top 3 --verify-pdf-previews 1
```

This opt-in check does **not** prove the entire source passage, a figure,
surrounding governing context, an LLM answer, or even that the page explains
the question. It never changes semantic search ranking, generates summaries,
modifies the native FileDB or creates another index. For complete passage
context and image/figure evidence, use the existing original-source evidence
reader and original-PDF page inspection. The private live regression receipt
and tests are in the Raspi `search-audit/continued-20261121-wave44` directory.

### Serialized import lock false-failure correction (November 21)

The two background summary workers already generate their source trees and
embeddings separately and serialize final writes to the **same** canonical
FileDB. A live source on the Coder worker completed its tree but timed out
waiting ninety seconds for the shared `.primary-kmd-content.lock`, while
another legitimate Q4-worker import completed shortly afterward. The source
was retained for automatic retry and the FileDB continued passing its
existing write checks; this is lock contention, not evidence of a broken
publisher PDF or failed model output.

The production import now waits **up to five minutes** for that lock using
the original bounded, cooperative, non-blocking flock polling. A real writer
still has exclusive access; waiting neither bypasses the lock nor alters
existing storage/embedding/page identity and does not remove the subsequent
foreign-key and full SQLite integrity checks. A test simulates another
legitimate writer releasing the lock only after one hundred and twenty-five
seconds, plus a separate never-released-lock timeout. If contention outlasts
the new bound, the already-present retry journal still handles it without
truncating or deleting the source. Do not run duplicate manual imports of a
failed original while either worker is active.

Only the identical worker-source checkout already registered for the two
Raspi systemd services can be safely fast-forwarded from canonical KMD
master; preserve any local changes first, and verify the exact installed
program, ongoing build jobs and live recovery separately. This change does
not claim that a long, full SQLite integrity scan has become inexpensive;
further throughput evaluation should not weaken data-integrity assurance
without independent tests.
