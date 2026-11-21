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

### Conservative original-source spelling equivalence in lexical evidence, November 21

Native `source-evidence` searches a **specified, SHA-verified original source**
with an ephemeral SQLite FTS5 index. A previously observable failure on the
pre-frozen four-domain original-PDF-page holdout was caused by the question
using the single term `nonzero` while the actual independent publisher
passage says `non-zero`. Without spelling equivalence, the supporting original
PDF physical page five was ranked seventh behind unrelated reference,
diagram and ablation-table pages. A query with `non zero` ranked the same
original page first, as independently verified against the publisher PDF.

`file_system_catalog.source_evidence._publisher_spelling_query` now applies a
**narrow, source-conditional spelling-only equivalence** for compounds of
`non` plus an alphabetic suffix of at least three letters. It never edits
publisher text, original source bytes, a derivative, or a native index. It
searches only the actual publisher-source body **after** the first source-unit
marker, not any KMD-generated catalog header. It rewrites a query term such
as `nonzero`, `nonlinear` or `nonparametric` to its separate FTS tokens
`non zero`, `non linear` or `non parametric` ONLY when the joined spelling
is absent from the publisher body and an actual matching hyphenated form
(such as `non-zero`, `non-linear`, `non-parametric`, or a physical typographic
hyphen) is present. If the existing joined spelling is in the source,
if no genuine hyphenated form exists, or for exact underscore identifiers,
there is no change. No arbitrary synonyms, guessed semantic paraphrases,
external model expansions, or replacements of absent publisher originals
are introduced. The original user question stays unchanged in query audit
and source identity, while `publisher_non_compound_spelling_aliases` records
any locally applied lexical form; quoted passages and offsets retain their
**actual unmodified source** and physical PDF-page proof.

The unchanged, independently publisher-page-verified **eleven-case** frozen
pre-change test spans CiteGuard, dense search-agent process supervision,
TexGaussian PBR materials, and *The Little Book of Deep Learning*. Baseline
real source-evidence retrieved nine of eleven expected original physical
pages at rank five, with six at rank one. With this spelling correction it
retrieved **ten of eleven at rank five** and **seven at rank one**. The dense
reward page moved from seventh to first, with a verified original physical
page, and the other ten gold page ranks did not worsen. The separate four-
case 2025 HCI reference control retained four of four at rank five. These
small hand-selected source-specific tests do **not** establish general
semantic retrieval or complete answer accuracy. The still-missed TexGaussian
text-conditioned material generation page remains an open morphological/
semantic ranking challenge. A read-only Porter stemming experiment recovered
that page in one shadow configuration, but reduced exact first-place matches
in other cases; it was **not** deployed. Global semantic embeddings and the
existing two independent corpus search channels remain unchanged.

Private SHA-frozen source/gold, unchanged baseline, patched actual PDF-page
results, HCI control and experimental non-production candidate ranks are
kept under `search-audit/continued-20261121-wave49/`; the programmatic
original-PDF verifier was run for all cited pages before the pre-change
queries and again before both patched live replays. Unit tests verify actual
publisher-only hyphen forms, Unicode hyphens, absent forms, already joined
source spellings, generated metadata exclusions, exact identifier isolation,
original source offsets/SHA and unchanged unverified-answer status.

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

### Publisher-original-grounded **post-hoc** retrieval review, November 21

When an older *single-original* gold misses its specific expected file, do not
silently conclude every competing retrieval result is irrelevant. A returned
source may be useful context, partly support the question, or concern a
completely different problem. The **new diagnostic only**
`scripts/raspi_source_review_audit.py` checks original publisher and extracted
source revisions against the completed acquisition ledger and read-only native
source map for a small manually selected qualitative review set. It refuses
changed originals, changed derivatives, duplicated/retired sources, symlinks,
source-header-only pseudo-evidence, altered frozen ranking identities, and
excerpt SHA/offset mismatches. The quoted evidence must occur AFTER the
actual publisher-source unit boundary in current UTF-8 derived text.
Optional `--verify-physical-pdf-pages` independently rehashes publisher PDF
bytes and uses the existing physical-page verifier for each quoted span.
Neither a digest nor a physical page is a semantic relevance or answer-truth
proof. A failed page match remains explicitly unverified; non-PDF source
excerpts are not misrepresented as physically verified PDF passages.

A review must have a SHA-256-pinned, existing frozen shadow, a separately
SHA-256-pinned private review file, the canonical source map, original root,
extracted source root, completed source acquisition ledger, and a NEW private
output path. Permitted provisional labels are `partial_support`,
`direct_support`, `different_topic_observed`, and `evidence_unclear`.
These are **manually supplied qualitative judgments**, not validated factual
answers, independent held-out labels, or license to tune relevance scoring.
The tool validates only the original identity, actual quoted source text,
original page location if applicable, and membership in an observed top-three
rank list. It never adds to the predeclared frozen gold, changes native
rankings, writes an original, or builds a second index.

Raspi private example (use the recorded SHA sums, not the review output itself
as a source of truth):

```bash
cd /data/src/github/KnowMoreDiRT
AUDIT=/data/var/knowmoredirt/knowledge-library/search-audit
W=$AUDIT/continued-20261121-wave46
F=$AUDIT/native-overview-retrieval-20261121/predeclared_native_overview_shadow_results.json
R=$W/posthoc-indirect-question-original-source-review.json
python3 scripts/raspi_source_review_audit.py \
  --shadow "$F" --shadow-sha256 "$(sha256sum "$F" | cut -c1-64)" \
  --review "$R" --review-sha256 "$(sha256sum "$R" | cut -c1-64)" \
  --original-root '/data/var/Knowledge Library' \
  --derived-root /data/var/knowmoredirt/knowledge-library/semantic/text \
  --completed-ledger '/data/var/Knowledge Library/_management/status.json' \
  --source-map /data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3 \
  --verify-physical-pdf-pages \
  --output "$W/NEW-posthoc-review.json"
```

The first real review deliberately selected three previously **observed**
competing candidates, not three unseen held-out questions. The source/derivative
revision and six excerpt SHA/offset proofs passed. On the Hick's Law decision-
time question, the retrieved Choice Overload original discusses closely related
decision behavior but is not the stronger law-specific source. On Unity texture
import settings, the retrieved Import a Texture original gives the correct
Inspector workflow and platform-specific override context but is not the
complete reference for texture filtering. A leading RGB highlight-removal
paper addressed a different research problem from the requested Blender
normal-map node; two sampled excerpts also matched its original physical PDF
page one. These findings **do not imply** all non-gold documents are wrong,
that the source answers the entire question, or that a particular replacement
ranking has been validated. The original frozen evaluation remains untouched.
Private original/derived SHA receipts, review, validated physical-page results
and regression evidence live under `search-audit/continued-20261121-wave46`.

### Read-only incremental indexing shard-balance planning, November 21

The two native recursive-summary workers retain source-disjoint assignments
using a stable original-relative-path SHA-256 modulo four. The Coder worker
currently owns hash buckets one, two and three; the Qwen Q4 worker owns bucket
zero. This remains reliable and resumable but **is not balanced by pending
source workload** after several thousand successful incremental insertions.

`/data/src/github/KnowMoreDiRT/scripts/raspi_summary_shard_plan.py` is a
read-only capacity diagnostic. It uses the installed canonical **source-map
and one native FileDB** via the existing SHA-respecting worker `scope()` to
count pending originals and complete extracted-text characters per hash bucket.
A separate timestamped, SHA-pinned worker throughput observation supplies a
measured characters-per-hour rate for each model. The pure function
`file_system_catalog.summary_shard_planner.compare_assignments` enumerates
every non-empty, source-disjoint assignment of up to sixteen hash buckets to
the two workers. It compares present ownership against both a minimal-churn
alternative and a purely mathematical best estimated completion duration,
with unique complete source coverage checked by tests. This is an analytical
sensitivity calculation, **not a reliable ETA, guaranteed speedup, active
scheduler or an automatic worker migration**. Relative model speed, source
tokenization, library-size distribution, current model sharing and per-source
quality/call count vary. It does not download models, touch the corpus/SQLite,
change thread counts, edit Infra, or call systemd.

On Raspi, use a new private output file and the recorded real observed-rate
file SHA, for example:

```bash
W=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave46
OBS=$W/native-worker-throughput-normalized.json
python3 scripts/raspi_summary_shard_plan.py \
  --source-map /data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3 \
  --filedb /data/var/knowmoredirt/knowledge-library/filesystem.sqlite3 \
  --observed-rates "$OBS" --rate-report-sha256 "$(sha256sum "$OBS" | cut -c1-64)" \
  --output "$W/NEW-shard-sensitivity.json"
```

The first live read-only comparison saw approximately eight thousand pending
originals on Coder's three shards versus about one thousand on Q4's single
shard. The document lengths differed, so counts are not workload-equivalent:
current pending **text** was about 1.86 billion vs 0.56 billion characters.
During the measured seven-hour historical window Coder completed roughly
0.65 million source characters/hour and Q4 roughly 0.80 million/hour; the
median completed **model generation** stage lasted roughly six minutes on
Coder and eight minutes on Q4, while source embedding/native import was
roughly half a minute. This does not support sacrificing transactional
FileDB integrity checks as the primary throughput solution.

The simplest non-overlapping two-versus-two allocation would retain Coder
hash buckets one and two and give Q4 buckets zero and three. At the captured
snapshot each would then have about 1.2 billion *remaining source characters*;
under the old sampled rates, the proxy largest remaining worker duration
would decrease from roughly 2,860 to 1,880 hours. These numbers **must not**
be presented as a true completion ETA or promised productivity gain; they
ignore future document size/call mixes and ongoing shared-model competition.
An unconstrained mathematical allocation reallocating three hash buckets has
a slightly smaller proxy duration but materially greater operational churn.

**No deployment was authorized or performed by this planner.** If rebalancing
is later approved for production, the active Infrastructure/systemd owner must
stage an atomic source-ownership-safe handoff while preserving in-flight
summary documents, generated trees, the source-map and FileDB. In particular,
never let both models own bucket three simultaneously. A safe ordering is to
release bucket three from the old Coder assignment at a verified document
boundary, then add it to Q4 after the reduced Coder assignment is definitely
active. Current installed worker units use `Restart=always` and a ninety-
second systemd stop timeout; a blind `systemctl restart` can interrupt a
multi-minute generation. Use a separately reviewed quiesce protocol, service
settings and rollback plan instead. The running Qwen/model services are shared
with other projects; do not stop them as part of this operation. Coordinate
with the System Integrity Supervisor and preserve other Infra agents' dirty
working tree. A model-speed change or source growth requires a new measured
rate snapshot and fresh read-only comparison.

Full private journal samples, source-char-normalized rates and live source-map
counterfactual receipts are in `search-audit/continued-20261121-wave46/`.

### Source-identity read lock contention during simultaneous native imports

The November twenty-first live Coder indexing journal recorded a different
transient failure from the earlier serialized *writer* lock timeout. A source
had a complete generated summary tree but `summary_rows()` failed with
`sqlite3.OperationalError: database is locked` during a **read-only native
source-anchor lookup**, before acquiring `.primary-kmd-content.lock`. This can
occur while the other summary worker is committing through SQLite's existing
DELETE journal mode: the standard Python SQLite connection defaults to only
five seconds of busy waiting. A later clean import from the worker does not
imply that this particular failed original was already reindexed; the retry
journal is the authoritative pending record.

`open_native_summary_read_connection()` now opens the **same** original KMD
FileDB in SQLite `mode=ro` with a **bounded two-minute reader timeout**, turns
on `PRAGMA query_only`, and safely closes the handle after the original-linked
summary row preparation. It rejects missing/symlinked native databases and
invalid wait settings. The separately bounded writer lock, normal foreign-key
checks, source revision protection, complete SQLite writer integrity check,
database journal mode, two workers and models remain unchanged. No new
connection permits native writes. The deferred worker automatically retries
its already-generated source after ordinary failure backoff.

The independent `test_native_reader_lock_parallel.py` fixture uses two real
SQLite connections, holds an EXCLUSIVE writer transaction, demonstrates a
blocked source read waits and succeeds when the writer releases, proves
`query_only` prevents inserts, checks the journal remains DELETE, and requires
bounded failure if the exclusive lock never releases. The test never opens
the actual original library or its native FileDB. After publishing, update the
clean active summary-worker source checkout only when no import is currently
running; the workers launch a fresh CLI subprocess per source, so future
imports use the corrected bound without stopping the existing model services.
Verify actual source retry/completion and any future journal errors separately.

### Safe *cooperative* source-boundary reassignment protocol (November 21)

**Important current deployment distinction:** The canonical KMD worker code
has an optional `hierarchical-shard-handoff.json` reader, but the two running
Raspi systemd worker Python processes have not yet restarted into that code.
The private manifest **does not exist**. Their existing source ownership
remains Coder buckets **1, 2, 3** and Qwen3.8 Q4 bucket **0**. No systemd unit,
shared resident model, source original, native FileDB or service setting was
changed by the implementation. The operator's real dry-run explicitly REFUSES
to proceed until **both** active processes demonstrably run the new code.
The prior read-only 2:2 plan is still a proposal, not a deployed speedup.

In a future coordinated maintenance window, the System Integrity Supervisor
must first establish a safe **document-boundary restart** (or let each worker
naturally restart), preserving the active model endpoints and ensuring no
current KMD import is interrupted. The systemd `KillMode=control-group` and
`TimeoutStopSec=90s` make a blind `systemctl restart` unsafe for the existing
multi-minute generation subprocess; it may discard ongoing model work. No
component in this source-shard protocol signals, kills, starts or restarts a
worker, modifies those service units, evicts a model or bypasses that guard.
Do not use the tool merely because sudo is available. The active project
exchange and resource broker must be checked before any maintenance.

Once both running workers have loaded the new source, their private status
JSON must report `allocation_protocol=kmd.two_worker_shard_handoff.v1`, a
`allocation_runtime_started_utc` exactly equal to the real current worker
`started_utc`, current installed main PIDs, generation zero and the original
command-line assignments. Verify this independently, not from Git HEAD alone.
The optional manifest is accepted only if a strict, regular, owner-only,
small JSON file with an allowed generation, exact two-worker bucket ownership,
no unexpected keys or symlinks. No file means original command-line behavior.
A manifest once active cannot be silently deleted, downgraded or overwritten
under the same generation. Illegal state makes a worker fail closed and retry
its non-destructive outer loop rather than silently claiming a released shard.

Use the owner-only, tested, atomic operator, which checks **both current live
systemd main PIDs**, their actual Python process command-lines, protocol
start markers, status freshness, current in-progress source shard, exact
manifest SHA and acknowledgement generation before each possible write:

```bash
cd /data/src/github/KnowMoreDiRT
python3 scripts/raspi_source_shard_handoff.py --target balanced
# READ-ONLY BY DEFAULT. Currently expected to refuse: workers too old.
# After both workers naturally run the new source and all checks pass:
python3 scripts/raspi_source_shard_handoff.py --target balanced --apply
# Wait until BOTH workers acknowledge source-boundary generation 1.
python3 scripts/raspi_source_shard_handoff.py --target balanced --apply
# Wait for Coder to COMPLETE its last bucket-3 source and for BOTH workers
# to acknowledge generation 2: Coder [1,2], Q4 [0].
python3 scripts/raspi_source_shard_handoff.py --target balanced --apply
# After Q4 safely acquires shard 3 at a fresh source boundary,
# both worker statuses eventually acknowledge generation 3, [1,2] / [0,3].
```

A second call before the prior generation's acknowledgements returns a
nonzero error **without** writing or starting any model. The private
`hierarchical-shard-handoff-operator.jsonl` journal records each actual
manifest stage change; the manifest is written atomically and fsynced.
The worker itself also verifies the releasing Coder's live protocol/PID and
source-boundary acknowledgement before Q4 can acquire shard three. The
rollback check is symmetric: if later restoring the old 3:1 split,
**Coder must check Q4's release** before it can reacquire shard three. This
protects against a manually forged full-assignment manifest bypassing the
operator. The worker rechecks the owner-only source manifest between source
documents, never during the model generation or native import of an active
source; on a change, it acknowledges and rebuilds the backlog immediately.
The SHA-based original-to-bucket mapping itself is unchanged, and no source
is duplicated between the two valid owners.

Rollback uses the same operator with `--target baseline`, also in two
acknowledged stages: Q4 releases shard three, then Coder reacquires it.
Never remove the active manifest as a rollback, alter the status JSON by hand,
or directly write `balanced` without the two-phase acknowledgement. A worker
whose saved generation refers to a missing manifest refuses to resume old
ownership automatically. On fatal or unavailable worker status, do not
apply more stages: preserve the current safe manifest and escalate the exact
source boundary to the Integrity Supervisor.

The isolated `test_source_shard_runtime.py`,
`test_source_shard_worker_boundary.py`, and
`test_source_shard_handoff_operator.py` verify strict manifest identity,
rollback, per-original boundary ordering, normal legacy behavior, stale PIDs,
missing acknowledgements, symmetrical acquisition proofs, atomic private
file writes and the append-only journal. Those tests use only disposable
fixtures, never the live process managers. The private wave47 receipt and the
real *refusing* dry-run log are retained under
`/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave47`.

### Explicit-source binding across both independently ranked native channels, November 21

The optional combined original/summary search must not confuse two different
source contracts. `raspi_dual_source_discovery.py` used the native
summary channel's exact publisher-name resolver while still issuing an
**unrestricted** original-passage search, so the explicit request
`According to CiteGuard, ...` could return the named source **and** other
semantically similar, unrelated publications. The source-linked `source-evidence`
operator correctly handled a named original; the combined two-channel
operator did not. The fix enforces the native summary search's existing
publisher-identity decision for both independent channels, not a second
parser or new relevance model.

- A unique, explicitly cited publisher original is the only permitted source
  in either channel. If its summary is not indexed yet, its independently
  available original passage may still be read from the existing source map,
  with the *exact* `--original-path` applied to the original-passage program.
- If an explicit title/identifier is absent or matches multiple original
  publications, the combined operator returns **zero** unrelated documents
  and describes the ambiguity/absence. Up to eight actual candidate paths
  can be displayed for a larger ambiguous source family, preserving the
  distinct total count. Users must choose a full title/path; vector ranking
  is not evidence that one edition was intended.
- Ordinary unqualified topic questions still run both global independent
  searches. There is no rank fusion, confidence score, source substitution
  or model retraining.

The publisher-source excerpt path also has two separate, explicit safeguards:

1. A native source-passage chunk can have a valid native SHA while existing
   **only** in the automatically generated extraction/catalog preamble rather
   than the current publisher's actual source unit. The combined CLI now
   explicitly *abstains* from presenting that candidate when the full
   original and derivative hashes match but the exact quoted passage does
   **not** occur after `[[SOURCE UNIT:...]]`. The candidate's original path,
   native rank and reason are recorded in the result. **Any changed source
   SHA, missing original, invalid chunk SHA or other integrity error still
   causes the entire query to fail closed**. The reusable pure combiner
   retains strict all-candidate rejection by default; only the installed
   combined CLI opts into the visible metadata-only candidate exclusion.
2. A derived-PDF search passage can include extractor-only page delimiters
   or span the boundary between two physical pages. These markers must not
   be displayed as literal publisher quotations. The preview reader selects
   only exact underlying text from one segment and tracks the source page
   label independently of the original chunk label. A SHA-256 of the
   **displayed preview** is retained alongside the original full-passage
   SHA, and physical-page evidence is still checked separately against the
   original source PDF bytes. Physical-page verification does not establish
   complete answer correctness or context outside the displayed excerpt.

The real Raspi same-query before/after outputs are kept privately under
`/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave47`.
An explicitly named `CiteGuard` question went from the original and two
unrelated originals to **only CiteGuard**, and the new publisher-text-only
preview was independently verified on original physical page **one**.
An ambiguous `ToolScope` name returned the two possible original editions
without substituting a source. An ordinary citation-auditing question kept
both search channels, returned three independently rehashed original-source
passage candidates, omitted one metadata-only candidate **with a visible
provenance rejection**, and verified two returned PDF excerpt previews on
original pages **ten** and **one**. The third retrieved item was a non-PDF
source and correctly carried no physical-PDF proof. No native model, source
original, summary, FileDB or corpus configuration changed.

This is a **source-identity/citation-integrity** correction, not a measured
improvement in open-domain semantic recall. The separate frozen experiment
`lexical-metadata-frozen-first-pass.json` also tested an in-memory title/topic
BM25 channel and recovered **zero additional known source labels** beyond the
incumbent two-channel union at top ten, across the frozen historical,
independent-paper and multi-source families. That untuned, metadata-only
channel was not added to production. For general relevance improvements,
new independently judged multi-answer and adversarial original-text labels
are needed, preserving the frozen pre-change evidence.

A fresh, publisher-source-byte-attested *live* comparison of all five
existing fixed multi-source questions on November 21 confirmed that the
summary channel recovered **10 of 11** predeclared known relevant original
identities at rank ten, the raw original-overview channel recovered **9 of 11**,
and the separate source-identity **union recovered all 11 of 11**, across all
five questions. The PBR microfacet publisher original missing from the current
summary top-ten list appeared in original-overview rank ten; in two Blender
normal-map questions the generated summaries recovered relevant publisher
references absent from the raw top ten. These are exact known source-label
hits under today's moving index, not exhaustive corpus precision, semantic
answer accuracy, or a promise that the same metrics persist as indexing grows.
The report, checked publisher and derivative SHA evidence, and per-source
rank details remain private under wave47. Continue to measure both independent
channels rather than replacing them with the weaker metadata-only option.

### Current original-byte-attested two-channel recall monitor (November 21)

`scripts/raspi_dual_channel_recall.py` measures *separate* top-K summary
and native original-source passage retrieval over the existing, SHA-pinned
predeclared five-query multi-source original set. This directly monitors a
known problem: as the corpus grows, first-stage source recall and rank may
change, and a relevant document can occur in one independent channel but not
in the other. The metric never turns a similar passage into an asserted answer.

Before any model search, the monitor validates the frozen original gold SHA
and checks every known HTML publisher original against the native source-map
original and derivative SHA, plus the original independently predeclared
publisher-text anchor. Every newly returned KMD candidate must be publisher
revision verified and each displayed source excerpt must have its own
verifiable SHA. Top-K ordering must be sequential and source paths unique.
It reports per-question exact original-source ranks and count of known
relevant documents found separately in the original-passage channel, the
hierarchical-summary channel, and their simple **set union**, never score
fusion or model confidence. Missing known-source labels are *candidate
retrieval misses*, not proof that an answer is impossible. The predeclared
originals were established before comparing new rankings, but this five-query
set is not exhaustive, and source SHA equality alone cannot prove that every
candidate contains a complete, contextually valid answer.

Use the existing private fixed gold and a NEW output file, without changing
source paths or downloading anything:

```bash
cd /data/src/github/KnowMoreDiRT
G=/data/var/knowmoredirt/knowledge-library/search-audit/multirelevant-originals-20261121/original_multirelevant_gold_frozen.json
OUT=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave47
/data/var/knowmoredirt/semantic-venv/bin/python scripts/raspi_dual_channel_recall.py \
  --gold "$G" --gold-sha256 "$(sha256sum "$G" | cut -c1-64)" \
  --dual-cli /data/src/github/KnowMoreDiRT/scripts/raspi_dual_source_discovery.py \
  --source-map /data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3 \
  --original-root '/data/var/Knowledge Library' \
  --derived-root /data/var/knowmoredirt/knowledge-library/semantic/text \
  --completed-ledger '/data/var/Knowledge Library/_management/status.json' \
  --output "$OUT/NEW-native-dual-source-holdout.json" --top 10
```

The original-SHA attestor handles the *publisher HTML/text* gold of this
fixed set; it explicitly refuses to pretend a PDF derivative is an original
physical-PDF byte quote. Physical PDF evidence still uses the existing
independent page verifier. Individual subsets may be run via `--case-id`,
but a subset must not be reported as the complete frozen population. All
program effects are read-only apart from the newly generated private test
report; it does not re-embed, download, modify a source or train ranking.
The live source-map and indexing workers change over time, so record the
fresh per-run timestamp and do not claim results reproduce byte-for-byte
across different library index states. A publisher/source identity result is
not a guarantee of truthful or complete answers.

### Independent newest-original SHA and complete hierarchy-integrity sweep — November 21

The live recursive-summary worker's normal `current_completed_originals()`
checks each current publisher SHA, derivative SHA, and whole-source root,
but a matching **root alone** is not proof that all recursively generated
leaves, parent-child source spans and current vector bytes remain intact.
The optional read-only source-integrity tool
`scripts/raspi_recent_summary_integrity.py` independently rechecks a bounded
sample of original documents from the actual two installed worker
`KMD_HIERARCHICAL_DOCUMENT_COMPLETE` journals. This is a verification
operation, **not** an index, new embedding corpus, or workload scheduler.

For each recent completed original, it requires the same completed acquisition
ledger receipt, exactly one current source-map identity, a full-byte matching
publisher original SHA-256, complete derived-text SHA, exactly one canonical
native FileDB original path, and current revision-linked summary rows. Every
current summary node must match its native source start/end character offsets,
actual derived-source span digest, summary-text digest, generated-source SHA
metadata, embedding-byte digest, declared embedding dimension and dtype, and
finite embedding norm. The verifier requires exactly one full-source root,
correct child order and nested summary levels, an unambiguous acyclic
parent-child graph, no orphaned nodes, and **contiguous complete source-text
coverage** across all leaves. Stale historical-revision nodes are counted but
not mistaken for current original representations. Checks do not require a
trusted language model or any remote service.

Invoke on Raspi with an exact local journal time and a **new private** output
filename:

```bash
cd /data/src/github/KnowMoreDiRT
W=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave48
/data/var/knowmoredirt/semantic-venv/bin/python \
  scripts/raspi_recent_summary_integrity.py \
  --source-map /data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3 \
  --native-db /data/var/knowmoredirt/knowledge-library/filesystem.sqlite3 \
  --original-root '/data/var/Knowledge Library' \
  --derived-root /data/var/knowmoredirt/knowledge-library/semantic/text \
  --completed-ledger '/data/var/Knowledge Library/_management/status.json' \
  --since '2026-11-21 14:30:00' --last 15 \
  --output "$W/NEW-recent-source-integrity.json"
```

The same command also accepts `--selection largest --last 12` to test
large books and multi-section sources instead of preferentially sampling
recent short worker completions. This mode queries only the **existing**
canonical FileDB's current source-spanning root candidates, selects the
twelve distinct originals with the longest extracted-text spans, and applies
all of the same independent original/source-map/root/leaf/vector SHA checks.
It never uses a different database, source downloader, model, or generated
reference document. The source root's presence is a *candidate* for audit,
not proof of a valid full hierarchy. Test fixture coverage confirms that
stale duplicate roots do not inflate the sample, invalid sizes fail, and
an entirely empty native index cannot pass as a successful large-book audit.
In a real November 21 read-only sample, twelve of the largest completed
originals passed, including a roughly 2.75-million-character HCI reference
with forty-two linked native summary nodes. The twelve large-original tests
covered 105 nodes and independently rehashed their original and derivative
full-file bytes and source-spanning tree/embedding structures. No source
content or native index was modified.

Only one to twenty-five distinct recent real completion events can be
checked in a call; the tool never treats arbitrary stdout or a title in a
research abstract as a completion event. Any original or derivative above
64 MiB is marked **unverified**, not silently skipped as verified. Other
source SHA, missing child, disconnected hierarchy, incorrect offsets,
modified summary text and incompatible embedding errors are clearly reported
as **failed provenance**, with a nonzero exit status. Existing output paths
and symlinked source paths are refused. All original, source-map and FileDB
paths are opened for read-only verification; only a new mode-0600 private
JSON report is written. The worker and shared model services are untouched.

The November 21 first full sample rehashed fifteen approved publisher
originals and their current extracted text. It verified forty-nine native
recursive summary nodes, including the complete leaf-to-root source coverage
and embedded vector fingerprints, with zero detected integrity failures.
That is a bounded recent sample, not a guarantee for all roughly half-indexed
originals or for the factual quality of AI-generated summaries. The source
identity and tree audit are complementary to independent PDF-page physical
citation validation, and neither replaces original documents or ordinary
filesystem/grep search. This audit is intentionally NOT executed on every
interactive search, to avoid the large original-byte scanning cost.

The private reproducible receipt is at
`/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave48/approved-recent-full-native-graph-check.json`.
The associated disposable SQLite regressions under
`tests/filesystem_database/test_recent_source_tree_audit.py` and
`test_recent_summary_integrity_cli.py` inject changed publisher bytes,
modified derivatives, incomplete roots, missing or reordered source children,
invalid source spans, digest mismatches, manipulated embedding shape and
missing approvals, and check that each fails rather than being promoted to
complete. Retain historical source-map error classifications for the two
existing image-only originals until independently verified OCR is available.

### Independently pinned physical-PDF-page recall instead of merely page-location integrity

A source-linked retrieved passage is not sufficient proof that the search
selected the **correct supporting physical PDF page**. The publisher source
may have been selected successfully while the passage selector returns an
unrelated page from the same book. Source page recall, source identity, page
location validity, complete answer support and summary truth are **separate
acceptance questions**. This distinction is emphasized in the October 2026
GroundLM papers *PageRecall* and *Does the Selected Object Reach the Reader?*;
external paper benchmark percentages are not estimates for this Knowledge
Library. Do not infer correctness from a generated summary or a valid SHA.

`src/file_system_catalog/original_page_recall.py` and
`scripts/raspi_original_page_recall.py` add a **read-only, original-publisher
PDF page recall** benchmark against a small, previously frozen gold file.
Each gold case specifies an exact approved original-relative PDF path,
full original and current derived source SHA-256, the question, an expected
*physical* PDF page, a publisher excerpt from that page and its digest, and
an independently checked original physical-page text SHA. Before any test
query the audit verifies the completed acquisition ledger, full publisher
original and derived bytes, source-unit page boundary and **again** validates
the frozen excerpt on the same physical page of the original PDF. Any stale
source, missing receipt, changed excerpt, wrong page, invalid Unicode/quoted
SHA, wrong rank/identity/engine or source mutation fails closed. The separate
native `source-evidence` results must identify the same original and show
verified PDF-page proof; unverified pages never count as successful hits.
The metric is **known gold-page recall at K**, not exhaustive relevance,
complete factual answer correctness, global source recall, or confidence.

The verified small four-question test was frozen from the approved **2025
Introduction to Human-Computer Interaction** reference *before* any of its
four page-lookup queries were replayed. The independently original-PDF-
verified supporting pages were physical 112 (GOMS), 339 (System Usability
Scale), 53 (signal detection theory), and 367 (information scent). Their
four exact source excerpts, SHA-256s and original page-text SHA evidence are
in the private `frozen-physical-hci-page-gold.json`. A first genuine source-
restricted FTS5 retrieval returned the original gold pages at respective
ranks **2, 1, 2, 1**. That is **4/4 at rank five** and **2/4 at rank one**
for this deliberately small reference sample, not a broad quality claim.
The independent original-PDF page verifier was invoked again before the
standalone benchmark's second replay, reproducing those numbers. Neither
run modifies source documents, embeddings, native FileDB, source-map, models
or original source approvals. KMD may append ordinary private query audit
records as its existing read-only lookup normally does.

Run on the existing Raspi canonical library with an entirely **new** private
output path, keeping the pre-frozen SHA independent of current search results:

```bash
cd /data/src/github/KnowMoreDiRT
W=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave48
G="$W/frozen-physical-hci-page-gold.json"
/data/var/knowmoredirt/semantic-venv/bin/python \
 scripts/raspi_original_page_recall.py \
 --gold "$G" --gold-sha256 "$(sha256sum "$G" | cut -c1-64)" \
 --source-map /data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3 \
 --original-root '/data/var/Knowledge Library' \
 --derived-root /data/var/knowmoredirt/knowledge-library/semantic/text \
 --completed-ledger '/data/var/Knowledge Library/_management/status.json' \
 --source-cli /data/src/github/KnowMoreDiRT/scripts/raspi_hierarchical_library.py \
 --top 5 --output "$W/NEW-page-recall.json"
```

The script refuses an existing output and duplicate or altered gold cases,
accepts no more than twenty source cases or ten results per case, and writes
the result under mode-0600 with explicit per-case page identities and hit
ranks. An optional `--case-id` subset is labeled **not full population**.
The original PDF excerpt/page gold has independent source bytes and a
physical-page SHA; no retrieval output can silently replace a label. This
metric is particularly useful for lengthy reference books where known
answer-bearing pages can be far apart. Broader unbiased answer-level
benchmarks still require independent human evaluation and more varied source
families before deciding to modify the native ranking algorithm.

Private result and per-original page SHA receipts are under
`search-audit/continued-20261121-wave48/official-attested-original-page-recall.json`.
The disposable SQLite and source-verifier test fixtures are in
`tests/filesystem_database/test_original_page_recall.py`. No synthesized test
PDF is ever claimed to prove production physical-page extraction.

### Optional RAM-only first-publisher-unit discovery for incompletely summarized originals

The current native summary and original-overview corpus-wide vector search is
source-revision checked and remains the **default**. Its first-stage source
selection can nevertheless miss the requested approved original when the
relevant first page uses terms absent from its overview embedding or when a
new document has not yet received a hierarchical summary. This is independent
of the existing `source-evidence` operation, which can locate and physically
verify relevant pages once an exact original path is supplied. In a separately
frozen **eleven-query, four-domain** test, only **two of eleven** known exact
approved originals appeared in the top five of either existing global vector
channel. That small set does not prove all other candidates irrelevant.

A separate **optional read-only** source discovery is provided by
`src/file_system_catalog/publisher_firstunit_discovery.py` and its command:

```bash
/data/var/knowmoredirt/semantic-venv/bin/python \
  /data/src/github/KnowMoreDiRT/scripts/raspi_publisher_firstunit_discovery.py \
  --query 'Which paper uses conformal false-discovery control in RAG?' \
  --top 5 \
  --output /data/var/knowmoredirt/knowledge-library/search-audit/NEW-firstunit.json
```

For several related questions, use one restricted private JSON file containing
`{"queries":[{"id":"question_one","query":"..."}, ...]}` and replace
`--query ...` with `--queries-file /absolute/private/questions.json`. One to
**thirty-two** questions can share the single temporary first-page index.
The operator writes only to a **new** mode-0600 JSON file, or stdout if no
`--output` is specified; it refuses an existing output, unsafe query paths,
invalid query fields, or unsupported query-size limits. It uses one in-memory
SQLite FTS5 index with two independent rank channels: literal Unicode61 and
Porter Unicode61. `bm25(title=3, source_topic=0.4,
first_source_unit_body=1)` provides *lexical candidate order only*, not
confidence or factual accuracy. The two lists are NOT fused or substituted for
native semantic ranking; other projects may still use ordinary folder and
grep searches directly on the preserved original files.

The operator normally reads **only sixteen kibibytes** from each completed
source-map derivative and selects the **earliest usable original publisher
source unit**. PDF title, copyright and cover pages may be empty, and some
pages have NUL-interleaved font codes; these are skipped instead of rejecting
the entire approved reference. If **and only if** no readable source unit
exists in this standard prefix, the operator reads at most **sixty-four
kibibytes** from that same file to seek a later readable section. It inspects
at most sixty-four genuine publisher source-unit markers in whichever
bounded prefix applies, without changing original text, stitching separate
pages into a fabricated quote or treating the preceding KMD-generated
catalog header as publisher evidence. The selected unit may be physical PDF
page two, three or later. Its actual derived label and original character
offsets are retained as derivative metadata, not independently proven
physical-page identity.
At most 6,500 characters from that **single** source unit are added to
temporary FTS per original. The corpus is not recopied, no independent
persistent DB is created, and the two existing summary workers and models
are not stopped, modified or given new work. An original without a readable
source unit within this bounded preview remains visibly **unindexed by this
optional search**, not missing from the library. A fact on a distant page
may still be unavailable even when the original is approved and catalogued.

**A first-stage lexical hit is not a verified original.** Before emitting
*each displayed result*, the CLI verifies its current source-map status,
source original SHA and derivative SHA/path/title against the source-map row
*again*, checks exactly one current completed acquisition receipt and a
single original `filesystem_entries` identity in the existing native FileDB,
rehashes **all bytes** of the original publisher file and its full extracted
text, validates exact unmodified source character intervals and excerpt SHA,
and excludes any stale, unreadable, unapproved, nonmatching or oversized
candidate with a recorded reason. Raw textual source previews remain
untrusted original content and never executable instructions. Their
`factual_answer_verified` and `quoted_text_is_original_publisher_bytes`
flags remain false. `--verify-pdf-previews 1` (maximum three per channel and
question) additionally checks a returned excerpt on the *original physical
PDF page*, with its own page-text SHA. A derived page label alone is not a
verified publisher page. The optional physical-page check never invents a
citation or repairs an unreadable PDF. Native original/derived/index writes,
external downloads, model calls and publisher-origin replacements are absent.

**Measured private results, November 21:** On the previous opening-page
baseline, the live Raspi scan inspected 17,715 completed source-map records,
indexed **17,677** first-unit previews from approximately 261 MiB of sampled
existing source bytes, and excluded **thirty-eight** approved originals:
fourteen had mostly unreadable NUL-interleaved font codes on the opening
page, and twenty-four had very short opening pages. The strictly bounded
**earliest-usable-source-unit** recovery found a later readable publisher
page for **thirty-seven** of these thirty-eight originals. The later real
published-source code replay scanned the same 17,715 completed originals,
indexed **17,714** early source-unit previews and left one difficult PDF
outside the standard sixteen-kibibyte preview. That last original retained
usable original text on its *derived* PDF page twelve, near sixty kibibytes
into the extracted file. The bounded exceptional fallback subsequently
indexed it too, yielding **17,715 out of 17,715 eligible completed original
source previews** with **zero** skipped in a fresh full batch replay. Only
one source used the extended read; the frozen original-source rankings
remained six literal, seven Porter, eight in the separate-channel union.
This counts *early-page candidate eligibility*, not full-document indexing
or physical-page truth. Its derived page-twelve label has **not** passed
independent verification against the original physical PDF page. All
original files remain present and unchanged. The eleven pre-frozen questions retained
**six of eleven** first-stage exact known-source hits in the literal channel,
**seven of eleven** in Porter, and **eight of eleven** in their candidate-set
union; no source ranking was silently replaced. On those same questions,
the pre-existing two global native channels jointly found only two selected
originals in their top five. These are **document discovery** measurements,
not complete answer truth or broad relevance estimates. A real test across
three previously omitted approved PDFs additionally rehashed the full
publisher original and full derived source for each and independently
verified original physical PDF page location on **two of three**. For the
third, the physical-page verification failed and no publisher-page citation
was invented. The optional tool still reports exact omitted original-relative
paths and bounded reasons, not source prose or private catalog metadata; it
limits printed omission identities to 150 with an explicit truncation flag,
without changing the full omission count. The separate real CiteGuard paper
test independently verified publisher physical PDF page one. Evidence and
pre-frozen source labels remain private under the wave-forty-nine and
wave-fifty search-audit folders; no copyrighted publisher text was copied
to Git.

An additional **non-production** original-wide stratified source FTS shadow
sampled six deterministic positions within the 17,715 existing derivatives,
representing about 421 million indexed source characters in temporary RAM.
On the unchanged frozen eleven questions its two channels alone retrieved
**five literal** and **seven Porter** known originals at top five. Adding
its separately observed candidates to those from the unchanged native and
earlier first-unit channels would have increased known selected-original
coverage from eight to **nine of eleven**. But the temporary index took
nearly two minutes to build, did not solve the remaining two misses, and its
candidate sources were not fully rehashed for publisher-quality citation.
It was **not** deployed as a new ranker, default retrieval system or persistent
cache. A separate read-only PDF text-layer audit of the fourteen NUL-font
cases compared `pdftotext` layout, raw and default extraction over the first
two physical pages. All fourteen retained NUL glyph failures; simply
switching extraction mode did **not** repair their first pages. Source data
and indexes were not mutated, and no OCR or new acquisition was performed.

**Known limitations and next evidence gate:** The query set is small,
source-selected from four original documents and hand-authored before either
search, not a representative sample of all possible questions. The extra
first-page lexical candidate set still misses questions whose answer-bearing
content is deeper in the document, and a keyword match need not be the
correct answer. Adding or tuning a default global rank fusion is unjustified
without independent hard-negative relevance labels, larger representative
question sets, full-source first-stage recall and answer support quality
checks. Candidate results are revision-checked only at the time of each
lookup; if an original or derivative changes later, it must be reverified
before being cited. The corpus remains automatically incremented by the
existing KnowMoreDiRT workers, not by this optional read-only search.

The disposable regression tests are
`tests/filesystem_database/test_publisher_firstunit_discovery.py` and
`test_publisher_firstunit_discovery_cli.py`. They test generated-metadata
exclusion, first-source-unit-only parsing, bounded cardinality and queries,
noncommercial local corpus previews, exact source offsets and hashes, invalid
source paths, malformed or duplicate approvals, current source-map revision
changes after an ephemeral index was built, unique native FileDB identity,
stale original/derivative bytes, and independently verified versus unverified
physical-page citations. Successful tests verify the specified safeguards;
they are not evidence that the corpus-wide lexical ranking is universally
correct or faster than native vector retrieval on all hardware.
