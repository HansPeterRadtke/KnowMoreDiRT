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

### Original PDF physical-page bounds: legacy extraction provenance faults, November 21

The optional source lookup now performs an **independent original-PDF
physical-page-count negative check** before displaying any selected PDF
preview. It first rehashes the entire source against the completed approved
acquisition receipt, the complete current extracted derivative against
`source-map.sqlite3`, and the unique original identity in native FileDB. It
then reads the *current original PDF* page count with a bounded, read-only
`pdfinfo` subprocess and compares it to the maximum **anchored** `PDF page N`
source-unit label in the **complete extracted derivative**. If any cached
page marker is above the original PDF's physical page count, the new
`source_pdf_page_map_integrity` object reports
`derived_page_labels_exceed_original_physical_pages`, the physical page count,
the current maximum derived label and a warning. Merely consistent page
bounds return `page_label_bounds_consistent_only_not_verification`, **not** a
verified citation; the selected source label is explicitly marked
`derived_source_text_only_not_physical_page_proof`. Failure or timeout to read
`pdfinfo` is reported as `physical_page_count_unavailable`; the tool never
claims it checked the original page when it did not. Replacement of the
source file during the check fails closed. **Only an independently confirmed
excerpt on the correct physical publisher PDF page** can receive a positive
`physical_pdf_page.physical_pdf_page_verified` field. The corpus-wide page
count cannot establish literal excerpt match, factual correctness or citation
accuracy for an in-range page. It does not change semantic scoring or alter
source bytes, derivatives, summary/vectors, or the native database.

This is not speculative. A private Raspi **full existing-corpus read-only
page-label audit** compared all **14,474** completed source-map PDF derivatives
against actual original physical-page counts already represented in native
KnowMoreDiRT FileDB PDF metadata. It found **327** candidates with out-of-range
cached `PDF page` labels. For **every one** of those 327, the audit then
independently rehashed the entire original publisher PDF and derivative
against their approved current source-map and acquisition receipt SHA-256 and
queried actual `/usr/bin/pdfinfo`; **all 327 mismatches were confirmed**,
with no failed attestation. The old November 2 nine-source warning set is
included in these 327; **318 confirmed currently affected sources were not
in that previous nine-source set**. All 327 have original-content vectors in
native FileDB; **119 already have summary representations**. The problem is
material to existing source-linked retrieval, even though an individual
verified passage may still be correct. Another 168 completed source-map PDFs
had no anchored `PDF page` label and were not classified as correct by this
overcount check; for the 13,979 with in-bounds labels, individual page
position remains unverified absent per-excerpt original PDF proof. The 327
represent confirmed lower-bound page overcounts, **not** measured overall
citation error frequency or total prevalence of all false in-bounds pages.

**Native-library citation guard activated without modifying the shared
Infra checkout.** The existing Raspi native original-vector search already
consults a private, exact original-and-derived-revision-gated historical
PDF warning registry. After independently proving all 327 current defects,
the Knowledge Library project extended that **same existing private registry**
with **318 new attested records**, preserving the prior nine records and
historical November 2 evidence, adding a dated extension receipt and keeping
a private exact pre-extension backup. An atomic publication and existing
source/derived SHA validation prevented publishing stale or ambiguous source
identities. The already-installed `knowledge_library_semantic_query`
`_page_scope_warning` was called against **all 327** confirmed exact current
original revisions and returned the expected original physical page count and
cached derived-label maximum for every one, while a deliberately wrong source
SHA returned no warning. A real `knowledge-library-semantic-search` query
restricted to the affected AgentSteerTTS publisher original returned a
native SHA-verified passage with `source_page_citation_warning`, and
`--pdf-page-context` correctly withheld the untrustworthy cached page context
as `historical_page_citation_warning`. **No source content, native vectors,
summary tree, model, resident worker, or shared dirty Infra code was changed.**
The private historical warning is tied to the **exact** original plus current
derivative SHA and will cease matching if a future correctly re-extracted
source revision is committed and propagated through the native index. It is a
risk warning, not an automatic repair or independent positive physical-page
citation proof.

**Reconstruction feasibility is independently tested, but live migration is
not yet safe.** The current Infra `_large_pdf_pages` extractor, invoked only
in private temporary scratch, produced the correct **nineteen** original
physical-page units from the previously 245-unit AgentSteerTTS derivative;
seven selected excerpts were independently verified on the original physical
PDF pages one, three, six, ten, twelve, fifteen and nineteen. Three further
originals from graphics, infrastructure and financial research were privately
re-extracted with the current physical-page-count fallback, producing their
exact 12, 23 and 35 physical-page units rather than 30, 87 and 56 old labels,
and fifteen independently source-PDF verified excerpts. Original and existing
derivative full SHA-256 remained unchanged. A private deterministic
**327-source repair manifest** records every original/old derivative SHA,
native FileDB original ID, physical page count and next safe migration path.
**323** affected PDFs fit the existing bounded pagewise extractor; **four**
with more than sixty physical pages require a separate tested recovery path.
### Private, non-destructive original PDF page-cache repair staging

The source-linked PDF page-overcount repair cannot be applied to the live
source-map text/semantic vectors/summaries while the two resident source
summarization workers and other native retrieval readers depend on those
identities. Use only the optional, **staging-only** script
`scripts/raspi_stage_original_pdf_page_repairs.py` to generate a new proposed
replacement text outside the live source tree. It reads the private
revision-pinned 327-original remediation plan, the authorized complete
acquisition ledger, the current curated source manifest, the current
read-only source-map and original FileDB. It rehashes the **entire existing
original and old extracted derivative** (not only prefixes or excerpts),
checks unique original FileDB IDs and requires a physically impossible
historic page-label overcount before regenerating anything. The original
PDF is processed through the currently installed Infra
`knowledge_library_semantic._large_pdf_pages` original physical-page
extractor. Only one-to-thirty-two existing safe-bound original PDFs are
processed per private batch; no non-PDF, unapproved, stale, too-large or
unknown source is silently admitted. All output files and receipts are
new/exclusive and private, including the full regenerated copyright-bearing
text; such text must never be committed or distributed.

Example on **Raspi** from an existing privately attested repair plan (use
fresh stage and output names; never overwrite a prior stage):

```bash
cd /data/src/github/KnowMoreDiRT
D=/data/var/knowmoredirt/knowledge-library/search-audit/NEW-STAGE-DIRECTORY
mkdir -m 700 -p "$D"
PYTHONDONTWRITEBYTECODE=1 /data/var/knowmoredirt/semantic-venv/bin/python   scripts/raspi_stage_original_pdf_page_repairs.py   --repair-plan /data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave50/read-only-pdf-page-attribution-repair-plan-327-originals.json   --offset 0 --limit 12   --stage-root "$D/first-twelve-original-page-caches"   --output "$D/first-twelve-original-page-caches-receipt.json"
```

**Exceptional four-publisher-original extension, still private staging only:**
The prior full-original, physically indexed wave-fifty shadow independently
proved exact pages for four approved PDFs whose **64, 76, 70 and 418** physical
pages exceed the current Infra automatic sixty-page fallback bound. For
**only those exact four source and derivative SHA revisions**, the staging
CLI accepts an explicit `--extended-shadow-proof` with the private,
independently attested `all-four-larger-PDF-physical-pagewise-shadow-verification.json`.
It accepts **one original per call**, requires matching approved original,
old cached derivative and prior physical-page samples, and rehashes the
entire current original/old derived text and rechecks the real PDF page count.
It then streams only individually requested original physical pages using
`pdftotext -f N -l N` inside the private stage. Each Poppler child is bounded
by a **512 MiB virtual-address-space limit**, **8 MiB output file limit**
and **eight-second page timeout**, and the selected original is capped at
450 pages / fifty MiB. PDF form-feed codes *within a physical page* become
newlines, **never additional PDF page numbers**. The source-map, source
original, cached derivative, native vectors and summaries remain untouched.
This is **not** a relaxation of the shared Infra production extractor.

Example for the originally attested index-24 PDF (indices 47, 244 and 286
were separately checked using new stage and output paths):

```bash
D=/data/var/knowmoredirt/knowledge-library/search-audit/NEW-STAGE-DIRECTORY
mkdir -m 700 -p "$D"
PYTHONDONTWRITEBYTECODE=1 /data/var/knowmoredirt/semantic-venv/bin/python   /data/src/github/KnowMoreDiRT/scripts/raspi_stage_original_pdf_page_repairs.py   --repair-plan /data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave50/read-only-pdf-page-attribution-repair-plan-327-originals.json   --offset 24 --limit 1   --extended-shadow-proof /data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave50/all-four-larger-PDF-physical-pagewise-shadow-verification.json   --stage-root "$D/extended-one-original"   --output "$D/extended-one-original-receipt.json"
```

The CLI **rejects all other long PDFs** absent a separately verified
four-original source-proof record. It creates no live migration capability,
model workload or source acquisition action. A future program version must
not turn this narrowly scoped staging exception into an unbounded or
unsupervised extraction default.

### Immediately usable private physical-page search on the 327 staged repair sources

Until the resident summary workers can be coordinated for a complete
rollback-safe original cache/semantic vector/summary migration, the approved
**327 staged source replacements** can be searched **without changing the
live KnowMoreDiRT native FileDB** using the opt-in, temporary
`scripts/raspi_staged_pdf_original_page_search.py`. This is a **supplement to
native and ordinary folder/grep search**, not a new persistent semantic
index, not an automatic publisher-data download and not a factual-answer
model. The script constructs a bounded **in-memory SQLite FTS5** index of
only the staged PDF source-unit bodies, excluding generated catalog
provenance headers. It supports an exact approved `--original-path` or all
327 sources, and restricts page candidates to at most 100, verified output
passages to at most 8, staged text input to 128 MiB and original count to
327. The RAM-only FTS database closes at the end of each invocation. There
is **no** persistent alternate FTS database to maintain and no model/GPU
resource lease required.

```bash
D=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave51
PYTHONDONTWRITEBYTECODE=1 /data/var/knowmoredirt/semantic-venv/bin/python \
 /data/src/github/KnowMoreDiRT/scripts/raspi_staged_pdf_original_page_search.py \
 'scientific calibration uncertainty benchmark evaluation' \
 --stage-preflight "$D/all-327-private-staged-original-PDF-preflight-complete.json" \
 --top 5 --candidates 25
```

An optional `--original-path 'ai-agents/.../document.pdf'` narrows the search
to one exact existing approved publisher original. The result is local JSON
with bounded publisher excerpt, **actual PDF physical page**, original
SHA-256, existing cached derived SHA, private corrected candidate SHA and
native original FileDB identity. Before any result appears, the verifier
independently rechecks the **complete original publisher PDF**, existing
old derivative, and exact staged candidate against their SHA-256 receipts,
plus the completed original-acquisition ledger, current source-map revision
and unique native FileDB entry. It then runs the existing original-PDF
`verify_pdf_page_excerpts`, returning a page citation **only if the claimed
page's original PDF text actually contains the excerpt**. A failed, timed-out
or ambiguous original-page match is returned as unverified and excluded from
confirmed hits. No source-model summary is promoted into a citation.

**Lexical relevance safeguard:** The temporary FTS5 indexes *only the source
body*; the original filename and generated title/header metadata are
**unindexed** and cannot drive search rank. For a multi-term query, an actual
quoted publisher excerpt must contain at least **two distinct normalized
query terms** before its physical PDF page can be offered as search evidence.
One-term queries require one matching term. Candidates passing physical-page
proof but matching only a generic isolated word are rejected as weak excerpt
coverage, *not misclassified as fake publisher citations*. The source excerpt
remains a search candidate, not a completed factual answer.

Independently observed on Raspi: the full 327-original input indexed
**7,157 staged original PDF physical-page text units**, a total of
**54,493,366 bytes**, into **RAM only**. Searching for *scientific calibration
uncertainty benchmark evaluation* returned **five independently original-
physical-page-verified citations** across five different publisher papers,
with no unverified hits shown. A separate exact-original AgentSteerTTS query
returned three publisher-page-matched passages, on actual pages **6, 10,
and 11**, instead of the old physically impossible 245-page derivative.
After adding the source-body-only and two-query-term relevance guards,
the same entire 327-book query again returned **five publisher-original-page
verified passages** and excluded one weak excerpt. A second read-only
predeclared **six-topic** original-page QA (scientific calibration, multi-agent
speech synthesis, 3D reconstruction, agent tool use, financial price history
and causal inference) produced **18 publisher-page-verified passages across
all six queries**, with one weak excerpt rejected. See
`continued-20261121-wave52/six-domain-private-physical-page-retrieval-evaluation.json`.
This verifies original physical PDF page evidence across topics, not source
relevance judgments by a domain expert or representativeness of the entire
17,715-source library.

These are observed **lexical** retrieval and original-page proof, not a
representative semantic recall benchmark or a claim that every passage in
all 327 sources has been individually verified. The private JSON receipts
for both tested queries are under `continued-20261121-wave52/`, with
publisher passages deliberately excluded from Git and owner reports.

The original PDF files, all other library files, current FileDB vectors,
old source-map/cache, recursively generated summaries and both running
indexing workers are unaffected. Once the full coordinated native migration
is safe, native source search should again serve the repaired revisions
without relying on this temporary staging-only path.

### Native discovery with opt-in original-verified corrected PDF pages (November 22)

The normal `raspi_hierarchical_library.py find-source-evidence` reader can now
add the already-verified corrected **publisher physical PDF page excerpts**
as a **separate optional field** for *only the PDF originals that native
search already selected*. The default command and default output are
unchanged. This directly makes corrected page references usable during
normal agent document search, before a safe full cache and vector cutover.
It does **not** change native rank, stored vector content, source map, old
cached text, summary trees or any original PDF. It is not a second
persistent search service and never considers other unpublished sources.

Opt in only when searching on Raspi with the previously independently
verified private stage receipt (never place the private source text or
private receipt in Git or external outputs):

```bash
P=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave51
PYTHONDONTWRITEBYTECODE=1 /data/var/knowmoredirt/semantic-venv/bin/python \
 /data/src/github/KnowMoreDiRT/scripts/raspi_hierarchical_library.py \
 find-source-evidence 'scientific calibration reliability evidence' \
 --source-name 'ai-agents/2026-acl-scientific-calibration/2026.acl-long.1423.pdf' \
 --private-staged-pdf-preflight "$P/all-327-private-staged-original-PDF-preflight-complete.json" \
 --documents 1 --passages 2
```

The additional `corrected_private_publisher_page_overlay` field reports the
native-selected publisher source and up to two independently verified
original-PDF **physical** page excerpts per selected original, or a
no-proof status. The private stage receipt must match all 327 approved
sources; a duplicate/malformed receipt fails closed. The operator requires
read-only source-map and original FileDB handles and matches the specific
native selected PDF's original SHA-256, old derivative SHA and native file
entry to the private candidate. It hashes the original publisher PDF, old
live derivative and corrected private candidate on each proof, and requires
original PDF text on the claimed page to contain the quoted excerpt.
Filename or catalog headers alone cannot supply source hits. An
incomplete/unavailable original or an incorrect page is not relabeled as
verified. For a native original that is *not* in the 327 repaired sources,
no private overlay is attached. A named original uses FTS5 ranking
**restricted to that verified original**, preventing other documents'
generic terms from crowding out its own pages. All queries operate on a
bounded RAM-only temporary text index; no new persistent database is made.

**Real independent behavior checked November 22:** Three different
existing approved publisher PDF originals for scientific calibration,
ACL evidence retrieval, and speech-science each returned **two**
corrected-private-page excerpts that were independently reproduced on
the original physical PDF (six verified page proofs). A deliberately
irrelevant five-term query to the calibration original returned **zero**
verified corrected-page passages. The ordinary native `documents`,
`evidence_kind_counts` and native selection basis were structurally identical
when the **same** query was repeated with and without this optional layer;
no private stage was loaded for the default command. Both background
summarization service process identities remained unchanged and active.
Private complete query receipts and source excerpt text are under
`continued-20261122-wave54/`; only the behavior and counts are described
here. A verified page is *source evidence*, not a measured semantic recall
or generated-answer correctness claim.

**Independently pre-frozen original-page retrieval check:** Before replaying
any retrieval results, the private QA script deterministically chose **32**
repaired publisher PDF originals by SHA-256 source-path ordering. For each,
it read an independent *original physical publisher PDF page* with Poppler,
froze its actual page text hash, one exact sentence and an associated
six-term **lexical** probe based on that original sentence; no staged
retrieval score chose the publisher page or query. The subsequent evaluation
reverified all **32** frozen full original PDF hashes, original-page text
hashes and excerpt matches, and applied the strict original/old-cache/new-
private-stage SHA and FileDB identity checks before considering a citation.
The corrected-page RAM-only path independently returned the expected original
physical page as its **top result in 30/32** cases and within its **first
three results in all 32/32** cases. It withheld **113** lower-ranked weak or
unverifiable excerpt candidates instead of displaying them as confirmed
publisher-page citations. The unchanged old native text on these *exact
same pre-frozen 32 source/page pairs* located the expected publisher page
within its top three results in **29/32** cases. The three failures consisted
of **two** original PDF searches with no verified excerpts and **one** case
where native search returned other verified pages but missed the frozen gold
page; the corrected private version found the expected gold page in all
three cases, each at top rank. This is a reproducible, controlled **exact-
source lexical page-location experiment**, not an independent natural-language
user query benchmark, semantic document-discovery recall test, or proof of
answer truth. Do not extrapolate these 32 cases to all other library sources.
The uncommitted private frozen-gold, all original-page verification evidence,
per-query comparison and both locally reusable evaluation scripts are under
`continued-20261122-wave54/`. This QA used original publisher files only:
no new material was downloaded, no source indexes/summaries were rewritten,
and both live summary workers remained active.

The real 327-source reindexing,
summary refresh, stale-warning removal and transactional cutover still remain
outstanding. Until then, use this read-only option when you need corrected
publisher-page citations, or use normal native search, source folders and
ordinary grep without any private staging input.

### No-action native cutover dependency inventory (before any real migration)

Preparing all corrected publisher-text derivatives does **not** by itself
repair the live corpus. Use the separately tested, *read-only* command
`scripts/raspi_pdf_repair_cutover_readiness.py` to inventory the exact
original-revision dependencies without stopping resident workers, acquiring
writer locks, modifying source-map files or adding duplicate vectors.
The tool requires the complete **327-source** private stage-preflight receipt,
rehashes every complete approved original PDF, every old live derivative and
every proposed private replacement, checks exact source-map and unique native
KnowMoreDiRT FileDB identities, and reads the current `content_chunks` and
`content_representations` counts for those originals. The independent
second-stage audit also derives the original/old-derived SHA-specific
hierarchical summary-tree and per-original progress-journal filenames using
the established `revision_identity` function, checks their private SHA-256
and original source header, refuses native summary rows with a missing or
wrong-revision source tree, and reports unexpected already-created trees for
the proposed new source revision. It **does not** remove any existing tree,
progress journal or embedding, and does not infer that a summary attached to
a bad-page old derivative is now correct. It directly checks both resident
summary worker service states. A missing/changed file, ambiguous
original, stale receipt, source-map mismatch or missing service observation
fails closed. Both SQLite connections are opened with `mode=ro` and
`PRAGMA query_only=ON` and the result is written to a **new mode-0600 JSON**
under private Raspi search-audit; no publisher text is exported in the result.

```bash
D=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave51
PYTHONDONTWRITEBYTECODE=1 /data/var/knowmoredirt/semantic-venv/bin/python \
 /data/src/github/KnowMoreDiRT/scripts/raspi_pdf_repair_cutover_readiness.py \
 --stage-preflight "$D/all-327-private-staged-original-PDF-preflight-complete.json" \
 --output "$D/NEW-unique-native-cutover-readiness-receipt.json"
```

The real independently run inventory checked all 327 staged source revisions:
**1,188 existing original-content vectors** and **12,623 native generated
summary representations/vectors** are attached to these originals, across
**119** documents with native summaries. Exactly **119 old-revision
hierarchical summary-tree JSON files** and **119 corresponding progress
journals** were independently identified and SHA-checked by the extended
readiness inventory; **zero** newer-revision summary trees had been created.
All detected native summary documents have their expected old original plus
derivative revision identity, so they can be targeted without confusing other
books' summaries. These rows must not be orphaned,
misattributed to the new derivative SHA, or silently duplicated during a
migration. Both workers were still `active`; the report explicitly returned
`workers_not_quiesced=true` and `safe_to_cutover=false`. A separate read-only quality gate compared *all 327* old versus privately
corrected PDF extracted texts after removing the KMD-generated provenance
header and derived `[[SOURCE UNIT: ...]]` page markers, using case-folded
readable word tokens and occurrence-weighted **three-word sequence counts**.
All **327 of 327** preserved **100% of these three-word sequences in both
directions**, with **zero word-count changes**. This is an independently
recorded lexical/text-retention result (private receipt
`continued-20261121-wave51/second-turn-private-327-corrected-pdf-word-trigram-retention.json`),
not a proof of identical byte layout, OCR quality, ordered section boundaries,
illustration fidelity, factual correctness, or every possible original-source
quotation. The real publisher physical-page sampled citation proofs still
matter; neither the legacy wrong page labels nor original source PDFs were
modified by the comparison.

The tool returns
`safe_to_cutover=false` **even when both services are inactive**, because
service inactivity alone does not acquire exclusive writer leases, prove no
other consumers, create a cross-file atomic transaction, or establish a
verified rollback. This tool is deliberately *incapable* of promoting the
staged sources; the real migration procedure remains unimplemented and
requires separately verified coordination and tests. This preserves all
native semantic search results and the original books as they are until the
full dependent-state replacement can be safely verified. See the private
readiness receipts `continued-20261121-wave51/second-turn-native-cutover-readiness-327.json`
and `continued-20261121-wave51/second-turn-native-cutover-summary-tree-readiness-327.json`.

### Sealed private recovery subset and forced-rollback rehearsal (November 22)

The next phase after source/page-readiness is a **fully private recovery
snapshot and simulated failure rehearsal**, **not** a live cache/database
migration. Use `scripts/raspi_pdf_repair_shadow_cohort.py` with the existing
full original/SHA-verified 327-stage preflight and a new exclusive private
folder. The tool reruns the exact 327 source-map and native ID preflight,
including SHA-256 of each original publisher PDF, both old and corrected
text revisions, current old summary tree and per-source progress journal.
It copies only the **affected old cached text, old summary tree and progress
files** into a protected recovery folder; all original books remain in their
ordinary public folder structure and are never copied, deleted or rewritten.
It reads the two live SQLite metadata stores with `mode=ro`,
`PRAGMA query_only=ON`, and separate read-only transactions, copying only
327 original source-map rows, the 327 corresponding native file entries,
their content chunks/vectors and source-linked generated representation rows
into **one new private SQLite clone** with the source table constraints and
foreign keys preserved. The original 3-GB database is never bulk-copied or
written. Foreign-key parent lookups in the **private clone** have dedicated
indexes, so the isolated cascading-delete rehearsal is bounded rather than
quadratic. Both registered indexing workers remain running; the script
never stops or restarts them, and never changes the live source-map, native
FileDB, old derived caches, summaries or search results.

The isolated SQLite clone then **simulates updating all 327 source revisions
and clearing their existing source-linked vectors/summary rows** and
*deliberately rolls the transaction back*. Logical digests of **every column,
including embedding BLOB bytes**, are compared across all four original
tables before and after. SQLite `quick_check` and `foreign_key_check` must
pass. The independently reusable verifier
`scripts/raspi_verify_pdf_repair_shadow_bundle.py` subsequently checks the
full SHA-256 of every saved publisher-text and old summary/progress file,
SHA-256 of the SQLite clone, exact source-map/native original identities,
all 327 required old text records, the old summary-tree/progress identities
for any original with native analysis rows, and the original full-column
logical table digests. An incomplete, modified, duplicate, foreign-key-
broken or wrong-revision archive fails closed.

```bash
P=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave51
D=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave53
V=/data/var/knowmoredirt/semantic-venv/bin/python
# Use NEW unused names: never overwrite a historical private recovery bundle.
$V /data/src/github/KnowMoreDiRT/scripts/raspi_pdf_repair_shadow_cohort.py \
 --stage-preflight "$P/all-327-private-staged-original-PDF-preflight-complete.json" \
 --new-private-bundle-dir "$D/NEW-private-versioned-recovery-bundle"
$V /data/src/github/KnowMoreDiRT/scripts/raspi_verify_pdf_repair_shadow_bundle.py \
 --private-bundle "$D/NEW-private-versioned-recovery-bundle"
$V /data/src/github/KnowMoreDiRT/scripts/raspi_rehearse_pdf_repair_file_db_restore.py \
 --private-bundle "$D/NEW-private-versioned-recovery-bundle" \
 --staged-preflight "$P/all-327-private-staged-original-PDF-preflight-complete.json" \
 --new-private-rehearsal-dir "$D/NEW-private-cross-file-fault-rehearsal"
```

A separate cross-file failure rehearsal uses **only copied old source text,
old summary tree/progress and a cloned native/source-map subset**, plus one
independently SHA-pinned corrected private source: it injects failures
**immediately after a text/summary swap** and **after a committed metadata
change and cascade removal**. Both failure points must be restored from the
sealed historical sources, including full original text SHA, summary/progress
SHA, all native old rows and metadata revisions. These tests do **not**
exercise live concurrent writers, cross-database atomic commit, a durable
writer lease, or end-user query continuity; a genuine live migration must
still be coordinated and independently verified.

**Actual result (private wave53):** 327 old text files plus 119 revision-pinned
summary trees and 119 corresponding progress journals (**565 files,
46,210,889 protected bytes**) saved. The selective private SQLite image was
**61,870,080 bytes** and contained **327 old source-map rows**, **327 original
native file entries**, **1,578 original-content chunks** (1,188 already had
vectors in the preceding preflight), and **12,623 original-linked generated
summary representations**. A full cohort rehearsal briefly removed all
1,578 chunks and their 12,623 dependent rows **only inside the cloned SQLite
transaction**, then recovered all four tables with exact logical byte hashes
and no foreign-key errors. The separate SHA-based archive verifier accepted
all 565 files and all 119 summary/progress identity pairs. In an additional
single-original cross-file test, both injected failure points correctly
restored its old source text, summary tree/progress, and all 9 original native
chunks with 244 original-linked representations. The sealed bundle occupies
about 105 MiB, and the private cross-file failure simulation has separate
protected scratch files; neither changes live publisher books or metadata.
Reproducible private proofs, command logs and full manifests are under
`continued-20261121-wave53/`. These are recovery *rehearsal* results,
**not permission or a ready procedure to promote the 327 staged files**.

### Read-only recovery-bundle freshness against ongoing source reindexing

A recovery bundle can pass its own archived hash and SQLite integrity checks
yet become **stale** as the original two summarization services continue to
append new source-linked representations, revised document chunks, or new
old-revision summary trees. The recovery folder is a *historical snapshot*;
correctly rebuilding it at one instant while workers remain active does not
make it a current exclusive rollback point.

Use the separately tested **read-only**
`scripts/raspi_pdf_repair_shadow_freshness.py` before relying even on an
existing protected historical recovery copy. It first independently
validates all archived original text, old summary/progress file SHA-256
revisions, the full selective native DB image, all embedding bytes,
foreign keys and source-map publisher identities using the existing
archive verifier. It then opens both current live SQLite sources with
`mode=ro` and `query_only`, compares the entire present cohort of 327
source-map identities and native original IDs, every original-content chunk
and generated representation with **full-column digests including embedded
vectors**, and exact current revision-pinned summary-tree/progress files
against the archived versions. It detects both added/deleted rows and
changed bytes **even when SQL row counts stay equal**. No file is removed,
rewritten or migrated and the running models are not stopped. Any mismatch,
missing source, unreadable old tree, or changed progress journal is explicit.
Even when everything happens to match, the tool ALWAYS refuses to authorize
cutover because concurrent readers/writers and multiple independent files
are not made atomic by this check.

```bash
D=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261122-wave54
B=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave53/shadow-recovery-bundle-327-20261121-v1
PYTHONDONTWRITEBYTECODE=1 /data/var/knowmoredirt/semantic-venv/bin/python \
 /data/src/github/KnowMoreDiRT/scripts/raspi_pdf_repair_shadow_freshness.py \
 --historical-private-bundle "$B" \
 --new-private-output "$D/NEW-private-current-source-drift-check.json"
```

**Observed on November 22:** The first private recovery bundle was intact
and retained the exact prior original versions, but it was **already stale**:
its 327-original index snapshot contained **1,578 chunks**, **12,623 summary
representations** and **119 summary trees**, while fresh live read-only
queries found **1,605 chunks**, **13,496 summary representations** and
**127 summary trees**. That is **27 additional original chunk rows**,
**873 additional generated representations** and **eight additional
old-revision source summary trees** since the isolated backup. No original
publisher revision or cached derivative had been promoted; the fresh
327-original SHA and dependency inventory still reported both workers
active, **zero** corrected-revision summary trees, and cutover unsafe.
The historical archive is suitable for independent testing and comparison,
**not** rollback of those newly produced live summaries. Replacing the
live FileDB with this old subset would destroy newer work. No automatic
live-worker stop, overwrite or snapshot promotion is permitted.
The independent private proof is under `continued-20261122-wave54/`.

### Full 327-source private forward migration and forced-recovery acceptance (November 22)

The sealed historical subset permits a much stronger **isolated forward
migration** experiment than testing that one document's old state can be
restored. The new `scripts/raspi_rehearse_pdf_repair_forward_generation.py`
uses the full, previously SHA-verified **327-PDF** private staging receipt,
independently validates the sealed recovery archive before and after each
simulation and performs the transition **only inside a new exclusive private
audit directory**. It never opens the real source-map/native FileDB for
writing, stops either worker or overwrites the originals. It also explicitly
refuses a destination within the sealed recovery bundle or the private
staged-source tree; nested files cannot silently contaminate the archive.

For each historical original, the rehearsal independently validates its
canonical original SHA, old SHA, new staged candidate SHA, current historical
native ID and PDF original physical-page markers. Before creating any output,
it checks the corrected candidate's canonical provenance JSON, the physical
page sequence, exact staged SHA and source-unit counts. It deliberately
computes `text_characters` as **source-unit body characters excluding the
synthetic KMD provenance header**, not the entire derivative file length.
It privately copies all **327** corrected text derivatives into a separate
folder, with new-file-only creation, full post-copy SHA, file sync and no
existing-file replacement. In an isolated cloned SQL database, it updates
all **327** original-derived SHA and physical-page metadata rows, sets their
status to `staged_needs_reindex` and removes all **historical original-content
chunks and generated summary representations** using the original source's
native identity and the existing foreign-key cascades. The unrelated
publisher original file identity/metadata is unchanged. The private
transaction commits only after all 327 text copies succeeded and after
relational integrity and orphan checks pass. Its result is intentionally
**NOT native-search-ready**: new content embeddings, source-linked recursive
summaries, native reader synchronization and real PDF original-page checks
must still be regenerated under an actual live writer/reader coordination
protocol. This prototype never publishes a post-cutover source-map or a
live migration switch.

Use new never-before-existing output names in the audit root; each invocation
also produces a separate private JSON receipt. For example:

```bash
D=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261122-wave55
B=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave53/shadow-recovery-bundle-327-20261121-v1
P=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave51/all-327-private-staged-original-PDF-preflight-complete.json
V=/data/var/knowmoredirt/semantic-venv/bin/python
PYTHONDONTWRITEBYTECODE=1 "$V" \
 /data/src/github/KnowMoreDiRT/scripts/raspi_rehearse_pdf_repair_forward_generation.py \
 --historical-private-bundle "$B" --staged-preflight "$P" \
 --mode forward --new-private-generation-dir "$D/UNUSED-full-private-forward-dir" \
 --new-private-result "$D/UNUSED-private-forward-result.json"
# For isolated failure-recovery tests, use --mode fail_after_164_files,
# --mode fail_after_database_commit, or --mode restore_old, each with a
# DIFFERENT new-private-generation-dir and new-private-result.
```

**Observed full-size rehearsal:** **327** corrected publisher PDF derivatives,
**7,580** labeled original physical PDF source units (including continued
sections), and **53,929,527** original source-unit body characters were
verified and privately staged. **7,157** source units were eligible for
temporary RAM lexical candidate retrieval after the existing malformed-
glyph/empty-body screening; the remaining **423** are not verified quoteable
page evidence. The isolated forward SQLite clone removed **1,578 historical
native original chunks** and **12,623 historical native summary rows**, with
zero stale dependent rows and all original file metadata preserved. Its 327
new source revisions have `staged_needs_reindex` status, so the new isolated
copy is expressly **not** a published or usable full semantic index.
Two full-cohort crash simulations were separately executed: failure after
**164** corrected private files, and failure after the **327th** corrected
file and the private SQLite source-revision/dependent-row transaction had
committed. Both left the protected historical archive intact. Independent
restoration from that sealed archive recovered **565 historical old cached
source/summary/progress files**, the exact private 327-original SQLite image,
all **1,578** historical original chunks and all **12,623** historical
summary representations, with exact full-byte file and native-row checks.
These are **isolated fault-injection recoveries from the historical snapshot**:
because resident workers have meanwhile created more live summaries, the
archive is *stale relative to the live system* and cannot restore those newer
results. Preserve every protected audit folder until the supervisor has
separately verified that it is no longer needed. The private command logs and
receipts are under `continued-20261122-wave55/`; no original text or
private archive is committed to Git.

### Independent *new-revision native search readiness* gate, November 22

**A source-map SHA update does not mean the source is indexed.** The 327-PDF
private migration experiment intentionally creates corrected derivative files
and `staged_needs_reindex` source-map rows while deleting old original-specific
chunks and summary vectors. Those private files are NOT yet native-search-ready.
Simply setting `status=complete` after copying corrected text would still
produce an incomplete or stale index. Native KMD has independent
**original-content overview/full** vectors and **recursive-summary keyword,
facet and whole-document root** vectors. A future release needs every source
channel, not merely a copied file, a hash and one root summary.

The new `scripts/raspi_verify_pdf_reindex_release_fidelity.py` is a strictly
**read-only private-candidate quality gate**, not a live publication tool.
It independently re-verifies the existing complete 327-original historical
recovery bundle, its original source-map/identity and old direct-index channel
inventory, the fixed private correction-stage receipt, and the 327 corrected
candidate text derivatives. It verifies source path and original/derived
SHA, canonical provenance wrapper, physical PDF page sequence, source-unit
body character count **excluding generated header**, complete new source-map
state, and immutable native original IDs. From the historical native snapshot
it additionally identifies which originals previously had searchable **full**
content channels, and refuses to silently drop those at migration.

For each publisher original it refuses to accept new native passage chunks
unless their source spans and text hashes match the **corrected** derivative.
Both directly embedded original overview/full chunks and all recursive
summary/facet representations must have 384 finite float values, valid
byte hashes, consistent model identities and normalized nonzero vectors.
A further native keyword retention guard compares each original against
its sealed historical summary-vector channel. If the old original had at
least one native keyword/keyphrase representation, the corrected candidate
must have at least one new-revision keyword/keyphrase representation, not
just a root summary. This is a necessary minimum channel-preservation test,
**not** a claim that every previously extracted keyword is semantically
reproduced or that summary accuracy is proven. Synthetic tests exercise a
case where the old keyword channel disappears entirely and verify that
adding a revision-checked new keyword vector satisfies the minimum gate.

Every recursive representation must refer to the same original SHA and **new**
derivative SHA as its complete summary tree, with exact source range,
summary text SHA and summary-node identity. The revision-specific complete
summary-tree JSON must pass the original native hierarchy/coverage validator;
all tree nodes must appear with valid native representations, including the
full-document root. The existing production completed-root reader is also
run on the accepted sources, rather than counting any stale leaf as complete.
No write access is granted to either SQL connection. This tool never opens
Raspi's actual 3 GB production database as a candidate or reads publisher
content outside the guarded private staging paths.

Use **new** private output filenames, never overwrite earlier evidence:

```bash
D=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261122-wave56
B=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261122-wave55/private-forward-complete-327
H=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave53/shadow-recovery-bundle-327-20261121-v1
P=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave51/all-327-private-staged-original-PDF-preflight-complete.json
V=/data/var/knowmoredirt/semantic-venv/bin/python
PYTHONDONTWRITEBYTECODE=1 "$V" \
 /data/src/github/KnowMoreDiRT/scripts/raspi_verify_pdf_reindex_release_fidelity.py \
 --staged-preflight "$P" --historical-private-bundle "$H" \
 --private-candidate-db "$B/private_after_transition_not_search_ready.sqlite3" \
 --private-derived-root "$B/corrected_source_text" \
 --private-summary-root "$D/private-new-revision-summary-trees-missing" \
 --new-private-report "$D/ANOTHER-UNUSED-new-revision-readiness.json"
```

**Actual full-cohort negative acceptance:** all **327** corrected candidate
text derivative SHA revisions independently reverified. **Zero of 327**
new-revision original documents are currently native-search-ready; every one
was rejected because it still has `staged_needs_reindex` and no rebuilt
original-plus-summary vector hierarchy. **No evidence of a lost original or
failed running worker** follows from that expected private negative result.
The positive isolated fixture with an exact current root, full tree, direct
source vectors and matching normalized embeddings passes. Tests separately
reject a missing original overview, a missing *historically present* full
content channel, stale source spans, stale keyword/facet metadata, incomplete
summary trees, malformed source revision, mismatched embedding SHA, finite
zero vectors and nonfinite embeddings. Even if a future isolated candidate
passes every check, its result explicitly says **safe_to_cutover_live=false**:
the indispensable writer/reader quiescence, synchronized source-map/native/
cache/summary generation switch, fresh rollback point and original-publisher
page-citation checks remain separate release gates.

### Opt-in cooperative source-boundary hold and live-acknowledgement gate (November 22)

A **strictly opt-in** source-boundary quiescence protocol is implemented by
`src/file_system_catalog/source_page_quiescence.py` and the existing native
`scripts/raspi_hierarchical_worker.py`. The absence of the owner-only control
file `hierarchical-publisher-revision-quiesce.json` deliberately leaves the
current worker behavior **unchanged**. No extra model concurrency, new
worker process, startup flag or change to installed systemd services is
introduced. The source-boundary hold is **not active on the running Raspi
workers**: both existing Python processes were started on November 8 and
have not loaded this newer in-memory worker code. Git checkout fast-forward
alone does not reload running Python. Never create a live hold marker,
restart/stop the long-running workers or evict their shared models merely
because the new code exists; coordinate a verified safe source-boundary
transition with the System Integrity Supervisor first.

A future separately coordinated maintenance operator may produce a strict,
owner-only, 0600, single-file, monotonic JSON generation with the exact
`kmd.publisher_pdf_document_boundary_quiescence.v1` schema, fixed scope and
`action=hold`. This module **provides only the pure encoder and validated
reader**; it is NOT the operator and cannot create the live control marker.
The operator must first independently establish that BOTH workers have
already naturally loaded the new protocol, retain their registered model
leases and have no interrupted source build/import. When an upgraded worker
sees a hold, it **finishes the existing source** before acknowledging, then
refuses to begin another source while waiting. On each bounded heartbeat it
atomically publishes its own actual `quiesce_generation`, control SHA,
resident PID, fresh timestamp, `current_original=None`, and
`phase=publisher_revision_quiesced`. A strictly later `action=release`
generation allows that worker to continue; same-generation rewrites,
rollback, deletion, symlink, group-readable markers and invalid ownership
fail closed. No direct signal or service control is involved.

`python3 scripts/raspi_publisher_pdf_quiesce_readiness.py` is a **read-only**
operator-facing prerequisite check. It examines the real owner-only status
files, installed systemd MainPIDs and actual `/proc` command lines and
requires BOTH currently running workers to acknowledge the identical hold
generation, digest, process identity and fresh heartbeat. It has no `--apply`
option and NEVER creates or modifies the marker, any source file, databases,
models, worker services or owner reports. A missing marker or an old worker
returns `both_workers_quiesced_at_complete_original_boundary=false`. The
observed current Raspi result had no hold marker, both resident workers had
**not** loaded the new pause protocol, and full native cutover remained
**unapproved**. This is expected, not an error to work around by forcing
restarts. Synthetic tests exercised the actual worker loop with a hold
arriving during the first source import, verified the import finished before
the boundary acknowledgement, proved the second source could not begin
until a later release, and verified a requested stop while held never
started a new source. Replayed manifests, missing controls after restart,
forged/stale status and invalid worker PID/commands also fail closed.

**Important independent release gates remain:** acknowledgement of BOTH
summary workers is only one *necessary* condition, not sufficient to freeze
*all* FileDB/source-map/cache/summary writes or existing query readers.
The production native FileDB writer lock, source-map updates, manual indexing
consumers and read-only search processes require a separately implemented,
independently tested all-store reader/writer lease and fresh complete snapshot
and rollback plan. Read-only pause readiness must always report
`safe_to_cutover_live=false` even if two workers correctly hold. The 327
corrected publisher texts and new native vectors/recursive summaries must
then be rebuilt with real physical PDF page provenance and retrieval quality
verified before any user-facing native revision becomes available. The current
old original sources and running indexer services remain intact.

**Remaining explicit release gate:** obtain actual verified coordination for
both live summary workers and all other native reader/writer consumers,
prevent stale citations during migration, establish a fresh fully quiesced
source-map/cache/native/summary state with rollback for **multiple independent
filesystem and SQLite stores**, re-run all original revision checks, then
incrementally reindex the corrected sources and independently verify physical
publisher page excerpts, retrieval and clean stale summary invalidation
before declaring native search fully repaired. A private snapshot taken while
summary workers are live is **historical evidence**, not a current release-
ready rollback point. Do not auto-stop shared services, remove older live
vectors, erase originals or weaken the existing citation warnings merely
because the isolated test passed.

**Independent full private stage acceptance, November 21:** All **327**
originals with independently confirmed stale out-of-range cached PDF page
labels now have exclusive, privately stored, physical-page-correct candidate
text derivatives. The existing automatic bounded parser staged **323**;
only the four previously attested and SHA-pinned longer originals used
individually bounded extended page extraction. Two initial SQLite reader-lock
rejections were corrected by staging exactly those two sources in a *new*
private path without replaying or overwriting their prior receipts. An
independent separate excerpt check matched **981 of 981** selected source
excerpts against the **actual original physical publisher PDF page** across
the 327 staged PDFs. Two original text-layer font issues required a shorter,
still exact, original-PDF-verified excerpt; neither had been accepted as a
citation before its successful second proof. The final read-only preflight
**rehashes the complete original PDFs, all existing old source-map
publisher-text derivatives and all staged replacements for all 327 again**,
verifies current completed acquisition receipts, exact existing source-map
revisions and unique original FileDB IDs, cross-checks the existing 327
SHA-gated warnings, validates increasing original physical page labels,
and asserts all 981 page-sample proof references. Its private checkpoint
is `continued-20261121-wave51/all-327-private-staged-original-PDF-preflight-complete.json`
(SHA-256 `46bfb4ae4e463168eeb10d9ef7bfd08a454307fc99739a8134bbfd2440e302f0`).
No private publisher text was copied into Git. **Staging is NOT live repair:**
all 327 old original-linked derivatives, the canonical native FileDB vector
and chunk representation, 119 previously observed affected recursive summary
trees and the two live workers were deliberately left unchanged. The existing
native source-revision-gated page citation warnings therefore remain active.
Only a separately coordinated transactional, rollback-safe migration across
all affected dependencies can clear those old source-revision warnings after
physical-page regression checks. This preflight also does not claim complete
page-by-page text or image-fidelity validation beyond sampled original PDF
page excerpt checks; the 168 OCR-only-marked PDFs are a separate review
population and were not silently reclassified as repaired.

The script refuses existing output receipts/staging folders, out-of-audit
output paths, stale source SHA revisions, invalid/duplicated FileDB original
identities, changed source-map rows, mutable original/derived bytes, original
physical PDF count disagreements, false/repeated/reordered/invented page
labels, untrusted derivative page delimiters inside publisher text, and
extraction outside the existing safe sixty-physical-page/fifty-MiB bounds.
Blank original pages may be absent from the staged text but the emitted
page identities must remain strictly increasing and within verified PDF
physical-page bounds. Stage receipts explicitly distinguish
`live_promoted=false` and `original_physical_page_excerpts_verified=false`:
matching source bytes and page marker bounds are **not physical page citation
proof**. Independently verify selected staged source excerpts against the
original PDF with `publisher_pdf_page.verify_pdf_page_excerpts` before any
future live migration. Staging intentionally provides **no** CLI to rewrite
the live source-map, native FileDB vectors, summary graph, original PDF or
existing derivative. The separately coordinated rollback-capable migration,
resident-worker/source-lease quiescence, invalidation/rebuild and physical
PDF page regression are still REQUIRED and not delegated to this command.
A successful staging receipt is **not** evidence that any live citation or
summary was repaired. Ordinary original folder/grep search remains usable.

**Raspi private acceptance, wave fifty-one:** The first twenty-four unique
revision-pinned affected PDFs were staged without touching the running
original index: twelve initial successes, ten successes plus two bounded
SQLite read-lock rejections in the next batch, then exactly the two
rejected originals successfully staged in a **new** private retry directory.
The tool uses an explicit read-only forty-five-second SQLite lock wait, not
a worker restart. The private receipts distinguish the successful/failed
source identities rather than treating a partial batch as complete. In a
separate publisher-source test, **all seventy-two sampled original physical
PDF page excerpts** across those twenty-four staged sources were matched
against the exact approved original SHA and the correct actual physical
page. Two difficult-page verifications needed a shorter still-exact excerpt
from the existing PDF text layer; neither was promoted as evidence before
independent physical page verification. Each staged output has a full staged
SHA-256 and the old original/derived source revisions remain unchanged.
The private evidence root is
`/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave51/`;
`first-24-staged-originals-all-72-publisher-PDF-page-proofs.json` contains
source hashes and page proofs **without copyrighted publisher passages**.
This validates the staging *method* for a first batch; it is deliberately
not a live native vector or hierarchical-summary migration.

The four beyond the existing sixty-page automatic fallback have now
**also been independently tested in private**, using bounded single-physical-
page `pdftotext -f N -l N` reads on the **unchanged, approved** original PDFs:
64, 76, 70 and 418 physical pages, for **628 physical page extractions**.
The private read-only shadow generated exactly those original physical-page
identities and independently verified eighteen source-PDF page excerpts from
sampled first, middle and late pages; it stored only SHA receipts and counts,
not copyrighted publisher text or replacement derivatives. The current
Infra implementation still **refuses live automatic pagewise extraction for
these four** due its existing safety bound; this was a test of a possible
bounded extension, **not a deployed parser change** or bulk source repair.
The separate private long-source evidence is under the wave-fifty audit
folder in `all-four-larger-PDF-physical-pagewise-shadow-verification.json`.

Do **not** directly overwrite the old SHA-pinned derivatives while the two
resident summary workers or other dependent native semantic tasks can use
them. Coordinate service leases and a rollback-capable, versioned, transactional
refresh of affected source-map, extracted text, native vectors and hierarchical
summaries; keep approved originals and ordinary grep searches untouched.
All technical evidence is private under the wave-fifty audit folder. The
remaining **168** completed PDFs without a standard `PDF page N` label were
independently classified as **OCR-only-marked** extracts, not mistakenly
counted as sound physical-page citations by the overcount audit; OCR text
requires separate original-page and figure-quality validation before citing.

The source of many excessive labels is older cached text extraction splitting
*every* Poppler form-feed, including form-feed controls embedded inside
physical pages by PDF fonts. The reproducible example `AgentSteerTTS` has
**19 actual PDF pages** but the cached derivative has **245 labeled source
units**; its original and derivative hashes can both validate while its
page labels are wrong. The current Raspi Infra PDF extractor already checks
the expected PDF page count and, for bounded eligible files, falls back to
per-physical-page extraction; it does not automatically repair unchanged
historical derivatives. The original files have **not** been modified and
neither stale derived pages nor dependent native vectors/summaries have been
silently rebuilt while the two resident indexing workers are active.

Private reproducible evidence:
`search-audit/continued-20261121-wave50/corpus-original-pdf-page-overcount-independent-audit.json`
contains original-relative paths, current original/derived SHA-256, native
metadata page count, independent actual `pdfinfo` count and maximum cached
page marker for each fully verified candidate; no publisher full text is in
the audit. `live-pdf-derivative-overcount-safety-proof.json` exercises the
new read-only lookup against one invalid original and one page-count control.
The current native indexing workers must be coordinated before any targeted
re-extraction, source-map revision update, invalidation of dependent vector
chunks, or recursive summary rebuild. Do not mark old source-revision-linked
citations verified simply because an original-file SHA matches. An unverified
page is a **source provenance gap**, not a source to replace by unrelated
papers or a request to download anything externally.

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

### Private, complete-generation pointer and two-database source-revision checks (November 22)

The publisher PDF repair pipeline cannot safely replace the live source-map
SQLite (WAL mode), native FileDB SQLite (DELETE journal mode), source-derived
text files and recursive summary JSON as one transaction. These are **separate
storage engines and filesystem operations**. One SQLite commit, a correct
source-map SHA, or two paused recursive-summary workers cannot constitute an
atomic multi-store release. The existing Raspi scheduled direct-embedding
worker and semantic extractor are independent database writers;
direct embedding uses the native FileDB content lock, while extraction
uses its own `semantic/extract.lock`. Neither is covered by the optional
recursive-summary document-boundary hold. The hourly `knowledge-library refresh`
job is different: its inspected implementation updates the generated source
catalog and README, **not** the native KMD vectors or semantic source-map.
It still belongs in overall source inventory and deployment coordination,
but is not itself evidence of an independent native database writer.
Installed legacy native-search readers also read the live mutable paths,
not a version-selected immutable generation.

For future architecture evaluation, the new
`src/file_system_catalog/private_source_generation.py` is a deliberately
**isolated and private-only** generation protocol. It refuses the actual
Knowledge Library and live KMD directories and has **no production CLI**.
The caller supplies a new version directory under a private audit root with
**two separately addressable SQLite files**, the exact new-revision derived
text for every selected original and that original's exact revision-pinned
complete recursive-summary JSON. An owner-only manifest lists the exact
relative paths, byte counts and hashes, plus the independent native revision
QA receipt. The verifier requires a complete unique source-map cohort of
PDFs, verifies every current original/derived SHA and exact associated text
file and complete summary tree, verifies the native root for every current
source, checks both private SQLite files' integrity, and binds the *positive*
original/vector/summary QA receipt to the exact bytes of **both** SQLite
files. A QA receipt declaring incomplete reindex or an old staged-only source
is rejected even when its publisher text SHA passes. Merely renaming a source
or updating all listed file hashes without preserving the original/source
identity is also rejected. It never claims a summary proves factual truth.

The related read-only `raspi_verify_pdf_reindex_release_fidelity.py` supports
`--private-source-map-db` for a **separate** isolated corrected source-map
snapshot; when omitted, it retains the earlier single-private-clone testing
behavior. It records separate native/source-map database SHA-256 values and
independently requires that neither private candidate file changed between
its first SHA and the end of its all-source evidence check. These are evidence
requirements for an eventual complete private generation, not authority to
promote a source.

For the private switch test, an exclusive owner-only file lock serializes
marker updates. The marker is **one** atomically replaced, fsynced file
pointing to a numbered prevalidated immutable generation. Readers acquire a
shared lock, verify all assets, then hold the selected generation's exact path
and SHA rather than rereading the mutable marker for each source component.
A prior generation is **never deleted** by this protocol. A failed validation,
missing tree, stale source, corrupt vector, inconsistent native QA,
symlinked or world-readable lock, replayed generation, or a simulated crash
**before** marker replacement retains the old selection. A simulated crash
**after** marker replacement requires reading the current marker and verifying
its selected contents instead of blindly replaying the uncertain operation.
A reader already pinned to the old generation can still validate that old
source while a second reader pins the newly selected generation. The
real native-source vector verifier is exercised on two isolated original-PDF
fixtures, including **separate source-map and native SQLite files**; this
is not merely a symbolic pointer test. All test paths and inputs are temporary
or owner-private copies, not the public publisher originals.

**Observed latest old-revision recovery snapshot:** A new protected private
bundle for all **327** corrected publisher PDF original identities preserved
**585** current old-version derivative and summary artifacts (46,484,409
bytes) and a **67,334,144-byte** selective native/source-map SQLite snapshot
with **1,612** source chunks and **13,767** dependent native summary
representations. A separate complete private-bundle SHA verification and a
fresh read-only comparison against the live original FileDB returned the
**same** 327 original/derived identities and those exact chunk/representation
counts, with **zero** original or native identity mismatches and no observed
cohort difference at comparison time. This is a substantially newer snapshot
than the preceding historical 12,623-representation recovery bundle, but
**still not a proven quiesced multi-store rollback point**. Uncoordinated
writers may continue changing other documents while the private snapshot is
captured. No full-library snapshot, live promotion or service interruption
was performed.

**Observed release result remains negative:** All 327 previously staged
corrected PDF text derivatives have verified private SHAs, but the private
forward-migration copy deliberately has zero rebuilt complete native summary
roots and still uses `staged_needs_reindex`. The new double-SQLite-SHA
read-only QA receipt likewise rejects all 327 as unfinished; no private
versioned pointer can replace missing real native vectors or required
source-page citations. The pointer protocol is an architectural **prototype**
only: it does **not** make the installed Raspi native readers version-aware,
does **not** quiesce all independent writers and readers, does **not** install
an executable live cutover operator, does **not** coordinate the existing
November 8 resident worker processes, does **not** copy original publisher
PDFs or authorize downloading and deliberately reports
`safe_to_cutover_live=false` even on isolated positive tests. A true release
requires independently coordinated all-store writer/reader admission,
complete original-preserving backup, tested recovery, full new-revision
embedding and recursive-summary rebuild, real publisher physical-page
verification, and real end-to-end legacy-reader integration. Until then the
old original/source-map/native paths remain authoritative and untouched.

### Independent native direct-writer and source-map extractor release admission (November 22)

The private `scripts/raspi_pdf_repair_additional_writer_readiness.py` is a
**read-only**, fail-closed Raspi admission inventory. It reads the installed
systemd service and timer states for the two writers **outside** the
recursive-summary workers: `knowledge-library-semantic-worker.service` and its
timer can prepare and write direct overview vectors into the native FileDB;
`knowledge-library-semantic-extract.service` and its timer can replace derived
text and update the separate semantic source-map. The actual deployed Python
writer implementations and their locks were inspected before assigning these
responsibilities. The tool also invokes the independent read-only
`raspi_publisher_pdf_quiesce_readiness.py`, which requires both resident
recursive-summary processes to acknowledge the **same real complete-document
boundary**, not merely an inactive or allegedly idle process. It reports
these separate blockers explicitly and **never** grants permission to
perform a live migration.

A timer unit's `systemctl show` normally omits `MainPID` because it is **not
a service**. The inventory accepts that absent field **only for an exact
allowlisted timer**, representing no timer main process, but requires a
reported actual MainPID from each installed service. Unknown, failed or
missing service/timer states fail closed. A process being inactive does not
make a still-armed timer incapable of restarting it during migration; manual
or otherwise unregistered writers and legacy readers remain unknown even if
all named services happen to be idle. The existing hourly
`knowledge-library-index.service` is correctly classified separately as a
**catalog/README refresh**, not a native-embedding or source-map writer; its
source was inspected rather than assuming that every service named "index"
writes the semantic DB.

Usage on Raspi is private-only and cannot overwrite an existing report:

```bash
D=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261122-wave57
PYTHONDONTWRITEBYTECODE=1 python3 \
 /data/src/github/KnowMoreDiRT/scripts/raspi_pdf_repair_additional_writer_readiness.py \
 --new-private-output "$D/ANOTHER-NEW-read-only-writer-inventory.json"
```

**Current independent live observations:** Both direct semantic/vector and
source-map extraction services were inactive, but **both scheduled timers
were armed**. The two resident recursive-summary worker processes were still
active and had **not** loaded or acknowledged the newer optional source-
boundary pause protocol. The source-map and native database use separate
journal modes; no universal read/write generation admission or rollback
lease exists. The read-only inventory therefore listed seven explicit
blockers, including the two scheduled timers, old worker code, missing
quiescence and unpinned legacy native readers. Its result
`safe_to_cutover_live=false` is **not** a malfunction of indexing; it is the
correct evidence that a coordinated release has not happened. No timer,
service, shared model, source file, native FileDB, extracted text or source-map
was stopped or changed. Mocked writer/timer/protocol failures and the real
unit types were independently tested, and the original source corpus remains
available through ordinary folder search and existing verified retrieval.

### Same-snapshot, physically source-checked three-channel retrieval (November 22 continuation)

The installed two-channel native discovery and standalone original-passage
search remain the live interfaces. A previously frozen seventy-three-question
retrieval evaluation was previously inconclusive because vector and optional
lexical results had been recorded on different dates as the live summary
index expanded. It was also observed that all eleven predeclared source-label
misses of the earlier forty-six-question benchmark **already had complete
current original-linked recursive summary roots**. Those specific failures
were retrieval ranking/discovery failures, not missing summaries.

The **optional, private read-only**
`scripts/raspi_frozen_same_snapshot_three_channel_eval.py` captures its three
candidate channels from **one** bounded native SQLite read transaction and
the same eligible complete-source revision list, then closes the canonical
native database before evaluating any question. It loads the exact native
original **passage chunk** embedding vectors (the installed full original
reader's additional *analysis facet representations* are **not** included),
the independent current original-linked summary/super-summary embeddings and
an ephemeral RAM-only SQLite FTS5 index of verified current summary-root text
and publisher titles. It uses the preexisting local CPU-only multilingual E5
model once, limits CPU inference to two threads and does not create a second
persistent semantic database. Every native raw and summary vector must match
the pinned model revision, dimensions, byte length, embedding SHA-256 and
source-map original/content revisions; generated summary text hashes and
source-linked JSON metadata must also be valid. Sources without current
full-root summary coverage are excluded **from all three channels**, ensuring
an equal candidate population for this diagnostic rather than silently
changing the eligible corpus by search method. Search outputs are original
source identities, not asserted factual answers or confirmed physical PDF
pages.

The benchmark and policies were frozen before re-reading its labels: the
first forty-six questions contain fifty-two predeclared relevant original
labels; the distinct independent twenty-seven-question set contains twenty-
seven. The source original and extracted derivative bytes of **all sixty-nine
distinct expected originals** were independently checked against SHA-256
completion receipts and current source-map revisions. In a separate subsequent
low-priority audit, the **seven hundred thirty-one distinct documents**
appearing within the top five positions across all five evaluated source-
ranking views, plus the known-relevant originals, were independently
physically rehashed as complete publisher original and current extracted
source bytes: **1,948,337,652 original bytes** and **26,252,359 extracted
bytes**, with **zero SHA mismatches**. This does not establish factual
relevance for all retrieved candidates, physical PDF page citations,
extraction correctness, or exhaustive independent source relevance gold.

The lexical query code intentionally reproduces the *exact* earlier private
benchmark's tokenization, including an observed error in its long stop-word
literal (the original set accidentally contained one unsplit multiword string).
Fixing that after seeing evaluation results would not be the same frozen
method. The unchanged lexical channel uses OR match, Porter stemming and
FTS5 BM25 weighted **five for titles and one for current summary roots**.
Two-channel and three-channel reciprocal-rank fusion are reported as
**experimental** equal-vote ranking with fixed `k=60`, without outcome-based
parameter fitting. Union coverage is separately shown: candidate union is
**not** the same as a single top-ten ranked answer list. The original and
summary native vector matrices used the same captured source revision cohort
for every one of the seventy-three questions; the metadata source-map and
native SQLite stores are separate read snapshots, **not an atomic
cross-store transaction**. Observed native vector read time was about five
seconds and reusing that in-memory snapshot for all questions took around ten
more seconds. No native original, source-map, persistent keyword index,
summary, model, systemd unit or deployed ranking configuration was changed.

**Verified current-source candidate results, known source-label recall only:**

- Earlier forty-six questions, fifty-two known relevant original labels, at
  ten **per individual channel**: original chunks **34**, source summary **37**,
  title/root lexical **32**. Union of original+summary **42**; union of all
  three **44**. Fixed two-channel RRF's actual top ten **40**, three-channel
  RRF's actual top ten **41**. The larger three-channel candidate set found
  two further known originals, but a single ranked top-ten answer list did
  **not** gain those two as top-ten results.
- Separate predeclared twenty-seven questions: original chunks **24**,
  source summary **24**, lexical **23**, original+summary candidate union
  **24**, all-three candidate union **26**. Fixed two-channel RRF and
  three-channel RRF both found **24** known labels in their actual top ten.
  The lexical channel by itself performs worse here, and only the **candidate
  set** improves. Previous November 7 vector/hybrid scores are not the
  same-date baseline and must not be reused to claim a causal improvement.
- At top one on the earlier forty-six questions the three-channel fixed RRF
  found **23** known original labels versus **16** for two-channel RRF; on
  the twenty-seven questions the corresponding values were **16** versus
  **15**. These are **source identity** comparisons, not graded answer quality
  or semantic truth. A more demanding independent hard-negative/authority-
  context validation is mandatory before changing any production ranking.

The source-only evaluator lives under the **canonical current master**;
source/publication metadata, private gold and observed outputs remain in the
protected Raspi `search-audit/continued-20261122-wave58/` directory, not
inside Git. The first private trial had a harmless timing-report variable
shadowing error; its ranking metrics were verified unchanged by a second
independent corrected run. The *corrected* receipt is
`frozen-73-same-snapshot-three-channel-native-label-recall-timing-corrected.json`,
with the full seven-hundred-thirty-one-original physical hash proof
`actual-original-and-current-extract-full-sha-top-five-all-three-and-rrf-candidates.json`.
The earlier incorrect timing receipt is preserved as negative evidence and
must **not** be cited for elapsed time. The current benchmark does **not**
prove a production ranker improvement; the actionable next step is an opt-in
separate-channel candidate discovery interface with source-level provenance
and a truly new, independent, adversarial context/negation/false-source test
before any default search reranking is enabled. The two resident recursive
summarizers and all existing project dependencies must remain protected.

### Opt-in original-verifying three-channel source navigation (November 22)

The validated earlier comparison supports **separate candidate discovery**,
not replacing the default original-passage/recursive-summary ranker. The new
`scripts/raspi_three_channel_source_discovery.py` implements an opt-in read-only, **source-navigation-only** CLI
using the same candidate population, original-chunk vectors, native recursive
summary vectors and ephemeral current-title/root-summary FTS5 keyword channel
as the fixed source-label evaluation above. The command's result returns
three **separately ranked** lists, never silently merges them into a single
supposedly authoritative answer. It does not change the installed primary
dual source query, ordinary folder/grep search, FileDB, service/timer,
extracted text, publisher originals, shared model service or current source
revision. Local E5 inference uses the pinned CPU-only ONNX model in the
existing `semantic-venv`, with two threads and no cloud API calls.

Example, with **no automatic report publication** and only a bounded maximum
of eight source candidates per channel:

```bash
cd /data/src/github/KnowMoreDiRT
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
 PYTHONPATH=/data/src/github/KnowMoreDiRT/src:/data/infra/libexec \
 /data/var/knowmoredirt/semantic-venv/bin/python \
 scripts/raspi_three_channel_source_discovery.py \
 'Research on evaluating evidence-dense question answering using multi-level text division and automatic merging of retrieved passages.' \
 --top 8
```

Source search results include the exact current original catalog path,
original title, official/source URL metadata, content kind, publisher
rights/context where recorded, full publisher-original SHA and current
extracted-derivative SHA. **Before returning any candidate**, the CLI
independently streams and verifies its complete original file bytes and
complete extracted text bytes, and rechecks the current source-map and
completed acquisition record against the earlier snapshot. A missing,
revised, symlinked or ambiguous original makes the **entire search fail
closed** instead of emitting an unrelated substitute source. Source text is
not reproduced; a selected publication must still be inspected via the
existing original-source-evidence tools for exact section, literal passage,
conditional/quoted context and physical PDF page correctness. Returning
an original's filename, title, similar abstract or nearest summary is **not
factual evidence** and does not establish publisher page numbers, answer
truth, or suitability for external redistribution/training.

An observed real AI research question returned its expected existing original
PDF in both the summary and lexical source channels (at respectively native
ranks eight and seven among ten), although its original-passage chunk channel
missed it in its top ten. Twenty-nine unique returned original files and
corresponding extracted source revisions passed full SHA verification. A
**separate** Unity GPU renderer question returned the official current GPU
Resident Drawer reference **first in the summary channel** and correctly
separate top-ranked candidate lists from the other channels, with nineteen
unique original source SHA proofs. The original two-channel production CLI
remains installed; normal Knowledge Library catalog browsing and grep are
unaffected. Strict query and per-channel result bounds, malformed source-map
revisions, modified original/derivative content, symlinks, read-only DB
admission and fail-closed source substitution are covered by focused tests.

This supplemental discovery uses an in-memory native snapshot and scans the
model's source-linked content vectors, so it is appropriate for bounded,
selective queries rather than replacing a low-latency default search daemon.
The most important remaining gap is **independent contextual correctness**:
future original-source QA must still challenge quoted falsehoods, negation,
dream versus real-world claims, hypothetical statements, conflicting revisions
and missing evidence, and verify that an assistant says a fact is unknown
from the corpus rather than taking a nearest-neighbor hit as true. The new
CLI makes no automatic factual claims, no external downloads or account
operations, and is not a reason to disturb the two existing live recursive-
summary workers or prematurely release the staged 327 PDF page corrections.

**Source-map commit-race guard and selected physical-page evidence.** The
optional source navigation now checks a **fresh** read-only source-map
snapshot **after** streaming all selected original and derivative SHA hashes.
Because the extractor's source-map database is WAL-mode, an extractor may
commit a new derivative while an earlier read transaction still sees the
old source revision. An actual separate SQLite-connection regression test
changes the source-map derivative after a previously correct original SHA
proof and verifies that the old result is **rejected**, not silently returned
as current. The real opt-in Unity search was repeated after this guard:
all nineteen returned publisher originals passed SHA and the final current
source-map revision fence. This remains a best-effort non-atomic live read;
only an immutable cross-store source-generation protocol could offer a
stronger global snapshot guarantee.

A separate private source-evidence audit inspected actual **original PDF
physical pages**, not merely a derivative file's page-like labels. The
existing Poppler-based `verify_pdf_page_excerpts` reproduced exact normalized
first-page excerpt bytes and verified the actual **physical page one** for
three real retrieved original documents: the HiChunk retrieval paper, the
multi-talker ASR/diarization paper and a loading-dock construction drawing.
Their publisher original SHA and current extracted text SHA were checked
independently. The fourth machine-recognized cognitive-walkthrough PDF
retains an **OCR-only** page label and was correctly reported
**`unverified: no_text_page_label`**, even though its OCR text contains a
plausible matching phrase. No broad PDF-citation completeness or factual
answer-truth claim follows from these three verified page snippets. All exact
source and negative-control receipts remain in the private
`search-audit/continued-20261122-wave58/` evidence folder; no protected
publisher file, already-indexed vector or service was modified.

### Publisher-original physical PDF neighboring context, without quotation-to-fact promotion (November 22 late)

A SHA-valid short retrieval excerpt can still omit a governing qualification
located on the **previous or next physical PDF page**. Existing source-evidence
keeps its default read-only context strictly within each separately labeled
source unit, appropriately preventing accidental mixing but offering no
adjacent original physical page expansion. Two **optional read-only** operators
now supply this missing navigation stage, never a generated factual answer:

- `scripts/raspi_original_pdf_page_neighbors.py` accepts one **exact** approved
  original PDF path, a **physical PDF page number**, zero or one adjacent page
  on either side, and an optional exact center-page phrase. It independently
  rehashes the complete current original and derived text against completed
  acquisition receipts and source-map, pins the original PDF **open file
  descriptor** while running bounded Poppler `pdfinfo` and `pdftotext`, requires
  the real physical page number to exist, and rechecks file identity and
  current source-map revision afterward. Each returned physical page has its
  own complete extracted text, source coordinates and raw page-text SHA. Up
  to **three separately labeled** original physical pages are returned. There
  is no cross-page synthetic quote, automatic OCR, hidden excerpt truncation,
  external network access, model service change or native FileDB write.
- `scripts/raspi_source_evidence_pdf_neighbors.py` accepts the exact approved
  PDF original path and a source-specific question. It reuses the **already
  installed** KMD original `source-evidence` reader (which verifies the exact
  source revision and performs independent physical PDF page proof). Only the
  first **physically verified** result from that same original may become the
  anchor for the optional full-page context. The new operator then reproduces
  that passage on the physical PDF page, verifies the **complete page text
  SHA matches the prior independent original-reader proof**, and displays
  the immediate original PDF pages separately. An ambiguous, OCR-only,
  absent, changed or uncitable passage **aborts instead of substituting a
  different source**. Native original lexical evidence continues to write
  its existing private query audit, but publisher files and native vectors
  remain unchanged.

Both operators have a hard three-page limit, independent page-body length
limits (no silently clipped negations or qualifiers), a per-invocation
Poppler execution deadline, a restricted child memory/output budget, and
strict no-symlink publisher file and source-map checks. Pages with missing
text layers require separate explicitly authorized original-page visual/OCR
inspection; they cannot silently acquire text citations from the cached
OCR derivative. `pdftotext` is a **physical-page-bound text extraction**, not
visual proof that all publisher glyphs, figures, formulas or reading order are
correct. A fully verified physical page is **not** evidence that its content
is a factual assertion: it may be an example, reported claim, quotation,
negation, hypothetical, test question or obsolete version.

Example, with **no external effects**, from an already approved first-party
ACL paper and its self-contained experimental question, including the paper's
neighboring pages:

```bash
cd /data/src/github/KnowMoreDiRT
PYTHONPATH=/data/src/github/KnowMoreDiRT/src:/data/infra/libexec \
 PYTHONDONTWRITEBYTECODE=1 \
 /data/var/knowmoredirt/semantic-venv/bin/python \
 scripts/raspi_source_evidence_pdf_neighbors.py \
 'Scott Derrickson Born' \
 --original-path 'ai-agents/2026-acl-source-retrieval/2026.acl-long.1372.pdf' \
 --neighbor-radius 1
```

The actual owner-private test performed the native exact-original
source-evidence lookup, verified the matched sample question on **real PDF
physical page two** of the original sixteen-page hierarchical-retrieval
paper, then independently extracted **pages one, two and three** of that
same SHA-checked original and matched the two independent physical page-text
hashes exactly. The sample is a **benchmark question inside a retrieval
research paper**, not first-party proof of any real-world person's date of
birth. A separate real multi-talker ASR/diarization scientific paper passed
source-evidence and physical original-page hash checks for its first two
pages. Negative controls confirmed the machine-OCR-only HCI template yields
**no inferred PDF citation**, and a former extracted-page label requesting
page thirty of a peer-reviewed PDF with only nineteen actual publisher pages
is **rejected** rather than silently treated as a physical page. An absent
assertion on the original physical page is also rejected. Entire originals,
current extracted text, native embeddings and resident workers were preserved.

Reproduction evidence, including complete source-original physical page
texts and SHA checksums, remains under private Raspi
`search-audit/continued-20261122-wave59/`, **not Git**. New unit tests cover
literal source/derivative SHA mismatch, source-map change after extraction,
zero or excessive page requests, page-count boundaries, Poppler unavailable,
OCR-only/blank page, unbounded page length, publisher file replacement during
reading, independent source-evidence SHA mismatch, a wrong original identity,
and refused citation substitution. The preexisting native two/three-channel
ranking and corpus keyword/grep tools are unchanged. A separate independent
answer-grounding evaluation is still required: physical page localization
and honest abstention **do not** automatically judge scope, meaning or truth.

**Verified old-extraction page relabeling, not just overcount.** A separate
fresh original-byte-attested read-only audit of **three** representatives
from the existing three-hundred-twenty-seven-document page-repair backlog
proved mismatches *inside* the physical page range, not only at oversized
labels. One current USENIX OSDI original contains text labeled cached
`PDF page 19` that Poppler independently locates on original physical
**page 13**; its old `PDF page 34` is actually original physical **page
19**. A robotics publication's old page-six excerpt is physically on
page five, and a security publication's former page-forty-nine excerpt is
physically on the original last page, **page 21**. The physical original and
current extracted derivative SHA were checked before and after the review;
these examples are a bounded audit, **not** an assertion that every page
or all 327 original documents has already been repaired.

The optional `source-evidence` → physical page neighbor handoff now surfaces
`original_derived_page_label_disagrees_with_physical_page`, the original
page label and the **correct original physical PDF page** as distinct
fields. It does not quietly cite a wrong cached page. In an **actual live
source-specific KMD lookup** for terms from that security research paper,
the existing source-evidence verifier returned original-page proof with
`unique_across_pdf` for cached page **49** → physical page **21**; the new
operator then reverified that exact complete original/derivative revision,
independently re-extracted original physical pages **20 and 21**, and
reproduced the independently verified source page-text SHA. No citation
was promoted from the cached label or changed in the native FileDB.

This makes the *read-only* per-source correction immediately usable for
selected evidence, while the broader 327-PDF source extraction repair still
**cannot be cut over** until both installed native/source-map writers and
readers are quiesced at proven boundaries, all affected vectors and
summaries are actually regenerated at the new source revision, and an
atomic, rollback-safe generation switch is verified end-to-end. Preserve
old originals, current writer processes and all already indexed sources
until that separate release admits them. Owner-private exact provenance and
positive/negative evidence are saved in
`search-audit/continued-20261122-wave59/`.

### Original-revision stability after staged PDF page proof, November 23

The optional, **private** temporary publisher-page reader for the 327 staged
original PDF corrections now revalidates source-map rows, canonical native
FileDB original identity, complete publisher original bytes, old live extracted
text, and privately staged replacement SHA **after** Poppler's independent
original physical-page verification and **before** returning any verified
source excerpt. This closes an earlier read-only time-of-check/time-of-use
window: a concurrent writer could previously change the old derivative,
private staged candidate or source-map record between the first SHA check and
completion of the physical-page match, and a stale indexed excerpt could
still receive a successful original-page proof.

Every verified result now includes
`source_map_native_id_and_three_file_sha_rechecked_after_original_pdf_page_proof`;
the page proof's `page_text_sha256` must also be exactly one lowercase
SHA-256 digest. The second check **raises** if any original, old derivative,
private corrected candidate, source-map identity or native original FileDB
identity changed. Negative tests actively modify each of those dependencies
*inside* the mocked publisher page proof and confirm no evidence is
returned. The first source-reader precondition, exact approved acquisition
ledger and last physical original page proof remain mandatory. The operator
still returns **physical source citations, never established factual
answers**; no model, live source-map/native index, original publisher PDF,
summary worker or vector is changed by this improvement.

Verified reproduction on the **actual approved** USENIX security publisher
original, with the live source-map and the privately staged corrected PDF:
`raspi_staged_pdf_original_page_search.py 'SGX bouncer memory barriers'
--stage-preflight <existing private full-327 receipt>
--original-path <the exact approved original PDF path> --top 3
--candidates 20`. The source-bound private RAM-only search saw twenty
physical-page units and verified actual publisher pages **twenty-one** and
**eighteen**, with correct original physical-page labels, all three current
full-source digests and the final native/source-map identity rechecked;
seventeen weaker competing page candidates were excluded. Exact output,
source revisions, hashes and command proof are saved only in the owner-private
`search-audit/continued-20261123-wave60/` folder. The prior old cached
page-forty-nine to physical-page-twenty-one discrepancy remains documented;
this independent staged retrieval does not mean any 327-document live
reindexing or cutover was done.

### Opt-in original physical PDF page neighbors for all 327 corrected staged sources, November 23

The 327 publisher-original PDF corrections have a verified private staged
source candidate and a RAM-only publisher-page search, but the current native
FileDB and its generated summaries must remain on the old revision while the
active workers cannot safely quiesce. **Source-bound search may now optionally
expand its verified staged page into adjacent real publisher physical PDF
pages** without cutting over those sources. Set the new
`--original-neighbor-radius 1` parameter on the existing
`scripts/raspi_staged_pdf_original_page_search.py` command, or zero to request
only the verified original page. Omitting this parameter retains all former
output and ranking behavior. The search first verifies the **same source**
against its acquisition record, current old derivative revision, staged
corrected derivative revision, native FileDB identity and independently
matched original PDF physical page. The opt-in stage then extracts at most
one original physical page either side, **without joining them into a
fabricated quotation**, and requires that the center-page text SHA match the
previous independent publisher-page proof. It verifies the three full file
SHA digests, approved acquisition receipt and live native/source-map identity
**again after** extracting the adjacent pages. Unverified, stale,
wrong-original, missing page, overlong, corrupted or image-only page text
cannot acquire a source-context citation.

The original full-text and all surrounding pages remain restricted to the
owner's machine. An actual publisher PDF page is not automatically a factual
claim: quoted statements, examples, negative evidence, old versions or
benchmark questions must still be assessed in context. This operator
neither edits native vectors nor restarts writer services. Actual original
publisher PDF evidence passed on three independent previously approved
sources with extraction-label problems: the USENIX security paper located
correct **pages twenty and twenty-one** for an actual security research
query, the RSS robot motion paper returned **pages twelve and thirteen**,
and the USENIX OSDI implementation paper returned **pages eighteen and
nineteen**. All hits reverified the original, old extracted text, corrected
private staged text, source-map and native FileDB identity *after* the
physical-page expansion; the first hit's original physical-page citation,
source excerpt, revision digests and rank stayed unchanged compared with
search without the optional neighbor expansion.

New read-only negative tests force a concurrent original, current live
extracted source, private staged candidate, source-map row or native FileDB
identity mutation **during neighbor extraction**, and require failure.
Additional tests reject incorrect original document/excerpt, cross-PDF
page SHA disagreement, missing/oversized and re-ordered page sets, invalid
neighbor radius and fabricated factual-answer status. Exact reproduction
receipts, real publisher physical-page bytes, hash proofs and current corpus
coverage are preserved owner-privately in the continuation wave-sixty audit
folder; never place publisher page bodies or private transcripts in Git.

### Physical-page text completeness and alternate extractor evidence, November 23

The 327 staged PDF repairs now have real **physical** page labels, but their
publisher text layers are not uniformly readable. A new **private all-327,
full-SHA-attested source-level audit** rechecked completed owner acquisitions,
original source hashes, live old extraction hashes, private staged extraction
hashes, original native FileDB identities and live source-map revisions.
Across **7,580 physical publisher pages**, all 7,580 source-page markers
appear in staged text with no missing labels. However, the current safe
temporary publisher-page index skips **423 pages across 170 PDFs** because
their text bodies have excessive NUL glyph codes; only **7,157 physical
pages** are temporarily searchable by that text-only RAM index. The separately
bounded original-page context reader flags **860 pages containing at least
12 NUL codes** as unsuitable for reliable text-only quotation. This
wider diagnostic includes additional glyph-contaminated pages which the
temporary index's original stricter length-proportional filter still admits:
page-label correctness and text quality are **independent requirements**.
Do not infer that the other 6,720 pages are visually or semantically perfect.

An already installed independent local Ghostscript `txtwrite` extractor was
compared with Poppler **on a separately frozen, predeclared, held-out sample
of 16 previously unexamined problematic original physical pages**, covering
16 originals across eight source categories. The same open original file
handle was SHA-checked before and after both bounded local extraction calls;
no external source was fetched and no native index, worker or private staged
candidate was changed. All sixteen Ghostscript outputs were decodable UTF-8.
**Fifteen of sixteen** no longer contained NUL glyph codes; one had three.
The original Poppler baseline of all sixteen had at least twelve. The
median multiset case-insensitive word-token overlap of Ghostscript relative
to Poppler was about **ninety-one percent**: this is only a lexical
observation and includes different tokenization/order, **not** a verified
accuracy improvement, visual-layout reproduction, or permission to replace
live cached text. At least one sampled page contains dense graphical figures
and two-column text; comparing a rasterized original page, rather than only
word counts, is essential before treating either extractor's output as
complete scholarly evidence. All source bytes, sample precommit hashes,
individual page-level counts, validity/timeout receipts and the private
publisher raster are retained under the protected owner-private wave-sixty
audit folder. Do not publish raw book pages or frozen private study results.

The source-reader's opt-in *adjacent page* view now handles actual PDF glyph
and long-layout failures without silently truncating or inventing text.
Horizontal spacing-only runs can be compacted for display **only when all
original non-whitespace codepoints and all line boundaries remain intact**;
raw original physical-page SHA and original character length are retained
and column alignment is expressly marked as altered. An adjacent physical
page with missing/invalid glyph text is retained as an explicit
`adjacent_original_pdf_page_text_unavailable` entry with its real physical
page number, optional Poppler-page byte SHA, and `extracted_page_text: null`.
A corrupt **center page** is never promoted to a verified quotation. For
private collection-wide search, that single unusable center candidate is
excluded with an explicit count, rather than aborting other independently
verified candidate pages. Original/old/private-staged SHA and live identity
remain revalidated after the neighbors are read.

The real all-327 private RAM-only search on the USENIX OSDI paper then
returned five independently publisher-PDF-page-verified passages after
one unusable center-page candidate was excluded. Two adjacent physical
pages were explicitly marked as visually requiring inspection; **no
fabricated page text** replaced their damaged text layers. Results and
ranked evidence remain source-bound, not asserted factual answers. This
functionality is a safe *read-only bridge while waiting for* the 327-PDF
transactional native reindex cutover, not a completed index repair.

### Opt-in provisional search for original PDF pages skipped by unreadable glyphs (November 23, wave sixty-one)

Source-page correctness is separate from the quality of text extracted from
that source page. A complete read-only source SHA, old live extraction SHA,
private corrected candidate SHA, original acquisition receipt and native
source-map/FileDB proof of all **327 privately staged original PDFs** found
**7,580 physical publisher page labels**. The previous safe temporary
physical-page lexical index admitted only **7,157 physical pages**, omitting
**423 glyph-corrupted pages across 170 original documents**. A larger group
of **860** physical pages contained twelve or more literal NUL glyphs, but
some of those met the separate temporary index's length-proportional
criteria; neither classification is a proof of visual accuracy.

A full source-byte-verified **read-only** alternative-extractor evaluation
has now inspected all **423** pages skipped by that temporary index. The
installed Ghostscript text extractor successfully returned UTF-8 page
text on **423 of 423**; **412** contained zero literal NUL codes. With a
less strict provisional quality candidate threshold allowing a small number
of NUL glyphs, **420 of 423** met the prior length and word-token rules.
**This is *not* 420 verified publisher-accurate pages.** Ghostscript often
recovers meaningful chart labels and text where Poppler's encoding is broken,
but text order, column interpretation, formulae, chart labels and sentence
scope remain to be independently checked against the rendered original PDF.
The entire source-by-source full-SHA diagnostic is private and contains
**only counts and page-text hashes**, not copyrighted paper bodies.

The new optional `scripts/raspi_staged_pdf_alternate_glyph_search.py` accepts
an exact completed, approved original relative PDF path, its existing
private 327-source stage receipt, a bounded natural-language query and
optionally one explicit physical PDF page. It operates *only* on the
original page units previously **omitted** by the standard private search,
not the current native index. No new files are downloaded; no OCR is invoked.
Before and after the temporary query, a fresh read-only source-map/native
FileDB snapshot checks the exact original ID and current extraction
revision, and complete hashes of the original publisher PDF, its old live
extracted text and privately staged corrected text must match the approved
records. A Ghostscript process is restricted to one selected publisher PDF
page at a time, with a hard time, virtual-memory and output-size limit,
using the same original file descriptor whose full SHA and true physical
page count are checked before and after. One query admits at most **32
previously omitted physical pages**. Larger documents require an explicit
page selection instead of silently dropping other damaged pages.

The production-quality **provisional reader is deliberately stricter** than
the exploratory 420-page test: it rejects any NUL or replacement-character
glyph, abnormally sparse text and text too long for untruncated output.
Only horizontal spacing may be compacted when all non-whitespace characters
and original line boundaries survive. The selected source's candidates are
ranked in **RAM-only** SQLite FTS5, with the same minimum two-distinct-word
excerpt coverage guard as existing staged physical-page search. A second
independent Poppler pass on the **same physical publisher page** may
corroborate the complete short excerpt. If it does not, the result is
explicitly named `provisional_alternate_machine_text_only_not_a_verified_quote`.
Even a corroborated source excerpt is **never** a proven factual answer or
guaranteed visual-layout reconstruction. Both statuses preserve original
publisher identity, original physical page, extracted-text digest and
review requirements.

Real private acceptance: the ACL source-attribution paper's original
physical page twenty-seven, previously excluded by corrupted character
codes, now yields a bounded original-PDF-linked *provisional* answer
candidate for an attribution/search question. The source-aware TTS agent
paper also yields an original-page-one candidate; among its eleven
previously omitted physical pages, ten meet the **strict, zero-NUL**
provisional extraction rules, with five ranked hits on a source-specific
speech query. The eleventh was **not** represented as recovered. None of
these examples was promoted as an independently verified PDF quotation or
factual answer, and no resident model, running summarizer, live source
cache, native vector, summary or source-map was changed. All original PDFs,
private diagnostic evidence, outputs, SHA receipts and unresolved reading-
order checks are under owner-private `search-audit/continued-20261123-wave61/`.

Until an actual PDF visual and phrase-reliability benchmark establishes
accuracy across hard cases, **do not automatically replace** the original
live extraction or certify these provisional pages as authoritative.
The 327-source rollback-safe, coordinated native reindex and summary/vector
revision cutover is still a separate outstanding step; a cleared source
page text-corruption warning cannot by itself authorize it.

### Independent exact-phrase corroboration instead of promoting a damaged page wholesale

For a provisionally readable page, a user query can still require stronger
evidence than a single machine's extraction. An optional secondary Poppler
pass first checks the returned complete short preview against the **same
real original physical PDF page**. When that long machine preview differs
because of graphics or two-column reading order, the new bounded fallback
tries at most eighteen **source-identical, query-relevant contiguous short
fragments** with multiple distinct query terms. The independent Poppler
physical-page verifier may confirm an **exact shorter fragment** and its
physical PDF page digest without confirming the surrounding Ghostscript-only
preview. Such a result has two clearly separated parts: a provisional
longer preview and a separately SHA-proven *exact* text fragment, with an
explicit warning that the rest of the page has **not** been verified.
The fallback accepts only an exact same-physical-page proof, not a similar
sentence on another page, not a relocated quote, not a vague shared keyword,
not a different PDF, and not a generic source match. It never asserts that
the manuscript's factual claims are true or that figure labels, scope and
reading order are correct. The full original PDF SHA and current
source-map/native original identity are rechecked after both extractions.

The actual approved ACL source-attribution research PDF's formerly omitted
physical page **twenty-seven** produced an independently *Poppler-and-
Ghostscript-confirmed exact fragment of two hundred and seventy-nine bytes*,
covering three distinct query terms, while the larger machine-extracted
preview remained **provisional**. The same test on the agent TTS source's
original physical page **one** returned a useful provisional text candidate
but **no query-relevant second-extractor-corroborated passage**; it was not
misreported as a verified original quotation. Actual original files and
full private readouts are retained outside Git. Synthetic hard-negative
tests verify that a phrase matched on a different physical page, a truncated
or malformed proof, an unrelated single generic query term, and the wrong
source identity cannot be elevated into a verified quotation. This is
opt-in evidence navigation only, not approval for native text/vector/summary
source revision replacement.

### Strictly private and unapproved physical-PDF glyph recovery preview assembly

A separate optional review-only tool, `scripts/raspi_stage_provisional_pdf_glyph_recovery.py`,
creates **one new isolated private preview directory** for one exact
publisher-approved original PDF. It accepts only an already verified full
327-source physical page stage receipt, current unmodified original/source
bytes and native source-map/FileDB identities. It identifies **only** page
units omitted from the safe temporary page search and obtains bounded,
source-SHA-attested alternative candidate text through the local guarded
Ghostscript reader. The strict preview assembly accepts only candidate
pages without any NUL/replacement-character glyphs; unavailable pages
retain their prior original Poppler text, with their omission disclosed.

The preview repeats exactly the existing staged publisher physical page
markers and preserves **every untouched page's text body byte-for-byte**
(in UTF-8 representation) and the document header. It replaces nothing
except specifically identified formerly excluded physical pages. Tests
refuse missing/duplicate/reordered labels, a forged new source-unit marker,
a tampered alternate text SHA, a changed unchanged page, unknown candidate
states, and made-up text on unavailable pages. The tool requires a new
restricted private directory, writes a candidate text file plus a separate
source-revision/page-level manifest, verifies both output hashes, and
explicitly declares `PRIVATE_UNAPPROVED_MACHINE_PDF_TEXT_PREVIEW_NOT_FOR_LIVE_CUTOVER`.
This metadata **cannot** be used as a native source-map generation or
release authorization; no live original, index, vector, summary or resident
model is changed.

A live source-real pilot with the original approved agent speech-synthesis
paper preserved **all nineteen physical page labels**. Its previous
corrected-physical-page text was searchable on **eight** pages because
**eleven** were omitted for damaged glyphs. The new private review-only
machine candidate recovered **ten** of those omitted pages under the
**zero-NUL** rule. Consequently an offline *unapproved preview* would be
searchable on **eighteen of nineteen** original physical pages; the last
unrecovered page remains visibly omitted. Both the original cached text
and the earlier private approved physical-page stage remain exactly
unchanged. Exact source SHA checks, output hashes and the candidate's
rejection of live publication are preserved under owner-private wave-sixty-
one audit files. A larger optional private cohort is only a research source
for later visual/semantic review; it **must never** silently replace native
KnowMoreDiRT originals, vectors or generated summaries.

### Source-bound, RAM-only recovered-page search across 170 approved PDF originals (November 23, wave sixty-two)

The existing 327 approved physical-page-corrected source derivatives and the
170 new **private, unapproved alternate-machine-text preview** files make a
bounded, reversible retrieval-quality comparison possible without touching
native KnowMoreDiRT. Across these 170 affected originals, the safe staged
temporary lexical index originally had **4,230** searchable physical page
sections of **4,653**. The strict glyph-rescue preview has **4,641**
provisional candidate sections: **411** recoverable and **12** still
unusable. Across all 327 original PDFs, that is **7,157** original-section
candidates before and **7,568** after. The increase is *provisional textual
coverage*, not verified PDF visual text correctness, restored native vectors,
semantic embedding recall or a released source revision.

The new **opt-in read-only** entry point
`scripts/raspi_private_recovered_pdf_corpus_search.py` takes one query, the
previously approved complete 327-source PDF stage receipt and the protected
170-source recovery-preview rollup; optional filters specify an exact
approved publisher-original relative path and bounded top/candidate counts.
No network, source download, OCR, external LLM or persistent index is used.
The input rollup is required to match the *same* approved 327-source receipt
SHA. It must contain **170 unique complete original PDF identities** and
exactly **411** provisional machine recovered page sections. It rehashes
every publisher original, current old live text derivative, initial
corrected physical-page stage, new private preview and native FileDB source
ID against the complete acquisition ledger and read-only source-map/native
databases. Source identity and every published result's PDF and private text
revisions are verified again after any page-level proof. Sources whose
single attempted machine extraction was rejected on all formerly damaged
pages (five approved originals in this cohort) stay in the inventory with
**zero recovered rows**, never fake indexed content.

Only the separately SHA-attested **411 newly recovered physical page bodies**
enter the short-lived in-memory SQLite FTS index. Existing original PDFs,
untouched pages and all normal lexical search remain outside this supplement
and unchanged. The user-facing output does **not** expose the original
Ghostscript-only text excerpt. A whole-corpus query selects an ephemeral
page candidate, then searches its *full internal provisional page* for a
source-relevant, verbatim short passage. That short passage is returned
**only if** an independently re-read Poppler extraction confirms its exact
text on the **same real physical original PDF page** with a source SHA and
page digest; otherwise the result is only the source and real page locator,
explicitly labeled **unverified**. Never infer complete-page accuracy,
readability of equations and figures or the truth of the source's claim.

Real owner-private end-to-end acceptance used a frozen attribution/source
retrieval query and the **entire 170-original cohort**. The query returned
an exact publisher-original physical page **27** from an ACL paper as its
highest-ranked candidate, and the final tool independently confirmed a
**279-byte** short text fragment on that page against both Ghostscript and
Poppler; the second-ranked page 35 from the same paper also corroborated a
short fragment. Both are original-source navigational evidence, not a model
answer. The strict **zero-recovered-page source** returned no results and
remained in the input source set. An unrelated hard-negative question also
returned no result. The tool did not return unconfirmed machine-only
quotes. These tests used the existing original source checks, not fabricated
page labels or new downloads.

The independently frozen six-query baseline comparison and its unmodified
source records remain owner-private. Its OSDI chart example was initially
misclassified as a missing page; post-hoc inspection established that it was
NUL-contaminated but already present in the prior temporary index. A
separate immutable erratum corrects the classification: only the ACL and
AgentSteerTTS examples were truly missing from the earlier *lexical* index,
and both became discoverable after provisional recovery. Existing security
and robotics control ranks were preserved and the hard negative stayed
negative. See owner-private `search-audit/continued-20261123-wave61/` for
full 170-source preview, all 411 physical pages and independent full-byte
checks, and `search-audit/continued-20261123-wave62/` for end-to-end
whole-corpus and regression results. Do not add copyrighted originals or
full private extracted-book texts to Git.

**Production boundary:** this new source-page navigation does *not* authorize
promoting a provisional preview into the live source-map, native FileDB,
source vectors, summary embeddings, summary-tree generations or hierarchical
indexing. A coordinated rollback-safe source-revision cutover and visual
original-source citation/reading-order acceptance remain outstanding. All
resident summary workers and shared service dependencies continue unchanged.

### Exact short-fragment provenance on glyph-damaged original PDF pages (November 23, wave sixty-two follow-up)

The independently retained owner-private source audit first checked **all
411** strict, provisionally recovered physical PDF pages against a *second*
original-PDF Poppler extraction on the identical SHA-attested original file
and physical page. **380** pages had at least one verbatim passage of sixteen
or more words, including **318** with at least twenty-eight matching words.
This is page-local machine agreement, not proof of original typography,
formulae, figures, complete sentence scope or a factual statement.

The other **31** pages were reviewed in a separate **predeclared shorter-window
original-PDF test**. Among those, **29** contained at least one further
identical eight-to-fourteen-word passage, at least sixty-five characters long
and with six or more distinct word tokens. The remaining **two** provisional
pages did not meet even that threshold and must not be promoted to confirmed
source text. Consequently, **409 of 411** recovered provisional physical
pages have at least one independently identical source-page passage at the
specified thresholds. **Twelve** original physical pages remain unrecovered
under the stricter no-NUL candidate rule. These two different deficits
must not be merged or reported as a false complete-index release.

The optional source proof selector now reserves a small **bounded part of
its single verifier batch** for 12-, 10- and 8-word candidate fragments,
while preserving the longer 36-, 28-, 20- and 14-word windows. Short
windows require **at least seventy characters**, **six distinct nonnumeric
word tokens**, and **two distinct matching terms in the user's current
question**; all candidates still require an independent *exact* Poppler
match with the **same SHA-verified original publisher physical page** before
any short quotation can be returned. Overlapping short windows may both be
probed because one extra column/layout-corrupted token can invalidate an
otherwise exact passage. The total independent candidate verification batch
remains at most eighteen by default, with a hard maximum of twenty-four;
there is no unbounded scan, remote model or automatic page promotion.

Additional negative tests refuse a matching phrase located on a **different
publisher physical page**, untrusted page-digest claims and generic
one-word match contexts. A real, source-SHA-checked test of the scholarly
market simulation original showed *two* previously excluded physical pages,
thirty-eight and forty-four, where **eight-word excerpts** were now verified
against the same original page by both local extractors. Their test queries
were selected *after inspection* from known shared-source words; this is a
**targeted extraction/reproducibility test, not a held-out retrieval-quality
improvement claim**. The original frozen global source-attribution query
still returned the same three independently verified exact excerpts after
the change. The owner-private reports and extracted passages remain outside
Git under the wave-sixty-two review directory. Never interpret any of these
machine-matched excerpts as confirmation of the full page, its diagrams,
claims or its semantic indexing completeness.

### Explicit, bounded streaming proof of large approved original PDFs (November 23, wave sixty-two)

The existing read-only `scripts/raspi_recent_summary_integrity.py` audits
native hierarchical source-summary graphs from independently completed
original publisher files. Previously, it intentionally returned
`unverified_oversized` if a source's original publisher PDF exceeded **64 MiB**,
even when its extracted source text was far smaller. This was an audit
safety limit, **not** proof that the original file or its generated native
hierarchy was corrupt. The default **64 MiB** original and **64 MiB** derived
limits remain intact for normal audit calls.

An additional **explicit opt-in** `--max-original-mib 128` (one through
**256 MiB** maximum) permits independent verification of a larger *original
file* only. The original PDF is SHA-256 hashed through a same-file-descriptor,
mutation-checked **streaming reader**; the content is **not** loaded wholesale
into RAM. The full derived source text is still capped at **64 MiB** and
inspected with the existing exact source-span, summary-node, current original
revision, native vector digest/dimension and complete hierarchical graph
checks. Invalid, Boolean, negative or above-256-MiB bounds are rejected.
The native database and source map remain query-only SQLite connections;
this command neither causes summary generation nor changes the workers.

A real original publisher graphics/renderer research PDF of **97,804,345**
bytes, previously reported merely as `unverified_oversized` in the
fifteen-source bounded recent-completion sample, now passes an independent
**128-MiB streaming original-file** proof. Its **71,225-byte** extracted
source text and **all three current native recursive summary nodes** were
fully SHA-linked to the exact publisher original and their native embeddings
and source spans were verified. All original source text was accounted for
by the descendant graph. The proof is private under the wave-sixty-two audit
directory. This validates provenance and structure **only**; it does not
establish that AI-generated summaries are factually accurate or that page
figures/diagrams were properly interpreted.

The special larger-source bound is an **optional QA setting**, never a global
relaxation of the ordinary worker, download, text-source memory, or native
indexer limits. The default normal source audit must continue to report
large originals as explicitly unverified unless the operator knowingly uses
this bounded streaming override and rechecks the exact bytes. Unresolved
large or derived-over-limit sources remain unverified, never automatically
declared valid.

### November 23 follow-up — original PDF figure-heavy pages and strict glyph-island retrieval

The previous visual/search review covered **423** physical publisher-PDF pages
that the earlier physical-page-corrected temporary parser could not safely
make fully searchable. An alternate original-PDF machine-text preview
provisionally supplied entire clean text for **411** of those pages. It
**correctly rejected the other 12**, under the strict no-NUL/no-replacement
glyph and minimum-word guards. The original PDFs **were not missing**:
independently SHA-verified renders of all twelve and the two further
under-corroborated pages showed nonblank content containing charts,
equations, tables and surrounding prose. This visual classification does
not validate any machine-transcribed numerical claim, chart axis or scientific
conclusion. Original folder/grep and PDF viewers remain available.

The twelve fully skipped *source-text pages* are now also partially
navigable through a **separate, opt-in, RAM-only clean-glyph island search**,
without weakening the default strict whole-page admission policy. The new
entry point is `scripts/raspi_private_strict_original_pdf_island_search.py`.
It requires the exact previous **327-original** approved physical-page stage
receipt and the protected complete **170-original** private preview rollup:

```bash
P=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261121-wave51/all-327-private-staged-original-PDF-preflight-complete.json
R=/data/var/knowmoredirt/knowledge-library/search-audit/continued-20261123-wave61/private-unapproved-170-original-PDF-glyph-review-cohort/private-170-original-PDF-provisional-machine-glyph-stage-rollup.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/src/github/KnowMoreDiRT/src \
 python3 /data/src/github/KnowMoreDiRT/scripts/raspi_private_strict_original_pdf_island_search.py \
 'image reconstruction matching template automated retrieval candidate' \
 --stage-preflight "$P" --private-170-rollup "$R" --top 3 --candidates 32 \
 --original-path 'ai-agents/2026-tool-use-evaluation/icml-2026-yang26bu.pdf'
```

The optional `--original-path` must exactly match an existing approved
original PDF in the **twelve-page** rejected cohort; omit it to search all
twelve pages. The caller receives no generated answer, only the **exact
original publisher relative PDF and physical page number**, and at most an
independently verified *short literal same-page source excerpt*. The complete
original PDF file, current old text derivative, private approved stage,
source-map and native FileDB original ID are verified again against the
approved ledger; PDFinfo physical page boundaries, owner-private preview
receipts and original file SHA are checked before/after extraction. Ghostscript
is run locally under the existing output, time and memory bounds. The
unavailable-page proof must still report its full text as **strictly
unavailable**. This helper does **not** replace it with an altered text body.

Every NUL, Unicode replacement and non-whitespace control character is
considered a **hard boundary** between provisional machine-text islands.
No sentence is fabricated by dropping a corrupted character and joining
its surrounding text. Only bounded clean segments enter a temporary SQLite
full-text index, limited to **twelve pages**, **512 islands**, **18,000 bytes
per candidate**, and **2 MiB of temporary candidate text**; matching requires
at least two distinct query terms. The returned *candidate* is **not a
quotation**. Before any short excerpt is emitted, the existing independent
Poppler original-physical-page verifier must confirm the literal same-page
passage on the same full-SHA-pinned original PDF. A nonmatching question may
return the source/page **without** a quotation; a fabricated negative query
returns no result. The client never prints provisional unconfirmed excerpts.
No OCR, external model, purchase, source download, source original mutation,
source-map/derivative/native index write, vector rebuilding, summary
replacement or worker interruption occurs.

The read-only reproducible investigation checked all **twelve** previously
strictly rejected physical pages and found at least one independently
identical **8- to 28-word** publisher-page passage on **each**. These are
*partial searchable islands*, **not** twelve complete, corrected pages.
A live targeted, original-SHA/native-source-bound query about template
matching returned a corroborated passage on physical page thirteen of its
source, while a different broad query left uncertain candidates clearly
unverified and an invented hard-negative query yielded no results. The
private end-to-end source evidence and exact SHA receipts are in
`search-audit/continued-20261123-wave63/`. These small source pages and
short excerpts are not proof of full visual/glyph layout, chart contents,
scientific truth, global search recall or an authorized native migration.

A later **coordinated, rollback-safe source revision cutover** for the 327
approved corrected-text candidates, rederived embeddings, and hierarchical
summary generations is still necessary. Retain the existing warnings on old
native PDF source citations until that cutover has been independently
validated with all active source consumers. Never write private recovered
book text, raw original PDFs, page screenshots or private owner recordings
into the Git repository.


### November 24 — source-body evidence must survive ranked-candidate caps

The two original PDF provisional-page discovery helpers and the general native
source-evidence reader must distinguish an **FTS candidate** from a qualified
source passage *before* applying the returned-result limit. A ranked one-term
hit, or a title-only hit for a long paper, is not original-publisher evidence.
The earlier fixed top-N prefilter could conceal real later physical PDF pages
or source sections. The temporary private recovered-page index is still capped
at 170 original PDFs, 423 replaced pages and 64 MiB of source text. Its
strict twelve-page glyph-island helper remains separately bounded by 512
islands; both apply query-term checks before returning the requested number
of candidates. They never authorize a PDF repair to enter live indexes.

The ordinary native **source-evidence** reader now streams ranked temporary
FTS candidates and checks actual source-body terms until it has enough
verified source chunks or has exhausted that one already bounded original.
Its per-document limits and exact original/derived SHA checks remain intact;
there is no unrestricted corpus scan or persistent search index. The public
candidate-list `BoundedPassageIndex.search()` still accepts at most two
thousand returned candidates; the separate streaming iterator is used by
source-evidence to avoid treating a first-hundred title-only ranking as
exhaustive. Closing early releases its SQLite cursor. A real source result
still requires exact source offsets, SHA and (for a publisher PDF physical
page citation) separate same-page original-PDF verification. Rank and exact
term overlap alone never establish a factual answer.

The red/green tests include a late page with two exact terms below many
high-ranked one-term or title-only hits, source-specific filters, early cursor
release, strict current source SHA, and hard-negative exclusion. Privately
replayed checks against 170 previously staged originals and 411 provisional
pages returned three source/page locators, only one with independent exact
same-page corroboration; no machine-only excerpt was promoted as verified.
A separate four-case frozen original-PDF physical-page holdout was found at
top five in all four cases; this is bounded page recall, not comprehensive
semantic or factual-answer evaluation. Historical corrected source-map,
native vectors and hierarchical summaries were not replaced; the previously
identified reader/writer/document-boundary cutover gates still apply.


### November 24 — final per-original source-reader revision fence after physical-page checks

The direct `raspi_hierarchical_library.py source-evidence` reader independently
verifies a publisher original, its acquisition receipt and current extracted
source-map text before source search. With concurrent source-map extraction and
native writer timers, a document revision can change *during* physical-page
confirmation. The reader now performs a **second read-only full-byte revision
fence immediately before returning evidence**: fresh source catalog and
completed receipt identity, original publisher SHA and byte size, derived
source full SHA using bounded streaming, current source-map row and native
FileDB original file ID, then a fresh second metadata check after streaming
both files. The guard rejects changed original bytes, a newer derivative,
missing native file, changed title/edition or source-map revisions observed
mid-search, and writes no original/vector/summary data. It records the
`final_current_source_revision_check` in both the returned result and the
owner-private query receipt. The full original text and search excerpts remain
private; only verified source coordinates are exposed.

This is an incremental **stale-citation refusal for one installed reader**,
not a multistore atomic generation or a release mechanism. Independent SQLite
transactions cannot prove that all other clients are simultaneously pinned,
so the field `cross_store_immutable_generation_pinned` remains **false** and
`safe_to_cutover_live` remains **false**. The 327-PDF source repair and two
HTML original-extraction upgrades must not replace current native rows or
summary/vector generations until all resident workers, scheduled direct
embedding/extractor writers and legacy readers have a coordinated revision
admission mechanism. Tests inject source-map changes both after first read
and during the derived-hash check, original file byte substitutions, metadata
edits, and a missing native ID; all fail closed. Positive acceptance uses a
real current publisher PDF with independently verified physical pages.


### November 24 — summary discovery post-ranking source and native representation fence

The normal `raspi_hierarchical_library.py search` operation ranks source-linked
recursive-summary nodes and verifies the original publisher bytes and current
source derivative *while* checking native ranked candidates. A concurrent
extractor or native summary writer can commit a different source-map revision,
original file or summary representation **after** that early verification.
Those previously valid in-memory candidate results must not become stale
source references. A final read-only check now takes place after ranking and
any explicit publisher selection, before results are returned or private
search-audit records are written. Every returned original is tied to its
completed acquisition receipt, current catalog record, source-map original
and derivative SHA, native FileDB original ID, and each exact ranked native
summary chunk/representation, including source-span and summary-text hashes
and source-revision generation metadata. Original and derivative full file
SHA-256 are then streamed a second time through stable no-follow readers,
and both the source map and the relevant native summary representations are
queried afresh *after* those reads. On any observed mismatch the entire query
fails closed rather than showing a partially stale or substituted document.

The returned result and private search audit include
`final_current_summary_source_revision_check` with the count of returned
originals rechecked, total original-plus-derivative bytes hashed, and explicit
`cross_store_immutable_generation_pinned=false` and
`safe_to_cutover_live=false`. No second database or source cache is created.
Physical publisher SHA checks across all returned documents are limited to
**one GiB per query**; larger searches must be narrowed by `--top` or an
explicit source name. This restriction must not be weakened silently to
maintain apparent search results. Some very large archived textbooks may
therefore require a narrow lookup. The new checks add a **second publisher
and derivative read per returned source**, not a read for each summary node.
The verified ranked model results, summary texts and native FileDB are not
mutated, and this guard does not grant permission to replace any of the
historical 327 staged PDF or two HTML derivatives. Exact native vector
changes or source-map writes detected after ranking trigger a negative
answer, not automatic healing or guessed publisher source truth.

Private synthetic red/green tests reproduce a source-map revision committed
after native ranking, same-length original publisher byte substitution,
removal of a selected native summary representation, and two late changes
committed between full-file hash and final metadata recheck. The known
uncited and unindexed publisher-name cases still return explicitly empty
results without calling an LLM. This is a best-effort per-request stale
revision refusal; all legacy readers and writers must still adopt an atomic
source-generation admission scheme before any live multistore cutover.


### November 24 — final publisher revision check for combined evidence commands

The `find-source-evidence` and `find-pdf-evidence` commands combine the
native discovery rank with a separate original-publisher physical PDF page
or extracted HTML/Office/RFC evidence search. The original source and its
native FileDB ID are checked before passage/page verification, but a
concurrent source-map extractor or FileDB writer could change a selected
original **after** its PDF page or RFC publisher-byte proof. Both combined
commands now run a shared *final* read-only source revision guard before
returning any verified publisher passage. For each returned source with a
verified passage it checks the originally selected source SHA and derivative
revision, the completed acquisition receipt and current catalog, the latest
source-map record, the canonical native original FileDB ID and the original
native discovery chunk if one was used. It independently re-reads the
complete original and current derivative bytes through the installed
full-source SHA/identity verifier, and reopens current metadata **again**
after those reads. An observed change refuses the whole response, with a
bounded failure receipt in the private query audit; no unverifiable partial
passage or invented alternate original may be emitted.

The guard is limited to five selected originals and one GiB of complete
publisher-plus-derived text per query. A larger source selection fails closed;
use `--documents 1` or the exact `--source-name`, not an ungrounded model
answer. Records with no independently accepted publisher evidence remain
explicitly `no_verified_excerpt` or `source_unavailable`, and cannot acquire
a citation simply by passing a raw vector or lexical search. In mixed-source
search the existing private staged-PDF overlay remains an **optional,
separate, non-production** physical source channel; its own private page
revision proofs are not declared a live source generation.

`final_current_original_source_revision_check` now appears in both combined
reader results and the owner-private query audit. It explicitly reports
`cross_store_immutable_generation_pinned=false` and
`safe_to_cutover_live=false`. This is one more independent stale-citation
refusal path, not a transaction spanning multiple database files or a green
light to replace 327 historical PDFs, the two revised publisher HTML
extractors, native embeddings or recursive summary trees. Synthetic negative
tests exercise same-size original substitution, newer source-map revision,
missing native discovery chunk, changes committed **after** the physical
source bytes were checked, malicious relative paths and total read-budget
exhaustion. A protected original publisher PDF and a source-linked official
RFC were then queried through the real mixed-evidence command, preserving
verified physical PDF page locations and exact original RFC byte intervals.


### November 24 — isolated dual-SQLite generation reader and durable private pointer

The existing `private_source_generation` proof already validates an isolated
original-PDF cohort containing one native FileDB SQLite database, its matching
source-map SQLite database, each corrected extracted-text file, each complete
recursive-summary tree, source revision checks, normalized embeddings and a
source-bound release-gate receipt. Both SQLite copies must be physically
complete and have no mutable WAL sidecars. The new *private-only* function
`open_pinned_private_read_view(base, allowed_root)` opens the two **same-
generation** SQLite snapshots as read-only connections, after selecting and
fully checking a single pointer. The yielded view exposes `.native`,
`.source_map`, `.generation`, and `.read_source_asset(kind, relative)` for
manifest-listed `corrected_derived_text` and `recursive_summary_tree` assets.
The source-asset reader is bounded (default thirty-two MiB, maximum sixty-four
MiB), checks permitted paths and rejects symlinks, and compares actual bytes
to their exact pinned manifest SHA. Closing the view hashes the entire pinned
generation again. Consumers must not expose their query results until the
`with` block has exited without error.

A separate two-original cohort acceptance also builds one full private
source-map and native FileDB containing **both corrected publisher PDFs**, their
current source SHA/derivative rows, two extracted-text files, and two complete
recursive summary trees. It passes the real private native vector reindex
gate, then opens them as one pinned same-generation reader. This verifies
multi-document cohort consistency, not only switching between single-PDF
fixtures. No live original, vector or summary was replaced.

Concurrent isolated publication can advance the shared pointer to generation
two while an existing reader continues querying the complete old generation
one pair: both the native and source-map rows, old original-derived text and
full summary tree remain matched, and new readers select generation two.
Mutating an old pinned source file is explicitly detected and refuses the
old reader on exit, even after another generation has been published. The
private publisher now explicitly calls `fsync` on each validated staged SQLite
file, corrected original-derived text, summary-tree JSON, private native QA
receipt, generation manifest and every containing directory **before** the
atomic pointer rename. It then syncs the pointer's parent directory before
the simulated post-switch crash callback. This avoids claiming a durable
pointer to unsynced assets. Tests inject failures into both staged file and
directory `fsync`: the previously selected generation remains active and the
new generation can be published only after a later successful retry. These
checks demonstrate correct syscall ordering and failure refusal in isolated
fixtures, not an actual hardware power-loss test or live production rollback.
Tests also cover crash before pointer rename,
replayed generation, unsafe production roots, attempted SQL writes, altered
source assets, missing manifest roles and oversized original text reads.

**Scope and release prohibition:** These functions work only inside isolated
private audit roots and are not used by any live Knowledge Library reader,
original-source extractor, scheduled native embedding writer or resident
summarization worker. Their positive tests are not permission to flip the
production source-map/FileDB files or to replace 327 corrected PDF and two
publisher HTML derivatives. Real migration still needs all production
readers and writers to honor one generation admission mechanism, healthy
source-boundary acknowledgments, a crash-safe staging/preflight process and
an independently verified rollback. The module continues to report
`safe_to_cutover_live=false`. Neither pointer nor assets are copied to
external services; originals stay grep-readable in their existing folders.


### November 24 — cooperative PRIVATE legacy reader and writer admission barrier

The isolated generation switch now has an additional **private-only and
cooperative** client-admission demonstration in
`file_system_catalog.private_source_admission`. An older-style private
reader or writer explicitly enters `admit_private_legacy_reader` or
`admit_private_legacy_writer` and retains a shared advisory lock for its
entire operation. The *private* rehearsal publisher calls
`publish_private_after_legacy_drain`: its nonblocking exclusive lock refuses
a switch while any cooperating old reader or writer is still active. Once
all participating old clients finish, it validates the exact staged
native/source-map and recursive-summary generation, durably seals new legacy
admissions, and invokes the already-tested private atomic pointer switch.
Older pinned generation-aware readers may continue using their immutable
source snapshots while newer readers select the next version. New
cooperating legacy sessions are refused after sealing, including after a
simulated crash *between the durable seal and the pointer rename*.

A concurrency flaw was reproduced in the original **file-lock-only**
private protocol: renaming the lock file and recreating that pathname let the
publisher acquire a different inode while an existing reader still held its
original shared lock. Both private reader/writer admission and publisher now
also lock the **same owner-controlled private generation-root directory**,
shared for old sessions and exclusive/nonblocking for private publication.
Directory locking is acquired before the file and pointer locks in a fixed
order, preventing that split-file-lock bypass for cooperating Linux processes.
Both same-process and independent-process regressions exercise the replaced
lock filename and confirm publication is refused until the old reader exits.
The additional directory lock does not protect unknown manual clients,
replaced root directories, other hosts or any production database instance.

The seal is owner-only, regular, bounded JSON pinned to the actual verified
source-generation manifest SHA and its ordinal; its existence is checked
while holding the admission lock. If the seal is deleted after a pointer has
been published, a new legacy session **still fails closed** rather than
reopening older source paths. A copied/tampered seal, symlinked or
world-readable lock, invalid staged PDF/vector cohort and replayed
publication ordinal are refused. An abandoned old seal temporary file is
left untouched rather than overwritten. These are verified in synthetic
private fixtures, including a separate operating-system process holding a
legacy reader open while another publisher is refused. No worker restart,
signal, native SQLite mutation or external source download is involved.

**Critical limitation:** advisory locking excludes **only cooperating old
clients** that explicitly use these private functions. An unregistered
manual SQL client, live native embedding timer, source extractor, resident
summarization worker, browser/UI reader or older code path can ignore the
lock. The existing bare `publish_private_only` is deliberately retained for
older private tests and has no client-admission proof. Consequently neither
the barrier, a durable tombstone, a positive test nor the existence of the
private pinned-generation read view demonstrates that **all actual KMD
production readers and writers** have migrated. The structured private
publication result explicitly records
`cooperating_legacy_reader_admissions_sealed=true`,
`unregistered_clients_excluded=false`,
`all_live_KMD_readers_admitted=false` and
`safe_to_cutover_live=false`. These controls are not installed into or
run against the production source map, native FileDB, original collection,
public retrieval service or shared resident models. The seven live release
blockers must stay open until an independently verified production-wide
client census, enrollment, admission deny/hold protocol, crash-safe
multi-store generation rollout, rollback and post-switch source QA exist.
This private rehearsal is not authorization to cut over 327 PDF or two HTML
sources or to discard historical source generations.


### November 24 — source-linked lookup inside one private pinned generation

A separate **private-only** `retrieve_private_source_locators(base,
allowed_root, original_path, query, top=3)` now demonstrates a real
provenance-bound document lookup without requiring a live model or changing
any native FileDB rows. It selects exactly one approved corrected PDF original
from the *current immutable private generation*, opens its native FileDB and
source-map database through `open_pinned_private_read_view`, validates the
original publisher SHA and current derived-text SHA, finds its unique native
file ID, and loads the exact manifest-listed derived source text plus its
complete revision-specific recursive summary tree. The native root summary
representation must match the same original source ID, derived source
revision and tree root ID. This is a **retrieval proof across the databases
and actual source text**, not just an index-integrity manifest check.

The lookup then performs temporary in-memory lexical passage discovery over
only that original's verified derived source text. Returned locators have the
exact source section, source character offsets, excerpt SHA and matched query
terms. The physical excerpt hashes are independently verified against the
pinned derived string before the entire generation is rehashed at the end of
the context. Deliberately **no publisher quotation or generated answer text**
is returned. A corrected provisional PDF machine-text locator has explicit
`original_publisher_pdf_physical_page_verified=false`,
`original_publisher_quote_verified=false`,
`source_factual_claims_verified=false` and `safe_to_cutover_live=false`.
Only a separately verified original publisher PDF page can support quoting.
The query is limited to one source and five locators; the existing private
asset reader caps each source at thirty-two MiB by default.

Isolated tests exercise two different original PDFs in one published private
generation, exact native/root/derivative identity, source-specific positive
and hard-negative terms, safe path rejection, summary-tree and native SQLite
mutation during a query, and pointer change to a new generation during the
old reader's search. The old result remains bound to its pinned revision,
while a subsequent query sees the new generation. The lookup also operates
successfully **after** all cooperating old private writer admissions are
sealed. None of this code is wired into production services or authorizes
source-map/native/summary replacement. A complete production service census,
reader-and-writer enrollment, safe writer holds and rollback are still
required before any corrected PDF or HTML production cutover.


### November 24 — frozen source-navigation recall drift across growing native corpus

The private `raspi_frozen_same_snapshot_three_channel_eval.py` captured **73
predeclared questions with 79 source-known relevant original document
labels** on November twenty-second and again November twenty-fourth. It
physically rehashed all **69 distinct known publisher originals and their
current derived source revisions** in both captures, using one consistent
native vector set per run. These are *two different read-only time snapshots*,
not a cross-SQLite atomic transaction. The count of originals with complete
source-linked native roots grew from **9,108 to 9,614**. At source-label
top-ten, raw original-passage hits changed from 58 to 57 out of 79, recursive
summary from 61 to 60, FTS titles/root summaries stayed 55 out of 79, and
the experimental three-channel reciprocal rank fusion held **65 out of 79**
with **zero net top-ten gains or losses**. These numbers measure finding the
correct *original documents*, not independent answer sufficiency, generated
answer accuracy or confirmed publisher PDF physical pages. Some individual
raw/summary ranks slipped as the indexed corpus expanded. No current native
ranking weights, embeddings or production search behavior were changed.

To avoid mistaking a newly extracted source revision for a ranking regression,
`file_system_catalog.frozen_source_retrieval_drift` and
`scripts/raspi_compare_frozen_source_recall.py` provide an additional
**read-only per-original** comparison. The comparator first checks the
frozen query SHA, gold source labels, encoder/ranking policies, publisher
original SHA and derivative SHA in both reports, and declines to compare a
source that changed revision. It produces top-one, top-five, top-ten,
top-twenty and top-thirty matched-source counts, gains and regressions
for each raw-vector, summary-vector, lexical FTS, two-channel and
three-channel ranking; private output records original paths and query
identifiers but not the query text. It explicitly refuses mismatched gold,
model revisions, duplicate or unsafe source paths, malformed ranks and
unattested original source bytes. The output must be a new private file under
the existing KMD search-audit tree and must not overwrite prior evidence.
This is an evaluator, **not** permission to tune a production retrieval
algorithm on frozen case answers. Future algorithm changes need a separately
held-out, adversarial evaluation, resource safety, and independently
verified original page/line evidence before publication.

Qualitative inspection of the persistent thirteen three-channel top-ten
misses identified important actual source-content gaps in *query vocabulary*:
first-party Blender Normal Map and Gradient Texture, Unity background asset
loading and sprites in text, and Adobe perspective correction pages **do
contain the relevant original explanations**. Their indirect task descriptions
use different vocabulary from the titles/body, so no known physical original
is absent; the current encoder/lexical ranking simply failed to put those
documents in the first ten. They are retained as reproducible frozen cases;
adding bespoke query-to-title shortcuts solely for those gold labels would
be benchmark leakage. A future reranker or document-alias/term expansion must
be tested against genuinely unseen sources and a factual sufficiency/negative
query set, not inferred from these positive source-label metrics.


### November 24 — independent original HTML article text anchors for source evidence

When the current `raspi_hierarchical_library.py source-evidence` or
`find-source-evidence` retrieves an excerpt from a SHA-approved original
`.html`/`.htm` document, the verified *derived* text was previously the only
passage-level proof. A correct whole-original SHA alone does **not** establish
that a transformed excerpt appears in the official HTML page. The new
`publisher_html_dom_evidence` module independently parses the bounded
original HTML publisher bytes (maximum eight MiB) using Python's built-in
HTML parser. It excludes script, style, template and other non-content
fields, and requires the complete whitespace-collapsed excerpt to occur
**uniquely in the original `article` or `main` element**. A match is recorded
as `publisher_html_dom_anchor` with its container name, SHA of the complete
approved original, SHA of the normalized matched excerpt and normalized
article text offsets. If there is no suitable article/main, matching text is
only present in scripts/navigation, or content appears ambiguously multiple
times, the source excerpt remains explicitly unverified against the original
DOM. The original and derived SHA checks and final source-map/native-ID
revision guard still run after passage selection; no library or database
row is rewritten.

**Crucial scope:** HTML DOM character references and whitespace have been
normalized. These are **not publisher byte coordinates, not browser visual
layout, not a PDF physical page and not proof that the passage answers the
user's question**. `literal_publisher_byte_interval_verified=false`,
`original_pdf_page_verified=false` and
`answer_factual_sufficiency_verified=false` remain explicit. The mixed-source
reader assigns the separate scope
`publisher_html_main_article_normalized_text` when such an anchor exists,
while retaining document-level extracted-text status for compatibility.
Publisher-original UTF-8 byte intervals for RFC/plain-text sources and
physically verified original PDF page citations remain different, stronger
evidence mechanisms. No JS execution, URL fetching or browser interaction is
used to parse the archived publisher HTML.

The first real regression cohort reused five previously frozen indirect
Blender, Unity and Adobe task descriptions. Four yielded exact derived-source
lexical passage candidates whose **complete normalized excerpts** also
matched their archived first-party publisher HTML articles or main content.
The Unity asynchronous-background-load question yielded **zero** lexical
source passages despite the original page containing relevant terminology.
This proves an independent *citation-provenance improvement* for four known
cases, not better whole-corpus document discovery, automatic query
paraphrasing, or answer correctness. A transient read-only SQLite contention
in one initial verification was not mistaken for a publisher-source failure.
Adversarial tests cover script-only fake quotes, two identical ambiguous
publisher copies, no main/article, short unspecific fragments, altered
publisher SHA and stale derived excerpt SHA. A broader independent source
answer-sufficiency benchmark remains required before anyone labels a
retrieval-generated answer "verified".


### November 24 — verified offline links from original HTML to already archived source pages

`raspi_archived_html_follow_links.py` now resolves a narrow but important
**source-navigation** gap without weakening the existing original-source
quotation rules: an explicitly selected official HTML overview can be a
*table of contents*, and the relevant explanation may be in the overview's
already archived child document. The new `archived_html_source_links` helper
parses only the **full SHA-verified original publisher `article` element** (or
`main` when no article exists), excluding scripts, navigation, sidebars and
footer. Each candidate link must resolve to a unique approved **HTTPS
same-origin URL** present in the existing curated source catalogue, with a
matching completed acquisition receipt. Links to external providers, script
URLs, ambiguous duplicate URLs and unrelated article references are rejected.
Relevant same-origin publisher references that are **not in the local
catalogue**, or have no approved complete acquisition receipt, are returned
in a separate bounded `unavailable_publisher_references_not_followed` section,
never as verified archived passages. The source's original HTML SHA and
article link text are proven, but the missing page's **contents are unknown**:
`publisher_destination_contents_unavailable=true` and
`not_a_source_excerpt_or_answer=true`. This identifies a real source gap
without silently downloading or inventing an answer. In the actual Unity
manual, the currently archived overview links to the first-party `Make a
texture or mesh compatible with asynchronous loading` page, which is not in
the current acquired catalogue. **No HTTP request, browsing, acquisition,
new index or source mutation takes place**. The current query terms are used
only to rank *possible references* in the original document. This does not
claim to discover the correct original across the whole library or prove any
candidate passage answers the question.

For at most five selected archive links, the read-only command opens each
**already downloaded publisher original**, full-SHA verifies its original
and current derived text, checks the matching unique native KnowMoreDiRT
original file ID, and invokes the existing original-source evidence search.
The candidate passage's current extracted-text offsets and SHA are retained;
for HTML, the existing independent full normalized `article`/`main` source
anchor is included when available. After searching, each original's
publisher bytes, source map revision and native identity are checked again.
A final bounded *collective* review checks all returned destinations after
any later destination search, and then the original linking page again.
An observed revision or identity change rejects the **entire** lookup rather
than emitting stale references. This is repeatable-source validation, **not
an atomic multi-database generation** or permission to update live source
maps, embeddings, or the 327 corrected PDF and two HTML source migrations.
The result explicitly reports `safe_to_cutover_live=false` and
`not_proof_of_correct_answer_or_atomic_cross_store_generation=true`.

Real original-source verification on November twenty-fourth demonstrated
`Unity 6.3 LTS — Loading textures in the background` linked in its approved
original article to `Texture and mesh loading`: the overview's indirect
frozen question had previously yielded **zero** matching source passage,
but following the publisher's archived reference yielded **one** original
HTML article-anchored source candidate for that same question. A second
separate first-party Unity graphics-profiling overview linked to three
already archived originals on checking the mesh upload pipeline, configuring
asynchronous texture/mesh uploads and uploading the data to the GPU, each
with an independently original-article-anchored candidate. This is a
material improvement in **explicit selected-document navigation** and
source provenance, not in global semantic ranking. Fourteen isolated tests
exercise same-origin URL and catalog/receipt identity, external navigation
refusal, no-candidate behavior, symlink/path and bounded request refusal,
parent/source-map/native revision changes, target revisions that change
only after another destination was processed and preservation of original
extracted-source versus independently verified HTML article scope.

Production direct folder/grep searching and existing exact-source
`source-evidence` remain available and unchanged. The CLI writes only a new
owner-private evidence file under the established KMD search-audit directory.
The owner-visible search UI is **not** automatically wired to this optional
reference-following command; this is an independently tested library tool.


### November 24 — standard source-evidence command can optionally follow archived publisher references

The established KnowMoreDiRT `raspi_hierarchical_library.py source-evidence`
command now accepts the **explicit opt-in** `--follow-archived-links` with
`--linked-originals N` (one to five, default three). The selected source must
be **one exact archived HTML original** (given by `--original-path` or a
uniquely resolved `--source-name`). RFC continuation cursors, PDF/text-only
originals, unbounded requests, and source-name ambiguities are rejected for
this mode. The existing default `source-evidence`, `search`,
`find-source-evidence` and `find-pdf-evidence` operations and native document
ranking are unchanged when the switch is absent. No network access is
introduced and the existing original folder/grep search remains usable.

For example, after selecting the exact Unity 6.3 LTS **Loading textures in
the background** original, the command is:

```sh
PYTHONPATH=src python3 scripts/raspi_hierarchical_library.py source-evidence \
  'How do I avoid noticeable stutters when transferring model mesh and image asset data from disk in a running game?' \
  --source-name 'Unity 6.3 LTS — Loading textures in the background' \
  --follow-archived-links --linked-originals 2 --top 2
```

The normal original-source response keeps **zero direct parent excerpts** for
that indirect question. A distinct
`linked_archived_original_references` object contains the separate selected
source-link proof and one already archived **Texture and mesh loading**
child, with the child's full original-source SHA, derived SHA, native FileDB
identity, a candidate extracted-text offset/hash and an independently
normalized **original publisher HTML article** text anchor. The same response
separately lists the official **Make a texture or mesh compatible with
asynchronous loading** link as
`not_in_existing_catalog` with **no acquired content**, no downloaded page,
and no invented quotation. It still never claims that the candidate is a
correct answer, literal publisher byte coordinate, physical PDF page, or
proof of an immutable live cross-store generation. A wholly unrelated
question about neutrinos returns **zero** verified archived links and zero
source passages. Without `--follow-archived-links`, the existing query has
exactly its prior zero-parent-passage structure and no linked response.

The opt-in source-link search uses the previously published bounded
`raspi_archived_html_follow_links` reader; the original parent and every
selected target are independently rechecked against completed acquisition
receipts, current original and derived SHA and native FileDB IDs **after**
passage selection. It checks the whole candidate set again at the end. A
revision mismatch refuses the entire optional response instead of
presenting stale source evidence. The added summary in the normal private
query audit contains counts and exact source paths but **no publisher
paragraphs**; it distinguishes parent direct excerpts from linked
publisher-article candidate passages and missing sources. The separate
standalone link-follow command remains available for source-navigation
research and reproducible private receipts. The opt-in is **live-read-only
CLI functionality**, not installation in the owner-facing web UI, default
semantic ranking or permission to stage new sources or repair 327 PDFs.

Five opt-in dispatch regressions exercise normal zero-change defaults,
unique source identity, fail-closed non-HTML/continued/unbounded requests,
late linked-source changes that reject the entire response, and hard-negative
queries without fallback. Real Unicode exact-title, direct-path and
hard-negative CLI checks confirm the behavior in the actual Raspi archive.
The overall Knowledge Library source migration remains blocked until the
seven known live reader/writer admission and rollback requirements are met.


The opt-in response now also includes one **bounded, directly readable**
`extracted_text_excerpt_candidate` per linked source passage. Before the
candidate text is returned, its exact start/end character offsets must
reproduce the current derived source text, and its UTF-8 SHA must equal the
attested original-derived excerpt digest. The entire original publisher HTML
and complete derived text are separately rehashed, the normalized original
`article` text anchor is independently verified, and all original source
revisions are checked again at the end. A change or forged excerpt, SHA or
position rejects the entire linked lookup. The preview is still explicitly
`extracted_text_excerpt_is_not_a_literal_publisher_byte_quotation=true` and
`factual_answer_sufficiency_verified=false`. It provides evidence the user
can actually read in one opt-in CLI request, without promoting a machine
extraction into a literal original-byte quotation or confirmed answer.


### November 24 — optional archived original references alongside native mixed-source discovery

The existing canonical
`scripts/raspi_hierarchical_library.py find-source-evidence` (which can
select one source by its unique official title or rank up to five originals
by the **unchanged native semantic search**) now offers
`--follow-archived-links --linked-originals N` as an **explicit** read-only
extension. Only an already selected, source-revision-verified **HTML
original** is eligible to have its article references followed. Linked
originals are **separate** evidence candidates: they never replace, rerank,
or pretend to be the previously returned native sources or PDF page quotes.
`--linked-originals` is a bounded **total** budget, one through five,
across all selected HTML parents. Source names, native order, direct
publisher evidence and the ordinary command JSON remain unchanged when the
opt-in flag is absent. This is distinct from the earlier
`source-evidence --follow-archived-links` feature, which accepts one
explicit original without native mixed-source discovery.

A reproducible real source query on Raspi is:

```sh
cd /data/src/github/KnowMoreDiRT
PYTHONPATH=src /data/var/knowmoredirt/semantic-venv/bin/python \
  scripts/raspi_hierarchical_library.py find-source-evidence \
  'How do I avoid noticeable stutters when transferring model mesh and image asset data from disk in a running game?' \
  --source-name 'Unity 6.3 LTS — Loading textures in the background' \
  --documents 1 --passages 2 --snippet-characters 850 \
  --follow-archived-links --linked-originals 2
```

The native-selected Unity overview stays `rank=1`, with
`no_verified_excerpt` and zero directly matching passages. The separate
`opt_in_archived_original_reference_navigation` output contains the
already archived first-party **Texture and mesh loading** child, with
one **readable** current derived-text excerpt, its exact UTF-8 SHA and
source offsets, an independent normalized original publisher HTML article
anchor and original/native source revision checks. It also reports a
linked official compatibility guide **not in the local catalogue** as
unavailable, never fabricating an original document or downloading it.
The optional audit record contains **only source paths, counts and scope
warnings**, not copyrighted publisher paragraph text. The result explicitly
says `not_a_verified_correct_answer=true`,
`cross_store_immutable_generation_pinned=false` and
`safe_to_cutover_live=false`. Existing PDF physical-page and original
plain-text byte-range proof remain separate and are unaffected.

The opt-in rejects RFC continuation, simultaneous private corrected-PDF
staging, requests exceeding the total five linked originals, stale or
swapped parent/child SHAs, duplicated source identities, invalid native
file IDs, and HTML candidates without a current approved source-map/native
original. A final **collective** pass reopens and rehashes all selected
HTML parent and linked child originals after the last link was searched;
changed original/derived bytes or native IDs refuse the entire response.
This is a *best-effort post-search consistency fence*, not an atomic
multi-SQLite writer admission protocol. Source-link quality also depends on
an HTML original being selected by native search: if a relevant document
never appears in the selected native top-five, following its links cannot
recover it. The capability is **live-read-only on the Raspi CLI and
committed to the official KnowMoreDiRT repository**, not installed in the
owner-facing web Explorer or authorized as a live PDF/HTML source cutover.

The fixture regression suite covers unchanged default native rank and direct
PDF/HTML excerpts, explicit named source selection, two native-selected
HTML parents sharing one total child budget, bogus child count and identity,
late changed parent/child revisions and unchanged direct result on an
unrelated question. Actual Unity original HTML source/derived SHA and native
ID were rechecked on Raspi after running the opt-in command. No source
acquisition, model replacement, service restart, index modification, or
publication of the library's private books is involved.


### November 24 — label overlap between native search ranks and followed original references

One real native `find-source-evidence` query for Unity Render Textures selected
five relevant archived HTML originals and, with the explicit optional
`--follow-archived-links --linked-originals 3` mode, checked three official
first-party linked originals. The full current original and derived SHA and
source-map/native-ID checks passed, and three exact source text excerpts also
matched their archived publisher HTML articles. **However, one of those
three linked references was already present at native search rank four.**
The retrieval thus discovered **two genuinely new distinct original sources,
not three**. Source-linked citations remain legitimate for overlapping
originals, but describing a repeat as a newly found document inflates
retrieval-quality metrics.

The optional mixed native lookup now keeps **both** facts separately. Each
linked candidate reports `already_in_native_selected_documents` and
`native_selected_original_rank_if_already_selected`; group counters report
`linked_original_sources_checked`,
`already_native_ranked_linked_reference_count`,
`distinct_new_linked_originals_not_in_native_selection` and
`distinct_linked_original_paths`. When an already ranked target is linked,
its publisher source SHA, derived text SHA and exact native original FileDB
identity MUST match the selected native document or the entire query fails.
An original already linked from an earlier HTML parent may also be a later
native-selected parent, but only when its exact source identity agrees.
Repeated links from different parents are not represented as multiple
newly acquired or discovered originals. Native rank, original source
passages, source-link evidence and original corpus are never changed by this
annotation. The audit records source IDs and counts, never original
paragraphs. No cross-store atomic generation is claimed.

Replaying the actual five previous native-ranked Unity documents against
fresh original source-map/native and publisher SHA checks confirmed exactly
three linked references, **one overlap at native rank four and two new
originals**, without rerunning the expensive all-vector scan. Synthetic
regressions cover native overlap, a linked child also serving as a later
native parent, differing native IDs/SHAs, per-request budget, late source
revision changes, and default behavior unchanged. The independent same
native semantic query required approximately **106 seconds wall time**
in this observed configuration and read gigabytes of vector data. This is
an observed individual query, **not a latency distribution or speed
benchmark**. The existing Infra native semantic reader imports a historical
worktree implementation of `search_semantic_entries` that scans all native
chunk and representation vector rows; it is not a persistent ANN index.
Do not silently replace that live implementation or the owner-facing web
portal during active indexing. A future performance release requires
isolated equivalent-result benchmarking, source revision consistency,
read/write admission and a safe deployment/rollback plan. The existing
explicit-source lookup remains much cheaper and source-exact.


### November 24 — independently verified low-memory native source search (opt-in)

Current native semantic search is slower than it needs to be for an
interactive library: one Unity Render Texture request took about 106 seconds
wall time in the observed Nov 24 cold/busy environment. A later detailed
**read-only one-query profiler** took 20.1 seconds: the 23 MB private source
catalogue was fully read twice (8.5 s), the native SQLite chunk and
representation scan took 7.0 s and the E5 model initialized in 2.2 s.
These were *different single requests, cache and worker contention differed*;
these two numbers must not be presented as a representative latency average,
or attributed entirely to embedding computation. The existing installed
Infra reader imports a historical native scanner that first materializes
**about 71,000 chunk/representation candidates** as Python dictionaries,
then selects the strongest candidate per original and sorts all 17,715
originals. Source revision checks, publisher SHA and excerpt SHA are essential
and **must not be removed for speed**.

A new standalone, user-invoked helper provides **lower peak Python allocations
with exactly the same search results**, without modifying the running
portal or other workers. `native_streamed_semantic` iterates the exact
original native SQLite SELECT rows in the same order, invokes the same
`vector_from_blob` and **unchanged** `cosine_similarity` for every candidate,
retains only the best-scoring candidate dictionary per file ID and uses
exactly the same final sort/first-winner tie contract. It still returns
*every* ranked original, because the installed source-verification reader
may need to skip several stale/uncitable top-ranked candidates.

A separate exact-score improvement now computes the unchanged current query
vector's NumPy L2 norm **once per scan**, rather than approximately 71,000
times. The denominator still uses precisely the legacy expression
`float(query_norm * np.linalg.norm(candidate_vector))`; the same
`np.dot(query_vector, candidate_vector) / denominator` conversion, zero
vector behavior, and **strict greater-than** winner comparison are retained.
It is a scalar operation reuse, **not** approximate vector normalization,
ANN indexing, rounding, model-revision changes or source filtering.
Corrupted nonfinite native vector regressions explicitly verify that NaNs
do not displace earlier correctly scored source candidates. The complete
read-only current native database was checked with three distinct real E5
query embeddings: each comparison returned **exactly the same 17,715 file
ranks, selected chunk IDs, native original IDs and float scores** and
unchanged SQLite data version. Observed legacy/full-streamed per-scan time
pairs were **6.617/4.189, 4.347/3.922, and 4.334/4.037 seconds**. The
scans always ran old then new; OS cache and concurrent writer activity can
explain part of the difference. These are evidence of successful *exactness*
and promising CPU savings, **not a controlled claim of user-visible query
latency reduction**. The source catalogue's repeated reads, CPU model
startup, original physical byte SHA and SQLite revision checks still incur
runtime. Only the optional one-shot lower-memory KMD scanner uses this
precomputed query norm; the active Infra default is unchanged.

The optional `scripts/raspi_low_memory_native_original_search.py` is a
**one-shot, single-Python-thread** command. It temporarily substitutes only
the process-local native-scanner function for one call to the installed Infra
semantic source reader and **restores the original function even after a
failure**. That reader still checks the current source catalogue, completed
acquisition receipts, source map, original/derived physical SHA, native
FileDB identity, exact passage SHA and final revision. No persistent cache,
ANN index, model replacement, network fetch or database write is introduced.
It rejects multiple Python threads, invalid/unbounded requests and any
return result missing an exact passage SHA or final source-generation safety
contract. Results include machine-extracted *candidate passages*, not factual
answer verification or literal publisher text/byte evidence.

Example (private output deliberately outside Git):

```sh
cd /data/src/github/KnowMoreDiRT
PYTHONPATH=src /data/var/knowmoredirt/semantic-venv/bin/python \
  scripts/raspi_low_memory_native_original_search.py \
  'Unity Render Texture Inspector create Render Texture view from camera' \
  --top 5 \
  --private-output /data/var/knowmoredirt/knowledge-library/search-audit/MY-NEW-PRIVATE-LOWMEM-SEARCH.json
```

**Measured evidence and honest limitations:** On the current real Raspi
native database, comparing both scanners in one read-only SQLite connection
with unchanged `PRAGMA data_version` produced **17,715 exactly equal result
records, ordered chunk IDs, original IDs and floating-point scores**. The
legacy scanner took **7.98 s** and streaming took **9.26 s** in that
uninstrumented comparison: **no verified speed improvement**. In a
separate `tracemalloc` comparison over the same 17,715 results, peak Python
allocation fell from **87,915,519 bytes to 24,535,717 bytes** (72.09%
reduction), with exact results and unchanged database version. Tracemalloc
does **not** include SQLite's OS file cache or every native model allocation;
its instrumented times do not represent normal search latency. A real
standalone opt-in query returned the **same five Unity source originals,
native chunk IDs, original SHA, passage SHA and scores** as the existing
reader, with full original/derived/native final revision checks. Fixture
regressions additionally check mixed float16/float32 vectors, first-winner
ties, excluded incomplete representations, scoped original CTEs, read-only
SQLite and restoring the old scanner on exception/multithread rejection.

**Deployment scope:** This helper is an independently tested private CLI
alternative, **not the live default web or source-evidence scanner**. The
native E5 search still requires significant I/O, and replacing the old live
reader would require coordination with agents changing the heavily dirty
Infra repository, equivalent-result tests under concurrent writers, and a
rollback plan. Source-boundary migration for corrected PDF/HTML documents
still has seven independent live reader/writer/admission blockers.


### November 24 — one short-lived local model and original-source vector snapshot for multiple verified searches

The **read-only** `scripts/raspi_batch_low_memory_original_search.py` supports
multiple source searches in one private command. The ordinary mode accepts
**two to six distinct questions**, reuses the installed, unchanged pinned
multilingual E5-small CPU model across all questions, and invokes the
existing installed source-verified Infra native reader with the earlier
exact-result low-memory streaming scanner. The original acquisition catalog,
current source map, native FileDB and publisher/derived bytes are still
checked fresh **on every question**, and all returned passages receive a
**final combined, full-original source-revision check after the last
question**. Earlier results cannot be returned as verified if their source
or native ID becomes stale while later queries run. The ordinary mode does
not cache any native vector or source metadata between queries.

For repeated broad searches there is an additional, explicit option
`--reuse-current-native-vectors`. This loads the **current read-only** native
`content_chunks` and complete representation rows **once** into a bounded,
short-lived in-memory candidate snapshot, checks the native SQLite
`PRAGMA data_version` and file inode, size, mtime and ctime during construction
and **before/after every question**, and uses bit-for-bit equivalent original
cosine scores and first-winner ties to rank the very same native original
sources. The snapshot remains private to this single-threaded CLI process;
no ANN index, on-disk source cache or new vector/index write is created. Any
**observed** native database change makes the entire batch fail closed,
not return stale candidates or silently use cached rankings. After the last
question, a fresh all-results source check independently opens the current
catalog, completed acquisition receipts, source-map SQLite and native
FileDB, rehashes the complete original and derived documents and rechecks
every chunk SHA/current native ID. This is **NOT** cross-store writer admission,
a guarantee of zero concurrent source-map changes or authorization to cut
over the corrected PDF/HTML indexes; live migration still has seven blockers.

The snapshot variant has stricter resource admission than ordinary batch
mode: **two to four** queries, at most four results each, current original
corpus mode only (not a filtered path or alternate source collection), at
most **100,000 native vector candidates and 30,000 original IDs**, at least
**two GiB Linux MemAvailable** before construction and at least **512 MiB**
while it runs. Consult and reserve the shared Raspi resource broker before
this optional higher-memory operation so resident summarization services
retain capacity. The snapshot's measured **whole process peak** was about
**762 MiB** for two searches and **1,013 MiB** for three searches in the
2026-11-24 Raspi environment; actual memory depends on native SQLite page
caching and corpus growth. The plain batch or one-shot low-memory source
search is preferable under tighter RAM pressure. The command restores the
original process-local E5 model class and vector scanner even on exceptions,
and never patches other running processes or the active web portal.

Concrete example, creating only a new owner-private receipt:

```sh
cd /data/src/github/KnowMoreDiRT
PYTHONPATH=src /data/var/knowmoredirt/semantic-venv/bin/python \
  scripts/raspi_batch_low_memory_original_search.py \
  --reuse-current-native-vectors \
  --query 'Unity Render Texture Inspector create Render Texture view from camera' \
  --query 'physically based roughness normal map material shading for industrial warehouse rendering' \
  --query 'agent evidence provenance factuality source citations verification retrieval' \
  --top 2 \
  --private-output /data/var/knowmoredirt/knowledge-library/search-audit/MY-NEW-PRIVATE-BATCH.json
```

**Independent correctness and performance evidence:** A read-only
three-question comparison with the actual native SQLite database found
**17,715 exactly identical ranked originals, full scores, selected native
chunk IDs and source identities in every question** compared with the legacy
full-materialization scanner, while SQLite `data_version` stayed unchanged.
After the five-second snapshot construction in that particular run, each
full-corpus candidate ranking took **0.47–0.50 seconds**, compared with
**6.27–7.73 seconds** for the uncached legacy scanner. Those are scanner
measurements, not end-to-end user-response times. On a two-question
source-evidence benchmark, the previous warm-model streaming batch took
**15.95 seconds** and the vector snapshot batch took **12.15 seconds**;
**all four returned original-source passages, hashes, scores and native IDs
matched exactly**, and the final source check physically rehashed all four
original/derivative revisions. A separate real three-topic acceptance
returned **six** publisher SHA-verified passage candidates in
**22.43 seconds**, including Unity documents, material-rendering research
and agent evidence verification. The native vector snapshot took 10.55
seconds to build in that more contended run, versus 5.23 seconds previously;
**do not claim a universal fixed latency or speedup** in the presence of
uncontrolled OS caches, background writers and CPU contention. A direct
one-shot scoped original query remains valuable when a source is known.

Adversarial fixtures test identical original/native ranking with 16- and
32-bit vectors; strict finite and non-finite winner behavior; source
collection/path scope refusal; a native writer committing during a batch;
new original revisions and altered publisher source passage hashes; model
or source guard failure after an earlier successful query; cross-thread
usage; total candidate and physical-memory admission; and full restoration
of the process-local scanner and model. No generated answer is given factual
authority solely because a source passage is hash-verified. The real
source-verified result includes exact passage SHA, original-source revision,
a link to its archived original, and explicit
`cross_store_immutable_generation_pinned=false` / `safe_to_cutover_live=false`.

This is a **working, tested private command-line capability**, not a live
replacement for the default semantic search, a deployed website feature, or
an authorized source acquisition. It improves repeated-query throughput
when the cached native database stays unchanged; any concurrent native
index update properly aborts this batch and can be retried in a new,
independently authorized read-only turn after resource admission.


### November 24 — frozen seventy-three-question original-article link recovery and safe URL normalization cache

A new **read-only quality evaluation** reused the exact previously frozen
73 independent research queries and 79 known publisher original-source
labels, with no changes to their query SHA, approved acquired source paths,
previous captured RRF-three rankings or gold identities. The official
first-party link following was applied as a **supplementary navigation
channel**: for each query, consider only the top five originals from the
previously captured reciprocal-rank-fusion source ranking. Parse only the
full-original-SHA-verified `article` (fallback `main`) element for any HTML
parents, preserve a fixed **maximum five same-origin, acquired, completed
and query-token-related link candidates per parent**, and inspect only
those pre-existing catalogued destinations. No source acquisition, external
HTTP request, summary or vector changes were made. The original and derived
source-map approval/revision values were compared with the completed ledger,
and the two recovered known originals were independently full-SHA-verified.

The fixed top-ten source ranking already found **65 of 79 known labels**.
Following its top-five publisher article links revealed **two additional
previously missed, already archived labels** among 421 candidate linked
references seen across **238 original HTML parents**:

* Unity 6.3 LTS **Reflection Probes** from the existing official
  *Reflections* overview (previously fusion rank **nineteen**).
* Unity 6.3 LTS **Event examples** from the existing official UI Toolkit
  *Events* overview (previously fusion rank **eleven**).

Both rescued labels belong to the independent 27-query holdout, and the
existing full publisher-original `article` evidence reader independently
verified exact extracted-text excerpts against each archived official HTML
page, current full original and derivative SHA, canonical native FileDB
identity and final source-map revision. The original pages were already in
the library; nothing was downloaded. This identifies a real **supplementary
source-discovery benefit**. It does **not** prove top-ten ranking improved
from 65 to 67 labels: adding as many as 25 linked candidates per query
increases the candidate budget, and the remaining 419 link occurrences
are not labelled false or true just because they lack frozen gold labels.
The benchmark is about **known original-source discovery**, not whether an
answer is factually sufficient or whether arbitrary original quotations can
be trusted. The result must not silently modify the default search.

The frozen study exposed an independent **performance problem**: the existing
`source_link_candidates` operation re-parsed the same large publisher URL
catalogue for every original HTML article. Pure `_canonical_url_string`
normalization now has a **bounded 32,768-entry in-process LRU** cache;
source catalogue entries, permitted URLs, completed acquisition ledger
status, source-map revisions, original files and native IDs are **not**
cached or pinned. Every call still consults the **current** caller-supplied
manifest and completed receipt, and still full-SHA-validates the original
HTML. Synthetic regressions deliberately invalidate a ledger receipt and
change the official catalogue URL **after the cache is warmed**; neither
stale original is admitted, and restoring the approved current source
restores the candidate. Unsafe or unhashable URL arguments remain rejected.

A second complete run of the exact same 73 frozen questions, original
SHA-approved HTML parents and 421 selected official article links produced
**identical, byte-for-byte candidate case records** and exactly the same
independent two-source recovery. Approximate shell start-to-private-artifact
times fell from **173 seconds to 92 seconds** (a 47 percent reduction in
this pair of observed runs), with no acquisition, native DB write or
model/runtime replacement. The timings are **not a controlled population
latency benchmark**, and other processes, disk cache state and source-map
writers can affect them. Source link navigation remains opt-in and exposes
explicit unarchived/unapproved gaps without inventing destination content.
The full original excerpt and native ID verification still runs for each
actual user-facing linked source candidate.


### November 24 — verified first-party Markdown source navigation and original literal publisher bytes

The offline library now treats **already archived first-party Markdown**
source links as verifiable publisher evidence, not just HTML `article`
links. This closes an actual source-discovery gap for first-party n8n,
agent-automation and technical documentation stored as `.md` originals.
The new read-only `archived_markdown_source_links.py` extracts only bounded,
ordinary inline `[label](href)` links from the **full-original-SHA-verified
publisher UTF-8 bytes**, excluding fenced/indented/inline code, Markdown
images, cross-origin links, non-HTTPS URLs and ambiguous or uncompleted
archive entries. Each candidate identifies the exact **literal Markdown
source link byte start/end** and independent SHA of that original byte
interval, and its unique HTTPS same-origin URL, current catalogued target
and completed acquisition receipt. A missing or incomplete publisher link
is explicitly marked **not available** and never fetched or quoted. This
is **not** a Markdown or HTML general-purpose renderer: unsupported nested
links/reference syntax are excluded rather than guessed.

The companion `scripts/raspi_archived_markdown_follow_links.py` follows at
most **five** approved same-origin links from one explicitly selected,
verified `.md` parent. It independently rechecks each target's complete
publisher original SHA, derived text SHA and native KnowMoreDiRT FileDB ID,
and ranks only **source-passage candidates** within the already acquired
target. For native original plain text, the reader independently verifies
the returned passage's exact **publisher UTF-8 original-byte interval**,
in addition to the current extracted-text char offsets and SHA. Publisher
Markdown **link** original byte coordinates and publisher **passage**
original byte coordinates are separate proof scopes. All parent and child
source SHA/source-map/native IDs are physically rechecked at the end,
refusing observed stale generations; cross-database atomic writer admission
is **not** claimed and live PDF/HTML migrations remain disabled.

The standard `raspi_hierarchical_library.py source-evidence` CLI now also
supports `--follow-archived-links` for one explicit `.md` original. An
optional `--exact-link-text` selects one publisher link by its literal
label, e.g. from an official table of contents, without any claim that the
label alone is semantically relevant or answers the user's question. This
flag requires the opt-in switch and a Markdown original; an HTML-only
source is still handled by the older independent DOM-anchored reader.
RFC `.txt`, PDFs and continuing RFC section cursors cannot enter the
Markdown link path. The default source-evidence response is unchanged when
`--follow-archived-links` is absent. The private query audit retains only
counts, source IDs and verification metadata, not the publisher prose.

Example using **already acquired** original n8n pages (no downloads):

```sh
cd /data/src/github/KnowMoreDiRT
PYTHONPATH=src /data/var/knowmoredirt/semantic-venv/bin/python \
  scripts/raspi_hierarchical_library.py source-evidence \
  'How can I reshape and map JSON fields between nodes in an n8n workflow?' \
  --original-path 'money-and-business/operational-ai-money-2026/automation-agents-and-reliable-workflows/work-with-data-e2147cc38.md' \
  --follow-archived-links --exact-link-text 'Transform data' --linked-originals 2 --top 2
```

This reproducibly returns the source-verified original **Work with data**
overview and its officially linked, independently byte-verified **Transform
data** original. A second explicit step from that overview into its
already archived **Approaches for transforming data** original returned
two more independently publisher-byte-verified source excerpts, including
which expressions and code nodes can restructure data. Direct source
excerpt truth remains an *evidence candidate*, not an automatic complete
answer. All current publisher bytes, derived revisions and native IDs were
rehashed and accepted in both real n8n paths. An unknown exact link name
must return **zero**, not replace it with a merely similar publication.

The earlier immutable same-corpus seventy-three-question benchmark had
**79 known original-source labels**, of which **65** were in the fixed
three-channel fusion top ten. Official first-party HTML article links
recovered two previously missed Unity original-source labels. Adding
**original Markdown publisher links** from the same fixed five fusion
parents per question recovered a **third** known original: n8n **Transform
data**, whose source itself is only a short navigational table of
contents. The complete evaluated Markdown scope was **22** original
SHA-checked parents and **88** query-term-matched approved link
occurrences. This three-source total is a **supplementary navigation
recovery**, NOT top-ten ranking or answer-accuracy improvement: each parent
can introduce up to five additional candidate originals, and a known-label
hub is not necessarily the best factual answer. The transformation question
is better answered by additional already archived first-party **Use the UI
mapper** and **Approaches for transforming data** guidance. Do not tune a
ranking model just to hit a benchmark's table-of-contents label.

Synthetic regressions cover exact UTF-8 publisher link byte intervals and
SHA, same-origin official completed receipt, changed catalog or receipt,
ambiguous duplicated URL, code/image/external spoof links, unsafe source
paths, missing targets, stale parent/child full source SHA and native ID,
modified extracted-text excerpt hash, modified publisher byte excerpt SHA,
and the default-HTML-source command remaining unchanged. A further adversarial
case exposed an actual source-spoofing bug in the original narrow Markdown
link parser: counting backticks modulo two incorrectly promoted links
inside **double-backtick and multiline inline code examples** as real
publisher references. The parser now tracks actual backtick delimiter run
length across lines and excludes inline/multiline HTML comments. Publisher
links after an explicitly closed code span continue to work; unmatched
code delimiters fail closed. Both official source-navigation behavior and
all **73 immutable benchmark questions, 22 Markdown parent original SHAs,
88 approved Markdown links and one recovered n8n gold original label** were
unchanged on independent full-current-source recheck. No source-publisher
approval status is ever cached or inferred from a quoted code sample. The tool is a
working **private, read-only CLI capability** in the canonical KnowMoreDiRT
repository, not deployed into the owner-facing web GUI and not permission
to acquire additional copyrighted manuals or change the production corpus.


### November 24 — native-ranked document discovery can now follow original Markdown as well as HTML

The existing `scripts/raspi_hierarchical_library.py find-source-evidence`
command now accepts its explicit `--follow-archived-links` option for
**verified native-selected `.md` publisher originals**, as well as HTML
`article`/`main` originals. This is a separate, read-only source-navigation
channel; the existing native semantic original ranking, selected document
order, original direct passage status and PDF physical-page evidence remain
unchanged. Linked sources never replace or rerank native originals. The
user need **not know an exact source title** to begin: the normal native
search selects up to five original documents first, after which at most
five *total* approved same-origin links across selected HTML/Markdown
parents may be followed. Only current completed originals with approved
catalog+ledger SHA, derived SHA and native FileDB identity are admissible.
The entire parent and child source revisions are checked again **after**
all linked passages have been assembled, refusing observed stale versions.

The optional mixed response distinguishes **original Markdown literal
hyperlink source bytes**, independently verified **linked publisher UTF-8
passage byte intervals**, and HTML's separately verified normalized
`article` text anchors. The original `source-evidence` and
`find-source-evidence` default modes are unchanged when the option is
absent. The private audit contains only source paths, ranks, counts and
proof metadata, not original prose. The result explicitly marks linked
texts as **candidates, not a verified factual answer**, and preserves
`cross_store_immutable_generation_pinned=false` and
`safe_to_cutover_live=false`. PDF physical-page quoting, continuing RFC
sections and temporary corrected PDF private overlays remain separate and
are not silently routed through HTML/Markdown reference matching.

A real reproducible query with no exact title is:

```sh
cd /data/src/github/KnowMoreDiRT
PYTHONPATH=src /data/var/knowmoredirt/semantic-venv/bin/python \
  scripts/raspi_hierarchical_library.py find-source-evidence \
  'How can I reshape and map JSON fields between nodes in an n8n workflow?' \
  --documents 5 --passages 2 --snippet-characters 850 \
  --follow-archived-links --linked-originals 5
```

On the **actual live-read-only Raspi library** this selected five native
original Markdown documents—*Work with data*, *Break workflows into
smaller parts*, *Use the UI mapper*, *Configure workflow settings*, and
*Export and import*. Starting from the original *Work with data* parent
(natively ranked first), optional publisher links found **five already
acquired child originals**: *Expressions versus data nodes*, *Reference
data*, *Transform data*, *Filter out unwanted data*, and *Pin and mock
data*. Each child yielded **two** source text passages, for **ten**
independently publisher-original UTF-8-byte-verified passage proofs.
The independently SHA-attested original parent's literal Markdown link
bytes, each entire original and derivative SHA, and each unique current
native FileDB original ID were rechecked. *Transform data* is the same
publisher source **missing from all top-thirty channels of the previous
frozen seventy-three-question benchmark**, now available through the
new explicit navigation channel. The lookup did **not** make it the
native top-five ranked result, automatically verify answer sufficiency or
download any outside page.

A second **explicit title** query for *Work with data* returned four
approved links with **eight** original-publisher byte-proven passages,
plus two clearly unavailable/unapproved links, and preserved the original
native rank. A missing literal Markdown link returns no invented sources.

For each Markdown publisher byte quote the independent reader checks not
only the literal original-byte range SHA, but also that decoding the selected
original UTF-8 bytes (using only declared universal newline normalization)
produces the **exact extracted-source candidate passage**. A second genuine
publisher text span with a valid SHA but unrelated content is **not**
accepted as a quotation for the retrieved excerpt. A new adversarial
regression alters the claimed physical publisher byte coordinates and
recalculates their genuine SHA; the combined lookup correctly refuses it.
This strengthens provenance beyond the extracted-text-only hash while
leaving PDF physical page and HTML normalized DOM proof scopes separate.
The optional reader was tested against changed parent and target SHA,
wrong native ID, missing approved acquisition, duplicated URL and final
source revision, as well as mixed HTML + PDF defaults, a shared five-child
budget and overlap with an already native-ranked Markdown child. No
other worker's source/index/cache was stopped, copied, replaced or
reindexed. All acceptance evidence remains in private KMD search-audit,
not Git or an owner-facing developer report.

At this point the optional HTML/Markdown source discovery still relies on
current native candidate ranking. A relevant parent excluded from the
native selected group cannot be reached by this *one-hop* mode. Source-link
candidates are a useful supplementary navigation path, **not** proof of
higher equal-budget top-ten recall or semantic answer accuracy. The
frozen benchmark now has **three** independent predeclared known originals
recovered through extra first-party HTML/Markdown references beyond the
original top-ten (two Unity, one n8n). Changing default rankings requires
independent precision/recall and answer-sufficiency tests with a fixed
equal candidate budget and negative examples. The 327-PDF and 2-HTML
production revision cutover still requires all seven reader/writer
synchronization safeguards.


### November 24 — fixed ten-source-slot recall guard, physical originals, and lexical stopword no-go

Improved original-source navigation is not evidence of a stronger *default*
source ranking. The optional HTML and Markdown link readers found three
previously missed, approved official source documents, but they added
additional retrieval candidates **beyond** the original ten native source
results. The new pure `file_system_catalog.frozen_reference_budget`
comparator and owner-private one-shot
`scripts/raspi_frozen_reference_budget_eval.py` provide a reproducible
**at-most-ten-result-slot comparison** before any such promotion. They take
the original SHA-attested **73 predeclared questions, 79 known approved
publisher original labels**, their frozen original/summary/lexical
three-channel fusion ranks, and the already completed original HTML and
Markdown first-party publisher link studies. Every link must originate
from one of the previous **top five native-ranked original parents**.
Links remain in their prior publisher reference order and never receive
an artificial relevance label or training reward from the frozen gold.

The fixed candidate-capacity policy is **K native original slots plus up
to 10−K first-party linked originals**, for K in **10, 9, 8, 7 and 5**.
A duplicated source or fewer available links consumes a slot rather than
secretly adding extra results. This is a fair **maximum result-capacity**
comparison, not a promise of exactly ten distinct original documents in
every query. It is based on **known-source identity**, NOT factual-answer
correctness, exact quote sufficiency, product research utility, physical PDF
page accuracy or statistical power. The already inspected independent
27-query cohort is **not a pristine blind holdout**. The benchmark uses
a fixed source corpus, not continuous live native search snapshots.

Observed source-known recall results (read-only November 24 cohort):

| Fixed maximum ten-source capacity | 46 earlier questions (52 labels) | 27 prior independent questions (27 labels) | All 73 (79 labels) | Gain/loss versus 10 native |
|---|---:|---:|---:|---|
| 10 native, 0 links | 41 | 24 | **65** | unchanged |
| 9 native, 1 link | 39 | 25 | **64** | +1 previously missed Unity guide; −2 existing known originals |
| 8 native, 2 links | 39 | 25 | **64** | same net loss |
| 7 native, 3 links | 39 | 25 | **64** | same net loss |
| 5 native, 5 links | 36 | 26 | **62** | +2 missed Unity guides; −5 previously found originals |

The n8n *Transform data* source recovered by unlimited supplementary link
navigation was **not** recovered by these constrained source-order budgets;
its official overview link is behind earlier links, which can occupy
all available slots. This is an observable reason that extra navigation
and source-aware reading should be a **separately selectable channel**,
not a blind replacement for native top-ten search results.

The reusable one-shot private evaluation additionally checks the **actual
current publisher original and derivative bytes** for every admitted frozen
gold source, selected linked source and publisher parent. On the original
accepted dataset it streamed full SHA-256 through **284 unique current
originals and their exact derived files**, checked completed first-party
catalogue/acquisition metadata, unique native FileDB original IDs, current
source-map revision, and re-opened both SQLite databases after the full
file review. Any observed current source revision, missing archive file,
symlink, incomplete acquisition receipt or changed native source identity
aborts the complete benchmark. It does **not** claim an atomic cross-SQLite
reader/writer admission lease: concurrent source updates may still race
between independent stores, and seven source-generation cutover blockers
remain. There are no model calls, HTTP requests, downloads, automatic
source/index changes, owner report effects or cached permanent vectors.

Example command creating a **new** private evidence receipt:

```sh
cd /data/src/github/KnowMoreDiRT
PYTHONPATH=src /data/var/knowmoredirt/semantic-venv/bin/python \
  scripts/raspi_frozen_reference_budget_eval.py \
  --private-output /data/var/knowmoredirt/knowledge-library/search-audit/NEW-PRIVATE-FROZEN-TEN-SLOT-QA.json
```

A separate fresh native E5 same-snapshot pilot evaluated correction of the
historical lexical query word filter, where a long stopword string was
incorrectly passed to `set(string)` and thus split into **characters** rather
than actual words. The pilot derived the intended stopword tokens directly
from the published source's literal text, kept the same original/summary
vectors, FTS5 title-5/summary-1 Porter OR weights, and RRF k=60, and scored
all 73 frozen original-source queries without changing native search code.
**Top-ten fusion recall remained 65/79** and top-five was slightly worse,
**60/79 versus 61/79** in that current snapshot. The corrected tokenizer
is therefore **not promoted** to installed search based solely on the
plausibility of a conventional stoplist. The current known original-source
corpus grew from **9,614** complete summary roots on the earlier November 24
snapshot to **9,709** on the later one with the exact same known-source
revisions, while native top-ten known-source recall stayed **65/79**.
One top-five historical source label improved; there was no observed top-ten
regression. This is not a proof of broadly improved factual answer accuracy.

Until an independently better fixed-budget retrieval policy is found, keep
all official first-party links **opt-in and supplementary**, preserve
existing original folder/grep search and native semantic ranks, and measure
actual publisher-context and answer-sufficiency on hard negative questions.
The full evaluation evidence is private, with exact report SHA records in
the Raspi project exchange. It must not be mistaken for a seller product,
an authorized external publication or a source-generation live migration.


**Continuous writer admission for the private quality evaluator:** A real
postpublication read-only audit originally refused to complete when the
resident native summarizers committed **another document's** index rows
during physical SHA review. Rejecting *every* global SQLite
`PRAGMA data_version` change made the evaluation unnecessarily unavailable
on the always-indexing Raspi, even if none of the 284 selected originals
changed. The quality evaluator now first captures **the exact selected
original source-map revisions and native FileDB IDs** before hashing any
publisher files, then checks those same selected identities during hashing
and once again through fresh read-only SQLite connections at the end.
Unrelated committed native/source-map rows are allowed; any **selected
source-map SHA, derived file identity or native original ID change** still
aborts the entire result. Synthetic real separate-connection SQLite writer
fixtures verify both harmless new unrelated rows and harmful selected
updates, including writes made during a long hash. The latest live-read-only
audit passed for all 284 originals with the source-map and FileDB
identities rechecked on both sides. **That individual live run did not
observe a concurrent database commit**; concurrency behavior is proven by
the injected-writer fixtures, not falsely claimed from it. This is not a
cross-SQLite atomic generation or live migration protocol. Do not infer
a permanent source pin from successful before/after rechecks, and retain
the separate seven production source-cutover blockers.


### November 24 — opt-in publisher passage structure review (navigation, quoted prompts, scope)

**Physical source provenance is not the same as answer sufficiency.** Two
important hard negatives were found in approved original material: the
*BAPO Boundary-Aware Policy Optimization for Reliable Agentic Search* paper
has an appendix page containing a literal example **"TIR System Prompt"**
whose instructions a retrieval assistant must treat only as quoted research
data, and the approved first-party n8n *Work with data* documentation has an
**"In this section"** link list rather than an explanation of how to
transform input values. Both passages independently match current original
publisher pages or original UTF-8 bytes and have valid native FileDB IDs.
There is no source corruption; their **usefulness for a particular factual
answer** is the separate question. In n8n, *Use the UI mapper* explicitly
states mapping references prior node values but **does not itself change
data**; this negative scope matters when answering a transformation
question. An original quote's negation should not be silently dropped.

The new pure `file_system_catalog.source_passage_utility` helper is a
conservative **structural warning layer**, not a model-based truth scorer.
It observes the verified current *individual original excerpt* for multiple
publisher link-list lines and section headings, quoted agent prompt
examples, very brief method overviews, literal query-token overlap and
negation or scope restrictions. It returns only the actual original path,
source section, current excerpt SHA, publisher proof type, bounded token
matches and **observed cautions**. No source body is copied into query
logs, and no excerpt, native rank, first-party link or physical-page
citation is removed or re-ranked. `publisher_location_proof_scope`
explicitly distinguishes physically verified PDF pages, original publisher
plain-text byte intervals, independently normalized HTML article locations
and extracted-text-only spans. The reviewer cannot *create* a source SHA
proof; it is called only after the existing original acquisition/source-map
SHA and native FileDB identity checks, and the existing PDF/HTML/plaintext
publisher excerpt checks have completed.

The existing `raspi_hierarchical_library.py source-evidence` and
`find-source-evidence` commands now accept an explicit
`--assess-passage-utility` switch. Without it their established default
outputs, original/derivative/PDF-page revision checks, source selection and
runtime remain unchanged. With it, each native-selected original receives
`optional_source_passage_utility_review`, alongside untouched source
excerpts and physical citations. The private audit captures only the total
number examined and warning counts, **never source prose or generated
answers**. Direct original review is bounded to at most five excerpts and
no RFC continuation; native mixed discovery already limits document and
passage counts to five each. No model, vector/FTS rank, web portal, source
file or SQLite database is changed by enabling the flag.

**Real, independent source proof and result-parity checks:** The same
original BAPO paper question returned the **identical two PDF excerpts,
original SHA, derived SHA, source offsets, native IDs, and physically
verified publisher pages** with and without the option. The appendix's
*TIR System Prompt* was flagged as quoted instruction-template data; the
other research excerpt remained available, not automatically accepted as
a correct answer. An existing n8n question also returned its **identical
two first-party UTF-8-byte-attested passages**, and flagged *In this
section* as navigation. Direct and native mixed named-source commands
reproduced those findings. These demonstrations show that the new warnings
can expose non-answer text without corrupting original citations. They do
**not** establish high recall, precision, scientific truth or whether an
unflagged excerpt conclusively answers a question. The current fixed
73-query, 79-known-original quality benchmark remains a **document
identity** benchmark, not an answer-entailment benchmark. Source-limited
negative examples must be included in any later answer-quality evaluation.

One-shot examples using only *already acquired* publisher originals:

```sh
cd /data/src/github/KnowMoreDiRT
PYTHONPATH=src /data/var/knowmoredirt/semantic-venv/bin/python \
  scripts/raspi_hierarchical_library.py source-evidence \
  'agent search policy optimization boundary-aware improve search stability reliable multi step reasoning' \
  --original-path 'ai-agents/agent-reliability-and-automation/2026-bapo-boundary-aware-policy-optimization-for-reliable-agentic-search-c078e669b6.pdf' \
  --top 2 --snippet-characters 850 --assess-passage-utility

PYTHONPATH=src /data/var/knowmoredirt/semantic-venv/bin/python \
  scripts/raspi_hierarchical_library.py find-source-evidence \
  'How can I reshape and map JSON fields between nodes in an n8n workflow?' \
  --source-name 'n8n official documentation — Work With Data' \
  --documents 1 --passages 2 --snippet-characters 850 \
  --assess-passage-utility
```

A verified paper's quoted prompt **never becomes instructions to execute**.
A source heading, publisher original digest or literal query-term overlap
**never establishes factual entailment**. This is a small, working private
reader-quality diagnostic, explicitly separated from live default ranking,
answer generation, automatic source downloading and the seven unresolved
source-generation migration safeguards.


**Source-linked child originals are also reviewed when both opt-ins are
explicitly enabled.** The two existing reader commands may combine
`--follow-archived-links` with `--assess-passage-utility` on already archived
Markdown or HTML originals. The reviewer keeps the native direct passages
and approved linked children in **separate** structures, without assigning
linked children native ranks or inventing evidence. Source-linked Markdown
passages retain their verified literal publisher UTF-8 byte scope; linked
HTML passages retain their **normalized article proof**, not a fictional
literal byte interval. The private audit stores only the native and linked
review counts, never the original text or retrieved quote. On the real n8n
query, four first-party linked original guides produced eight SHA-verified
publisher passages. A further exact-source review exposed a **false
navigation warning**: the *Other data transformation nodes* section
contains explanatory method descriptions after links (*Remove Duplicates*,
*Sort*, *Split Out*, *Summarize*) and guidance on when to use them, not just
a bare table of contents. The structural classifier now distinguishes
these explanatory link entries from headings explicitly named *In this
section* or *Related resources*. On the same current eight linked n8n
excerpts, **two** are now flagged as navigation-only, while the useful
method descriptions remain unflagged. A separate brief *Transform data*
overview is marked as requiring further context. A
*Reference data* passage also explicitly warns that mapping a value does
not modify the underlying data. Crucially, comparison with the previously
verified original n8n retrieval proved **identical** native document ranks,
all four linked original identities and their full publisher source
passages. The optional cautions are source-reading aids, not a ranking or
factual-answer verdict.


### November 24 — exact publisher Markdown table cells restore cropped answer evidence without changing native ranks

A reproducible **answer-evidence** gap (not a document-discovery gap)
was confirmed in the already approved, archived first-party n8n source
*Expressions versus data nodes*. The ordinary native `find-source-evidence`
query **“Is the AI Transform node available to self-hosted n8n workflows?”**
ranks that correct original **first**, but its two extracted-text excerpts
omit the original table cell **“Available on: Cloud only”**. Even expanding
normal passage retrieval to the first five excerpts does not include the
cell, because the relevance excerpt is truncated before the row, while
the paragraph about the AI Transform node separately lacks its platform
restriction. Source authenticity and native ranking were correct; the
missing answer lay in a **structured publisher table**.

New read-only `file_system_catalog.source_markdown_tables` and opt-in
`--inspect-original-tables` on BOTH existing commands,
`raspi_hierarchical_library.py source-evidence` and
`find-source-evidence`, provide a separate **exact publisher-table
source-evidence channel**. They parse bounded plain original Markdown
pipe tables with original header and all column names, strict row shape,
current complete first-party acquisition SHA, full current derived SHA,
unique native FileDB original ID, exact UTF-8 original **header and row
byte coordinates and SHA**, and independently checked derivative row
character coordinates and SHA. The existing native source revision guard
runs after lookup and refuses observed source-map or native-ID changes.
In `find-source-evidence`, only already native-selected Markdown
originals are inspected, up to five documents and **two** source rows per
document; **native document ranks and the original passage excerpts
remain unchanged**. Direct named original requests admit at most three
row candidates. The optional audit logs **counts only** and never the
publisher row/quote text; on a read or revision failure, the complete
request fails and the safe private audit records the stage. No model,
network retrieval, source downloading or original database write occurs.

The row selector checks literal query words against **publisher column
names**, the **first column identifying the subject**, the source section
heading and the full row, retaining the publisher original row coordinates.
A longer exact multi-word subject phrase wins over coincidental matches
of a product name in another row. This is a **candidate ordering** for a
bounded separate source channel, not an entailment or correctness score.
For ambiguous tables, source heading context is preserved; heading words
are used for selection but are not confused with quoted row evidence.
Fenced Markdown code, indented code, HTML comments, unescaped/ambiguous
pipe syntax, malformed column separators, duplicate derivative row spans
and unrecognized table shapes fail closed or are ignored. No source
header/row is synthesized from another document.

Three **real already archived originals** independently verify the need:

1. **n8n — Expressions versus data nodes:** The first native-ranked
   document's physically source-SHA-checked table row has columns
   *Approach*, *Use when you need to...*, *Examples*, *Available on*.
   The **AI Transform node** row has *Cloud only*, whereas the **Code node**
   row has *Cloud and Self-hosted*. These are the **acquired publisher
   snapshot's own values**, not a separately checked live product
   subscription or feature policy. The original two normal excerpts
   still do not include the critical *Cloud only* cell; the new
   optional source row does, with current original and derived SHA.
2. **MaterialX physically based rendering specification:** The question
   asking for the mathematical **surface-normal symbol** must not be
   answered by an unrelated shader input *Port* named `normal`. The
   original table under *Physical Material Model / Scattering Framework /
   Symbols* contains the exact *Symbol* → **`$n$`**, *Description* →
   *Unit surface normal vector at `$p$`* relationship. The first table
   parser had selected a different but equally genuine `normal` input
   port row. Comparing **table column headings and original section
   context** fixed the ambiguity; current original-byte table header,
   source row and native FileDB SHA/provenance were independently checked.
3. **n8n — Use metrics to measure quality:** A question about the
   **Self-hosted Business** plan's maximum parallel test cases must not
   be answered using the adjacent *Self-hosted Community, n8n Cloud Pro*
   plan. The archived publisher table has **Self-hosted Business → 3**
   and **Community/Cloud Pro → 1 (sequential)**. An initial row selector
   mixed these up when generic *n8n* and *self-hosted* terms tied; matching
   the longer literal **`self-hosted Business`** first-column subject
   phrase corrected the result. All row bytes and headings remain
   source-verbatim, without silently accepting other plan terms.

A **predeclared private twenty-case** original-grounded retrieval study
uses **seven** exact original n8n sources and manual source-claim labels:
**twelve supported claims, six refuted claims and two deliberately
unsupported questions**. For each answerable case, the reviewer chose
one exact original-publisher phrase, bounded by the approved original
full SHA and byte coordinates **before** scoring retrieval. The prior
fixed **two source-excerpt slots** contained the requisite publisher
quote in **17/18** answerable cases, including **12/12 supported** and
**5/6 refuted**. The new **additional opt-in table** channel supplies
that one missing source quote: **18/18** in this deliberately narrow
sample, without changing those two normal passages, native ranks or
publisher citations. The unsupported questions have no manually
attested answer quote; merely retrieving other words from an original
**never establishes** an answer. These twenty manual examples were
chosen after inspecting official sources, and are **not** a pristine
independent unseen holdout. The table channel adds extra evidence beyond
the two passage slots, so the observed quote recovery is **not an
identical-budget top-two passage ranking gain**. Nor does including a
quoted original cell prove that a generated answer interprets negation,
version scope, plan eligibility or scientific claims correctly.

Reproduce a current source-verified example without changing live search:

```sh
cd /data/src/github/KnowMoreDiRT
PYTHONPATH=src /data/var/knowmoredirt/semantic-venv/bin/python \
  scripts/raspi_hierarchical_library.py find-source-evidence \
  'Is the AI Transform node available to self-hosted n8n workflows?' \
  --documents 3 --passages 2 --snippet-characters 850 \
  --inspect-original-tables

PYTHONPATH=src /data/var/knowmoredirt/semantic-venv/bin/python \
  scripts/raspi_hierarchical_library.py source-evidence \
  'Which MaterialX PBR symbol represents the unit surface normal vector at p?' \
  --original-path 'graphics/materials-color/materialx-pbr-spec.md' \
  --top 2 --inspect-original-tables
```

The existing default source-evidence path is unchanged without the
explicit flag. Ordinary folder, grep, native semantic result ranking,
HTML normalized article evidence, PDF **physical publisher page** checks
and background indexing continue independently. Do not extend this
source-verbatim Markdown table candidate policy to silent source
summarization or automatic fact claims without stronger independent
negative-scope and edition/version evaluation. This is a private,
working **retrieval and original-source quoting** improvement, not an
authorized production PDF/HTML generation cutover; all seven cross-store
reader/writer migration blockers remain.


#### Postpublication source-row semantic ambiguity: incident radiance is not the incident direction

A new, distinct **MaterialX** source question was posed *after* the
first table-row implementation and Git publication: **“Which MaterialX
PBR symbol denotes incident radiance at p along direction omega_i?”**
The initial method returned the publisher's real `$\omega_i$` row,
which defines the **incident direction**, instead of `$L_i$`, which
identifies the **incident radiance** measured along that direction. Both
rows have correct publisher literal-byte SHA and the same verified
*Symbols* section, but their **meaning differs**. This is a practical
example of why a valid source row is not automatically a correct answer.
The failure was reproduced in a checked-in adversarial synthetic source
test **before correction** and rechecked on the actual approved original.

The optional table candidate selector now separately scores contiguous
**informative query phrases within publisher definition/description
columns**, keeping the text physically scoped to each individual cell.
This supports an explicit question asking for *incident radiance along
direction omega_i* rather than incorrectly promoting the direction
`omega_i` solely because that identifier appears in the question.
Crucially, only columns whose genuine publisher headings indicate
*Description*, *Definition*, *Meaning*, *Details* or *Use when...* get
this descriptive-phrase treatment. Other facts such as a n8n plan's
*Available on* column are **not** promoted ahead of a strongly named
row subject just because the user mentions a platform type. Source
column meaning and original table section continue to be preserved.

Post-fix **five source-verified actual original lookups** passed:

* n8n *AI Transform node* row → `Cloud only`, not an adjacent Code
  node's `Cloud and Self-hosted` entry;
* n8n *Self-hosted Business* plan → three parallel test cases, not the
  nearby Community/Cloud Pro one sequential case;
* MaterialX *unit surface normal mathematical symbol* → `$n$`, not
  an unrelated `normal` shader input port;
* MaterialX *incident radiance along incident direction omega_i* →
  **`$L_i$`** (`Incident radiance at $p$ along direction $\omega_i$`),
  not the separate direction symbol `$\omega_i$`;
* MaterialX *measured_edf normal input port* → `normal`, type `vector3`,
  default `Nworld`, correctly kept distinct from the scattering equation
  symbol table.

The first-party original publisher SHA, current derivative SHA, byte
ranges for the exact selected row/header, unique native FileDB ID and
final current source-map generation check remained valid in every case.
The **same exact twenty predeclared n8n source-quotation cases** were
re-run after the correction: the old two ordinary passage slots remain
**17/18** for exact manually chosen source quotes, and the optional
additional table channel still makes **18/18** available with no new
answer quote for the two unsupported questions. The manual annotations
are not a pristine semantic evaluation and **no automatic factual
entailment verdict is claimed**; richer human/adjudicated semantic cases
are required before changing default search ranking, source-answer
synthesis or owner-facing GUI behavior.


### November 24 — exact source-unique Markdown row and column field lookup with abstention

The existing opt-in `--inspect-original-tables` restores complete
publisher table rows and column headings, but still leaves the consumer
with a dangerous choice when multiple tables contain the same row name.
KnowMoreDiRT now also supports explicit `--table-subject` and
`--table-column` arguments for **literal** first-column row identifiers
and **exact** publisher column headings. The optional
`--table-exact-section` argument disambiguates the source table when
several table rows share the exact same identifier and column. Both
`source-evidence` (a known original) and `find-source-evidence` (the
previously native-ranked approved originals) accept the flags **only
along with** `--inspect-original-tables`. Normal source passages and
native document ranks are **unchanged**. The operation is read-only and
never acquires additional sources or calls a generative model.

A resolved cell is explicitly marked `status=resolved`, comes from
**one unique** current first-party original source row, and includes
the **literal publisher cell value**, exact original **cell, row and
column-header byte spans and their SHA**, full original SHA, full
current derivative SHA, unique native FileDB original ID and source
section heading. The normal KMD final revision guard still rejects a
source-map/native-ID change after the evidence is assembled. Duplicate
identical table header text is supported because the actual physical
header and row byte spans in the SHA-attested original disambiguate the
source, rather than inventing a unique derivative-header occurrence.
A nonunique extracted **row** span instead fails closed. The exact
publisher cell is a **quotation**, not a verified semantic conclusion,
current external product subscription entitlement or model-generated
answer. A user asking whether a plan supports something must still
interpret the original publisher's field and its dated edition correctly.

This deliberately conservative lookup returns one of three outcomes:

* `resolved`: exactly one publisher row/column value matched, with a
  byte-for-byte verified cell and original source citation.
* `not_found`: the literal subject+column is absent from the **named
  original**; for the native discovery command, the result is explicitly
  **not_found_in_selected_native_originals**, NOT a claim that the full
  archive contains no answer. No artificial cell is emitted.
* `ambiguous`: the same literal subject+column exists in two publisher
  tables or distinct native-ranked originals, or another source has
  multiple matching rows. Even the highest-ranked native document
  **does not automatically win**. The unique-cell field is null; the
  caller must constrain the original publisher and exact section.

Ordinary English subject names, such as *Self-hosted Business*, may be
casefold-matched; **mathematical and code identifiers preserve exact
case**. A real MaterialX regression revealed that treating LaTeX
`$\omega_i$` and `$\Omega_i$` as case-insensitive confused a
**single incident direction** with the **hemisphere of incident
directions**. The bug was found during frozen real-source evaluation,
then corrected with adversarial tests. `$L_i$` and `$l_i$`, and code
identifiers such as `` `dataValue` `` and `` `DataValue` ``, also remain
distinct. No fuzzy acronym, alias, stemming or cross-document guess is
accepted as a literal source identity.

The first-party n8n and MaterialX original corpus was checked with a
**manual, source-revision-frozen sixteen-case** audit: **eleven uniquely
resolved exact fields, one ambiguous field, four truly missing literal
fields**. Three approved full original SHA versions, three current
derivative versions and their native FileDB IDs were independently
rechecked. The original cells, table headers and rows were rehashed
from actual publisher UTF-8 bytes; no source or index was modified.
The same-source ambiguity example is MaterialX's **`normal`** input:
**`conical_edf`** and **`measured_edf`** both use the label and column
*Default*. Without an exact section the result is **ambiguous**, with no
chosen field; with an explicit exact section the publisher result is
**`Nworld`** for *measured_edf*. An invented n8n *Teleportation engine*
or price column yields **not_found**, not hallucinated support.

Both commands also support literal cell resolution with existing native
SHA-ranked original discovery; if multiple selected documents match, the
result is **ambiguous**, independent of their native rank. The private
query audit stores **status and counts only**, never the returned
original cell, source prose or arbitrary model completion. The
original selected source passages, link navigation, folder/grep search,
semantic rank, source map, publisher originals and background indexers
remain unchanged. When flags are omitted, no original-cell lookup runs.

Example exact first-party original:

```sh
cd /data/src/github/KnowMoreDiRT
PYTHONPATH=src /data/var/knowmoredirt/semantic-venv/bin/python \
  scripts/raspi_hierarchical_library.py source-evidence \
  'Is the AI Transform node available to self-hosted n8n workflows?' \
  --original-path 'money-and-business/operational-ai-money-2026/automation-agents-and-reliable-workflows/expressions-versus-data-nodes-4bf00f99a.md' \
  --top 2 --inspect-original-tables \
  --table-subject 'AI Transform node' --table-column 'Available on'

PYTHONPATH=src /data/var/knowmoredirt/semantic-venv/bin/python \
  scripts/raspi_hierarchical_library.py find-source-evidence \
  'Is the AI Transform node available to self-hosted n8n workflows?' \
  --documents 3 --passages 2 --inspect-original-tables \
  --table-subject 'AI Transform node' --table-column 'Available on'
```

The archived publisher's *AI Transform node* row explicitly states
`Available on: Cloud only`; the *Code node* row states
`Cloud and Self-hosted`. These are two separate **source statements**,
not inferred eligibility or present-day terms. The ability to retrieve
one exact cell has not established a general cross-document
fact-checker, a semantic truth/contradiction classifier, a better
identical-result-budget native rank, or the right to migrate live
corrected PDF/HTML source generations. Those remain separate research
and engineering tasks, with the seven live cutover blockers unchanged.


### November 24 — literal assertion checks against an exact original publisher cell (NVIDIA and ADK)

The original-source table lookup from the preceding section recovers an
exact archived publisher field, but it did **not** directly compare a
separately submitted literal assertion against that field. A private,
read-only, opt-in `--expect-table-value` option now works in BOTH existing
`raspi_hierarchical_library.py source-evidence` and
`find-source-evidence` commands, **only** when paired with
`--inspect-original-tables`, an exact `--table-subject` and an exact
`--table-column`. The caller may specify `--table-exact-section` where
several original tables share a literal row subject. The expected value
must be one bounded nonempty line without control characters; it is not
silently edited, uppercased, translated, synonymized or coerced.

The exact publisher-cell source reader must first verify **current
complete original and extracted derivative SHA**, the unique native
FileDB identity, the literal publisher header/row/**cell** byte ranges
and their SHA, the original source section and a current source-map
revision check. The optional comparison only consumes that exact
**same-call verified original** field. It cannot follow unapproved
URLs, alter native semantic ranks, replace source excerpts, execute
agent instructions found within a source document or write original
source/index data. The private query audit records the result's
**status and counts only**: neither the claimed literal nor the
publisher source field text appears in it. The normal option-free
query returns exactly its existing ranked sources and excerpts.

Four explicit outcomes prevent turning a plausible source quote into a
fabricated factual assertion:

* `literal_match` — the user-supplied string is **exactly equal** to
  one uniquely selected, physically source-verified original table
  cell; this is not verification of current external product policy.
* `literal_different` — the asserted characters **differ** from that
  one original cell; this does **not** prove that a differently worded
  semantic claim is false. For example, synonyms or date/edition
  differences may change the literal string without changing the
  meaning. A model must not auto-label this result `false`.
* `source_ambiguous_no_literal_comparison` — multiple original rows
  or native-selected original documents match, and **no value** may be
  chosen because it ranked first. Constrain the original and section.
* `source_not_found_no_literal_comparison` — no exact matching field
  exists in the inspected original(s). This does **not** mean the
  entire Knowledge Library has no answer. No fabricated value appears.

A genuine **native discovery bug** was found in the first real
cross-domain evaluation. For the question *“Does NVIDIA cuOpt Server
support QP quadratic programming?”*, native document search ranked
the relevant approved NVIDIA cuOpt Server agent skill **second**.
However, its Markdown table was initially skipped as though it did not
exist, because the acquisition catalog correctly calls these
third-party approved `SKILL.md` originals **kind=`skill`**, not
`kind=text`. The exact-cell reader now accepts both *text* and *skill*
when the **current approved source path really ends in `.md` or
`.markdown`**. HTML, PDF and unknown source kinds remain excluded.
The source's literal original table row says **`QP → ✗`**. Comparing
the assertion `✓` now produces `literal_different` with its
independently verified physical original field, instead of the old
incorrect `not_found_in_selected_native_originals`. All four
previously native-ranked documents and original passage results were
shown to remain **byte-for-byte identical** before/after this fix.
These symbols describe the **archived publisher skill snapshot**, not
a separate fresh verification of NVIDIA's present-day API.

A **predeclared source-literal evaluation** was frozen *before* adding
the comparison helper, using **five distinct approved first-party
NVIDIA and Google ADK reference originals** that had not been in the
preceding n8n/MaterialX source-row checks. It contains **41 manually
selected field assertions**: **27 exact literal matches, nine
intentionally different literal strings, five nonexistent literal
fields**. The original source files, completed acquisition receipts
and publisher URLs were checked independently; the expected literal
fields were confirmed by a **separate minimal physical Markdown-line
reader**, not the production table parser under evaluation. Every
case was subsequently run through full original and derived SHA,
original field-byte positions, native FileDB identity and the final
source revision guard. All **41/41** produced their predeclared
literal-comparison status, with no fabricated answer for the five
missing fields. These are new **cross-domain source examples** relative
to the previous n8n and MaterialX corpus, but they were chosen after
reading the documents and are **not** an unseen blinded semantic
entailment holdout. In particular, nine different strings are **not
nine proven false factual claims**.

Those five approved source originals cover NVIDIA cuOpt routing
troubleshooting, cuOpt Server routing/LP/MILP/QP support and REST API
field names, Google ADK user-simulation expert/novice/evaluator
personas, runtime cancellation behavior and service-memory
capabilities. For example, the archived ADK novice persona has
`Correct Agent Inaccuracies: No`, whereas the expert persona has
`Correct Agent Inaccuracies: Yes`; exact field comparison can show
that distinction without implementing a speculative semantic
interpreter. Any source instructions, shell commands or model prompts
inside these archived documents are **untrusted publisher text**, not
instructions to the retrieving agent.

Reproducible already-approved local source examples:

```sh
cd /data/src/github/KnowMoreDiRT
PYTHONPATH=src /data/var/knowmoredirt/semantic-venv/bin/python \
  scripts/raspi_hierarchical_library.py source-evidence \
  'Does NVIDIA cuOpt Server solve quadratic programming?' \
  --original-path 'skills/nvidia/cuopt-server-api-python-3f546f3/SKILL.md' \
  --top 2 --inspect-original-tables \
  --table-subject QP --table-column Supported \
  --expect-table-value '✓'

PYTHONPATH=src /data/var/knowmoredirt/semantic-venv/bin/python \
  scripts/raspi_hierarchical_library.py find-source-evidence \
  'Does NVIDIA cuOpt Server support QP quadratic programming?' \
  --documents 4 --passages 2 --inspect-original-tables \
  --table-subject QP --table-column Supported \
  --expect-table-value '✓'
```

Both will show a literal difference when the selected current approved
publisher original contains a uniquely verified **`✗`** value. The
comparison is a **source-bound literal assertion aid**, not a general
truth/falsehood adjudicator, current online standards checker, model
hallucination detector, or permission to publish sources. Before any
model-generated answer is called correct, independent evaluation must
check claim scope, meaning, negation, temporal applicability and
multiple potentially conflicting first-party originals. No source
acquisition, native reranking, live source-generation cutover or
indexer stoppage was authorized or performed by these additions.

### Possible bibliography context in verified PDF excerpts (November 25, 2026)

A physically verified passage can quote a paper's bibliography rather than
its own experiments. In a real material-transfer topical query, the second
returned excerpt was a bibliography. The existing `--assess-passage-utility`
option on `source-evidence` and `find-source-evidence` now supplies a separate
`possible_bibliography_context` boolean and, when observed, the structural flag
`publisher_bibliography_style_entries_require_separate_claim_source`. The
warning uses bounded numbered/author-entry patterns, year and publication
metadata cues, and stronger evidence for single references in mixed-column
passages. It runs only on an excerpt with a verified physical publisher PDF
page. The result is an observation about the provided excerpt, not about the
whole page or document. Absence of a warning is not proof of original findings.

Bibliographic content can be useful (particularly for reference searches), and
mixed passages may contain substantive results. Neither the excerpt, source
rank, original/derived revision checks, nor page anchors are changed. It is
not an automatic rejection, relevance score, or factual entailment decision.
The existing navigation/prompt-only diagnostic retains its meaning. Private
query-audit records add only the aggregate `bibliography_context_candidates`
count, not the matched names, text, or publication identifiers. Default
queries without the optional flag retain their previous behavior.

Evaluation used already-approved local publisher originals. All 89 excerpts
across sixteen papers were independently rechecked against original SHA,
Poppler physical-page SHA and normalized page span. The final rules recognized
all 34 positive and 31 negative tuning examples across twelve papers, but
these were used to refine the rules and are not a held-out estimate. On a
separate final set of four different papers (labels frozen before scoring),
eight of eleven reference-containing excerpts were flagged, and none of
thirteen negative excerpts were flagged. This is a small, deliberately
reference-enriched, manually labeled sample, not an unbiased estimate of
corpus-wide precision/recall. Three clipped or mixed reference passages were
missed. The flag must therefore remain advisory and optional. Named-original
and direct live A/B comparisons retained identical returned source passages
and ranks; a named reference query produced two warnings, and the original
material-transfer topical query produced one direct warning. No source
acquisition, alternate index, model generation or live generation cutover
was required.

Example using the existing approved ICCV paper:

```sh
cd /data/src/github/KnowMoreDiRT
PYTHONPATH=src /data/var/knowmoredirt/semantic-venv/bin/python \
  scripts/raspi_hierarchical_library.py source-evidence \
  'material transfer geometry illumination reflectance' \
  --original-path 'graphics/iccv2025-original-published-material_photorealistic_pbr/Huang_MaTe_Images_Are_All_You_Need_for_Material_Transfer_via_ICCV_2025_paper.pdf' \
  --top 2 --snippet-characters 900 --assess-passage-utility
```

Private evaluation evidence is in the existing Raspi search-audit directory
under the `20261125-wave101-bibliography-` prefix. Publisher excerpts and
manual labels are not committed to this repository.

### Revision-bound original PDF context and alternate reading order (November 25, 2026)

The handoff from native source evidence to physical PDF page context now
binds both complete original and extracted-source revision digests, in
addition to the physical page text digest. A different original revision
can preserve the exact same center page while changing its other pages;
a changed extracted-source revision can also leave this page unchanged.
Previously both conditions could pass the page-only handoff. Both now fail
closed, even when the returned page digest is identical. This is a
cross-operator revision check, not atomic cross-store generation pinning.

The read-only `raspi_original_pdf_page_neighbors.py` and
`raspi_source_evidence_pdf_neighbors.py` operators also accept
`--include-center-reading-order`. The optional center-page projection uses
Poppler's default reading order on the same open, hashed physical original
file and under the same bounded extraction deadline. It retains the
existing layout proof, exact quotation offsets, and neighbor pages as
separate fields. It supplies its own text and digest, and explicitly
indicates whether the required layout excerpt also occurs contiguously
in the alternate projection. The handoff rejects a different center page,
layout digest, changed alternate text, or inconsistent contiguity claim.
Unavailable or oversized alternate text is explicitly marked unavailable;
no silently clipped text is presented as the full page. This is not a
visual reading-order proof or factual claim verifier. The default remains
layout-only, and neither the native retrieval ranks nor the live index
are modified.

For example, to obtain original page context for an existing approved PDF:

```sh
cd /data/src/github/KnowMoreDiRT
PYTHONPATH=src /data/var/knowmoredirt/semantic-venv/bin/python \
  scripts/raspi_source_evidence_pdf_neighbors.py \
  'PRAC subspace rank for linear and non-linear layers pre-training' \
  --original-path 'ai-agents/retrieval-language-and-memory/2026-prac-principal-random-subspace-for-llm-activation-compression-and-memory-effic-1f0d96954c.pdf' \
  --neighbor-radius 0 --include-center-reading-order
```

A live A/B comparison of this query retained the exact original revision,
excerpt hash and layout page result; the alternate page digest matched an
independent original-file extraction. In a small four-paper test, twelve
source-scoped claim anchors were frozen before search. All twelve were
recovered by explicitly requested physical-page reading-order extraction;
nine of these were not contiguous in the normalized layout projection.
Several claim cases share the same evidence phrase. This is context access
at a known physical location, not a retrieval recall measurement. In the
separate retrieval evaluation (24 direct queries, 12 claims paired with
primary and unrelated secondary originals), 53 returned excerpts were
independently re-attested across 26 physical pages. The preselected primary
page was among the first three results in 11 of 12 cases and first in seven.
Manual review found four supporting and two contradicting results with
sufficient claim-specific text, two partial contradiction contexts and
four cases with insufficient scope-matched evidence. None of the secondary
results established the requested claim; three returned no excerpts.
These small, manually judged source-specific results are neither a blinded
corpus-wide evaluation nor an automated entailment score. Lack of a
returned passage never proves that the full document lacks evidence.

Private gold, query receipts, page text and manual evaluation are retained
in the existing private Raspi search-audit directory with the prefix
`20261125-wave101-claim-` or `20261125-wave101-cross-original-claim-`.
No publisher page contents are committed. The next remaining problems
include scope-aware semantic adjudication, evidence clipping, and safe
immutable generation pinning across the live stores.

#### Expand an exact previously reviewed ranked source passage

The source-evidence context operator defaults to the first physically verified
passage. A lower-ranked result may contain the relevant limiting condition
or explicit contradiction. An agent can now select a specific returned rank
(one through three) using `--passage-rank`, together with the required
`--expected-original-sha256`, `--expected-derived-sha256`,
`--expected-excerpt-sha256`, `--expected-physical-page`, and
`--expected-page-text-sha256`. Obtain those values from a matching
`source-evidence` query with `--top 3 --snippet-characters 500`. The context
operator repeats that exact bounded search and rejects a changed source or
derivative, changed excerpt at the selected rank, missing rank, unverified
physical page, or a different physical page or layout digest. Identical text
repeated on another page cannot be substituted for the reviewed context. It will not silently substitute another result. The existing
first-verified default and native ranking are unchanged. The response adds
`selected_source_excerpt_rank` and `source_excerpt_selection`; the old
`first_verified_source_excerpt_rank` is null in explicit selection mode
because the selected result need not be the first verified one.

Live checks expanded both partial contradiction cases from the earlier
four-paper evaluation. The PRAC claim selected rank one, physical page seven,
recovering the actual unequal linear/non-linear layer ranks. The SegmentDreamer
claim selected rank three, physical page two, recovering the explicit
subtrajectory partition statement. Both selected exact prior excerpt identities
and both returned the independently hashed full center reading-order text
containing the predeclared evidence anchor. A live negative check supplied a
wrong expected excerpt digest at rank three and returned no context.
The final page-bound version also rejected a wrong prior physical page even
when the original, derivative, excerpt and layout digests matched.
These are manually selected, source-bound evidence recoveries, not improved
ranking scores or an autonomous claim classifier. No original source contents
or private search receipts are included in the repository.

#### Private source-scoped claim evaluation (experimental, not deployed)

`scripts/raspi_attest_source_claim_dataset.py` independently checks frozen
private evidence spans against their approved original PDF bytes, extracted
derivative, exact physical page, reading-order digest, source metadata,
normalized offsets, and current native identities. Source map and native
identities are rechecked after all page reads. Synthetic fixtures are explicitly
not publisher-attested. A source hash does not establish a semantic gold label.

`scripts/benchmark_source_claim_assessment.py` accepts a private frozen
dataset, an already authorized loopback-only model port and exact model ID,
a bounded deadline, and explicit experimental variants. It does not start a
model, contact a remote provider, write live indexes, or expose gold answers
to the model. Invalid replies remain distinct from valid insufficient-evidence
judgments. The pairwise variant asks independently about each supplied source,
then combines validated relations deterministically; this does not make the
model's individual judgments authoritative. Private exclusive-created
checkpoints and final receipts record completed cases and exact paired scores.

On 2026-11-25, 44 manually labeled cases were frozen before model inference,
including ten attested spans from eight approved original PDFs and twelve
synthetic spans. With the existing Qwen2.5-VL-3B-Instruct Q4_K_M model in
text-only mode, the baseline joint prompt obtained 11/44 correct labels
(10/44 with exact decisive source IDs). A richer scope-aware joint prompt
obtained 1/44. The independently judged source variant, developed after
reviewing the joint results, obtained 30/44 correct labels and exact source IDs,
but made four unsupported decisive judgments and three polarity reversals.
Its valid output rate was 44/44; this is not an accuracy guarantee. The
pairwise run reused the frozen cases, not an independent blind holdout.
The initial Qwen2.5-7B CPU trial timed out before a paired comparison could
be completed. No live fact-checking classifier is enabled by these results.

Reproduce only under an authorized local model lease using the private
dataset; first run the attestation command with a new private output path,
then benchmark with `--dataset`, `--output`, `--port`, `--model`,
`--variant baseline --variant scope` or `--variant pairwise`.
Exact private data, source attestations, model run and checkpoint receipts
are in the Raspi search-audit directory under prefix
`20261125-wave102-`. Publisher text and private gold/receipts are not
committed to Git. Model candidates are experimental and should not be
presented as verified facts. The live immutable generation and publication
gates remain unchanged.

#### Retrieve intact evidence within an explicitly verified physical page

`scripts/raspi_original_pdf_page_passages.py` ranks bounded contiguous
sentence windows within a fresh reading-order extraction of one approved
original PDF page. This is an opt-in read-only operator for an already known
physical page, not a replacement for native document ranking.

Example (use the exact approved original path and verified page):

```sh
python scripts/raspi_original_pdf_page_passages.py 'query' \
    --original-path 'exact/approved/original.pdf' --page 2 \
    --top 3 --max-characters 900 --include-inflection-matches
```

The operator independently checks the original/derivative revision, current
physical page and both extraction digests. Supply all three expected
original, derivative and reading-order-page hashes if using a previously
reviewed source identity. It rejects a changed revision or native original ID
instead of substituting a result. Source and native identities are rechecked
after ranking. Each result is an exact contiguous span of whitespace-normalized
reading-order text with its own offset and hash. Adjacent sentence context is
preserved within the character budget. Sentence splitting is heuristic;
overlong units are explicitly marked as partial. Page boundaries are not
silently crossed. Numeric literals preserve sign and percent versus
percentage-point units. Optional lower-weight ordinary-English inflection
matching does not rewrite the returned evidence. Neither lexical ranking nor
the alternate extraction proves visual reading order or a factual claim.

Final 2026-11-25 regression: frozen, manually selected anchors for 28 queries
on 14 known original pages, using the same verified normalized reading-order
page for all three methods and 900 characters per candidate. The old
fixed-length excerpt chooser recovered 18/28 full anchors at rank one;
sentence windows recovered 26/28 at rank one and 27/28 within three;
optional Porter matching recovered 26/28 at rank one and 28/28 within three.
No previous first-rank success regressed. On twelve queries from six
different papers whose anchors were frozen before ranking that cohort,
first-rank recovery improved from 8/12 to 12/12. The final operator was also
run directly with exact expected hashes: the MONDAY manual-annotation
contradiction was recovered intact at rank two and its anchor hash verified.
These are known-page, non-blinded retrieval tests, not an end-to-end
corpus-recall or fact-verification benchmark.

Frozen anchor receipts, reproduction scripts, final regression and exact
operator receipts remain in the private Raspi search-audit directory under
prefix `20261125-wave102-`. No publisher text is committed. Ordinary
folder/grep access and existing native retrieval remain unchanged. The
operator does not pin an immutable cross-store generation and does not
authorize a live source-generation migration.

#### Expand the selected verified PDF pages without changing document ranking

Both existing source-evidence commands now accept the optional
`--expand-pdf-pages N` (zero by default; at most five). For example:

```sh
python scripts/raspi_hierarchical_library.py find-source-evidence 'query' \
    --documents 5 --passages 3 --snippet-characters 900 --expand-pdf-pages 5
```

For the exact-original `source-evidence` command, use its existing
`--original-path` and at most five passages with this option. Only already
verified physical pages can be expanded. Distinct pages are allocated
round-robin in original rank order; every repeated anchor on a selected page
must agree with the same exact layout extraction and offset. Up to three
independently checked reading-order windows per page are returned separately
from the unchanged original passages. Unavailable or noncontiguous alternate
extractions are explicit; an identity mismatch fails closed. The normal
final revision guard runs after expansion. The audit excludes publisher
prose. This option cannot be combined with a continued section or private
corrected-page overlay.

In the 2026-11-25 wave103 private replay, 28 questions on 14 named originals
were evaluated with the physical page withheld. Existing top-five snippets
contained five full frozen reading-order anchors. Expansion recovered 26
(24 using only the first window of each of up to five pages); the first
window on the first selected page recovered 17 versus four with the original
first snippet. Two target pages were never selected by the earlier page
locator. All earlier full-anchor successes were retained. These are manually
selected anchor containment measurements, not factual-answer accuracy.
The fifteen-window maximum is a larger evidence budget than five snippets,
and layout and reading-order text may not have identical contiguity.

Seven additional source-linked speech research questions, with neither the
original nor the page provided, reached three of seven preselected documents
in the native top five. The expansion found two complete frozen anchors,
versus zero in the original snippets. The existing independent first-unit
lexical channels both retrieved all seven target documents at rank one on
this particular frozen set. Their RAM-only scan covered all 17,715 complete
source-map records. Neither result demonstrates generally superior ranking;
the questions are manually selected, and other relevant originals are not
exhaustively labeled. The native default ranking and all index data remain
unchanged. Exact private query, source and evaluation receipts reside under
`search-audit/20261125-wave103-evaluation/` on Raspi.

#### Independent original-PDF discovery and lexical-result evidence

The read-only `raspi_original_pdf_document_passages.py` operator can locate
evidence across an already known approved original PDF without requiring a
derivative page label or known physical page. Bounds: 32 MiB original,
64 physical pages, 2 MiB extracted text, five selected pages, three
sentence windows per page. Its complete reading-order text scan is only a
locator. It independently rereads the selected physical pages and checks
the exact reading-order page digests and normalized passage offsets.
If extracted form-feed counts disagree with the actual PDF page count,
it requests each numbered physical page explicitly within the bounded
deadline rather than guessing boundaries. Unavailable pages and complete
scan failures are explicit. Source revisions and native original identities
are rechecked after all reads; no live index is written.

```sh
python scripts/raspi_original_pdf_document_passages.py 'question' \
    --original-path 'exact/approved/paper.pdf' --top-pages 5 --snippet-characters 900
```

The separate `raspi_firstunit_selected_pdf_evidence.py` operator accepts
the exact SHA-256 of a private receipt from
`raspi_publisher_firstunit_discovery.py`. It preserves independent literal
and Porter lexical ranks and expands distinct approved PDF candidates
round-robin without fusing scores or changing native semantic rank.
It reattests each source before expansion and collectively rechecks all
selected source and native identities afterward. It writes exclusively to
a new private search-audit output, not to Git.

```sh
python scripts/raspi_firstunit_selected_pdf_evidence.py \
    --lexical-receipt /data/var/knowmoredirt/knowledge-library/search-audit/PRIVATE_RECEIPT.json \
    --expected-sha256 EXACT_RECEIPT_SHA256 --originals-per-query 1 \
    --top-pages 5 --snippet-characters 900 \
    --output /data/var/knowmoredirt/knowledge-library/search-audit/NEW_PRIVATE_OUTPUT.json
```

The wave103 final replay found all 35 preselected physical pages when
only the original was specified, and 32 complete frozen evidence anchors.
For the earlier 28 named-original cases, the union of existing selected-page
expansion and independent whole-document search recovered all 28 anchors;
neither method alone did so. On seven speech research questions, neither
target original nor page was supplied to the lexical search. Both lexical
channels reached all seven preselected documents at rank one; verified PDF
expansion reached all seven target pages and recovered five complete anchors.
Native semantic retrieval reached three target documents in its top five
and selected-page expansion recovered two anchors. These are small,
manually selected, non-exhaustively labeled cohorts, not a general
corpus-recall or factual-answer accuracy guarantee. All results are
optional retrieval candidates, not entailment judgments. The seven
atomic live source-generation migration safeguards remain unresolved.

Private evidence and reproducible scripts are retained under
`search-audit/20261125-wave103-evaluation/`. Wave105's compatible
filesystem regression suite passed all 1092 tests, excluding the separate
pytest-only resource-safety test for which the framework was unavailable.
