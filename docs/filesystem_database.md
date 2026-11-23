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
