# KnowMoreDiRT: bounded hybrid evidence candidates, 7 November 2026

## Scope and ownership

This is an isolated **KMD-side retrieval experiment**, not a production replacement
for the original KnowMoreDiRT filesystem database, and not an Infra ranking
deployment. Implementation is in `file_system_catalog.hybrid_search`.
The pinned Raspi production KMD worktree, dirty Infra integration, original
files, source-map, source ledger, native vectors, source versions and generation
state have not been modified. The private pilot executable and evidence live
under the existing Raspi `/data/var/knowmoredirt` runtime and the
Knowledge Library's private UX-discovery evidence directory; no book contents
or database copies are committed.

The original KMD FileDB continues to own original-file metadata and semantic
vectors. The offline Knowledge Library already has format-specific text
extraction, a source-map of exact original/derivative checksums, and a
verified-source query wrapper. The latter now supports exact matched native
passages and optional verified PDF-page context. Neither feature fixes
**global evidence discovery**, where an embedding samples only three short
sections of a long book and the native search retains one best vector per
document. The `content_representations` table still has no generated
discourse-analysis records. The separate DRT/DSPG question-answering system
is not integrated into this live knowledge-library retrieval path.

## Reusable KMD primitive

`BoundedPassageIndex` accepts caller-validated, source-linked original FileDB
entry IDs and text sections. It performs a bounded **in-memory SQLite FTS5**
search and returns each original ID, exact source path, section/page label,
original and derivative SHA-256 identifiers, exact passage SHA, and complete
passage. The source text is not normalized, and no disk-backed DB or FileDB
schema object is created. Query expressions are constructed from bounded,
escaped Unicode term tokens instead of interpreting arbitrary MATCH syntax.
Lexical matching requires more than a single incidental term for multi-term
queries, and all index/return sizes have explicit upper bounds.

`fuse_search_candidates` combines native cosine-result ranks and independently
retrieved lexical-section ranks through weighted reciprocal-rank fusion.
Unlike the original KMD native one-best-chunk-per-document search, the
fusion deduplicates by original document **but keeps several distinct matched
pages**. No cosine similarity is called a factual confidence, and every output
is explicitly labeled as a retrieval candidate pending original-revision,
derivative-revision, parent-context, and discourse-scope verification.

This is intentionally **not** wired into `search_semantic_entries` or the
live Infra wrapper. It is a core component ready for guarded integration,
not a declaration that an entire library can be searched by FTS on every
query. The system's two-operation core public API is unchanged.

## Actual Raspi read-only pilot

The UX corpus discovered and independently accepted earlier on 7 November
contains **seventy-nine completed originals**. The pilot read the actual
source-map and download ledger, joined each original to the canonical
FileDB's original `filesystem_entry_id`, independently recalculated all
seventy-nine *derived-text* file hashes, and loaded **997 nonempty page/section
passages covering 3,270,506 characters** into a temporary in-memory index.
These original-source checks rely on the earlier original SHA ledger validation;
the pilot did not rehash all binary PDFs. It used the actual complete native
semantic ranking of **17,714 original documents** for four real queries,
not a synthetic or truncated semantic ranking.

The general visual-design query placed the HCI textbook at native semantic
document rank **2,772**; with *only the seventy-nine UX originals* available
to the lexical component, the hybrid pilot placed it fourth and surfaced
relevant sections including physical PDF page **506**. On the GUI windows/icons
query, HCI moved from native rank **1,257** to pilot rank **3**, with the
graphical-user-interface chapter around physical PDF page **504**. The
Hick's-law question preserved the appropriate UX titles near the top. The
Android weather-app query still produced generic documents and was not
reliably solved. Exact top candidates are in private
`isolated_kmd_hybrid_pilot_20261107.json`.

**Important experimental limitation:** The semantic ranking searched all
17,714 indexed originals, whereas the FTS data contained just the manually
selected new UX collection. The change in UX top-ten counts is therefore
affected by the privileged lexical candidate pool and cannot be presented as
an unbiased whole-library accuracy improvement. It demonstrates that source
passage information that vectors missed is retrievable with a compact,
non-persistent lexical index. Independent all-domain positive/negative
holdouts, candidate-selection recall, authority discrimination, and calibrated
bad-source rates still require work.

A read-only source-title/path overlap probe also placed this HCI textbook
around position **1,014**. Selecting FTS source documents only by filename and
title relevance would therefore repeat the same loss of recall. A bounded,
generic source discovery policy must include topic-family/neighbor expansion,
semantic candidates beyond the top handful, and model-generated query concepts
where appropriate. It must not hard-code “GUI” versus “agent” into the
retrieval implementation. The native FileDB's approximately 640-MiB storage
checkpoint also precludes an unmeasured full-book vector or unbounded FTS
rollout.

## Next integration boundaries

Keep the one canonical original FileDB, its native original identifiers,
and its compact vector policy. The next step is to test a query-time source
selection plan over *all* eligible document categories, retrieving selected
verified page/section text into bounded ephemeral FTS before fusing rankings.
If an optional persistent lexical accelerator is justified by measured query
latency, it must remain a versioned **part of the same canonical FileDB**,
with database-size accounting, migrations, rollback, source-version invalidation,
full SQLite integrity checks, and no alternate source-of-truth.

Query-time page extraction must require the current original SHA-256, current
derived-text SHA-256 and source-span/page validity already enforced by the
Infra wrapper. Passing a correctly shaped hash is not the same as verifying
the source bytes. Selected passages must be expanded to governing headers,
neighbors or discourse-linked spans; a paragraph quoted inside a dream,
hypothesis, historical report, or negative condition must not be promoted to
an unqualified assertion. DRT/DSPG integration and grounded answer synthesis
are separate acceptance targets.

The regression set contains ten new focused tests for the in-memory FTS
boundary, unsafe and adversarial paths, Unicode/exact passages, source hash
requirements, retrieval scope, output bounds, multi-page preservation,
semantic-only fallback, score interpretation, deterministic fusion, and query
injection. At this checkpoint, **ten new tests and nineteen existing native
content/schema tests pass** with system Python's unittest runner. The host
has no pytest installed in the checked interpreters. Native embedding
client tests cache successful model requests by hash, so a repeat run without an
isolated test-cache directory caused two mock-expectation failures after the
first clean pass; rerunning with a fresh KMD_MODEL_CALL_CACHE_DIR passed all
existing tests without code changes. The module is pure Python and SQLite FTS5;
production database/schema deployment is **not done**.
