# Raw-Folder Architecture Audit

This audit covers the current KnowMoreDiRT core package, public docs, tests, and external-evaluation adapter boundary. It was written after stopping the interrupted external run so architecture work could return to KMD’s only valid system contract: `initialize(folder_path)` followed by `question(text)`.

## Current System

KMD is a raw-folder discourse knowledge system. The core package recursively scans readable files, stores natural filesystem metadata, chunks raw text, extracts source-grounded spans, mentions, referents, contexts, frames, temporal edges, and generic relations, and answers from bounded retrieval over those internal DSPG records.

The public API remains only:

- `initialize(folder_path)`
- `question(text) -> string`

## Generic Primitives Extracted From Old DRT Work

The old development system contained useful generic mechanisms mixed with dataset-shaped vocabulary. The mechanisms retained for KMD are:

- **Bounded retrieval**: combine lexical sentence search, referent/chunk lookup, relation/frame lookup, document-neighbor expansion, source metadata as retrieval prior, and noise downweighting.
- **Evidence ranking**: rank by anchor overlap, predicate overlap, relation/frame match, temporal/context suitability, and source quality.
- **Graph traversal**: use DSPG documents, chunks, spans, mentions, referents, contexts, frames, frame arguments, temporal edges, and relations rather than answering only from flat text.
- **Relation resolution**: store label/value relations, identifiers, URLs, file-like references, copular assertions, active/passive events, negation relations, table cells, record/object values, and timestamped facts.
- **Context handling**: preserve asserted, negated, reported, quoted, believed, fictional/dream-like, uncertain, and low-semantic-quality contexts as source-grounded records.
- **Temporal reasoning**: store ordered state events and answer current/final-state questions by target-anchored latest evidence.
- **Identifier resolution**: treat references, codes, URLs, paths, hashes, and prefixed IDs as generic identifiers.
- **Query-frame execution**: answer questions by matching requested relation text, target anchors, answer type, constraints, temporal scope, and evidence requirements against grounded DSPG records.
- **Local-model bounded reasoning**: optional localhost-only query planning can produce constrained generic plans, but execution remains source-grounded and independent of external evaluation harnesses.

## Removed Or Refactored Concepts

The core package no longer contains dataset-shaped or old external-evaluation vocabulary. The cleanup removed or refactored:

- external benchmark/process terminology from core modules,
- hidden-label and scoring terms from core modules,
- old prepared-input and wrapper markers from core modules,
- old reference subtype intents such as `which_pr`, `which_ticket`, and similar family-shaped names,
- old domain-shaped role terms in core routing,
- old intent/role enums that routed by handwritten semantic categories,
- query-planner prompts that exposed relation-specific operation names instead of generic query frames.

## Generic Architecture Changes

- The optional local planner now emits generic query frames rather than semantic intent enums.
- Core answer logic no longer uses old prepared-input, scorer, family, or external benchmark terms.
- Identifier handling is expressed as generic reference/case/code lookup.
- Answer cleanup now relies on broad answer-type compatibility and source grounding rather than enumerated role handlers.
- Architecture tests now fail if core source contains external-evaluation markers, old wrapper markers, old family-shaped intent names, or core vocabulary such as exact dataset family terms.

## Boundary Decision

Adapter scripts may remain outside `src/knowmoredirt` for operational evaluation glue. They must not influence core reasoning. The only acceptable data passed into KMD is a folder path and question text.

## Remaining Concerns

- Some self-written fixtures still contain legacy software/project-management words as regression data. That is acceptable only as input text, not as core architecture vocabulary.
- KMD’s deterministic answer layer is now relation-frame first, but accuracy regressed on self-written fixtures after removing procedural semantic handlers; future work should recover accuracy through generic extraction, traversal, aggregation, context propagation, and local-model verification.
- Optional local-model planning is generic and disabled by default; it still needs broader non-fixture validation before being treated as a mature reasoning path.
