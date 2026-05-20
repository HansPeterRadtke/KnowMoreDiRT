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
- **Relation resolution**: store label/value relations, identifiers, URLs, file-like references, active/passive events, state/status statements, negation, meaning/plural relations, aliases, table cells, and timestamped facts.
- **Context handling**: preserve asserted, negated, reported, quoted, believed, fictional/dream-like, uncertain, and low-semantic-quality contexts as source-grounded records.
- **Temporal reasoning**: store ordered state events and answer current/final-state questions by target-anchored latest evidence.
- **Identifier resolution**: treat references, codes, URLs, paths, hashes, and prefixed IDs as generic identifiers.
- **Role assignment**: answer actor/role questions through generic role labels, active/passive event patterns, and source-grounded relations.
- **Local-model bounded reasoning**: optional localhost-only query planning can produce constrained generic plans, but execution remains source-grounded and independent of external evaluation harnesses.

## Removed Or Refactored Concepts

The core package no longer contains dataset-shaped or old external-evaluation vocabulary. The cleanup removed or refactored:

- external benchmark/process terminology from core modules,
- hidden-label and scoring terms from core modules,
- old prepared-input and wrapper markers from core modules,
- old reference subtype intents such as `which_pr`, `which_ticket`, and similar family-shaped names,
- old domain-shaped role terms in core routing,
- the scratch `bounded_dspg.py` migration module because it preserved old vocabulary instead of clean generic primitives,
- `legacy_drt_path.py`, renamed and refactored as `model_planner.py` with generic intents only.

## Generic Architecture Changes

- The optional local planner now exposes generic intents only: `role_lookup`, `reference_lookup`, `url_lookup`, `file_lookup`, `state_lookup`, `context_lookup`, `identity_lookup`, `grouped_search`, and `unknown`.
- Core answer logic no longer uses old prepared-input, scorer, family, or external benchmark terms.
- Identifier handling is expressed as generic reference/case/code lookup.
- Person-name cleanup now strips generic role-prefix nouns instead of enumerating domain-specific roles.
- Architecture tests now fail if core source contains external-evaluation markers, old wrapper markers, old family-shaped intent names, or core vocabulary such as exact dataset family terms.

## Boundary Decision

Adapter scripts may remain outside `src/knowmoredirt` for operational evaluation glue. They must not influence core reasoning. The only acceptable data passed into KMD is a folder path and question text.

## Remaining Concerns

- Some self-written fixtures still contain legacy software/project-management words as regression data. That is acceptable only as input text, not as core architecture vocabulary.
- KMD’s deterministic answer layer still includes many general English phrase patterns. These should continue moving toward relation-first graph execution and broader generated holdout tests.
- Optional local-model planning is generic and disabled by default; it still needs broader non-fixture validation before being treated as a mature reasoning path.
