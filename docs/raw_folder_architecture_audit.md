# Raw-Folder Architecture Audit

## Scope

This audit covers the current KnowMoreDiRT source package, benchmark adapter scripts, tests, docs, and uncommitted changes created during the interrupted capability-parity work.

## Current System Contract

KnowMoreDiRT is a raw-folder knowledge system. The public contract remains exactly:

1. `initialize(folder_path)`
2. `question(text) -> string`

The core package reads arbitrary nested folders and raw readable text files, derives internal DSPG structures, and answers from source-grounded text. External evaluation adapters may read evaluator-specific question files and write prediction files, but they must pass only the raw folder path and question text into KMD.

## Findings

### Generic components

- `scanner.py`, `ingest.py`, `store.py`, `relations.py`, `query.py`, and `engine.py` implement raw-folder scanning, filesystem metadata capture, sentence/chunk storage, mentions, referents, contexts, frames, relations, and source-grounded answers.
- Generic support for identifiers, URLs, file paths, dates, status/state, people, organizations/accounts, claims, negation, and context is appropriate for a raw-text knowledge system.
- The benchmark runner under `scripts/benchmarks/` is adapter glue only. It reads evaluator question IDs and writes scorer files, but core KMD receives only the folder path and question text.

### Suspicious or benchmark-shaped components

- The uncommitted `bounded_dspg.py` port used legacy intent names such as specific PR/ticket/customer variants. Even though it did not contain exact benchmark entities or gold data, the internal vocabulary was too close to external question-family categories.
- `legacy_drt_path.py` used older role/intent names including specific reference subtypes. This was refactored to generic intents: `role_lookup`, `reference_lookup`, `url_lookup`, `file_lookup`, `state_lookup`, `context_lookup`, `identity_lookup`, `grouped_search`, and `unknown`.
- The public docs contained a migration report with local run paths and score-recovery language. That was removed from the official docs surface.

## Changes Made

- Stopped the interrupted external evaluation process and did not resume it.
- Removed the uncommitted `bounded_dspg.py` core module rather than preserving a benchmark-shaped bounded executor.
- Removed engine integration with `bounded_dspg.py`.
- Refactored the optional local model query-plan vocabulary to generic raw-folder knowledge operations.
- Strengthened architecture tests so core source fails on external-evaluation markers, hidden-label terms, old prepared-input markers, and legacy question-family intent names.
- Deleted migration-result docs from `docs/` and replaced architecture/evaluation wording with generic external-evaluation language.
- Kept benchmark adapter resume/checkpoint behavior outside core because it is operational glue and does not affect answer logic.

## Deliberately Rejected

- No benchmark-run result, score, question family, hidden label, or external evaluator field was added to core behavior.
- No exact question IDs, entities, products, expected answers, or source wrapper markers were added to core behavior.
- The interrupted external run artifacts were not committed and are not used as tests.

## Remaining Concerns

- The deterministic engine still contains broad domain words such as customer, employee, ticket, and PR-like identifier handling. These are currently used as generic text-domain concepts and fixture coverage terms, not evaluator-specific families. They should be progressively generalized to actor, organization, case, reference, and identifier terminology where doing so does not remove useful general capability.
- The optional local-model path remains developmental and disabled by default.
- Self-written fixture scores remain regression checks, not proof of broad generalization.
