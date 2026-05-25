# DRT-Centered Rebuild Report

Date: 2026-05-25

## Scope

This report records the current KnowMoreDiRT rebuild step toward a Kamp-style DRT architecture. HERB was not run. The public API remains `initialize(folder_path)` and `question(text) -> string`. The core still receives only raw folder contents and question text.

## DRT Design Basis

Kamp’s “A Theory of Truth and Semantic Representation” treats a discourse representation as a partial model built incrementally from discourse. A DRS introduces discourse referents and conditions, and the truth of the discourse is evaluated by whether that structure can be embedded into a model. The important architectural consequence for KMD is that question answering should not be a list of procedural handlers. It should bind variables over a grounded discourse representation: referents, conditions, contexts, temporal information, polarity, modality, and source evidence.

KMD’s engineering form is still DSPG, but DSPG is now being constrained to behave as a storage and execution substrate for DRS-like structures rather than as a hand-written answer extractor. Predicate words and relation labels remain data. They are not intent enums and do not select domain-specific code branches.

## What Changed

The bounded query executor was rebuilt around generic DRS condition binding. It now selects a bounded SQLite subgraph, loads documents, chunks, source spans, frames, frame arguments, relations, contexts, temporal edges, and metadata records, and attempts to bind an answer variable by matching target anchors, requested predicate text, context accessibility, temporal scope, broad answer type, and evidence. It no longer contains the previous large procedural answer extraction routes.

The deterministic relation extractor was reduced to universal surface-structure extraction. It now stores label/value text, JSON/object-like scalar values, table cells, identifiers, URLs, and timestamps. It no longer creates deterministic active/passive event relations, copular assertion relations, negation-as-semantics records, or person-pattern event frames. Semantic events, semantic roles, claims, reports, dreams, polarity, modality, and other discourse interpretation are expected to come from local-model DRS frame extraction with exact evidence text.

Ingestion keeps filesystem scanning, raw text chunking, natural metadata capture, source spans, mentions, referents, context carriers, deterministic surface records, and optional local-model DRS frames. Deterministic relation rows are still normalized into frames for provenance and graph mechanics, but they are treated as structural records rather than semantic interpretations.

The local model path remains isolated and localhost-only. When enabled, it is used for chunk-to-DRS frame extraction, question-to-query-frame parsing, bounded evidence answer extraction, and answer verification. Fake-model tests prove that model chunk frames are stored, model query frames are called, bounded execution uses grounded model frame arguments, verifier calls are counted, and unsupported or incompatible model answers are rejected.

The local llama.cpp endpoint was checked at `http://127.0.0.1:14829/v1/models` during this pass and reported `Qwen2.5-14B-Instruct-Q4_K_M.gguf`. No cloud API was used.

Static architecture tests were strengthened so future core changes fail if deterministic semantic event regexes such as active/passive event extraction, copular assertion extraction, person-pattern semantic extraction, or polarity-marker semantic extraction return to `relations.py`.

## What Became Pure Logic

The remaining deterministic core is intended to be infrastructure: recursive folder scan, readable text loading, chunking, exact source spans, SQLite schema/storage, stable cache keys, source-grounded metadata, lexical indexing, source-quality signals, exact surface recognizers for URLs, identifiers, timestamps, key/value text, object-like scalar values, and table cells, plus bounded graph loading, equality-style term matching, context accessibility checks, temporal ordering where temporal records exist, broad answer-type validation, aggregation over candidate bindings, and evidence grounding.

This is not complete DRT entailment yet. It is a practical vertical slice of a DRS/DSPG runtime that has removed the largest procedural semantic extraction layer and now requires the local-model DRS path to supply semantic conditions.

## Validation

Commands run:

```bash
python3 -m py_compile src/knowmoredirt/*.py tests/**/*.py scripts/evaluate_fixture.py
PYTHONPATH=src pytest -q tests/unit/test_generic_relations_and_metadata.py tests/unit/test_capability_recovery.py tests/unit/test_answer_type_and_model_extraction.py tests/unit/test_architecture_contract.py tests/smoke/test_public_api.py
PYTHONPATH=src pytest -q
PYTHONPATH=src python3 scripts/evaluate_fixture.py --corpus tests/fixtures/messy_raw_corpus --qa tests/fixtures/messy_raw_corpus_qa.json --json-out /tmp/kmd_eval_messy_drt_refactor.json
PYTHONPATH=src python3 scripts/evaluate_fixture.py --corpus tests/fixtures/broad_raw_world --qa tests/fixtures/broad_raw_world_qa.json --json-out /tmp/kmd_eval_broad_drt_refactor.json
PYTHONPATH=src python3 scripts/evaluate_fixture.py --corpus tests/fixtures/hardcore_noise --qa tests/fixtures/hardcore_noise_qa.json --json-out /tmp/kmd_eval_noise_drt_refactor.json
PYTHONPATH=src python3 scripts/evaluate_fixture.py --corpus tests/fixtures/hard_raw_reasoning --qa tests/fixtures/hard_raw_reasoning_qa.json --json-out /tmp/kmd_eval_hard_drt_refactor.json
```

Results:

- `py_compile`: passed.
- Targeted non-evaluation tests: `28 passed`.
- Fake-model tests passed for model-enabled chunk-frame extraction, model query-frame parsing, bounded frame-argument binding, verifier invocation, and grounded answer acceptance.
- Full pytest: failed only on strict fixture evaluation gates.
- Messy fixture: `18/60`.
- Broad fixture: `27/65`.
- Noise fixture: `7/8`.
- Hard fixture: `45/134`.

The strict fixture gates were not lowered. These failures are real and expected after removing procedural semantic interpretation from deterministic code. They show that the LLM-centered DRS replacement is not complete enough yet for the internal fixtures.

## Remaining Gaps

The main gap is not missing owner, reviewer, manual, customer, ticket, or other relation-specific handlers. Those must not return. The missing work is generic DRT/DSPG capability: practical cached local-model frame extraction over full fixtures, stronger model prompt reliability, progress logging for model-enabled initialization, richer context hierarchy and accessibility, stronger identity/coreference hypotheses, temporal interval/state validity, multi-hop traversal across referents and frame arguments, aggregation over variable bindings, table/object group traversal as DRS structures, and verifier-driven unknown gating.

The deterministic fallback is now intentionally shallow. That is the cost of removing hand-written semantic parsing before the local-model DRS path is mature enough to replace it. The next correct implementation step is to make model-enabled fixture evaluation practical and reproducible, then improve generic frame construction and binding until the fixtures recover through DRS semantics rather than semantic handler branches.

## HERB Status

HERB was not run.
