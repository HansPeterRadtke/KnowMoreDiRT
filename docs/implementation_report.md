# KnowMoreDiRT Implementation Report

## Package Structure

- `src/knowmoredirt/__init__.py`: exports only `initialize` and `question`.
- `src/knowmoredirt/public.py`: two-function public API and module-level initialized engine.
- `src/knowmoredirt/scanner.py`: raw folder traversal, UTF-8 text reading, file metadata, SHA-256 hashing, and text-unit extraction.
- `src/knowmoredirt/store.py`: normalized SQLite DSPG store for documents, chunks, source spans, mentions, referents, contexts, frames, and frame arguments.
- `src/knowmoredirt/ingest.py`: raw-text ingestion pipeline that fills the DSPG store from arbitrary readable files.
- `src/knowmoredirt/index.py`: bounded lexical index over raw text units.
- `src/knowmoredirt/extractors.py`: generic regex extractors for URLs, IDs, emails, names, and labels.
- `src/knowmoredirt/engine.py`: KMD engine that builds DSPG records, then answers through bounded retrieval over the same raw text units.
- `src/knowmoredirt/model.py`: optional isolated local model client hook; not required for default operation.
- `src/knowmoredirt/evaluation.py`: internal fixture evaluation helpers.
- `scripts/evaluate_fixture.py`: developer evaluation command.

## Public API

The intended public API exposes only:

```python
initialize(folder_path)
question(text) -> string
```

`__all__` is exactly `['initialize', 'question']`. Internal modules/classes exist for implementation and diagnostics, but they are not the intended user-facing API.

## Package Style Findings

I searched `/data/src/github` and installed Python package metadata for CAPOV/capov examples. No local CAPOV package or installed package metadata was found. Because there was no clear HPR/CAPOV namespace pattern to follow, this implementation keeps the clean PyPI-oriented `src/knowmoredirt` package layout that was already started in the foundation commit.

## Old DRT Reference Use

Old `/data/src/github/devtests/DRT_tests` code and reports were reviewed before implementation. KMD uses the old system as architectural reference for:

- raw-folder-only public interface discipline,
- normalized DSPG records,
- SQLite persistence shape,
- exact source spans and mentions,
- local referents,
- scoped contexts,
- event/frame records,
- bounded retrieval before answer extraction,
- optional local-model isolation.

The KMD code is a clean refactor/rewrite. It does not move old scripts wholesale, does not include HERB/benchmark logic, and does not depend on prepared metadata or old DSPG runtime paths.

## Current Engine Behavior

Implemented:

- recursive raw-folder scanning,
- arbitrary filename/extension support,
- readable text ingestion only,
- natural file metadata collection: relative path, size, mtime, ctime, SHA-256,
- sentence/line chunking,
- normalized SQLite DSPG graph construction,
- sentence and mention source-span storage,
- URL/identifier/name mention extraction,
- local referent construction,
- assertion/belief/dream/fiction/allegation/report/negation context records,
- lightweight event/frame extraction with frame arguments,
- lexical retrieval,
- deterministic answer patterns for direct facts, IDs, URLs, temporal states, tables, claims, beliefs, dreams, fiction, allegations, quotes, and unanswerable cases,
- architecture tests that assert no old benchmark/prepared-input markers appear in core package code.

Not implemented yet:

- durable user-configurable database path,
- full old staged model-assisted extraction,
- full model query planner/reranker,
- confidence-calibrated graph answer selection,
- long-run progress/checkpointing,
- external benchmark adapter code.

## Tests Run

- `python3 -m py_compile src/knowmoredirt/*.py tests/**/*.py scripts/evaluate_fixture.py`: passed
- `PYTHONPATH=src pytest -q`: `15 passed`
- `PYTHONPATH=src python3 scripts/evaluate_fixture.py --json-out /tmp/kmd_eval_dspg.json`: `60/60 (1.000)`

## Known Risks and Next Steps

The 60-question fixture now passes and initialization builds real DSPG tables, but this is still a single self-written corpus. The next step should add fresh generated holdout folders with renamed entities, changed wording, new distractors, and graph/query stress cases before treating the engine as generally robust. The model-assisted staged extraction and query planning from the old DRT development line should be ported next as an optional local-only component.
