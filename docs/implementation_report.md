# KnowMoreDiRT Implementation Report

## Package Structure

- `src/knowmoredirt/__init__.py`: exports only `initialize` and `question`.
- `src/knowmoredirt/public.py`: two-function public API and module-level initialized engine.
- `src/knowmoredirt/scanner.py`: raw folder traversal, UTF-8 text reading, file metadata, SHA-256 hashing, and text-unit extraction.
- `src/knowmoredirt/index.py`: lightweight lexical index over raw text units.
- `src/knowmoredirt/extractors.py`: generic regex extractors for URLs, IDs, emails, names, and labels.
- `src/knowmoredirt/engine.py`: first deterministic QA engine over the raw-text index.
- `src/knowmoredirt/evaluation.py`: internal fixture evaluation helpers.
- `scripts/evaluate_fixture.py`: developer evaluation command.

## Public API

The intended public API exposes only:

```python
initialize(folder_path)
question(text) -> string
```

`__all__` is exactly `['initialize', 'question']`.

## Package Style Findings

I searched `/data/src/github` and installed Python package metadata for CAPOV/capov examples. No local CAPOV package or installed package metadata was found. Because there was no clear HPR/CAPOV namespace pattern to follow, this implementation keeps the clean PyPI-oriented `src/knowmoredirt` package layout that was already started in the foundation commit.

## Old DRT Reference Use

Old `/data/src/github/devtests/DRT_tests` code was used only as conceptual reference for:

- raw-folder scanning discipline,
- exact artifact extraction patterns for URLs/PRs/bugs/tickets,
- lexical candidate retrieval before answer extraction,
- context distinctions such as asserted fact versus belief, dream, fiction, allegation, and quote/report.

The KMD code is a clean rewrite. It does not move the old scripts, does not include HERB/benchmark logic, and does not depend on prepared metadata or old DSPG database paths.

## Current Engine Behavior

Implemented:

- recursive raw-folder scanning,
- arbitrary filename/extension support,
- readable text ingestion only,
- natural file metadata collection: relative path, size, mtime, ctime, SHA-256,
- sentence/line chunking,
- lexical retrieval,
- generic URL/identifier/name/label extraction,
- deterministic answer patterns for direct facts, IDs, URLs, temporal states, tables, claims, beliefs, dreams, fiction, allegations, quotes, and unanswerable cases.

Not implemented yet:

- persistent database backend,
- full graph/DSPG representation,
- model-assisted reasoning,
- confidence-calibrated answer selection,
- unseen adversarial holdout generation,
- source/evidence API at the public boundary.

## Tests Run

- `python3 -m py_compile src/knowmoredirt/*.py tests/**/*.py scripts/evaluate_fixture.py`: passed
- `PYTHONPATH=src pytest -q`: `11 passed`
- `PYTHONPATH=src python3 scripts/evaluate_fixture.py --json-out /tmp/kmd_eval_final.json`: `60/60 (1.000)`

## Known Risks and Next Steps

The 60-question fixture now passes, but this is still a single self-written corpus. The next step should add generated holdout folders with renamed entities, changed wording, and new distractors before treating the engine as generally robust.
