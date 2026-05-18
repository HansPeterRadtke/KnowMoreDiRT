# KnowMoreDiRT

KnowMoreDiRT (KMD) is a raw-folder knowledge system prototype. The intended public API is deliberately small:

```python
import knowmoredirt as kmd

kmd.initialize("/path/to/random/raw/folder")
answer = kmd.question("Who reviewed PR-8042?")
```

## Public Contract

1. `initialize(folder_path)`
   - input is only a folder path
   - the folder may contain arbitrary nested folders and arbitrary filenames/extensions
   - every readable file is treated as raw text
   - no schema, prepared corpus, metadata wrapper, or benchmark conversion is required
2. `question(text) -> string`
   - input is only a question string
   - output is only an answer string at the public boundary

Internals may use indexes, extracted spans, evidence, and diagnostics, but only `initialize` and `question` are exported as the intended public module API.

## Current Implementation

This first implementation is deterministic and local-only. It scans raw files, records natural filesystem metadata, chunks text into sentence/line units, builds a lexical index, extracts common IDs/URLs/files/names, and applies conservative source-grounded answer patterns for the fixture categories.

It is not the final DRT reasoning engine. The fixture score is a starting point for regression testing, not a claim of broad real-world generalization.

## Development

```bash
python3 -m pip install -e '.[test]'
PYTHONPATH=src pytest -q
PYTHONPATH=src python3 scripts/evaluate_fixture.py --json-out /tmp/kmd_eval.json
```

## Test Layout

- `tests/unit/` validates raw fixture contracts, scanner behavior, and indexing.
- `tests/smoke/` validates the two-function public API.
- `tests/evaluation/` runs the messy-corpus QA evaluation harness.
- `tests/fixtures/messy_raw_corpus/` contains the raw text corpus.
- `tests/fixtures/messy_raw_corpus_qa.json` contains source-grounded QA pairs.

## Current Fixture Score

`60/60 (1.000)` on the 60-question messy raw-text corpus.

See `docs/implementation_report.md` and `docs/evaluation_report.md` for details.
