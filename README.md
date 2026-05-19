# KnowMoreDiRT

KnowMoreDiRT (KMD) is a raw-folder discourse knowledge system. It reads arbitrary text files, builds an internal discourse provenance graph, and answers questions through a deliberately small public API:

```python
import knowmoredirt as kmd

kmd.initialize("/path/to/random/raw/folder")
answer = kmd.question("Who reviewed PR-8042?")
```

## Public Contract

KMD exposes only two intended user-facing operations:

- `initialize(folder_path)`: read one folder tree containing arbitrary readable text files.
- `question(text) -> string`: answer one natural-language question as a plain string.

The input folder may contain nested folders, arbitrary filenames, arbitrary extensions, files without extensions, prose, logs, tables, transcripts, JSON-like text, and noisy text. KMD does not require schemas, prepared corpora, metadata wrappers, or benchmark-specific conversion.

See [`docs/public_api.md`](docs/public_api.md) for the exact API contract.

## DRT and DSPG

KnowMoreDiRT is grounded in **Discourse Representation Theory (DRT)**, the dynamic semantic framework introduced by Hans Kamp and developed by Kamp, Reyle, and others for discourse referents, anaphora, tense, context, and discourse-dependent meaning. DRT explains why meaning cannot be reduced to isolated sentence facts: each sentence updates a structured discourse representation that preserves referents, conditions, scope, and temporal/contextual dependencies.

KMD uses **DSPG** (Discourse Source Provenance Graph) as the engineering representation layer evolved from practical DRT system work. DSPG keeps the DRT commitments that matter for a working knowledge system—mentions, referents, contexts, frames, temporal evolution, source spans, and provenance—while storing them in queryable graph/database structures over raw text.

Theory and architecture docs:

- [`docs/theory.md`](docs/theory.md)
- [`docs/architecture.md`](docs/architecture.md)

Selected DRT references:

- [Stanford Encyclopedia of Philosophy: Discourse Representation Theory](https://plato.stanford.edu/entries/discourse-representation-theory/)
- Hans Kamp, [“A Theory of Truth and Semantic Representation”](https://www.degruyterbrill.com/document/doi/10.1515/9783110867602.1/html)
- Hans Kamp and Uwe Reyle, [*From Discourse to Logic*](https://books.google.com/books?vid=ISBN079232403X)
- Kamp, van Genabith, and Reyle, [“Discourse Representation Theory” handbook chapter](https://www.ims.uni-stuttgart.de/archiv/kamp/files/2011.kamp.van.genebith.reyle.discourse.representation.theory.handbook.pdf)

## Current System

KMD currently provides a first DSPG-backed vertical slice:

- raw-folder scanning and text ingestion,
- natural filesystem metadata capture,
- sentence/line chunking,
- SQLite DSPG persistence,
- source spans, mentions, referents, contexts, frames, and frame arguments,
- bounded lexical/referent retrieval,
- text-quality/noise contexts for random-character, hex/blob-like, OCR-corrupted, word-salad, plausible-babble, and meaningful-discourse sources,
- conservative source-grounded answering,
- isolated optional local-model integration hooks.

This is not presented as a finished reasoning engine. The current fixture score is a regression baseline; broader generated and real-world holdouts are required before claiming general robustness. See [`docs/evaluation.md`](docs/evaluation.md).

## Development

```bash
python3 -m pip install -e '.[test]'
PYTHONPATH=src pytest -q
PYTHONPATH=src python3 scripts/evaluate_fixture.py --json-out /tmp/kmd_eval.json
```

## Test Layout

- `tests/unit/` validates raw fixture contracts, scanner behavior, and indexing.
- `tests/smoke/` validates the two-function public API.
- `tests/evaluation/` runs fixture QA evaluation harnesses.
- `tests/fixtures/messy_raw_corpus/` contains the original project-style regression corpus.
- `tests/fixtures/broad_raw_world/` contains a broad heterogeneous raw-world corpus.
- `tests/fixtures/hardcore_noise/` contains random-character, OCR, base64/hex-like, and word-salad pollution tests.

## Current Fixture Score

Current regression scores:

- original messy corpus: `60/60 (1.000)`
- broad raw-world corpus: `65/65 (1.000)`
- hardcore noise corpus: `8/8 (1.000)`

See [`docs/evaluation.md`](docs/evaluation.md) for details and limitations.
