# Evaluation

## Current Fixture

The current self-written fixture is:

- corpus: `tests/fixtures/messy_raw_corpus/`
- QA file: `tests/fixtures/messy_raw_corpus_qa.json`
- files: 30
- questions: 60

The fixture is designed to exercise the raw-folder contract. It contains arbitrary nested folders, arbitrary filenames, mixed file endings, files without extensions, prose, tables, chats, logs, transcript turns, JSON-like text, dreams, beliefs, allegations, contradictions, noisy text, IDs, URLs, and distractors.

## Current Score

Current result:

- total: `60/60`
- score: `1.000`

The current score is a regression baseline for the fixture, not evidence that KMD is generally solved.

## Categories Covered

The fixture covers:

- direct facts,
- exact IDs and URLs,
- source-grounded questions,
- temporal/final-state questions,
- table lookup and table context,
- beliefs versus facts,
- dreams and fiction versus assertions,
- claims, counterclaims, allegations, and contradictions,
- transcript/chat speaker attribution,
- noisy text,
- raw JSON-like text,
- entity ambiguity,
- multi-hop dependencies,
- aggregation,
- unanswerable questions.

## Architecture Checks

The unit tests also verify that initialization creates normalized DSPG structures:

- documents,
- chunks,
- source spans,
- mentions,
- referents,
- context records,
- frames,
- frame arguments,
- temporal edges.

Architecture tests assert that the core package contains no old benchmark/prepared-input markers and that `knowmoredirt.__all__` exports only `initialize` and `question`.

## Limitations

The fixture is self-written and currently friendly to the implementation. It is useful for regression, but it should not be treated as proof of broad real-world generalization.

Next evaluation work should add:

- generated holdout folders,
- entity-renamed mutation suites,
- paraphrased question suites,
- adversarial context/scope tests,
- larger random raw-folder corpora,
- independent external benchmark runs only after internal generalization improves.
