# Hard Raw-Reasoning Fixture

## Purpose

This fixture converts external failure observations into internal, self-written regression coverage without copying external benchmark entities, questions, answers, labels, or source records. The corpus is raw text only: arbitrary nested folders, odd filenames, prose, logs, tables, JSON-like text, and noise files.

## Failure Classes Sampled

The latest external diagnostic run showed recurring structural failures rather than one isolated bug. The internal hard fixture covers these generic failure modes:

- wrong answer type, such as returning a URL or identifier for a person or organization question;
- identifier-family confusion when several IDs of different meanings appear in the same source region;
- URL confusion when several links appear near the same entity but only one relation-scoped link is requested;
- person versus organization confusion in sources that contain names, group names, IDs, and URLs together;
- content-phrase extraction when the requested answer is a change, claim, explanation, or statement rather than a nearby structural value;
- unanswerable false positives where a related entity exists but the requested relation does not;
- nested JSON/object-like raw text with sibling distractors and arrays;
- two-hop lookup through reference and role relations;
- temporal/final-state reasoning with contradictory earlier states;
- discourse context, including dreams, beliefs, allegations, corrections, denied claims, and no-decision discussions;
- cache/lock/noise pollution that repeats target names with wrong values;
- minimal canonical output for exact IDs, URLs, paths, meanings, plurals, counts, and table values;
- aggregation/counting under row constraints;
- table/log lookup with repeated entities and distractors;
- mixed natural text, JSON-like text, tables, logs, and noise in one raw folder.

## Corpus and QA Size

- Corpus root: `tests/fixtures/hard_raw_reasoning/`
- QA file: `tests/fixtures/hard_raw_reasoning_qa.json`
- Source files: 12
- Questions: 134
- Categories: 15

## Anti-Overfitting Boundary

The fixture uses invented entities and values. It is structurally inspired by observed external failure modes, not by external benchmark wording or answer content. The KMD core still receives only a folder path and question strings, and the core package test suite scans for forbidden benchmark or fixture-specific vocabulary.

## Current Result

The hard fixture currently passes at `134/134 (1.000)`. This is a stricter internal regression milestone, not a claim of external benchmark generalization.
