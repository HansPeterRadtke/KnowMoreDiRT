# Raw Text Test Corpus Report

The fixture at `tests/fixtures/messy_raw_corpus/` is a deliberately messy raw-folder corpus for future KnowMoreDiRT development.

## Contents

- Corpus files: 30
- Corpus folders below the fixture root: 34
- Ground-truth QA pairs: 60
- Nested folders use meaningless names such as `m7/`, `q2/no_ext/`, and `rubble/`.
- Files use mixed endings (`.txt`, `.log`, `.md`, `.jsonish`, `.tsv`, `.eml`, `.chat`, `.weird`) and several files with no extension.
- Content styles include prose notes, design descriptions, incident logs, meeting transcripts, chats, debates, dreams, beliefs, claims/counterclaims, timelines, table-like text, raw JSON-like text, email chains, noisy gibberish, misleading distractors, near-duplicate names, IDs, URLs, PR-like identifiers, bug IDs, and commit-like hashes.

## Ground Truth

`tests/fixtures/messy_raw_corpus_qa.json` contains source-grounded question-answer pairs. Evidence entries point to raw corpus files and exact snippets; there is no hidden metadata. Some questions intentionally expect `unknown`.

## Why This Enforces the Contract

The fixture is usable only as a raw folder tree. It contains no metadata wrappers, prepared manifests, benchmark-specific headers, or schema fields required by the public system. A future solver must discover structure from the file contents.

## Current Implementation Status

An initial deterministic raw-text engine is now implemented. It is sufficient to use this fixture as an executable regression baseline, but it is not the final DRT reasoning system and should be expanded with unseen holdouts before any broad claims are made.
