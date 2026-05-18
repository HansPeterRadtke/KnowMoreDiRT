# KnowMoreDiRT Evaluation Report

## Fixture

- Corpus: `tests/fixtures/messy_raw_corpus/`
- QA file: `tests/fixtures/messy_raw_corpus_qa.json`
- Corpus files: 30
- QA pairs: 60

The corpus is raw text only: arbitrary folders, filenames, extensions/no extensions, prose, tables, chats, logs, transcript turns, raw JSON-like text, fiction/dream/legal contexts, IDs, URLs, noisy text, and distractors.

## Current Score

Total: `60/60 (1.000)`

## Category Breakdown

- `aggregation`: `2/2` (`1.000`)
- `ambiguity`: `1/1` (`1.000`)
- `belief_vs_fact`: `3/3` (`1.000`)
- `claim_vs_fact`: `2/2` (`1.000`)
- `code_context`: `1/1` (`1.000`)
- `contradiction_resolution`: `2/2` (`1.000`)
- `customer`: `3/3` (`1.000`)
- `direct_fact`: `2/2` (`1.000`)
- `discussion_disagreement`: `1/1` (`1.000`)
- `discussion_no_decision`: `1/1` (`1.000`)
- `distractor_avoidance`: `1/1` (`1.000`)
- `dream_vs_fact`: `2/2` (`1.000`)
- `entity_resolution`: `3/3` (`1.000`)
- `exact_id`: `3/3` (`1.000`)
- `exact_url`: `2/2` (`1.000`)
- `fiction_vs_fact`: `1/1` (`1.000`)
- `multi_hop`: `2/2` (`1.000`)
- `noisy_text`: `2/2` (`1.000`)
- `not_over_answering`: `1/1` (`1.000`)
- `quote_vs_assertion`: `2/2` (`1.000`)
- `raw_json_text`: `3/3` (`1.000`)
- `source_grounded`: `2/2` (`1.000`)
- `table_context`: `2/2` (`1.000`)
- `table_lookup`: `1/1` (`1.000`)
- `technical_document`: `2/2` (`1.000`)
- `temporal`: `5/5` (`1.000`)
- `temporal_context`: `1/1` (`1.000`)
- `transcript`: `2/2` (`1.000`)
- `unanswerable`: `4/4` (`1.000`)
- `who_claimed_what`: `1/1` (`1.000`)

## Interpretation

The first KMD implementation reaches 100% on the current 60-question fixture. That result is useful as a regression baseline, not as proof of broad generalization. The engine is still deterministic and pattern-heavy; it should be challenged next with fresh generated holdouts and paraphrased questions before adding more advanced graph/model components.

## Anti-Overfitting Notes

- The code does not contain exact answer lookup tables.
- The public API accepts only raw folder path plus question string.
- The engine contains generic extraction/retrieval patterns, but it is still tuned against the current reasoning categories and needs unseen tests.
- No HERB/benchmark data, questions, or entities are used in this repository.
