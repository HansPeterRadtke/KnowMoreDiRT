# HERB Failure Analogue Coverage

This development diagnostic maps every failed row from the latest external run to synthetic internal hard-fixture analogues. It intentionally omits external answers, source text, entity names, product names, and question wording. The core KMD package does not import or read this file.

- Source run: `kmd_final_generic_full_20260520_221320`
- Failed rows inspected: 1318
- Synthetic analogue suite: `tests/fixtures/hard_raw_reasoning`
- Hard QA total after expansion: 134

## Failure Signals
- exact_set_miss: 730
- false_positive: 503
- false_unknown: 189
- low_token_f1: 968
- retrieval_miss: 815

## Abstract Clusters
- `unanswerable_relation_false_positive`: 219 failed rows -> hrq024, hrq095, hrq101, hrq108, hrq112, hrq127
- `wrong_actor_identifier_set`: 202 failed rows -> hrq085, hrq086, hrq087, hrq088, hrq089, hrq132
- `wrong_relation_scoped_reference`: 176 failed rows -> hrq091, hrq092, hrq093, hrq094, hrq095, hrq125, hrq131
- `unanswerable_actor_identifier_false_positive`: 115 failed rows -> hrq090, hrq126
- `wrong_organization_set`: 107 failed rows -> hrq097, hrq098, hrq099, hrq100, hrq101
- `unanswerable_reference_false_positive`: 96 failed rows -> hrq095, hrq127
- `content_phrase_or_field_value`: 93 failed rows -> hrq109, hrq111, hrq128, hrq129, hrq130
- `missing_content_relation`: 87 failed rows -> hrq108, hrq109, hrq111, hrq128, hrq129, hrq130
- `missing_actor_identifier_role_chain`: 62 failed rows -> hrq085, hrq086, hrq087, hrq090, hrq126
- `unanswerable_temporal_false_positive`: 44 failed rows -> hrq103, hrq108, hrq112
- `unanswerable_organization_false_positive`: 29 failed rows -> hrq101
- `state_assignment_lookup`: 22 failed rows -> hrq105, hrq106
- `organization_relation`: 20 failed rows -> hrq097, hrq098, hrq101
- `missing_organization_relation`: 19 failed rows -> hrq101
- `missing_relation_scoped_reference`: 11 failed rows -> hrq095, hrq127
- `missing_temporal_state`: 10 failed rows -> hrq102, hrq103, hrq104, hrq112
- `temporal_final_current_state`: 6 failed rows -> hrq051, hrq052, hrq053, hrq102, hrq103, hrq104

## Coverage Rule
Every failed external row has one entry in `tests/fixtures/hard_raw_reasoning_failure_map.json` linking its `question_id` to one abstract cluster and one or more internal synthetic QA IDs. Clusters are shared where rows require the same reasoning capability.
