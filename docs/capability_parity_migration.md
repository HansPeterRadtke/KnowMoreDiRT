# Capability-Parity Migration Report

This report records the exact old DRT path that produced the 59.14% HERB run, what was migrated into KnowMoreDiRT, and what was rejected as contamination.

## Old 59% Entrypoint

- Scored run: `/data/var/herb_benchmark/runs/drt_dspg_model_query_pure_raw_20260517_193059`
- Score: answerable accuracy `0.5914110429447853`
- Batch driver: `/data/var/herb_benchmark/drt_prepared/model_query_fix_batches100_20260517_141511/run_remaining_1_15.sh`
- Query command: `scripts/dspg_query.py --use-model-query --bounded-doc-limit 40` over `/data/var/herb_benchmark/drt_prepared/dspg/herb_drt.sqlite`
- Merge/scoring chain: `scripts/merge_drt_query_batches.py` then `scripts/run_drt_batched_model_query_scoring.py` and the local HERB scorer.

## Old Function Chain

1. `dspg_query.INTENT_GRAMMAR` constrained the local model query-plan JSON.
2. `dspg_query.deterministic_plan` generated a baseline plan from the question.
3. `dspg_query.call_model_query_plan` called local llama.cpp `/completion` with the grammar and planner prompt.
4. `dspg_query.normalize_model_plan` reconciled model and deterministic plan fields.
5. `dspg_query.rank_document_candidates` and `load_bounded_records` built a bounded subgraph from the old SQLite DSPG.
6. `dspg_query.execute_plan` answered by executing the normalized plan over bounded records.
7. `dspg_query.run_query` wrote per-question logs/checkpoints and final query results.
8. Batch merge and the HERB adapter serialized predictions for the scorer.

## Migration Map

| Old component | KMD component |
| --- | --- |
| `INTENT_GRAMMAR` | `src/knowmoredirt/legacy_drt_path.py` |
| `deterministic_plan` | `src/knowmoredirt/legacy_drt_path.py` |
| `call_model_query_plan` | `src/knowmoredirt/legacy_drt_path.py` + `LocalModelClient.complete_json(grammar=...)` |
| `normalize_model_plan` | `src/knowmoredirt/legacy_drt_path.py` |
| model-plan branch of `run_query` | `KnowMoreDiRTEngine._answer_with_migrated_model_query` |
| `execute_plan` | `KnowMoreDiRTEngine._execute_migrated_plan`, executing the migrated plan against KMD raw-folder DSPG handlers |
| progress/checkpoint logs | `scripts/benchmarks/run_herb_kmd_raw_folder.py` per-question JSONL checkpoint/progress |
| scorer adapter | benchmark glue under `scripts/benchmarks`, outside core `src/knowmoredirt` |

## Clean Capability Ported

- The old constrained model query-plan grammar and prompt style are now active in KMD.
- The old intent enum and model/deterministic normalization behavior are ported.
- `LocalModelClient` supports llama.cpp grammar-constrained `/completion` calls.
- `KMD_USE_LOCAL_MODEL=1` activates the migrated model-query path inside `initialize(folder_path)` / `question(text)`.
- Benchmark glue can record per-question prompt/response hashes, model parse counts, accepted-plan counts, and evidence counts.

## Contamination Removed or Rejected

- The old prepared DRT source corpus is not used by core KMD.
- `HERB RAW ARTIFACT` metadata/text wrappers are rejected.
- Prepared fields such as `product_id`, `product_name`, `source_title`, `employee_ids`, and `customer_id` are not part of core KMD reasoning.
- Gold answers, answerability labels, evaluator fields, and official family/type labels are not used for query behavior.
- Filename/folder-path semantics are not treated as content facts inside core KMD.

## Active Model Proof

A fresh local-model proof was run against an invented temporary raw folder.

- Local model enabled: `True`
- Model call count: `1`
- Parsed count: `1`
- Accepted count: `1`
- Model-answer count: `1`
- Answer: `Nia Vale`
- Grounded evidence count: `1`
- Reason: `migrated DRT model-query plan: who_owns`
- Prompt hash: `e215bd6ffd88f5f62e52b24ea5c4c9436b1cb913ed12f1657098ae6005fc3abf`
- Response hash: `3f1cb1d1fe4a3089c4133cee9b326a9ef98b3d887e6de91b22cc545cfe28b840`

The model response was parsed and normalized into the old query-plan path; the answer was accepted only because it was grounded in the temporary raw-text evidence.

## Side-by-Side Sample Comparison

A 50-question sample was run through KMD with `--use-local-model` and compared against the old 59% run artifacts. This was not a full HERB benchmark rerun.

- KMD sample run: `/data/var/knowmoredirt/herb_runs/kmd_parity_model_sample50_20260520_094158`
- Comparison JSON: `/data/var/knowmoredirt/reports/kmd_parity_model_sample50_20260520_094158_parity_comparison.json`
- Old answered: `32/50`
- New answered: `24/50`
- New evidence-bearing: `24/50`
- Same serialized answer: `8/50`
- Model enabled: `True`
- Model call count: `50`
- Model parsed/accepted: `50/50`
- Model-answer count: `12`

This proves the old model-query planning path is being invoked from KMD. It does **not** prove full 59.14% capability parity: the old scored path executed over the old prepared SQLite graph and source representation, which is intentionally rejected in KMD.

## Validation

- `python3 -m py_compile src/knowmoredirt/*.py scripts/benchmarks/run_herb_kmd_raw_folder.py tests/**/*.py scripts/evaluate_fixture.py`
- `PYTHONPATH=src pytest -q`: `32 passed`
- Messy fixture: `60/60`
- Broad fixture: `65/65`
- Hardcore noise fixture: `8/8`

## Full Clean Raw-Folder HERB Rerun

A full official local HERB run was executed after the clean capability-port work using the current KMD raw-folder public interface and no local model calls. The run used the raw HERB source folder directly; no prepared DRT corpus, metadata wrapper, gold answer, answerability label, official family/type label, or scorer field was used by KMD query behavior.

- Run report: `/data/var/knowmoredirt/reports/kmd_recovery_raw_folder_full_20260520_093611.json`
- Run directory: `/data/var/knowmoredirt/herb_runs/kmd_recovery_raw_folder_full_20260520_093611`
- KMD commit at run time: `22a058a9c85a274ec6a7775a6725fcb27b64a501`
- Questions completed: `1514/1514`
- Answered count: `726`
- Evidence-bearing count: `726`
- Model used: `no` for the full rerun. The local-model path is ported and tested separately, but the full model run remains too slow/weak to use as the final full result on this Jetson configuration.

Scores from the official local scorer:

```json
{
  "answerable_accuracy": 0.5558282208588957,
  "deterministic_exact_match": 0.3527406922378419,
  "token_f1": 0.3531197849946884,
  "unanswerable_accuracy": 0.6094420600858369,
  "retrieval_recall": 0.0,
  "source_citation_precision": 0.0,
  "source_citation_recall": 0.0
}
```

This recovers most of the old 0.5914 answerable score while preserving the clean raw-folder contract. It does **not** claim full parity: the old 59.14% run still had a prepared SQLite/source representation that KMD intentionally rejects. The remaining gap is raw JSON-as-text bounded evidence routing and citation/source-ID mapping, especially for customer/content families.

## Current Gap

The old local-model planner path is alive in KMD and the clean full rerun now reaches answerable accuracy `0.5558282208588957`. The old `execute_plan` / `load_bounded_records` behavior was not copied byte-for-byte because it was tied to the old DSPG schema and contaminated prepared corpus. KMD executes migrated plans and deterministic bounded retrieval over raw-folder DSPG handlers instead. Recovering or exceeding the old `0.5914110429447853` result without contamination requires better raw JSON-as-text bounded evidence routing, customer/content extraction, and source/citation ID mapping.
