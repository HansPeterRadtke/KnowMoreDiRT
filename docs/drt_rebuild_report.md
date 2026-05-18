# KMD DRT/DSPG Rebuild Report

## Status

This phase replaces the initial fixture-oriented engine with a cleaned DRT/DSPG vertical slice inside the professional `src/knowmoredirt` package. The public API remains intentionally small: `initialize(folder_path)` and `question(text) -> string`.

The implementation is not a blind move from the old development repository. It ports the old DRT architectural shape into KMD modules while keeping HERB adapters, prepared-corpus workflows, benchmark metadata routing, and old CLI scaffolding out of the package.

## Old DRT Material Reviewed

Reviewed old git history and reports from `/data/src/github/devtests/DRT_tests`, including cleanup and benchmark history around:

- raw-folder-only public interface cleanup,
- pure raw text architecture enforcement,
- no-prepare-run benchmark workflow removal,
- pure raw model-query HERB result,
- fresh public raw-folder HERB rerun.

Reviewed old core/reference files:

- `/data/src/github/devtests/DRT_tests/drt.py`
- `/data/src/github/devtests/DRT_tests/dspg.py`
- `/data/src/github/devtests/DRT_tests/dspg_store.py`
- `/data/src/github/devtests/DRT_tests/extract.py`
- `/data/src/github/devtests/DRT_tests/contracts.py`
- `/data/src/github/devtests/DRT_tests/config/dspg_system.yaml`
- `/data/src/github/devtests/DRT_tests/scripts/dspg_ingest_folder.py`
- `/data/src/github/devtests/DRT_tests/scripts/dspg_query.py`

Key reference conclusions retained in KMD:

- DRT input must be a raw folder path only.
- Internal structure should be normalized DSPG records, not fixture-only answer lookup.
- SQLite is a useful first persistence layer for documents, chunks, spans, mentions, referents, contexts, frames, and arguments.
- Model integration must stay optional and isolated.
- HERB/prepared metadata code must not enter core KMD.

## Components Ported or Refactored

- `src/knowmoredirt/store.py`: new SQLite-backed DSPG store with normalized tables and indexes for documents, chunks, source spans, mentions, referents, mention-referent links, contexts, frames, and frame arguments.
- `src/knowmoredirt/ingest.py`: raw folder ingestion into DSPG records using the existing KMD scanner plus cleaned deterministic span, mention, context, referent, and frame extraction.
- `src/knowmoredirt/engine.py`: initialization now builds the DSPG store first, then constructs the existing bounded lexical query index over the same raw sentence records.
- `src/knowmoredirt/model.py`: optional local model client hook is isolated from the default deterministic path and uses no cloud API.
- `tests/unit/test_dspg_store.py`: verifies real DSPG tables are populated after initialization, not only answer strings.
- `tests/unit/test_architecture_contract.py`: verifies the core package does not contain old benchmark/prepared-input markers and that the public export remains `initialize`/`question` only.

## Intentionally Not Ported

- HERB scorer adapters, run wrappers, ID mapping, and benchmark runtime directories.
- Prepared corpus generation, metadata wrappers, `HERB RAW ARTIFACT` markers, and `allow_prepared_metadata` behavior.
- Metadata-only deterministic HERB intents and product/customer/employee routing based on prepared fields.
- Old one-shot full-DSPG JSON generation experiments.
- Full staged llama.cpp extraction and grammar contracts; KMD only contains an isolated local-model client hook in this phase.
- Old script-heavy CLI workflow; KMD keeps production internals as importable modules.

## Current Architecture

`initialize(folder_path)` now performs:

1. Recursive raw-folder scan over arbitrary readable files.
2. Natural filesystem metadata collection: relative path, size, ctime, mtime, content hash.
3. Chunking into text units.
4. SQLite DSPG graph construction:
   - document records,
   - chunk records,
   - sentence and mention source spans,
   - exact mentions and local referents,
   - assertion/report/belief/dream/fiction/allegation/negation contexts,
   - lightweight event/frame records and frame arguments.
5. Bounded lexical index construction for current query execution.

`question(text)` still returns only a string at the public boundary. Internals can inspect DSPG counts and integrity for tests/diagnostics, but those are not exported from `knowmoredirt.__all__`.

## Forbidden Benchmark/Prepared-Structure Audit

The core package was scanned for old input-structure markers:

- `HERB RAW ARTIFACT`
- `allow_prepared_metadata`
- `DRT_HERB_PREP_ROOT`
- `artifact_manifest_by_rel_path`
- `source_corpus`
- `product_id`
- `source_title`

No findings were found in `src/knowmoredirt/*.py` by the architecture contract test. KMD does not contain HERB entities, questions, scorer logic, or benchmark adapters.

## Validation

Final validation for this phase:

- `python3 -m py_compile src/knowmoredirt/*.py tests/**/*.py scripts/evaluate_fixture.py`
- `PYTHONPATH=src pytest -q`
- `PYTHONPATH=src python3 scripts/evaluate_fixture.py --json-out /tmp/kmd_eval_dspg.json`

The 60-question messy-corpus fixture remains a regression baseline and currently scores `60/60 (1.000)`.

## Known Weaknesses

- The DSPG store is currently in-memory by default; durable database path/configuration is next work.
- Mention, context, and frame extraction are deterministic and shallow compared with the old staged model-assisted extraction experiments.
- Query execution still relies heavily on the existing bounded lexical/deterministic answer layer rather than a full graph query planner.
- Optional model integration is only a clean hook; staged model-assisted extraction/query planning is not active yet.
- The 60-question fixture is useful but friendly. Fresh generated holdout folders, paraphrases, and adversarial raw-folder tests are still required before claiming broad generalization.
- External benchmark evidence from old DRT showed raw-folder architecture is viable but retrieval/citation recall can remain weak; that remains explicit next work.
