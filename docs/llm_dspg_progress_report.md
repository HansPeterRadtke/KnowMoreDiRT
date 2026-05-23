# LLM-Centered DSPG Progress Report

Date: 2026-05-23

## Scope

This report records the current KnowMoreDiRT implementation state after continuing the generic DRT/DSPG rebuild. HERB was not run. The work stayed inside the KMD raw-folder architecture: the public API remains `initialize(folder_path)` and `question(text) -> string`, and the core still receives only raw folder contents and question text.

## Local Model Status

The local llama.cpp text endpoint was checked directly:

- Endpoint: `http://127.0.0.1:14829/v1`
- Model reported by `/v1/models`: `Qwen2.5-14B-Instruct-Q4_K_M.gguf`
- Context reported by server metadata: `32768`
- Chat completion smoke test: passed during this work window
- Cloud APIs: not used
- HERB: not run

The vision endpoint on port `14830` is unrelated to this KMD text task and was not modified.

## What Changed

### Generic local-model plumbing

- Added `src/knowmoredirt/semantic_cache.py` for stable JSON caching of local-model chunk frame extraction.
- Extended the local model client to support llama.cpp OpenAI-compatible chat completions with stricter localhost-only safety and more robust JSON extraction.
- Added generic model-planner calls for:
  - chunk-to-discourse-frame extraction,
  - question-to-query-frame planning,
  - bounded evidence answer verification.
- Added counters/traces for model plan, chunk-frame, verifier, parse, accept, and reject activity.
- Kept local model usage explicit and optional. Default deterministic behavior still works without the model.

### Generic DSPG ingestion improvements

- Ingest now can attach local-model frames to grounded source spans when enabled.
- Added generic section/record grouping so label-value, table, and object-like text can inherit source-local structural context without requiring a prepared schema.
- Added object-like raw-text record extraction for brace-delimited raw text. This treats JSON-like data as text and stores grounded key/value relations rather than relying on an external schema.
- Added table/header relation extraction for delimited raw-text rows.
- Preserved source spans and provenance for deterministic and model-extracted relations.

### Generic bounded graph execution improvements

- Strengthened target and relation constraint matching in the bounded DSPG executor.
- Added low-priority/noise source filtering when better grounded non-noise candidates exist.
- Added generic relation-constraint filtering so broad answer-type terms do not satisfy missing relation requests by themselves.
- Improved count aggregation over grouped table/record/label relations.
- Added generic validity/context priors for current/active versus stale/obsolete source groups.
- Kept the implementation relation-agnostic: relation words remain data, not control-flow handlers.

### Generic extraction and canonicalization fixes

- Improved active-event extraction in mixed structural text so label-value lines do not suppress unrelated event sentences in the same chunk.
- Added a generic conversion from action-like label values into event-like relations when the label supplies a predicate and the value is a grounded action phrase.
- Improved token cleanup and name/person descriptor normalization without fixture-specific entity branches.

## Anti-Hardcoding Position

The current architecture intentionally avoids benchmark-specific and fixture-specific answer routes. Semantic words from source text or questions are used as relation/constraint data. They are not allowed to drive control-flow branches such as special owner/reviewer/manual/organization handlers.

Static architecture tests were strengthened earlier and still run under `pytest`; they now pass. The remaining failures are strict fixture-score failures, not static contamination failures.

## Validation Run

Commands run:

```bash
PYTHONPATH=src /data/venv/bin/python -m py_compile src/knowmoredirt/*.py tests/**/*.py scripts/evaluate_fixture.py
PYTHONPATH=src /data/venv/bin/python -m pytest -q
PYTHONPATH=src /data/venv/bin/python scripts/evaluate_fixture.py --corpus tests/fixtures/messy_raw_corpus --qa tests/fixtures/messy_raw_corpus_qa.json --json-out /tmp/kmd_eval_messy_final.json
PYTHONPATH=src /data/venv/bin/python scripts/evaluate_fixture.py --corpus tests/fixtures/broad_raw_world --qa tests/fixtures/broad_raw_world_qa.json --json-out /tmp/kmd_eval_broad_final.json
PYTHONPATH=src /data/venv/bin/python scripts/evaluate_fixture.py --corpus tests/fixtures/hardcore_noise --qa tests/fixtures/hardcore_noise_qa.json --json-out /tmp/kmd_eval_noise_final.json
PYTHONPATH=src /data/venv/bin/python scripts/evaluate_fixture.py --corpus tests/fixtures/hard_raw_reasoning --qa tests/fixtures/hard_raw_reasoning_qa.json --json-out /tmp/kmd_eval_hard_final.json
```

Results:

- `py_compile`: passed
- `pytest -q`: failed because strict fixture score gates are still not restored
- Unit/static tests outside strict fixture gates: passed in the final pytest run
- Messy fixture: `29/60` (`0.483`)
- Broad fixture: `41/65` (`0.631`)
- Hardcore noise fixture: `7/8` (`0.875`)
- Hard raw reasoning fixture: `71/134` (`0.530`)

The strict evaluation tests currently expect:

- Messy: `60/60`
- Broad: `65/65`
- Noise: `8/8`
- Hard: `134/134`

Those gates are intentionally still strict and were not lowered.

## Remaining Limitations

The current implementation is not done. The main remaining gaps are generic reasoning gaps, not benchmark adapter problems:

- Context/discourse accessibility is still weak, especially belief/dream/fiction/quote/denial separation.
- Multi-hop binding still misses many actor/reference chains and relation-scoped identifiers.
- Table/log row reasoning needs stronger row grouping, argmax/min, and exact constraint satisfaction.
- Unknown gating still allows some false positives when broad lexical evidence exists but a complete relation binding is absent.
- Canonical output sometimes returns a broader phrase than the expected minimal grounded value.
- The local-model path is wired and tested technically, but full fixture evaluation still primarily exercises deterministic DSPG execution; cached model-frame evaluation needs a dedicated, reproducible test mode before it can become a normal gate.
- Runtime use of the live local model is too slow for full fixture loops without additional caching and batching controls.

## Next Correct Work

The next implementation steps should remain generic:

1. Add a cacheable LLM-frame evaluation mode for fixtures so local semantic frames can be used reproducibly without expensive repeated calls.
2. Build a proper generic context-accessibility layer over assertion, quote, report, belief, dream, fiction, allegation, denial, and uncertainty contexts.
3. Strengthen relation role alignment and answer-variable binding in the graph executor instead of adding semantic keyword handlers.
4. Add generic row/object group traversal with aggregation operators such as count, latest, argmax, and scoped list extraction.
5. Use the local-model verifier as a bounded entailment check for high-risk candidates, with grounding and type validation enforced before returning a non-unknown answer.

## Conclusion

KMD has moved closer to the intended generic hybrid architecture: deterministic code builds and searches grounded DSPG structures, and the local LLM path now has concrete ingestion, planning, and verification hooks with caching. However, the internal fixture gates are not restored. This state should be treated as committed progress, not completion.
