# LLM-Centered DSPG Progress Report

Date: 2026-05-25

## Scope

This report records the current KnowMoreDiRT implementation state after continuing the generic DRT/DSPG rebuild. HERB was not run. The work stayed inside the KMD raw-folder architecture: the public API remains `initialize(folder_path)` and `question(text) -> string`, and the core still receives only raw folder contents and question text.

## Local Model Status

The local llama.cpp text endpoint was checked directly at `http://127.0.0.1:14829/v1/models`. It reported `Qwen2.5-14B-Instruct-Q4_K_M.gguf` with server metadata showing a 32768 context. Cloud APIs were not used. A one-document live local-model smoke passed and answered from a model-produced DRS frame, but it took about 61 seconds. A live full fixture pass with the local model was not completed because the local model path is still too slow for the full internal fixture loop without additional batching and cache controls.

## What Changed

This pass added an explicit generic DRT layer in `src/knowmoredirt/drs.py`. The new layer defines discourse referents, discourse arguments, discourse conditions, and discourse contexts as relation-agnostic Python objects. These objects are not semantic handlers. They are normalized containers for predicates, arguments, modality, polarity, temporal text, confidence, and exact evidence text.

The SQLite DSPG schema was extended with `identity_hypotheses`. Ingestion now creates same-surface identity hypotheses when deterministic or model-produced frame arguments align with existing mention referents. This is a small first step toward the old DRT_tests identity machinery, but without importing benchmark-shaped identity categories.

Ingestion now converts deterministic source-grounded surface records into generic discourse conditions and frame arguments. The deterministic extractor is deliberately limited to universal structures such as labels, object-like key/value text, table cells, URLs, identifiers, and timestamps. It no longer creates active/passive event relations, copular assertions, or negation-as-semantics records. Those semantic decisions now belong to local-model DRS frame extraction.

The local-model chunk-frame path now normalizes accepted frame dictionaries through the DRS layer. Accepted model frames create referents, frame arguments, semantic relations, modality contexts, temporal edges when supplied, and identity hypotheses. The model output must still be grounded by exact evidence text from the chunk before it is stored.

The bounded DSPG executor now uses generic query-frame binding as the normal path. It loads a bounded SQLite subgraph, treats rows as grounded DRS conditions, and binds answer variables by checking target anchors, requested predicate text, context accessibility, temporal scope, broad answer type, and evidence. When local-model frames are present, it binds the same query frame against grounded frame arguments. This is closer to Kamp-style variable binding than the earlier procedural answer extraction path. It does not branch on owner, reviewer, manual, runbook, customer, ticket, or any other domain relation name.

Unit tests were extended to prove that local-model chunk frames can become queryable generic frame arguments and that the store exposes the new identity-hypothesis table. The static architecture tests still reject benchmark/prepared markers and obvious semantic-handler branches in core code.

## Validation Run

Commands run:

```bash
python3 -m py_compile src/knowmoredirt/*.py tests/**/*.py scripts/evaluate_fixture.py
PYTHONPATH=src pytest -q
```

The py_compile command passed. Targeted non-evaluation unit, smoke, architecture, store, model, and capability tests passed with `28 passed`. Full pytest still fails on the strict fixture-score gates, which were not lowered. The current standalone fixture scores are messy 18/60, broad 27/65, noise 7/8, and hard 45/134.

HERB was not run.

## Interpretation

The current implementation is still not complete. It is materially closer to the intended architecture because the package now has explicit DRT-style condition objects, identity-hypothesis storage, model-frame normalization, grounded frame-argument variable binding, and a deterministic layer restricted to surface structure. However, the strict internal fixtures remain far below the earlier procedural-handler scores. That failure is expected at this stage and should not be hidden: KMD has removed procedural semantic routing, but the replacement generic LLM-centered DRT/DSPG machinery is not yet strong enough to recover all fixture behavior.

The main remaining gap is not a missing owner/reviewer/manual-style handler. The missing work is generic. KMD still needs robust LLM chunk-frame coverage with cacheable fixture execution, stronger context accessibility, stronger temporal/event/state modeling, stronger multi-hop referent traversal, stronger table/log/object group traversal, aggregation operators over generic frame bindings, and verifier-driven unknown gating. The current deterministic fallback is useful infrastructure, but it is not a full semantic parser.

## Next Required Work

The next correct step is to make local-model frame extraction practical for full fixture evaluation by adding durable cache reuse, progress logging, and batch controls, then run the strict internal fixtures in model-enabled mode without changing the public API. After that, the bounded executor should be extended to traverse frame-to-referent neighborhoods, identity hypotheses, context assignments, temporal edges, and grouped object/table records as generic DRT structures. Only then should fixture failures be treated as evidence of missing generic mechanisms rather than as prompts to reintroduce semantic handlers.
