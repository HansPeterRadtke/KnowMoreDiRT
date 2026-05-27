# DRT Generalization Run 2026-05-27

UTC start recorded for this run: `2026-05-27T12:32:31Z`.

This checkpoint restores the intended direction: semantic decisions are owned by the local model and represented as query or chunk DRS/DSPG JSON. Deterministic code is limited to schema repair, exact grounding checks, cache keys, source span storage, graph binding, accessibility checks, temporal ordering over already-interpreted temporal values, structural value validation, and final formatting.

## Architecture Changes

- Removed deterministic question answer-type inference and semantic temporal/negation fallbacks.
- Removed deterministic numeric/equality answer binding and source-path priority handling based on path words.
- Removed deterministic person/organization regex classification and answer rewrites that stripped query-slot words.
- Generalized model query frames with answer variables, scope requirements, modality requirements, normalized temporal-scope operators, and grounded query terms.
- Extended chunk frame extraction with grounded identity hypotheses and context holders.
- Preserved model-produced context holders in modality contexts instead of destructively merging all same-modality contexts.
- Added same-span temporal edge projection for structural records without interpreting relation labels.
- Kept deterministic temporal binding only when the query DRS supplies a temporal operator.

## Validation

Focused validation passed:

```text
python3 -m py_compile src/knowmoredirt/*.py
git diff --check
KMD_AUTO_LOCAL_MODEL=0 PYTHONPATH=src python3 -m pytest tests/unit/test_answer_type_and_model_extraction.py tests/unit/test_capability_recovery.py tests/unit/test_dspg_store.py tests/unit/test_generic_relations_and_metadata.py tests/unit/test_architecture_contract.py -q
31 passed
```

Local-model-disabled fixture slices:

```text
messy: 11/60
broad: 19/65
noise: 4/8
hard: 49/134
```

Full pytest with `KMD_AUTO_LOCAL_MODEL=0` still fails the strict fixture gates:

```text
broad expected 65/65, got 19/65
noise expected 8/8, got 4/8
hard expected 134/134, got 49/134
messy expected 60/60, got 11/60
```

The remaining failures are expected after removing shortcut-style semantic fallbacks. They should be addressed through better chunk-to-DRS construction, query-DRS construction, identity/context accessibility, bounded DRS binding, and verifier behavior, not by adding deterministic semantic handlers.
