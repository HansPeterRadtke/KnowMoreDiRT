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

After the second generalized cleanup checkpoint, deterministic fallback slices are:

```text
messy: 10/60
broad: 20/65
noise: 4/8
hard: 44/134
```

The second checkpoint removed agentive morphology generation, made all non-asserted `modality:*` contexts inaccessible unless the query DRS requests that context, treated source-quality contexts as retrieval metadata rather than inaccessible discourse boxes, and allowed model-produced unary DRS predicates to bind non-structural answer variables when no non-target argument exists. The score regression is accepted for this checkpoint because it removes deterministic semantic guessing and improves the LLM-first DRS path.

An isolated live-model probe on a tiny scoped-state corpus showed that the chunk model can mark a frame as `modality="reported"` and the query model can produce a grounded query frame, but it may encode the scoped value as a unary predicate rather than a separate argument. Graph binding now supports that unary DRS shape. The verifier still rejected the graph candidate for this probe, while bounded model evidence extraction returned the correct grounded value; this points to verifier calibration as a generic model-stage issue rather than a reason to add deterministic semantic parsing.

An isolated live-model count probe on `Alpha/Beta/Gamma unit status` records showed a graph aggregation flaw: the query DRS contained `requested_relation=status`, `relation_terms=(units, ready)`, and `aggregation=count`, but the executor counted a `blocked` unit because it only required two matched prefixes from the whole flattened relation term set. The aggregation matcher now treats each model-produced relation term or constraint token as its own DRS term group and requires every group to hold in a counted source span. This is a pure query-DRS satisfaction check; it does not interpret the raw language or add a count-domain handler. The same live probe now returns `2`.

Eager model-frame ingestion now mirrors lazy materialization for polarity scope: non-positive model-produced DRS conditions are stored in nested `polarity:*` contexts, and context IDs include their parent and evidence surface so separate subordinate boxes are not destructively merged. Bounded accessibility no longer treats any satisfied context requirement as permission to pass through every inaccessible context; negative contexts remain blocked unless the query DRS is negated or explicitly asks for that polarity scope.

Frame validation now also requires model-produced `temporal_text` to be an exact substring of the source chunk before temporal DRS records are accepted. This extends the existing grounding checks for evidence, arguments, identity hypotheses, and context holders.

After the count aggregation repair, local-model-disabled fixture slices are:

```text
messy: 11/60
broad: 20/65
noise: 4/8
hard: 44/134
```

Full pytest with `KMD_AUTO_LOCAL_MODEL=0` still fails the strict fixture gates and the known noise canonicalization check:

```text
broad expected 65/65, got 20/65
noise expected 8/8, got 4/8
hard expected 134/134, got 44/134
messy expected 60/60, got 11/60
direct noise query expected "Dr. Pella", got "Dr. Pella watered the greenhouse fern"
```

The remaining failures are expected after removing shortcut-style semantic fallbacks. They should be addressed through better chunk-to-DRS construction, query-DRS construction, identity/context accessibility, bounded DRS binding, and verifier behavior, not by adding deterministic semantic handlers.
