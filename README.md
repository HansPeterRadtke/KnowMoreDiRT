# KnowMoreDiRT

KnowMoreDiRT answers questions over arbitrary raw folders through a model-owned semantic planner and generic deterministic tools.

```python
import knowmoredirt as kmd
kmd.initialize("/path/to/folder")
print(kmd.question("What changed most recently?"))
```

The core has no benchmark-specific routes and no Python question semantics. The model creates a strict structured query program, generic tools execute it with provenance, and a second strict model call produces a grounded answer. Every semantic call uses native JSON Schema constrained decoding.
