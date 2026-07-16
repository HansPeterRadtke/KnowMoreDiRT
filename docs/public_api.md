# Public API

KnowMoreDiRT exposes exactly two intended user-facing functions.

## `initialize(folder_path)`

```python
import knowmoredirt as kmd

kmd.initialize("/path/to/folder")
```

`folder_path` is the only input. The folder may contain:

- nested subfolders,
- arbitrary filenames,
- arbitrary file extensions or no extension,
- readable prose files,
- logs,
- transcripts,
- tables written as text,
- JSON-like text,
- code-like text,
- noisy mixed text.

Every readable file is treated as raw text. KMD does not require or accept prepared corpora, a special external schema, metadata wrapper, manifest, source conversion layer, or semantic adapter format.

Normal runtime requires a reachable localhost llama.cpp-compatible endpoint. The public API does not expose model selection as an argument; configure the endpoint with `KMD_LOCAL_MODEL_ENDPOINT` when the default `http://127.0.0.1:14829/v1` is not correct.

## `question(text) -> string`

```python
answer = kmd.question("Which reference fixed the cache regression?")
```

`text` is the only question input. The return value is a plain answer string.

The internal system may keep diagnostics, source evidence, graph records, confidence estimates, and model-call traces. Those are implementation details, not part of the intended public API.

## Error Behavior

Calling `question` before `initialize` raises `RuntimeError`.

Calling `initialize` without reachable localhost llama.cpp raises a clear runtime error. If llama.cpp becomes unreachable during `question`, KMD raises a clear runtime error with request/cache context for diagnostics rather than returning an answer from a hidden no-model semantic path.

Empty or unsupported questions return `unknown` rather than fabricated answers.
