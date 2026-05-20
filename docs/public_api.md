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

## `question(text) -> string`

```python
answer = kmd.question("Which reference fixed the cache regression?")
```

`text` is the only question input. The return value is a plain answer string.

The internal system may keep diagnostics, source evidence, graph records, confidence estimates, and model-call traces. Those are implementation details, not part of the intended public API.

## Error Behavior

Calling `question` before `initialize` raises `RuntimeError`.

Empty or unsupported questions return `unknown` rather than fabricated answers.
