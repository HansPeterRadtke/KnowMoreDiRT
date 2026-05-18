# KnowMoreDiRT

KnowMoreDiRT is planned as a raw-folder knowledge system. The public contract is intentionally small:

1. `initialize(folder_path)`
   - accepts only a folder path
   - reads arbitrary nested folders and arbitrary readable text files
   - does not require schemas, prepared corpora, metadata wrappers, or benchmark conversion
2. `question(text) -> string`
   - accepts only a question string
   - returns only an answer string at the public interface boundary

This repository currently contains a test-first foundation: package skeleton, documentation, a deliberately messy raw-text fixture, and validation tests. It does **not** implement the final reasoning engine yet.

## Development

```bash
python3 -m pip install -e '.[test]'
pytest
```

The fixture under `tests/fixtures/messy_raw_corpus/` is deliberately unstructured. Tests validate the raw-folder-only contract and the ground-truth QA specification without pretending that a solver exists.

