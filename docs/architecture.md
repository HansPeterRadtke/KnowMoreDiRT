# Architecture

KnowMoreDiRT is a raw-folder DRT/DSPG system. It accepts a folder tree, reads all readable files as raw text, builds an internal discourse provenance graph, and answers questions from that graph and its supporting text index.

## Public Boundary

The public boundary is intentionally minimal:

- `initialize(folder_path)`
- `question(text) -> string`

No prepared corpus, metadata wrapper, benchmark adapter, manifest, or schema is part of the input contract.

## Ingestion Pipeline

Initialization performs these steps:

1. **Folder scan**: recursively traverse arbitrary folders and filenames.
2. **Text read**: read each readable file as text.
3. **Natural metadata capture**: record path, relative path, size, ctime, mtime, byte count, and SHA-256 content hash.
4. **Chunking**: split text into sentence/line-sized units while preserving source offsets.
5. **Source spans**: store both chunk spans and mention spans.
6. **Mention extraction**: extract source-grounded IDs, URLs, file-like values, names, and artifact strings.
7. **Referent construction**: create local referents from exact mentions without requiring destructive global merging.
8. **Context assignment**: mark sentence-level contexts such as asserted, reported, believed, alleged, dreamed, fictional, and negated.
9. **Frame extraction**: create lightweight event/proposition frames with predicates and argument links.
10. **Text-quality scoring**: store generic structural signals for low-semantic-content files such as random-character blobs and symbol-heavy corruption.
11. **Indexing**: build bounded retrieval structures over both raw chunks and DSPG records.

## SQLite DSPG Store

The current store is SQLite-backed and normalized. It includes:

- `extraction_runs`
- `documents`
- `chunks`
- `source_spans`
- `mentions`
- `referents`
- `mention_referents`
- `contexts`
- `frames`
- `frame_arguments`
- `temporal_edges`

The current implementation uses an in-memory database by default. A durable user-configurable store path is planned.

## Retrieval and Query Execution

The current query path combines:

- lexical retrieval over raw sentence chunks,
- referent-centric retrieval through mentions and referents,
- frame-aware retrieval through predicates and frame arguments,
- temporal state retrieval for state changes with dated evidence,
- text-quality downweighting so noise files do not dominate normal questions,
- conservative deterministic answer extraction over bounded candidates.

This is a first vertical slice of the full DSPG query architecture. It avoids full-corpus reasoning per question and avoids assuming external input structure. Future work should strengthen graph traversal, entity resolution, uncertainty handling, and model-assisted query planning.

## Optional Local Model Integration

KMD includes an isolated local model client hook. The default system does not require a model and does not call cloud APIs. Future staged model integration should use bounded inputs and constrained outputs for:

- mention classification,
- frame extraction,
- context/scope classification,
- identity hypotheses,
- query plan generation,
- answer verbalization with source grounding.

Model output must remain optional, validated, and source-grounded.

## Provenance

DSPG objects are grounded in exact source spans. Answers at the public boundary are strings, but internal answer records keep evidence objects with relative source path, source text, and score. Future public diagnostic APIs can expose provenance without changing the simple `question(text) -> string` user contract.

## Current Weaknesses

- The extractor is still mostly deterministic and shallow.
- Entity resolution is local and conservative.
- Context propagation is sentence-level rather than fully hierarchical.
- Temporal modeling handles simple dated state statements but not full interval logic.
- Noise handling is structural and conservative; it is not a semantic gibberish detector.
- The local model path is isolated but not yet part of the default staged pipeline.
- The fixture suite is now broader, but it is still self-written and not proof of broad generalization.
