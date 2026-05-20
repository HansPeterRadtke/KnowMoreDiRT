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
3. **Natural metadata capture**: record filename, suffixes, parent path, directory depth, mode/permissions, uid/gid where available, inode/device where available, atime/ctime/mtime, symlink status, MIME guess, line count, word count, byte count, and SHA-256 content hash.
4. **Chunking**: split text into sentence/line-sized units while preserving source offsets.
5. **Source spans**: store both chunk spans and mention spans.
6. **Mention extraction**: extract source-grounded IDs, URLs, file-like values, names, and named text spans.
7. **Referent construction**: create local referents from exact mentions without requiring destructive global merging.
8. **Context assignment**: mark sentence-level contexts such as asserted, reported, believed, alleged, dreamed, fictional, and negated.
9. **Frame extraction**: create lightweight event/proposition frames with predicates and argument links.
10. **Generic relation extraction**: store label/value pairs, raw text key/value pairs, table cells, identifier values, meaning/plural relations, active/passive events, negation/proof/status relations, aliases, and timestamp relations as source-grounded DSPG relations.
11. **Text-quality/context scoring**: store generic structural signals and document-level contexts for low-semantic-content files such as random-character blobs, hex/blob-like text, OCR corruption, word salad, plausible babble, and meaningful discourse.
12. **Indexing**: build bounded retrieval structures over both raw chunks and DSPG records.

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
- `relations`

The current implementation uses an in-memory database by default. A durable user-configurable store path is planned.

Document metadata stores natural filesystem/read metadata and text-quality metrics, including printable ratio, symbol ratio, token diversity, OCR-like token ratio, a low-semantic-noise flag, and a semantic-quality label. The same classification is also represented as a `quality:*` context so noisy source material remains preserved and queryable rather than discarded.

## Retrieval and Query Execution

The current query path combines:

- lexical retrieval over raw sentence chunks,
- referent-centric retrieval through mentions and referents,
- frame-aware retrieval through predicates and frame arguments,
- relation-aware retrieval through generic label, identifier, event, status, temporal, and table relations,
- temporal state retrieval for state changes with dated evidence,
- text-quality downweighting so noise files do not dominate normal questions,
- conservative deterministic answer extraction over bounded candidates,
- ranking by anchor match, predicate/label match, context validity, temporal recency, and text-quality signals.

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
- Noise handling is structural and conservative; it labels and downweights low-semantic-content sources for ordinary fact retrieval while preserving them as source-grounded contexts.
- The local model path is isolated but not yet part of the default staged pipeline.
- The fixture suite is now broader, but it is still self-written and not proof of broad generalization.

## Optional Local Query Planner

KMD includes an optional local planning path for development. Candidate selection remains bounded before reasoning: lexical sentence search, DSPG relation/frame matches, neighboring discourse units, and natural filesystem metadata may contribute retrieval priors. Filesystem metadata can help locate a raw file, but answer facts must still be grounded in readable raw text spans.

When enabled, the local-model path uses a localhost llama.cpp-compatible endpoint to produce constrained JSON query plans, normalizes those plans with the deterministic planner, and executes them against KMD raw-folder DSPG records. This path is disabled by default, never uses cloud APIs, and should not introduce dataset- or scorer-specific behavior.
