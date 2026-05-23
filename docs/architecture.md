# Architecture

KnowMoreDiRT is a raw-folder DRT/DSPG system. It accepts a folder tree, reads all readable files as raw text, builds an internal discourse provenance graph, and answers questions from that graph and its supporting text index.

## Public Boundary

The public boundary is intentionally minimal:

- `initialize(folder_path)`
- `question(text) -> string`

No metadata wrapper, manifest, semantic adapter, or external schema is part of the input contract.

## Ingestion Pipeline

Initialization performs these steps:

1. **Folder scan**: recursively traverse arbitrary folders and filenames.
2. **Text read**: read each readable file as text.
3. **Natural metadata capture**: record filename, suffixes, parent path, directory depth, mode/permissions, uid/gid where available, inode/device where available, atime/ctime/mtime, symlink status, MIME guess, line count, word count, byte count, and SHA-256 content hash.
4. **Chunking**: split text into sentence/line-sized units while preserving source offsets.
5. **Source spans**: store both chunk spans and mention spans.
6. **Mention extraction**: extract source-grounded IDs, URLs, file-like values, names, and named text spans.
7. **Referent construction**: create local referents from exact mentions without requiring destructive global merging.
8. **Context assignment**: mark sentence-level assertion and discourse-scope contexts from source-grounded carriers.
9. **Frame extraction**: create lightweight event/proposition frames with observed predicates and argument links.
10. **Optional local-model discourse frames**: when `KMD_USE_LOCAL_MODEL=1` and LLM ingestion is enabled, each meaningful source chunk is sent to the localhost-only model for generic DRT/DSPG frame JSON. Accepted frames must be grounded by exact evidence text from the chunk before they are stored. Model arguments are converted into referents, frame arguments, same-surface identity hypotheses, and source-grounded semantic relations.
11. **Generic relation extraction**: store label/value pairs, JSON-like/object-as-text key/value pairs, table cells, identifier values, copular assertions, active/passive events, negation relations, and timestamp relations as source-grounded DSPG relations.
12. **Text-quality/context scoring**: store generic structural signals and document-level contexts for low-semantic-content files such as random-character blobs, hex/blob-like text, OCR corruption, word salad, plausible babble, and meaningful discourse.
13. **Indexing**: build bounded retrieval structures over both raw chunks and DSPG records.

## SQLite DSPG Store

The current store is SQLite-backed and normalized. It includes:

- `extraction_runs`
- `documents`
- `chunks`
- `source_spans`
- `mentions`
- `referents`
- `mention_referents`
- `identity_hypotheses`
- `contexts`
- `context_carriers`
- `context_assignments`
- `frames`
- `frame_arguments`
- `temporal_edges`
- `relations`
- `metadata_records`

The current implementation uses an in-memory database by default. A durable user-configurable store path is planned.

KMD now has an explicit Python DRT layer in `knowmoredirt.drs`. It defines discourse referents, discourse arguments, discourse conditions, and discourse contexts as relation-agnostic objects. These objects are normalized into the SQLite DSPG store. Predicate and role labels remain data from source text or model output. They are not intent enums and do not select bespoke answer handlers.

Document metadata stores natural filesystem/read metadata and text-quality metrics, including printable ratio, symbol ratio, token diversity, OCR-like token ratio, a low-semantic-noise flag, and a semantic-quality label. The same classification is also represented as a `quality:*` context so noisy source material remains preserved and queryable rather than discarded. Generic filesystem/read metadata is also normalized into `metadata_records`, while source quality, filesystem time, sentence context, and event-time signals are represented as context carriers and assignments.

## Retrieval and Query Execution

The current query path combines:

- lexical retrieval over raw sentence chunks,
- referent-centric retrieval through mentions and referents,
- frame-aware retrieval through observed predicates and frame arguments,
- relation-aware retrieval through generic label, identifier, event, assertion, temporal, table, and record relations,
- bounded SQLite subgraph execution over selected documents/chunks, source spans, mentions, referents, contexts, frames, frame arguments, temporal edges, and relations,
- local-model frame argument binding when semantic frames are present,
- temporal state retrieval for state changes with dated evidence,
- text-quality downweighting so noise files do not dominate normal questions,
- conservative deterministic answer extraction over bounded candidates,
- ranking by anchor match, requested-relation term match, relation completeness, context validity, temporal recency, and text-quality signals.

Questions are parsed into generic query frames containing target anchors, requested relation text, relation terms, constraints, answer type, temporal scope, negation, aggregation, and evidence requirements. Relation words from a source or question remain data inside the frame; they do not select content-specific code branches.

This is a first vertical slice of the full DSPG query architecture. It avoids full-corpus graph loading per question and avoids assuming external input structure. Future work should strengthen graph traversal, entity resolution, uncertainty handling, aggregation, discourse context propagation, and deeper model-assisted extraction.

The bounded SQLite graph executor is part of the normal non-model answer pipeline for query plans that can be mapped to generic DSPG operations. The optional local model path uses the same executor after producing a constrained plan, so model assistance refines planning rather than replacing grounded graph execution.

When model frames are available, answer candidates can be produced by binding the query frame against frame arguments rather than by using a relation-name handler. The executor checks that the target anchors, requested predicate text, context, and expected answer type are jointly satisfied, then returns compatible non-target arguments as possible answer-variable bindings.

Before returning a non-unknown answer, KMD now infers a broad expected answer type from the question: person/actor, organization, identifier, URL, file path, count, state, date/time, boolean, content phrase, metadata value, or unknown. Candidate answers are rejected if the value type is incompatible with the question. This prevents structural references such as URLs, file paths, IDs, and metadata-only hits from satisfying person, organization, state, or content questions unless the question explicitly asks for that type. Metadata records remain valid answer sources only for metadata questions; otherwise they serve as retrieval priors.

## Optional Local Model Integration

KMD includes an isolated local model client hook. The default system does not require a model and does not call cloud APIs. When explicitly enabled, model use is bounded and constrained in three roles:

1. **Chunk frame extraction**: convert raw chunks into generic DRT/DSPG frames with predicates, argument roles, polarity, modality/context, temporal text, confidence, and exact evidence text.
2. **Question frame parsing**: convert the question into the same generic query-frame language used by deterministic planning.
3. **Answer verification/extraction**: verify candidate answers against bounded evidence and discourse frames, or extract the shortest grounded answer from bounded evidence when graph execution cannot bind an answer.

The model is never allowed to use outside knowledge or external labels. All accepted output must be JSON, localhost-only, and source-grounded. Model-derived chunk frames are cached under a local cache directory keyed by chunk text and extraction version so repeated initialization does not repeat work.

## Provenance

DSPG objects are grounded in exact source spans. Answers at the public boundary are strings, but internal answer records keep evidence objects with relative source path, source text, and score. Future public diagnostic APIs can expose provenance without changing the simple `question(text) -> string` user contract.

## Current Weaknesses

- The deterministic fallback is still shallow and currently below the strict fixture gates after removal of semantic answer handlers.
- Entity resolution is local and conservative.
- Context propagation is sentence-level rather than fully hierarchical.
- Temporal modeling handles simple dated state statements but not full interval logic.
- Noise handling is structural and conservative; it labels and downweights low-semantic-content sources for ordinary fact retrieval while preserving them as source-grounded contexts.
- The local model path now includes chunk-frame extraction, query-frame parsing, bounded verification, and evidence extraction, but live-model throughput and JSON reliability are still active engineering constraints.
- The fixture suite now includes hard failure-driven raw reasoning tests, but it is still self-written and not proof of broad generalization.

## Optional Local Query Planner

KMD includes an optional local planning path for development. Candidate selection remains bounded before reasoning: lexical sentence search, DSPG relation/frame matches, neighboring discourse units, normalized metadata records, and natural filesystem metadata may contribute retrieval priors. Filesystem metadata can help locate a raw file, but answer facts must still be grounded in readable raw text spans unless the user explicitly asks about file metadata itself.

When enabled, the local-model path uses a localhost llama.cpp-compatible endpoint to produce generic JSON query frames, normalizes those frames with the deterministic frame builder, executes a bounded SQLite DSPG subgraph, verifies candidates from bounded evidence, and can fall back to source-grounded bounded evidence extraction when the graph does not support an answer. This path is disabled by default, never uses cloud APIs, and must remain independent of any external evaluation harness.
