# Architecture

KnowMoreDiRT is a raw-folder DRT/DSPG system. It accepts a folder tree, reads all readable files as raw text, builds an internal discourse provenance graph, and answers questions from that graph and its supporting text index.

## Public Boundary

The public boundary is intentionally minimal:

- `initialize(folder_path)`
- `question(text) -> string`

No metadata wrapper, manifest, semantic adapter, or external schema is part of the input contract.

## Ingestion Pipeline

Initialization performs these steps:

1. **Folder scan**: recursively traverse arbitrary folders and filenames, excluding only KMD's own configured/generated cache directories when they are inside the scanned root.
2. **Text read**: read each readable file as text.
3. **Natural metadata capture**: record filename, suffixes, parent path, directory depth, mode/permissions, uid/gid where available, inode/device where available, atime/ctime/mtime, symlink status, MIME guess, line count, word count, byte count, and SHA-256 content hash.
4. **Chunking**: split text into sentence/line-sized units while preserving source offsets.
5. **Source spans**: store both chunk spans and mention spans.
6. **Mention extraction**: extract source-grounded IDs, URLs, file-like values, names, and named text spans.
7. **Referent construction**: create local referents from exact mentions without requiring destructive global merging.
8. **Context assignment**: attach source-grounded sentence, quality, and temporal carriers to explicit context records.
9. **Surface-structure extraction**: record universal source structures such as label/value text, object-like key/value text, table cells, identifiers, URLs, and timestamps. These records preserve source structure but do not perform semantic event or role interpretation.
10. **Optional local-model discourse frames**: when `KMD_USE_LOCAL_MODEL=1` and LLM ingestion is enabled, each meaningful source chunk is sent to the localhost-only model for generic DRT/DSPG frame JSON. Accepted frames must be grounded by exact evidence text from the chunk before they are stored. Model arguments are converted into referents, frame arguments, same-surface identity hypotheses, and source-grounded semantic relations.
11. **Generic relation storage**: store deterministic surface records and model-produced DRS conditions in the same grounded SQLite representation, while keeping relation words as data rather than control-flow selectors.
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
- `drs_boxes`
- `drs_referents`
- `drs_conditions`
- `drs_condition_arguments`
- `drs_identity_hypotheses`
- `temporal_edges`
- `relations`
- `metadata_records`
- `model_attempts`

The current implementation uses an in-memory database by default. A durable user-configurable store path is planned.

KMD now has an explicit Python DRT layer in `knowmoredirt.drs`. It defines discourse referents, discourse arguments, discourse conditions, and discourse contexts as relation-agnostic objects. These objects are normalized into the SQLite DSPG store. Predicate and role labels remain data from source text or model output. They are not intent enums and do not select bespoke answer handlers. Generic identity hypotheses carry source-span provenance, and bounded identity expansion only follows accepted/same-referent-style hypotheses grounded in the currently loaded source spans rather than every hypothesis ever stored for the run. Expansion is iterative: newly loaded identity sources can expose another grounded equality edge, causing one more bounded rerank/reload round until a small fixed point is reached. Candidate or ambiguous DRS identity hypotheses remain preserved as DRS provenance, but they are not used as equality edges.

When the model returns full chunk DRS objects, KMD also materializes the declared DRS structure directly. Deterministic materialization validates declared IDs, source grounding, a single asserted root box, acyclic subordinate box links, temporal references, identity references, acyclic condition arguments, and provenance. It may dereference a declared referent ID to the model-provided referent label, or a declared box/condition ID to its model-provided evidence surface, so the graph executor can bind variables over first-class `frame_arguments`. It may repair only provenance-level issues: unescape evidence strings when the repaired string is an exact source substring, replace an ungrounded evidence string with the model-declared label/value only when that label/value is an exact source substring, clear stray target IDs on literal arguments, remove identity hypotheses whose evidence does not mention both declared sides, and drop temporal records that no condition references. Conditions with a model-declared `temporal_id` are projected into `temporal_edges`; if the model also supplied literal condition arguments, those literal values become the ordered value payloads. Monolithic chunk DRS decoding is schema-constrained to stable referent, box, condition, and temporal ID namespaces, and condition/argument evidence can be constrained to deterministic delimiter-boundary source-span candidates. In staged chunk DRS fallback, stage 1 is constrained to stable referent, box, and temporal id namespaces, stage 2 receives those declarations and must attach temporal records through `temporal_id`, and stage 1 is prompted to declare distinct subordinate boxes for scoped embedded content so conditions do not self-reference their containing box. Stage 2 condition evidence uses the same source-span constraint; the model still chooses conditions, predicates, roles, arguments, and temporal links, while deterministic code only supplies exact span options and validates the result. If validation finds a condition argument that points to a missing DRS box, KMD can make one additional constrained local-model call that emits only declarations for those missing box IDs, then accepts the merge only when the full DRS passes the same structural and exact-source grounding checks. Deterministic code does not infer temporal or scope semantics when the model leaves those links absent. If a validated model DRS contains referents and boxes but no conditions, or a delimiter-rich record validates with fewer than two model-produced conditions despite multiple field-like source spans, KMD may run the same staged constrained extraction path and keep it only when the staged DRS validates and adds conditions. It does not infer missing semantic roles or meanings from raw text.

In the staged chunk DRS path, stage 1 referent, box, and temporal evidence is also schema-constrained to deterministic source-span candidates. The model still chooses the DRS declarations and semantics; deterministic code only supplies grounded span options, stable ID namespaces, source-aware output budgets, cache keys, and validation.

For nested delimiter-rich or field-heavy chunks, KMD may raise the Stage 1 skeleton token budget before calling the model so the constrained decoder has room to declare the model-chosen referents, boxes, and temporal records. Flat delimiter records keep the smaller skeleton budget. Compact non-temporal chunks use a separate Stage 2 condition budget floor found by staged DRS readouts, while timestamped, long, or dense chunks keep the larger condition budget. These are budget policies only; they do not add semantic fields or Python-side role interpretation.

For chunks with several delimiter-boundary field-like source spans, KMD may schedule the staged constrained DRS extractor before the monolithic extractor. This is a call-order policy only: the model still produces all DRS referents, boxes, conditions, roles, and temporal links, and deterministic code still only supplies spans, stable namespaces, cache keys, and validation. If staged extraction fails, KMD can still try the monolithic path.

Document metadata stores natural filesystem/read metadata and text-quality metrics, including printable ratio, symbol ratio, token diversity, OCR-like token ratio, a low-semantic-noise flag, and a semantic-quality label. The same classification is also represented as a `quality:*` context so noisy source material remains preserved and queryable rather than discarded. Generic filesystem/read metadata is also normalized into `metadata_records`, while source quality, filesystem time, sentence context, and event-time signals are represented as context carriers and assignments.

## Retrieval and Query Execution

The current query path combines:

- lexical retrieval over raw sentence chunks,
- referent-centric retrieval through mentions and referents,
- frame-aware retrieval through observed predicates and frame arguments,
- relation-aware retrieval through generic label, identifier, temporal, table, and record relations,
- bounded SQLite subgraph execution over selected documents/chunks, source spans, mentions, referents, contexts, frames, frame arguments, temporal edges, and relations,
- local-model frame argument binding when semantic frames are present,
- temporal state retrieval for state changes with dated evidence or model-declared temporal values,
- text-quality downweighting so noise files do not dominate normal questions,
- conservative deterministic answer extraction over bounded candidates,
- ranking by anchor match, requested-relation term match, relation completeness, context validity, temporal recency, and text-quality signals.

Questions are parsed into generic query frames containing target anchors, requested relation text, relation terms, constraints, answer type, temporal scope, negation, aggregation, and evidence requirements. Relation words from a source or question remain data inside the frame; they do not select content-specific code branches. Query DRS JSON schemas cap array sizes for answer variables, target referents, temporal records, requested conditions, box requirements, constraints, and condition arguments so constrained llama.cpp decoding cannot run on by repeating argument objects.

For query DRS objects, deterministic validation requires the declared question string, schema version, namespace references, and evidence text to be grounded in the question. KMD may repair provenance strings by using model-declared labels, values, or role-label surface variants that are exact question substrings, may restore exact question casing, may align an argument's target kind with the namespace of its declared target ID, may repair an answer-variable argument to a declared target referent when its evidence exactly names that referent, and may prune duplicate answer-variable argument references inside a single requested condition. Condition and box evidence may fall back to the full question. It does not add query semantics or infer answer intent from Python-side wording rules.

This is a first vertical slice of the full DRS/DSPG query architecture. It avoids full-corpus graph loading per question and avoids assuming external input structure. The deterministic layer no longer attempts to parse active/passive events, copular assertions, or discourse modality by hand. Those semantic decisions belong to the local-model DRS extraction path. Future work should strengthen graph traversal, entity resolution, uncertainty handling, aggregation, discourse context propagation, and live-model throughput.

The bounded SQLite graph executor is part of the normal non-model answer pipeline for query plans that can be mapped to generic DSPG operations. The optional local model path uses the same executor after producing a constrained plan, so model assistance refines planning rather than replacing grounded graph execution.

For materialized DRS condition rows, relation-level modality can satisfy a query DRS scope requirement when the surrounding context is otherwise asserted, and it blocks unscoped access to that condition. This lets graph binding use scoped condition data supplied by the model without requiring Python-side discourse parsing, and without treating polarity fields as answer values.

When model frames are available, answer candidates can be produced by binding the query frame against frame arguments rather than by using a relation-name handler. The executor checks that the target anchors and requested predicate text match condition-local predicate/argument material, not merely unrelated text nearby in the same chunk or box. It then checks answer-slot text, context, and expected answer type before returning compatible non-target arguments as possible answer-variable bindings. For typed answers, a model frame with no matching role or value-type slot does not fall back to arbitrary arguments. Compound model slot labels such as `report_link` are expanded into lexical matching variants so they can unify with source-grounded structural fields while remaining data rather than code branches. Generic question-word answer placeholders are ignored as slot terms. Identity expansion starts from exact referent surfaces or exact token-set matches before following stored identity hypotheses; incidental substrings inside URLs, file paths, or longer labels do not seed identity expansion. This is the intended path for semantic roles, claims, reports, dreams, temporal events, and other discourse conditions.

When equally grounded, accessible answer candidates disagree across disjoint source spans and the query supplies no temporal/list/count operator that can resolve the disagreement, the bounded executor returns no answer and records internal conflict diagnostics with source paths, span IDs, chunk order, character offsets, source kind, and source text snippets. Temporal ordering only considers model-declared or structural temporal candidates whose DRS context is accessible to the query frame; a later reported state cannot override an earlier asserted state unless the query asks for reported scope. Unscoped temporal candidates also return no answer when model-declared temporal values support different bound values, even if those values came from the same source span. Explicit bounded conflicts and temporal ambiguities block local-model evidence fallback, so the system does not spend another call or let an evidence-only answer choose one side of an unresolved scoped graph conflict. Other no-answer paths attach bounded source provenance samples, including document metadata, chunk IDs, span IDs, span kind, chunk order, character offsets, and original text snippets. Local-model evidence answers use the same internal provenance shape, so a model-derived answer can still be traced back to exact file, chunk, span, offsets, document metadata, and original text. The public API still returns `unknown`, but the answer path preserves enough provenance for future diagnostic surfacing.

Before returning a non-unknown answer, KMD validates the answer against the answer type supplied by the query DRS. Deterministic code may classify only non-semantic value shapes, such as URL, identifier, file path, count, or date/time; person, actor, organization, state, and content-role readings remain model/query-DRS decisions. It does not infer the requested answer type from question wording. If the local-model path is active and cannot produce a complete grounded answer, the engine returns `unknown` instead of falling through to a second deterministic semantic interpretation. When the model path is inactive, the deterministic fallback can still bind source-grounded structural records with an `unknown` answer type and validate the final value by shape.

## Optional Local Model Integration

KMD includes an isolated local model client hook. The default system does not require a model and does not call cloud APIs. When explicitly enabled, model use is bounded and constrained in four roles:

1. **Chunk frame extraction**: convert raw chunks into generic DRT/DSPG frames with predicates, argument roles, polarity, modality/context, temporal text, confidence, and exact evidence text.
2. **Chunk DRS extraction**: optionally convert raw chunks into schema-constrained DRS objects with referents, boxes, conditions, temporal records, and identity hypotheses. If a monolithic chunk DRS fails JSON, schema, or exact-grounding validation, KMD can retry with staged extraction: stage 1 declares referents, boxes, and explicit temporal records with stable IDs and source-span-constrained evidence; stage 2 emits conditions constrained to those declared IDs. The same staged path can retry a validated but structurally sparse DRS, and the replacement is accepted only if it passes validation and adds conditions.
3. **Question frame parsing**: convert the question into the same generic query-frame language used by deterministic planning.
4. **Answer verification/extraction**: verify candidate answers against bounded evidence and discourse frames, or extract the shortest grounded answer from bounded evidence when graph execution cannot bind an answer.

The model is never allowed to use outside knowledge or external labels. All accepted output must be JSON, localhost-only, and source-grounded. Model-derived chunk frames are cached under a local cache directory keyed by chunk text and extraction version so repeated initialization does not repeat work. Validation failures that produce no grounded frames, such as invalid JSON, schema rejection, or grounding rejection, are cached as accepted=false empty-frame results; they are retried only when the prompt/schema/model cache key changes, and they are never inserted into the DSPG graph. During incremental ingest, `model_attempts` records chunk-frame attempts by run, source span, source, and model/prompt/schema policy cache context so unchanged deterministic validation failures are skipped even when no frame rows were materialized, while transient `request_failed` attempts remain retryable. Already materialized local-model frame rows are reused only when the current cache context also has an accepted materialized attempt recorded; when the model, prompt, schema, grammar, runtime, or policy cache context changes for the same source span, older local-model frame materialization rows are removed before the replacement attempt is recorded so stale predicates, arguments, identity edges, and frame temporal edges cannot remain active beside the current interpretation. The lazy per-question frame materializer uses the same cache-keyed attempt ledger and retry rule, so a bounded query does not repeatedly spend live model calls on a source span whose equivalent deterministic chunk-frame attempt already failed but can recover from endpoint/request interruptions. Bounded model evidence answers use the same discipline for malformed JSON/schema failures while avoiding caches for external request failures. The model fingerprint used by those cache keys includes the llama.cpp endpoint, model identity, context size and source, timeout, sampler settings, and completion transport settings such as completion/chat API mode and streaming mode.

Streaming llama.cpp responses are bounded by the local-model wall timeout as well as the socket timeout. If a stream keeps producing partial content without a complete JSON object past the configured budget, the request is aborted instead of waiting indefinitely.

Chunk DRS staged extraction uses the same cache discipline for non-request JSON failures at each constrained stage: malformed stage output is cached under the stage prompt/schema/model key, while request failures remain uncached so transient endpoint problems can be retried. During incremental ingest, `model_attempts` also records DRS attempts by run, source span, task, source, and model/prompt/schema policy cache context. Unchanged deterministic failed attempts are skipped on later ingest runs against the same store, transient `request_failed` attempts are retried in the same cache context, and all attempts are retried automatically when the model identity, runtime settings, prompt/schema/grammar versions, policies, or relevant request parameters change. Already materialized chunk DRS rows follow the same active-cache rule as chunk frames: current-cache materializations are reused, while older same-source-span model DRS boxes, conditions, identity hypotheses, temporal edges, frames, arguments, and orphan support rows are removed before a changed-cache attempt is accepted or skipped. Raw source spans, deterministic records, model attempts, and unrelated chunks remain intact.

During re-ingest, current document rows, metadata records, and filesystem-time context carriers are refreshed when the same content-stable document ID is encountered again. Bounded retrieval carries current sentence-derived chunk IDs instead of only `(document_id, chunk_order)` pairs, so a scan-policy change cannot load stale chunks that share an old order with the current file.
Document-local identity expansion is also bounded to the current sentence-derived chunk IDs for selected documents, so a previous chunking policy cannot reintroduce stale identity hypotheses from old source spans while still allowing current document-local bridge chunks outside the initial retrieval window.

When both chunk-frame and chunk-DRS ingestion are enabled, cached or previously materialized frame attempts skip only the frame call. They no longer skip the DRS ingest path for that sentence, so enabling DRS after an earlier frames-only run can materialize the missing DRS rows without repeating the frame call.

## Provenance

DSPG objects are grounded in exact source spans. Answers at the public boundary are strings, but internal answer records keep evidence objects with relative source path, source text, and score. Future public diagnostic APIs can expose provenance without changing the simple `question(text) -> string` user contract.

## Current Weaknesses

- The deterministic fallback is intentionally shallow after removal of semantic answer handlers and is currently below the strict fixture gates.
- Entity resolution is local and conservative.
- Context propagation is sentence-level rather than fully hierarchical.
- Temporal modeling handles simple dated state statements but not full interval logic.
- Noise handling is structural and conservative; it labels and downweights low-semantic-content sources for ordinary fact retrieval while preserving them as source-grounded contexts.
- The local model path now includes chunk-frame extraction, query-frame parsing, bounded verification, and evidence extraction, but live-model throughput, durable cache reuse, and JSON reliability are still active engineering constraints.
- The fixture suite now includes hard failure-driven raw reasoning tests, but it is still self-written and not proof of broad generalization.

## Optional Local Query Planner

KMD includes an optional local planning path for development. Candidate selection remains bounded before reasoning: lexical sentence search, DSPG relation/frame matches, neighboring discourse units, normalized metadata records, and natural filesystem metadata may contribute retrieval priors. Filesystem metadata can help locate a raw file, but answer facts must still be grounded in readable raw text spans unless the user explicitly asks about file metadata itself.

When enabled, the local-model path uses a localhost llama.cpp-compatible endpoint to produce generic JSON query frames, normalizes those frames with the deterministic frame builder, executes a bounded SQLite DSPG subgraph, verifies candidates from bounded evidence, and can fall back to source-grounded bounded evidence extraction when the graph does not support an answer. This path is disabled by default, never uses cloud APIs, and must remain independent of any external evaluation harness.

The model path is currently staged as:

1. chunk-to-DRS frame extraction with exact-span grounding and cache keys tied to prompt/schema/model settings,
2. question-to-query DRS construction,
3. bounded DRS/DSPG graph binding,
4. verifier validation of candidate answer compatibility, scope, provenance, and evidence,
5. short-timeout evidence fallback only when bounded graph binding cannot provide a complete answer.

Verifier output is rejected when its proposed answer is incompatible with the query DRS answer type. This prevents an absence statement, URL, identifier, or other wrong-shape value from passing a person, organization, URL, identifier, boolean, count, or content query merely because it is present in nearby evidence.

For scoped DRS questions, verifier prompting treats the candidate as the bound embedded proposition or scoped value when the query frame requires a reported/modal scope. The verifier still must ground that binding in raw evidence and discourse records; deterministic code does not rewrite the candidate into a scoped reading.

When `KMD_PROGRESS=1` or `KMD_EVAL_PROGRESS=1`, eager LLM ingestion emits concise per-chunk stdout lines for model-frame start and completion, including chunk counters, source path/order, cache/fresh result, validation reason, frame count, model elapsed time, and cumulative ingest elapsed time. This is observability only; it does not change extraction semantics.
