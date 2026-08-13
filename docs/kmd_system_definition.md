# KnowMoreDiRT System Definition

## Status and purpose

This document defines the target behavior and architecture of KnowMoreDiRT (KMD). It is the engineering specification for how the system must ingest information, represent discourse and scope, retrieve evidence, reason over evidence, answer questions, verify answers, evaluate itself, cache model calls, and operate at scale.

The source-derived KMD voice requirements are preserved separately in `docs/kmd_recording_requirements.md`. That document is intentionally not polluted by later design decisions. This document incorporates every substantive requirement from those recordings and resolves their open questions using the current KMD implementation, the independent DRT/devtests lineage, current test evidence, and external primary research.

This is a target specification, not a claim that every line is already implemented. Where the current implementation already provides a suitable mechanism, this document standardizes it rather than replacing it. Where current behavior is weaker than the target, the target wins.

## Design sources and evidence used

The design was derived from five evidence classes.

1. **KMD recordings.** Both KMD recordings from Jetson were reread in full. They define the non-negotiable product behavior: source-only factual answers, real-world default scope, explicit handling of differently scoped evidence, long-range discourse context, vector retrieval, DRT/DRS as operational structures, bounded context, and semantic answer judging.
2. **Current KMD implementation.** The current store already contains documents, chunks, source spans, mentions, referents, contexts, context carriers, context assignments, frames, DRS boxes, DRS referents, DRS conditions, temporal edges, general relations, provenance fields, confidence, and model-attempt records. The existing architecture is therefore close to the required representational direction.
3. **Current KMD tests.** The recording-focused tests currently pass and cover source-only answering, real-world versus dream/report scope, requested subordinate scope, implicit sleep-only scope, retroactive dream scope, hypothetical headers, dated sections, semantic vector retrieval, bounded DSPG behavior, qualified unknowns, and semantic judging.
4. **Independent DRT/devtests lineage.** `research/devtests_drt_reference` is independent recovery/acceptance evidence and includes raw-folder, no-overfit, benchmark, and DRT-oriented contracts. It is evidence for behavior and black-box validation, not production code to import blindly.
5. **External research.** Primary research supports the design decisions around scoped meaning representations, discourse structure, retrieval, bounded context, attribution, and semantic evaluation.

Key external references:

- Discourse Representation Structure parsing and the IWCS shared task describe DRS as a scoped meaning representation capable of representing negation, modality, quantification, and presupposition: https://aclanthology.org/W19-1201/
- The SDRT reference describes discourse as labeled segments linked by rhetorical/discourse relations such as Explanation and Contrast: https://homepages.inf.ed.ac.uk/alex/sdrt.html
- DRS graph conversion demonstrates that DRT meaning representations can be represented as directed labeled graphs: https://aclanthology.org/2020.conll-shared.2/
- Retrieval-Augmented Generation formalizes explicit non-parametric memory as a complement to parametric model memory: https://arxiv.org/abs/2005.11401
- Dense Passage Retrieval establishes dense semantic retrieval as a practical high-recall retrieval mechanism: https://arxiv.org/abs/2004.04906
- ColBERT shows the value of fine-grained late interaction rather than relying only on a single document vector: https://arxiv.org/abs/2004.12832
- Dense Hierarchical Retrieval shows that short passage representations can be misleading without document-level context: https://arxiv.org/abs/2110.15439
- Reciprocal Rank Fusion is a simple established method for combining independent rankings: https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
- Lost in the Middle demonstrates that merely supplying long context does not guarantee reliable use of information, especially when critical evidence is in the middle: https://arxiv.org/abs/2307.03172
- Self-RAG provides evidence that fixed indiscriminate retrieval can hurt and that retrieval/evidence should be evaluated rather than accepted blindly: https://arxiv.org/abs/2310.11511
- ALCE separates answer quality from citation/evidence quality and shows that citation support remains a distinct failure mode: https://arxiv.org/abs/2305.14627
- Answer-equivalence research documents that a single literal gold answer is insufficient because semantically equivalent answers can differ superficially: https://aclanthology.org/2021.emnlp-main.757/
- Reliable, Adaptable, and Attributable Language Models with Retrieval motivates retrieval-based systems specifically for reliability, adaptability, and verifiability: https://arxiv.org/abs/2403.03187

## Core system definition

KMD is a **corpus-grounded discourse-aware question-answering system**. Its factual output is constrained by an explicit external datastore. Language models are semantic parsers, query planners, evidence interpreters, reasoners, canonicalizers, and judges; they are not hidden factual databases for KMD answers.

A valid KMD answer must satisfy three independent conditions:

1. **Semantic correctness:** the answer addresses the user's question correctly.
2. **Scope correctness:** every factual claim is valid in the context/circumstance in which KMD presents it.
3. **Evidence support:** every factual claim is supported by identifiable corpus evidence, with traceable provenance.

Failure of any one condition makes the answer invalid even if the prose sounds plausible.

## System boundary and public contract

KMD is a technical retrieval, semantic-representation, and reasoning service. It is not the conversational agent. Dialogue management, tone, markdown, user-facing clarification strategy, and social/conversational behavior belong to a layer above KMD. That layer may rephrase a verified technical result but may not manufacture evidence, change scope, or turn an `unknown`/conflict into a guess.

The intended user-facing KMD API remains deliberately small:

- `initialize(folder_path)` ingests and initializes a raw folder.
- `question(text) -> str` asks one technical question and returns a plain rendered answer string.

The folder is the source boundary. It may contain nested folders, arbitrary filenames/extensions, prose, logs, transcripts, tables-as-text, JSON-like material, code-like material, malformed text, or meaningless text. KMD does not require a prepared external schema, semantic wrapper, benchmark manifest, or gold-bearing adapter as its public input contract. Specialized benchmark preparation may transform a benchmark into an equivalent raw source corpus only to enforce benchmark leakage boundaries; it is not a requirement imposed on ordinary KMD users.

Internally KMD keeps the structured answer state, evidence, contexts, provenance, confidence/sufficiency, graph records, and diagnostics defined later in this document. The stable public string API does not prevent richer internal/debug interfaces, but such interfaces must not change the semantics of the two public functions. Calling `question` before successful initialization is an error. Unanswerable, empty, malformed, or unsupported questions return an unknown-style answer rather than fabricated factual content.

## Division of responsibility: models versus deterministic code

Language models translate natural language into constrained structured representations and can perform bounded semantic interpretation over supplied evidence. Deterministic code owns validation and execution. In particular:

- models may propose query DRS, source DRS, context classifications, referent hypotheses, canonicalizations, and bounded answers;
- deterministic code validates schemas, IDs, source grounding, scope accessibility, polarity, time, identity links, cache/model compatibility, and provenance;
- vectors retrieve candidates but never determine truth;
- DRT/DRS/DSPG structures constrain reasoning;
- the persistent store preserves the source-grounded semantic state;
- invalid or ungrounded model output is rejected or marked unresolved rather than silently repaired into a different semantic claim.

This boundary prevents KMD from degenerating into a model-owned hidden reasoning path that merely decorates outputs with database-looking metadata.

## Non-negotiable invariants

### Source-only factuality

All factual claims in a normal KMD answer must be supported by corpus evidence or by an explicit derivation whose premises are themselves corpus-supported. The model may use pretrained linguistic competence, semantic associations, and general reasoning procedures to interpret evidence or propose retrieval hypotheses, but may not promote unsupported world facts from its weights into evidence, semantic state, or answer claims.

This boundary must exist in every model-call prompt that can influence factual output. It is insufficient to put the instruction only in the final-answer prompt because an earlier extractor, canonicalizer, reasoner, or verifier can otherwise inject unsupported facts into an intermediate representation. Query expansion is a special non-evidentiary operation: a model may use parametric semantic knowledge to suggest search terms or hypotheses, but those suggestions are never facts and can affect the answer only after corpus retrieval supplies supporting evidence.

If a user explicitly requests model/world knowledge outside the KMD corpus, that is a different operating mode and must be explicit in the public API and answer metadata. It must never be silently enabled.

### Real-world default scope

When the user asks an underspecified factual question, the default requested scope is the ordinary real world. A subordinate context such as a dream, hypothetical exercise, fictional story, quotation, report, simulation, counterfactual, or dated historical state cannot directly satisfy that default query unless the context is compatible with real-world assertion.

This does not mean KMD hides subordinate evidence. If such evidence is relevant, it is surfaced as related evidence with its scope stated explicitly.

### No scope promotion

A proposition must never be promoted from a narrower or incompatible scope into a broader scope merely because its text looks factual.

Examples:

- A rule in a dream is not a real-world rule.
- A hypothetical exercise is not an operational policy.
- A person's report is not automatically an independently established fact.
- A quotation is attributed to the speaker/source and is not automatically endorsed by the enclosing document.
- A statement under a date applies to that temporal frame unless the document establishes broader validity.
- A negated proposition cannot be retrieved as a positive fact.

### Qualified unknown

When compatible evidence does not establish the requested proposition, KMD must return a qualified unknown rather than inventing an answer. A qualified unknown has two layers:

1. A direct status: the requested proposition is not established by the available corpus in the requested scope.
2. Optional related evidence: relevant evidence from another scope, source, date, report, or uncertain frame, clearly qualified so the user can still benefit from it.

The canonical semantic behavior is therefore not just `unknown`; it is `unknown in requested scope + scoped related evidence when useful`.

The canonical recording-derived acceptance example is explicit. Suppose the corpus contains no official German or common traffic-law source establishing that bicycles may not drive faster than cars, but it does contain a file in which Timmy dreams that somebody explains exactly such a rule. For an ordinary real-world question about whether that bicycle rule is law, KMD must not answer from pretrained knowledge and must not promote the dream. The direct answer is that the available corpus does not establish the requested real-world law or that no supporting official law text was found. KMD should then surface the relevant dream evidence with its scope stated clearly: the corpus contains a file in which Timmy dreams that somebody describes such a rule. If the user instead explicitly asks what rule appeared in Timmy's dream, that dream-scoped proposition is direct evidence for the requested dream-scoped question.

### Provenance is mandatory

Every proposition used for answering must remain traceable to source spans, documents, and context carriers. Generated summaries and canonicalizations may accelerate reasoning but never replace the original evidence chain. Derived claims must additionally preserve a derivation trace linking the claim to its corpus-supported premises and the deterministic or validated reasoning operation that produced it.

### Scope is not truth

Scope describes the world, circumstance, speaker, modality, time, or discourse environment in which a source presents a proposition. Scope compatibility determines whether evidence is eligible to answer a query, but it does not by itself prove that the proposition is objectively true. An ordinary `real_world` assertion means that the source presents the proposition as an ordinary-world claim; it is not a certificate that the source is correct. Source authority, directness, contradiction, temporal applicability, and provenance remain separate evidence dimensions. KMD reports what its corpus establishes under those dimensions; it must not claim external verification that the corpus does not provide.

## Information model

KMD uses a layered semantic graph, referred to operationally as the DSPG/DRS representation. DRT supplies scoped semantic boxes and referents; discourse structure supplies relationships among segments and contexts; KMD adds provenance, confidence, temporal information, retrieval metadata, and operational IDs.

### Layer 1: immutable source layer

The source layer preserves what was actually ingested.

Each document has at least:

- stable document ID;
- absolute and relative path or source identifier;
- content hash;
- size and file timestamps when applicable;
- media/type metadata;
- extraction metadata;
- immutable source text or a reproducible pointer to it.

Each chunk has at least:

- stable chunk ID;
- document ID;
- deterministic chunk order;
- source character offsets;
- exact text;
- token estimate;
- content hash.

Each evidence span has:

- span ID;
- document ID;
- chunk ID;
- exact source offsets;
- exact surface text;
- span kind.

The source layer is never rewritten by model inference.

### Layer 2: referential layer

Mentions and referents represent entities and values while preserving uncertainty.

A mention is a source-grounded surface occurrence. A referent is a canonical entity/value hypothesis. Mention-to-referent links carry status and confidence. Identity hypotheses must remain hypotheses until sufficiently supported; entity merging must never destroy the original mention provenance.

### Layer 3: context/scope layer

A `context` is a semantic environment in which propositions are interpreted. Contexts form a parent-child hierarchy but may additionally participate in discourse relations.

Required core context kinds are open-ended but must include at least:

- `real_world` / ordinary asserted context;
- `dream`;
- `hypothetical`;
- `counterfactual`;
- `fictional` or narrative fiction;
- `reported` / attributed speech or hearsay;
- `quoted`;
- `conditional`;
- `document_frame` / document-level declared frame when a document or section explicitly establishes one;
- `temporal` or dated state;
- `simulation` / exercise;
- `uncertain_scope` for context inferred but not established strongly enough for hard promotion.

A context carries:

- context ID;
- kind;
- parent context;
- holder/speaker/experiencer when applicable;
- evidence surface establishing the context;
- confidence;
- source provenance;
- optional temporal interval;
- optional authority/source classification;
- optional modality and epistemic status.

A **context carrier** is the text that creates, modifies, closes, or retroactively establishes a context. Examples include “I dreamed that…”, a section heading, “suppose that…”, “then I woke up and it had all been a dream”, a quote delimiter plus speaker attribution, or a dated section header.

A **context assignment** links a context to the propositions, spans, sections, chunks, or DRS boxes it governs.

### Layer 4: DRS layer

A DRS box is a scoped semantic unit. It may contain referents and conditions and may be nested under another box.

A DRS condition must represent at least:

- predicate;
- arguments;
- box/context;
- polarity;
- modality;
- temporal value/edge where relevant;
- evidence surface;
- confidence;
- source provenance.

This structure is specifically required so negation, modality, conditionals, reported content, hypothetical content, quantification, and other scope-sensitive phenomena are not flattened into unqualified triples. DRS accessibility rules are part of execution semantics: a referent or condition available inside a subordinate box does not automatically escape into its parent or sibling scope. Quantifiers, negation, implication, and modal boxes must preserve their accessibility constraints during retrieval and reasoning.

### Layer 5: discourse graph

Document meaning is not only a tree of boxes. KMD must represent discourse relationships among segments when they affect interpretation. At minimum, the graph must support relations such as:

- continuation/sequence;
- elaboration;
- explanation;
- contrast;
- condition;
- consequence/result;
- attribution/reporting;
- correction/revision;
- scope-open;
- scope-close;
- retroactive-scope;
- temporal-before/after/overlap;
- same-topic/section membership.

The relation vocabulary may evolve, but relation edges must be typed, source-grounded, confidence-bearing, and versioned.

### Layer 6: operational semantic layer

Frames, normalized relations, bounded DSPG propositions, retrieval embeddings, and model-derived canonical forms are operational indexes over the source/DRS graph. They can be regenerated and cached. They are never authoritative independently of their source chain.

## Storage, persistence, schema versioning, and concurrency

The DSPG/DRS database is persistent semantic state, not a disposable cache. Raw source remains the ultimate evidence, but accepted documents, spans, referents, contexts, DRS structures, temporal edges, relations, provenance, and model-attempt metadata are durable queryable records.

SQLite is the reference/local backend because it is transactional, inspectable, and sufficient for development and moderate corpora. A file-backed SQLite store must use foreign-key enforcement, write-ahead logging, a bounded busy timeout, and durable synchronization appropriate to production correctness. In-memory stores are test-only.

Initialization/update writes are transactional at coherent boundaries. A crash must not expose half-materialized semantic objects as a completed corpus state. Model-derived records are associated with extraction/run IDs and source/config fingerprints so incomplete runs can be distinguished, resumed, rolled back, or superseded.

The logical schema is versioned. Schema migration is explicit and tested; production code must not silently reinterpret old rows under new semantics. Regenerable indexes and model-derived annotations carry implementation/prompt/model fingerprints. Immutable source identity and provenance survive migrations.

For corpora that exceed practical SQLite concurrency or size limits, scaling must replace the physical backend while preserving the logical store contract. A server database may add partitioning, concurrent ingestion, online backup, and monitoring, but must not weaken DRT accessibility, provenance, transactions, or query semantics. Storage scale is never a reason to replace the discourse model with raw model output.

### Filesystem semantic catalog

The filesystem semantic catalog is the fast source/index layer. It preserves filesystem metadata, extracted text, chunks, hashes, embeddings, literal retrieval, and semantic retrieval. It may also contain regenerable model-generated retrieval annotations. It is not the authoritative DRT/DSPG reasoning store and does not independently establish truth. KMD consumes it for candidate evidence and retrieval acceleration while DRT/DSPG remains the persistent semantic reasoning layer.

## Scope semantics and propagation rules

### Rule 1: explicit carriers dominate local surface form

If an explicit carrier establishes a scope, all governed propositions inherit that scope regardless of whether their local text independently looks factual.

### Rule 2: scope persists until closed or superseded

A scope opened at a header, paragraph, section, quote, or narrative point remains active through its governed region until an explicit close, structural boundary, or stronger carrier ends it.

### Rule 3: retroactive scope is first-class

Later text may establish scope over earlier text. The ingestion/context pass must therefore support revising assignments after later evidence is seen. A purely left-to-right irreversible chunk parser is insufficient.

The canonical Timmy case—“then I woke up and it had all been a dream”—must attach the prior narrative region to a dream context even though the carrier occurs later.

### Rule 4: local absence of a carrier is not evidence of global real-world scope

A chunk with no special marker must inherit document/section/parent context before it is treated as ordinary real-world assertion.

### Rule 5: uncertain implicit scope is conservative

When a model infers that a passage is probably a dream/hypothetical/etc. without decisive evidence, KMD must store the inference with confidence and `uncertain_scope` semantics. Such a proposition is not eligible to establish an unconditional real-world answer. It may be surfaced as uncertain related evidence.

This resolves the recording's “proven dream versus merely dream-like” problem conservatively: uncertainty restricts promotion rather than broadening it.

### Rule 6: date is scope, not decoration

Dated sections create temporal contexts or temporal constraints. Two otherwise identical propositions under different dates remain distinct evidence states. Query time constraints select compatible states rather than collapsing them.

### Rule 7: declared authority and verified authority are distinct

A header such as “Official German Traffic Law” establishes that the document declares or frames itself as official. That declaration is source evidence and can govern interpretation of the section, but self-description alone does not prove external legal authority. KMD must keep `declared_authority` separate from `verified_authority` or equivalent provenance-backed authority metadata. Verified authority requires corpus-visible provenance or metadata sufficient to establish it. Professional-sounding prose or an “official” label alone cannot create verified authority.

### Rule 8: quotations and reports preserve attribution

Reported or quoted propositions carry the reporting/quoting holder. They may answer questions about what someone said or reported. They do not automatically answer whether the proposition itself is true in the real world.

### Rule 9: contradictions are preserved

Conflicting propositions are not merged into a single value. They remain separate evidence objects with source, scope, time, authority, and confidence. Answering selects or summarizes conflicts according to query scope and evidence quality.

## Ingestion architecture

Ingestion has two distinct passes because local chunking alone cannot solve long-range context.

### Pass A: deterministic source extraction

1. Discover files/sources deterministically.
2. Extract text without model interpretation.
3. Preserve source offsets and metadata.
4. Segment into chunks using deterministic rules and stable IDs.
5. Store documents, chunks, and source spans.
6. Compute embeddings for retrieval units.

### Pass B: semantic/discourse analysis

1. Analyze document-level structure, headings, sections, narrative boundaries, quotations, dates, and other carriers.
2. Create context carriers and context assignments.
3. Parse semantic propositions/DRS conditions.
4. Resolve cross-chunk referents and discourse relations.
5. Perform a backward/retroactive reconciliation pass so later carriers can affect earlier spans.
6. Validate that each model-derived proposition has source evidence and a context assignment.
7. Materialize bounded operational DSPG structures.

The semantic pass may use bounded windows and staged analysis internally, but it must operate with document-level state so cross-window scope survives.

## Query processing architecture

Every query proceeds through explicit stages.

### Stage 1: query semantic analysis

The query planner converts the user question into a structured query frame containing at least:

- requested proposition/intent;
- entities and values;
- requested scope;
- temporal constraints;
- authority/source constraints;
- polarity;
- expected answer type;
- whether the user explicitly asks about a subordinate context;
- retrieval concepts/terms;
- optional query expansions.

If no scope is stated, requested scope defaults to real-world assertion.

### Stage 2: bounded query expansion

The LLM may generate semantically related retrieval concepts, as required by the recordings. Expansion is bounded and recorded. The default policy is:

- keep the original query as the primary representation;
- generate a small configurable set of semantically distinct expansions;
- preserve the original named entities, exact identifiers, dates, and quoted strings unchanged while permitting additional aliases or related forms as separate expansion hypotheses;
- reject expansions that change the requested scope or answer type;
- deduplicate semantically redundant expansions;
- log expansions for reproducibility.

There is no universal magic expansion count. The count is a production setting and must be calibrated on held-out retrieval tests. The specification requires boundedness and intent preservation, not a hardcoded number derived without evidence. As a concrete recording-derived example, a broad query such as “what is a cat” may retain `cat` as the primary query while adding retrieval hypotheses such as pets, household animals, biology, or other semantically related concepts. Those expansions increase recall but never become answer facts until corpus evidence supports them.

### Stage 3: candidate retrieval

KMD uses multiple retrieval channels.

1. **Dense/vector retrieval is mandatory.** The recordings explicitly require embedding-distance search. Production mode must therefore not silently disable vector retrieval.
2. **Lexical/exact retrieval is complementary.** Exact identifiers, names, codes, dates, and rare terms are often better served by lexical search. Lexical retrieval does not replace dense retrieval.
3. **Structured retrieval.** Known referents, dates, contexts, predicates, document metadata, and DRS relations are queried directly when the planner provides structured constraints.

The result lists are fused deterministically. Reciprocal Rank Fusion is the default fusion algorithm because it combines heterogeneous rankings without requiring raw-score comparability. The fusion algorithm and constant remain configurable and benchmarked.

### Stage 4: scope/discourse expansion

Raw top-k hits are not the final evidence set. For every candidate proposition/chunk KMD expands the evidence graph to include interpretation-critical context:

- governing context ancestors;
- context carriers;
- section/header carriers;
- retroactive scope edges;
- preceding/following discourse segments linked to the candidate;
- temporal anchors;
- attribution holder/speaker;
- source/provenance spans;
- contradictory propositions in the same requested scope when relevant.

This stage is the direct architectural answer to the Timmy problem. The previous ten chunks are not included merely because “ten” was mentioned in the recording. The graph determines what context is semantically required. A configurable physical-neighbor fallback exists only when semantic structure is incomplete.

### Stage 5: compatibility filtering

Each candidate is classified relative to the query as:

- `direct_support` — scope compatible and proposition supports the requested fact;
- `direct_contradiction` — scope compatible and proposition contradicts it;
- `related_scoped` — relevant but scope incompatible;
- `uncertain_scope` — relevant but context confidence is insufficient;
- `background` — useful for interpretation but not direct answer evidence;
- `irrelevant`.

Only `direct_support` and `direct_contradiction` relative to the requested scope can establish the main answer. For an underspecified factual query that requested scope is the real world; for an explicit dream, historical, reported, hypothetical, or other scoped query, direct evidence is evaluated inside that requested scope. Related or uncertain evidence can appear in qualifications but cannot silently determine the main answer.

### Stage 6: reranking

Candidates are reranked using a score that considers:

- semantic relevance;
- exact-term relevance;
- query-scope compatibility;
- answer-type compatibility;
- temporal compatibility;
- authority/source compatibility;
- evidence confidence;
- graph distance from the matched proposition to its required carriers;
- redundancy/diversity.

Scope compatibility is a hard semantic feature, not just another weak relevance score.

### Stage 7: bounded evidence pack construction

The final model context is a bounded evidence pack, not a raw dump of top-k chunks. A flat “question + every retrieved chunk” prompt is still a valid baseline and may work well in ordinary cases when the result set is small, self-contained, and comfortably within the model context. It is not sufficient as the sole production architecture because it provides no guarantee that required discourse scope is preserved or that large result sets remain usable. The recording's concrete scale example must remain part of capacity testing: a query may retrieve roughly 100 chunks of roughly 1,000 tokens each, already approaching or exceeding the usable context budget of many models before the question, instructions, provenance, and scope carriers are added. KMD must handle that case without assuming that the whole raw result set can be pasted into one prompt.

The pack is assembled in priority order:

1. user question and structured query frame;
2. direct evidence spans;
3. the scope carriers required to interpret those spans;
4. provenance/source metadata;
5. contradictory evidence where applicable;
6. related subordinate-scope evidence;
7. optional explanatory background.

Required scope carriers cannot be evicted while their dependent proposition remains in the pack. If the token budget cannot contain a proposition plus the context necessary to interpret it, that proposition is removed or replaced by a source-faithful structured representation that preserves every scope, polarity, attribution, temporal, and provenance feature required for the current reasoning step. KMD must not claim that arbitrary semantic compression is globally lossless.

This explicitly avoids the long-context failure mode documented by Lost in the Middle: KMD does not assume that increasing context length automatically solves retrieval/attention problems.

## Malformed, noisy, and meaningless input

KMD applies the same source-grounded semantics to clean prose and ugly input. It must not introduce special domain-specific handlers merely because a question is profane, childish, grammatically broken, vague, or nonsensical.

If a gibberish token or malformed passage exists in the corpus, it remains literal retrievable evidence. If it does not exist and no grounded semantic relation supports the query, KMD returns unknown. The test corpus must therefore include database-backed gibberish, absent gibberish, fantasy, children's writing, word salad, malformed grammar, profanity, hostile phrasing, ambiguous names, vague requests, and serious questions about meaningless documents.

Noise classification may prevent useless model work during ingestion, but it must not destroy retrievability of exact source material or invent semantic meaning for nonsense.

## Similarity threshold policy

The recordings require a vector similarity threshold but do not justify a permanent numeric value. The system therefore uses an empirically calibrated threshold.

The production threshold must be selected using held-out retrieval cases covering:

- direct lexical matches;
- paraphrases;
- distant semantic matches;
- exact identifiers;
- hard negatives with similar wording;
- scope-carrier chunks that do not repeat query terms;
- multi-hop/discourse-linked evidence.

The optimization target is not just top-k answer recall. It is **scope-complete evidence recall at bounded candidate volume**. A threshold that retrieves the proposition but misses its necessary carrier is insufficient.

Threshold calibration must be repeated when the embedding model, preprocessing, chunking strategy, or corpus distribution changes.

## Answer-generation contract

The answer generator receives only the structured query and approved evidence pack. Its prompt must contain the source-only invariant.

Every generated answer has an internal structured form even if the public API renders plain text.

Required internal fields:

- `status`: `answered`, `unknown`, `conflicted`, or `partially_answered`;
- `answer_text`;
- `requested_scope`;
- `direct_evidence_ids`;
- `related_evidence_ids`;
- `contradiction_ids`;
- `confidence` or evidence sufficiency assessment;
- `provenance`;
- `scope_qualifications`;
- `answer_type`.

### Answered

Use when compatible evidence establishes the requested proposition.

### Unknown

Use when the corpus does not establish the proposition in the requested scope. The rendered answer should say that KMD did not find sufficient compatible evidence. It may then mention useful differently scoped evidence.

### Conflicted

Use when credible compatible evidence supports mutually incompatible answers and no deterministic source/authority/time rule resolves the conflict. KMD must report the conflict rather than choose arbitrarily.

### Partially answered

Use when the question has multiple requested components and only a subset is supported.

## Evidence and citation contract

Every factual sentence in an answer must be supportable by one or more evidence IDs. KMD should expose citations/source references in public output when the interface supports them.

Citation correctness and answer correctness are separate. A fact can be semantically correct but cited to the wrong scope or source; that must fail evidence validation.

The verifier must check:

- each factual claim has evidence;
- cited evidence entails/supports the claim in the displayed scope;
- no cited subordinate context is presented as direct real-world evidence;
- no source-free factual clause was introduced;
- contradictions are not omitted when they materially change the answer.

This adopts the separation emphasized by ALCE and attributable-retrieval research: answer fluency/correctness does not prove evidence support.

## Pretrained-memory isolation

It is impossible to erase pretrained knowledge from an LLM at inference time, so KMD enforces a behavioral boundary rather than pretending the model has no memory.

The enforcement stack is:

1. source-only prompts at every factual model stage;
2. structured evidence input, not open-ended “answer from what you know” prompts;
3. answer schemas requiring evidence IDs;
4. a verifier that rejects unsupported claims;
5. synthetic counterfactual tests where corpus facts intentionally contradict common world knowledge;
6. source-absence tests such as “What is the capital of France?” with no supporting corpus entry, which must return unknown;
7. audit logs of model inputs/outputs, cache keys, and accepted/materialized attempts.

A model may use its linguistic/world knowledge to understand that “woke up” can signal dream termination, but it may not use world knowledge to answer the bicycle-law proposition unless the corpus supports it.

## Uncertainty model

KMD separates three concepts that must not be conflated:

- **model confidence:** how certain the parser/model is about its interpretation;
- **evidence strength:** how directly the source establishes the proposition/context;
- **answer sufficiency:** whether evidence is adequate to answer the query in its requested scope.

Confidence values are metadata, not truth probabilities. A high model confidence cannot turn a dream proposition into a real-world fact.

For implicit contexts, the default policy is conservative:

- explicit/strongly established context -> normal scoped assignment;
- plausible but ambiguous context -> `uncertain_scope`;
- insufficient evidence for any special scope -> inherit validated parent/document context.

## Open-world semantics, negative evidence, and completeness

KMD is open-world by default. Failure to find evidence for proposition `P` does not establish `not P`, and failure to find evidence for `not P` does not establish `P`. Absence of evidence produces `unknown` unless the corpus explicitly supplies completeness semantics for the relevant domain.

Negative propositions require evidence just like positive propositions. Explicit statements such as “the valve is not open” directly support a negative proposition. Observational statements such as “inspection found no crack” can support an absence proposition when their semantics and inspection scope establish that claim; they must not be reduced to simple keyword negation.

Closed-world reasoning is allowed only when corpus evidence establishes a closed set or completeness condition, for example an authoritative exhaustive inventory, a declared complete table, or an explicitly bounded database relation. The completeness condition itself must be represented and cited.

Questions containing `all`, `every`, `none`, `only`, exact counts, superlatives, or exhaustive-list intent require completeness reasoning. KMD may return a complete enumeration or exact count only if it can justify that the relevant search space is closed or exhaustively covered. Otherwise it returns a partial answer or qualified unknown and states the limitation.

Derived arithmetic, comparison, date calculation, set operations, and other deterministic computations are permitted when all input premises are corpus-supported and the derivation is recorded. Whenever a deterministic operation is available, it is preferred over asking an LLM to perform exact arithmetic or set bookkeeping.

## Contradiction and source-authority policy

KMD never resolves contradictions using pretrained model preference.

Resolution uses only corpus-visible features:

1. requested scope compatibility;
2. explicit source authority metadata;
3. temporal applicability;
4. directness of evidence;
5. correction/supersession discourse relations;
6. provenance and evidence strength.

Model/parser confidence may help decide whether an extraction is reliable enough to use, but it must never override stronger source evidence or act as a truth vote between conflicting propositions. If corpus-visible scope, authority, time, correction/supersession, directness, and provenance do not resolve the conflict, the answer is `conflicted`.

Official-looking wording alone is not verified authority evidence.

## Document-context strategy

The recording explicitly left open “whole file versus structured context.” The decision is hybrid and deterministic.

KMD's primary mechanism is structured context representation. Whole-document/raw-section loading is a fallback, not the normal answer path.

Use whole-document or large-section context during ingestion when needed to discover carriers and cross-chunk relations. During query answering, retrieve only the minimal scope-complete evidence subgraph that fits the bounded context budget.

Fallback raw-neighbor expansion is allowed when:

- semantic parsing failed or is incomplete;
- a candidate lacks a validated context assignment;
- the relevant carrier cannot be resolved structurally;
- a diagnostic/repair pass is explicitly running.

Fallback radius is configurable and expands adaptively until a structural boundary, token budget, or confidence criterion is reached. A fixed “ten chunks” rule is not the architecture.

## DRT/DRS role

DRT/DRS is not a decorative semantic export. It is part of the execution path.

It must affect at least:

- scope compatibility;
- negation handling;
- modality/hypothetical handling;
- attribution;
- temporal selection;
- referent resolution;
- evidence expansion;
- contradiction detection;
- bounded context construction.

A test that passes while DRS is disabled or ignored is not evidence that the DRS implementation works. The benchmark suite must include ablations that verify DRS/context structures materially affect hard cases.

## Model-call architecture

Model calls are deterministic services around explicit contracts.

Each model call has:

- task name;
- exact model identity/revision;
- prompt/schema version;
- normalized input hash;
- output schema;
- timeout;
- retry policy;
- token budget;
- cached result when the complete call fingerprint matches;
- persistent attempt log.

No model call may silently switch to another model or endpoint under the same cache key.

### KMD-wide model-call cache

All model-call caches use one canonical KMD-wide root:

`/data/var/knowmoredirt/model_cache`

The cache root is independent of benchmark, suite, corpus, or run. Cached outputs are not independent of their inputs: a model call is reusable only when its complete fingerprint matches, including all corpus-derived text or structured evidence supplied to that call. Cache reuse therefore cannot turn outputs from one corpus into facts for another corpus unless the complete model-call input is actually identical. Cache namespaces such as chunk DRS, query DRS, query plan, evidence answer, verifier, canonicalization, identity, source resolution, document context, and evaluation judge live under the same root.

Cache keys must include every factor that can alter model output, including model identity/revision, prompt/schema version, relevant generation parameters, normalized input, and task implementation version.

Ambiguous legacy collisions are never arbitrarily reused. If the same key has different historical outputs, the active entry is removed and the call is recomputed.

### Corpus/index caches are separate

Filesystem/vector catalogs are not model-call caches. They are corpus-derived indexes and therefore use corpus/config fingerprints. They remain separate from the global model-call cache and are reusable only when corpus, chunking, embedding model/revision, and relevant index settings match.

## Untrusted corpus text and instruction isolation

Corpus text is data, never control-plane instruction. A source document may contain text such as “ignore previous instructions,” fake tool commands, fake system prompts, or instructions addressed to an AI. Unless the user's question is specifically about those instructions as document content, they have no authority over KMD's prompts, retrieval policy, tools, configuration, evidence rules, or answer contract.

Every model prompt that includes corpus text must delimit that text as untrusted evidence. Source text cannot change the source-only invariant, request external tools, alter model/cache settings, suppress citations, or redefine the user's query. If a document contains executable/code-like material, KMD interprets it as source content unless a separate explicitly authorized execution subsystem exists; ordinary KMD ingestion and question answering never executes corpus-provided commands.

## Reliability and transport

Every external/local model or embedding transport must have:

- explicit connect/read timeout;
- bounded retries;
- retryable-status allowlist;
- exponential backoff with bounded maximum;
- structured error logging;
- no duplicate retry loops stacked accidentally at multiple layers;
- explicit model-ID verification where an endpoint advertises models.

A failed semantic extraction cannot silently become fabricated structured data. It must be logged and either retried, represented as unresolved, or handled by an explicit fallback.

## Configuration

All tunable production parameters and operational behavior must be controlled by the centralized XML configuration system with documented settings, defaults, types, units, ranges/choices, descriptions, risk/change-frequency metadata, and precedence. Core semantic invariants defined by this specification are code contracts and must not be made optional merely because a configuration switch could bypass them.

Precedence is:

1. explicit environment override;
2. user/config XML override;
3. packaged production defaults.

Critical benchmark settings must be frozen into a run-compatibility manifest so a resumed run cannot silently change model, prompt, corpus, retrieval, context, or evaluation configuration.

No benchmark wrapper should invent hidden constants that bypass the central configuration contract.

## Logging and observability

Persistent rotating logs are required for:

- initialization;
- source ingestion;
- semantic extraction;
- model attempts and retries;
- cache hit/miss behavior;
- retrieval candidate generation;
- scope expansion;
- evidence selection;
- answer generation;
- verification;
- benchmark phase transitions;
- scoring.

Logs must include stable run IDs and enough identifiers to reconstruct why a proposition or answer was accepted without logging secrets unnecessarily.

## Public API behavior

The public API continues to return a simple answer string through `question(text) -> str`, while the internal answer state preserves structured diagnostics. A debug/internal API may expose this state without changing the stable public contract.

At minimum, the internal answer object contains:

- rendered answer;
- status;
- evidence list;
- source/document identifiers;
- relevant source spans;
- scope/context labels;
- optional temporal labels;
- confidence/sufficiency metadata;
- diagnostic reason for unknown/conflicted answers when requested.

The default user-facing response should remain concise. Deep provenance/DRS details are available through structured output or debug interfaces rather than dumped into every answer.

## Evaluation architecture

KMD evaluation has multiple independent layers.

### 1. Deterministic structural checks

Before using an LLM judge, verify properties that do not require semantic judgment:

- expected question ID exists;
- answer schema is valid;
- required evidence IDs exist;
- evidence belongs to allowed source corpus;
- no gold answer leaked into retrieval input;
- requested scope metadata is preserved;
- real-world answers do not cite only incompatible subordinate contexts;
- cache/run compatibility matches.

### 2. Semantic answer-equivalence judge

An LLM judge compares candidate and gold semantics, not wording. The judge receives the question, gold/reference semantics, candidate answer, and the expected scope behavior. It must accept grammatical/paraphrastic equivalents and reject meaning changes.

Judge calls must be deterministic as far as practical: temperature zero, fixed prompt/schema, exact model identity, KMD-wide cache, and structured JSON result.

### 3. Evidence/scope judge

A separate check determines whether the answer is supported by the cited evidence under the correct scope. This prevents a semantically plausible answer from passing solely because it matches the gold by coincidence.

### 4. Retrieval evaluation

Measure retrieval separately from end-to-end answering:

- proposition recall;
- scope-carrier recall;
- scope-complete evidence recall;
- false-positive rate;
- candidate volume;
- rank of first valid evidence;
- retrieval behavior for exact identifiers and paraphrases.

### 5. Ablation evaluation

Hard context fixtures must be run with selected mechanisms disabled to prove the mechanism matters. Examples:

- vector retrieval disabled;
- context expansion disabled;
- DRS scope filtering disabled;
- retroactive context pass disabled.

The hard fixtures should fail under relevant ablations. Otherwise a passing test may be accidental.

## Required recording-derived acceptance cases

The following categories are mandatory and must remain in the suite:

1. dream content cannot answer an ordinary real-world question;
2. the same dream content can answer a question explicitly asking about the dream;
3. reported content cannot establish an official real-world fact by itself;
4. reported content can answer what the reporter said;
5. explicitly official/asserted corpus evidence can answer the corresponding ordinary question;
6. semantic paraphrase retrieval must work through vectors;
7. common world facts absent from the corpus must return unknown;
8. implicit sleep/dream context without the literal word “dream” must be recognized conservatively;
9. retroactive “then I woke up; it had all been a dream” must scope earlier propositions;
10. hypothetical scope established in a header must govern later content;
11. multiple dated sections must keep otherwise identical entities/properties separated by date;
12. related differently scoped evidence must be surfaced without being promoted;
13. evaluator must accept semantic paraphrases and reject wrong meanings;
14. scope carriers outside the matching chunk must be recoverable;
15. huge retrieval result sets must remain answerable within bounded model context.

## Additional mandatory tests beyond the recordings

A full system definition requires more than the two recordings. The suite must also include:

### Negation

- “The bridge is not open” must not be retrieved as support for “the bridge is open.”
- Double negation and scoped negation require explicit tests.

### Conditionals

- “If the alarm is active, door A locks” is not evidence that door A is currently locked unless the condition is established.

### Counterfactuals

- “If the server had restarted, the queue would have cleared” cannot establish that the queue cleared.

### Quotations and attribution

- A quoted claim is evidence of what was said, not automatically of the claim itself.

### Corrections and supersession

- “Earlier we said X; correction: Y” must mark X as superseded for the applicable scope while retaining historical provenance.

### Conflicts

- Two authoritative sources with incompatible current values must produce a conflict unless a source/time/supersession rule resolves it.

### Entity identity

- Similar names must not be merged without evidence.
- Alias resolution must preserve mention provenance.

### Multi-hop evidence

- Answers requiring two or more disconnected supporting propositions must retrieve and combine all required hops.

### Adversarial pretrained knowledge

- Synthetic corpus facts intentionally contrary to common knowledge must be answered from the corpus.
- Questions with obvious pretrained answers but no corpus support must remain unknown.

### Retrieval hard negatives

- Nearly identical wording in the wrong scope must rank below compatible evidence after scope filtering.

### Open-world and completeness

- Missing evidence for `P` must not become evidence for `not P`.
- Missing evidence for `not P` must not become evidence for `P`.
- Explicit negative observations must remain usable as negative evidence when their scope supports the claim.
- `all`, `none`, exact-count, and exhaustive-list questions must return partial/unknown unless completeness is established.
- Declared closed sets must permit exact enumeration and counting when the completeness evidence is cited.

### Corpus prompt injection

- Source text containing fake system prompts, “ignore previous instructions,” tool commands, or answer directives must remain inert evidence and must not alter KMD control behavior.

### Context-distance stress

- Scope carrier 1, 10, 50, and many chunks away;
- carrier after proposition;
- carrier in header/footer;
- nested scopes;
- scope closes before a later similar proposition.

### Scalability

- thousands to millions of chunks;
- large numbers of semantically similar candidates;
- bounded memory and token usage;
- deterministic resume and cache reuse.

## Benchmark rules

A benchmark must never expose gold/reference answers to KMD retrieval or answer generation. Gold data is introduced only after predictions are frozen, for scoring.

Wrong answers should be recorded and benchmark execution should continue by default. `stop-on-failure` is a debugging option, not normal benchmark behavior.

Every benchmark run records:

- exact source tree hash;
- query set hash;
- model identities;
- embedding identity;
- prompt/schema hashes;
- code hashes;
- configuration fingerprint;
- cache configuration;
- initialization statistics;
- question predictions;
- evaluator outputs;
- failures/retries.

A run cannot be called comparable if these compatibility fields differ materially.

## Decision log resolving the recording's open alternatives

### Whole file versus structured context

**Decision:** structured DRS/discourse context is primary; whole-file/large-section context is an ingestion and repair fallback. Query-time answers use minimal scope-complete evidence packs.

Reason: this satisfies the recording's intent and avoids long-context positional failures.

### Neighboring chunk count

**Decision:** semantic graph expansion first; adaptive physical-neighbor fallback second. No fixed ten-chunk rule.

Reason: the necessary carrier may be one chunk away or hundreds of chunks away, and irrelevant fixed windows waste context.

### Similarity threshold

**Decision:** configurable and empirically calibrated per embedding/chunking configuration. Current numeric defaults are provisional until held-out scope-complete retrieval calibration.

### Query expansion

**Decision:** bounded LLM expansion with preserved entities/scope and logged expansions. Exact limit is configurable and calibrated, not hardcoded as an unsupported magic number.

### Direct raw context versus structured representation

**Decision:** hybrid. Raw source evidence is always retained for verification/citation, but DRS/DSPG is the operational context representation used to select and interpret evidence.

### Grounding prompt wording

**Decision:** a short invariant appears in every factual model stage: use supplied KMD evidence for factual content; do not add factual claims from pretrained memory; if evidence is insufficient, return unknown/unresolved.

Prompts may add task-specific instructions but cannot weaken that invariant.

### Uncertain implicit scope

**Decision:** store as explicit uncertainty and restrict promotion. Ambiguous subordinate scope cannot establish an unconditional real-world fact.

### Retrieval mode

**Decision:** mandatory dense/vector candidate retrieval plus complementary lexical and structured channels. Current production configuration should move away from a semantic meaning of “vector optional” for normal KMD operation because that conflicts with the recording requirement.

### Ranking fusion

**Decision:** deterministic multi-channel rank fusion, with RRF as the default initial method. It is simple, score-scale independent, and independently established. It remains replaceable by a validated learned fusion if held-out evaluation demonstrates a consistent improvement.

### Semantic judge

**Decision:** model-backed equivalence judging plus deterministic structural/evidence checks. LLM judgment alone is insufficient for provenance and scope correctness.

## Current implementation alignment

The current KMD tree already contains much of the required substrate:

- explicit source documents/chunks/spans;
- contexts, carriers, and assignments;
- DRS boxes/referents/conditions;
- polarity, modality, temporal information;
- confidence and provenance;
- bounded DSPG;
- document-context processing;
- vector retrieval;
- qualified unknown logic;
- semantic evaluator;
- model attempt records;
- centralized XML runtime configuration;
- persistent logging;
- retry/backoff logic;
- KMD-wide model-call cache root.

Recording-derived unit tests currently pass for the implemented scope/retrieval/evaluation behavior. This is evidence that the current direction is viable, not proof that the complete target specification is finished.

Known target-level gaps should be tracked against this document, especially:

- proving vector retrieval is mandatory in normal production mode rather than merely available;
- broader discourse relation coverage beyond current fixtures;
- stronger explicit contradiction/correction semantics;
- systematic retrieval threshold calibration;
- explicit retrieval ablation tests;
- fine-grained answer-to-evidence support verification for every generated factual clause;
- large-scale stress tests of scope-complete retrieval;
- complete conformance coverage for every requirement in this document.

## Operational lifecycle

### Initialization

1. Load validated configuration.
2. Verify model/embedding identities.
3. Fingerprint source corpus and relevant code/config.
4. Reuse compatible corpus/index caches.
5. Ingest changed sources.
6. Run semantic/discourse extraction for changed material.
7. Materialize/update vector and structured indexes.
8. Validate referential/context graph integrity.
9. Emit initialization diagnostics.

### Query

1. Parse query frame.
2. Generate bounded expansions.
3. Run dense + lexical + structured retrieval.
4. Fuse candidate rankings.
5. Expand through context/discourse graph.
6. Filter by scope/time/authority compatibility.
7. Rerank.
8. Build bounded scope-complete evidence pack.
9. Generate structured answer.
10. Verify support/scope.
11. Repair or downgrade to unknown/conflicted if verification fails.
12. Render answer with provenance.

### Update

When source files change, KMD invalidates only derived data whose source/config fingerprints changed. Model-call cache entries remain globally reusable when their complete call fingerprints match. Corpus-specific vector/index caches invalidate on corpus/chunking/embedding changes.

Incremental updates are content-hash based. Unchanged source material retains stable identities where the logical content/boundaries remain stable; changed or deleted material invalidates dependent spans, semantic records, embeddings, and graph edges. No stale proposition may survive as active evidence after its source has been removed or materially changed. Re-ingestion must be idempotent for an unchanged corpus/configuration.

## Safety properties

KMD must fail conservatively.

If semantic parsing fails, preserve raw source and mark semantic state incomplete.

If scope is ambiguous, do not promote.

If evidence is absent, answer unknown.

If evidence conflicts, report conflict.

If model identity mismatches expected configuration, fail preflight rather than mixing caches/results.

If a cache key is ambiguous or corrupted, recompute.

If required provenance is missing, the proposition cannot support a final factual answer.

If the evidence pack cannot preserve the context needed to interpret a proposition, remove or defer that proposition rather than presenting it out of context.

## Performance requirements

Correctness dominates latency, but the architecture must scale.

- Dense embeddings and reusable indexes are precomputed.
- Model calls are cached KMD-wide by complete call fingerprint.
- Retrieval is staged so expensive model reasoning only sees a bounded candidate set.
- DRS/discourse graph expansion retrieves required context without loading entire documents in the common case.
- Persistent catalogs are resumable.
- Initialization and benchmark phases expose progress metrics.
- Memory limits and context budgets are explicit configuration values.

Performance optimization may never remove required scope/provenance information merely to reduce tokens.

## Definition of done

KMD conforms to this specification only when all of the following are true:

1. Every recording-derived hard requirement in `kmd_recording_requirements.md` has an automated acceptance test or a documented reason why it requires a higher-level benchmark.
2. Vector retrieval is exercised in normal KMD answering and verified by tests/telemetry.
3. Scope-sensitive cases fail correctly when DRS/context mechanisms are ablated.
4. Real-world default queries never use incompatible dream/hypothetical/report evidence as direct support.
5. Differently scoped related evidence remains discoverable and correctly qualified.
6. Retroactive and distant scope carriers work across chunk boundaries.
7. Pretrained-world-knowledge leakage tests pass.
8. Negation, conditionals, quotations, reports, dates, contradictions, corrections, and entity identity have explicit coverage.
9. Every answer claim can be traced to source evidence and context.
10. Semantic equivalence evaluation accepts paraphrases and rejects meaning changes.
11. Evidence/scope support evaluation is independent from answer equivalence.
12. Retrieval thresholds are calibrated on held-out data for the active embedding/chunking configuration.
13. Long-result/context-budget stress tests pass without flattening scope.
14. Model-call caches use the single KMD-wide canonical root and exact fingerprints.
15. Corpus/index caches use strict corpus/config compatibility fingerprints.
16. Full production configuration, logging, timeout/retry, package/build/install, and resume tests pass.
17. No benchmark leaks gold/reference answers into retrieval or generation.
18. The complete KMD test suite passes on the exact release tree.
19. At least one full model-backed benchmark containing both ordinary and extreme scope cases completes without unrecorded failures.
20. The implementation is audited against this document before release; passing old tests alone is not sufficient if the spec has uncovered untested requirements.
21. Raw-folder ingestion, incremental invalidation, schema migration, transactional interruption/recovery, and backend/store invariants have explicit tests.
22. The stable `initialize(folder_path)` / `question(text) -> str` boundary remains usable without a prepared semantic corpus.
23. Malformed/gibberish/profane/vague input cases obey the same grounding rules and do not trigger fabricated special handling.
24. A conversational layer can rephrase KMD output but cannot alter verified evidence/scope/status semantics.
25. Open-world, negative-evidence, closed-world/completeness, exhaustive-enumeration, and exact-count tests pass.
26. Corpus prompt-injection tests prove that source text cannot modify KMD control instructions or tool behavior.
27. Derived arithmetic/set/date results retain machine-checkable provenance to corpus-supported premises.
28. Declared authority and verified authority remain distinguishable in storage, retrieval, and answer rendering.

## Final system summary

KMD is not a generic “vector search plus LLM” application. Its defining property is **scope-preserving, source-grounded reasoning over a discourse-aware semantic datastore**.

The system first preserves raw evidence, then builds a scoped DRS/discourse graph that records who said or experienced what, under which circumstances, when, with what polarity/modality, and from which exact source spans. A user question is converted into a structured requested scope and retrieval plan. Dense vector search is mandatory for semantic recall, complemented by exact/structured retrieval. Retrieved hits are expanded through the discourse graph so the proposition is never separated from the context that determines what it means. Only scope-compatible evidence can establish the direct answer; incompatible or uncertain evidence can still be surfaced as explicitly qualified related information. The final answer is bounded, source-only, verifiable, and semantically evaluated rather than string-matched.

That architecture directly implements the central recording insight: the dangerous failure is not merely failing to retrieve a sentence. It is retrieving the sentence while losing the circumstance that makes the sentence true, false, hypothetical, reported, fictional, dated, or otherwise limited. KMD's entire representation, retrieval, verification, and test architecture must therefore treat context as part of the fact rather than as optional surrounding text.
