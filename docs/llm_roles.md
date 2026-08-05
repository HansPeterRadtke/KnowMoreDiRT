# Language Model Roles

KnowMoreDiRT uses local language models in three separate conceptual roles. Keeping these roles distinct prevents presentation behavior from contaminating reasoning.

## Source parser

The source parser converts a bounded source chunk into grounded DRT JSON. It identifies discourse referents, boxes, conditions, arguments, scope, temporal structures, polarity, and identity hypotheses. Every accepted evidence string must be grounded in the source chunk.

The parser proposes semantics. It does not write directly into persistent storage. Deterministic validation decides whether the proposal is structurally valid and sufficiently grounded.

## Query parser

The query parser converts the user question into a grounded query DRS. It declares what is being asked, which referents and conditions are relevant, the expected answer variable and type, and any scope, polarity, temporal, aggregation, or evidence requirements.

The query parser does not answer the question from memory. It creates an executable request against the bounded DSPG evidence set.

## Answer presenter

The answer presenter converts a verified technical result into readable language. It may improve wording, explain uncertainty, and format provenance for a conversational agent. It must not add facts or override the symbolic result.

## Deterministic boundary

Deterministic code owns schema validation, source grounding checks, cache identity, persistent storage, bounded retrieval orchestration, graph execution, arithmetic, aggregation, conflict detection, proof validation, and final acceptance or rejection.

The intended flow is therefore:

source text -> source parser -> validated DRT -> DSPG

question -> query parser -> validated query DRS -> bounded symbolic execution

verified result -> answer presenter -> human-readable response
