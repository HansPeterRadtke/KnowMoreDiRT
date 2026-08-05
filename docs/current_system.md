# Current System

This document describes the system that exists now. It is not a proposal for a future rewrite.

## System boundary

KnowMoreDiRT accepts a raw folder and answers questions about the material in that folder. Its public interface remains:

```python
initialize(folder_path)
question(text) -> str
```

The folder may contain prose, notes, logs, tables, transcripts, structured-looking text, malformed text, or meaningless text. KMD does not require a schema or manifest.

## Processing model

The current architecture is layered:

1. The filesystem database persists files, metadata, extracted text, chunks, hashes, embeddings, and retrieval indexes.
2. Local language-model parsing converts source chunks into grounded DRT structures.
3. The DSPG store persists referents, conditions, scope, time, polarity, identity evidence, and provenance.
4. Retrieval selects a bounded evidence set using lexical, metadata, and vector signals.
5. A local language model converts the user question into a structured query representation.
6. Deterministic execution binds referents, follows permitted graph relations, applies scope, polarity, time, and identity constraints, and produces candidate answers.
7. Verification checks difficult candidates against bounded evidence.
8. A final presentation step converts the technical result into a readable answer.

The governing division of responsibility is:

- vectors retrieve;
- DRT reasons;
- the filesystem database persists;
- language models translate between natural language and structured representations;
- deterministic code validates and executes.

## Initialization

Initialization recursively scans the folder, reads supported files, captures natural filesystem metadata, computes hashes, extracts text, creates bounded chunks, stores exact source spans, computes embeddings, and builds retrieval indexes.

Meaningful chunks are sent to the configured localhost model. The model emits grounded DRT JSON. Deterministic validation checks schema integrity, ID references, source grounding, scope structure, temporal references, identity references, and provenance before accepted objects enter the DSPG store.

The deterministic layer does not invent missing semantic roles, event meanings, modality, or discourse scope. Invalid or ungrounded model output is rejected rather than silently converted into a different semantic interpretation.

## Question answering

A question is parsed into a structured query DRS. The query representation carries answer variables, target referents, requested conditions, constraints, answer type, temporal requirements, polarity, scope, aggregation, and evidence requirements.

The retriever uses lexical matches, filesystem metadata, embeddings, referent surfaces, relation material, and bounded identity expansion to locate candidate evidence. It does not load the entire corpus graph for each question.

The symbolic executor then performs bounded graph matching. It checks local condition structure rather than relying on unrelated words in the same chunk. It enforces discourse accessibility, polarity, temporal compatibility, identity constraints, answer type, and provenance. Conflicting equally grounded answers produce `unknown` rather than an arbitrary choice.

## Failure behavior

KMD returns `unknown` when it cannot produce a grounded, structurally valid answer. This includes unusable query DRS output, inaccessible scope, unresolved temporal ambiguity, incompatible answer type, unsupported identity binding, conflicting evidence, and insufficient proof structure.

No-answer paths retain internal provenance and diagnostics so failures can be inspected without exposing implementation details through the public string API.

## What is not part of the current contract

KMD is not a general conversational agent. It is the technical reasoning service beneath one. Markdown style, conversational recovery, social tone, and broad dialogue management belong in an agent layer above KMD.

KMD also does not treat vector similarity as truth, does not use one global negation vector, does not accept unverified identity merges, and does not use domain-specific question handlers as the normal semantic path.
