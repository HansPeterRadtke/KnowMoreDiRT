# Cache Contract

A cached result may be reused only when every input capable of influencing that result is identical.

## Principle

Each cache key is a deterministic digest over the complete effective input state. The cache must never depend only on source text when model choice, prompt, schema, decoding policy, retrieval settings, or algorithm version can change the output.

## Model-backed computation keys

A model-backed cache key includes, as applicable:

- exact input text;
- model identifier and revision;
- endpoint/runtime identity when behavior may differ;
- prompt text and prompt version;
- JSON schema and grammar version;
- decoding parameters and token budgets;
- staged versus monolithic extraction policy;
- source-span option set;
- parser and validator versions;
- grounding and repair policy versions.

## Ingestion keys

Ingestion and chunk caches include:

- file content hash;
- relevant filesystem metadata when used by the computation;
- text extraction implementation and version;
- chunking algorithm and parameters;
- normalization policy;
- embedding model and revision;
- embedding dimensions and normalization policy;
- index implementation and version.

## Retrieval and execution keys

Question-path caches include:

- normalized question and exact original question where casing or evidence grounding matters;
- query parser context;
- corpus or database state hash;
- retrieval algorithm and parameters;
- lexical, metadata, vector, and identity-expansion settings;
- bounded graph-executor version;
- verifier version;
- answer-type and conflict policies.

## Negative cache entries

Deterministic failures such as invalid JSON, schema rejection, or grounding rejection may be cached when the complete cache context is identical. Transient transport failures must remain retryable.

## Invalidation

Changing any influential input creates a new key. Existing entries do not need destructive mutation; they simply become unreachable from the new context. This makes experiments reproducible and prevents stale semantic output from masquerading as a current result.
