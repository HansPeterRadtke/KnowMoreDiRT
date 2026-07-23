# KnowMoreDiRT

KnowMoreDiRT is a raw-folder question-answering system built around Discourse Representation Theory, a persistent discourse graph, local language-model parsing, and deterministic graph execution. The public API remains intentionally small:

```python
import knowmoredirt as kmd

kmd.initialize("/path/to/raw/folder")
answer = kmd.question("Who reviewed the field report?")
```

The input may be an arbitrary folder tree containing prose, logs, tables, transcripts, JSON-like text, noisy files, files without extensions, and mixed source formats. KMD does not require schemas, manifests, or metadata wrappers around the raw text. KMD scans the folder, preserves source spans and provenance, builds an internal discourse representation, and answers questions from the resulting evidence graph.

## Current status

The current model-backed benchmark result is `263/273`. This is the strongest current system state, but it is not evidence that the architecture is finished. The remaining failures cluster around referent identity, target binding, scope, temporal roles, reported versus asserted content, and multi-hop retrieval. Recent experiments show that cross-chunk identity resolution is the most important architectural bottleneck.

The current production path is stable enough that new retrieval and identity ideas should be tested separately before integration. Isolated vector and identity-resolution experiments now exist in the filesystem devtest repository and under `tests/experiments/identity_resolution/` in this repository.

The authoritative continuation plan is [`docs/current_work.md`](docs/current_work.md).

## What the system does today

Initialization scans the raw folder, reads text-bearing files, normalizes source records, segments content into bounded passages, invokes the configured local model for semantic extraction, validates the returned structures, and stores source-backed discourse objects in SQLite. The stored representation includes source documents, spans, mentions, referents, contexts, frames, frame arguments, conditions, temporal information, and provenance links.

Question answering parses the question into a bounded semantic query, retrieves a relevant subgraph, executes deterministic graph and relation logic, and asks the local model only for bounded semantic tasks where language interpretation is required. Answers remain source-grounded and are returned as plain strings.

The local model is required for normal runtime. KMD does not silently replace unavailable model semantics with a fake deterministic parser. Deterministic code is responsible for infrastructure, schema validation, graph storage, candidate filtering, relation execution, proof checks, and answer formatting.

## Why DRT remains part of the architecture

Vector retrieval and language models can find and interpret relevant text, but neither gives the system a persistent exact world model. DRT and the engineering DSPG representation provide stable discourse referents, explicit conditions, source provenance, scope, time, modality, belief contexts, and negation. Their strongest practical purpose is not merely formal logic. It is preserving identity and context across a corpus so that later reasoning refers to the same entities and the same scoped claims rather than reconstructing them from text on every question.

The current difficulty is constructing that world model correctly. A referent created in one chunk may be mentioned by a pronoun, alias, role, or relational description in another chunk. Once those mentions are attached to the correct referent, the graph becomes valuable. Before that attachment is correct, the representation can confidently preserve the wrong identity. The next phase therefore focuses on safe cross-chunk mention resolution.

## Current target architecture

The most promising target is a layered system rather than a pure DRT engine or a pure retrieval-augmented language model.

The filesystem vector database acts as global semantic memory. It stores source chunks and, experimentally, mention vectors, identity-profile vectors, alias vectors, and relation-neighborhood vectors. It retrieves a bounded candidate set from a corpus that may be far larger than any model context window.

KnowMoreDiRT owns discourse semantics. It converts retrieved text into provisional mentions and conditions, resolves mentions against persistent referents, preserves ambiguity when identity is not established, and stores the resulting DRS or DSPG structures with exact provenance and scope.

The local model performs bounded interpretation. It parses passages and questions, compares a current mention with a small candidate set, and explains verified answers. It must not be asked to remember the entire corpus or invent identity when the evidence is insufficient.

Deterministic code performs hard validation. It rejects incompatible entity types, unique-identifier conflicts, impossible temporal overlaps, invalid proof chains, and unsupported merges. A vector result or model judgment becomes a persistent fact only after passing these checks.

## Vector research conclusions

The vector experiments produced one strong production conclusion and one weaker research conclusion.

The strong conclusion is that vectors are excellent semantic navigation tools. Exact comparison against one million normalized 1,024-dimensional vectors took about 7.6 seconds on Thor even when vector generation and normalization were included in the timed loop. A preloaded matrix, GPU implementation, or approximate-nearest-neighbor index can do substantially better. This enables semantic search over corpora far beyond an LLM context window.

The weaker conclusion is that some model-specific vector operators can generate points near likely logical conclusions. Implication chaining, conjunction, and relational composition showed strong results in selected models and held-out domains. However, invalid-premise controls proved that a plausible conclusion vector is not a proof. These operators are useful as additional search transformations, not as truth-producing inference rules.

A global negation direction was not reliable. Negation behaves differently across constructions such as `is not`, `cannot`, `does not remain`, and `it is false that`. Construction-specific operators can work, but polarity should remain explicit in the discourse representation rather than being inferred from one universal vector offset.

## Identity-resolution experiment conclusions

The existing filesystem database is a useful base and should not be replaced. Whole-chunk vectors alone are insufficient, but multiple semantic views retrieve the correct referent candidate reliably enough to justify the architecture.

On the synthetic adversarial benchmark, multi-vector retrieval reached 90.3% top-ten and 96.7% top-twenty candidate recall. Deterministic filtering with all available role, organization, city, product, and email evidence raised top-one accuracy from 38.1% to 66.7% and top-ten accuracy to 94.3%. A constrained local-model resolver received the top ten candidates, achieved 95.5% candidate coverage, correctly rejected all tested genuinely new entities, and improved exact overall resolution beyond raw vector rank. It still preserved ambiguity for generic role and description mentions that did not contain enough information to identify one entity safely.

The operational conclusion is that vectors should retrieve candidates, not assign identity. Referent resolution must combine several vector views, exact identifiers, aliases, local discourse recency, relation evidence, type constraints, temporal constraints, and an explicit ambiguous state.

## Immediate work

No production integration should begin with a broad rewrite. The next decisive step is a real labeled identity benchmark built from actual KMD documents and current failures. The benchmark must measure whether the correct referent appears in the top one, five, ten, and twenty candidates; how often hard constraints remove it; how often the model chooses correctly; how often genuinely new entities are falsely merged; and how delayed resolution improves outcomes when later evidence arrives.

After that benchmark validates the synthetic findings, the system can add provisional mentions and candidate identity links behind a feature flag, followed by delayed reconciliation and vector-assisted question retrieval. The detailed phases, acceptance criteria, and experiment results are in [`docs/current_work.md`](docs/current_work.md).

## Public API

KMD exposes two intended user-facing operations:

```python
initialize(folder_path)
question(text) -> str
```

See [`docs/public_api.md`](docs/public_api.md) for the exact contract.

## Development

```bash
python3 -m pip install -e '.[test]'
export KMD_LOCAL_MODEL_ENDPOINT=http://127.0.0.1:14829/v1
PYTHONPATH=src pytest -q
```

The main test areas are unit behavior, smoke coverage of the two-function API, model-backed evaluation, noise handling, and isolated identity experiments. The current continuation work must preserve the public API and keep experimental identity logic outside production paths until its acceptance criteria are met.

## Documentation

The main documents are [`docs/current_work.md`](docs/current_work.md), [`docs/architecture.md`](docs/architecture.md), [`docs/theory.md`](docs/theory.md), [`docs/evaluation.md`](docs/evaluation.md), [`docs/storage_architecture.md`](docs/storage_architecture.md), and [`docs/public_api.md`](docs/public_api.md).
