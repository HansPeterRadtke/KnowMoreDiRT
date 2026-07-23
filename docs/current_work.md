# Current work: vector-assisted persistent discourse identity

## Purpose of this document

This document is the authoritative continuation plan for KnowMoreDiRT after the vector-geometry, logical-operator, vector-database, and cross-chunk identity experiments completed in July 2026. It is intended for a new developer or agent with no conversation history. It explains the current system, the experiments already performed, the conclusions that are supported by evidence, the conclusions that remain speculative, the target architecture, the next implementation phases, and the acceptance criteria that must be met before production integration.

The central conclusion is that the most promising system is not a pure language-model question-answering pipeline and not a pure symbolic DRT parser. It is a layered semantic-memory system. Vectors provide global semantic discovery over a corpus too large for a model context. DRT and DSPG provide persistent identity, scope, time, modality, provenance, and deterministic graph structure. A local language model performs bounded interpretation over retrieved evidence and candidate referents. Deterministic code validates every persistent merge and every claimed inference.

## Current system overview

KnowMoreDiRT exposes `initialize(folder_path)` and `question(text)`. Initialization scans arbitrary raw files, creates source records and spans, invokes the local model for semantic parsing, validates the returned structures, and persists a discourse graph in SQLite. Question answering parses a question, retrieves a bounded subgraph, executes relation and graph logic, and produces a source-grounded natural-language answer.

The internal engineering representation is DSPG, a Discourse Source Provenance Graph derived from practical DRT requirements. It contains source documents and spans, mentions, referents, contexts, frames, frame arguments, conditions, temporal structures, and provenance. DRT remains the semantic commitment: discourse referents persist across statements, conditions attach to those referents, and scope distinguishes asserted facts from beliefs, hypotheticals, negations, reports, and other contexts.

The latest completed model-backed benchmark scored 263 of 273. No run is currently active. The result shows that the current system is useful and stable enough for controlled continuation, but the remaining errors expose architectural weaknesses rather than isolated formatting problems. The important failure families include incorrect referent or target binding, nested-field binding, temporal role confusion, world or scope confusion, reported versus asserted content, source authority, multi-hop relation selection, and retrieval scope. These failures are closely connected because a fact can be retrieved correctly yet attached to the wrong entity, time, role, or discourse context.

## Why the project still needs DRT and DSPG

A standard retrieval-augmented language model can answer many questions when the relevant text is placed in its context. That does not make DRT unnecessary. It clarifies its role.

Vectors solve global discovery. They answer where the relevant evidence may be located. A model solves bounded language interpretation. It explains what a small piece of text probably means. DRT and DSPG solve persistence and exact structure. They record which mention refers to which entity, which condition is asserted in which context, when it holds, where it came from, and which later facts may be combined with it.

The unique practical value of DRT is therefore not that an LLM is incapable of understanding one sentence about a belief or a pronoun. The value is that the system can preserve millions of such decisions as one inspectable world model instead of reconstructing them probabilistically for every question. Persistent identity, explicit scope, temporal state, and provenance enable deterministic graph operations and make mistakes traceable.

The largest obstacle appears before this benefit is available. Cross-chunk identity resolution must determine whether a new mention refers to an existing referent, introduces a new referent, or remains ambiguous. A wrong merge contaminates every later condition attached to that referent. A missed merge fragments one real entity into several referents. The next phase must therefore improve world-model construction before expanding symbolic reasoning.

## Vector-geometry experiments

The first experiments tested direct geometry in Qwen3-Embedding-0.6B-Q8 sentence space. Midpoints and extrapolations between semantically related sentences produced meaningful neighborhoods. Moving between a dog defecating in a corner and in a garden preserved the event while changing location. Moving between customer and employee produced client, worker, staff, and combined-role meanings. Moving between hot and cold produced warm or cool intermediate concepts and heat or freezing concepts beyond the endpoints. Belief-attribution examples also moved continuously between attributed and direct assertions.

These results establish that sentence-vector geometry contains interpretable semantic directions. They do not establish that every direction has one stable symbolic meaning. The practical use is semantic exploration and query expansion. A midpoint, extrapolation, or weighted combination should be treated as another database query, never as a fact.

## Logical-operator experiments

The first narrow logical tests searched mathematical vector formulas for modus ponens, modus tollens, conjunction, disjunctive syllogism, implication chaining, and relational transitivity. Strict template-based held-out results initially appeared strong: modus ponens reached about 89.5%, conjunction 64.2%, implication chaining 92.9%, relational transitivity 99%, and disjunctive syllogism 100%. Modus tollens reached only about 33%.

Those numbers were not sufficient because template similarity can dominate semantic geometry. A stronger robustness experiment added paraphrases, a broad candidate pool, invalid-premise substitutions, and shuffled-premise controls. Implication chaining remained convincing at 94.4% top one and 100% top ten, while falling to 26.3% for invalid chains and 1.3% with a shuffled second premise. Modus ponens reached 40.6% top one and 89.4% top ten, while falling to 13.1% on invalid antecedents and 3.8% with shuffled premises. Disjunctive syllogism collapsed to 1.9% top one, proving that its earlier result was a template artifact. Relational transitivity reached 65.6% top one and 96.3% top ten, but invalid and shuffled controls remained high. The operator could construct the expected-looking relation without verifying that the intermediate chain licensed it.

The next round fitted offsets, scalar-plus-offset maps, low-rank maps, full affine maps, and two-premise operators on train-test splits. Random splits produced approximately 98% to 100% accuracy for several operators. Entire held-out semantic domains caused the large affine maps to collapse toward zero. The maps had learned interpolation over shared vocabulary and templates rather than universal logic.

A lower-capacity component representation used separate vectors for premises, entities, predicates, relations, coordinate products, and differences with only a few shared scalar coefficients. This transferred better. Qwen reached 93.1% to 100% for conjunction and 79.2% to 100% for transitivity across held-out domains. Multilingual E5-large reached 100% for implication chaining in every held-out domain, 93.8% to 100% for transitivity, and 56.2% to 100% for modus ponens. Model behavior differed substantially, proving that the operators are embedding-model-specific.

The supported conclusion is limited but useful. Some vector operators can calculate a point near a likely conclusion. They are suitable for hypothesis generation and secondary semantic search. They are not proof rules. A separate symbolic or structural validator must check referent identity, predicate compatibility, intermediate variables, scope, time, and rule applicability before accepting any conclusion.

## Negation experiments

Negation was tested on hundreds of positive and negative pairs. Positive and negated sentences remained very close, with mean cosine near 0.884, because they share almost all semantic content. The direction from a positive sentence to its negation was inconsistent, with low pairwise directional agreement. One global negation direction retrieved the exact held-out negation only around 14% of the time. Average offsets and low-rank maps reached about 10% exact retrieval. A full affine map overfit and generalized worse.

Construction-specific operators worked better. In Qwen, `it is true that` to `it is false that` reached roughly 34% to 81% across held-out domains, while plain `is` to `is not` was nearly unusable. In multilingual E5-large, `true that` to `false that` reached 100% across all held-out domains, `can` to `cannot` reached roughly 59% to 98%, and `remains` to `does not remain` reached roughly 33% to 92%. Plain `is` to `is not` still failed.

The conclusion is that negation is not one universal algebraic inverse. The DRS must represent polarity explicitly. Construction-specific opposition operators may be useful for contradiction search, but they must not define truth or falsity.

## Vector-scale experiment

An exact scan benchmark on Thor compared one normalized 1,024-dimensional query against generated normalized vectors. One hundred thousand vectors took about 0.76 seconds, 250,000 took about 1.89 seconds, and one million took about 7.59 seconds. The loop included generation and normalization, so a preloaded matrix is expected to be faster. GPU or approximate-nearest-neighbor implementations can scale further.

This is the strongest practical vector result. A model cannot inspect millions of chunks in one context. A vector database can search them and send only the most relevant evidence onward. This capability is required regardless of whether the final reasoner is symbolic, neural, or hybrid.

## Cross-chunk identity-retrieval experiments

The identity benchmark used the existing filesystem repository's embedding client and SQLite-oriented architecture. It created 120 persistent people, 720 known mentions, and 40 genuinely new mentions. Mention forms included names, aliases, generic roles, relational descriptions, pronouns, and relation-only descriptions. Entities deliberately shared names, roles, organizations, cities, products, and other properties to prevent trivial matching.

Whole-chunk matching was weak. Noisy mention chunks compared with entity introduction chunks reached 18.2% top one, 50% top ten, and 69.2% top twenty. Sentence-level matching improved to 34.2% top one and 81.9% top ten. Generated identity-profile queries reached 36% top one and 79.4% top ten. Relation-profile queries reached 25.7% top one, 86.8% top ten, and 98.3% top twenty.

The best general candidate generator used several independent vector views rather than one combined identity vector. A multi-vector maximum over name, sentence, identity-profile, and relation-profile evidence reached 44.4% top one, 74.3% top five, 90.3% top ten, and 96.7% top twenty. A fixed weighted fusion reached 38.1% top one, 63.1% top five, 87.8% top ten, and 96.1% top twenty. A simple arithmetic sum of surface and relation vectors did not outperform the multi-view method.

Mention types behaved differently. Weighted fusion reached 100% top one for full names and 83.3% for aliases. Multi-vector retrieval reached 73.3% top one and 98.3% top ten for pronouns, and 46.7% top one and 90% top ten for relation-only references. Generic roles and descriptions remained weak because several candidates genuinely satisfied the same known properties.

Deterministic filtering was essential. Filtering with all available role, organization, city, product, and email evidence raised weighted-fusion top-one accuracy from 38.1% to 66.7%, top-five to 80.7%, and top-ten to 94.3%. Type or organization alone contributed little because many candidates shared them.

Open-set identity detection showed why one similarity threshold is unsafe. On this synthetic corpus, threshold 0.59 retained 84.4% of known mentions with no false merges among the forty new entities. Lowering the threshold to 0.58 raised recall to 91.3% but falsely merged 20% of new entities. The threshold is corpus-specific and must not be copied into production.

A constrained Qwen3.5-27B candidate-selection experiment used forty-four balanced cases and the top ten vector candidates. The correct candidate was present for 95.5% of cases. The raw vector's first candidate was correct for only 36.1% of known mentions. The model raised exact known resolution to 52.8% and overall accuracy to 61.4%, while correctly rejecting all eight genuinely new entities. It resolved every tested full name and pronoun, half the aliases, and two thirds of relation-only cases. It preserved ambiguity for generic roles and descriptions whose evidence did not distinguish one entity.

The result validates candidate reduction, not automatic identity assignment. Vectors can usually retrieve the correct referent neighborhood. They cannot decide identity safely by themselves. The resolver must support `existing`, `new`, and `ambiguous` outcomes.

## DRS-side resolver contract experiment

An isolated resolver under `tests/experiments/identity_resolution/` tests the required decision behavior without modifying production code. It covers exact unique identifiers, exact aliases, pronouns supported by relational context and local recency, same-role ambiguity, entity-type conflicts, temporal incompatibility, explicit new entities, and close candidate margins. All eight tests pass.

The contract is intentionally conservative. A candidate is rejected when its type conflicts, a unique identifier conflicts, or the mention falls outside its active temporal interval. Exact identifiers and aliases can outweigh a weaker vector score. A candidate is accepted only when its score exceeds an acceptance threshold and its margin over the runner-up is sufficient. Otherwise the decision remains ambiguous. No uncertain merge is forced.

## Architectural decision

The existing filesystem database should remain the global storage and retrieval foundation. Rebuilding vector storage inside KnowMoreDiRT would duplicate working infrastructure and would not solve identity. Integration should occur through a narrow interface rather than by importing the entire implementation deeply into KMD.

The filesystem layer should eventually expose retrieval for source chunks, mentions, referent profiles, aliases, and relation neighborhoods. KnowMoreDiRT should own provisional mentions, persistent referents, DRS or DSPG conditions, contexts, confidence, candidate identity links, accepted merges, rejected merges, and unresolved ambiguity.

No single vector should represent a referent. A referent should have multiple indexed views: canonical name and aliases, descriptive identity profile, role profile, relation-neighborhood profile, source-context mentions, and optionally time-specific profiles. Retrieval should union candidates from these views and retain per-view evidence rather than collapsing everything into one centroid.

## Target processing pipeline

A raw passage is ingested with exact source provenance. The local parser extracts provisional mentions, candidate conditions, roles, contexts, time expressions, and source spans without assigning permanent cross-document identity prematurely.

Each provisional mention produces multiple retrieval queries. The system embeds the surface phrase, the containing sentence, a compact identity description, and a normalized relation-neighborhood description. It asks the filesystem database for candidates from each matching index and forms a bounded union.

Deterministic filters remove impossible candidates. These include entity-type conflict, incompatible unique identifiers, impossible time intervals, contradictory account or employee identifiers, impossible document-level constraints, and explicitly forbidden merges. Exact aliases and stable identifiers receive strong positive evidence.

The local resolver receives the mention, its source context, recent active discourse referents, and a small candidate set with vector and structural evidence. It returns one existing referent, a new referent, or ambiguity. The output must use constrained structured decoding and cite the evidence used.

Only high-confidence decisions with sufficient margin are committed as permanent mention-to-referent links. Ambiguous mentions remain provisional and may carry ranked candidate links. Later passages can add evidence and trigger reconciliation. The system must preserve the original source mention and every prior decision so that merges can be audited or reversed.

Conditions attach to resolved or provisional referents with exact context, polarity, time, and provenance. A provisional referent may participate in local reasoning only under rules that preserve uncertainty. It must not be globally merged merely because its vector is similar to an existing entity.

Question retrieval first uses vector search over source chunks and semantic objects. Optional midpoint, extrapolation, or learned operator queries may expand the search, but every generated vector is labeled as a search hypothesis. Retrieved evidence is converted into a bounded DRS subgraph. Deterministic graph logic validates identity, scope, temporal compatibility, and proof applicability. The model then explains the verified result.

## Implementation phases

### Phase 1: real labeled identity benchmark

Build a benchmark from actual KMD documents and known failures. It must include full names, aliases, pronouns, role descriptions, relational descriptions, repeated names, repeated roles, entity re-entry after long gaps, explicit new entities, ambiguous cases, temporal role changes, and references across files. Each mention must have a gold state of existing referent, new referent, or unresolved ambiguity. Existing cases must identify the gold referent.

The benchmark should initially contain at least two hundred cross-chunk mentions, then grow as failures are discovered. Synthetic cases remain useful for controlled stress testing but cannot determine production thresholds.

Measure top-one, top-five, top-ten, and top-twenty candidate recall for each vector view and their union. Measure by mention type, document distance, number of distractors, source genre, and whether unique structural evidence exists. Measure open-set false-merge rate separately from closed-set candidate recall.

Do not proceed to automatic production merging unless the correct referent appears in the top ten for at least 95% of resolvable real mentions and the top-twenty recall is at least 98%. These thresholds are initial engineering targets, not theoretical guarantees.

### Phase 2: experimental mention and referent profile store

Define a narrow retrieval interface between KMD and the filesystem database. The interface should accept vector queries, metadata filters, source-range constraints, and requested semantic view. It should return candidate IDs, scores by view, source evidence, and structural metadata.

Add experimental records for provisional mentions and referent profiles without changing the public API. Preserve source chunk vectors, but add surface, sentence, identity-profile, and relation-profile vectors. Keep model identifier, revision, dimensionality, normalization state, and generation prompt with every vector so indexes can be rebuilt safely.

Do not overwrite one referent vector after every mention. Maintain multiple observations and one or more derived profiles. Time-specific or role-specific profile clusters may be needed when one entity changes role or context.

### Phase 3: conservative resolver behind a feature flag

Implement the isolated resolver contract in production-quality form behind a disabled feature flag. The resolver must support `resolved`, `new`, and `ambiguous`. It must record candidate scores, hard filters, positive evidence, negative evidence, margin, model decision, and final commit status.

Use exact identifiers and hard conflicts before model judgment. The model receives at most a bounded candidate set. It must not see raw database-wide state or produce arbitrary referent IDs. Constrained decoding should restrict it to the supplied candidates plus `new` and `ambiguous`.

Evaluate precision and recall separately. Precision of committed merges is more important than coverage because a false merge poisons the world model. Ambiguity is an acceptable outcome. A reasonable first acceptance target is at least 99% precision on committed merges with measured coverage reported separately.

### Phase 4: delayed reconciliation

Introduce provisional referents and candidate-equivalence links. Later evidence may confirm a candidate, reject it, or merge two provisional referents. Reconciliation should combine stable identifiers, accumulated aliases, relation neighborhoods, source continuity, time compatibility, and model judgment.

Every merge must be reversible or reconstructible. Store merge provenance and the evidence state at decision time. Re-run affected conditions and cached answers when a merge changes graph identity.

Test scenarios where the first mention is impossible to resolve but a later email address, account identifier, explicit name, or relation makes the identity clear. Compare immediate forced resolution with delayed resolution. The delayed system should reduce false merges without unacceptable referent fragmentation.

### Phase 5: vector-assisted question retrieval

Use the filesystem database as semantic memory for questions. Embed the complete question and model-generated subquestions or expected-answer descriptions. Retrieve source chunks, mentions, referent profiles, conditions, and local graph neighborhoods.

Test conservative search expansion. Start with the nearest results, generate pairwise midpoints or weighted combinations only among highly related candidates, search again, and measure whether new relevant evidence is discovered. Five initial candidates produce ten unordered pairs, or twenty ordered extrapolation directions if direction matters. Expansion must be bounded and deduplicated.

Compare baseline nearest-neighbor retrieval with multi-query retrieval, multi-vector referent views, midpoint expansion, extrapolation, and learned implication-chain search operators. Measure answer accuracy, evidence recall, irrelevant-context growth, latency, and model-token usage. Search expansion is accepted only when answer or evidence recall improves more than context noise and cost.

### Phase 6: validated vector hypothesis operators

After retrieval is stable, expose selected vector operators as optional query generators. Each operator must declare its embedding model, training dataset, semantic purpose, and known failure controls. Implication chaining is the strongest initial candidate because it survived paraphrase and shuffled-premise tests. Relational transitivity may generate useful completions but must never be labeled valid without structural proof.

Every generated point should carry provenance showing the source vectors and operation. Search results produced by an operator remain hypotheses until matched against source evidence or validated DRS conditions.

### Phase 7: graph reasoning and answer verification

Improve deterministic reasoning only after identity and retrieval are reliable. The graph engine should explicitly validate variable identity, relation direction, transitivity declarations, scope, polarity, source authority, and temporal compatibility. Model-generated answer text must cite or retain the supporting source and graph path.

Regression tests should target the remaining benchmark failures and new identity cases. The current 263 of 273 score must not regress. New metrics should include identity precision, candidate recall, ambiguity rate, graph-proof validity, and source-evidence coverage.

## Experiments still necessary

The broad architectural direction is sufficiently supported to stop exploring unrelated alternatives. Several targeted experiments remain necessary before integration.

The real-data identity benchmark is mandatory because all current retrieval numbers come from an adversarial synthetic corpus. It must establish whether the correct referent is usually retrievable and whether real ambiguity resembles the synthetic benchmark.

Document-distance tests are required. Candidate recall should be measured when antecedents occur in the previous sentence, previous chunk, previous file section, another file, or hundreds of pages earlier. Local discourse recency and global semantic retrieval should be compared directly.

Entity-profile update experiments are required. Compare one centroid, several view-specific centroids, all mention vectors, clustered temporal profiles, and relation-neighborhood vectors. Measure both retrieval quality and index cost.

Delayed-resolution experiments are required. Measure false merges, fragmented entities, and final resolved coverage when ambiguity is preserved until later evidence arrives.

Real question-answering retrieval expansion tests are required. Midpoints and learned operators are interesting only if they find supporting evidence that nearest-neighbor and multi-query retrieval miss without flooding the context with irrelevant material.

A second embedding model should be tested for identity retrieval because logical experiments showed strong model-specific differences. At minimum compare Qwen3-Embedding with multilingual E5-large on the same real labeled identity set.

Threshold calibration and confidence calibration are required. No synthetic threshold may be copied into production. Calibrate scores by mention type and semantic view, and report uncertainty rather than pretending all cosine values are comparable.

## Decisions already made

The existing filesystem vector database remains the retrieval foundation. Do not replace it with a new KMD-specific vector store without evidence of a missing capability.

DRT and DSPG remain the persistent world-model layer. Do not reduce the system to storing only chunks and summaries because that loses stable identity, explicit context, time, polarity, and provenance.

Vectors are search and candidate-generation mechanisms. Do not treat vector arithmetic as proof.

The local model performs bounded semantic interpretation. Do not ask it to reconstruct or remember the complete corpus.

Ambiguity is a valid persistent state. Do not force every mention to an existing or new referent immediately.

False merges are more damaging than missed merges. Optimize committed merge precision before coverage.

Experiments stay isolated until their acceptance criteria are met. Preserve the current production benchmark and public API.

## Integration checkpoint completed

The filesystem semantic database has now been copied into this repository as the isolated `file_system_catalog` package. Its original devtest copy remains intact. KMD exposes a thin `knowmoredirt.filesystem` facade and standalone CLI entry points, so fast indexing, semantic retrieval, and grounded LLM question answering can run without DRT initialization. The copied regression suite is retained under `tests/filesystem_database/`. This checkpoint does not yet make the production DRT initializer depend on the database; that dependency will be introduced only after the real identity benchmark validates the retrieval interface.

## Immediate next task

The next implementation task is to create the real labeled cross-chunk identity benchmark and a runner that uses the current filesystem retrieval interface. Start by harvesting identity-related failures and representative source passages from the existing KMD fixtures and model-backed benchmark outputs. Label each mention with its gold referent or ambiguity state. Reproduce the synthetic benchmark metrics on this real set. Only after those results are known should the production schema or initialization pipeline change.

A new agent should begin by reading this document, the root README, `docs/architecture.md`, the isolated tests in `tests/experiments/identity_resolution/`, and the filesystem experiment report in the devtests file-system repository. The first code change should be benchmark infrastructure, not production entity merging.
