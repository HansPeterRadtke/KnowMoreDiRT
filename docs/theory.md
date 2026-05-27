# Theory: DRT and DSPG

## Classical Discourse Representation Theory

Discourse Representation Theory (DRT) is a formal approach to natural-language meaning introduced by Hans Kamp and developed further by Kamp, Uwe Reyle, and others. Its central insight is that interpretation is dynamic: a sentence is not evaluated in isolation, but updates a discourse representation that already contains entities, conditions, temporal information, and accessibility constraints from earlier discourse.

Classical DRT represents discourse in structures often called DRSs: boxes containing discourse referents and conditions over those referents. This made DRT especially important for phenomena where sentence-local semantics is insufficient:

- anaphora and pronoun resolution,
- indefinites and definites,
- tense and temporal reference,
- quantification across discourse,
- presupposition and accessibility,
- attitude reports and embedded contexts.

Useful references:

- Stanford Encyclopedia of Philosophy, [“Discourse Representation Theory”](https://plato.stanford.edu/entries/discourse-representation-theory/)
- Hans Kamp, [“A Theory of Truth and Semantic Representation”](https://www.degruyterbrill.com/document/doi/10.1515/9783110867602.1/html)
- Hans Kamp and Uwe Reyle, [*From Discourse to Logic*](https://books.google.com/books?vid=ISBN079232403X)
- Kamp, van Genabith, and Reyle, [“Discourse Representation Theory” handbook chapter](https://www.ims.uni-stuttgart.de/archiv/kamp/files/2011.kamp.van.genebith.reyle.discourse.representation.theory.handbook.pdf)
- Johan Bos, [“Wide-Coverage Semantic Analysis with Boxer”](https://aclanthology.org/W08-2222/)
- Parallel Meaning Bank / DRS parsing shared-task work, represented in the local paper archive under `docs/papers/`

## Why Flat Triples Are Not Enough

A conventional subject-predicate-object triple can store a fact-like assertion, but it usually loses the discourse conditions that decide whether the assertion should be used as a fact:

- Was it asserted, quoted, denied, believed, alleged, hypothetical, fictional, or dreamed?
- Which source span supports it?
- Which mention introduced the referent?
- Is a later mention the same entity, a different entity, or unresolved?
- Which temporal context governs the statement?
- Did a later source revise or contradict it?
- Is the statement globally asserted, or only available inside an embedded context?

These distinctions are not optional for practical question answering over messy folders. A system must know the difference between “Anna deleted `debug.tmp`,” “Martin said Anna deleted `debug.tmp`,” “Anna dreamed she deleted `debug.tmp`,” and “The audit found `debug.tmp` still exists.”

## DSPG: Engineering Representation Evolved from DRT

KnowMoreDiRT uses DSPG, the Discourse Source Provenance Graph, as the engineering representation layer. DSPG is not a replacement theory for DRT. It is a practical storage/query representation inspired by DRT and extended for raw-text knowledge systems.

DSPG preserves the DRT commitments that matter operationally:

- **Mentions**: exact source-grounded spans such as names, pronouns, URLs, IDs, dates, file paths, and descriptions.
- **Referents**: local entity hypotheses linked to mentions without forcing premature merges.
- **Contexts**: asserted, quoted, reported, believed, alleged, conditional, fictional, dreamed, negated, uncertain, temporal, and document-level scopes.
- **Frames**: event/proposition records with predicates and roles.
- **Source spans**: exact file/chunk offsets and surfaces that ground every extracted object.
- **Temporal evolution**: state/event ordering and revision history.
- **Provenance**: the path from answer back to source text and extraction run.

Classical DRT provides the semantic rationale. DSPG provides the database-backed operational form needed to ingest arbitrary file trees, preserve provenance, and answer questions without assuming that the input already has a schema.

## Implementation Notes from DRT References

The Kamp, van Genabith, and Reyle handbook chapter treats DRT as dynamic context update: discourse introduces referents and conditions, and later interpretation is constrained by accessibility. For KMD this means source chunks should update a global DRS/DSPG rather than populate question-specific handlers.

Boxer and the Parallel Meaning Bank show the practical value of machine-readable DRS structures with normalized predicates, roles, discourse referents, and scoped conditions. KMD follows that engineering lesson by asking the local model for strict JSON DRS/DSPG structures, then accepting only grounded spans that can be mapped into SQLite referents, contexts, conditions, arguments, identity hypotheses, and provenance records.

## Raw Discourse Grounding

KMD’s public input contract is intentionally strict: a folder path and raw readable file contents. The system must discover structure from text rather than relying on source-side metadata wrappers or external schemas. This is the engineering consequence of the DRT perspective: meaning is built by tracking discourse updates, referents, contexts, and source-grounded conditions inside the system.
