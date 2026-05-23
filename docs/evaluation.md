# Evaluation

## Fixture Suites

KMD currently keeps four self-written fixture suites:

- `tests/fixtures/messy_raw_corpus/`: original project-style regression corpus.
- `tests/fixtures/broad_raw_world/`: heterogeneous raw-world corpus across school, family, household, fiction, law-like notes, medical appointment notes, veterinary notes, geography, language learning, recipes, travel, sports, art, research, logistics, accounting, schedules, diagrams, tables, OCR-like text, multilingual fragments, aliases, and conflicts.
- `tests/fixtures/hardcore_noise/`: random-character, base64/hex-like, word-salad, multilingual nonsense, OCR garbage, plausible babble, and adversarial distractor pollution.
- `tests/fixtures/hard_raw_reasoning/`: hard failure-driven raw-text suite covering type safety, relation-scoped IDs/URLs, unanswerable false positives, nested object text, multi-hop lookup, temporal state, context, tables/logs, and noise pollution.

All fixtures are raw text only. They use arbitrary nested folders, arbitrary filenames, mixed file endings, files without extensions, prose, tables, chats, logs, transcript turns, JSON-like text, dreams, beliefs, allegations, contradictions, noisy text, IDs, URLs, and distractors.

## Current Score

Current generic hybrid query-frame results after adding the DRS condition layer, identity-hypothesis storage, local-model frame normalization, and model-frame argument binding:

- original messy corpus: `29/60 (0.483)`
- broad raw-world corpus: `41/65 (0.631)`
- hardcore noise corpus: `7/8 (0.875)`
- hard raw-reasoning corpus: `71/134 (0.530)`

These scores are intentionally reported after removing procedural semantic routing and relation-specific answer handlers from the core and keeping the fixture gates at exact correctness. The strict gates currently fail; the numbers above are the honest current state. The next implementation work is to recover accuracy through generic discourse parsing, graph traversal, context propagation, aggregation, cacheable local-model frame extraction, and bounded entailment verification rather than semantic handler branches.

## Categories Covered

Across the fixture suite, categories include:

- direct facts,
- exact IDs and URLs,
- source-grounded questions,
- temporal/final-state questions,
- table lookup and table context,
- school homework, teacher feedback, math word problems, science notes, language learning, history and geography notes,
- diary entries, dream journals, family messages, household notes, recipes, travel plans, sports/music/art notes,
- fiction, fantasy lore, fictional and real letters, forum posts, legal-style notes, incident notes,
- invented medical appointment and veterinary notes,
- gardening, farming, construction, appliance manuals, scientific abstracts, lab notebooks,
- debates, belief statements, neutral fictional civic arguments, accounting, shipping, calendars, spatial layouts, diagrams,
- multilingual fragments and OCR-like corruption,
- beliefs versus facts,
- dreams and fiction versus assertions,
- claims, counterclaims, allegations, and contradictions,
- transcript/chat speaker attribution,
- noisy text,
- raw JSON-like text,
- entity ambiguity,
- multi-hop dependencies,
- aggregation,
- unanswerable questions.

## Noise/Gibberish Robustness

The hardcore noise suite checks that:

- random-character and near-binary text remains ingestible,
- symbol-heavy files receive low-semantic-content flags and `quality:*` contexts,
- noise categories such as random-character noise, hex/blob-like text, OCR corruption, word salad, plausible babble, and meaningful discourse are preserved as metadata/context,
- noisy files do not dominate normal meaningful answers,
- gibberish-only questions return `unknown`,
- meaningful facts mixed with noise remain answerable.

## Architecture Checks

The unit tests also verify that initialization creates normalized DSPG structures:

- documents,
- chunks,
- source spans,
- mentions,
- referents,
- identity hypotheses,
- context records,
- context carriers and assignments,
- frames,
- frame arguments,
- temporal edges,
- generic relation records,
- normalized metadata records.

Architecture tests assert that the core package contains no external-evaluation markers, wrapper assumptions, hidden-label terms, or dataset-shaped routing, and that `knowmoredirt.__all__` exports only `initialize` and `question`.
They also scan the core package for fixture/domain-shaped literals from the regression corpora so future changes do not quietly reintroduce content-specific answer branches.

The current answer path uses generic DSPG mechanisms:

- label/value and raw text key/value relations,
- JSON-like/object-as-text key/value relations with source-grounded record paths,
- active/passive event relations,
- negation relations,
- temporal state relations,
- table row/cell relations,
- filesystem/read metadata and document quality contexts.
- broad expected-answer type validation for person/actor, organization, identifier, URL, file path, count, state, date/time, boolean, content phrase, and metadata answers.

Additional unit and hard-fixture coverage checks that type-unsafe candidates are rejected: person questions do not return URLs, paths, or IDs; URL questions return URLs rather than nearby names; organization questions reject bare structural identifiers; metadata-only hits cannot answer non-metadata questions; JSON-like raw text supports generic key/value lookup; and low-semantic/cache-like text is downweighted without being discarded. Fake local-model tests verify that chunk-frame extraction, query-frame calls, bounded verification, and evidence extraction are invoked, counted, source-grounded, and rejected when proposed evidence is incompatible.

## Limitations

The fixtures are self-written and useful for regression, but they should not be treated as proof of broad real-world generalization. The broad raw-world score is reported honestly, but the next step should be fresh generated holdouts and mutation/paraphrase variants that were not visible while implementing these deterministic fixes.

Next evaluation work should add:

- generated holdout folders,
- entity-renamed mutation suites,
- paraphrased question suites,
- adversarial context/scope tests,
- larger random raw-folder corpora,
- independent external evaluations only after internal generalization improves.

## External Evaluation Diagnostics

External evaluations are treated as diagnostics rather than training data. Any adapter may only pass a folder path to `initialize(folder_path)` and question strings to `question(text)`, then format the returned strings for the outside harness. Core KMD must not consume hidden answers, labels, dataset categories, metadata wrappers, or source conversions.
