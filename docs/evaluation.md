# Evaluation

## Fixture Suites

KMD currently keeps three self-written fixture suites:

- `tests/fixtures/messy_raw_corpus/`: original project-style regression corpus.
- `tests/fixtures/broad_raw_world/`: heterogeneous raw-world corpus across school, family, household, fiction, law-like notes, medical appointment notes, veterinary notes, geography, language learning, recipes, travel, sports, art, research, logistics, accounting, schedules, diagrams, tables, OCR-like text, multilingual fragments, aliases, and conflicts.
- `tests/fixtures/hardcore_noise/`: random-character, base64/hex-like, word-salad, multilingual nonsense, OCR garbage, plausible babble, and adversarial distractor pollution.

All fixtures are raw text only. They use arbitrary nested folders, arbitrary filenames, mixed file endings, files without extensions, prose, tables, chats, logs, transcript turns, JSON-like text, dreams, beliefs, allegations, contradictions, noisy text, IDs, URLs, and distractors.

## Current Score

Current results:

- original messy corpus: `60/60 (1.000)`
- broad raw-world corpus: `65/65 (1.000)`
- hardcore noise corpus: `8/8 (1.000)`

The current fixtures all pass. This is still a self-written regression suite, not proof of broad real-world generalization.

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
- context records,
- frames,
- frame arguments,
- temporal edges.
- generic relation records.

Architecture tests assert that the core package contains no old benchmark/prepared-input markers and that `knowmoredirt.__all__` exports only `initialize` and `question`.
They also scan the core package for fixture/domain-shaped literals from the regression corpora so future changes do not quietly reintroduce content-specific answer branches.

The current answer path has been refactored toward generic DSPG mechanisms:

- label/value and raw text key/value relations,
- active/passive event relations,
- negation/proof/status relations,
- temporal state relations,
- identity/alias relations,
- table row/cell relations,
- filesystem/read metadata and document quality contexts.

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

External evaluations are treated as diagnostics rather than training data. KMD adapters may format public `question(text)` outputs for a scorer, but core KMD must not consume hidden answers, answerability labels, dataset category labels, prepared metadata wrappers, or source conversions. Low retrieval/citation scores identify engineering work rather than reasons to add shortcuts.
