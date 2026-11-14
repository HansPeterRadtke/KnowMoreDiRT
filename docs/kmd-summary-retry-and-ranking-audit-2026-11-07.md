# Recursive summary reliability increment — 7 November 2026

This increment focuses on live summarization failures and independent search
relevance rather than changing the user's one-native-FileDB architecture.

## Indexing worker status at investigation

The continuously running Raspberry Pi hierarchical worker was active with
no unexpected restarts. Around 13:49 CET it had indexed 140 of 17,714
extracted original documents (about 0.79 percent), with 142 summary vectors.
It continues in the original KMD FileDB. The small-first scheduling policy
means this is document progress, not proportional text coverage.

A small number of short Unity documents repeatedly failed with
"local summary did not finish within reserved generation tokens".
The original retry increased the output allowance from 320 to 480 tokens
but used the same verbose instructions, allowing deterministic
output-length failures to recur.

## Fix: source-complete but bounded short-summary fallback

The summarization implementation now retries only actual output-length
exhaustion with a deliberately shorter, source-grounded system instruction.
It asks the local model to finish a concise summary of at most 115 words,
retaining actual source topics, key conditions, negations and important
scope qualifiers. It recomputes the prompt's exact active-model token
cost and never exceeds either the configured context/safety budget
or the small E5 embedding input window. Other errors remain errors.

The SummaryNode stores its actual generation token allowance, prompt
token count and prompt policy. The native FileDB's generation provenance
carries the policy, so fallback-produced texts can later be audited
rather than presented as standard summary generation. Old saved nodes
with no prompt policy are marked legacy-unrecorded on reading.

A targeted *real local Qwen* test retried the previously repeatedly
failing complete Unity "Packages" original, generated a 663-character
complete original-linked summary and inserted its vector into the
original native FileDB under all existing provenance/idempotence checks.
This was a short original, not a whole-book proof. The test left the
background worker running.

The worker's private failure count also included entries for sources
already completed in native FileDB through another authorized test run.
The worker now discards stale, already-processed or no-longer-eligible
failure receipts when refreshing its inventory. It retains genuine
pending failures and its existing exponential retry delay.

Focused synthetic tests cover the source-complete strict fallback,
the live context/output reservation, source-provenance recording,
non-length failures and stale retry pruning. Unmodified native source
and storage suites are run in an isolated cache to avoid old mocked
model-call collisions.

## Exploratory retrieval ranking work — rejected for live deployment

The existing private known-answer and difficult-paraphrase benchmark
contains 56 expected original-document queries. An exploratory parameter
sweep tested cosine similarity from generated summary nodes, old sampled
overview vectors, and query-term/title SQLite FTS5 rank fusion against
the same candidate originals. Some weighted blends modestly improved
rank-one recovery on the existing queries, but this was a tuning set.

An independently written follow-up holdout fixed 27 newly authored
questions from original already-indexed Unity, Blender and n8n
documents *before* running the tested blend. Against the 151 then
indexed originals, the pure summary-vector search returned 22 of 27
correct target originals at rank one and 25 within rank five. The
candidate rank blend returned only 20 correct rank-one results, though
it did place all 27 targets within rank five. Some exact source titles
were suppressed by lexical/reused-old evidence.

An additional simple summary-plus-old-overview top-five union did
not recover the two missing targets on a later enlarged corpus.
Consequently the rank blend and union are **not deployed** to the
production search path. Exact current native summary cosine ranking
remains authoritative for the primary document order. Search audit
logging and explicit incomplete-index/nearest-neighbor warnings
continue unchanged.

These experiments show that summary vector retrieval is useful but
the best-scoring original cannot be presumed correct, and simple
lexical/title weighting is not a universally reliable correction.
Future ranking should be judged on a larger source-balanced heldout
set, source context, explicit authority and an independently calibrated
reject/abstain mechanism. The private exploratory scripts and results
remain under the Raspi search-audit data directory, not committed to Git.

## Safety and next goals

The local model's initial context is about 16 thousand tokens and the
existing E5 embeddings are limited to 512. The complete source text
and original source SHA remain the retrieval authority; all summary
levels are lossy discovery aids with original file IDs and spans.

The active service is to be restarted only on a real completed-original
boundary after the new KMD commit is installed in its pinned checkout.
That preserves the current source process, native vectors and private
other agents' concurrently dirty Infra files. After restart, inspect
service PID, progress, failed count, summary vectors, query history,
SQLite integrity and retry cases rather than claiming success from
a service-active flag alone.

## Compact native facet retry on 10 November 2026

Both live recursive indexers used the compact native facet prompt and a
2,400-token output reservation. On a generation cutoff, the fallback
selected the same prompt, allowance, source and seed. A regression test
reproduced an identical second request. Nine pending originals had a
cutoff as their most recent error. Their ledgers recorded 119 source
attempts in total, which do not establish how many requests failed
for this particular reason.

The compact-mode fallback now asks for a summary of up to 60 words,
up to three facets and up to two short terms per facet. These are model
instructions. The existing native JSON schema, safety reservation and
embedding admission checks still apply. The engine remeasures the prompt,
passes the full original source group and records the policy on the
resulting node. A second cutoff leaves the source incomplete. A different
error bypasses the fallback. This change applies during generation.
Previously saved trees that exceed the embedding limit require separate
recovery.

The tests exercise the workers' real output allowance, the rendered
transport requests, cache identity and rejection of partial responses.
The operational receipt and live outcomes are kept privately on Raspi
in the search audit named `compact-length-retry-20261110`.

The first successful compact retry for a real IHK handover checklist
repeated an incorrect year in the extracted text's management header.
The publisher printed its edition on physical page one. The pilot kept
the misleading output as private rejected evidence before import.
Summary builds now include the curated edition in the model's document
label when the exact edition is also present on a line of the extracted
publisher content. Header-only dates are ineligible. The complete source
bytes, offsets and digest still refer to the original verified derivative.

A second private pilot correctly dated the summary but also emitted the
superseded catalog year as a keyword. For an extractor-owned header whose
source path and SHA match but whose year differs from the corrected
manifest, the generator now sees only the publisher content. Native
offsets, the complete derivative digest and the archived header remain
available for provenance. The first real publisher passage owns that
prefix in storage. The header cannot become a separate generated leaf.
The tests include a nonmatching source identity, genuine publisher
preface, unchanged year, oversized first page and multilevel coverage.

### Verified native outcome

The outgoing-owner checklist had 20 recorded source attempts and no
recursive summary rows before the live pilot. Its first successful
length fallback was preserved outside the index because it repeated
the wrong catalog year. After the source record was corrected and
the stale management header excluded from the model input, the normal
compact request succeeded. The reviewed summary identifies the May 2022
edition. Its source span covers the complete archived derivative and
all six of the publisher's pages. The accepted summary, six topics and
18 keyphrases contributed 25 verified vectors. All 256 sampled older
vectors kept their recorded byte digests.

Of two questions fixed before import, the German question asking for the
IHK outgoing-owner checklist went from absent in the first eight to first.
The broader English question on preparing a family-business handover
still missed this particular original in the first eight. Neither result
establishes the accuracy of other summaries or complete corpus coverage.
The private evidence includes both rejected intermediate outputs and
their reasons. Seventy-two distinct KnowMoreDiRT tests, fifteen Infra
business tests and twelve documentation tests passed.


## Recovering a source that exceeds the passage encoder budget

The native builder measures each generated summary, topic, and representation
with the deployed passage tokenizer, including its prefix and special tokens.
One shorter native-schema request can recover a model length cutoff or a
measured passage overflow. It receives the same complete source group. Both
requests count against the pilot's model-request limit, including cache hits.
Progress receipts count processed nodes, adapter requests, successful calls,
and reused saved nodes separately.

For an existing completed tree that failed import, run the usual native build
with the same model and add `--recover-embedding-context`. The command checks
the publisher and derivative revisions and confirms an actual overflow. It
preserves the saved tree and writes the complete recovery beside it, keyed to
the SHA-256 of the predecessor's exact bytes. The ordinary import selects and
validates that recovery. A dry import computes every vector before admission.
Concurrent writers expose one complete file atomically. A different result
cannot replace an existing file.

Recovery reuses intact nodes when their source interval and exact child IDs
match. On a rejected node, admissible original terms retain their native facet.
A shorter model result can supply the rejected summary or additional fitting
terms. The original tree remains available for comparing both outputs.

In the Raspi dry runs, the NVIDIA setup guide had a 523-token summary and
MICAS had a 642-token keyphrase against the encoder's 512-token limit. The
corrected trees retained all 46 and 67 previously admissible fields,
respectively. Two of the three MICAS nodes matched their saved records in
full, including the root. The ordinary importer prepared 47 and 76 vectors.
A read-only replay also matched all 139,966 indexed representation identities,
ranks, kinds and text hashes in the fixed 4,045-original regression cohort.


### Checking exact identifiers in the original

After document discovery, pass an exact configuration variable or tax-code
identifier to the existing `source-evidence` command. Its in-memory FTS and
excerpt selection recognize internal underscores as part of the name. Source
evidence requires the literal in the publisher body. Generated catalog headers,
unit labels, longer identifiers and space-separated lookalikes cannot substitute
for it. Excerpt offsets and hashes identify the exact verified derivative bytes.

The two recovered documents prompted a live comparison. The earlier passage
locator returned no excerpt for any of the nine distinct uppercase configuration
identifiers in the NVIDIA publisher file or the 135 native source-anchored
Stripe identifiers. The corrected locator found every one of the 144 and
verified its excerpt against the complete derivative. Two absent identifiers
returned zero passages. Five ordinary questions across three documents kept
their original ranked section lists.


### Live admission and regression results on 11 November

The canonical FileDB accepted the NVIDIA ontology setup guide, MICAS,
and VisionZip. The three sources contributed seven hierarchical nodes
and 191 source-linked vectors. All three publisher and derivative hashes,
all node intervals, all representation identities, and all normalized
384-dimensional vectors were verified after import. Both pre-existing
summary files retained their original byte digests.

Five of six predeclared questions placed their intended original in the
first eight results after admission, including four first-place results.
All six had missed before admission. The remaining MCP configuration
question put the setup guide eleventh under summary search and thirteenth
with facets. The existing filesystem `find --path nvidia-ontology-setup
--name SKILL.md` command resolved that guide uniquely. A separate
file-name comparison included one skill with two archived revisions,
for which a name alone could not choose a unique publisher file.

The paired experiment added exactly the 191 admitted vectors to the
previously frozen 139,966-row cohort. Across 39 previous questions in
both search modes, every one of the 624 top-eight positions retained its
source path, rounded similarity, matched node identity, and text digests.
A separate comparison of every field and stored blob in the live
pre-admission prefix matched all 75,866 chunk rows and 140,911
representation rows. The workers continued adding other new rows.

At the time of the capacity audit, 13,639 eligible sources remained.
In the preceding 24 hours, 659 newly indexed originals contributed
28.6 million source characters. At that throughput, indexing the remaining
2.74 billion characters would take about 96 days. Applying the same
observed storage density to those characters would add 6.24 GiB for vector
blobs, generation metadata and representation text alone. The configured
4 GiB checkpoint had about 2.85 GiB of headroom. Both projections depend
on the future document mix and throughput. The detailed per-source,
per-query, and storage measurements are under the private Raspi
`search-audit/embedding-context-recovery-20261111` directory.

### Finding a named original and citing its publisher text

To find a named guide, use an exact catalog title, original path, or a publisher-qualified archived skill name. For example:

```sh
knowledge-library-summaries search 'What does the NVIDIA ontology setup guide say about configuring the MCP connection?'
knowledge-library-summaries source-evidence 'What does the NVIDIA ontology setup guide say about configuring the MCP connection?'
knowledge-library-summaries source-evidence --source-name 'NVIDIA Agent Skill: nvidia-ontology-setup' 'MCP connection'
```

Search brings a uniquely named original to the first position only after its native rows, original file, derivative and character intervals pass verification. Scores retain their original cosine values. If no verified summary for that original is available, the result says so. Direct source evidence can still inspect an extracted original with a canonical FileDB entry. A shared name offers up to eight exact catalog paths, along with the total number of matches. Choosing an exact path keeps different archived revisions distinct.

The evidence locator uses publisher Markdown headings and excludes the generated catalog preamble. Shell comments in fenced code examples and YAML front matter do not create headings. Results include zero-based, half-open character offsets into the verified derivative. For UTF-8 Markdown and plain text up to eight MiB with a single plain-text extraction unit that reproduces the publisher file, the additional `publisher_excerpt` field gives half-open **byte** offsets and a digest for the raw publisher slice. Decode that slice as UTF-8 and normalize newlines to recover the displayed excerpt. PDF, HTML and other converted formats retain their extraction-unit labels and verified derivative offsets.

### November 11 named retrieval and provenance checks

A paired replay froze 142,100 native representations and ran 62 questions in both native search modes. The intended original came first in all 22 explicitly named runs, including the NVIDIA setup question that had previously ranked eleventh and thirteenth. Fourteen named runs improved. All 624 top-eight positions across the 39 older control questions kept their paths, scores, node identities and revision hashes. The 102 runs without a unique verified preference retained all ranked results.

In twelve publisher-evidence questions, the required details appeared together in the first five excerpts on all twelve, up from five with full questions on the previous version. Eleven additional questions from eight other publisher guides increased from two qualifying full questions to eleven. Ten of the eleven had the supporting passage first. Thirty-four older HTML questions kept identical ranked excerpts, offsets and scores. All 144 positive identifier checks found a literal in verified source text, and both absent identifier checks returned zero excerpts.

A separate sweep matched all 5,327 stored source intervals for 4,081 originals to an independent whole-text hash. It verified 5.82 billion original bytes and 102 million derivative bytes. Independent checks across all 733 archived skills reproduced 4,210 excerpt byte ranges from the actual publisher files. Of the 729 publisher-qualified names, 725 identified one path and four identified two paths each.

The search command now counts only the two coverage fields it displays. Over six alternating measurements on one read-only database snapshot, the median coverage query decreased from 1.30 seconds to 0.07 seconds, with identical counts. In the 62-question replay, median end-to-end retrieval decreased from 1.49 to 0.95 seconds for summaries and from 5.23 to 4.52 seconds with facets. Timings include concurrent index workers. Private protocols, per-query rankings and file-verification results are in the Raspi audit directory `named-source-retrieval-20261111`.

An additional ten-question panel found two letter-for-letter lookup misses. One source says “updates” where the question used “update”. Another puts likelihood filenames under “Outputs”. A bounded Markdown fallback now looks for regular English singular or plural forms after scoring literal matches. A multi-term fallback requires another distinct query concept in the displayed excerpt. For filenames, a complete underscore-delimited component can supply that second concept. Explicit identifier questions retain their whole-token requirement.

All 33 positive questions across the three panels then had the expected details together in a verified top-five excerpt. The fallback kept all 113 previously returned excerpts byte-for-byte and in order. Across 200 unrelated question and guide pairs it reproduced the same five incidental excerpts. Blanket promotion of publisher headings was also tested. It improved the NVIDIA connection passage but worsened several previously first-ranked answers, so the deployed ranking retains the original order.

### Recovering a missing search term from a printed PDF classification

A saved native summary can contain valid topics but no keywords or keyphrases. The importer checks the verified first PDF page for a `CCS Concepts` heading and a complete, single-line subject entry in the form `• category → subject;`. When a native import requires terms and the entire tree lacks them, the importer adds the printed subject once on the root. It records the extractor, page and half-open character interval in the verified derivative. Subject indexing uses a separate embedding batch because a change in padding can affect quantized vectors for existing terms. A search that returns the classification checks it against the current publisher and derivative revisions and the exact first-page classification block. The private search receipt includes the same coordinates. If the format is unsupported and the tree still has no terms, the import reports the unmet requirement.

The Eurographics paper *On the Accuracy of Surface Scattering Theories* had a complete saved summary but had failed native import eight times. Read-only preparation using the local encoder reproduced all ten original rows and their exact vector hashes, then added the publisher keyphrase “Reflectance modeling”. A separate check across thirty-two archived PDFs found five supported classifications and verified all five against independent extraction of page one from their publisher files.

The installed importer added one source chunk and eleven representations for the paper. All persisted fields and vector hashes matched the read-only preparation, apart from insertion timestamps. The saved summary tree retained its original hash. On a fixed 173,277-row baseline plus this paper, it went from absent in the first eight results to first in seven of eight query and search-mode combinations. The remaining case was the bare subject phrase in summary-only mode. Searching the printed phrase with facets enabled put the paper first, and source-evidence returned the classification on page one as its first excerpt. All sixty-four control positions retained their paths, scores and matched node identities.

Worker status presents both the failure count in the saved worker snapshot and the count still pending against current source revisions and complete FileDB roots. Historical receipts remain available after a separate verified admission. At this check the older snapshot still recorded the paper, while the index showed it was complete. Five archived skills remained on the ordinary processing queue, each with a verified local source and no recorded failure.

### Offline lookup by DOI or arXiv identifier

Paste a DOI, a DOI resolver URL, a modern arXiv reference such as `arXiv:2304.06828v2`, or an HTTPS arXiv abstract or PDF URL. The same identifiers work with a topic in a direct evidence query.

```sh
knowledge-library-summaries search '10.21437/Interspeech.2026-2080'
knowledge-library-summaries source-evidence 'According to 10.21437/Interspeech.2026-120, what does pairwise verification cost?'
knowledge-library-summaries source-evidence --source-name 'arXiv:2304.06828v2' 'incrementality'
```

DOI lookup uses the catalog's explicit DOI for a PDF. A related interview or reference page can link to a paper without providing its text. arXiv lookup takes the identifier from the catalog's archive download URL and checks it against any arXiv reference or identifier recorded for the same file. A versioned request requires the version on the download URL. A versionless request can match a pinned PDF, and the result reports its archived identifier in `source_name_match.matched_identifier`. Multiple matching paths leave the choice to the reader. The CLI checks the acquisition URL against the completed download receipt before verifying the file, extraction and source intervals. A well-formed identifier with no matching record returns zero search results.

The November 12 catalog had 507 DOI records and 270 arXiv PDFs. All 777 matched their acquisition receipts. Across 2,335 identifier queries, every supported spelling tested selected the independently expected path. Asking for an unarchived version of each arXiv record returned zero catalog matches. Of 35 archived PDFs inspected directly, 28 printed the identifier on the first page. Catalog matches use the recorded acquisition metadata and can also cover books or proceedings whose first page omits the identifier.

Two indexed papers previously missed the top three for their own identifiers. Both came first in summary and facet search after the change. Two papers awaiting a summary were available through direct evidence. For these four papers, eight entire excerpts were reproduced on their cited pages by extracting the physical PDFs again and normalizing whitespace. Four control questions over the same frozen native rows retained all 40 ranked positions, scores, node identities and source digests. The private protocol and per-query results are in the Raspi audit directory `publication-identifiers-20261112`.

## November 14, 2026 — native FileDB budget versus full-library indexing

The old four-GiB FileDB checkpoint was a deliberately bounded safeguard, not
a completed-corpus size estimate. At the November 14 live checkpoint the native
FileDB contained roughly 1.63 billion bytes, with 5,387 of 17,715 eligible
originals fully summarized; the total extracted original text was approximately
2.84 billion characters, only about 168 million of them under completed
recursive summaries. Both workers were active and reported zero pending errors.
The existing source inventory was fully downloaded; no new sources or original
text were fabricated for this capacity change.

A read-only 48-hour measurement counted 852 source-revision-linked documents,
approximately 47.9 million original characters and 107.5 million bytes across
three representation fields: native vectors, generated JSON and representation
text. Extrapolating **only those fields**, the remaining original characters
would add approximately 5.98 billion bytes at that observed density, with significant uncertainty.
That is not a complete FileDB size forecast: SQLite indexes, btree pages,
metadata, future document mix and other writers make the real required space
larger. The four-GiB checkpoint would therefore stop the two workers long
before full coverage.

The new bounded operating ceiling is **32 GiB** on the same original KMD
SQLite FileDB. It is a maximum, not a preallocation, demand to index every
source immediately, or permission to erase historic rows. The existing
transactional page-count and foreign-key checks remain mandatory.
Independent admission requires at least **32 GiB of unprivileged filesystem
free space plus 256 MiB for SQLite write headroom** before the shared
content-writer lock can proceed. The guard is checked after acquiring that
lock and rejects missing or symlinked FileDB paths. Failed admission does not
delete or replace sources or native vectors. The upper bound in the generic
native summary writer was raised from 16 to 32 GiB solely to support this
explicit, tested production checkpoint; smaller caller-specified limits
remain supported. Both worker programs use the same policy for newly started
processes, and the operator reports actual disk capacity and admission state
without claiming a broker disk reservation.

Raspi's data filesystem had approximately 317.7 billion *unprivileged*
available bytes at the measurement; this exceeded the safety floor.
The read-only reproducible monitor is:

```sh
/data/var/knowmoredirt/semantic-venv/bin/python \
  /data/src/worktrees/knowmoredirt-hierarchical-summary-20261107/scripts/raspi_hierarchical_capacity_audit.py \
  --window-hours 48
```

This command reports measured payload and a conservative *partial* remaining
projection, not a completion date or unconditional guarantee that 32 GiB is
sufficient. The actual database and progress remain authoritative. Additional
tests exercise low-space admission, unsafe paths, exhaustion at an explicit
limit, rejection beyond 32 GiB, and source-hash-preserving native rows.
The original long-running workers are not automatically restarted or
interrupted by publishing this policy.
