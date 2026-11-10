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
