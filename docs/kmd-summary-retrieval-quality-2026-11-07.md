# Real native KnowMoreDiRT summary-retrieval quality evaluation, 7 November 2026

The owner requested independent actual-query and original-file acceptance rather
than relying on database integrity or merely successful vector insertion.
This report records executed read-only tests against the ORIGINAL KMD FileDB
and the continuously running Raspberry Pi recursive-summary index.

## Search command and exact persistent query audit

The Raspi-only operator command knowledge-library-summaries search now
automatically records every completed query and its complete ranked original
source metadata in a private daily JSONL file. Failed queries record their
error state, rather than being silently forgotten. Each receipt contains UTC
timestamp, query, model, canonical database path, result count, elapsed time,
ranked original paths, titles, raw cosine similarities, best summary and
matching summary-node levels, source and summary revision digests and
short summary excerpts. Newer receipts also contain the live summary-index
document/vector coverage and an explicit nearest-neighbor-not-factual-answer
warning. All private files use owner-only permissions and OS-level append locks.
Logs are on the Raspi high-capacity data partition, not in GitHub.

The operator can review the latest receipts with knowledge-library-summaries
search-log --last 10. This is a read-only view of persistent search history,
not a replay or a new provider request. The daily JSONL is held under the
private KnowMoreDiRT knowledge-library/search-audit area, with dated per-day
rotation and a 64-MiB-per-day cap. A newly added unit suite verifies exact
query/rank retention, source revisions, file permissions, symlink refusal,
bounded record sizes, the CLI's mandatory success/failure logging and
round-trip review.

## First real benchmark: 28 known source documents and five absent topics

The thirty-three query prompts and expected original paths were fixed before
running the tests and retained in private evidence. Every query went through
the same real public CLI, local multilingual E5 query embedding, canonical
native summary-vector search and source-revision validation, with actual
results saved automatically in the JSONL.

Of 28 known-answer source-finding queries, **28 found the expected original
at rank one**. These were hand-authored queries closely aligned to titles
and topics of original documents already known to be summarized:
Unity rendering and UI Toolkit, Blender material nodes, Adobe Substance,
UX design, dock-leveler engineering drawings, a financing spreadsheet and
the HiChunk paper. This is a positive functional smoke suite, NOT an unbiased
claim of 100-percent real-world retrieval precision.

Five requests whose desired source was not yet in the recursive-summary
index still returned nearest neighbors, including an unrelated
operating-systems errata paper for an ETF portfolio question. Cosine
similarity is not probability of correctness or proof of topic coverage.

Private fixture, complete logs and result report:
search-audit/known-answer-benchmark-20261107/.

## Harder independent paraphrase and multilingual benchmark

Thirty-six additional prompts were independently written and fixed before
their live results were inspected. Of 28 known-answer indirect, paraphrased
or German-language questions, the exact expected original was:
- rank one: **18 of 28**
- in the top three: **25 of 28**
- in the top five: **26 of 28**
- in the top ten: **28 of 28**

All six deliberately German-language known-answer probes ranked their
expected original first. However, difficult English questions about surface
normals and text-embedded icons had the correct original only at ranks
six and eight. Several other questions had an arguably relevant alternative
in the first slot, such as a generic Unity lighting page instead of the
more focused shadow-cascade visualizer. Exact-single-target scoring can
underestimate useful alternatives but captures the user cost of locating
the specific intended original.

Eight unsupported or not-yet-summarized requests also returned nearest
neighbors, sometimes with high similarities. One question about procedural
Geometry Nodes found a real, newly indexed shader Geometry Node document:
this was partially topically relevant but did not establish the requested
procedural point-distribution instructions, and must NOT be counted as
verified source coverage. A question specifically about Unity HDRP ray
tracing returned Unity HDR color documentation with a similarity around
0.87. A fixed cosine threshold chosen on the first easy test suite would
not cleanly separate all correct, partial and irrelevant hits.

Private fixture and full audit:
search-audit/hard-retrieval-benchmark-20261107/.

## Fair like-for-like original-sample versus generated-summary comparison

Both retrieval methods were compared against the SAME fixed set of 84
documents that had both old excerpt vectors and new generated-summary
vectors at the time of the experiment, using the same E5 question
embeddings. The combined two positive test suites supplied 56 known
expected source documents.

Old three-source-excerpt cosine search found:
- top one: **40 of 56**
- top five: **52 of 56**
- mean reciprocal rank at ten: **0.8081**

Generated hierarchical-summary cosine search found:
- top one: **45 of 56**
- top five: **55 of 56**
- mean reciprocal rank at ten: **0.8737**

Thirteen questions improved rank, six worsened rank and thirty-seven
remained equal. This shows a genuine benefit of meaningful generated
summaries over three arbitrary original snippets on these candidate
documents. It also proves that summary generation alone does not solve
relevance on every question. The study controls candidate source set,
but is still a deliberately selected test collection rather than a blind
external benchmark.

Private reproducible metric output:
search-audit/same-doc-overview-vs-summary-20261107.json.

## Source and stored-vector integrity

An independent read-only provenance check re-read the actual original
binary files and derived source text for 75 already indexed originals,
verifying every complete source SHA, derivative SHA, summary node
source-span SHA, complete root coverage, per-level source range continuity,
parent/child links, summary digest and serialized E5 vector digest.
It checked 77 live summary vectors, including actual embedding token
lengths between 86 and 404 (below the model's 512-token input limit).
There were zero verification failures. SQLite quick_check returned OK
and foreign_key_check had zero errors. This proves database/source
consistency for the checked documents, not correctness of every
LLM-generated statement.

Private raw diagnostic receipt:
search-audit/source-provenance-readonly-audit-20261107.json.

## Current reliability status and priorities

The native recursive summary database and original-file discovery command
are already usable on the documents that have been summarized. Both
the easy and harder functional searches ran with zero request failures
and their exact query/ranking receipts were preserved.

**Not yet reliable across the full Knowledge Library:** the worker
continues to index from a very small percentage of more than seventeen
thousand eligible originals. The summary search returns candidates even
for entirely unsupported questions; it cannot yet reliably abstain.
The rank-one result is sometimes merely a related document. No
model-independent source-grounded factual answer verification has been
implemented by nearest-vector search.

Do not call an uncalibrated cosine score a probability or infer complete
whole-library coverage. Future evaluation should add a truly blind
holdout, alternative-relevant-document annotations, domain-balanced
positive and negative questions, multilingual source coverage and
a refusal/abstention model grounded in actual source text. A hybrid
query/ranking pipeline with title, terminology and source context may
improve hard paraphrases, but it must be benchmarked instead of assumed.

The private full query/rank JSONL and two benchmark receipts are preserved
for future retrieval regression comparisons as more originals are
summarized. No downloaded books, full extracted texts, complete vector
databases or private query logs are committed to Git.
