# Raspi native KnowMoreDiRT summary indexing: real acceptance

Date: 7 November 2026. This is an implemented, completed two-document
production FileDB pilot, NOT a claim of complete offline library indexing.

## Original problem

The older native KMD content pipeline had recursive facet-reduction code, but
the active Raspi library worker used only three separate sampled excerpts per
original document. It generated no complete-book LLM summary and no searchable
summary-of-summary representations. It was designed to reduce LLM compute
and keep the original FileDB below approximately 640 MiB, but it did not
implement the user's intended source-complete hierarchy.

The active local Qwen endpoint advertises 16,384 configured context tokens.
The E5 multilingual embedding model accepts no more than 512 tokens.
Source packing must use the actually configured model context and reserve
the summary output. Every summary must fit the embedding context or be
rejected; source text must never be silently truncated.

## Implemented components

The hierarchical_summary module reads every character of the current
validated derived source, splits at real page markers or chapter headings,
packs as many sections as fit the real model's tokenizer/context while
reserving output capacity, and recursively summarizes the summaries until
the root covers the entire document.

Every node retains original document identity and source text offsets,
a source SHA, its summary text and SHA, a level, child IDs, model context
and prompt/generation budgets. Every generated summary is a separately
indexed retrievable item, not just the final root.

The model adapter uses the private local Qwen endpoint with source-guarded
constrained JSON generation and exact model-call caching. The native
storage writer creates one source-linked KMD chunk plus one embedding
representation per summary node in the EXISTING original FileDB, with
half-precision 384-dimensional vectors and source-revision checks.
There is no second semantic database. Storage writes are opt-in,
idempotent, and guarded by foreign keys and a database-size checkpoint.

The new native summary search reads every level and groups ranked
summary hits under the original filesystem entry. It returns multiple
matched summary nodes per document, not only the document's best chunk.

The operational Raspi command is /data/bin/knowledge-library-summaries:
the commands status, search, build and import support private
document discovery, verified complete-tree generation, and explicit
native FileDB insertion. Original/derived checksums are verified before
generation and import, and query-time source revision checks are
applied before returning indexed summaries.

## Real complete-document results

The complete original HiChunk 2026 ACL research PDF has 107,083
extracted source characters in 17 source sections. It produced
two first-level summaries and one root summary-of-summaries.
The native FileDB contains all three summaries and all three
real E5 vectors, with the correct original path, source spans,
two root child references, source SHA hashes and embedding hashes.

A second actual original, the Hick's Law reference article, has
4,218 source characters and generated one full-document summary
and one vector. The FileDB now contains four summary representations
for two original documents, plus 70,575 unchanged earlier passage
vectors. SQLite integrity and foreign keys pass. Reimporting
the HiChunk hierarchy added zero duplicate rows.

Read-only on-disk checks verified every summary, source span,
summary vector, original source SHA and derivative SHA, and
confirmed the full-book roots covered all extracted characters.
Two real semantic document queries correctly preferred
HiChunk for automatic hierarchical evidence retrieval and
Hick's Law for choice-related decision time.

An earlier HCI textbook experiment tested a three-page
GUI chapter summary, which ranked very highly against all
existing native document vectors. That partial chapter
experiment has not been mislabeled as a complete textbook
summary or imported as one.

## Production raw-passage compatibility

The existing Infra semantic query wrapper previously ranked
native representations before checking whether their source
excerpts could be resolved. A higher-scoring summary could
therefore suppress the best verified original passage.

A small live compatibility guard excludes summary collection
rows from legacy raw-passage search BEFORE best-per-document
collapse. Dedicated native summary search still sees every
summary vector. Five new dedicated Infra guard tests and
nine earlier native-scope tests pass. Two unrelated
GUI/Hick raw-passage controls retained identical chunk IDs,
while a HiChunk raw search recovered the actual original
passage after the summary shadowing defect was prevented.

The guard was changed narrowly in the existing concurrently
dirty Infra query wrapper, with its prior bytes preserved
privately. It is not staged with another agent's existing
uncommitted changes. The isolated KnowMoreDiRT research
branch owns the new summarizer, native storage/search, operator,
tests and documentation.

## Research conclusions and remaining work

RAPTOR, ICLR 2024, supports embedding and retrieving both
local and recursively generated summary levels. HiChunk,
ACL 2026, is available in this exact Raspi offline library
and supports adaptive retrieval over hierarchical structure.
Context-Aware Hierarchical Merging, ACL Findings 2025,
demonstrates factual errors can compound during recursive
summarization. Summaries are therefore source-linked
retrieval clues rather than factual answers.

The first functionality, finding the original document
through any summary level, now works on two complete real
sources. The remaining source corpus is NOT recursively indexed.
Further work includes a measured all-library generation
schedule, actual multi-level long-book runs, held-out relevance
evaluation, robust resumption, and a suitable disk budget.
Do not claim perfect search accuracy or complete document
coverage before independently testing that larger workload.

## Continuous worker startup, 7 November 2026

The owner subsequently explicitly authorized starting the whole-library
recursive-summary run. The enabled Raspi systemd unit
knowledge-library-hierarchical-summary.service now executes
scripts/raspi_hierarchical_worker.py continuously, under low CPU and
I/O priority, a memory/cgroup bound and a singleton file lock. It uses
the local Thor Qwen endpoint and the native Raspi E5 vector model.

The worker selects all completed, source-mapped originals, creates
the complete recursive tree for each, verifies hashes and source spans,
and imports *every* node/embedding into the canonical native FileDB.
It records a current document, completed/remaining counts, percentage,
retry state, and last successful source in a private JSON status receipt.
It refreshes the authoritative database counts every ten successful
originals and updates progress after each completed document.

The original 640-MiB sample-only FileDB checkpoint has been superseded
by a bounded 4-GiB cap for this owner-authorized full-library task.
The earlier native embedding worker's systemd cap is aligned separately
in the Infra research branch. The data disk had around 303 GiB free
at startup. This is still a hard limit, not unbounded disk growth.

The first real live samples completed and were inserted successfully
after service activation. The worker restarted safely on a completed
document boundary to apply the progress counter correction and continued
processing, with no observed process failure. Each failed source
receives a persisted exponential retry while other originals continue.

The active KMD reader has a separate native summary-search API so
ordinary hash-verified original passages cannot be hidden by higher-scoring
summary embeddings. Searches validate the best-ranked candidate
documents' revisions rather than hashing all seventeen thousand
original derivatives for every query. Original publisher binary SHA checks
use bounded streaming reads, rather than materializing a large book
entirely in memory.

The service is running, but *only* the observed completed files have
recursive summaries. Generated source and summary data remain private
outside Git; the worker source and synthetic tests belong in the
separate KnowMoreDiRT research branch.
