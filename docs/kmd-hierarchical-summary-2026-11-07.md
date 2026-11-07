# Source-complete hierarchical summaries in KnowMoreDiRT

Implementation and real test evidence: 7 November 2026. This isolated
research branch is not installed into the Raspi production search worker.

## What was wrong

The current KnowMoreDiRT FileDB on Raspi keeps one canonical metadata
database plus its native semantic vectors. The Infra worker does **not**
make full-book summaries. Its compact overview usually samples three
distant excerpts, then embeds those fragments. The original native
content_representations table has zero generated analysis rows.

The older native KMD ContentSemanticPipeline._analyze_file already
recursively reduces **facet analyses**, but that is not a general
whole-book summary and is not used by the live Raspi source ingestion.
The native search also keeps one best vector per document before the
Infra wrapper checks citation provenance.

## New implementation in this branch

The hierarchical_summary module reads complete extracted text, preserves
every source character and all original offsets, detects page or chapter
breaks, and packs neighboring source sections into the model's context.
The packing algorithm reserves a real generation-token allowance and
safety allowance and uses exact active chat-template/tokenizer counts
for every accepted group. Large isolated sections split without dropping
text. Every first-level group gets a summary, and ordered groups of
summaries are recursively compressed until one document root remains.
All intermediate summaries remain separately addressable by exact
source ranges and child links. A bounded pilot that stops early
explicitly fails as incomplete and does not pretend to have summarized
the entire document.

The hierarchical_summary_model module uses an explicitly configured
local llama.cpp model endpoint. It enforces a fixed output-token
reservation, a strict summary-only JSON response, finite timeouts,
the KMD untrusted-source guard and the existing local model-call cache.
Summaries are marked as lossy retrieval aids rather than source truth.

The hierarchical_summary_store module maps every summary node into
the existing native KMD content_chunks and content_representations
tables without creating a new database or schema. It validates original
file IDs and sizes, supplied completed-source checksums, complete
derived-text checksums, exact source-span hashes, summary-tree integrity,
embedding shape and model identity. Each vector is stored in the native
half-precision 384-dimensional format, requiring 768 bytes per
vector. A separate explicitly opted-in transaction guards database
capacity, foreign keys, idempotence and conflicting revisions.

**Do not yet enable this writer against live FileDB.** The current
Infra semantic result resolver only understands its pre-existing
overview/full source-unit mappings. It needs a summary-aware source
resolver and correct best-per-document candidate handling first.
Activating summary embeddings before then risks hiding otherwise
valid hits when a summary wins KMD's global cosine ranking.

## Measured live model pilot

Thor's available Qwen model advertises 16,384 tokens of configured
context, despite a larger pretraining window. The exact current HCI
book extraction contains 2,752,515 characters. It has 829 physical
page markers and 830 complete sections including its prologue.
A full-book deterministic tokenizer-estimate planning check found
48 first-level groups, covering all 2,752,515 characters. This is a
planning estimate, **not** 48 real generated summaries or exact Qwen
token counts for the full book.

A separate, exact live Qwen tokenizer check of the first eighty
sections found four context-compatible groups with prompt tokens
15,017, 15,359, 14,845 and 7,343. Each group reserves 640
output tokens and 384 safety tokens. The fullest therefore uses
15,359 + 1,024 = 16,383 out of 16,384 configured tokens.
Every one of the eighty sections was covered. The improved planner
completed the bounded exact check in approximately 31 seconds.
The prior whole-book, overly broad exact-token planning probe was
stopped after repeated expensive network tokenizations and replaced
with the more efficient nearby-size search. It generated no text
and changed no database.

A real three-page section from the 2025 HCI textbook, pages
503 through 505, was read from its checked original-derived
source and summarized by Qwen with a clean completion. That
summary took approximately 27 seconds and produced 1,657
characters, measuring 401 tokens in the existing local
multilingual E5 embedding model. On three GUI questions,
that summary's vector would have ranked fifth, second and third
among the existing native document candidates; the previous
sampled source representations ranked thousands of positions
lower. These ranks are *hypothetical read-only vector comparisons*
for one chapter, not a full-book accuracy guarantee.

An independent in-memory FileDB with the original KMD tables,
native real original-file ID and real E5 vector then accepted
the Qwen summary as one native summary_long representation.
The 768-byte vector was searchable through unmodified KMD
semantic search, its original source offset and SHA matched,
SQLite integrity was OK, and foreign keys passed. No live
database or original document was written or replaced.

## Research and quality limits

The local Raspi Knowledge Library has the full 2026 ACL HiChunk
paper under ai-agents/2026-acl-source-retrieval/2026.acl-long.1372.pdf.
RAPTOR, ICLR 2024, describes recursively summarizing, embedding
and retrieving multiple abstraction levels:
https://proceedings.iclr.cc/paper_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html

Context-Aware Hierarchical Merging, ACL Findings 2025, shows
that recursive merging can compound hallucinations and benefit
from grounding in the actual source:
https://aclanthology.org/2025.findings-acl.289/

The root summary is useful for broad document discovery but is
necessarily lossy. The production search should retrieve both root
and lower-level chapter summaries, then verify exact original
passages and document-level context. Summary text must never
be presented as the original authoritative evidence.

New tests cover complete section coverage, splitting, multi-level
tree provenance, explicit context and output budgets, incomplete
run rejection, real local-model request constraints, native
representation embedding, half-precision storage, source identity,
idempotence, FK safety and rollback on incompatible revisions.
Existing hybrid and native FileDB tests also pass with isolated
model-call cache state.

Next: finish summary-aware source resolution in the dirty Infra
integration safely, run a real multi-level **full-book** pilot,
benchmark quality with held-out queries and contradictions, and
measure per-document storage and inference cost before enabling
recurring source summarization under the existing approximately
640 MiB FileDB checkpoint. The full-book model generation has
not been completed or deployed.
