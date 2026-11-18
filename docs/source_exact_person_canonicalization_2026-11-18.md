# KMD source-exact person canonicalization test — November 18, 2026

## Purpose

Avoid a redundant model formatting call for a complete person name already present verbatim in original-source evidence, while retaining the independent model entailment verification. This change does not bypass strict query DRS planning, model-supported source DRS ingestion, source provenance or independent verification.

The shortcut applies only to answer-type person, two-to-four-token candidates, unchanged canonical formatting, and full-token-boundary exact appearance in source evidence. Other types, partial names, and absent evidence retain the existing model canonicalization path.

## Measured results

Using a fictional single-file corpus, the actual public KMD initialize operation ingested one chunk and accepted a compact source DRS from a real local Qwen Q4 model. The graph contains two DRS conditions and three DRS referents. A separate real local Qwen Q8 model produced an accepted query DRS for the question "Who reviewed the Falcon Test Plan?". KMD's native bounded DSPG executor returned "Mira Chen" with the original file and exact supporting text. A later public question operation on this branch, after Q4 ingestion and controlled Q8 query and verification, returned "Mira Chen", with the Q8 independent model verifier accepting the source entailment.

Cold structured model calls remain slow under concurrent load. The cached plan and bounded graph execution were fast. This does not certify a general production deployment or end-to-end single-model uncached performance. No existing service or production KMD checkout was changed.

## Tests

Four new unit cases cover the exact source match, substring collision, other answer types, and missing evidence. The unit and smoke suite passes when excluding one confirmed pre-existing count-evidence regression, which reproduces independently on the untouched original checkout. See the private Raspi KMD demonstration logs and the linked Multiverse source-evidence explorer for authentic results.

## Source-revision cache safety check — November 18 follow-up

A second isolated, owner-authored fictional corpus used the same relative file name and the same question, "Who reviewed the Falcon Test Plan?", but replaced the name with Elena Frost. The first separate corpus's independently model-verified answer was Mira Chen.

The changed source was actually reprocessed by the existing Qwen Q4 ingestion model, which successfully materialized two DRS conditions and three referents. The first bounded run stopped at its time limit after completing ingestion; a second run reused the newly accepted cached source DRS and the still-valid unchanged Qwen Q8 query plan. The public KMD question() call returned Elena Frost, with the revised original source. The Qwen Q8 independent verifier returned an accepted, freshly generated verdict (fresh_or_cached=fresh). It did not reuse the earlier Mira Chen verifier response. Original changed-source SHA-256, query, verification stage outcome and exact source evidence are in the separate private Raspi test report, and a sanitized comparison is published under website assets/kmd/revision-proof.json and kmd/evidence/#source-revision.

This is a controlled two-source-snapshot test of cache invalidation, not an in-place file-change update benchmark, large-archive test, or general correctness claim. The revised source's cold structured ingestion remained slow under shared model load. Existing KMD workers and their production data were unchanged.

The KMD public API retains independent entailment verification even when the new exact-source-person formatting shortcut applies. The source branch is isolated and should not be merged without review by the active KMD project owner.
