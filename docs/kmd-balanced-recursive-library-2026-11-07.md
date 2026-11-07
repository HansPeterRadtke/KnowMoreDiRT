# Balanced source-length scheduling for KnowMoreDiRT

The original Raspberry Pi continuous worker processed shortest extracted documents first. That produced reliable initial throughput, but after over three hundred originals had been processed, the document count was more than one percent complete while source text coverage was still approximately two hundredths of one percent. The active worker had not yet exercised many medium or large real source hierarchies.

The new queue selection interleaves document sizes without increasing language model concurrency. Every eighth scheduled original is selected from the 60,000- to 199,999-character category, and every fortieth is selected from sources at least 200,000 characters long. Other positions select shorter sources. Each category proceeds from its smallest remaining document upward. When a category is exhausted, the scheduler selects from the others and eventually visits every eligible original.

The schedule is computed from the current count of revision-valid completed originals and the remaining source-map backlog. The same original identity, source SHA-256 checks, native single-FileDB vector storage, singleton worker lock, resumable generation cache, bounded model-call count, failure retries, and low-resource systemd constraints remain in effect. No additional model process or second database is started.

The real read-only backlog preview at 306 indexed originals scheduled a medium source of 60,001 characters at global ordinal 312 and a large 200,129-character source at ordinal 320. This provides earlier real multi-level validation instead of waiting until nearly every short page has been summarized. The worker status records the scheduling policy after the next safe restart.

A deterministic test corpus confirms one medium source every eight positions, one larger source every forty positions, no duplicate or omitted original, monotonic size ordering within each category, correct behavior when restarting partway through the sequence, and fallback when a category has no remaining originals. The implementation does not assert all very large books are already indexed. Long documents can require multiple sequential local model calls, so actual elapsed time per source will vary.

This is an operational quality change rather than a relevance-ranking change. It keeps the already benchmarked native batched semantic search and exact query audit unchanged.
