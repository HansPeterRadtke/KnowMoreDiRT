# Native KnowMoreDiRT summary-search scalability — 7 November 2026

This experiment improves source-linked search without interrupting the running Raspberry Pi summarization worker or creating another database.

## Bottleneck and replacement

The previous search loaded every generated summary and its provenance JSON for every query, then calculated cosine distance one vector at a time. This creates unnecessary work as the original KnowMoreDiRT FileDB grows.

The new batched implementation reads source IDs, vector identities and embedding bytes first. It validates each vector's SHA-256 and dimension and calculates cosine values in temporary batches of 1,024 vectors. It retrieves and verifies complete summary text and provenance only for the highest-ranked original documents. The old scalar implementation remains available for regression tests.

The new implementation preserves source revision checks, all matching hierarchy levels and the original file reference. It does not add a persistent cache, an additional metadata database or new SQL schema objects. It does not implement factual-answer verification or abstention.

## Measured benchmarks

Disposable native-schema SQLite indexes containing 10,000 synthetic vectors yielded the same top-eight originals: original scalar search 0.440 seconds, batched search 0.219 seconds.

At 30,000 synthetic vectors, the original scalar search took 1.375 seconds and the batched search 0.712 seconds, again with the same ordered top eight originals. The largest similarity-score difference was below one millionth.

Twenty-seven actual logged queries were then checked against the live growing FileDB. All 27 returned identical ordered original documents and matching summary-node counts; maximum absolute cosine difference was 0.00000102. Aggregate search scan time was 6.147 seconds versus 5.665 seconds. This does not include cold E5 model startup and does not guarantee future latency.

A separate direct CLI query retrieved the Unity Visual Effect Graph Property Binder original first and wrote a query audit recording native_batched_cosine_v1 as the retrieval engine.

## Safety and limitations

Native KMD schema tests reject corrupt vector checksums, wrong vector sizes, invalid summary-text digests, unsupported model versions, nonfinite input, and rejected source verification callbacks. They compare complete scalar and batched results and verify search causes no persistent SQLite writes.

The output remains nearest original-document candidates rather than verified answers. Unsupported questions can still return unrelated content. No lexical re-ranking was deployed because earlier independent holdout tests showed it could reduce first-place accuracy.

All real-query benchmarks and logs remain private on Raspi under the KnowMoreDiRT search-audit data directory. Only implementation, synthetic tests, and this report belong in Git.
