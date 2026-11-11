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

## Physical source verification in ranked search, November 11

Summary search now checks both physical files behind each candidate. After the native row matches a unique manifest entry, a unique completed download receipt and the extracted source map, the resolver streams the publisher file and the derivative through SHA-256. A changed, missing or unsafe file disqualifies that revision in either search mode. Ranking continues until it fills the requested number of places or runs out of verified candidates. The query reuses the outcome across nodes of the same revision and checks the files again on the next search. Audit records include the verification policy, original and derivative digests, and the stored passage digest beside its character offsets. Claims still require inspection of the referenced passage.

Previously, search hashed the extracted text and compared the publisher digest stored in the index with the download receipt. A controlled local copy of the winning source retained its byte count after one byte changed. The previous search still ranked it first. With the fix, search rejected that copy and filled all three requested places with the next three verified sources in order. The actual publisher originals and the canonical index remained intact throughout the comparison.

A read-only Raspi audit independently hashed every original and derivative referenced by the fixed search cohort. All 4,045 originals, 5,263 distinct source spans and 139,966 native representations passed. Checks included passage offsets and digests, represented text, embedding bytes, vector dimensions and norms. The original files totaled 5,755,045,203 bytes. These counts describe the indexed cohort at the recorded row boundary.

The comparison ran 39 fixed queries in both default and facet-assisted search against the same cohort. All 624 result positions retained their document, matched representation IDs, digests and reported scores. The expected source appeared in the first eight for 34 questions in default mode and 32 in facet mode. Median observed query times went from 1.44 to 1.51 seconds in default mode and from 5.10 to 5.73 seconds in facet mode. The two passes ran sequentially on the active host, so these timings include variations in load. The full protocol, controlled-file checks and per-source receipts are private on Raspi under search-audit/publisher-search-verification-20261111.

Three additional interleaved default-mode checks exercised larger graphics publications. The conservation imaging and hair reconstruction papers ranked first, and the rough diffuse shading slides ranked fourth. Both implementations returned the same top eight on every run. Across the eight returned documents, the first revised runs spent 0.37 to 0.76 seconds hashing 121 to 262 MB of publisher files.
