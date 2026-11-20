# Native source-linked retrieval ranking regression, November 20, 2026

The owner's canonical FileDB and source-map remain unchanged. This is a
**retrieval evaluation**, NOT a replacement index, automated semantic
truth test or synthetic claim of progress from a passed unit test.

The original November 7 `hard-retrieval-benchmark-20261107` created a
private, predeclared population of 36 queries before observing retrieval
scores, with 28 hand-selected known original source paths and eight
examples that originally had no matching indexed source. The remaining
eight are not known-unanswerable questions; they are only **historically
unlabeled** and could become answerable after library growth.

On November 20, the exact same 36 questions were replayed using the
**live, unchanged native batched summary search** over roughly 8,300
summarized originals. All command executions returned cleanly. Exact-
original top-one retrieval fell from 18 of 28 in the earlier small
corpus to 4 of 28; top-three fell from 25 to 8; top-five to 14;
top-ten fell from 28 to 17, with MRR@10 0.264. These are the old gold
queries on a **different, larger corpus**, not a randomized blinded
benchmark or proof the retrieval algorithm itself recently changed.
The expected sources all retained original FileDB entries and at least
one summary collection, so this is ranking dilution/quality deterioration,
not an accidental deletion of the previous sources.

An isolated existing `--with-facets` search experiment covered only the
first five originally strong queries before it was stopped for poor
observed ranking and substantially increased CPU time; it found no
improvement on those five cases. That is **not** a statistically complete
facet-mode benchmark. A separate read-only highest-level-root-vector
comparison scanned 16,538 SHA-valid current summary vectors, filtered
original and derivative source-map revisions and retained one highest-
level summary vector for each of about 8,300 originals. Its top-one was
5 of 28 and top-ten was 16 of 28. Root-only ranking also did **not**
restore the historical result. This rules out indiscriminately
replacing all-level search with root-only search on this small fixture.

No scoring weight, model, native vector, source title, keyword or publisher
content was altered to fit these examples. Results are logged with per-
query ranks and old/new top-three paths. Retrieval similarity is neither
calibrated relevance nor proof that an answer exists. A broad new index
will naturally introduce many competing legitimate sources, so a much
larger, multi-answer independently judged ranking benchmark is needed
before any production reranker or hybrid search deployment.

## Repeatable read-only guard

`src/file_system_catalog/retrieval_regression.py` and
`scripts/raspi_retrieval_regression.py` provide one canonical KMD guard:

- The expected original paths and questions must come from a separately
  versioned external, bounded predeclared gold JSON (private, not Git).
- The tested original KMD search must report its original/derivative
  publisher-SHA validation contract, and results must be unique original
  paths. Each per-query current source is independently looked up in the
  *existing* canonical original FileDB and read-only source-map.
- The guard now validates both the original SHA in the actual native
  `content_chunks` and the source-map derivative SHA in the native
  `content_representations.generation_json`, rather than mistaking a
  matching original SHA for complete source-revision correctness. This
  checks metadata/version consistency; it does not rehash every original
  publisher file. The normal search performs its separate revision and
  source-passage checks before returning candidates.
- An unmatched gold is counted as a ranking miss, not automatically as
  missing source content. A historical no-gold question is marked
  `no_gold_predeclared_not_unanswerable`, never a true negative.
- Full-population historical totals are compared **only** if every gold
  query is replayed. An explicit `--case-id` subset is a targeted smoke
  test and has no full-population comparison claim.
- The script writes one private reproducibility JSON, refuses overwrite,
  uses SQLite read-only/PRAGMA query_only, and does not alter any native
  source rows, embeddings, live worker or the original corpus. A query
  error fails the run without issuing an incomplete successful report.

The November 20 private benchmark receipts, SHA values, root-vector
comparison, limited facet pilot and real smoke-test logs are under
`/data/var/knowmoredirt/knowledge-library/search-audit/retrieval-drift-20261120/`.

For a bounded independent smoke test, the operator can run:

```sh
PYTHONPATH=/data/src/worktrees/knowmoredirt-retrieval-regression-20261120/src \
 /data/var/knowmoredirt/semantic-venv/bin/python \
 /data/src/worktrees/knowmoredirt-retrieval-regression-20261120/scripts/raspi_retrieval_regression.py \
 --gold /data/var/knowmoredirt/knowledge-library/search-audit/hard-retrieval-benchmark-20261107/predeclared-gold.json \
 --database /data/var/knowmoredirt/knowledge-library/filesystem.sqlite3 \
 --source-map /data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3 \
 --search-cli /data/src/worktrees/knowmoredirt-hierarchical-summary-20261107/scripts/raspi_hierarchical_library.py \
 --case-id normalmap_indirect --case-id pit_spec \
 --output /data/var/knowmoredirt/knowledge-library/search-audit/retrieval-drift-20261120/NEW-private-smoke.json
```

The new output destination must not already exist. Source-content
ranking needs multi-source gold, larger top-k retrieval, hard irrelevant
counterexamples, query-to-citation provenance and user relevance
judgments. A reranking/hybrid candidate should be evaluated strictly
without using these old 28 examples to fit parameters.
