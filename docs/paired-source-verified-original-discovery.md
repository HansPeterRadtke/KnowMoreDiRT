# Paired original-source discovery without unverified score fusion

This is an **optional, read-only** KnowMoreDiRT search. It gives a worker two
different ways to discover an already archived original in one operation,
without hiding either channel's individual ranks or pretending their scores
have the same meaning.

The first channel is the existing source-verified native passage search. It
uses the installed local E5 query encoder and returns bounded exact native
passages, with verified publisher-original and extracted-text revisions. The
second channel is the separate **current native-summary lexical** search. It
finds possible original documents by words in generated summary roots, then
rehashes publisher and derivative bytes and independently verifies the native
recursive summary graph for returned documents.

The union retains each native result in its unchanged native order, followed
by any additional lexical candidates, capped at ten distinct originals.
Repeated originals are recorded with **both original channel ranks** and must
have matching original source revisions. This is **not** an interleaved
top-ten ranking, a new ensemble confidence score, a new permanent index, a
model-generated answer or a claim that each retrieved document is relevant.

After both searches finish, the original and extracted source files of
**every distinct returned document** are fully SHA-rehashed again against the
approved acquisition ledger, current source map and native FileDB identity.
The source-map revision and approved acquisition receipt are checked again
after the complete physical file hashing, so a concurrent revision change
during that final proof is rejected instead of accepted. This still is not
an atomic transaction across independently updated stores.
The native scanner intentionally leaves its file-content SHA blank on some
large originals. The paired checker accepts that documented skipped-large
case **only after independently verifying the complete original bytes**.
Any contradictory stored SHA, missing original, changed derivative, receipt
or native ID aborts the response. Publisher-original hashing is streamed
with a maximum of one hundred twenty-eight mebibytes per file; extracted
source text has a separate, stricter sixty-four-mebibyte bound. The combined
physical source-read budget remains two hundred fifty-six mebibytes across
all returned files. These independent bounds are necessary because actual
accepted publisher PDFs can exceed sixty-four mebibytes even though their
text derivatives are much smaller. A search exceeding those limits fails
closed; use the existing narrower original-search commands instead. Only a
known read-only SQLite busy/locked error in the native query receives a
single bounded retry. Other errors are never retried automatically, and
the active indexing workers remain untouched.

## Run a single query on Raspi

From /data/src/github/KnowMoreDiRT:

~~~bash
umask 077
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/data/var/knowmoredirt/semantic-venv/bin/python \
  scripts/raspi_paired_original_discovery.py \
  'How can an agent recover after a malicious tool response?' \
  --top 5 \
  --private-output \
  /data/var/knowmoredirt/knowledge-library/search-audit/NEW-PAIRED-QUERY.json
~~~

This uses the existing local CPU embedding runtime, not an external model
API. The script does not download documents, change summaries, modify the
canonical source map/FileDB, change embeddings, start a persistent search
server, restart the two live indexers or alter global search rankings.
A new, owner-only JSON receipt stores the two complete independently verified
search results, original hashes, distinct document IDs, separate ranks and
the final physical-source recheck. The console shows only high-level counts
and the receipt path. Existing receipts are never overwritten.

The output includes original passage evidence **only on the native channel**.
A lexical-only candidate is a verified original *identity*, not a verified
passage establishing a particular claim. Use the existing original-file
inspection or the verified source-evidence operator to inspect the actual
topic and qualifications before writing a factual answer. Neither semantic
nor lexical similarity proves the requested statement.

This paired command is intentionally **unscoped**. A question naming an
archived Blender release does not safely restrict the native channel by
itself. For an exact publisher-version restriction use the existing
source-verified archived HTML article search, or the standalone
summary-lexical discovery with its explicit publisher URL directory argument.
Do not claim that an unscoped paired result establishes the requested edition.

## Current evidence and limitations

On November twenty-sixth an **independent thirteen-question, source-verified
known-positive set** created November twentieth was replayed against both
current search readers. Both the native source-verifying search and the
Porter summary lexical search contained the known source among their first
five for **nine of thirteen** questions. Their combined sets contained
the known source for **eleven of thirteen**. This demonstrates useful
complementarity, not superior relevance ranking, universal completeness,
or an unbiased factual-answer-quality estimate. Other retrieved originals
were not exhaustively judged irrelevant, and a larger union shows more
candidates at the cost of inspection and additional physical hashing.

The full paired end-to-end replay separately verified **all thirteen**
frozen queries against the real current source archives. It returned the
previously known original in eleven of thirteen cases, with every one of
the one hundred twenty-one distinct returned original-document instances
independently rehashed after both channels. The first run recorded three
real failures: a transient read-only SQLite writer lock and two publisher
PDFs over the former sixty-four-mebibyte original-file limit. The bounded
SQLite read retry and independent streamed original limit fix were tested,
and exactly those three previously failed queries were rerun without erasing
the first failure receipts. The repaired results have their own private
frozen-query and publisher-revision-verified evidence report.

A further **older** multi-relevant-original set, frozen before this paired
tool existed, has five questions and eleven known supporting original
labels. The native-only channel returned seven of those eleven labels,
the separate summary-lexical channel eight, and the union eight. All thirty-
six returned original-document instances were independently byte-checked.
This demonstrates no universal gain from combining the channels and
shows that both can miss relevant material. In particular, a materials
roughness question missed both previously verified originals. This is a
specific remaining discovery-quality gap, not proof that the papers were
absent or irrelevant. It is important to preserve the distinction between
**candidate recall among predeclared originals** and genuine verified
answers, and between the union of two top-five lists and one fused top-five
ranking.

These results and the source checks are retained privately under Raspi's
restricted search-audit tree, not committed with the project source.

No immutable, atomic source-generation snapshot is pinned across the
separate canonical SQLite stores. Verification is fresh at response time,
but later writer changes remain possible. This is a point-in-time
original-source discovery tool and is **not approved** to replace the
installed default search path without a broader held-out relevance,
incorrect-source and edition-scope evaluation.

## Reproducible tests

~~~bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/filesystem_database:. \
python3 -m unittest -q \
  tests.filesystem_database.test_paired_original_discovery \
  tests.filesystem_database.test_summary_lexical_discovery \
  tests.filesystem_database.test_recent_source_tree_audit
~~~

The fixtures cover two-channel rank preservation, complementary source
identities, duplicate/contradictory SHA rejection, real original and
derivative byte changes, source-map revision changes during the final hash,
ledger changes, native FileDB identity mismatch,
intentionally skipped native hashes for independently rehashed large
files, distinct original streaming/derivative limits, one known SQLite busy
retry, no retry for unrelated database errors, and unsafe path rejection. The actual corpus replay is separate from
the synthetic tests and must not be described as running merely because
they pass.
