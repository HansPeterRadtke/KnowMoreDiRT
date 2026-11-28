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

## Small, literal-folder first-unit search when global vectors miss the source

The existing `scripts/raspi_publisher_firstunit_discovery.py` has an
**opt-in** `--path-prefix` for the canonical original-source directory
(for example `graphics/photorealism-guides-2026/blender45-original-material-render-nodes/`).
It must end with a slash. This is an **exact string prefix**, not a
search over untrusted URLs, not a glob, and not SQLite `LIKE`: folder
names containing `%` or `_` are literal. Unsafe or empty folders fail
closed rather than report that the requested research topic has no answer.

Run this directly on Raspi for one question, or supply a batch file
containing independently selected `{"queries":[{"id":"...","query":"..."}]}`
records to reuse one small, temporary in-memory first-unit index:

~~~bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /data/var/knowmoredirt/semantic-venv/bin/python   scripts/raspi_publisher_firstunit_discovery.py   --query 'An RGB image makes the mesh seem bumpy without changing its geometry'   --path-prefix 'graphics/photorealism-guides-2026/blender45-original-material-render-nodes/'   --top 10
~~~

The optional folder filter is evaluated **against current source-map
original paths before any derived file reading or FTS insertion**. The
existing literal and Porter stemmed channels remain independent; only
current retrieved candidates have their complete publisher original,
complete derivative, approved download receipt and unique native FileDB
identity checked. Original files, summaries, embeddings, global rankings
and indexing workers are never changed. The output states which folder
was inspected, how many extracted originals were eligible, how many first
units were usable, and whether every returned source stayed within it.
This is not a complete full-document or multi-folder search; facts that
occur after the first indexed unit, in figures, or outside the chosen
folder may be missed. The matched source excerpt is current derived text,
not automatically an independently verified physical PDF page.

A November twenty-eighth **frozen source-identity** check covered five
predeclared Blender material questions within twenty-seven available
originals. The Porter channel found all five intended originals within
ten results (four ranked first, one eighth); the separate literal
channel found four of five within ten. A different, previously frozen
ten-question Unity group searched 575 original documentation files:
the union of the two untouched lexical lists found four known sources
within ten, and missed six. No query terms, source labels, weights, or
channel scores were tuned to these questions. The older global native
summary search had found only one of the first four Blender originals
within ten; this is a **limited scope-specific discovery gain**, not a
general accuracy claim or a complete holdout. Retrieved documents still
require actual original passage inspection before any factual answer.

## Optional third channel: the original publisher first source unit

The earlier two-channel command remains the default. When a query is phrased
using terms **present in a paper's opening** but omitted from its generated
summary, add `--include-source-firstunit` to that same command. It reuses the
existing `FirstUnitRAMIndex` and approved source map to scan a bounded prefix
of **every completed original**, including originals with no generated
summary. It then physically hashes each chosen publisher original and its
full extracted derivative, checks the acquisition receipt and unique native
FileDB document identity, and preserves the independent Porter rank.

Example (single question, with an original-source excerpt included separately):

~~~bash
umask 077
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/data/var/knowmoredirt/semantic-venv/bin/python \
  scripts/raspi_paired_original_discovery.py \
  'How do album-understanding benchmarks use trips and wedding photos?' \
  --top 3 --include-source-firstunit \
  --private-output \
  /data/var/knowmoredirt/knowledge-library/search-audit/NEW-THREE-CHANNEL.json
~~~

Returned first-unit candidates also contain a `literal_query_coverage_review_only`
measurement. It displays exact question-word matches in the attested
first-source-unit excerpt separately from matches in catalog title metadata.
This is useful when a nonsense or weakly relevant OR query returns
candidates; it does not filter them or classify whether a document truly
answers the question. Porter stemming and synonyms can retrieve real sources
with few exact literal words, so absence is not proof of irrelevance.

The separate `raspi_publisher_firstunit_discovery.py` command additionally
offers `--require-source-phrase 'AgentMemRec'` as an *explicit* precision mode.
A batch can use each question's optional `require_source_phrase` key to
check several independent names during one RAM-only corpus scan. This is a
bounded, source-byte-SHA-verified first-unit phrase gate, **not** a
document-wide proof of absence, a relevance classification, or a replacement
for either normal search channel. It never silently changes the ranker, and
a phrase only in the catalog title or on an unsearched later page is
intentionally insufficient. Under concurrent index writes, a persistent
SQLite lock aborts the whole bounded phrase query rather than mislabeling
the original as absent. With this phrase requirement, the existing
`--verify-pdf-previews 1` option can also confirm the **exact matched phrase
and its source context** against the publisher PDF's original physical text
page. The old no-required-phrase mode remains unchanged, and no method
silently claims semantic truth or a verified original image. See
`docs/filesystem_database.md` for usage, limits and source-span evidence.

This option is **significantly slower at cold start**. On the current Raspi
archive it sampled approximately 274 megabytes of original-derived first-unit
text across 17,715 originals before ranking. Its selected excerpts are
SHA-linked to the **current extracted source text** and marked separately
from verified original publisher PDF bytes; physical PDF page verification
requires an additional explicit original-page tool. A generated summary
remains only an optional search hint; a candidate is never an answer to the
research question by itself.

The result keeps the native-passage, summary-root and publisher-first-unit
ranks **independent**, without reranking or inflating a confidence estimate.
At most five sources per channel and fifteen unique originals can be
returned. Every distinct returned original and derivative is physically
rehashed again, and an overlapping original must have identical source
and derivative revisions and native identity. A different revision, missing
approved receipt or document outside the original byte budget causes refusal.
Large PDFs are SHA-streamed as documented above. For multiple questions,
use the existing `raspi_publisher_firstunit_discovery.py --queries-file`
interface instead to amortize its initial first-unit index build; this
three-channel command currently runs a fresh source scan for each query.

A November twenty-sixth original-source comparison used thirteen already
frozen known-positive research questions. The existing two-channel candidate
union contained eleven known original documents among its two top-five
lists. Independently running the existing Porter publisher-firstunit search
found eleven of thirteen, including **both** originals the two-channel union
missed. Counting the *union across those separately executed read-only
searches* therefore covered thirteen of thirteen known supporting documents.
On a separate five-question, eleven-label multi-relevant set, the existing
two-channel union identified eight labels, while publisher-firstunit search
found ten. These are **not** same-snapshot three-channel query tests and
should not be described as a measured unified ranker improvement or
exhaustive relevance evaluation. Two actual live three-channel calls also
verified the previously missing album and overlapping-speech research
originals, including exact original/source SHA, native identity, and final
excerpt offsets. Their original physical PDF first-page passages were
independently checked in separate read-only receipts.

The real single-query cold start was considerable: the archive prefix index
sampled roughly two hundred seventy-four megabytes for 17,715 sources, and
the two actual live three-channel queries required approximately sixty-six
and eighty-three seconds. Do not use it by default for time-critical
questions. Batch original-first-unit questions with its existing batch
interface, or use direct native search or source-name lookup when enough.
Two deliberately unanswerable-looking queries nevertheless produced other
OR-matched candidates; a populated result alone is not an answer.

If a physical PDF page is needed, the **same third-channel query** accepts
`--verify-firstunit-pdf-pages 1` (up to three). This asks the existing
publisher-PDF verifier to independently read the ranked PDF originals and
compare the selected page excerpt to the physical page. It reports the number
actually confirmed and leaves other pages explicitly unverified. A correct
physical page citation is still **not proof** that the paper entails the
requested factual proposition. This check is optional and adds work; when
omitted, the first-source-unit page label remains derived-text-only.

The original first-unit text can miss important facts on later pages, tables,
figures and images. It is a **bounded original-source discovery complement**,
not a full-document search or proof that the source entails a suggested answer.
Nonsensical queries containing common words can still return other documents
because independent lexical candidate discovery uses an OR search; an empty
or populated candidate set is **not** a verified unanswerability judgment.

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
