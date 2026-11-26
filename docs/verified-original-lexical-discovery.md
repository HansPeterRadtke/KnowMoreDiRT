# Source-verified, read-only lexical discovery over current native summaries

This is an **optional complementary search** for the existing offline
Knowledge Library. It does not replace the native semantic search, original
folder/grep access or verified source-passage inspection.

An original may be genuinely relevant to a plain-language question but absent
from the semantic reader's first five matches. This tool searches the titles
and generated summary roots already in the canonical KnowMoreDiRT FileDB,
using a temporary in-memory SQLite FTS5 index. For **each returned candidate**,
it then independently checks its approved original-file bytes, the derived
text bytes and its complete native summary/embedding tree. A generated summary
is a *discovery hint*, never evidence that the scientific claim is true.

The tool never writes to the original files, approved acquisition ledger,
source map, canonical native database, existing summaries or embedding vectors.
It does not download anything, use a language model, create a persistent
second index or change the running indexing workers.

## One-shot use on Raspi

Run in /data/src/github/KnowMoreDiRT with a fresh private output filename:

~~~bash
umask 077
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. nice -n 10 python3 \
  scripts/raspi_source_verified_summary_discovery.py \
  'How can an agent recover from malicious instructions in a tool result?' \
  --top 5 \
  --private-output \
  /data/var/knowmoredirt/knowledge-library/search-audit/UNIQUE-DISCOVERY.json
~~~

A requested publisher edition is a **hard source boundary**. For example,
add the following argument when only the archived Blender four-point-five
manual is permitted:

~~~bash
--publisher-url-prefix 'https://docs.blender.org/manual/en/4.5/'
~~~

This is an exact HTTPS URL **directory** prefix, including the final slash.
It is compared to the canonical source-map publisher URL before generated
summary ranking, and checked again after each returned original has been
physically verified. A wrong or unavailable edition returns **zero** matches
instead of silently substituting a paper from another release or website.
The filter reuses the repository's established exact publisher-archive URL
validation rules, including rejection of encoded path escapes and URL ports.
For **actual passages from archived HTML manuals**, use the existing
source-verified article search instead:
`raspi_publisher_html_article_discovery.py`, documented in
`docs/publisher_html_article_discovery.md`. It indexes the full verified
original article, whereas this summary-only tool returns document-discovery
candidates. The scope does not navigate to the URL or download anything. It is not a
semantic filter on dates or versions inferred from the query alone: callers
must explicitly supply the needed publisher release. A source absent from
the archive is not evidence that it does not exist publicly.

Unicode-only tokenization can be selected with --unicode61; the default is
Porter stemming. Query text is passed as data to a bounded, escaped FTS
expression, not as raw SQL. At most five independently verified original
candidates are returned. New reports are private and never overwrite an
existing receipt.

The output reports the original path, full publisher and extracted-text SHA,
native summary metadata SHA, the source-tree verification count and the
lexical rank among currently completed native summary roots. It labels
scores as sorting values, **not confidence**. It never presents generated
summary prose as source-backed quotations or automatic factual answers.
To cite a claim, open the actual original or run an independently verified
source passage search with relevant governing context.

Only originals with **currently completed recursive summary roots** enter
the ephemeral ranking. With a publisher-directory restriction, the initial
in-memory index contains only matching stored publisher URLs; the returned
originals are checked against that same restriction again. An unscoped query
may otherwise surface *different archived versions* of an apparently identical
manual page, so do not use an unscoped result as an answer to a version-specific
question. Neither text similarity nor successful original-byte validation
alone establishes that the **right version** was selected. In-progress or unindexed originals may still contain
a good answer and will not appear here. The rest of the library remains
accessible through ordinary folder/grep and other permitted search methods.
The source/derived/native metadata of all candidate roots is revision
cross-checked; complete physical source+derived SHA and source/embedding
verification is performed **only for returned candidates**, not every
document scanned for words. A candidate that fails those checks is dropped.
Neither a missing result nor a low rank means that a source lacks the
information.

This check is point-in-time. It does not pin an immutable cross-database
generation against another writer acting after verification. As with any
search over summaries, a relevant-looking result can be misleading, so read
and evaluate the actual original before relying on a claim.

## Evidence and limits

On November twenty-sixth, a three-question set was frozen before testing,
covering RecoverAgent, T1-Bench and multi-turn table-reasoning research.
Among those three **known supporting originals**, the installed native
semantic search returned one within five. A separate in-memory experiment
over the approximately ten thousand currently completed summary roots
placed all three known originals first using either Unicode or Porter
matching. The guarded, source-verified tool was then run on those same
frozen questions and also returned the three known originals first,
with the selected original bytes and native summary trees checked.

This is **not a general accuracy comparison**: three known positives are
not an unbiased sample of user searches, the lexical candidate population
excludes originals with incomplete summaries, and other retrieved papers
were not exhaustively relevance-labeled. There are no quality or latency
superiority claims. Before any automated/default ranking replacement,
evaluate broad independent queries and negative controls across all
document categories, source-version changes, latency and memory use.

The frozen private comparisons, source-hash checks and one-shot retrieval
receipts are kept under the existing restricted Raspi
/data/var/knowmoredirt/knowledge-library/search-audit tree. Do not commit
private original papers, owner transcripts or those receipts.

## Regression tests

~~~bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/filesystem_database:. \
  python3 -m unittest -q \
  tests.filesystem_database.test_summary_lexical_discovery \
  tests.filesystem_database.test_recent_source_tree_audit
~~~

Tests include malicious query syntax, both tokenizers, bounded input, the
current native summary revision, a changed original, stale source-map
metadata, source-identity verification and clear separation of discovered
original candidates from original-passage factual proof.
