# Search archived publisher HTML within an explicit version directory

The optional Raspi command searches the article text of already approved
archived HTML. Give the publisher's exact directory URL to restrict the source
population, for example one archived Blender manual version. It does not
infer the installed version, choose "latest", fetch missing pages, or fall
back to another version. Native semantic retrieval and the existing first-unit
discovery command keep their current behavior.

The output gives original paths, publisher URLs, native FileDB identities,
current original and derivative checksums, and reproducible excerpts. A
retrieved passage is not a verified answer or proof of applicability.

## Run on Raspi

    /data/var/knowmoredirt/semantic-venv/bin/python \
      /data/src/github/KnowMoreDiRT/scripts/raspi_publisher_html_article_discovery.py \
      --publisher-url-prefix 'https://docs.blender.org/manual/en/4.5/' \
      --query 'How does roughness affect reflected light?' --top 5

The scope must be an ASCII HTTPS URL with a plain hostname and a directory
path ending in a slash. Hostname case is normalized; paths remain
case-sensitive. Credentials, ports, queries, fragments, whitespace, control
characters, percent escapes, repeated path slashes and dot segments are
unsupported and refused. The directory boundary prevents a request for
4.5/ from matching 4.50/. This conservative identity contract is not a
general-purpose URL normalizer.

An unavailable archived scope returns empty ranked lists and
source_scan.scope_status=no_archived_sources_in_requested_scope. This says
nothing about whether the requested publication exists elsewhere. A scope
containing only non-HTML or unusable articles reports that separate condition.

To amortize the bounded archive scan, use --queries-file PRIVATE_JSON:

    {
      "queries": [
        {"id": "roughness", "query": "How does roughness affect reflected light?"},
        {"id": "lighting", "query": "Which controls change emitted light energy?"}
      ]
    }

One batch uses one explicit scope. It accepts one through thirty-two distinct
question IDs and questions of three through six hundred characters; at least
one informative query term is required. --top is one through ten, default
five, independently for the literal and Porter stemming channels. Porter
stemming can match inflected word forms. Neither channel's BM25 score is a
probability, and their ranks are not fused.

Use --output NEW_PRIVATE_JSON to save a result with owner-only permissions.
Existing files are never overwritten. Output under original or derivative
trees is refused, including resolved symlink aliases. Output inside the native
database working tree is permitted only under its existing search-audit
directory; direct database/source-map directories remain protected. No result
file is written until all selected-source validation has passed.

## Population and article selection

The canonical catalog, completed acquisition ledger, current source map and
native FileDB are read only. Every selected HTML source must have mutually
consistent catalog and source-map publisher URLs, a complete acquisition,
matching recorded byte count, current original and derivative digests, and one
native file identity.

The temporary SQLite FTS5 index exists only in memory. It indexes the full
selected normalized article, not the first derivative prefix. The parser
requires one explicitly closed article container, or one closed main if
there is no article. Multiple, incomplete, explicitly hidden or excessively
large containers are excluded with a reason. It removes navigation, aside,
footer, script, style, template, SVG, canvas and button content, plus explicit
hidden/inert/ARIA-hidden or navigation-role content. It does not interpret
external CSS, execute JavaScript, render figures, or prove browser visibility.

Bounds are fixed and reported: at most twenty-five thousand metadata rows,
sixty-four MiB per metadata file, five hundred twelve scoped HTML records,
eight MiB per original HTML, one hundred twenty-eight MiB of reserved HTML
read budget per scope, 2,097,152 article characters per document, sixteen
MiB per selected derivative, and one hundred twenty-eight distinct selected
originals per batch. Choose a narrower directory or split the question batch
when a global bound is reached; there is no silent partial global scan.
Individual invalid, oversized or ambiguous sources are skipped explicitly.
Read source_scan before interpreting results as coverage.

original_html_byte_budget_used includes reserved reads that subsequently
fail. It is not a claim that every reserved byte was successfully read.
Non-HTML exclusion counts are separate from HTML failures. The scope inventory
is observed during the scan, not frozen against concurrent acquisitions.

## Reproduce a passage and interpret its limits

Each result contains source_excerpt, its SHA-256 and half-open
normalized_article_start_char / normalized_article_end_char coordinates.
The coordinates refer to whitespace-normalized selected original HTML text.
They are not offsets in raw HTML, derivative text, a PDF or a browser render.

The following Python sketch reproduces one saved result from its original:

    import hashlib
    from file_system_catalog.publisher_html_article_discovery import extract_article

    raw = (original_root / hit["original_path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == hit["publisher_original_sha256"]
    selector, body = extract_article(raw)
    assert selector == hit["article_selector"]
    assert hashlib.sha256(body.encode()).hexdigest() == hit["normalized_article_sha256"]
    excerpt = body[hit["normalized_article_start_char"]:hit["normalized_article_end_char"]]
    assert excerpt == hit["source_excerpt"]
    assert hashlib.sha256(excerpt.encode()).hexdigest() == hit["source_excerpt_sha256"]

The excerpt follows actual FTS match positions, including stemmed word forms,
rather than returning the opening when only an inflection matches later in
the article. It remains bounded to five hundred twenty characters. It can
omit a condition, select an adjacent discussion, or be insufficient to answer
the question. Read the governing section and original before asserting a
technical conclusion. matched_article_forms reveals matches actually inside
the excerpt; a title-only candidate can have no article matches.

After ranking the entire batch, the command reopens the catalog, ledger,
source map and native database and rehashes every selected original and
derivative. It then performs another fresh metadata/native/file-identity
check. A source URL, revision, title, receipt, native identity or observed
file change causes refusal instead of a successful stale result. This is
fresh revalidation, not atomic generation pinning across independent stores.
A later uncoordinated change remains possible. Existing generation-migration
gates and live workers are unaffected.

All retrieved text remains untrusted source data. Original rights and
acquisition permissions still govern sharing or subsequent use; a successful
search adds no redistribution, training or execution authority.

## Verification

Focused tests use authored synthetic originals and independent mutation
controls, without copying publisher content into Git:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/filesystem_database \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m unittest -v \
      test_publisher_html_article_discovery \
      test_publisher_html_article_discovery_cli

The November 26 private evaluation froze twelve known-source manual questions
and an unavailable-scope control before examining rankings. Baseline and article
comparisons used the same approved populations: twenty-seven HTML originals for
the archived four-point-five manual, and one hundred fifty-one for five-point-two,
with two ZIP archives explicitly excluded.

For the five-point-two questions, known-target top-five retrieval improved from
two of six to six of six for literal search and from three of six to six of six
for Porter. Each new channel placed three known targets first and three second.
Four-point-five retained all six known targets first. Independent verification
reproduced all one hundred twenty displayed excerpts from forty-eight distinct
original sources, checking both full file digests, metadata, native identities
and normalized character positions.

This is a small purposive source-finding diagnostic, not a general accuracy
estimate. Several snippets still omit necessary surrounding explanation, even
when the right document ranks first. Private original-linked questions, results
and independent semantic/provenance receipts remain under the existing Raspi
search-audit project evidence tree. The new command does not change native
semantic ranks, stored vectors, derivatives, acquisitions or running workers.
