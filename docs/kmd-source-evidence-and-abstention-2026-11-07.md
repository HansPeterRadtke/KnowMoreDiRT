# Source-linked evidence inspection and abstention research — 7 November 2026

This bounded KnowMoreDiRT research increment continues the original Knowledge Library work on Raspi. It does not change production semantic ranking, source ingestion, recursive summary generation, or the authoritative original KMD FileDB.

## Research basis

The offline Knowledge Library already contains the complete original 2026 ICML paper *Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation* and the original 2026 EMNLP paper *Uncertainty-Aware Hybrid Retrieval for Long-Document RAG*. Both distinguish semantic similarity from evidence appropriate to the reader. The first considers whether the retrieved evidence is sufficient for reasoning; the second emphasizes query-dependent uncertainty and the tradeoff between full-document context and small retrievable passages.

The original 2026 paper *Relevance Supervision Can Teach Neural Retrievers to Prefer LLM-Generated Text* warns that retrievers may develop preferences for fluently generated material over independently written originals. *Don't Be Misled by Style*, ACL 2026, similarly shows that style can bias ranking. These are reasons not to replace the source text with the generated summaries or treat fluent summary language as a citation.

Recent independent work on evidence sufficiency and answer abstention (Zhang and Wu, Computers, Materials & Continua 2026, DOI 10.32604/cmc.2026.086343) also reports poor model abstention under conflicting evidence. A confidence declaration by an LLM alone would not validate a source claim.

Research sources and full publisher articles:
- https://proceedings.mlr.press/v306/sun26p.html
- https://aclanthology.org/2026.emnlp-main.2104/
- https://aclanthology.org/2026.findings-emnlp.2038/
- https://aclanthology.org/2026.acl-long.302/
- https://doi.org/10.32604/cmc.2026.086343

## Empirical unsupported-result experiment (private source corpus)

A read-only experiment ran 104 queries against the original native KMD FileDB with about 499 summarized originals at the checkpoint. The labels were drawn from the existing easy and difficult source-finding benchmark, the previously fixed positive holdout, and an additional small held-out unsupported-query set. The positive label means precisely that the top-ranked original path equals the predeclared gold path; it does NOT mean a factual answer has been independently verified.

In the training split (69 queries), 39 top-ranked results matched the predetermined source; 30 were different or unsupported. The held-out split (35 queries) contained 17 exact expected top-ranked originals and 18 other/unsupported first results. Negatives had cosine scores of approximately 0.782 to 0.877 on the holdout, overlapping the correct-result score range of roughly 0.868 to 0.919.

Using a score cutoff of 0.88 selected 11/17 correct holdout top results but still accepted 5/18 incorrect/unsupported results. Requiring source-summary term overlap of 0.30 selected 14/17 correct top results while still accepting 7/18 incorrect/unsupported. Title-term overlap reduced some false positives but rejected many correct source matches. Top-one/two score margin also failed to distinguish outcomes robustly. Therefore NO cosine, lexical, score-margin or fused acceptance threshold was added to production search.

Private reproducible test program and complete metric receipt:
- search-audit/evidence_gate_experiment_20261107.py
- search-audit/source-sufficiency-diagnostic-20261107.json

These data are not a blind model-calibration benchmark. Source identities and some questions were authored with knowledge of the small existing index, and an apparently wrong first document can still be topically relevant. The outcome is negative evidence for blindly deploying a simple threshold, not a certified estimate of deployment-wide factual accuracy.

## New experimental original-evidence command

The new file_system_catalog.source_evidence module can inspect an **explicit original document** after canonical source selection. The Raspi-only CLI adds:

    knowledge-library-summaries source-evidence --original-path '...' 'graphical interfaces windows icons menus'

The operation calls the existing complete original-publisher binary SHA-256, completed-download ledger, source-map and derived-text SHA verification. It finds the exact original KMD FileDB entry and loads the current complete extracted source. An in-memory, bounded SQLite FTS5 index locates lexical matches across original page/section units. It never persists another database, modifies original text, or replaces the native semantic vectors.

Each returned candidate has the exact original FileDB ID, original and derivative SHA, source unit/page label, original derived-text start/end offsets, source-span SHA-256, selected bounded excerpt offsets and SHA, literal matched query words, and an explicit status identifying it as a **source lexical candidate rather than a verified factual answer**. The query and exact returned source-span metadata are appended to the existing owner-only private query JSONL log. The full source or publisher text is not copied into Git.

Extraction-specific KMD provenance JSON precedes real publisher pages and often repeats the source title and search terms. The evidence locator explicitly excludes that generated metadata prologue from evidence candidates (while retaining the source offsets and checksum). An in-memory test using actual RFC 9866 source confirmed that the first evidence candidate is from the publisher's Original plain text section rather than the metadata prelude.

## Actual full-book verification

The original current HCI textbook extraction contains 2,752,515 characters and 830 source sections. The prototype indexed the complete verified text in RAM in about a second and located GUI pages 504 and 503 first for a question about graphical interface windows, icons, menus and pointers. Page 504/506 material was also returned for visual hierarchy. Every returned source excerpt's offsets and hashes matched the original verified extracted text.

An unrelated question about German building-permit applications still produced lexical page matches for incidental words such as application, building, and legal. This demonstrates directly why full-text results cannot be promoted to a claim that the book supports the user's answer.

Separate real-source tests found matching original sections in RFC 9866 and the 200,796-character PACER causal-discovery paper, correctly retaining the current publisher source and derivative checksums. All lookups were logged in the existing private daily query-audit file.

## Safety and next evaluation

- Keep summary-vector similarity as **document discovery** only.
- Treat source-page FTS as a second retrieval stage that gives the user inspectable original evidence, never as a truth oracle.
- For question answering, verify the actual passage with its preceding/following and governing chapter/discourse context, including negation, historical dates, quotations, and conflicts.
- Benchmark the eventual evidence sufficiency / abstention layer with independently labeled answerable, partially supported, irrelevant, absent, and contradictory queries rather than raw cosine cutoffs.
- Do not enable a new answer-rejection or document-reranking policy until it improves independent held-out relevance without suppressing valid originals.

The existing continuous KnowMoreDiRT summarization worker remains independently enabled and is not restarted for this optional source-inspection feature.

## Physical PDF page citations on Raspi

The installed `knowledge-library-summaries source-evidence` command checks PDF excerpts against physical pages automatically. For example, this query finds passages about query latency in the archived Duhu paper.

```sh
knowledge-library-summaries source-evidence \
  --original-path infrastructure/2026-usenix-osdi26-full-peer-reviewed-papers/osdi26-men.pdf \
  'How does shared disaggregated memory affect query latency?'
```

Use `passages[].publisher_pdf_page.page` for the one-based physical page when its status is `verified`. The existing `source_section` and derivative character offsets describe the indexed extraction. In the Duhu test, the first result carried the old label “PDF page 28”. The complete excerpt occurred on physical page 13 of the 19-page PDF.

The checker hashes the open publisher PDF against the expected revision and gives Poppler that same open file. It first reads each nominated physical page once. If a label fails and the document has up to 32 pages, the checker can correct it when exactly one page in a complete scan contains the entire excerpt. Longer documents get direct checks of the nominated pages. An ambiguous passage, incomplete scan or OCR-only label can still be returned as a retrieval candidate with an explicit unverified reason. The query audit retains the same per-passage result.

To reproduce a citation, run `pdftotext -f N -l N -layout -enc UTF-8 original.pdf -` for the reported page. The SHA-256 of its output bytes must equal `page_text_sha256`. Decode UTF-8, collapse whitespace with `" ".join(text.split())`, then take the half-open character range `[start_char:end_char]`. That slice reproduces the entire excerpt after the same whitespace normalization, preserving case, punctuation and word order. These coordinates refer to extracted page text. The check does not assess whether an excerpt answers a question or establish visual content in a scanned figure.

In the November 12 replay, independent extraction located all 26 sampled excerpts from 13 PDFs. Fourteen historical labels needed correction across seven papers and twelve labels were accurate. A separate panel of four natural questions returned twelve excerpts, nine of which needed new page numbers. Three citations in a longer textbook verified on their nominated pages. An image-only OCR result retained an unverified page status. Repeated phrases and deliberately altered excerpts in two real papers received no physical citation. The private Raspi protocol and per-page checks are under `search-audit/physical-pdf-citations-20261112`.

## RFC numbers and section citations on Raspi

Use an RFC number to find the archived standard. Use a section citation to read the opening of that section directly from the publisher text, even while the standard is waiting for recursive summaries.

```sh
knowledge-library-summaries search 'RFC 9607' --top 3
knowledge-library-summaries source-evidence 'RFC 9110, Section 9.2.2'
knowledge-library-summaries source-evidence 'https://www.rfc-editor.org/rfc/rfc9000.html#appendix-A.1'
knowledge-library-summaries source-evidence --source-name 'RFC 9110' 'HTTP request semantics'
```

A verified native summary of an explicitly requested RFC ranks first. Its cosine score retains its original meaning. The operator checks the official plain-text acquisition URL, the completed download receipt, the number in the publisher header, and the original, derivative and indexed passage hashes. An unarchived RFC returns zero documents. An archived RFC still awaiting a verified summary reports `no_verified_index_match`. Use the section command to inspect it directly.

Exact section evidence identifies a unique body heading and compares its title with the printed contents. Some standards omit deep subsections from their contents. For those, the lookup verifies the chain of enclosing body headings back to an ancestor listed in the contents. A missing or conflicting section returns zero passages and `section_not_verified`. An ordinary topic query within a named standard uses the existing lexical search.

Read `rfc_section_lookup.scope` to distinguish a heading with an opening paragraph from a heading that immediately introduces a child section. The excerpt ends at the configured character limit, and `truncated` records whether more of the opening is available. Read the rest of the section before treating the opening as a complete requirement. To reproduce the citation, use the half-open `publisher_excerpt.start_byte` and `end_byte` range in the original file. The SHA-256 of those bytes must equal `publisher_excerpt.sha256`. Decode UTF-8 and normalize line endings to compare with `source_excerpt`. The query receipt records the same section, extent, and hashes.

The November 12 audit checked 344 archived publisher texts. It matched 12,524 numbered body heading positions, including 816 whose enclosing headings supplied the contents check. A separate sample fixed 694 raw byte positions before implementation. All 694 returned excerpts reproduced the publisher bytes at those positions. An additional deep subsection in each of 72 standards also reproduced the expected raw bytes. One invented section number in each original returned zero passages. In the two native search comparisons, RFC 9607 moved from outside the first three results to first, and RFC 9941 moved from third to first. The private reproduction protocol and receipts are in `search-audit/rfc-section-citations-20261112`.

### Reading past the opening

An RFC section result carries a continuation cursor when more text follows the excerpt. This lets a reader follow a condition, exception, or example across several excerpts. In RFC 9110 section 9.2.2, the opening defines idempotence. The restriction on automatic proxy retries comes at the end of the section.

```sh
citation='RFC 9110, Section 9.2.2'
first=$(knowledge-library-summaries source-evidence "$citation")
cursor=$(printf '%s' "$first" | jq -er '.rfc_section_lookup.context.next_cursor')
knowledge-library-summaries source-evidence "$citation" --continue-section "$cursor"
```

Repeat with each returned `next_cursor` until it is null. The next excerpt starts at the previous excerpt’s exclusive end. Concatenate `source_excerpt` values directly to reproduce the text with normalized line endings. Each page retains its own exact byte range and SHA-256 in the publisher file. The cursor binds its position to the catalog path, RFC number, section, original checksum and extraction checksum. A different source revision requires a fresh opening. Page size can change between calls.

The context covers the heading’s own text up to its next child, sibling, unnumbered division or the end of the file. The reader preserves indented lists and code, and recognizes flush-left box drawings followed by publisher figure captions. A null cursor with `boundary=unclassified_margin` means a flush-left block could not be classified. Inspect the original for the remaining material in that case.

Across 344 archived RFCs, 12,507 of 12,524 numbered sections reached their independently located structural boundary. The other 17 stopped early at an unclassified margin. Across 350 selected sections, 1,557 consecutive excerpts reconstructed every expected publisher byte interval. A separate revision test on 32 archived texts changed publisher bytes or the extraction in memory. All 64 old cursors were rejected, and fresh lookups reproduced the updated revision. The private protocol and receipts are in `search-audit/rfc-section-context-20261112`.

### Captioned figures at the left margin

The section reader can continue past a printed figure that uses several rows of boxes, parallel columns, or a two-ended arrow. It checks the shape of each intervening block and looks for the publisher’s numbered figure caption. It also recognizes a captioned JSON object or array. For the syntax check, a printed newline after column 72 can act as whitespace so a long encoded string can wrap. The excerpt and its byte coordinates always come from the original file, including the printed newline.

RFC 9964 appendices A.1 and A.2 are practical examples. Their openings contain only the heading. Follow the opening’s continuation cursor to read the three JOSE examples and three COSE examples. To reproduce an example, concatenate the excerpts in order and check their individual publisher byte ranges before interpreting the printed data.

The November 12 audit revisited all 17 previously shortened sections and restored 10 across nine standards. The recovered text covers 182,596 additional characters. All six reconstructed example figures matched their original bytes. The complete archive check reached the independently identified boundary of 12,517 of 12,524 numbered sections. Seven sections across six standards still report `unclassified_margin` and require inspection of the original. The private protocol and receipts are under `search-audit/rfc-captioned-blocks-20261112`.

The reader also follows a protocol exchange when the preceding paragraph identifies the figure by number, the printed messages fit the drawing, and the caption repeats that number. It recognizes exchanges with participant columns, labeled handshake phases and symbol legends. It can follow a printed CoAP error response with a header, a content format and an indented payload. For example, continuing RFC 9846 section 2 now reaches the qualification about early application data and peer authentication after the handshake diagram.

A follow-up check recovered four section endings and 12,532 characters. Across the same 344 RFCs, 12,521 of 12,524 numbered sections now reach the next independently located boundary. RFC 9919 appendices B.3 and B.5 and RFC 9578 section 4 still stop at an unclassified margin. The before-and-after publisher byte ranges are under `search-audit/rfc-referenced-figures-20261112`.

## Two ACL papers on citation attribution and redundant evidence

On November 12, the offline collection already held the 2026 ACL papers *CiteGuard* and *RARE* with CC BY 4.0 recorded in the source manifest. Each had three overview vectors. Neither had native full-text passages. CiteGuard tests citation attribution against human references and examines alternative citations. RARE evaluates retrieval when the same fact appears in several documents. Both are relevant to the shared request for source-revision-aware retrieval evaluation. [CiteGuard](https://aclanthology.org/2026.acl-long.282/) and [RARE](https://aclanthology.org/2026.acl-long.923/).

The existing incremental writer added 81 CiteGuard passage vectors and 187 RARE passage vectors to the original FileDB. Re-extracting each of the 43 physical PDF pages reproduced its stored page text. The audit verified every new vector checksum and source span, coverage of every non-whitespace character in both derivatives, and the byte-for-byte contents of all 70,575 previous passage and overview rows. All 21 planned units were present afterward. Eight controls using in-memory changes to revision records confirmed rejection of stale or missing provenance.

Four questions about the papers’ methods were fixed before indexing. Previously, the expected papers ranked third and fourth on two questions and fell outside the first eight on the other two. After indexing, they ranked first on three questions and second on the fourth. For all four questions, the best semantic passage missed the preselected methods page. Passing the same wording and the selected original path to `knowledge-library-summaries source-evidence` returned a physically verified answer in its first excerpt for each question. Three excerpts matched the preselected page. On the fourth, the expected explanation of rank fusion was on page 3, and the first excerpt independently supported it on page 19. These measurements cover two papers and four questions.

On Raspi, the following read-only check verifies the saved results against the publisher PDFs, the native rows, and the original query receipts. The same directory contains the frozen questions, before-and-after rankings, revision controls, and the final checkpoint. To rerun the four semantic queries against the current index, run the neighboring `retrieval.py` with the semantic Python interpreter.

```sh
python3 /data/var/knowmoredirt/knowledge-library/search-audit/acl-citation-redundancy-20261112/replay.py
```

## November 13 — abstain from unreadable PDF font extraction

Verifying an extracted excerpt against its original PDF page proves physical
location, not human readability. A recovered ten-page, 97.8 MB rendering
paper, *Multi-Layer Reservoir Splatting for Temporal Reuse under Disocclusion*,
has some pages where the publisher's embedded font mapping produces
NUL-interleaved strings. In six previously physically verified search
excerpts, four included 52–165 NULs per 950 characters. Despite matching the
source's PDF text layer, those excerpts should not be presented as normal
readable source evidence.

The original-source evidence locator now skips PDF excerpts with at least
twelve NUL characters and at least two percent NUL density. It records
`unreadable_pdf_excerpts_dropped` and explicitly directs the reader back to
the publisher PDF; it does not modify or "correct" original or derivative
text, any source byte position, vector, document rank, or original checksum.
This is a bounded heuristic for one demonstrated encoding failure, not a
semantic or OCR correctness classifier; low-NUL excerpts can still be bad,
and source images or equations may require direct reading.

On the same query, four unreadable excerpts were omitted and the normal
candidate search returned five alternative, physically verified page excerpts,
all with zero NULs and exact derivative character offsets and checksums.
The original PDF was unchanged. Five CiteGuard and five RARE control excerpts
retained their exact page, order, text hash, coordinates and verification
status. The 244 unittest-compatible KMD filesystem tests passed, excluding
the separate pytest-only resource-safety module because pytest is not
installed in the selected Python environment. New synthetic cases cover a
garbled PDF page, its clean alternative, control strings and non-PDF scope.

Private reproducibility evidence and before/after query JSON are under
`/data/var/knowmoredirt/knowledge-library/search-audit/large-ambiguous-pdf-20261113/`.
This code does not automatically run in other deployed commands until their
version is updated; no shared dirty tree or original source was altered.

## November 14 — discover documents, then verify their PDF page evidence

The previously separate steps of semantic document discovery and source
passage inspection can now be run together against the *same* canonical
KnowMoreDiRT FileDB:

```sh
knowledge-library-summaries find-pdf-evidence \
  'How does RARE check whether atomic facts from different documents are redundant?' \
  --documents 3 --passages 5 --snippet-characters 1600
```

This is **local, read-only source retrieval**, not generated question
answering. The command discovers up to five current source documents using
the existing native semantic reader, preserving their original similarity
scores and positions. Non-PDF results remain explicitly labeled `not_pdf`;
missing, stale or reassigned FileDB IDs produce `source_unavailable`, not
an invented citation. For each eligible PDF, it independently verifies
the original publisher checksum, the exact completed download receipt,
the current derived extraction checksum and native FileDB entry ID.
It then runs bounded original-source lexical passage search and checks
each selected excerpt against the physical PDF pages using Poppler, the
original's SHA-256 and exact text occurrence. Only page-`verified`
excerpts may be returned as candidate evidence. Unverifiable physical
pages and heavily NUL-corrupted text excerpts remain omitted, with separate
drop counts and explicit source-status fields.

Passages are candidate evidence, never a proof of the answer. Their selection
first maximizes **distinct query terms present in the actual excerpt**, and
uses original lexical rank to break ties. The original lexical rank is
retained as `source_lexical_rank` alongside the number of covered terms.
Document rankings and similarity scores are not overwritten by this
within-document ordering. The original span offsets, passage hashes,
publisher-page SHA and source revision remain unchanged. Audit logs are
private and omit quoted `source_excerpt` text, preserving source
identities and hashes for later reproduction.

The pre-registered ACL CiteGuard/RARE questions from November 12 test exactly
this distinction. All four questions located their intended paper within
native document ranks 1–2. Before within-paper evidence ordering, three
of four returned their expected physical page and both required anchors
within five excerpts. RARE's question about CRRF rank aggregation and
confidence scores located the correct paper first but placed its physically
verified page 3 seventh in the lexical passages. Sorting physically
verified candidates by distinct query-term coverage lifted that same
original page into the first five. A complete independent repeat recovered
the intended page and both anchors in **all four of four** held-out cases.
Those four successes do not establish open-domain QA precision or
faithfulness beyond the included source excerpts.

Tests additionally simulate stale publisher revisions, a different native
filesystem-entry ID, absent physical page evidence, zero matching excerpts,
unreadable publisher text and correct accounting of rejected candidates.
No downloaded original, PDF, derivative source, model weights, native
vectors or concurrent agent worktrees were modified. Private frozen
questions, pre-change failures, results, and exact repetition scripts live
in `/data/var/knowmoredirt/knowledge-library/search-audit/provenance-integration-20261114/`.

## November 14 — mixed-source, explicitly graded original provenance

The source library is not PDF-only. It contains official publisher HTML and
Markdown, original RFC UTF-8 text, structured Office references and approved
archived course material. The native vector document search can rank these
documents, but the older combined `find-pdf-evidence` correctly refused
non-PDF results without offering an equivalent verified derivative excerpt.
The new `find-source-evidence` command closes that gap without converting an
extracted HTML or Office string into a fictitious exact publisher quotation:

```sh
knowledge-library-summaries find-source-evidence \
  'OpenPBR standard surface material model shader coating dielectric roughness' \
  --documents 5 --passages 3

knowledge-library-summaries find-source-evidence \
  'What does RFC 9110 say about safe idempotent HTTP methods GET and HEAD?' \
  --passages 5

knowledge-library-summaries find-source-evidence \
  --source-name 'NVIDIA Agent Skill: nvidia-ontology-management' \
  'Which ontology management operations and configuration does the document describe?'
```

The input is still bounded to five ranked originals, five excerpts per
original and two thousand characters per excerpt. Source-name resolution uses
the existing single-original publisher/RFC/DOI/skill resolver, not an LLM
guess. A strict affirmative content question such as “What does RFC 9110
say about...” resolves the RFC if the catalog contains one unambiguous exact
publisher source, extracting just the question's subject for within-file
reading. Comparisons and unrelated mentions are *not* silently pinned.
`--source-name` requires an exact title, original catalog path, recognized
publisher identifier or pinned skill name; an absent or ambiguous match fails
without substituting a vector result. A named source reports no semantic
similarity or synthetic native chunk identity, and explicitly labels its
`selection_basis`. Otherwise the command preserves the native semantic
document ranks.

Every source must satisfy the completed original downloader receipt, current
original SHA-256, existing KMD FileDB path and entry ID, current extracted-text
SHA-256 and exact excerpt offset/hash. A PDF additionally requires the physical
publisher-page verifier; no unverified pages are emitted. For HTML, ZIP, Office
and other transformed formats, passages are labeled
`verified_extracted_text_only`: the excerpt is *exactly* reproducible from
the verified current derivative, but its bytes or layout may differ from the
publisher's original. For unmodified UTF-8 publisher text, where the existing
byte-offset reproducer can independently map the full derivative to the
original, the passage gains `publisher_utf8_byte_interval` and the original
byte range/hash. No physical page proof is invented for any non-PDF format.

`evidence_kind_counts` distinguishes physical PDF page proofs, verified
publisher UTF-8 byte intervals, and derivative-only excerpts. Returned text
is strictly potential evidence, not an established real-world answer.
An archived skill is *untrusted publisher data*, never executed as agent
instructions. Audits retain provenance and hashes but omit full
`source_excerpt` text. Ordinary lexical retrieval, native search and the
original PDF-only command remain available.

On November 14, an independent Raspi replay checked seven actually downloaded
originals: three HTML sources, two PDFs, one RFC plain-text standard and one
NVIDIA skill. All twenty returned excerpts exactly matched the pinned
source-map derivative and SHA-256, including eight independently reproduced
original UTF-8 publisher-byte intervals and six physically verified PDF-page
proofs. In a separate negative control, the native search returned five
semantically ranked candidates for a fabricated nonsense query, while the
source-evidence layer correctly reported zero supporting excerpts. Before the
named-source fix, the standalone RFC 9110 question ranked five *other* RFCs;
afterward the intended publisher RFC was selected explicitly, with no invented
semantic score. These are bounded correctness and provenance observations,
not a complete demonstration of document-search recall or answer truth.

The private question, exact original receipts, byte checks, before/after
results and read-only reproduction script remain under Raspi's
`/data/var/knowmoredirt/knowledge-library/search-audit/owner-continuation-20261114-1338/`.
The original documents, converted text and source index have not been
rewritten or copied into Git.

### Cross-format archive and workbook acceptance on November 14

The installed exact-source command additionally recovered three individually
SHA-verified derivative excerpts from the original IHK Rostock liquidity
planning XLSX and three from the original *10 detailed posters* ZIP of
embedded usability-heuristic PDFs. The workbook's worksheet labels and the
ZIP's archive-member page labels are preserved as **extracted-text** scope;
no original XLSX cell byte interval or independently verified embedded PDF
physical page is implied. The independent private replay was extended to
nine originals and 26 current excerpt hashes, including eight exact original
UTF-8 intervals and six physically verified standalone PDF pages. It did
not execute workbook formulas, archived scripts, skills or source code.

## November 15 — exact governing-context windows for source excerpts

The original mixed-source retrieval command previously returned a small
query-matching extract from an otherwise SHA-verified source. That is
insufficient when the *source section's own opening or neighboring text*
qualifies the matched statement as a dream, hypothetical, historical draft,
quotation, denial or superseded practice. A valid excerpt checksum does
not preserve the meaning of prose it omits.

`knowledge-library-summaries find-source-evidence` now includes bounded
`scope_context` alongside each returned excerpt. It carries up to three
individually hashed spans: `section_opening` when the excerpt is far from
the start of the current PDF source unit, Markdown section or original
text section; `preceding` text adjacent to the excerpt; and `following`
text that may contain a qualification or retraction. Each span contains
zero-based half-open character offsets, its own SHA-256 and exact text from
the *same verified current derivative*. No span crosses the parsed original
source-unit or Markdown section boundary. A large source section split into
bounded search chunks still retains its original section opening, rather than
pretending the start of the search chunk is the governing context.

The default is `--scope-context-characters 320` for each span. Use
`--scope-context-characters 0` to retain the previous compact result, or
set an integer from 80 through 600. The original matched excerpt, its SHA
and native document rank are unchanged. This does not rebuild the corpus or
change stored vectors. The separate `find-pdf-evidence` and
`source-evidence` interfaces remain unchanged unless deliberately upgraded
and tested later.

These added spans are marked
`basis=verified_current_derivative_section_only` and
`publisher_page_or_byte_proof=false` even when the *matched* excerpt has
a separately verified physical PDF page or original UTF-8 byte interval.
They are untrusted publisher content, not model instructions or a
statement of truth. A local LLM must inspect scope and, where important,
the complete original rather than declaring a rule true from a retrieved
quote. The original cannot be called simultaneously current and superseded,
and conflicting sources cannot be resolved by a single span. The query
audit retains the relative source positions and SHA-256s of each context
piece, but strips all `text` values, along with the primary
`source_excerpt`, before recording the evidence.

Four independent synthetic hard negatives—fictional dream legislation,
superseded shipment policy, conditional laboratory instruction and a
disputed perpetual-motion quote—showed the same exact excerpt bytes
before and after the change. The old 260-character excerpt lacked the
scope-setting opening in all four cases; the new section-opening proof
recovered it in all four. Additional tests cover explicit retractions
after a clipped passage, opposing statements on different physical pages
that must not be combined, Markdown headings, a long section split at
60,000 characters, stale derivative revisions, strict bounds and privacy
of audit logs. These tests measure *availability of real source context*,
not correctness of a generated answer or universal scope recognition.

Independent examples on the actual OpenPBR HTML specification, the
CiteGuard paper and RFC 9110 reproduced context SHA/offsets exactly,
with PDF excerpts retaining their previously verified physical-page status
and plain UTF-8 excerpts retaining their original-byte proofs. All added
context remains derivative-only evidence. Results and the model-free
reproducible negative-case script are under
`/data/var/knowmoredirt/knowledge-library/search-audit/source-scope-context-20261115/`.
No source originals, FileDB entries, vectors or model servers were
modified by this read-only retrieval extension.

## November 15 — named RFC sections must use publisher body structure

The combined mixed-source operator previously handled exact source names,
but a citation such as `RFC 9110, Section 9.2.2` was reduced to lexical
query terms. On the actual downloaded RFC 9110, this returned SHA-matching
passages from unrelated body locations and an index at the back of the
standard. Valid source hashes did NOT establish section relevance. The
existing dedicated RFC reader already checks the publisher's numbered
body heading against the original table of contents and source revision.

The combined `find-source-evidence` operator now invokes that *existing*
structural source reader, not FTS5 or model similarity, when an exact
unambiguous RFC publisher source and valid numbered section are requested.
It retains the named original and its original publisher SHA, independent
native FileDB entry ID, current exact derivative SHA and verified original
UTF-8 publisher-byte interval. The result has
`passage_selection_basis=verified_rfc_body_section_boundary`, one
`verified_rfc_section_opening` or `verified_rfc_section_continuation`,
and the original `rfc_section_lookup` metadata. It reports no invented
lexical rank or count of query terms for a structurally resolved opening.
The RFC table of contents cannot become an answer substitute.

For a missing section, the result is explicitly
`rfc_section_not_verified`, with zero excerpts and no lexical fallback.
A malformed or source-mismatched original-byte proof is refused. A revision-
and-section-bound continuation cursor cannot be borrowed for another
section or source, and no continuation is accepted without an exact RFC
source and numbered section. The same operator can follow a verified
section window by passing `--continue-section` the prior
`rfc_section_lookup.context.next_cursor`; it never crosses the
independently located publisher section boundary.

```sh
knowledge-library-summaries find-source-evidence \
  'RFC 9110, Section 9.2.2' --documents 1 --passages 2 --snippet-characters 520

# Extract .documents[0].rfc_section_lookup.context.next_cursor and then:
knowledge-library-summaries find-source-evidence \
  --source-name 'RFC 9110' --continue-section "$cursor" \
  'Section 9.2.2' --documents 1 --passages 1 --snippet-characters 520
```

The November 15 live negative control was the *actual* original RFC 9110.
Before the change, `find-source-evidence 'RFC 9110, Section 9.2.2'`
returned five unrelated original-text passages with valid but irrelevant
publisher-byte proofs. Afterward, it returned **Idempotent Methods** in
Section 9.2.2, with a 333-character opening, a publisher-byte SHA match and
a continuation cursor. Following that cursor returned source characters
333–853, exactly contiguous with the opening, with a second verified
publisher-byte interval. Reusing the first cursor against Section 9.2.1
returned `source_unavailable` with an explicit source/section cursor
mismatch and no quoted text. Section 999.999 did not exist and produced
zero excerpts. These tests preserve the source semantics but do not
decide an answer's truth or validity beyond that section.

Private before/after results and publisher-byte checks are under
`/data/var/knowmoredirt/knowledge-library/search-audit/rfc-direct-section-20261115/`.
The original `source-evidence` specialized RFC section reader is
unchanged, and the same proof routine is reused rather than cloned.
No original text, native vectors or background inference workers have
been modified.

## November 15: verified source link from All2Text PDF figure crops to native KMD

All2Text's isolated feat/pdf-figure-provenance-20261115 feature extracts
source-linked raster image groups from already approved original PDF pages,
with original full-file SHA-256, exact 1-based physical page, original PDF
figure bbox, Poppler-rendered page SHA, bounded crop bbox and crop PNG SHA.
Captions are PDFMiner extracted text, only heuristically associated with
nearby images; neither captions nor figures are original publisher-byte
quotes or authenticated factual claims. It does not discover vector-only
figures or convert scanned pages into trusted text.

The independent read-only KMD operator,
scripts/raspi_visual_pdf_link.py, checks each crop's original full-file
SHA-256 against the current source-map revision, one canonical FileDB entry
ID, full current extracted text SHA if available, pixel-exact crop equality
against its rendered page, and an independent second Poppler re-render of
that original physical page whose full raster SHA must match. It finds
one unique initial 96 normalized caption characters inside only the
exact verified [SOURCE UNIT: PDF page N] source section, mapping the match
back to literal derived character offsets and SHA-256. Duplicated or
absent caption matches abstain. Visual-only originals remain marked without
completed text. No originals, KMD databases, vectors or current summaries
are rewritten.

Observed real-original examples:
- OpenPBR SIGGRAPH 2025 shading source PDF, physical pages 8 and 14:
  3 grouped raster figures (Figure 4, 5 and 9), all 3 caption-prefix
  anchors independently verified, native FileDB entry 14180 with
  exactly 3 overview vectors, not full-text coverage.
- Multi-Layer Reservoir Splatting original PDF, page 1:
  a 13-panel rendered comparison grouped as one source figure, Figure 1
  derived caption anchor, native FileDB entry 19069 with 67 full-text
  vector chunks and 3 overview vectors.
- Philippines RA11232 original scan, page 1:
  one full-document-page image, native FileDB entry 9913, source-map
  status error for incomplete OCR, zero native vectors and no invented
  caption proof.

The three source original physical pages also passed independent
whole-page Poppler raster SHA reproduction. The current read-only linkage
JSON and exact hashes are under Raspi's private
/data/var/knowmoredirt/knowledge-library/search-audit/pdf-layout-grounding-20261115/kmd_links/
with *_with_native_chunks.json names. The All2Text input figure/context
receipts, independent source-raster copies and a review contact sheet stay
in the parent audit folder; cropped publisher figures stay outside Git.

The KMD full unittest-compatible suite passed 294 tests, including nine new
source/derived revision, duplicated/wrong-page caption, source-map failure,
and crop tamper cases. This provides reproducible image provenance, NOT
a visual description accuracy score, complete OCR, correct chart values,
or photo/render authenticity. Ordinary text, semantic retrieval and ongoing
KMD background summarization continue to use the existing native FileDB.

A final fail-closed output guard checks the report's resolved destination
before doing any original/filedb reads: it rejects an existing report,
original-corpus or derived-text destination, a direct native database-root
or source-map folder report, and symlinked parent paths into protected
storage. Inside the KMD runtime tree, only the purpose-built private
search-audit directory is permitted. A synthetic test independently
exercised all these cases and preserved existing file contents.

## November 19 — schema-level native summary bounds on one stalled original

The existing Qwen3-Coder shard worker remains active and has completed
thousands of other source originals. Its persisted failure ledger shows a
single pending original, the EVATok video-tokenization paper (matching
original and derivative revisions); it has retried nine times. The precise
failure is `EmbeddingContextExceeded: native summary exceeds safe embedding
context` raised by `hierarchical_summary_model.generate_retrieval_analysis`.
The native E5 encoder uses a maximum of 512 tokens and only 400 are safe
for the document summary after source/embedding prefix reservations. A
*word-count instruction alone* had not constrained the model to that
budget reliably even after a shorter retry. Do not treat repeated
scheduled retries as evidence of progress or mark this original complete.

The isolated KMD change adds a portable, schema-constrained `maxLength`
and `minLength` to the original native `document_summary` field **before
model generation**: a bounded initial 1200-character field and an even
shorter 480-character field on the strict/compact retry. These bounds
apply ONLY to the generated summary, not to the publisher's original
source text, original complete-source group, native facets or the native
KMD FileDB schema. The context-aware JSON schema still uses the same
original KMD facet structure; the original FILE_ANALYSIS_SCHEMA constant
is unchanged. The generated output still has to pass the independent
400-*model-token* check before being embedded or cached. The new branch
also refuses an overlong cached summary, and treats a model/server that
ignores maxLength as an explicit recoverable `EmbeddingContextExceeded`,
not a reason to truncate generated text or publish an incomplete node.

Five new isolated tests cover both decoder-side bounds, model ignoring
the cap, unsafe cached output, original facet schema preservation and
ordinary short native summaries; the former retry request-equality test
was updated to compare unchanged facet schema while requiring the shorter
retry's deliberate new summary cap. The complete KMD unittest-compatible
suite passes **300 tests, zero failures** on the isolated branch.

This is a verified **code-level guard**, not yet proof that the historical
EVATok source now completes against the real shared Qwen server. At
inspection the shared Coder model slot was busy serving the existing
worker. Do not issue an uncontrolled parallel request or restart the
worker mid-document merely to claim recovery. Install and independently
confirm the safe code revision, then allow or coordinate a documented
safe worker handoff before declaring the pending source resolved.
