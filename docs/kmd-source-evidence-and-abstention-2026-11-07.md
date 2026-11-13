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
