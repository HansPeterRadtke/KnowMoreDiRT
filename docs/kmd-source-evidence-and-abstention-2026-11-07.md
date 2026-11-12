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
