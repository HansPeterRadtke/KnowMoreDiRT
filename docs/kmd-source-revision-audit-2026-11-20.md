# KnowMoreDiRT curated original revision and native-index audit — November 20

This operator runs entirely offline, and never changes downloaded originals, the
source-map, the one native KMD FileDB, index rows or background workers.

scripts/raspi_source_revision_audit.py does two DIFFERENT levels of verification:

1. FULL METADATA reconciliation of every source-map original against the approved
   source catalog, downloaded completion ledger and native FileDB. It verifies
   unique canonical names, original size and SHA receipt, source kind, approved
   publisher title/URL, one matching native entry ID/size, and counts of indexed
   overview, summary and full source chunks. SQLite quick_check and foreign-key
   checks verify the existing FileDB. This does not read all original file bytes.
2. DETERMINISTIC STRATIFIED full-byte rehashing of a bounded original sample
   across PDFs, HTML, skills, text, ZIPs and XLSX. Selection uses a stable
   SHA of seed plus source path, and chooses both recursively summarized and
   not-yet-summarized originals where available. Both intentionally unextracted
   scanned PDFs are included. Every selected original's actual bytes are checked
   against completed source SHA receipts; successfully converted text is checked
   against the source-map derivative SHA and correct embedded source identity.
   A stable file's device/inode/size/time is verified around reading; path
   traversal and symlinks are refused.

The source-map text_characters count excludes the KMD-generated provenance
header. This audit verifies the generated header as metadata, including source
path/SHA/kind, and checks text_characters against the publisher source-unit
blocks only. An initial audit incorrectly included the generated header, making
ninety-eight false-positive character-count findings. The original failed
auditor receipt was preserved; a corrected replay of the SAME original sample
proved those derivatives have valid source/body accounting.

Real acceptance, November 20, 2026:

- Exactly 17,717 curated original source-map paths, catalog sources and completed
  download receipts were reconciled. No path duplicates or mismatches of stored
  publisher SHA, source size, kind, publisher title/URL or original native FileDB
  file entry were observed. This is metadata verification, not full-file SHA
  authentication across the entire library.
- The single native database had 17,715 original files with overview embeddings,
  8,205 with recursive summary embeddings at the snapshot, and 115 with
  historical full-text embeddings. Two known scanned PDF originals retain
  honest error/unextracted state. SQLite quick_check was okay, with zero
  foreign-key violations.
- The deterministic sample rehashed 100 original files and 98 current derivative
  texts, reading about 320 MB total. Both image-only original PDFs were fully
  SHA verified but had no derivative to verify. No sampled original/derivative
  mismatches, skipped files or byte-budget overruns were found.
- This is NOT an audit of every publisher original byte, every vector payload,
  source page accuracy, human factual claims, mathematical formulas or diagram
  interpretation. It does not infer a valid citation merely from vector presence.

The previously blocked EVATok source is now indexed with three source-linked
native summary chunks and ranks first for a directly relevant semantic query.
Four source excerpts independently reproduced their actual publisher PDF pages
one, two, four and seven, with raw original PDF SHA, extracted page text SHA,
source derivative SHA, exact excerpt offsets and SHA. An unrelated hard negative
returned zero source passages. This proves one original has working end-to-end
retrieval and page provenance; it does not calibrate corpus-wide relevance.

A separate source-type check independently reproduced six excerpt spans and
derivative hashes from the Philippine Supreme Court law HTML original, PBRT
fourth-edition Reflection Models HTML chapter and RFC 9110 idempotence section.
Only the RFC excerpt was independently verified as a literal raw publisher-byte
interval. HTML publisher documents contain markup, so the other five positions
are exact current DERIVED TEXT coordinates, not original HTML byte citations.
No source query result automatically proves factual truth or answer sufficiency.

Private reproducibility receipts are stored on Raspi under:

/data/var/knowmoredirt/knowledge-library/search-audit/source-revision-audit-20261120/

Use the new, private operator with the source root, derived-text root, source
catalog, completed ledger, source-map SQLite and the same native FileDB.
The output path MUST be a NEW name outside original, derivative and protected
native data. The tool refuses existing output names, large unchecked files,
symlinks and traversal. See the command's --help for bounded sample sizes.

Remaining work: finish recursive source summaries and independently evaluate
source-passage relevance/answer sufficiency on original held-out documents,
including hard negatives, contradictions and historical scope. Continue
quality evaluation of OCR and figures for the two large scanned PDFs.
The sampled audit does not authorize downloading new originals, index resets,
model downloads, or replacing the original KnowMoreDiRT FileDB.

A second deterministic seed, selected independently after the initial QA,
repeated the full metadata check and a new 100-original / 98-derivative
byte audit. Both runs succeeded with zero metadata mismatches, zero
actual source/derivative mismatches and zero audit-budget skips. Only ten
publisher originals occurred in both samples: together, this directly
verified 190 **distinct** originals and 188 **distinct** current derived
texts, reading more than 650 MB of real bytes across the two replays.
This increases coverage but is still a deterministic sample, **not** full
library source-byte verification. The exact two independent report hashes,
selection overlap and counts are saved as two_seed_independent_results_20261120.json
in the same private audit directory. The legacy false-positive first
run remains preserved separately so the reason for the corrected text
header accounting is reproducible.
