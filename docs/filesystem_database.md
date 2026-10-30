# Filesystem semantic database subsystem

KnowMoreDiRT now includes the filesystem catalog code from the `devtests/file-system` repository as an isolated package named `file_system_catalog`. The devtest copy remains unchanged for independent research. The KMD copy is vendored so the fast semantic database can be initialized and queried without building the slower DRT/DSPG world model.

The subsystem scans arbitrary folder trees into SQLite, preserves lossless filesystem metadata, extracts text, chunks files, creates embeddings, optionally creates LLM-generated semantic representations, supports literal and semantic retrieval, plans bounded searches with a constrained local model, and produces grounded answers with explicit evidence records.

It can be used directly through the vendored package or through the KMD facade:

```python
from knowmoredirt.filesystem import (
    initialize_filesystem_database,
    question_filesystem_database,
)

initialize_filesystem_database(
    "/path/to/raw/folder",
    "/path/to/catalog.sqlite3",
)

result = question_filesystem_database(
    "/path/to/raw/folder",
    "/path/to/catalog.sqlite3",
    "Which report mentions the failed pressure test?",
)
```

The command-line entry points are `kmd-filesystem` for initialization and grounded questions and `kmd-filesystem-search` for direct semantic or literal retrieval.

This subsystem does not initialize DRT, create persistent KMD referents, or perform DRS proof validation. It is the fast retrieval and basic LLM question-answering layer. Future integration will let KMD consume this database for candidate evidence, mention profiles, and referent-resolution candidates while retaining DRT/DSPG as the persistent identity and reasoning layer.

The source was copied from devtests commit `6faaccc9`. Regression tests from that repository are preserved under `tests/filesystem_database/`. Any future synchronization must be explicit and reviewed; changes in either copy do not automatically propagate to the other repository.

## Full Raspi Knowledge Library deployment (October 2026)

The owner also uses this original filesystem subsystem for the complete offline Knowledge Library on host `raspi`; the library contains PDFs, HTML, Markdown, original spreadsheets and ZIP courses, so **do not** call the text-only semantic `kmd-filesystem init` on that raw source root. Its `ContentSemanticPipeline._extract_text` decodes file bytes as text and will otherwise embed meaningless binary PDF/ZIP bytes. The original metadata scanner works on any file kind; semantic text extraction is a separate capability that requires tested format-specific extraction and a reachable local embedding model.

The **Infra-owned** Raspi command is `/data/bin/knowledge-library-filesystem`; full usage and deployment verification steps live in `/data/infra/docs/knowledge-library-knowmoredirt-filesystem.md`. The actual source corpus is `/data/var/Knowledge Library/`, while its private metadata SQLite is `/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3`. The database must stay **outside** the scanned root and all Git repositories. The deployed code lives in a clean, pinned checkout of KnowMoreDiRT; the original local `master` and separate discourse-research worktree were not reset or edited.

Usage on Raspi:

```bash
# Check actual original-file coverage, schema and SQLite integrity:
/data/bin/knowledge-library-filesystem status --verify

# Search KMD native filesystem metadata, no LLM involved:
/data/bin/knowledge-library-filesystem find --name 'Digital Marketing' --limit 10
/data/bin/knowledge-library-filesystem find --extension pdf --path money-and-business --limit 20
/data/bin/knowledge-library-filesystem find --extension pdf --sort size_desc --limit 10

# Initialize from scratch only if the database does not yet exist:
/data/bin/knowledge-library-filesystem scan --fast-samples --skip-exiftool --max-hash-bytes 65536 --progress-every 1000

# Atomic metadata refresh after library additions. Refuses to erase populated semantic vectors:
/data/bin/knowledge-library-filesystem refresh --fast-samples --skip-exiftool --max-hash-bytes 65536 --progress-every 1000
```

`--fast-samples` and `--skip-exiftool` are **Infra wrapper options**, not an option in upstream `python -m file_system_catalog`. It invokes the unmodified original KMD scanner, but caps the per-file byte-structure histogram at a one-megabyte exact-analysis threshold and three sixty-four-kilobyte samples for larger items. The resulting original metadata indicates `analysis_complete=false` for a sampled file. The deployed wrapper additionally skips ExifTool deep decoding on all files to reduce CPU load, explicitly stores its unavailable status and retains other original KMD PDF/Office/archive parsers. This choice reduces detailed Exif coverage, **not the number of indexed filesystem entries**. The hash cap independently marks files larger than sixty-four kilobytes `skipped_large` with head/tail hashes; it does **not** invent a complete SHA-256. For the original downloaded files, the separately maintained offline Knowledge Library downloader ledger retains their full SHA-256 verification, so the cheaper native KMD metadata scan need not reread the entire multi-gigabyte archive. This distinction remains visible in the status output. All original files and native metadata types are included in the scan, with KMD's unchanged schema and original source paths.

`status --verify` checks that all currently downloaded originals are present in native `filesystem_entries`, the source root is correct, the SQLite schema/integrity/foreign keys are valid, and reports whether native `content_chunks` and `content_representations` have been populated. Metadata alone is not a completed semantic vector index; preserve that limitation for agents and user reports. Full text PDF page search currently also exists in a **separate** Infra-managed, business-only lexical index; it is not part of this native metadata database. All downloaded text, full-file caches, outputs, SQLite, source inventories and privacy-sensitive metadata remain exclusively under private `/data/var`, not in the KMD or Infra Git repository.
