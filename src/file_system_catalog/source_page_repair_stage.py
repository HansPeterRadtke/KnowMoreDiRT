"""Stage corrected physical PDF page derivatives without touching the live corpus.

This is an offline, source-revision-checked preparation step, NOT a migration.
Existing source-map, original FileDB, live derivative, vectors, summaries,
services and publisher PDFs are never modified. Staged full extracted text is
private local data and must not be copied to Git or distributed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from collections.abc import Callable, Iterable

from .source_review_audit import safe_file, sha_file

SHA = re.compile(r"[0-9a-f]{64}\Z")
PDF_UNIT = re.compile(r"(?m)^\[\[SOURCE UNIT: PDF page ([1-9][0-9]{0,5})\]\]")
MAX_PAGES = 60
MAX_ORIGINAL_BYTES = 50 * 1024 * 1024
MAX_PUBLISHER_TEXT_CHARS_PER_PAGE = 32 * 1024 * 1024
SPLIT_CHARS = 1024 * 1024
MAX_ITEMS = 327


def approved_identity(item: dict, source_entry: dict, source_map: sqlite3.Connection,
                      native_db: sqlite3.Connection, completed_receipts: dict[str, str],
                      original_root: Path, derived_root: Path) -> dict:
    """Require current approved original, cached derivative and native FileDB ID.

    A previous audit/repair plan is only a *proposal*. All revisions are
    verified again against the present source-map, receipt, and full files.
    """
    if not isinstance(item, dict) or not isinstance(source_entry, dict):
        raise ValueError("untrusted staged original reference")
    rel = item.get("source_original_path")
    if not isinstance(rel, str) or not rel.lower().endswith(".pdf"):
        raise ValueError("repair only approved original PDF files")
    if source_entry.get("path") != rel or source_entry.get("kind") != "pdf":
        raise ValueError("repair candidate absent from approved source manifest")
    orig_sha = item.get("original_sha256")
    prev_sha = item.get("old_derived_sha256")
    if (not isinstance(orig_sha, str) or not SHA.fullmatch(orig_sha)
            or not isinstance(prev_sha, str) or not SHA.fullmatch(prev_sha)):
        raise ValueError("invalid pinned source and old derivative SHA")
    rows = source_map.execute(
        "SELECT source_sha256,derived_sha256,derived_path,original_title,status "
        "FROM originals WHERE source_path=?", (rel,)).fetchall()
    if (len(rows) != 1 or rows[0][4] != "complete" or rows[0][0] != orig_sha
            or rows[0][1] != prev_sha):
        raise ValueError("source-map complete original revision changed")
    drel = rows[0][2]
    if drel != rel + ".txt":
        raise ValueError("unexpected existing source derivative mapping")
    if rows[0][3] != source_entry.get("title"):
        raise ValueError("publisher title differs from current source-map")
    if completed_receipts.get(rel) != orig_sha:
        raise ValueError("approved complete original receipt missing or stale")
    idrows = native_db.execute(
        "SELECT id FROM filesystem_entries WHERE relative_path_display=? "
        "AND entry_type='file'", (rel,)).fetchall()
    if (len(idrows) != 1 or type(idrows[0][0]) is not int
            or idrows[0][0] != item.get("native_original_file_entry_id")):
        raise ValueError("original lacks unchanged unique native FileDB identity")
    orig = safe_file(original_root, rel)
    old = safe_file(derived_root, drel)
    if orig.stat().st_size > MAX_ORIGINAL_BYTES:
        raise ValueError("original exceeds existing safe pagewise fallback bound")
    if sha_file(orig) != orig_sha or sha_file(old) != prev_sha:
        raise ValueError("publisher source or existing derivative full SHA changed")
    physical = item.get("publisher_physical_pdf_pages")
    claimed_max = item.get("old_derived_max_page_label")
    if (type(physical) is not int or not 1 <= physical <= MAX_PAGES
            or type(claimed_max) is not int or claimed_max <= physical):
        raise ValueError("candidate is not confirmed within bounded physical PDF repair scope")
    if old.stat().st_size > 128*1024*1024:
        raise ValueError("cached derivative exceeds bounded whole-source recheck")
    text = old.read_text(encoding="utf8", errors="strict")
    labels = [int(x) for x in PDF_UNIT.findall(text)]
    if not labels or max(labels) != claimed_max:
        raise ValueError("legacy original PDF page-label overcount changed")
    return {"source_path": rel, "original_path": orig, "old_derived_path": old,
            "source_sha256": orig_sha, "old_derived_sha256": prev_sha,
            "source_entry": source_entry, "native_original_id": idrows[0][0],
            "expected_physical_pdf_pages": physical,
            "old_max_cached_pdf_page_label": claimed_max}


def original_derivative_header(entry: dict, relative: str, source_sha: str) -> str:
    """Match canonical Infra extraction provenance header without rewording."""
    data = {"source_path": relative, "source_sha256": source_sha,
            "title": entry.get("title"),
            "publisher_url": entry.get("source_url", entry.get("url")),
            "kind": "pdf", "publication_year": entry.get("publication_year"),
            "rights_note": entry.get("rights_note")}
    return ("KMD source provenance (the original PDF/HTML/ZIP/XLSX, not a new publisher book):\n"
            + json.dumps(data, ensure_ascii=False) + "\n\n")


def write_exact_physical_pages(stream, pages: Iterable[tuple[str, str]],
                               expected_pages: int) -> dict:
    """Reproduce canonical source-unit serialization from original page N.

    Require original-numbered increasing physical PDF page identities.
    Infra's extractor omits truly blank pages; gaps are valid but invented,
    duplicate, out-of-order or beyond-end page identities are never valid.
    An embedded publisher text marker must not spoof new derived page units.
    """
    previous = 0
    emitted = 0
    chars = 0
    nonempty = 0
    for label, raw in pages:
        match = re.fullmatch(r"PDF page ([1-9][0-9]{0,5})", label) if isinstance(label,str) else None
        page = int(match[1]) if match else -1
        if (page <= previous or page > expected_pages or not isinstance(raw,str)):
            raise ValueError("invented, out-of-range, backward or duplicate physical PDF page identity")
        previous = page
        if len(raw) > MAX_PUBLISHER_TEXT_CHARS_PER_PAGE:
            raise ValueError("unexpected excessively long extracted physical PDF page")
        if re.search(r"(?m)^\[\[SOURCE UNIT:", raw):
            raise ValueError("untrusted original text contains spoofable source page delimiter")
        body = raw.strip()
        if body:
            nonempty += 1
            for n, start in enumerate(range(0, len(body), SPLIT_CHARS), 1):
                part = body[start:start+SPLIT_CHARS]
                heading = label if len(body) <= SPLIT_CHARS else f"{label}, continued section {n}"
                block = f"\n\n[[SOURCE UNIT: {heading}]]\n{part}\n"
                stream.write(block)
                emitted += 1
                chars += len(block)
    if nonempty == 0 or chars < 30:
        raise ValueError("no useful publisher physical PDF page text")
    return {"publisher_pdf_physical_page_count": expected_pages,
            "highest_extracted_original_page": previous,
            "physical_page_labels_emitted": nonempty,
            "physical_page_count_not_text_coverage_proof": True,
            "staged_source_units": emitted, "staged_text_characters": chars}


def stage_verified_original(item: dict, source_entry: dict,
                            source_map: sqlite3.Connection, native_db: sqlite3.Connection,
                            completed_receipts: dict[str, str], original_root: Path,
                            derived_root: Path, stage_dir: Path,
                            extract_page_units: Callable[[Path,Path],Iterable[tuple[str,str]]],
                            actual_pdf_count: Callable[[Path],int]) -> dict:
    """Write one new private staging file exclusively; never overwrite live.

    The physical page count and the current original/derived hashes are
    rechecked after extraction and before stage publication.
    """
    state = approved_identity(item, source_entry, source_map, native_db,
                              completed_receipts, original_root, derived_root)
    src = state["original_path"]
    actual = actual_pdf_count(src)
    if type(actual) is not int or actual != state["expected_physical_pdf_pages"]:
        raise ValueError("independent current publisher PDF physical page count differs")
    rel = state["source_path"]
    final = stage_dir / (rel + ".txt")
    if not final.resolve(strict=False).is_relative_to(stage_dir.resolve(strict=True)):
        raise ValueError("staged PDF path escapes private stage root")
    if final.exists() or final.is_symlink():
        raise FileExistsError("private staged original already exists; never replace")
    parent = final.parent
    base = stage_dir.resolve(strict=True)
    current = base
    for part in Path(rel).parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError("staging destination path contains symlink")
        current.mkdir(mode=0o700, exist_ok=True)
        os.chmod(current, 0o700)
    temporary = None
    staged_here = False
    try:
        fd, name = tempfile.mkstemp(prefix=".source-page-stage-", suffix=".part", dir=parent)
        temporary = Path(name)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf8") as out, tempfile.TemporaryDirectory(
                prefix=".pdf-conversion-scratch-", dir=stage_dir) as scratch:
            out.write(original_derivative_header(source_entry, rel, state["source_sha256"]))
            counters = write_exact_physical_pages(
                out, extract_page_units(src, Path(scratch)), actual)
            out.flush()
            os.fsync(out.fileno())
        if (sha_file(src) != state["source_sha256"]
                or sha_file(state["old_derived_path"]) != state["old_derived_sha256"]):
            raise ValueError("original or existing derivative mutated during offline staging")
        # Recheck source-map revision right before private stage publication.
        row = source_map.execute(
            "SELECT source_sha256,derived_sha256,derived_path,status FROM originals WHERE source_path=?",
            (rel,)).fetchall()
        if (len(row) != 1 or row[0] !=
                (state["source_sha256"], state["old_derived_sha256"], rel+".txt", "complete")):
            raise ValueError("source-map changed during offline staging")
        # Hardlink is an atomic, exclusive creation: never replace any prior
        # staged result. The disposable temporary remains in the same tree.
        os.link(temporary, final, follow_symlinks=False)
        staged_here = True
        os.chmod(final, 0o600)
        staged_sha = sha_file(final)
        if staged_sha == state["old_derived_sha256"]:
            raise ValueError("page-map stage unexpectedly retained the incorrect old derivative")
        return {"status": "staged_private_only", "source_path": rel,
                "original_sha256": state["source_sha256"],
                "old_derived_sha256": state["old_derived_sha256"],
                "staged_sha256": staged_sha, "native_original_file_entry_id": state["native_original_id"],
                "old_invalid_cached_last_pdf_page": state["old_max_cached_pdf_page_label"],
                "current_physical_pdf_pages": actual,
                "staged_file": str(final), "staged_bytes": final.stat().st_size,
                "original_and_live_derived_full_sha_reverified": True,
                "no_live_source_map_fileDB_summary_vector_original_or_old_derivative_writes_by_stager": True,
                "live_promoted": False,
                "original_physical_page_excerpts_verified": False,
                **counters}
    except BaseException:
        if staged_here:
            final.unlink(missing_ok=True)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
