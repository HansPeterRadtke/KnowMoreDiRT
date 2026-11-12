"""Reproduce complete retrieval excerpts on physical pages of a verified publisher PDF."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable

_PAGE = re.compile(r"PDF page ([1-9][0-9]{0,5})(?=$|, continued section | \| | \[(?:part|source offset) )")
_MAX_RELOCATION_PAGES = 32
# A verified large-layout page expands to 7.2 MB of mostly spaces.
_MAX_OUTPUT = 8 * 1024 * 1024
_MAX_SECONDS = 20


def _unverified(reason: str) -> dict:
    return {"status": "unverified", "reason": reason}


def locate_page_excerpts(passages: list[dict], page_count: int | None,
                         read_page: Callable[[int], bytes | None]) -> list[dict]:
    """Verify stated pages first. Relocation requires a unique match in a complete scan."""
    proofs = []
    pending = {}
    for i, hit in enumerate(passages):
        label = _PAGE.match(hit["source_section"])
        needle = " ".join(hit["source_excerpt"].split())
        if not label:
            proofs.append(_unverified("no_text_page_label"))
        elif not needle:
            proofs.append(_unverified("empty_excerpt"))
        elif not page_count:
            proofs.append(_unverified("page_count_unavailable"))
        else:
            pending[i] = (int(label[1]), needle)
            proofs.append(None)

    if not pending:
        return proofs

    matches = {i: {} for i in pending}

    def inspect(page: int) -> bool:
        raw = read_page(page)
        if raw is None:
            return False
        try:
            normalized = " ".join(raw.decode("utf-8").split())
        except UnicodeDecodeError:
            return False
        digest = hashlib.sha256(raw).hexdigest()
        for i, (label, needle) in list(pending.items()):
            start = normalized.find(needle)
            if start < 0:
                continue
            anchor = {"status": "verified", "page": page, "page_text_sha256": digest,
                      "start_char": start, "end_char": start + len(needle)}
            if label == page:
                proofs[i] = {**anchor, "basis": "derivative_label_confirmed"}
                del pending[i]
            else:
                matches[i][page] = anchor
        return True

    nominated = {label for label, _ in pending.values() if label <= page_count}
    unreadable = set()
    for page in sorted(nominated):
        if not inspect(page):
            unreadable.add(page)
    if pending and page_count <= _MAX_RELOCATION_PAGES:
        for page in range(1, page_count + 1):
            if page not in nominated and not inspect(page):
                unreadable.add(page)
    for i, (label, _) in pending.items():
        found = matches[i]
        if len(found) > 1:
            proofs[i] = _unverified("multiple_matching_pages")
        elif page_count > _MAX_RELOCATION_PAGES:
            proofs[i] = _unverified("page_extraction_unavailable" if label in unreadable else "relocation_limit")
        elif unreadable:
            proofs[i] = _unverified("incomplete_page_scan")
        elif found:
            proofs[i] = {**next(iter(found.values())), "basis": "unique_across_pdf"}
        else:
            proofs[i] = _unverified("excerpt_absent")
    return proofs


def _poppler(args: list[str], fd: int, deadline: float) -> bytes | None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    # A PDF can contain very long text runs. Limit the child's output at
    # the OS boundary before reading it into memory.
    try:
        with tempfile.TemporaryFile() as output:
            proc = subprocess.run(
                ["/usr/bin/prlimit", f"--fsize={_MAX_OUTPUT}", "--as=536870912", "--", *args],
                pass_fds=(fd,), stdout=output, stderr=subprocess.DEVNULL,
                env={**os.environ, "LC_ALL": "C"}, timeout=min(remaining, 6),
            )
            if proc.returncode or output.tell() >= _MAX_OUTPUT:
                return None
            output.seek(0)
            return output.read()
    except (OSError, subprocess.TimeoutExpired):
        return None


def verify_pdf_page_excerpts(path: Path, expected_sha256: str, passages: list[dict]) -> list[dict]:
    """Hash the same open file description supplied to Poppler and detect any later mutation."""
    if not passages:
        return []
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
    with os.fdopen(os.open(path, flags), "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or hashlib.file_digest(source, "sha256").hexdigest() != expected_sha256:
            raise ValueError("publisher PDF differs from the verified original revision")
        fd = source.fileno()
        name = f"/proc/self/fd/{fd}"
        deadline = time.monotonic() + _MAX_SECONDS
        info = _poppler(["/usr/bin/pdfinfo", name], fd, deadline)
        counts = re.findall(rb"(?m)^Pages:[ \t]+([1-9][0-9]{0,5})[ \t]*$", info or b"")
        page_count = int(counts[0]) if len(counts) == 1 else None

        def read_page(page: int) -> bytes | None:
            return _poppler(["/usr/bin/pdftotext", "-f", str(page), "-l", str(page),
                             "-layout", "-enc", "UTF-8", name, "-"], fd, deadline)

        proofs = locate_page_excerpts(passages, page_count, read_page)
        after = path.stat(follow_symlinks=False)
        if (not os.path.samestat(before, after) or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns or before.st_ctime_ns != after.st_ctime_ns):
            raise ValueError("publisher PDF changed during page verification")
        return proofs
