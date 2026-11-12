"""Offline RFC identifiers and exact, publisher-anchored section openings."""

from __future__ import annotations

import hashlib
import re

from .publisher_excerpt import attach_publisher_excerpts

_FLAGS = re.IGNORECASE | re.ASCII
_NUMBER = r"[1-9][0-9]{0,4}"
_RFC_NAME = re.compile(rf"RFC[ \t]*({_NUMBER})", _FLAGS)
_URL = re.compile(
    rf"https://www\.rfc-editor\.org/(?:rfc/rfc({_NUMBER})(?:\.txt|\.html)?|info/rfc({_NUMBER}))"
)
_KEY = r"(?:[1-9][0-9]*|[A-Z])(?:\.[0-9]+)*"
_SECTION = re.compile(rf"(?:section[ \t]+|§[ \t]*)({_KEY})|appendix[ \t]+([A-Z](?:\.[0-9]+)*)", _FLAGS)
_CITATION = re.compile(rf"(RFC[ \t]*{_NUMBER})(?:[ \t]*,[ \t]*|[ \t]+)(.+)", _FLAGS)
_FRAGMENT = re.compile(r"(section|appendix)-([A-Za-z0-9.]+)")
_HEADER = re.compile(rb"(?m)^Request for Comments:[ \t]+([0-9]+)(?=[ \t\r\n]|$)")
_HEADING = re.compile(rf"(?:Appendix [A-Z]|{_KEY})\.[ \t]+\S")
_BODY_LABEL = re.compile(rf"(?m)^(?:Appendix )?({_KEY})\.[ \t]+\S")
_CONTINUATION = re.compile(r"\n( +)(\S[^\n]*)")
_PARAGRAPH = re.compile(r"\n(?:[ \t]*\n)+[ \t]+\S[^\n]*(?:\n[ \t]+\S[^\n]*)*")
_UNIT = re.compile(r"(?m)^\[\[SOURCE UNIT: Original plain text\]\][ \t]*\r?\n")


def rfc_number(value: str) -> str | None:
    if not isinstance(value, str):
        return None
    match = _RFC_NAME.fullmatch(value) or _URL.fullmatch(value)
    return next((part for part in match.groups() if part), None) if match else None


def section_key(value: str) -> str | None:
    match = _SECTION.fullmatch(value.strip())
    return (match[1] or match[2]).upper() if match else None


def split_rfc_citation(value: str) -> tuple[str, str] | None:
    match = _CITATION.fullmatch(value)
    if match:
        name, selector = match.groups()
    else:
        name, _, anchor = value.partition("#")
        fragment = _FRAGMENT.fullmatch(anchor)
        if not fragment or not _URL.fullmatch(name):
            return None
        selector = " ".join(fragment.groups())
    key = section_key(selector)
    return (name, "Section " + key) if key else None


def matching_rfc(record: dict, number: str) -> str | None:
    if record.get("url") != f"https://www.rfc-editor.org/rfc/rfc{number}.txt":
        return None
    reference = record.get("source_url")
    if reference is not None and (
        not isinstance(reference, str) or not _URL.fullmatch(reference) or rfc_number(reference) != number
    ):
        return None
    return "RFC " + number


def rfc_header_matches(raw: bytes, number: str) -> bool:
    return _HEADER.findall(raw[:4096].removeprefix(b"\xef\xbb\xbf")) == [number.encode("ascii")]


def _headings(text: str, key: str, *, contents: bool):
    indent = r"( +)" if contents else r"()"
    label = "Appendix " + key if len(key) == 1 and key.isalpha() else key
    pattern = re.compile(rf"^{indent}{re.escape(label)}\.[ \t]+(\S[^\n]*)", re.MULTILINE)
    headings = []
    for match in pattern.finditer(text):
        title, end = match[2], match.end()
        while (line := _CONTINUATION.match(text, end)) and len(line[1]) > len(match[1]):
            if _HEADING.match(line[2]):
                break
            title += ("" if title.endswith("-") else " ") + line[2]
            end = line.end()
        headings.append((match.start(), end, " ".join(title.split())))
    return headings


def _section_basis(toc: str, body: str, key: str, start: int, title: str) -> str | None:
    listed = _headings(toc, key, contents=True)
    if listed:
        return "contents_and_body" if len(listed) == 1 and listed[0][2] == title else None
    # Deep headings can be omitted from the printed table. Require the whole
    # enclosing number chain, and a unique body/contents match for an ancestor.
    parts = key.split(".")
    if len(parts) == 1:
        return None
    enclosing = []
    for match in _BODY_LABEL.finditer(body, 0, start):
        ancestor = match[1]
        while enclosing and enclosing[-1].count(".") >= ancestor.count("."):
            enclosing.pop()
        enclosing.append(ancestor)
    # The preceding heading may be a sibling or one of that sibling's children.
    enclosing = [ancestor for ancestor in enclosing if ancestor.count(".") < len(parts) - 1]
    if enclosing != [".".join(parts[:n]) for n in range(1, len(parts))]:
        return None
    for ancestor in reversed(enclosing):
        printed = _headings(body, ancestor, contents=False)
        if len(printed) != 1:
            return None
        listed = _headings(toc, ancestor, contents=True)
        if listed:
            if len(listed) == 1 and listed[0][2] == printed[0][2]:
                return "body_under_verified_parent"
            return None
    return None


def _opening(text: str, key: str):
    toc = re.search(r"(?m)^Table of Contents[ \t]*$", text)
    if not toc:
        return None
    first = re.compile(r"(?m)^1\.[ \t]+\S").search(text, toc.end())
    if not first:
        return None
    body = text[first.start() :]
    headings = _headings(body, key, contents=False)
    if len(headings) != 1:
        return None
    start, end, title = headings[0]
    basis = _section_basis(text[toc.end() : first.start()], body, key, start, title)
    if basis is None:
        return None
    paragraph = _PARAGRAPH.match(body, end)
    return (
        first.start() + start,
        first.start() + (paragraph.end() if paragraph else end),
        title,
        "opening_paragraph" if paragraph else "heading_only",
        basis,
    )


def rfc_section_evidence(source: dict, entry: int, number: str, key: str, limit: int) -> dict:
    """Confirm the opening against the contents, derivative and exact original text bytes."""
    if not 160 <= limit <= 2000:
        raise ValueError("invalid source snippet size")
    raw = source.get("publisher_text_bytes")
    if not raw or not rfc_header_matches(raw, number):
        raise ValueError("publisher text does not identify the requested RFC")
    text = source["derived_text"]
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != source["derived_sha256"]:
        raise ValueError("RFC derivative differs from its recorded revision")
    marker = _UNIT.search(text)
    if not marker or hashlib.sha256(raw).hexdigest() != source["original_sha256"]:
        raise ValueError("RFC publisher revision or plain-text extraction marker is missing")
    body = text[marker.end() :]
    if body != raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").strip() + "\n":
        raise ValueError("RFC derivative does not reproduce the publisher text")
    lookup = {"rfc": number, "section": key, "status": "section_not_verified"}
    rows = []
    opening = _opening(body, key)
    if opening:
        start, end, title, scope, basis = opening
        start += marker.end()
        end += marker.end()
        excerpt_end = min(start + limit, end)
        excerpt = text[start:excerpt_end]
        rows.append(
            {
                "original_path": source["original_path"],
                "filesystem_entry_id": entry,
                "original_sha256": source["original_sha256"],
                "derived_sha256": source["derived_sha256"],
                "source_section": f"RFC {number}, Section {key}. {title}",
                "source_start_char": start,
                "source_end_char": end,
                "source_span_sha256": hashlib.sha256(text[start:end].encode()).hexdigest(),
                "excerpt_start_char": start,
                "excerpt_end_char": excerpt_end,
                "excerpt_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
                "source_excerpt": excerpt,
                "status": "verified_rfc_section_opening",
            }
        )
        attach_publisher_excerpts(raw, source["original_sha256"], text, rows, body_start=marker.end())
        if "publisher_excerpt" not in rows[0]:
            raise ValueError("RFC section could not be reproduced in the publisher original")
        lookup.update(status="verified", title=title, scope=scope, basis=basis, truncated=excerpt_end < end)
    return {
        "original_path": source["original_path"],
        "original_sha256": source["original_sha256"],
        "derived_sha256": source["derived_sha256"],
        "rfc_section_lookup": lookup,
        "passages": rows,
        "returned_passages": len(rows),
        "source_coordinates": "zero_based_half_open_character_offsets_in_verified_derivative",
        "evidence_warning": "Exact section lookup returns its heading and opening paragraph when present. Later paragraphs can add qualifications.",
    }
