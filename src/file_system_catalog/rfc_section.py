"""Offline RFC identifiers and exact, publisher-anchored section openings."""

from __future__ import annotations

import hashlib
import json
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
_BLANK_GAP = re.compile(r"\n(?:[ \t]*\n)+")
_FIGURE_CAPTION = re.compile(r"[ \t]*Figure ([1-9][0-9]*)(?::[^\n]+)?[ \t]*(?=\n|\Z)")
_PARTICIPANTS = re.compile(r"[A-Za-z][\w-]*(?:[ \t]{2,}[A-Za-z][\w-]*)+")
_PHASE_ROW = re.compile(r"[A-Za-z][\w-]*[ \t]+[|^v][ \t]+\S")
_PHASE_MARKER = re.compile(r"(?:^|[ \t])[|^v](?:[ \t]|$)")
_FLOW_ARROW = re.compile(r"<[-=]{2,}>?|[-=]{2,}>")
_LEGEND = re.compile(r"([ \t]+)(?:[+*]|\{\}|\[\])[ \t]+\S[^\n]*(?:\n\1[ \t]+\S[^\n]*)*")
_DISPLAY_FOLD = re.compile(r"(?m)^([^\n]{72})\n")
_BOX_EDGE = re.compile(r"[+.'`][-+=]{2,}[+.'`]")
_DOUBLE_ARROW = re.compile(r"<[-=]{2}[^<>]*[-=]{2}>")
_BOX_BORDER = re.compile(r"\+(?:[-=]+\+)+[ \t]*")
_MARGIN_LINE = re.compile(r"(?m)^([^ \t\n][^\n]*)")
_CONTENTS = re.compile(r"(?m)^Table of Contents[ \t]*$")
_FIRST_BODY = re.compile(r"(?m)^1\.[ \t]+\S")
_CURSOR = re.compile(r"([1-9][0-9]{0,8})\.[0-9a-f]{64}")
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
    toc = _CONTENTS.search(text)
    if not toc:
        return None
    first = _FIRST_BODY.search(text, toc.end())
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


def _diagram_row(line: str) -> bool:
    if line.count("|") + line.count("║") >= 2 or _BOX_BORDER.fullmatch(line) or _DOUBLE_ARROW.fullmatch(line):
        return True
    # Parallel boxes can share a row with arrows or labels between their borders.
    edge = _BOX_EDGE.match(line)
    return edge is not None and _BOX_EDGE.search(line, edge.end()) is not None


def _diagram_block(block: str) -> bool:
    lines = block.split("\n")
    return any(_diagram_row(line.strip(" \t")) for line in lines) and all(
        _diagram_row(line.rstrip()) for line in lines if line and line[0] not in " \t"
    )


def _invalid_json_constant(value: str):
    raise ValueError(f"non-JSON constant {value}")


def _json_figure(block: str) -> bool:
    # The RFC text renderer can hard-wrap long encoded strings at column 72.
    # A space at the physical fold permits a syntax check without fusing tokens.
    # Offsets, digests and excerpts always use the unmodified publisher bytes.
    try:
        json.loads(_DISPLAY_FOLD.sub(r"\1 ", block), parse_constant=_invalid_json_constant)
    except (ValueError, RecursionError):
        return False
    return True


def _preceding_paragraph(body: str, start: int) -> tuple[int, str]:
    """Return the immediately preceding block only across a blank line."""
    pos = start
    while pos:
        begin = body.rfind("\n", 0, pos - 1) + 1
        if body[begin : pos - 1].strip(" \t"):
            break
        pos = begin
    if pos == start or pos == 0:
        return pos, ""
    end = pos - 1
    while pos:
        begin = body.rfind("\n", 0, pos - 1) + 1
        if not body[begin : pos - 1].strip(" \t"):
            break
        pos = begin
    return pos, body[pos:end]


def _printed_response(block: str) -> bool:
    # These are display boundaries. The indented payload can be diagnostic notation.
    lines = block.split("\n")
    return (
        len(lines) >= 6
        and re.fullmatch(r"Header: \S[^\n]* \(Code=[0-9]\.[0-9]{2}\)", lines[0]) is not None
        and re.fullmatch(r"Content-Format: [0-9]+ \([^()\s]+\)", lines[1]) is not None
        and lines[2:4] == ["Payload:", "{"]
        and lines[-1] == "}"
        and all(line[:1] in (" ", "\t") for line in lines[4:-1])
    )


def _flow_row(line: str, phases: bool) -> bool:
    if _FLOW_ARROW.search(line):
        return True
    if not phases:
        return False
    if line.startswith((" ", "\t")):
        return _PHASE_MARKER.search(line) is not None
    return _PHASE_ROW.match(line) is not None


def _referenced_figure_end(body: str, first_row: re.Match, contents: set[str]) -> int | None:
    row = first_row[1].rstrip()
    response = row == "Response:"
    phases = _PHASE_ROW.match(row) is not None
    if not (response or phases or _PARTICIPANTS.fullmatch(row)):
        return None
    start = first_row.start()
    before, intro = _preceding_paragraph(body, start)
    if phases:
        if not intro.startswith((" ", "\t")) or not _PARTICIPANTS.fullmatch(intro.strip(" \t")):
            return None
        _, intro = _preceding_paragraph(body, before)
    numbers = set(re.findall(r"\bFigure[ \t]+([1-9][0-9]*)\b(?!\.[0-9])", intro))
    if len(numbers) != 1 or not all(line.startswith((" ", "\t")) for line in intro.splitlines()):
        return None

    pos = start
    left = right = payload = legend = False
    while gap := _BLANK_GAP.search(body, pos):
        block = body[pos : gap.start()]
        if _BODY_LABEL.search(block) or any(
            line[1].rstrip() in contents for line in _MARGIN_LINE.finditer(block)
        ):
            return None
        if pos == start and not phases:
            if block.rstrip(" \t") != row:  # A participant header or response marker stands alone.
                return None
        elif response:
            if payload or not _printed_response(block):
                return None
            payload = True
        elif _LEGEND.fullmatch(block):
            if not (left and right):
                return None
            legend = True
        else:
            arrows = _FLOW_ARROW.findall(block)
            if legend or not arrows or not all(_flow_row(line, phases) for line in block.splitlines()):
                return None
            left |= any(arrow.startswith("<") for arrow in arrows)
            right |= any(arrow.endswith(">") for arrow in arrows)
        pos = gap.end()
        if caption := _FIGURE_CAPTION.match(body, pos):
            if (
                caption[1] in numbers
                and caption.group().strip() not in contents
                and (payload or (left and right))
            ):
                return caption.end()
            return None
    return None


def _caption_after_figure(body: str, first_row: re.Match, contents: set[str]) -> int | None:
    start = first_row.start()
    json_block = first_row[1].rstrip() in ("{", "[")
    if not json_block and not _diagram_row(first_row[1]):
        return _referenced_figure_end(body, first_row, contents)
    pos = start
    while gap := _BLANK_GAP.search(body, pos):
        # Never consume a numbered division or a back-matter heading printed in the contents.
        if _BODY_LABEL.search(body, pos, gap.start()) or any(
            row[1].rstrip() in contents for row in _MARGIN_LINE.finditer(body, pos, gap.start())
        ):
            return None
        if not json_block and not _diagram_block(body[pos : gap.start()]):
            return None
        caption = _FIGURE_CAPTION.match(body, gap.end())
        if caption:
            if caption.group().strip() in contents or (
                json_block and not _json_figure(body[start : gap.start()])
            ):
                return None
            return caption.end()
        pos = gap.end()
    return None


def _context_extent(body: str, opening_end: int) -> tuple[int, str]:
    """Limit context to this heading's own text, including verified captioned artwork."""
    line = _MARGIN_LINE.search(body, opening_end)
    contents = set()
    if line and not _BODY_LABEL.match(body, line.start()):
        toc = _CONTENTS.search(body)
        first = _FIRST_BODY.search(body, toc.end())
        contents = {
            item.strip() for item in body[toc.end() : first.start()].splitlines() if item[:1] in (" ", "\t")
        }
    while line:
        if _BODY_LABEL.match(body, line.start()):
            boundary = "numbered_heading"
            break
        if line[1].rstrip() in contents:
            boundary = "contents_heading"
            break
        caption_end = _caption_after_figure(body, line, contents)
        if caption_end is None:
            boundary = "unclassified_margin"
            break
        line = _MARGIN_LINE.search(body, caption_end)
    else:
        boundary = "end_of_text"
    stop = line.start() if line else len(body)
    return opening_end + len(body[opening_end:stop].rstrip(" \t\n")), boundary


def _cursor(source: dict, number: str, key: str, offset: int) -> str:
    parts = (
        "rfc-context-v1",
        source["original_path"],
        source["original_sha256"],
        source["derived_sha256"],
        number,
        key,
        str(offset),
    )
    return f"{offset}." + hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def rfc_section_evidence(
    source: dict, entry: int, number: str, key: str, limit: int, *, cursor: str | None = None
) -> dict:
    """Verify a section opening or its next contiguous window against both source revisions."""
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
    if cursor is not None and opening is None:
        raise ValueError("continuation requires the same verified RFC section")
    if opening:
        heading_start, opening_end, title, scope, basis = opening
        context_end, boundary = _context_extent(body, opening_end)
        length = context_end - heading_start
        offset = 0
        span_end = opening_end
        if cursor is not None:
            token = _CURSOR.fullmatch(cursor)
            if not token:
                raise ValueError("invalid RFC section continuation cursor")
            offset = int(token[1])
            if offset >= length or cursor != _cursor(source, number, key, offset):
                raise ValueError("continuation cursor does not match the RFC section and source revision")
            scope = "section_continuation"
            span_end = context_end
        start = marker.end() + heading_start
        end = marker.end() + span_end
        excerpt_start = start + offset
        excerpt_end = min(excerpt_start + limit, end)
        excerpt = text[excerpt_start:excerpt_end]
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
                "excerpt_start_char": excerpt_start,
                "excerpt_end_char": excerpt_end,
                "excerpt_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
                "source_excerpt": excerpt,
                "status": "verified_rfc_section_continuation"
                if cursor is not None
                else "verified_rfc_section_opening",
            }
        )
        attach_publisher_excerpts(raw, source["original_sha256"], text, rows, body_start=marker.end())
        if "publisher_excerpt" not in rows[0]:
            raise ValueError("RFC section could not be reproduced in the publisher original")
        next_offset = excerpt_end - start
        context = {
            "start_offset": offset,
            "end_offset": next_offset,
            "length_characters": length,
            "boundary": boundary,
            "next_cursor": _cursor(source, number, key, next_offset) if next_offset < length else None,
        }
        lookup.update(
            status="verified",
            title=title,
            scope=scope,
            basis=basis,
            truncated=excerpt_end < end,
            context=context,
        )
    return {
        "original_path": source["original_path"],
        "original_sha256": source["original_sha256"],
        "derived_sha256": source["derived_sha256"],
        "rfc_section_lookup": lookup,
        "passages": rows,
        "returned_passages": len(rows),
        "source_coordinates": "zero_based_half_open_character_offsets_in_verified_derivative",
        "evidence_warning": (
            "Later paragraphs can qualify the opening. Follow context.next_cursor to read the source in order. "
            "If the boundary is unclassified_margin, inspect the original for the rest of the section."
        ),
    }
