"""Map verified plain-text excerpts back to byte ranges in the publisher's UTF-8 file."""
from __future__ import annotations

import hashlib
import re


def _byte_offsets(original: str, normalized: str, positions: set[int]) -> dict[int, int]:
    # Each CRLF is one extra byte in the publisher file. A lone CR has the same
    # encoded width as LF. Walk each source segment and CRLF occurrence once.
    crlfs = (match.start() - index for index, match in enumerate(re.finditer("\r\n", original)))
    next_crlf = next(crlfs, None)
    previous = byte_position = 0
    offsets = {}
    for position in sorted(positions):
        byte_position += len(normalized[previous:position].encode("utf-8"))
        while next_crlf is not None and next_crlf < position:
            byte_position += 1
            next_crlf = next(crlfs, None)
        offsets[position] = byte_position
        previous = position
    return offsets


def attach_publisher_excerpts(raw: bytes, original_sha256: str, derived_text: str, passages: list[dict],
                              *, body_start: int) -> None:
    """Add raw coordinates only after the entire extraction matches the original plain text."""
    if hashlib.sha256(raw).hexdigest() != original_sha256:
        raise ValueError("publisher bytes differ from the verified revision")
    try:
        original = raw.decode("utf-8")
    except UnicodeDecodeError:
        return
    normalized = original.replace("\r\n", "\n").replace("\r", "\n")
    core = normalized.strip()
    if not core or derived_text[body_start:] != core + "\n":
        return
    prefix = len(normalized) - len(normalized.lstrip())
    extents = []
    for row in passages:
        start = row["excerpt_start_char"] - body_start
        end = row["excerpt_end_char"] - body_start
        if not 0 <= start < end <= len(core) + 1:
            continue
        # Extraction appends LF even when the publisher ends at the last character.
        # Drop that synthetic character only if the original cannot reproduce it.
        if end > len(core) and normalized[prefix + len(core):prefix + end] != "\n":
            end -= 1
        if start < end:
            extents.append((row, prefix + start, prefix + end))
    offsets = _byte_offsets(original, normalized, {pos for _, lo, hi in extents for pos in (lo, hi)})
    for row, lo, hi in extents:
        start_byte, end_byte = offsets[lo], offsets[hi]
        fragment = raw[start_byte:end_byte]
        excerpt = fragment.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        derived_start = row["excerpt_start_char"]
        derived_end = derived_start + len(excerpt)
        if derived_text[derived_start:derived_end] != excerpt:
            raise ValueError("publisher interval does not reproduce the derivative excerpt")
        row.update(source_excerpt=excerpt, excerpt_end_char=derived_end,
                   excerpt_sha256=hashlib.sha256(excerpt.encode("utf-8")).hexdigest())
        row["publisher_excerpt"] = {
            "start_byte": start_byte, "end_byte": end_byte,
            "sha256": hashlib.sha256(fragment).hexdigest(),
            "encoding": "utf-8", "newline_normalization": "universal",
        }
