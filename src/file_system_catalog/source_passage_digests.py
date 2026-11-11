"""Hash UTF-8 source intervals, in character coordinates, in one bounded-memory pass."""
from __future__ import annotations

import codecs
from collections import deque
from collections.abc import Iterable
import hashlib


def source_passage_digests(blocks: Iterable[bytes], intervals: Iterable[tuple[int, int]]):
    """Return the complete byte digest and the digests of fully covered nonempty intervals."""
    pending = deque(sorted({(a, b) for a, b in intervals
                            if type(a) is int and type(b) is int and 0 <= a < b}))
    full = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    active = {}
    finished = {}
    offset = 0
    for block in blocks:
        full.update(block)
        text = decoder.decode(block)
        end = offset + len(text)
        while pending and pending[0][0] < end:
            active[pending.popleft()] = hashlib.sha256()
        completed = []
        for interval, digest in active.items():
            lo, hi = interval
            piece = text[max(0, lo - offset):min(len(text), hi - offset)]
            digest.update(piece.encode("utf-8"))
            if hi <= end:
                finished[interval] = digest.hexdigest()
                completed.append(interval)
        for interval in completed:
            del active[interval]
        offset = end
    decoder.decode(b"", final=True)  # Reject a truncated multibyte character at EOF.
    return full.hexdigest(), finished
