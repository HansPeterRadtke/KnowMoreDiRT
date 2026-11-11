"""Passage verification uses character offsets over the exact UTF-8 derivative."""
import hashlib
import random
import unittest

from file_system_catalog.source_passage_digests import source_passage_digests


class PassageDigestTests(unittest.TestCase):
    def test_overlapping_unicode_spans_and_clipped_invalid_endpoints(self):
        source = "ä\r\ne\u0301 → 🛰️\n日本語" * 17
        raw = source.encode()
        rng = random.Random(41)
        intervals = [(0, len(source)), (0, 1), (1, len(source) + 1), (-1, 4), (3, 3),
                     (True, 7), (len(source) - 1, len(source))] + [
            tuple(sorted(rng.sample(range(len(source) + 1), 2))) for _ in range(64)]
        expected = {(i, j): hashlib.sha256(source[i:j].encode()).hexdigest()
                    for i, j in intervals if type(i) is int and 0 <= i < j <= len(source)}
        for block_size in (1, 2, 7, 19, len(raw)):
            with self.subTest(block_size=block_size):
                blocks = (raw[i:i + block_size] for i in range(0, len(raw), block_size))
                full, spans = source_passage_digests(blocks, intervals + intervals)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), full)
                self.assertEqual(expected, spans)

    def test_invalid_utf8_in_or_after_matched_region_fails_the_whole_revision(self):
        for blocks in ([b"body", b"\xf0\x9f"], [b"body", b"\xff"]):
            with self.subTest(blocks=blocks), self.assertRaises(UnicodeDecodeError):
                source_passage_digests(blocks, [(0, 4)])

    def test_empty_source_and_nonempty_out_of_bounds_spans_do_not_create_anchors(self):
        for blocks in ([], [b""]):
            self.assertEqual(source_passage_digests(blocks, [(0, 1), (0, 0)]),
                             (hashlib.sha256(b"").hexdigest(), {}))
