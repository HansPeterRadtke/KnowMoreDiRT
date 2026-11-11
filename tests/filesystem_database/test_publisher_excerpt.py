"""File byte offsets remain exact under UTF-8, mixed line endings, and extraction trimming."""
from hashlib import sha256
import unittest

from file_system_catalog.publisher_excerpt import attach_publisher_excerpts
from file_system_catalog.source_evidence import source_evidence_candidates


def digest(raw):
    return sha256(raw).hexdigest()


class PublisherCoordinateTests(unittest.TestCase):
    def test_raw_coordinates_distinguish_duplicate_unicode_excerpts_and_line_endings(self):
        for ending in ("", "\n", " \r\n", "\r", "\r\n"):
            original = ("  \r\n# Résumé\r\n\r\n## Réglage\rClé 北🌱 VALUE_KEY.\r\n"
                        "## Réglage\r\nClé 北🌱 VALUE_KEY." + ending)
            raw = original.encode()
            body = original.replace("\r\n", "\n").replace("\r", "\n").strip()
            source = "KMD source provenance header\n\n[[SOURCE UNIT: Original plain text]]\n" + body + "\n"
            results = source_evidence_candidates(
                original_path="guides/settings.md", original_title="Setup", filesystem_entry_id=8,
                original_sha256=digest(raw), derived_sha256=digest(source.encode()), derived_text=source,
                publisher_text_bytes=raw, query="VALUE_KEY", top=5, snippet_characters=200)
            with self.subTest(ending=ending):
                rows = results["passages"]
                self.assertEqual(2, len(rows))
                self.assertEqual(2, len({h["publisher_excerpt"]["start_byte"] for h in rows}))
                for row in rows:
                    a = row["publisher_excerpt"]
                    excerpt = raw[a["start_byte"]:a["end_byte"]]
                    self.assertEqual(digest(excerpt), a["sha256"])
                    visible = excerpt.decode().replace("\r\n", "\n").replace("\r", "\n")
                    self.assertEqual(source[row["excerpt_start_char"]:row["excerpt_end_char"]], visible)
                    self.assertEqual(row["source_excerpt"], visible)
                    self.assertEqual(row["excerpt_sha256"], digest(visible.encode()))
                    self.assertIn("北🌱", visible)

    def test_changed_text_or_lossy_decoding_never_claims_a_raw_coordinate(self):
        for raw in (b"edited UTF8 text", b"bad\xffsource"):
            rows = [{"excerpt_start_char": 0, "excerpt_end_char": 4}]
            attach_publisher_excerpts(raw, digest(raw), "body\n", rows, body_start=0)
            self.assertNotIn("publisher_excerpt", rows[0])
        with self.assertRaisesRegex(ValueError, "revision"):
            attach_publisher_excerpts(b"other", digest(b"publisher"), "body\n", [], body_start=0)

    def test_multiple_source_units_are_never_treated_as_one_plain_file(self):
        raw=b"one two\nthree four"
        src="KMD source provenance example\n[[SOURCE UNIT: Original plain text, continued section 1]]\none two\n"
        src+="[[SOURCE UNIT: Original plain text, continued section 2]]\nthree four\n"
        out=source_evidence_candidates(original_path="long.txt",original_title="Long",filesystem_entry_id=1,
            original_sha256=digest(raw),derived_sha256=digest(src.encode()),derived_text=src,
            publisher_text_bytes=raw,query="three four")
        self.assertEqual(1,out["returned_passages"])
        self.assertNotIn("publisher_excerpt",out["passages"][0])
