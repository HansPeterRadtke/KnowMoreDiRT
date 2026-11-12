"""Physical page provenance is independent of derivative page labels."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from file_system_catalog.publisher_pdf_page import locate_page_excerpts, verify_pdf_page_excerpts


def hit(label, text):
    return {"source_section": label, "source_excerpt": text}


class PhysicalPdfPageTests(unittest.TestCase):
    def test_correction_is_unique_across_the_whole_pdf_and_preserves_derivative(self):
        src = [hit("PDF page 4", "Česká  measurement\n= 3.5%"),
               hit("PDF page 1 | Chapter 1", "First page")]
        before = [row.copy() for row in src]
        pages = {1: b"First page\n", 2: "One Česká\n measurement = 3.5%.\f".encode(), 3: b"Other"}
        info = locate_page_excerpts(src, len(pages), pages.get)
        proof = info[0]
        self.assertEqual((proof["status"], proof["page"], proof["basis"]), ("verified", 2, "unique_across_pdf"))
        self.assertEqual(proof["page_text_sha256"], hashlib.sha256(pages[2]).hexdigest())
        normalized = " ".join(pages[2].decode().split())
        self.assertEqual(normalized[proof["start_char"]:proof["end_char"]], "Česká measurement = 3.5%")
        self.assertEqual(info[1]["basis"], "derivative_label_confirmed")
        self.assertEqual(src, before)

    def test_ambiguous_absent_and_ocr_text_do_not_gain_a_physical_page(self):
        src = [hit("PDF page 5", "shared text"), hit("PDF page 5", "Shared text"),
               hit("OCR PDF page 2", "unique token"), hit("PDF page 1", "    ")]
        pages = {1:b"shared text", 2:b"shared text and unique token"}
        info = locate_page_excerpts(src, 2, pages.get)
        self.assertEqual([v["reason"] for v in info],
                         ["multiple_matching_pages", "excerpt_absent", "no_text_page_label", "empty_excerpt"])
        self.assertTrue(all(v["status"]=="unverified" and "page" not in v for v in info))

    def test_unreadable_page_prevents_unique_correction_but_preserves_direct_match(self):
        pages={1:b"Wrong label", 2:b"Right words", 3:None}
        info=locate_page_excerpts([hit("PDF page 1","Right words"),hit("PDF page 2","Right words")],3,pages.get)
        self.assertEqual(info[0]["reason"], "incomplete_page_scan")
        self.assertEqual(info[1]["page"], 2)
        self.assertEqual(locate_page_excerpts([hit("PDF page 7","Text")],None,pages.get)[0]["reason"],
                         "page_count_unavailable")

    def test_only_requested_pages_open_when_labels_match_or_book_is_long(self):
        calls=[]
        def read(n):
            calls.append(n)
            return b"Page contains word and another matching excerpt"
        src=[hit("PDF page 2, continued section 1 [part 3]","contains"),
             hit("PDF page 2 [source offset 700]","another matching excerpt")]
        proofs=locate_page_excerpts(src,100,read)
        self.assertEqual(calls,[2])
        self.assertTrue(all(v["page"]==2 for v in proofs))
        calls.clear()
        miss=locate_page_excerpts([hit("PDF page 6","absent")],100,read)[0]
        self.assertEqual((calls,miss["reason"]),([6],"relocation_limit"))

    def test_unreadable_nominated_page_in_long_book_reports_extraction_failure(self):
        proof = locate_page_excerpts([hit("PDF page 49", "text")], 70, lambda page: b"\xff")[0]
        self.assertEqual(proof, {"status": "unverified", "reason": "page_extraction_unavailable"})

    def test_unavailable_temporary_storage_keeps_excerpts_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.pdf"
            path.write_bytes(b"fixture")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with mock.patch("file_system_catalog.publisher_pdf_page.tempfile.TemporaryFile", side_effect=OSError):
                proofs = verify_pdf_page_excerpts(path, digest, [hit("PDF page 1", "sample text")])
            self.assertEqual(proofs, [{"status": "unverified", "reason": "page_count_unavailable"}])

    def test_mutation_during_extraction_invalidates_already_located_excerpt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.pdf"
            path.write_bytes(b"original")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()

            def replace_after_hash(args, fd, deadline):
                if args[0].endswith("/pdfinfo"):
                    return b"Pages: 1\n"
                replacement = Path(tmp) / "replacement.pdf"
                replacement.write_bytes(b"replaced")
                replacement.replace(path)
                return b"Requested excerpt"

            with mock.patch("file_system_catalog.publisher_pdf_page._poppler", side_effect=replace_after_hash):
                with self.assertRaisesRegex(ValueError, "changed during"):
                    verify_pdf_page_excerpts(path, expected, [hit("PDF page 1", "Requested excerpt")])

    def test_changed_file_and_symlink_do_not_receive_proofs(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf=Path(tmp)/"paper.pdf"; pdf.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"revision"):
                verify_pdf_page_excerpts(pdf,"0"*64,[hit("PDF page 1","text")])
            link=Path(tmp)/"link.pdf"; link.symlink_to(pdf)
            with self.assertRaises(OSError):
                verify_pdf_page_excerpts(link,hashlib.sha256(pdf.read_bytes()).hexdigest(),[hit("PDF page 1","text")])

    @unittest.skipUnless(all(shutil.which(v) for v in ("pdftotext","pdfinfo","prlimit")),"Poppler and prlimit required")
    def test_real_operator_reproduces_page_of_repository_paper(self):
        root=Path(__file__).resolve().parents[2]
        pdf=root/"docs/papers/bos-2008-wide-coverage-semantic-analysis-with-boxer.pdf"
        raw=subprocess.check_output(["pdftotext","-f","1","-l","1","-layout","-enc","UTF-8",str(pdf),"-"])
        excerpt=" ".join(raw.decode().split())[:140]
        proof=verify_pdf_page_excerpts(pdf,hashlib.sha256(pdf.read_bytes()).hexdigest(),
                                       [hit("PDF page 1",excerpt)])[0]
        self.assertEqual(proof["page"],1)
        self.assertEqual(proof["page_text_sha256"],hashlib.sha256(raw).hexdigest())
