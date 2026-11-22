"""Exact physical page neighborhoods preserve provenance and fail closed."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from file_system_catalog.verified_pdf_neighbor_context import verified_page_neighbors


class VerifiedPdfPhysicalNeighborsTests(unittest.TestCase):
    def setUp(self):
        temporary=TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.pdf=self.root/'source.pdf'
        self.pdf.write_bytes(b'Synthetic source PDF bytes for physical page test; not publisher text')
        self.sha=hashlib.sha256(self.pdf.read_bytes()).hexdigest()
        self.pages={1:b'FIRST: preceding page contains a hypothetical, not an official legal statement.\n',
                    2:b'SECOND: The example asks When was Scott Derrickson Born?\n',
                    3:b'THIRD: a following page qualifies the question as a benchmark input.\n'}
        self.calls=[]

    def poppler(self,cmd,fd,deadline):
        self.calls.append(cmd)
        if cmd[0]=='/usr/bin/pdfinfo':return b'Pages: 3\n'
        self.assertEqual('/usr/bin/pdftotext',cmd[0])
        page=int(cmd[cmd.index('-f')+1])
        self.assertEqual(str(page),cmd[cmd.index('-l')+1])
        return self.pages.get(page)

    def test_explicit_original_page_and_neighbors_are_separate_provenanced_sources(self):
        with mock.patch('file_system_catalog.verified_pdf_neighbor_context._poppler',side_effect=self.poppler):
            x=verified_page_neighbors(self.pdf,self.sha,requested_page=2,
               required_center_excerpt='The example asks When was Scott Derrickson Born?')
        self.assertEqual([1,2,3],[row['physical_page'] for row in x['pages']])
        self.assertTrue(x['original_pdf_revision_verified_before_and_after'])
        self.assertTrue(x['required_center_excerpt_verified'])
        self.assertTrue(x['separate_original_pages_not_joined_into_one_quotation'])
        self.assertTrue(x['context_cannot_by_itself_prove_a_factual_claim'])
        self.assertEqual(3,len(self.calls)-1)
        for page in x['pages']:
            self.assertEqual(page['physical_page_poppler_sha256'],
                             hashlib.sha256(self.pages[page['physical_page']]).hexdigest())
            self.assertEqual(self.pages[page['physical_page']].decode(),page['extracted_page_text'])
            self.assertFalse(page['factual_claim_truth_verified'])

    def test_first_last_page_radius_clamped_without_invented_page(self):
        with mock.patch('file_system_catalog.verified_pdf_neighbor_context._poppler',side_effect=self.poppler):
            x=verified_page_neighbors(self.pdf,self.sha,requested_page=1)
            self.assertEqual([1,2],[p['physical_page'] for p in x['pages']])
            y=verified_page_neighbors(self.pdf,self.sha,requested_page=3)
            self.assertEqual([2,3],[p['physical_page'] for p in y['pages']])
            z=verified_page_neighbors(self.pdf,self.sha,requested_page=3,neighbor_radius=0)
            self.assertEqual([3],[p['physical_page'] for p in z['pages']])

    def test_page_beyond_real_pdf_page_count_rejected_not_relocated(self):
        with mock.patch('file_system_catalog.verified_pdf_neighbor_context._poppler',side_effect=self.poppler):
            with self.assertRaisesRegex(ValueError,'outside actual original physical page count'):
                verified_page_neighbors(self.pdf,self.sha,requested_page=4)
        self.assertEqual(1,len(self.calls))

    def test_source_query_not_on_original_physical_page_refused(self):
        with mock.patch('file_system_catalog.verified_pdf_neighbor_context._poppler',side_effect=self.poppler):
            with self.assertRaisesRegex(ValueError,'required source excerpt absent'):
                verified_page_neighbors(self.pdf,self.sha,requested_page=2,
                   required_center_excerpt='the real-world birthdate is established')

    def test_ocr_only_page_cannot_become_a_citable_text_page(self):
        self.pages[2]=b'\x0c'
        with mock.patch('file_system_catalog.verified_pdf_neighbor_context._poppler',side_effect=self.poppler):
            with self.assertRaisesRegex(ValueError,'lacks usable text'):
                verified_page_neighbors(self.pdf,self.sha,requested_page=2)

    def test_no_silent_context_truncation_before_a_disclaimer(self):
        self.pages[2]=b'A'*2001+b'\nAfterward comes the source limitation and retraction.'
        with mock.patch('file_system_catalog.verified_pdf_neighbor_context._poppler',side_effect=self.poppler):
            with self.assertRaisesRegex(ValueError,'no silent truncation'):
                verified_page_neighbors(self.pdf,self.sha,requested_page=2,max_page_characters=2000)

    def test_corrupt_source_or_symlink_is_not_an_original(self):
        with self.assertRaisesRegex(ValueError,'SHA differs'):
            verified_page_neighbors(self.pdf,'0'*64,requested_page=1)
        link=self.root/'link.pdf';link.symlink_to(self.pdf)
        with self.assertRaises(OSError):verified_page_neighbors(link,self.sha,requested_page=1)

    def test_replaced_original_after_poppler_extract_never_returns_text(self):
        def mutate_on_page(cmd,fd,deadline):
            result=self.poppler(cmd,fd,deadline)
            if cmd[0]=='/usr/bin/pdftotext':
                new=self.root/'new.pdf';new.write_bytes(b'Replaced after original revision proof')
                new.replace(self.pdf)
            return result
        with mock.patch('file_system_catalog.verified_pdf_neighbor_context._poppler',side_effect=mutate_on_page):
            with self.assertRaisesRegex(ValueError,'changed during page extraction'):
                verified_page_neighbors(self.pdf,self.sha,requested_page=1,neighbor_radius=0)

    def test_invalid_anchoring_options_not_converted_to_default(self):
        for invalid in (0,True,-1,1000000):
            with self.subTest(page=invalid):
                with self.assertRaises(ValueError):
                    verified_page_neighbors(self.pdf,self.sha,requested_page=invalid)
        with self.assertRaises(ValueError):
            verified_page_neighbors(self.pdf,self.sha,requested_page=1,neighbor_radius=2)
        with self.assertRaises(ValueError):
            verified_page_neighbors(self.pdf,self.sha,requested_page=1,required_center_excerpt='a')

if __name__=='__main__':unittest.main()
