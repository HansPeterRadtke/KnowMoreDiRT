"""No-network original physical PDF alternate glyph candidate provenance and guards."""
from __future__ import annotations
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from file_system_catalog.alternate_pdf_page_text import (
    quality_of_alternate_page,extract_alternate_original_pdf_pages,
)

WORDS=('The original publisher source physically documents a long bounded factual passage '
       'but a machine extraction cannot independently establish visual reading order and '
       'diagram context or factual interpretation in a scientific benchmark result. '
       'This sentence is repeated for stable unit test length and lexical coverage.\n')
TEXT=(WORDS*3).encode()

class AlternatePdfPageTextTests(unittest.TestCase):
    def setUp(self):
        d=TemporaryDirectory();self.addCleanup(d.cleanup)
        self.path=Path(d.name)/'original.pdf';self.path.write_bytes(b'%PDF-not-a-real-PDF; approved original test')
        self.sha=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.calls=[]

    def pdfinfo(self,args,fd,deadline):
        self.calls.append(('info',args,fd))
        self.assertEqual('/usr/bin/pdfinfo',args[0])
        self.assertTrue(args[-1].startswith('/proc/self/fd/'))
        return b'Pages: 3\n'

    def page_reader(self,fd,page,deadline):
        self.calls.append(('read',page,fd))
        self.assertIn(page,(1,2,3))
        return TEXT

    def check(self,**kw):
        return extract_alternate_original_pdf_pages(self.path,self.sha,pages=[1,2],
               expected_physical_page_count=3,page_reader=self.page_reader,
               pdfinfo_reader=self.pdfinfo,**kw)

    def test_physically_attested_original_pages_and_SHA_bound_to_same_open_file(self):
        x=self.check()
        self.assertEqual([1,2],[r['physical_pdf_page'] for r in x['pages']])
        self.assertTrue(x['original_pdf_sha_verified_before_and_after_same_file_description'])
        self.assertTrue(x['all_page_numbers_from_original_pdfinfo_not_fake_derivative_labels'])
        self.assertTrue(x['machine_extracted_text_is_provisional_not_visually_verified'])
        self.assertEqual(self.sha,x['original_pdf_sha256'])
        for row in x['pages']:
            self.assertEqual('source_bounded_provisional_machine_text',row['status'])
            self.assertEqual(TEXT.decode(),row['text'])
            self.assertFalse(row['publisher_visual_or_semantic_text_fidelity_verified'])
            self.assertTrue(row['source_page_content_not_an_authoritative_quote'])
            self.assertEqual(hashlib.sha256(TEXT).hexdigest(),row['raw_machine_text_sha256'])
        self.assertEqual(3,len(self.calls))
        # The PDF descriptor passed to both extractor invocations is the
        # same descriptor that pdfinfo inspected.
        self.assertEqual(self.calls[0][-1],self.calls[1][-1]);self.assertEqual(self.calls[0][-1],self.calls[2][-1])

    def test_text_with_one_NUL_or_replacement_is_not_a_verified_quote(self):
        for bad in (b'\0'+TEXT,TEXT+b'\xef\xbf\xbd',b'\x00'*34,\
                    b'',None,b'\xffbad encoding',b'only ten words are available'):
            with self.subTest(bad=bad[:20] if isinstance(bad,bytes) else bad):
                self.assertEqual('unavailable',quality_of_alternate_page(bad)['status'])

    def test_long_horizontally_padded_source_retains_every_original_word_and_line(self):
        raw=b'Original source condition.\n'+b' '*19100+TEXT
        x=quality_of_alternate_page(raw)
        self.assertEqual('source_bounded_provisional_machine_text',x['status'])
        self.assertIn('horizontal_padding_compacted',x['layout_policy'])
        self.assertEqual(''.join(x['text'].split()),''.join(raw.decode().split()))
        self.assertEqual(x['text'].count('\n'),raw.decode().count('\n'))
        self.assertLess(x['display_character_count'],x['raw_character_count'])
        self.assertFalse(x['publisher_visual_or_semantic_text_fidelity_verified'])

    def test_overlong_uncompacted_text_is_rejected_with_no_truncation(self):
        raw=b'OriginalVisualWord'*2000+b' '+TEXT
        x=quality_of_alternate_page(raw)
        self.assertEqual('unavailable',x['status'])
        self.assertIn('oversized',x['reason'])
        self.assertNotIn('text',x)

    def test_page_count_rejects_wrong_original_physical_pdf_metadata(self):
        with self.assertRaisesRegex(ValueError,'physical PDF page count differs'):
            extract_alternate_original_pdf_pages(self.path,self.sha,pages=[1],
                 expected_physical_page_count=3,page_reader=self.page_reader,
                 pdfinfo_reader=lambda *_:b'Pages: 2\n')

    def test_bad_page_or_count_never_renumbered_or_silently_clamped(self):
        for pages in ([0],[4],[1,1],[2,1],[True],list(range(1,34))):
            with self.subTest(pages=pages),self.assertRaises(ValueError):
                extract_alternate_original_pdf_pages(self.path,self.sha,pages=pages,
                     expected_physical_page_count=3,page_reader=self.page_reader,
                     pdfinfo_reader=self.pdfinfo)
        with self.assertRaises(ValueError):
            self.check(max_pages=1)

    def test_original_publisher_pdf_hash_mismatch_refused_before_parser(self):
        with self.assertRaisesRegex(ValueError,'approved SHA-256'):
            extract_alternate_original_pdf_pages(self.path,'0'*64,pages=[1],
                expected_physical_page_count=3,page_reader=self.page_reader,
                pdfinfo_reader=self.pdfinfo)
        self.assertEqual([],self.calls)

    def test_source_original_pdf_symlink_rejected_even_with_right_hash(self):
        alias=self.path.parent/'alias.pdf';alias.symlink_to(self.path)
        with self.assertRaises(OSError):
            extract_alternate_original_pdf_pages(alias,self.sha,pages=[1],
                expected_physical_page_count=3,page_reader=self.page_reader,
                pdfinfo_reader=self.pdfinfo)

    def test_mutated_original_during_machine_extraction_refused_afterward(self):
        def changing_reader(fd,page,deadline):
            self.path.write_bytes(b'changed publisher original after the hash')
            return TEXT
        with self.assertRaisesRegex(ValueError,'changed during alternate extraction'):
            extract_alternate_original_pdf_pages(self.path,self.sha,pages=[1],
                expected_physical_page_count=3,page_reader=changing_reader,
                pdfinfo_reader=self.pdfinfo)

    def test_atomically_replaced_original_after_open_is_also_refused(self):
        def changing_reader(fd,page,deadline):
            another=self.path.parent/'replacement.pdf'
            another.write_bytes(b'replaced while the old file descriptor stays valid')
            another.replace(self.path)
            return TEXT
        with self.assertRaisesRegex(ValueError,'changed during alternate extraction'):
            extract_alternate_original_pdf_pages(self.path,self.sha,pages=[1],
                expected_physical_page_count=3,page_reader=changing_reader,
                pdfinfo_reader=self.pdfinfo)

    def test_failed_page_extraction_is_not_fake_or_partial_source_text(self):
        def unavailable_reader(fd,page,deadline):return None
        x=extract_alternate_original_pdf_pages(self.path,self.sha,pages=[1],
                expected_physical_page_count=3,page_reader=unavailable_reader,
                pdfinfo_reader=self.pdfinfo)
        self.assertEqual('unavailable',x['pages'][0]['status'])
        self.assertNotIn('text',x['pages'][0])

if __name__=='__main__':unittest.main()
