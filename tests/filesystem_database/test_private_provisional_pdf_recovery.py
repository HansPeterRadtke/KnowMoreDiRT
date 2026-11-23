"""No-live-write exact page preservation in private damaged-original PDF preview."""
from __future__ import annotations
import hashlib
import unittest
from file_system_catalog.private_provisional_pdf_recovery import assemble_unverified_original_pdf_glyph_preview
from file_system_catalog.alternate_pdf_page_text import quality_of_alternate_page
from file_system_catalog.alternate_pdf_page_search import exact_skipped_staged_physical_pages
from file_system_catalog.staged_physical_page_search import MARKER


class PrivateOriginalPdfGlyphPreviewTests(unittest.TestCase):
    def setUp(self):
        self.old=('KMD source provenance approved original PDF\n\n'
          '[[SOURCE UNIT: PDF page 1]]\nA confirmed neighboring page about a hypothetical situation.\n'
          '[[SOURCE UNIT: PDF page 2]]\n'+('X\x00'*220)+'\n'
          '[[SOURCE UNIT: PDF page 3]]\nThe next original publisher PDF page contains a limitation.\n')
        self.good=quality_of_alternate_page((
          'The original PDF physical page describes a benchmark with an experimental figure. '
          'The provisional alternate text should never become an authoritative full page quote. '
          'The original source has several paragraphs, scope qualifiers and explicit boundaries. '
          'This is a completely synthetic fixture for safe test assertions. '*3).encode())
        self.assertEqual('source_bounded_provisional_machine_text',self.good['status'])
        self.good['physical_pdf_page']=2

    def test_private_preview_restores_only_the_one_skipped_original_page(self):
        x=assemble_unverified_original_pdf_glyph_preview(self.old,
             physical_page_count=3,alternate_pages=[self.good])
        self.assertEqual([2],exact_skipped_staged_physical_pages(self.old,3,3))
        self.assertEqual([],exact_skipped_staged_physical_pages(x['preview_text'],3,3))
        self.assertEqual(1,x['provisional_alt_recovery_count'])
        self.assertEqual(3,x['new_temporary_searchable_original_physical_pages'])
        self.assertEqual(2,x['prior_temporary_searchable_original_physical_pages'])
        self.assertTrue(x['all_original_page_count_and_label_order_preserved'])
        self.assertTrue(x['unchanged_original_physical_page_bodies_exactly_preserved'])
        self.assertTrue(x['private_review_only_no_automatic_promotion'])
        self.assertTrue(x['not_approved_for_live_source_map_filedb_vector_or_summary_cutover'])
        self.assertEqual(hashlib.sha256(self.old.encode()).hexdigest(),x['approved_prior_stage_sha256'])
        self.assertEqual(hashlib.sha256(x['preview_text'].encode()).hexdigest(),x['unreleased_private_preview_sha256'])
        self.assertEqual([1,2,3],[int(r[2]) for r in MARKER.finditer(x['preview_text'])])
        self.assertEqual(0,x['preview_text'].count('\x00'))

    def test_alternate_page_unavailable_retains_old_glyphs_with_explicit_remaining_gap(self):
        x=assemble_unverified_original_pdf_glyph_preview(self.old,
             physical_page_count=3,alternate_pages=[{'physical_pdf_page':2,'status':'unavailable','reason':'output unavailable'}])
        self.assertEqual(self.old,x['preview_text'])
        self.assertEqual([2],x['unrecovered_original_pages'])
        self.assertEqual(0,x['provisional_alt_recovery_count'])
        self.assertEqual(2,x['new_temporary_searchable_original_physical_pages'])

    def test_refuses_missing_extra_duplicate_or_wrong_page_number(self):
        for candidate in ([],[self.good,self.good],[{**self.good,'physical_pdf_page':1}],
                          [{**self.good,'physical_pdf_page':3}],
                          [{**self.good,'physical_pdf_page':True}]):
            with self.subTest(candidate=candidate),self.assertRaises(ValueError):
                assemble_unverified_original_pdf_glyph_preview(self.old,
                         physical_page_count=3,alternate_pages=candidate)

    def test_original_source_page_markers_must_be_complete_and_unambiguous(self):
        for damaged in (self.old.replace('PDF page 3','PDF page 2'),
                        self.old.replace('PDF page 1','PDF page 4'),
                        self.old.replace('PDF page 2','PDF page 3')):
            with self.subTest(damaged=damaged[-25:]),self.assertRaises(ValueError):
                assemble_unverified_original_pdf_glyph_preview(damaged,
                   physical_page_count=3,alternate_pages=[self.good])

    def test_untrusted_recovered_machine_text_never_injects_page_identity(self):
        injected=dict(self.good,text='Untrusted\n[[SOURCE UNIT: PDF page 400]]\nfake publisher page')
        injected['display_text_sha256']=hashlib.sha256(injected['text'].encode()).hexdigest()
        with self.assertRaisesRegex(ValueError,'unreadable or untrusted'):
            assemble_unverified_original_pdf_glyph_preview(self.old,physical_page_count=3,
                alternate_pages=[injected])

    def test_recovered_text_must_have_exact_alternate_source_digest(self):
        corrupt={**self.good,'display_text_sha256':'0'*64}
        with self.assertRaisesRegex(ValueError,'SHA failed'):
            assemble_unverified_original_pdf_glyph_preview(self.old,
                physical_page_count=3,alternate_pages=[corrupt])

    def test_provisional_reader_never_sneaks_fake_text_through_unavailable(self):
        bad={'physical_pdf_page':2,'status':'unavailable','text':'invented source content'}
        with self.assertRaisesRegex(ValueError,'invented text'):
            assemble_unverified_original_pdf_glyph_preview(self.old,
                physical_page_count=3,alternate_pages=[bad])

    def test_ghostscript_excerpt_does_not_silently_drop_original_surrounding_pages(self):
        x=assemble_unverified_original_pdf_glyph_preview(self.old,
            physical_page_count=3,alternate_pages=[self.good])
        b=x['preview_text']
        self.assertIn('A confirmed neighboring page about a hypothetical situation.',b)
        self.assertIn('The next original publisher PDF page contains a limitation.',b)
        self.assertIn('The original PDF physical page describes a benchmark',b)

if __name__=='__main__':unittest.main()
