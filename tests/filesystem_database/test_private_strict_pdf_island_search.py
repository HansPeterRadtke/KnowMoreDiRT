"""No source quote escapes from a corrupted physical page without a new proof."""
from __future__ import annotations
import hashlib
import unittest
from file_system_catalog.private_strict_pdf_island_search import (
    candidate_original_glyph_islands,OriginalGlyphIslandRAMIndex,MAX_PAGES,
    MAX_ISLAND_BYTES,MAX_RAW_BYTES,
)

def sha(x):return hashlib.sha256(x.encode()).hexdigest()

class StrictOriginalGlyphIslandsTests(unittest.TestCase):
    def setUp(self):
        self.path='science/current-private-publisher-paper.pdf'
        self.good=('Physical original-source provenance retrieval has to preserve the exact '
                   'publisher page identity and require a second independent extractor '
                   'before citing source text, including citations from complex layouts. ')*6
        self.raw=(self.good+'\x00'+'An invalid glyph boundary cannot be bridged into '
                  'a new source phrase, even when the neighboring words appear plausible. '
                  +self.good).encode()
    def add(self,idx,raw=None,page=13,path=None,**extra):
        return idx.add_pinned_page(path=path or self.path,original_sha256='a'*64,
           physical_page=page,original_page_count=30,raw_machine_text=self.raw if raw is None else raw,
           native_original_file_id=42,current_old_derived_sha256='b'*64,
           corrected_staged_derived_sha256='c'*64,**extra)
    def test_no_glyph_join_and_exact_two_separate_islands(self):
        pages=candidate_original_glyph_islands(self.raw)
        self.assertEqual(2,len(pages))
        self.assertEqual(self.good.strip(),pages[0])
        self.assertTrue(pages[1].startswith('An invalid glyph boundary'))
        self.assertNotIn('layouts. An invalid',pages[0])
        with OriginalGlyphIslandRAMIndex() as index:
            self.assertEqual(2,self.add(index))
            index.commit()
            self.assertEqual(1,len(index.keys))
            self.assertEqual(2,len(index.entries))
            hits=index.ranked('original-source provenance retrieval publisher identity')
            self.assertGreaterEqual(len(hits),1)
            self.assertTrue(all(x['source_excerpt_not_corroborated_or_quoted'] for x in hits))
            self.assertTrue(all('provisional_machine_excerpt' not in x for x in hits))
    def test_no_fragments_cross_separate_unreadable_runs(self):
        raw=('one two three four five six seven '+ '\x00'+ 'eight nine ten eleven twelve thirteen fourteen').encode()
        self.assertEqual([],candidate_original_glyph_islands(raw))
        with OriginalGlyphIslandRAMIndex() as index:
            self.assertEqual(0,self.add(index,raw=raw))
            index.commit()
            self.assertEqual([],index.ranked('eleven twelve thirteen fourteen'))
    def test_invalid_utf8_and_oversized_input_refused(self):
        for raw in (b'',b'broken\xfftext',b'a'*MAX_RAW_BYTES,None):
            with self.subTest(rawtype=type(raw).__name__),self.assertRaises(ValueError):
                candidate_original_glyph_islands(raw)
        with self.assertRaises(ValueError):candidate_original_glyph_islands(self.raw,max_islands=0)
        with self.assertRaises(ValueError):candidate_original_glyph_islands((self.good+'\x00'+self.good).encode(),max_islands=1)
    def test_refuse_private_page_markers_from_untrusted_machine(self):
        with self.assertRaisesRegex(ValueError,'page marker'):
            candidate_original_glyph_islands(('[[SOURCE UNIT: PDF page 99]] '+self.good).encode())
    def test_source_physical_page_native_original_sha_and_path_pinned(self):
        with OriginalGlyphIslandRAMIndex() as index:
            self.add(index);index.commit()
            rows=index.ranked('provenance publisher page identity')
            self.assertTrue(rows)
            row=rows[0]
            body=index.private_fragment_for_second_extractor(row_id=row['row_id'],path=row['path'],
                original_sha256=row['original_sha256'],physical_page=row['physical_pdf_page'],
                body_sha256=row['body_sha256'])
            self.assertEqual(row['body_sha256'],sha(body))
            for change in ({'path':'science/wrong.pdf'},{'original_sha256':'f'*64},{'physical_page':7},{'body_sha256':'d'*64}):
                args=dict(row_id=row['row_id'],path=row['path'],original_sha256=row['original_sha256'],
                          physical_page=row['physical_pdf_page'],body_sha256=row['body_sha256'])
                args.update(change)
                with self.subTest(change=change),self.assertRaises(ValueError):index.private_fragment_for_second_extractor(**args)
            with self.assertRaises(ValueError):self.add(index)
            with self.assertRaises(ValueError):self.add(index,path='../escaped.pdf')
            with self.assertRaises(ValueError):self.add(index,page=True)
            self.assertEqual([],index.ranked('intergalactic impossibility device'))
            self.assertEqual([],index.ranked('publisher nebula zetafoo'))
            with self.assertRaises(ValueError):index.ranked('publisher',original_path='not-in-cohort.pdf')
        with self.assertRaises(ValueError):index.ranked('provenance publisher')
    def test_invalid_query_and_limits_refused(self):
        with OriginalGlyphIslandRAMIndex() as index:
            self.add(index);index.commit()
            for query in ('',20,'x'*401):
                with self.subTest(query=query),self.assertRaises(ValueError):index.ranked(query)
            with self.assertRaises(ValueError):index.ranked('source publisher',limit=0)
            with self.assertRaises(ValueError):index.ranked('source publisher',limit=61)
    def test_temporary_page_budget_and_no_fabricated_publisher_text(self):
        with OriginalGlyphIslandRAMIndex() as index:
            for page in range(1,MAX_PAGES+1):self.add(index,page=page)
            with self.assertRaisesRegex(ValueError,'invalid or duplicated'):
                self.add(index,page=MAX_PAGES+1)
            index.commit()
            self.assertEqual(MAX_PAGES,len(index.keys))
            self.assertEqual(MAX_PAGES*2,len(index.entries))

if __name__=='__main__':unittest.main()
