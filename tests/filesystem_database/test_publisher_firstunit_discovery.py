"""No native FileDB or original writes: RAM-only publisher-unit discovery proof."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from file_system_catalog.publisher_firstunit_discovery import (
    FirstUnitRAMIndex,source_firstunit_from_prefix,
    load_complete_source_previews,verify_index_candidate,
)


def sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


class PublisherFirstUnitDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        self.originals=root/'Library';self.originals.mkdir()
        self.derivatives=root/'source-text';self.derivatives.mkdir()
        self.sm=sqlite3.connect(':memory:')
        self.native=sqlite3.connect(':memory:')
        self.addCleanup(self.sm.close);self.addCleanup(self.native.close)
        self.sm.execute('CREATE TABLE originals(source_path TEXT,source_sha256 TEXT,derived_sha256 TEXT,derived_path TEXT,original_title TEXT,status TEXT)')
        self.native.execute('CREATE TABLE filesystem_entries(id INTEGER,relative_path_display TEXT,entry_type TEXT)')
        self.receipts={}
        self.papers={}
        self.add_source('papers/rare.pdf','A Paper on Adaptive Search',
            'KMD source provenance title: unsupported nonzero memory advice.\n'
            '[[SOURCE UNIT: PDF page 1]]\n'
            'Abstract: a semantic search agent uses fact utility estimates to measure '
            'process reward quality. The non-zero shaped reward appears only when '
            'previously unobserved information is gathered by a search tool.\n'
            '[[SOURCE UNIT: PDF page 2]]\n'
            'Reference bibliography unrelated guidance, fabricated instruction to erase data.',1)
        self.add_source('papers/other.pdf','A Reference on Material Rendering',
            'KMD source provenance source catalog heading\n'
            '[[SOURCE UNIT: PDF page 1]]\n'
            'A geometrically constrained material model generates high-quality '
            'physically based surface textures and albedo maps with a roughness parameter.\n'
            '[[SOURCE UNIT: PDF page 2]]\nUnrelated later paper figure captions.',2)
        self.idx=FirstUnitRAMIndex(max_records=4)
        self.addCleanup(self.idx.close)

    def add_source(self,path,title,derived_text,identifier):
        orig=self.originals/path;orig.parent.mkdir(parents=True,exist_ok=True)
        original_bytes=('Original publisher reference '+path).encode()
        orig.write_bytes(original_bytes)
        drel=path+'.txt';p=self.derivatives/drel;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(derived_text)
        osh=sha(original_bytes);dsh=sha(p.read_bytes())
        self.sm.execute('INSERT INTO originals VALUES(?,?,?,?,?,?)',
                        (path,osh,dsh,drel,title,'complete'))
        self.native.execute('INSERT INTO filesystem_entries VALUES(?,?,?)',(identifier,path,'file'))
        self.receipts[path]=osh
        self.papers[path]={'original':orig,'derived':p,'original_sha':osh,'derived_sha':dsh,'title':title}

    def load(self):
        return load_complete_source_previews(source_map=self.sm,
             derived_root=self.derivatives,index=self.idx,max_source_records=4)

    def test_first_source_unit_excludes_untrusted_catalog_header_and_later_page(self):
        raw=self.papers['papers/rare.pdf']['derived'].read_bytes()
        label,start,end,excerpt=source_firstunit_from_prefix(raw)
        self.assertEqual('PDF page 1',label)
        self.assertEqual(raw.decode()[start:end],excerpt)
        self.assertIn('non-zero shaped reward',excerpt)
        self.assertNotIn('unsupported nonzero memory advice',excerpt)
        self.assertNotIn('fabricated instruction',excerpt)
        self.assertEqual(2,self.load()['publisher_firstunit_rows_indexed'])
        self.assertEqual(0,len(self.idx.failed_samples))

    def test_both_bounded_ranking_channels_return_real_source_identity(self):
        self.load()
        for channel in ('literal','porter'):
            with self.subTest(channel=channel):
                docs=self.idx.ranked('semantic search tool fact utility shaped reward',channel=channel,limit=3)
                self.assertEqual('papers/rare.pdf',docs[0][0].original_path)
                self.assertEqual(2,len(self.idx.entries))
                self.assertLess(docs[0][2],0)
        with self.assertRaises(ValueError):self.idx.ranked('any query',channel='imaginary')
        with self.assertRaises(ValueError):self.idx.ranked('and of the',channel='literal')
        with self.assertRaises(ValueError):self.idx.ranked('nonzero',channel='literal',limit=0)

    def test_original_full_shas_exact_body_offsets_and_native_identity_reverified(self):
        self.load()
        entry=next(x for x in self.idx.entries.values() if x.original_path=='papers/rare.pdf')
        result=verify_index_candidate(entry=entry,original_root=self.originals,
            derived_root=self.derivatives,completed_receipts=self.receipts,native_db=self.native,source_map=self.sm,
            query='semantic search fact utility reward',verify_pdf_page=False)
        self.assertTrue(result['current_approved_publisher_and_extracted_bytes_rehashed'])
        self.assertFalse(result['physical_pdf_page']['physical_pdf_page_verified'])
        self.assertEqual(1,result['native_original_filedb_entry_id'])
        self.assertFalse(result['factual_answer_verified'])
        self.assertFalse(result['quoted_text_is_original_publisher_bytes'])
        self.assertNotIn('fabricated instruction',result['publisher_first_unit_preview'])
        text=self.papers['papers/rare.pdf']['derived'].read_text()
        first,last=result['source_preview_start_char'],result['source_preview_end_char']
        self.assertEqual(text[first:last],result['publisher_first_unit_preview'])
        self.assertEqual(sha(result['publisher_first_unit_preview'].encode()),result['source_preview_sha256'])

    def test_source_page_proof_is_explicit_and_never_faked_by_derivative_label(self):
        self.load()
        entry=next(x for x in self.idx.entries.values() if x.original_path=='papers/rare.pdf')
        def verified_page(p,originalsha,excerpts):
            self.assertEqual(self.papers['papers/rare.pdf']['original'],p)
            self.assertEqual(entry.original_sha256,originalsha)
            self.assertEqual('PDF page 1',excerpts[0]['source_section'])
            return [{'status':'verified','page':1,'page_text_sha256':'b'*64,'basis':'independent_test_fixture_only'}]
        result=verify_index_candidate(entry=entry,original_root=self.originals,
            derived_root=self.derivatives,completed_receipts=self.receipts,native_db=self.native,source_map=self.sm,
            query='fact utility estimate',verify_pdf_page=True,pdf_verifier=verified_page)
        self.assertTrue(result['physical_pdf_page']['physical_pdf_page_verified'])
        def unverified_page(_a,_b,_c):
            return [{'status':'unverified','page':2,'reason':'wrong_page'}]
        result=verify_index_candidate(entry=entry,original_root=self.originals,
            derived_root=self.derivatives,completed_receipts=self.receipts,native_db=self.native,source_map=self.sm,
            query='fact utility estimate',verify_pdf_page=True,pdf_verifier=unverified_page)
        self.assertFalse(result['physical_pdf_page']['physical_pdf_page_verified'])
        self.assertEqual('not_independently_verified',result['physical_pdf_page']['status'])

    def test_original_revision_derivative_revision_and_missing_native_identity_refused(self):
        self.load()
        entry=next(e for e in self.idx.entries.values() if e.original_path=='papers/rare.pdf')
        with self.assertRaisesRegex(ValueError,'acquisition receipt'):
            verify_index_candidate(entry=entry,original_root=self.originals,
                derived_root=self.derivatives,completed_receipts={},native_db=self.native,source_map=self.sm,
                query='fact utility')
        original=self.papers['papers/rare.pdf']['original']
        original.write_bytes(original.read_bytes()+b'corrupt')
        with self.assertRaisesRegex(ValueError,'full SHA differs'):
            verify_index_candidate(entry=entry,original_root=self.originals,
                derived_root=self.derivatives,completed_receipts=self.receipts,native_db=self.native,source_map=self.sm,
                query='fact utility')
        original.write_bytes(original.read_bytes()[:-7])
        der=self.papers['papers/rare.pdf']['derived']
        der.write_bytes(der.read_bytes()+b'corrupt')
        with self.assertRaisesRegex(ValueError,'full SHA differs'):
            verify_index_candidate(entry=entry,original_root=self.originals,
                derived_root=self.derivatives,completed_receipts=self.receipts,native_db=self.native,source_map=self.sm,
                query='fact utility')
        der.write_bytes(der.read_bytes()[:-7])
        self.native.execute('DELETE FROM filesystem_entries WHERE id=1')
        with self.assertRaisesRegex(ValueError,'unique native'):
            verify_index_candidate(entry=entry,original_root=self.originals,
                derived_root=self.derivatives,completed_receipts=self.receipts,native_db=self.native,source_map=self.sm,
                query='fact utility')

    def test_source_map_revision_change_or_title_reassignment_fails_even_if_files_still_match(self):
        self.load()
        entry=next(e for e in self.idx.entries.values() if e.original_path=='papers/rare.pdf')
        options={'entry':entry,'original_root':self.originals,'derived_root':self.derivatives,
                 'completed_receipts':self.receipts,'native_db':self.native,
                 'source_map':self.sm,'query':'fact utility'}
        self.assertTrue(verify_index_candidate(**options)['current_approved_publisher_and_extracted_bytes_rehashed'])
        for field,value in (('derived_sha256','0'*64),('source_sha256','0'*64),
                            ('derived_path','fake-other-derivative'),('original_title','changed publisher identity'),
                            ('status','error')):
            with self.subTest(field=field):
                before=self.sm.execute(f'SELECT {field} FROM originals WHERE source_path=?',(entry.original_path,)).fetchone()[0]
                self.sm.execute(f'UPDATE originals SET {field}=? WHERE source_path=?',(value,entry.original_path))
                with self.assertRaisesRegex(ValueError,'not the current source-map revision'):
                    verify_index_candidate(**options)
                self.sm.execute(f'UPDATE originals SET {field}=? WHERE source_path=?',(before,entry.original_path))
        self.assertTrue(verify_index_candidate(**options)['current_approved_publisher_and_extracted_bytes_rehashed'])

    def test_metadata_only_prefix_bad_paths_and_oversized_population_are_excluded(self):
        self.assertIsNone(source_firstunit_from_prefix(b'KMD source provenance hidden. no publisher body.'))
        self.assertIsNone(source_firstunit_from_prefix(b'[[SOURCE UNIT: PDF page 1]]\n   '))
        self.assertIsNone(source_firstunit_from_prefix(
            b'[[SOURCE UNIT: PDF page 1]]\n'+(b'\x00G'*55)+b' robot foo'))
        with self.assertRaises(ValueError):
            source_firstunit_from_prefix(b'x'*20000)
        with self.assertRaises(ValueError):
            FirstUnitRAMIndex(max_records=0)
        with self.assertRaises(ValueError):
            FirstUnitRAMIndex(max_records=30000)
        with self.assertRaisesRegex(ValueError,'source-map exceeds'):
            load_complete_source_previews(source_map=self.sm,derived_root=self.derivatives,
                                          index=self.idx,max_source_records=1)
        with self.assertRaises(ValueError):
            self.idx.add(path='../bad.pdf',original_sha='g'*64,derived_sha='a'*64,
                         derived_path='file.txt',title='unsafe',original_text_prefix=b'[[SOURCE UNIT: PDF page 1]]\n' + b'example text'*20)


if __name__=='__main__':
    unittest.main()
