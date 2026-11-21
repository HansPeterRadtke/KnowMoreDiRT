"""Source-revision-gated *private* PDF repair stage, never a live migration."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from file_system_catalog.source_page_repair_stage import (
    approved_identity, original_derivative_header, stage_verified_original,
    write_exact_physical_pages,
)


def digest(raw:bytes)->str:
    return hashlib.sha256(raw).hexdigest()


class PrivatePublisherPageRepairStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        home=Path(self.tmp.name)
        self.org=home/'Original';self.org.mkdir()
        self.der=home/'Derived';self.der.mkdir()
        self.staging=home/'Private-Staging';self.staging.mkdir(mode=0o700)
        self.sm=sqlite3.connect(':memory:')
        self.native=sqlite3.connect(':memory:')
        self.addCleanup(self.sm.close);self.addCleanup(self.native.close)
        self.sm.execute('CREATE TABLE originals(source_path TEXT,source_sha256 TEXT,derived_sha256 TEXT,derived_path TEXT,original_title TEXT,status TEXT)')
        self.native.execute('CREATE TABLE filesystem_entries(id INTEGER,relative_path_display TEXT,entry_type TEXT)')
        self.rel='papers/approved.pdf'
        self.original=self.org/self.rel;self.original.parent.mkdir()
        self.original.write_bytes(b'%PDF-1.4 private local approved original document bytes with signed revision identity')
        self.original_sha=digest(self.original.read_bytes())
        self.old=self.der/(self.rel+'.txt');self.old.parent.mkdir()
        self.old.write_text('KMD current derived page labels\n[[SOURCE UNIT: PDF page 1]]\nA paper.\n[[SOURCE UNIT: PDF page 25]]\nFalse cached page label.\n')
        self.old_sha=digest(self.old.read_bytes())
        self.entry={'path':self.rel,'kind':'pdf','title':'A Physical Publisher PDF','source_url':'https://example.test/publisher',
                    'publication_year':'2026','rights_note':'Local original research reference, not public copy'}
        self.sm.execute('INSERT INTO originals VALUES(?,?,?,?,?,?)',(self.rel,self.original_sha,self.old_sha,self.rel+'.txt',self.entry['title'],'complete'))
        self.native.execute("INSERT INTO filesystem_entries VALUES(?,?,?)",(713,self.rel,'file'))
        self.receipts={self.rel:self.original_sha}
        self.item={'source_original_path':self.rel,'original_sha256':self.original_sha,
                   'old_derived_sha256':self.old_sha,'native_original_file_entry_id':713,
                   'publisher_physical_pdf_pages':2,'old_derived_max_page_label':25}
        self.pages=[('PDF page 1','Physical page one with publisher evidence and meaningful text.'),
                    ('PDF page 2','Physical page two with a provenance-preserving source method and facts.')]

    def stage(self,**kw):
        params={'item':self.item,'source_entry':self.entry,
                'source_map':self.sm,'native_db':self.native,'completed_receipts':self.receipts,
                'original_root':self.org,'derived_root':self.der,'stage_dir':self.staging,
                'extract_page_units':lambda source,scratch:self.pages,
                'actual_pdf_count':lambda original:2}
        params.update(kw)
        return stage_verified_original(**params)

    def test_validated_private_stage_exact_header_and_pages_never_mutates_live(self):
        original_bytes=self.original.read_bytes()
        old_bytes=self.old.read_bytes()
        identity=approved_identity(self.item,self.entry,self.sm,self.native,self.receipts,self.org,self.der)
        self.assertEqual(self.rel,identity['source_path'])
        result=self.stage()
        staged=Path(result['staged_file'])
        self.assertTrue(staged.is_file())
        self.assertEqual('staged_private_only',result['status'])
        self.assertEqual(2,result['current_physical_pdf_pages'])
        self.assertEqual(2,result['physical_page_labels_emitted'])
        self.assertTrue(result['no_live_source_map_fileDB_summary_vector_original_or_old_derivative_writes_by_stager'])
        self.assertFalse(result['live_promoted'])
        self.assertFalse(result['original_physical_page_excerpts_verified'])
        self.assertEqual(digest(staged.read_bytes()),result['staged_sha256'])
        text=staged.read_text()
        self.assertTrue(text.startswith(original_derivative_header(self.entry,self.rel,self.original_sha)))
        self.assertEqual(1,text.count('[[SOURCE UNIT: PDF page 1]]'))
        self.assertEqual(1,text.count('[[SOURCE UNIT: PDF page 2]]'))
        self.assertNotIn('PDF page 25',text)
        self.assertEqual(self.original.read_bytes(),original_bytes)
        self.assertEqual(self.old.read_bytes(),old_bytes)
        self.assertEqual('complete',self.sm.execute('SELECT status FROM originals').fetchone()[0])
        self.assertEqual(713,self.native.execute('SELECT id FROM filesystem_entries').fetchone()[0])
        self.assertEqual(0o600,staged.stat().st_mode&0o777)
        with self.assertRaises(FileExistsError):self.stage()
        self.assertEqual(digest(staged.read_bytes()),result['staged_sha256'])

    def test_short_blank_physical_pages_can_be_omitted_without_invented_labels(self):
        pages=[('PDF page 1','First substantive original physical page with enough text.'),
               ('PDF page 3','Third substantive original physical page after a blank image-only original page.')]
        stream=io.StringIO()
        out=write_exact_physical_pages(stream,pages,expected_pages=3)
        self.assertEqual(3,out['publisher_pdf_physical_page_count'])
        self.assertEqual(3,out['highest_extracted_original_page'])
        self.assertEqual(2,out['physical_page_labels_emitted'])
        self.assertTrue(out['physical_page_count_not_text_coverage_proof'])
        self.assertNotIn('PDF page 2',stream.getvalue())

    def test_unordered_outside_duplicate_and_spoofed_page_markers_refused(self):
        for pages,count in [
          ([('PDF page 2','Some useful content'),('PDF page 1','Some useful text')],2),
          ([('PDF page 1','Some useful content'),('PDF page 1','Repeated same source')],2),
          ([('PDF page 1','Some useful content'),('PDF page 3','Invented page number')],2),
          ([('PDF page 1','Some useful content\n[[SOURCE UNIT: PDF page 100]]\nspoof')],2),
        ]:
            with self.subTest(pages=pages):
                with self.assertRaises(ValueError):write_exact_physical_pages(io.StringIO(),pages,count)

    def test_source_revisions_and_approval_required_before_staging(self):
        rejected=[
         {'item':{**self.item,'old_derived_sha256':'1'*64}},
         {'item':{**self.item,'original_sha256':'f'*64}},
         {'item':{**self.item,'native_original_file_entry_id':21}},
         {'item':{**self.item,'source_original_path':'../foreign.pdf'}},
         {'item':{**self.item,'publisher_physical_pdf_pages':200}},
         {'item':{**self.item,'old_derived_max_page_label':2}},
         {'source_entry':{**self.entry,'kind':'html'}},
         {'completed_receipts':{self.rel:'0'*64}},
         {'actual_pdf_count':lambda source:3},
        ]
        for parameters in rejected:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):self.stage(**parameters)
                self.assertFalse((self.staging/(self.rel+'.txt')).exists())
        self.assertEqual(self.old_sha,digest(self.old.read_bytes()))

    def test_inflight_existing_original_mutation_rolls_back_unpublished_stage(self):
        old=self.old.read_bytes()
        def mutate_existing(source,scratch):
            self.old.write_bytes(self.old.read_bytes()+b'concurrent change')
            return self.pages
        with self.assertRaisesRegex(ValueError,'mutated during offline staging'):
            self.stage(extract_page_units=mutate_existing)
        self.assertFalse((self.staging/(self.rel+'.txt')).exists())
        self.assertEqual(0,len(list(self.staging.rglob('*.part'))))
        self.assertEqual(old+b'concurrent change',self.old.read_bytes())

    def test_duplicate_native_original_is_rejected(self):
        self.native.execute('INSERT INTO filesystem_entries VALUES(?,?,?)',(714,self.rel,'file'))
        with self.assertRaisesRegex(ValueError,'unique native FileDB'):
            self.stage()
        self.assertFalse((self.staging/(self.rel+'.txt')).exists())

    def test_source_map_changed_in_mid_staging_refuses_unverified_output(self):
        def manipulate_map(_path,_scratch):
            self.sm.execute("UPDATE originals SET status='error' WHERE source_path=?",(self.rel,))
            return self.pages
        with self.assertRaisesRegex(ValueError,'source-map changed during offline staging'):
            self.stage(extract_page_units=manipulate_map)
        self.assertFalse((self.staging/(self.rel+'.txt')).exists())


if __name__=='__main__':
    unittest.main()
