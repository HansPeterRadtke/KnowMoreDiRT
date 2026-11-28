"""Source-current, in-memory complete PDF-section search regression fixtures.

Synthetic originals only; no external publications, model loads or index writes.
"""
from __future__ import annotations
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from file_system_catalog.original_fullsection_fts import (
    MAX_DERIVED_BYTES_PER_SOURCE,OriginalSectionRAMIndex,
    add_current_scope,bounded_original_pdf_sections,verify_current_source_window,
)


def sha(payload:bytes)->str:
    return hashlib.sha256(payload).hexdigest()


class OriginalFullPDFSectionTests(unittest.TestCase):
    def setUp(self):
        temp=TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.ori=self.root/'originals';self.der=self.root/'derivatives'
        self.ori.mkdir();self.der.mkdir()
        self.db=self.root/'native.db';self.sm=self.root/'source_map.db'
        with sqlite3.connect(self.sm) as m:
            m.execute('CREATE TABLE originals(source_path TEXT,source_sha256 TEXT,'
                      'derived_sha256 TEXT,derived_path TEXT,original_title TEXT,'
                      'original_bytes INTEGER,original_kind TEXT,status TEXT)')
        with sqlite3.connect(self.db) as m:
            m.execute('CREATE TABLE filesystem_entries(id INTEGER,relative_path_display TEXT,entry_type TEXT)')
        self.path='research_%/primary.pdf'
        self.phrase='Double helix albatross tomography'
        self.intro='A short reference page describes overall scientific goals. '*9
        self.later=('This later paper page explains ' + self.phrase
                    + ' with repeated direct references. ')*5
        self.text='[[SOURCE UNIT: PDF page 1]]\n'+self.intro+'\n[[SOURCE UNIT: PDF page 7]]\n'+self.later
        self.receipts={}
        self.add(self.path,self.text,1,'Real textual matter heading')
        self.add('research_A/decoy.pdf',
                 '[[SOURCE UNIT: PDF page 1]]\nAn unrelated document about source reading. '*9,
                 2,'A different original')

    def add(self,path,text,ident,title):
        ori=self.ori/path
        der=self.der/(path+'.txt')
        ori.parent.mkdir(parents=True,exist_ok=True)
        der.parent.mkdir(parents=True,exist_ok=True)
        orig_data=('test-only synthetic original '+path).encode()
        ori.write_bytes(orig_data)
        der.write_text(text,encoding='utf8')
        self.receipts[path]=sha(orig_data)
        with sqlite3.connect(self.sm) as m:
            m.execute('INSERT INTO originals VALUES(?,?,?,?,?,?,?,?)',
                      (path,sha(orig_data),sha(text.encode()),path+'.txt',title,
                       len(orig_data),'pdf','complete'))
        with sqlite3.connect(self.db) as m:
            m.execute('INSERT INTO filesystem_entries VALUES(?,?,?)',(ident,path,'file'))

    def search(self,scope='research_%/'):
        index=OriginalSectionRAMIndex()
        with sqlite3.connect(self.sm) as m:
            scan=add_current_scope(source_map=m,derived_root=self.der,prefix=scope,index=index)
        return index,scan

    def verify(self,window,query=None,phrase=None):
        with sqlite3.connect(self.sm) as m,sqlite3.connect(self.db) as n:
            return verify_current_source_window(
                window=window,query=query or self.phrase,phrase=phrase,
                original_root=self.ori,derived_root=self.der,
                source_map=m,native_db=n,completed_receipts=self.receipts)

    def test_late_source_sections_are_retrieved_without_first_page_text(self):
        self.assertNotIn(self.phrase,self.intro)
        with self.search()[0] as index:
            for channel in ('literal','porter'):
                ranked=index.ranked('double helix albatross tomography',channel=channel)
                self.assertTrue(ranked)
                first=ranked[0][0]
                self.assertEqual(self.path,first.original_path)
                self.assertEqual('PDF page 7',first.label)
                self.assertGreater(first.source_start,len(self.intro))
                result=self.verify(first,phrase=self.phrase)
                self.assertEqual(1,result['current_native_original_filedb_entry_id'])
                self.assertTrue(result['complete_original_and_derived_bytes_full_sha_reverified'])
                self.assertFalse(result['original_pdf_physical_page_verified'])
                self.assertFalse(result['factual_answer_verified'])
                self.assertIn(self.phrase,result['extracted_source_excerpt'])
                self.assertEqual(sha(result['extracted_source_excerpt'].encode()),
                                 result['extracted_source_excerpt_sha256'])

    def test_literal_percent_and_underscore_scope_does_not_include_decoy(self):
        with self.search()[0] as index:
            self.assertEqual({self.path},set(index.documents))
        with self.search('research_A/')[0] as other:
            self.assertEqual({'research_A/decoy.pdf'},set(other.documents))

    def test_safe_index_import_rejects_changed_source_map_derivative(self):
        path=self.der/(self.path+'.txt')
        path.write_text(path.read_text()+' modified',encoding='utf8')
        with OriginalSectionRAMIndex() as index,sqlite3.connect(self.sm) as m:
            with self.assertRaisesRegex(ValueError,'original extracted PDF bytes differ'):
                add_current_scope(source_map=m,derived_root=self.der,
                                  prefix='research_%/',index=index)

    def test_stale_original_publisher_or_derived_sha_refuses_result(self):
        with self.search()[0] as index:
            window=index.ranked('tomography albatross',channel='literal')[0][0]
            self.ori.joinpath(self.path).write_bytes(b'tampered test-only original bytes')
            with self.assertRaisesRegex(ValueError,'publisher original/derived bytes differ'):
                self.verify(window)

    def test_stale_source_map_revision_refused_before_reading_publisher(self):
        with self.search()[0] as index:
            entry=index.ranked('tomography',channel='literal')[0][0]
            with sqlite3.connect(self.sm) as m:
                m.execute('UPDATE originals SET original_title=? WHERE source_path=?',
                          ('changed owner metadata',self.path))
            with self.assertRaisesRegex(ValueError,'publisher source-map revision changed'):
                self.verify(entry)

    def test_native_duplicate_is_not_accepted(self):
        with self.search()[0] as index:
            entry=index.ranked('albatross',channel='literal')[0][0]
            with sqlite3.connect(self.db) as n:
                n.execute('INSERT INTO filesystem_entries VALUES (?,?,?)',
                          (99,self.path,'file'))
            with self.assertRaisesRegex(ValueError,'one unique original'):
                self.verify(entry)

    def test_source_window_is_exact_utf8_original_span(self):
        data=('[[SOURCE UNIT: PDF page 1]]\nSome description with an emoji 🔭. '*100
              +'\n[[SOURCE UNIT: PDF page 3]]\nAnother substantial source page. '*200)
        windows=bounded_original_pdf_sections(
            self.path,'a'*64,'b'*64,self.path+'.txt','A book',data)
        self.assertGreater(len(windows),2)
        for w in windows:
            self.assertEqual(data[w.source_start:w.source_end],w.original_body)
            self.assertLessEqual(len(w.original_body),3600)
            self.assertTrue(w.label.startswith('PDF page '))
        self.assertIn('PDF page 3',{w.label for w in windows})

    def test_actual_operator_search_reopens_and_verifies_current_original(self):
        import sys
        scripts=str(Path(__file__).resolve().parents[2]/'scripts')
        sys.path.insert(0,scripts)
        try:
            from raspi_original_fullsection_fts import search
            ledger=self.root/'ledger.json'
            ledger.write_text(json.dumps({'items':[
                {'path':k,'status':'complete','sha256':v} for k,v in self.receipts.items()]}))
            questions=[{'id':'late','query':'albatross tomography',
                        'require_source_phrase':self.phrase}]
            response=search(prefix='research_%/',questions=questions,top=2,
                            source_map=self.sm,native_db=self.db,
                            original_root=self.ori,derived_root=self.der,
                            completed_ledger=ledger)
            self.assertEqual('kmd.original_complete_pdf_section_source_scoped_lexical_v1',
                             response['schema'])
            self.assertTrue(response['complete_approved_original_and_derived_sha_rechecked_after_all_queries'])
            self.assertTrue(response['every_returned_pdf_excerpt_has_current_derived_offsets_and_sha'])
            self.assertFalse(response['source_publisher_actual_PDF_physical_pages_unverified'] is False)
            self.assertEqual(1,response['source_scan']['source_map_eligible_pdf_originals'])
            self.assertEqual(1,response['unique_returned_originals'])
            for channel in ('literal','porter'):
                ranked=response['questions'][0]['independent_rank_channels'][channel]['source_originals']
                self.assertEqual(1,len(ranked))
                result=ranked[0]
                self.assertEqual(self.path,result['original_path'])
                self.assertIn(self.phrase,result['extracted_source_excerpt'])
                self.assertFalse(result['factual_answer_verified'])
                self.assertEqual(sha(result['extracted_source_excerpt'].encode()),
                                 result['extracted_source_excerpt_sha256'])
        finally:
            sys.path.remove(scripts)

    def test_current_derived_mutation_after_index_cannot_return_unverified_source(self):
        import sys
        from unittest import mock
        scripts=str(Path(__file__).resolve().parents[2]/'scripts')
        sys.path.insert(0,scripts)
        try:
            import raspi_original_fullsection_fts as cli
            ledger=self.root/'ledger2.json'
            ledger.write_text(json.dumps({'items':[
                {'path':k,'status':'complete','sha256':v} for k,v in self.receipts.items()]}))
            original=cli.verify_current_source_window
            seen=[]
            def tamper_after_first_proof(**kw):
                value=original(**kw)
                seen.append(value['original_path'])
                if len(seen)==1:
                    target=self.der/(self.path+'.txt')
                    target.write_bytes(target.read_bytes()+b' silently changed')
                return value
            with mock.patch.object(cli,'verify_current_source_window',side_effect=tamper_after_first_proof):
                with self.assertRaisesRegex(ValueError,'publisher original/derived bytes differ'):
                    cli.search(prefix='research_%/',
                               questions=[{'id':'late','query':'albatross tomography'}],
                               top=2,source_map=self.sm,native_db=self.db,
                               original_root=self.ori,derived_root=self.der,
                               completed_ledger=ledger)
            self.assertEqual([self.path],seen)
        finally:
            sys.path.remove(scripts)

    def test_unsafe_path_and_nul_corrupt_page_fail_closed(self):
        text='[[SOURCE UNIT: PDF page 1]]\n'+'\x00'*50+'text'
        for p in ('../escape.pdf','/root.pdf','research_%/../illegal.pdf','C:\\foo.pdf'):
            with self.subTest(path=p),self.assertRaises(ValueError):
                bounded_original_pdf_sections(p,'a'*64,'b'*64,p+'.txt','good title',text)
        with self.assertRaisesRegex(ValueError,'NUL-corrupted'):
            bounded_original_pdf_sections(self.path,'a'*64,'b'*64,
                                           self.path+'.txt','good title',text)

    def test_explicitly_skips_corrupted_original_pdf_page_but_keeps_intact_later_pages(self):
        text=('[[SOURCE UNIT: PDF page 1]]\n'+'\x00'*120+'bad scanned glyphs'
              +'\n[[SOURCE UNIT: PDF page 2]]\n'
              +('The certified safe original section discusses salmonella and hydration. '*12))
        spans,omitted=bounded_original_pdf_sections(
            self.path,'a'*64,'b'*64,self.path+'.txt',
            'Synthetic source',text,report_corrupt_units=True)
        self.assertEqual(1,len(omitted))
        self.assertEqual('PDF page 1',omitted[0]['source_unit_label'])
        self.assertTrue(omitted[0]['original_section_omitted_from_FTS'])
        self.assertTrue(all(z.label=='PDF page 2' for z in spans))
        self.assertTrue(all('salmonella' in z.original_body for z in spans))
        self.assertFalse(any('bad scanned glyphs' in z.original_body for z in spans))
        original=self.der/(self.path+'.txt')
        original.write_text(text,encoding='utf8')
        with sqlite3.connect(self.sm) as m:
            m.execute('UPDATE originals SET derived_sha256=? WHERE source_path=?',
                      (sha(text.encode()),self.path))
        with self.search()[0] as idx:
            self.assertEqual(1,len(idx.unreadable_source_units_explicitly_omitted))
            self.assertTrue(idx.ranked('salmonella hydration',channel='literal'))
            self.assertFalse(idx.ranked('bad scanned glyphs',channel='literal'))

    def test_short_and_empty_extracted_original_source_units_explicitly_reported(self):
        text=('[[SOURCE UNIT: PDF page 1]]\n'
              '  \n[[SOURCE UNIT: PDF page 2]]\n'
              'Tiny.\n[[SOURCE UNIT: PDF page 3]]\n'
              +('This full usable source section discusses evidence and original provenance. '*7))
        sections,skipped=bounded_original_pdf_sections(
            self.path,'a'*64,'b'*64,self.path+'.txt','Synthetic',text,
            report_corrupt_units=True)
        self.assertEqual(2,len(skipped))
        self.assertTrue(all(x['reason']=='empty_or_too_short_extracted_original_PDF_source_unit' for x in skipped))
        self.assertEqual({'PDF page 3'},{x.label for x in sections})
        self.assertTrue(all(x.original_body in text for x in sections))

    def test_bounded_source_count_and_missing_prefix_refuse(self):
        with sqlite3.connect(self.sm) as m, OriginalSectionRAMIndex() as idx:
            for prefix in ('','/root/','research_','research_%/../','missing/'):
                with self.subTest(prefix=prefix),self.assertRaises(ValueError):
                    add_current_scope(source_map=m,derived_root=self.der,
                                      prefix=prefix,index=idx)

    def test_required_phrase_cannot_be_satisfied_by_title_metadata(self):
        with self.search()[0] as index:
            entry=index.ranked('helix tomography',channel='literal')[0][0]
            with self.assertRaisesRegex(ValueError,'lacks required exact phrase'):
                self.verify(entry,phrase='text absent from current source')
            result=self.verify(entry,phrase='Double helix albatross tomography')
            self.assertIn('Double helix albatross tomography',result['extracted_source_excerpt'])


if __name__=='__main__':
    unittest.main()
