"""Paired search composes two independently verified channels, not their scores."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from scripts.raspi_paired_original_discovery import (
    independent_original_union,final_original_union_revision_guard,
    augment_original_union_with_firstunit,verified_publisher_firstunit_candidates,
)


def native(*, paths=("papers/a.pdf","papers/b.pdf"),sha="a"*64):
    return {
        'count':len(paths),
        'results':[
            {'original_path':p,'original_source_sha256':sha,
             'exact_passage_hash_verified':True,'source_passage_sha256':"b"*64}
            for p in paths
        ],
        'final_current_original_source_revision_check':{
            'full_publisher_original_and_current_derivative_SHA_rechecked':True,
            'safe_to_cutover_live':False,
        },
    }


def lexical(*, paths=("papers/b.pdf","papers/c.pdf"),sha="a"*64):
    return {
        'originals_physically_verified_and_returned':len(paths),
        'candidates':[
            {'original_path':p,'original_sha256_reverified':sha,
             'source_original_and_derived_bytes_verified':True,
             'source_citable_passage_verified_by_this_discovery':False}
            for p in paths
        ],
        'not_a_factual_answer_or_source_passage_quote':True,
        'cross_store_immutable_generation_pinned':False,
    }


def firstunit(*,paths=("papers/c.pdf","papers/d.pdf"),sha="a"*64,
              derivative_sha="d"*64,native_id=42,preview_text=None):
    preview=(preview_text if preview_text is not None else
             "The original paper describes a source unit with concrete experimental evidence.")
    return {
        'returned':len(paths),
        'source_scan':{'publisher_firstunit_rows_indexed':len(paths)},
        'unverified_source_candidates_excluded':[],
        'source_unit_text_is_derived_original_not_physical_pdf_page_verification':True,
        'factual_answer_verified':False,
        'candidate_originals':[{
            'source_rank':i,
            'original_path':path,
            'publisher_original_sha256':sha,
            'current_derivative_sha256':derivative_sha,
            'native_original_filedb_entry_id':native_id,
            'publisher_first_unit_preview':preview,
            'source_preview_start_char':0,
            'source_preview_end_char':len(preview),
            'source_preview_sha256':hashlib.sha256(preview.encode()).hexdigest(),
            'current_approved_publisher_and_extracted_bytes_rehashed':True,
            'quoted_text_is_original_publisher_bytes':False,
            'factual_answer_verified':False,
            'source_unit_label_scope':'derived_source_text_only_not_physical_page_proof',
        } for i,path in enumerate(paths,1)],
    }


class ThreeIndependentOriginalSourcesTest(unittest.TestCase):
    def test_first_unit_adds_missing_original_without_reranking_prior_channels(self):
        base=independent_original_union(native(),lexical())
        rows=augment_original_union_with_firstunit(base,firstunit())
        self.assertEqual(['papers/a.pdf','papers/b.pdf','papers/c.pdf','papers/d.pdf'],
                         [x['original_path'] for x in rows])
        self.assertEqual([1,2,None,None],[x['native_source_passage_rank'] for x in rows])
        self.assertEqual([None,1,2,None],[x['lexical_summary_candidate_rank'] for x in rows])
        self.assertEqual([None,None,1,2],[x['publisher_firstunit_porter_rank'] for x in rows])
        self.assertTrue(rows[1]['both_independently_returned'])
        self.assertIsNone(rows[3]['native_original_passage_sha256'])
        self.assertEqual('d'*64,rows[3]['publisher_firstunit_derived_sha256'])
        self.assertEqual(42,rows[3]['publisher_firstunit_native_original_id'])
        self.assertEqual(len(rows),len(set(x['original_path'] for x in rows)))

    def test_untrusted_or_tampered_firstunit_evidence_rejected(self):
        base=independent_original_union(native(),lexical())
        wrong=firstunit(sha='f'*64)
        with self.assertRaisesRegex(ValueError,'inconsistent original SHA'):
            augment_original_union_with_firstunit(base,wrong)
        wrong=firstunit(paths=('papers/c.pdf','papers/c.pdf'))
        with self.assertRaisesRegex(ValueError,'duplicate firstunit'):
            augment_original_union_with_firstunit(base,wrong)
        wrong=firstunit()
        wrong['candidate_originals'][0]['publisher_first_unit_preview']='unverified changed excerpt'
        with self.assertRaisesRegex(ValueError,'lacks independent current-original proof'):
            augment_original_union_with_firstunit(base,wrong)
        wrong=firstunit()
        wrong['candidate_originals'][0]['source_unit_label_scope']='actual original PDF quote'
        with self.assertRaisesRegex(ValueError,'lacks independent current-original proof'):
            augment_original_union_with_firstunit(base,wrong)
        wrong=firstunit()
        wrong['candidate_originals'][0]['factual_answer_verified']=True
        with self.assertRaisesRegex(ValueError,'lacks independent current-original proof'):
            augment_original_union_with_firstunit(base,wrong)
        wrong=firstunit()
        wrong['source_unit_text_is_derived_original_not_physical_pdf_page_verification']=False
        with self.assertRaisesRegex(ValueError,'untrusted original firstunit'):
            augment_original_union_with_firstunit(base,wrong)

    def test_three_channel_limit_fifteen_vs_old_two_channel_ten(self):
        base=[
            {'original_path':f"docs/{i}.pdf",'original_source_sha256':'a'*64,
             'native_source_passage_rank':i+1 if i<5 else None,
             'lexical_summary_candidate_rank':i-4 if i>=5 else None}
            for i in range(10)
        ]
        added=firstunit(paths=tuple(f"docs/{i}.pdf" for i in range(10,15)))
        rows=augment_original_union_with_firstunit(base,added)
        self.assertEqual(15,len(rows))
        too_many=firstunit(paths=tuple(f'docs/{i}.pdf' for i in range(10,16)))
        with self.assertRaisesRegex(ValueError,'untrusted original firstunit'):
            augment_original_union_with_firstunit(base,too_many)

    def test_existing_paired_union_without_firstunit_unmodified(self):
        rows=independent_original_union(native(),lexical())
        self.assertTrue(all('publisher_firstunit_porter_rank' not in r for r in rows))


class PairedSourceChannelTest(unittest.TestCase):
    def test_union_retains_per_channel_ranks_and_no_overlapping_duplicate(self):
        rows=independent_original_union(native(),lexical())
        self.assertEqual(['papers/a.pdf','papers/b.pdf','papers/c.pdf'],
                         [r['original_path'] for r in rows])
        self.assertEqual([1,2,None],[r['native_source_passage_rank'] for r in rows])
        self.assertEqual([None,1,2],[r['lexical_summary_candidate_rank'] for r in rows])
        self.assertEqual([False,True,False],[r['both_independently_returned'] for r in rows])
        self.assertEqual(["b"*64,"b"*64,None],[r['native_original_passage_sha256'] for r in rows])

    def test_one_channel_only_never_requires_fake_source(self):
        rows=independent_original_union(native(paths=()),lexical(paths=('papers/c.pdf',)))
        self.assertEqual(1,len(rows))
        self.assertIsNone(rows[0]['native_source_passage_rank'])
        rows=independent_original_union(native(paths=('papers/a.pdf',)),lexical(paths=()))
        self.assertEqual(1,len(rows))
        self.assertIsNone(rows[0]['lexical_summary_candidate_rank'])

    def test_rejects_duplicate_stale_or_contradictory_source_identity(self):
        with self.assertRaisesRegex(ValueError,'contradictory'):
            independent_original_union(native(),lexical(sha='e'*64))
        with self.assertRaisesRegex(ValueError,'duplicate native'):
            independent_original_union(native(paths=('papers/a.pdf','papers/a.pdf')),lexical())
        with self.assertRaisesRegex(ValueError,'duplicate lexical'):
            independent_original_union(native(paths=()),lexical(paths=('papers/c.pdf','papers/c.pdf')))
        x=native();x['final_current_original_source_revision_check']['safe_to_cutover_live']=True
        with self.assertRaisesRegex(ValueError,'revision proof'):
            independent_original_union(x,lexical())
        x=lexical();x['not_a_factual_answer_or_source_passage_quote']=False
        with self.assertRaisesRegex(ValueError,'caution'):
            independent_original_union(native(),x)

    def test_wrong_native_passage_digest_or_lexical_unproven_candidate(self):
        x=native();x['results'][0]['exact_passage_hash_verified']=False
        with self.assertRaisesRegex(ValueError,'provenance'):
            independent_original_union(x,lexical())
        x=lexical();x['candidates'][0]['source_original_and_derived_bytes_verified']=False
        with self.assertRaisesRegex(ValueError,'verified original'):
            independent_original_union(native(),x)


class FinalPhysicalOriginalRecheckTest(unittest.TestCase):
    def setUp(self):
        tmp=TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.home=Path(tmp.name)
        self.orig=self.home/'originals';self.orig.mkdir()
        self.derived=self.home/'derived';self.derived.mkdir()
        self.path='some/paper.pdf'
        self.origfile=self.orig/self.path;self.origfile.parent.mkdir()
        self.origfile.write_bytes(b'%PDF-original-publisher-source-\n' * 5)
        self.derivedfile=self.derived/'some/paper.pdf.txt'
        self.derivedfile.parent.mkdir()
        self.derivedfile.write_bytes(b'[[SOURCE UNIT: PDF page 1]]\nsource text')
        self.sha=hashlib.sha256(self.origfile.read_bytes()).hexdigest()
        self.dsha=hashlib.sha256(self.derivedfile.read_bytes()).hexdigest()
        self.ledger=self.home/'status.json'
        self.ledger.write_text(json.dumps({'items':[
            {'path':self.path,'status':'complete','sha256':self.sha}
        ]}))
        self.sm=sqlite3.connect(':memory:');self.native=sqlite3.connect(':memory:')
        self.addCleanup(self.sm.close);self.addCleanup(self.native.close)
        self.sm.execute('CREATE TABLE originals(source_path TEXT,source_sha256 TEXT,derived_sha256 TEXT,derived_path TEXT,original_bytes INTEGER,status TEXT)')
        self.sm.execute('INSERT INTO originals VALUES(?,?,?,?,?,?)',(
            self.path,self.sha,self.dsha,'some/paper.pdf.txt',self.origfile.stat().st_size,'complete',
        ))
        self.native.execute('CREATE TABLE filesystem_entries(id INTEGER,relative_path_display TEXT,entry_type TEXT,size_bytes INTEGER,content_sha256 TEXT,hash_status TEXT)')
        self.native.execute('INSERT INTO filesystem_entries VALUES(?,?,?,?,?,?)',(
            42,self.path,'file',self.origfile.stat().st_size,self.sha,'complete',
        ))
        self.rows=[{'original_path':self.path,'original_source_sha256':self.sha,
                    'native_source_passage_rank':1,'lexical_summary_candidate_rank':None}]

    def check(self):
        return final_original_union_revision_guard(
            records=self.rows,source_map=self.sm,native_db=self.native,
            original_root=self.orig,derivative_root=self.derived,ledger=self.ledger,
        )

    def test_final_original_source_byte_shas_and_native_identity(self):
        report=self.check()
        self.assertEqual(1,report['originals_reverified_after_both_channels'])
        self.assertFalse(report['cross_store_immutable_generation_pinned'])
        self.assertFalse(report['is_original_passage_relevance_or_truth_proof'])
        self.assertEqual(42,report['verification_records'][0]['native_original_file_entry_id'])
        self.assertEqual(self.dsha,report['verification_records'][0]['derived_sha256'])

    def test_changed_original_derivative_ledger_or_native_id_rejected(self):
        self.origfile.write_bytes(self.origfile.read_bytes()+b'evil')
        with self.assertRaises(ValueError):self.check()
        self.origfile.write_bytes(b'%PDF-original-publisher-source-\n' * 5)
        self.derivedfile.write_bytes(self.derivedfile.read_bytes()+b'bad')
        with self.assertRaisesRegex(ValueError,'bytes changed'):
            self.check()
        self.derivedfile.write_bytes(b'[[SOURCE UNIT: PDF page 1]]\nsource text')
        self.ledger.write_text(json.dumps({'items':[
            {'path':self.path,'status':'complete','sha256':'0'*64}
        ]}))
        with self.assertRaisesRegex(ValueError,'approved published'):
            self.check()
        self.ledger.write_text(json.dumps({'items':[
            {'path':self.path,'status':'complete','sha256':self.sha}
        ]}))
        self.native.execute('UPDATE filesystem_entries SET content_sha256=?',('b'*64,))
        with self.assertRaisesRegex(ValueError,'native original SHA contradicts'):
            self.check()

    def test_malicious_original_path_escapes_rejected(self):
        self.rows=[{'original_path':'../outside.pdf','original_source_sha256':self.sha}]
        self.sm.execute('INSERT INTO originals VALUES(?,?,?,?,?,?)',(
            '../outside.pdf',self.sha,self.dsha,'some/paper.pdf.txt',
            self.origfile.stat().st_size,'complete',
        ))
        with self.assertRaises(ValueError):self.check()


    def test_native_skipped_large_hash_is_valid_only_with_independent_original_sha(self):
        self.native.execute(
            'UPDATE filesystem_entries SET content_sha256=NULL,hash_status=?',
            ('skipped_large',),
        )
        result=self.check()
        row=result['verification_records'][0]
        self.assertFalse(row['native_stored_original_sha_reverified_if_present'])
        self.assertEqual('skipped_large',row['native_scanner_hash_status'])
        self.assertTrue(row['publisher_original_sha_independently_rehashed_even_when_native_skipped_large'])
        self.native.execute(
            'UPDATE filesystem_entries SET hash_status=?',('hash_error',),
        )
        with self.assertRaisesRegex(ValueError,'hash missing'):
            self.check()


    def test_streamed_publisher_budget_is_separate_from_derived_memory_cap(self):
        from unittest.mock import patch
        original_size=self.origfile.stat().st_size
        derivative_size=self.derivedfile.stat().st_size
        self.assertGreater(original_size,0)
        with patch('scripts.raspi_paired_original_discovery.MAX_SINGLE_ORIGINAL_BYTES',
                   original_size-1):
            with self.assertRaisesRegex(ValueError,'protected byte limit'):
                self.check()
        with patch('scripts.raspi_paired_original_discovery.MAX_SINGLE_ORIGINAL_BYTES',
                   original_size):
            self.assertEqual(1,self.check()['originals_reverified_after_both_channels'])
        with patch('scripts.raspi_paired_original_discovery.MAX_SINGLE_DERIVED_BYTES',
                   derivative_size-1):
            with self.assertRaisesRegex(ValueError,'protected byte limit'):
                self.check()
        with patch('scripts.raspi_paired_original_discovery.MAX_TOTAL_SOURCE_BYTES',
                   original_size+derivative_size-1):
            with self.assertRaisesRegex(ValueError,'physical read budget'):
                self.check()

    def test_firstunit_identity_guard_rejects_current_derived_or_native_movement(self):
        source=firstunit(paths=(self.path,),sha=self.sha,
                         derivative_sha=self.dsha,native_id=42,
                         preview_text=self.derivedfile.read_text())
        rows=augment_original_union_with_firstunit([],source)
        self.rows=rows
        self.assertEqual(1,self.check()['originals_reverified_after_both_channels'])
        self.rows[0]['publisher_firstunit_derived_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'first-unit passage identity moved'):
            self.check()
        self.rows[0]['publisher_firstunit_derived_sha256']=self.dsha
        self.rows[0]['publisher_firstunit_native_original_id']=77
        with self.assertRaisesRegex(ValueError,'first-unit passage identity moved'):
            self.check()

    def test_final_source_excerpt_offsets_rechecked_against_current_derivative(self):
        original=firstunit(paths=(self.path,),sha=self.sha,derivative_sha=self.dsha,
                           native_id=42,preview_text=self.derivedfile.read_text())
        self.rows=augment_original_union_with_firstunit([],original)
        self.assertEqual(1,self.check()['originals_reverified_after_both_channels'])
        self.rows[0]['publisher_firstunit_excerpt_start_char']=1
        with self.assertRaisesRegex(ValueError,'first-unit excerpt no longer occupies'):
            self.check()

    def test_optional_third_channel_is_never_run_without_explicit_flag(self):
        from unittest.mock import patch
        from scripts.raspi_paired_original_discovery import paired_verified_discovery
        self.sm.commit();self.native.commit()
        sm_path=self.home/'temp-map.sqlite3';db_path=self.home/'temp-native.sqlite3'
        with sqlite3.connect(sm_path) as c:self.sm.backup(c)
        with sqlite3.connect(db_path) as c:self.native.backup(c)
        base_n=native(paths=(self.path,),sha=self.sha)
        base_l=lexical(paths=())
        extra=firstunit(paths=(self.path,),sha=self.sha,derivative_sha=self.dsha,
                        native_id=42,preview_text=self.derivedfile.read_text())
        def invoke(flag):
            return paired_verified_discovery(query='verified source query',top=1,
                include_source_firstunit=flag,
                original_root=self.orig,derived_root=self.derived,
                source_map_path=sm_path,native_db_path=db_path,ledger=self.ledger)
        with patch('scripts.raspi_paired_original_discovery.native_low_memory_search',return_value=base_n),\
             patch('scripts.raspi_paired_original_discovery.verified_original_discovery',return_value=base_l),\
             patch('scripts.raspi_paired_original_discovery.verified_publisher_firstunit_candidates',return_value=extra) as third:
            original=invoke(False)
            third.assert_not_called()
            self.assertEqual('kmd.opt_in_paired_native_passage_plus_summary_lexical_original_discovery.v1',original['schema'])
            self.assertNotIn('publisher_firstunit_verified_original_candidates',original)
            extended=invoke(True)
            third.assert_called_once()
            self.assertEqual('kmd.opt_in_verified_three_independent_original_channels.v1',extended['schema'])
            self.assertEqual(1,extended['publisher_firstunit_verified_original_candidates'])
            self.assertEqual(1,extended['unique_original_candidates'])
            self.assertEqual(1,extended['candidates_found_in_at_least_two_of_three_source_channels'])
            self.assertEqual(0,extended['candidates_found_exclusively_by_original_firstunit'])
            self.assertFalse(extended['no_aggregate_confidence_or_factual_answer'] is False)
            self.assertEqual(1,extended['final_current_publisher_source_byte_and_native_file_identity_check']['originals_reverified_after_both_channels'])

    def test_native_busy_retry_only_for_one_known_readonly_sqlite_conflict(self):
        from unittest.mock import patch
        from scripts.raspi_paired_original_discovery import paired_verified_discovery
        def sqlite_busy():
            err=sqlite3.OperationalError('database is locked')
            err.sqlite_errorcode=sqlite3.SQLITE_BUSY
            return err
        mapfile=self.home/'map.sqlite3'
        nativefile=self.home/'native.sqlite3'
        self.sm.commit()
        self.native.commit()
        with sqlite3.connect(mapfile) as saved:self.sm.backup(saved)
        with sqlite3.connect(nativefile) as saved:self.native.backup(saved)
        expected=native(paths=(self.path,),sha=self.sha)
        with patch('scripts.raspi_paired_original_discovery.native_low_memory_search',
                   side_effect=[sqlite_busy(),expected]) as run,\
             patch('scripts.raspi_paired_original_discovery.verified_original_discovery',
                   return_value=lexical(paths=())) as scan,\
             patch('scripts.raspi_paired_original_discovery.time.sleep') as delay:
            result=paired_verified_discovery(
                query='verified original',top=1,original_root=self.orig,
                derived_root=self.derived,source_map_path=mapfile,
                native_db_path=nativefile,ledger=self.ledger,
            )
            self.assertEqual(2,run.call_count)
            delay.assert_called_once_with(0.35)
            scan.assert_called_once()
            self.assertEqual(1,result['unique_original_candidates'])
            self.assertEqual(1,result['native_read_sqlite_busy_retry_limit'])
        with patch('scripts.raspi_paired_original_discovery.native_low_memory_search',
                   side_effect=sqlite3.OperationalError('no such table: OOPS')) as run,\
             patch('scripts.raspi_paired_original_discovery.time.sleep') as delay:
            with self.assertRaisesRegex(sqlite3.OperationalError,'no such table'):
                paired_verified_discovery(query='verified original',top=1,
                    original_root=self.orig,derived_root=self.derived,
                    source_map_path=mapfile,native_db_path=nativefile,
                    ledger=self.ledger)
            self.assertEqual(1,run.call_count)
            delay.assert_not_called()
        with patch('scripts.raspi_paired_original_discovery.native_low_memory_search',
                   side_effect=[sqlite_busy(),sqlite_busy()]) as run,\
             patch('scripts.raspi_paired_original_discovery.time.sleep') as delay:
            with self.assertRaises(sqlite3.OperationalError):
                paired_verified_discovery(query='verified original',top=1,
                    original_root=self.orig,derived_root=self.derived,
                    source_map_path=mapfile,native_db_path=nativefile,
                    ledger=self.ledger)
            self.assertEqual(2,run.call_count)
            delay.assert_called_once_with(0.35)


    def test_derived_source_map_revision_switch_during_hash_fails_closed(self):
        from unittest.mock import patch
        from file_system_catalog.source_review_audit import sha_file
        def original_sha_then_change_source_map(path):
            digest=sha_file(path)
            if path==self.derivedfile:
                self.sm.execute('UPDATE originals SET derived_sha256=?',('f'*64,))
            return digest
        with patch('scripts.raspi_paired_original_discovery.sha_file',
                   side_effect=original_sha_then_change_source_map):
            with self.assertRaisesRegex(ValueError,'source-map revision moved'):
                self.check()


class FirstUnitActualSourceFixture(unittest.TestCase):
    def test_existing_original_firstunit_will_find_unsummarized_publisher_bytes(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);original=root/'original';der=root/'derived'
            original.mkdir();der.mkdir()
            path='papers/original.md'
            (original/'papers').mkdir();(der/'papers').mkdir()
            original_file=original/path
            original_file.write_text('Original publisher article about wedding photo collections')
            data=('Catalog metadata header is not publisher evidence.\n'
                  '[[SOURCE UNIT: text section]]\n'
                  'Research about organizing trip wedding photographs into related '
                  'personal photo albums and understanding multi-image content.\n')
            source_file=der/(path+'.txt');source_file.write_text(data)
            source_sha=hashlib.sha256(original_file.read_bytes()).hexdigest()
            derived_sha=hashlib.sha256(source_file.read_bytes()).hexdigest()
            ledger=root/'status.json';ledger.write_text(json.dumps({'items':[
                {'path':path,'sha256':source_sha,'status':'complete'}]}))
            with sqlite3.connect(':memory:') as sm, sqlite3.connect(':memory:') as native:
                sm.execute('create table originals(source_path,source_sha256,derived_sha256,derived_path,original_title,status)')
                sm.execute('insert into originals values(?,?,?,?,?,?)',(
                    path,source_sha,derived_sha,path+'.txt','Personal album original','complete'))
                native.execute('create table filesystem_entries(id,relative_path_display,entry_type)')
                native.execute('insert into filesystem_entries values(?,?,?)',(7,path,'file'))
                def search():
                    return verified_publisher_firstunit_candidates(
                        query='How can agents organize wedding photographs into related trip albums?',top=3,
                        source_map=sm,native_db=native,original_root=original,
                        derivative_root=der,ledger=ledger)
                r=search()
                self.assertEqual(1,r['returned'])
                self.assertEqual(1,r['source_scan']['publisher_firstunit_rows_indexed'])
                hit=r['candidate_originals'][0]
                self.assertEqual(source_sha,hit['publisher_original_sha256'])
                self.assertEqual(derived_sha,hit['current_derivative_sha256'])
                self.assertEqual(7,hit['native_original_filedb_entry_id'])
                self.assertFalse(hit['factual_answer_verified'])
                self.assertFalse(hit['quoted_text_is_original_publisher_bytes'])
                self.assertEqual(hashlib.sha256(hit['publisher_first_unit_preview'].encode()).hexdigest(),
                                 hit['source_preview_sha256'])
                original_file.write_text('Unapproved changed publisher document')
                bad=search()
                self.assertEqual(0,bad['returned'])
                self.assertEqual(1,len(bad['unverified_source_candidates_excluded']))



if __name__=='__main__':
    unittest.main()
