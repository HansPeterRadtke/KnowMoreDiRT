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


if __name__=='__main__':
    unittest.main()
