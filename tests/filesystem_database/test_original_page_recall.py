"""Independent PDF page gold, publisher SHA, native candidate proofs only."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from file_system_catalog.original_page_recall import (
    validate_publisher_page_gold,score_source_evidence_pages,
)


def sha(s:bytes)->str:
    return hashlib.sha256(s).hexdigest()


class PublisherSourcePageRecallTests(unittest.TestCase):
    def setUp(self):
        tmp=TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root=Path(tmp.name)
        self.originals=root/'originals';self.originals.mkdir()
        self.derivatives=root/'extracted';self.derivatives.mkdir()
        self.path='papers/verified.pdf'
        self.orig=self.originals/self.path
        self.orig.parent.mkdir()
        # Text bytes here are only a disposable fixture; the independent
        # physical PDF page verifier is injected, never faked as live proof.
        self.orig.write_bytes(b'FAKE PDF ORIGINAL, independently hashed bytes for protocol fixture')
        self.o_sha=sha(self.orig.read_bytes())
        self.derived_path='papers/verified.txt'
        self.derived=self.derivatives/self.derived_path;self.derived.parent.mkdir()
        self.anchor='A source-grounded statement on chapter markers and quote evidence.'
        self.derived.write_text('[[SOURCE UNIT: PDF page 2]]\nOther publisher section content.\n'
                                '[[SOURCE UNIT: PDF page 3]]\n'+self.anchor+'\n')
        self.d_sha=sha(self.derived.read_bytes())
        self.page_sha='c'*64
        self.source_map=sqlite3.connect(':memory:')
        self.addCleanup(self.source_map.close)
        self.source_map.execute('CREATE TABLE originals (source_path TEXT,source_sha256 TEXT,derived_sha256 TEXT,derived_path TEXT,status TEXT)')
        self.source_map.execute('INSERT INTO originals VALUES(?,?,?,?,?)',(
            self.path,self.o_sha,self.d_sha,self.derived_path,'complete'))
        self.case={'id':'quote_from_page3','query':'Which publisher page explains the current source-grounded quote?',
                   'source_original_path':self.path,'original_sha256':self.o_sha,
                   'derived_sha256':self.d_sha,'gold_page':3,
                   'publisher_source_excerpt':self.anchor,
                   'publisher_excerpt_sha256':sha(self.anchor.encode()),
                   'physical_page_verified':True,
                   'original_pdf_physical_page_text_sha256':self.page_sha}
        self.gold={'schema':'kmd.publisher_original_physical_page_gold.v1',
                   'frozen_before_query_replay':True,'cases':[self.case]}
        self.receipts={self.path:self.o_sha}

    def fake_physically_verified(self,original,original_sha,parts):
        self.assertEqual(self.orig,original)
        self.assertEqual(self.o_sha,original_sha)
        self.assertEqual(self.anchor,parts[0]['source_excerpt'])
        return [{'status':'verified','page':3,'page_text_sha256':self.page_sha,
                 'basis':'test_fixture_only_not_actual_pdf'}]

    def validate(self,**kwargs):
        return validate_publisher_page_gold(frozen=kwargs.pop('frozen',self.gold),
                 source_map=self.source_map,original_root=self.originals,
                 derivative_root=self.derivatives,
                 completed_receipts=kwargs.pop('receipts',self.receipts),
                 verifier=kwargs.pop('verifier',self.fake_physically_verified),**kwargs)

    def row(self,page,status='verified'):
        quote='Visible original passage from the PDF, with checked page identity.'
        return {'original_path':self.path,'original_sha256':self.o_sha,
                'derived_sha256':self.d_sha,
                'status':'source_lexical_candidate_not_verified_answer',
                'source_excerpt':quote,'excerpt_sha256':sha(quote.encode()),
                'publisher_pdf_page':{
                    'status':status,'page':page,'page_text_sha256':'e'*64}}

    def search(self,*rows):
        return {'original_path':self.path,'original_sha256':self.o_sha,
                'derived_sha256':self.d_sha,'query':self.case['query'],
                'retrieval_engine':'verified_original_ephemeral_fts5_v1',
                'passages':list(rows)}

    def test_source_original_and_derivative_page_sha_are_reverified_before_query(self):
        verified=self.validate()
        self.assertEqual(1,len(verified))
        self.assertEqual(3,verified[0]['gold_physical_pdf_page'])
        self.assertTrue(verified[0]['original_publisher_and_derived_bytes_rehashed'])
        self.assertEqual(self.page_sha,verified[0]['frozen_original_pdf_page_text_sha256'])

    def test_gold_page_second_counts_and_unverified_other_page_is_not_a_hit(self):
        gold=self.validate()[0]
        row=score_source_evidence_pages(gold=gold,result=self.search(self.row(2),self.row(3)),top=5)
        self.assertEqual(2,row['first_verified_gold_page_rank_at_top'])
        self.assertTrue(row['gold_page_hit'])
        self.assertTrue(row['page_recall_not_answer_truth'])
        row=score_source_evidence_pages(gold=gold,
                                       result=self.search(self.row(3,status='unverified'),self.row(2)),top=5)
        self.assertFalse(row['gold_page_hit'])
        self.assertIsNone(row['first_verified_gold_page_rank_at_top'])

    def test_unrelated_original_revision_or_wrong_query_refused(self):
        gold=self.validate()[0]
        for update in ({'original_path':'other.pdf'},{'query':'wrong question'},
                       {'original_sha256':'a'*64},{'derived_sha256':'f'*64},
                       {'retrieval_engine':'unsafe_ranker'}):
            payload=self.search(self.row(3));payload.update(update)
            with self.subTest(update=update),self.assertRaisesRegex(ValueError,'source-page query differs'):
                score_source_evidence_pages(gold=gold,result=payload,top=5)
        row=self.row(3)
        row['source_excerpt']='changed text'
        with self.assertRaisesRegex(ValueError,'exact digest checked'):
            score_source_evidence_pages(gold=gold,result=self.search(row),top=5)
        row=self.row(3)
        row['status']='model_factual_answer_verified'
        with self.assertRaisesRegex(ValueError,'lacks current publisher'):
            score_source_evidence_pages(gold=gold,result=self.search(row),top=5)

    def test_frozen_source_bytes_changed_or_absent_receipt_rejected(self):
        self.orig.write_bytes(self.orig.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'bytes differ'):
            self.validate()
        self.orig.write_bytes(self.orig.read_bytes()[:-7])
        with self.assertRaisesRegex(ValueError,'source-map identity'):
            self.validate(receipts={})
        self.derived.write_text(self.derived.read_text()+'altered')
        with self.assertRaisesRegex(ValueError,'bytes differ'):
            self.validate()

    def test_gold_page_wrong_or_quote_only_in_metadata_fails_closed(self):
        c=deepcopy(self.gold)
        c['cases'][0]['gold_page']=4
        with self.assertRaisesRegex(ValueError,'expected extracted physical page'):
            self.validate(frozen=c)
        c=deepcopy(self.gold)
        c['cases'][0]['original_pdf_physical_page_text_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'independently reverified'):
            self.validate(frozen=c)
        c=deepcopy(self.gold)
        c['cases'].append(deepcopy(c['cases'][0]))
        with self.assertRaisesRegex(ValueError,'gold lacks independent'):
            self.validate(frozen=c)
        c=deepcopy(self.gold)
        c['cases'][0]['publisher_excerpt_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'gold lacks independent'):
            self.validate(frozen=c)

    def test_untrusted_pdf_source_path_and_oversized_original_are_refused(self):
        with self.assertRaises(ValueError):
            self.validate(max_bytes=5)
        c=deepcopy(self.gold)
        c['cases'][0]['source_original_path']='../unapproved.pdf'
        with self.assertRaises(ValueError):self.validate(frozen=c)
        copy=self.orig.with_name('original-copy.pdf');copy.write_bytes(self.orig.read_bytes())
        self.orig.unlink();self.orig.symlink_to(copy)
        with self.assertRaises(ValueError):self.validate()


if __name__=='__main__':
    unittest.main()
