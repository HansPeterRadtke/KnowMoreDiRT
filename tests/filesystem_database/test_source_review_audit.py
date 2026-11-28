"""Post-hoc source diagnostic must never trust changed/quoted model evidence."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from file_system_catalog.source_review_audit import validate_reviews


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class SourceReviewProofTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.orig=self.root/'originals';self.orig.mkdir()
        self.deriv=self.root/'derived';self.deriv.mkdir()
        self.path='research/paper.html'
        p=self.orig/self.path;p.parent.mkdir()
        p.write_bytes(b'<html><p>Official publisher original.</p></html>')
        self.s0=digest(p.read_bytes())
        self.derived_path='safe/paper.txt'
        d=self.deriv/self.derived_path;d.parent.mkdir()
        self.source='KMD source provenance title: TEST CLAIM IN HEADER\n[[SOURCE UNIT: HTML original publisher document]]\nOfficial sentence supporting some related statement.\n'
        d.write_bytes(self.source.encode())
        self.s1=digest(d.read_bytes())
        self.map=sqlite3.connect(':memory:')
        self.addCleanup(self.map.close)
        self.map.execute('CREATE TABLE originals (source_path TEXT, source_sha256 TEXT,derived_sha256 TEXT,derived_path TEXT,status TEXT)')
        self.map.execute('INSERT INTO originals VALUES(?,?,?,?,?)',(self.path,self.s0,self.s1,self.derived_path,'complete'))
        self.receipts={self.path:self.s0}
        self.shadow={'schema':'kmd.read_only_raw_overview_vs_hierarchical_summary_shadow.v1','cases':[
            {'family':'historical','id':'q1','known_originals':['research/expected.html'],
             'raw_top_three':[self.path],'summary_top_three':['research/another.html']}
        ]}
        excerpt='Official sentence supporting some related statement.'
        self.review={'schema':'kmd.posthoc_top_three_source_review.v1','cases':[
            {'family':'historical','case_id':'q1','candidate_path':self.path,
             'review_grade':'partial_support','review_notes':'This is a bounded related source, not complete primary evidence.',
             'original_sha256':self.s0,'derived_sha256':self.s1,
             'evidence_anchors':[{'source_start_char':self.source.index(excerpt),
                                  'exact_derived_excerpt':excerpt,'excerpt_sha256':digest(excerpt.encode())}]}
        ]}

    def check(self,review=None,**kwargs):
        return validate_reviews(shadow=kwargs.pop('shadow',self.shadow),
                                review=review or self.review,source_map=self.map,
                                original_root=self.orig,derived_root=self.deriv,
                                receipts=kwargs.pop('receipts',self.receipts),**kwargs)

    def test_real_source_and_citable_derivative_phrase_are_verified_not_relevance(self):
        response=self.check()
        self.assertEqual('POSTHOC, manually interpreted, NOT independent or exhaustive relevance gold',response['review_status'])
        self.assertEqual(1,response['reviewed'])
        self.assertEqual({'partial_support':1},response['grade_counts'])
        proof=response['cases'][0]
        self.assertEqual(1,proof['matched_original_overview_rank'])
        self.assertIsNone(proof['matched_summary_rank'])
        self.assertEqual(self.s0,proof['source_original_sha256_reverified'])
        self.assertEqual(self.s1,proof['source_derivative_sha256_reverified'])
        self.assertEqual(1,len(proof['source_text_anchors_sha_verified']))
        self.assertFalse(response['native_corpus_or_ranker_changed'])

    def test_prologue_is_not_publisher_source_evidence(self):
        r=deepcopy(self.review)
        excerpt='TEST CLAIM IN HEADER'
        a=r['cases'][0]['evidence_anchors'][0]
        a.update(source_start_char=self.source.index(excerpt),
                 exact_derived_excerpt=excerpt,excerpt_sha256=digest(excerpt.encode()))
        with self.assertRaisesRegex(ValueError,'publisher text'):
            self.check(r)

    def test_mutated_pub_and_derivative_cannot_support_stale_claim(self):
        p=self.orig/self.path
        original=p.read_bytes();p.write_bytes(original.replace(b'publisher',b'adversary'))
        with self.assertRaisesRegex(ValueError,'original publisher source bytes changed'):
            self.check()
        p.write_bytes(original)
        d=self.deriv/self.derived_path
        raw=d.read_bytes();d.write_bytes(raw.replace(b'Official sentence',b'Fabricated text'))
        with self.assertRaisesRegex(ValueError,'derivative source bytes changed'):
            self.check()
        d.write_bytes(raw)
        self.assertEqual(1,self.check()['reviewed'])

    def test_missing_receipt_or_changed_source_revision_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'source-map/receipt'):
            self.check(receipts={})
        with self.assertRaisesRegex(ValueError,'source-map/receipt'):
            self.check(receipts={self.path:'a'*64})
        r=deepcopy(self.review)
        r['cases'][0]['derived_sha256']='a'*64
        with self.assertRaisesRegex(ValueError,'source-map/receipt'):
            self.check(r)

    def test_unknown_candidate_duplicates_or_gold_tampering_rejected(self):
        r=deepcopy(self.review)
        r['cases'][0]['candidate_path']='research/other.html'
        with self.assertRaises(ValueError):self.check(r)
        r=deepcopy(self.review)
        r['cases'].append(deepcopy(r['cases'][0]))
        with self.assertRaisesRegex(ValueError,'duplicate'):
            self.check(r)
        shadow=deepcopy(self.shadow)
        shadow['cases'][0]['raw_top_three']=[]
        with self.assertRaisesRegex(ValueError,'frozen top-three'):
            self.check(shadow=shadow)
        shadow=deepcopy(self.shadow)
        shadow['cases'][0]['known_originals']=[self.path]
        with self.assertRaisesRegex(ValueError,'predeclared original gold'):
            self.check(shadow=shadow)

    def test_wrong_phrase_offset_or_sha_must_not_pass(self):
        r=deepcopy(self.review)
        r['cases'][0]['evidence_anchors'][0]['source_start_char']+=1
        with self.assertRaisesRegex(ValueError,'quoted review span'):
            self.check(r)
        r=deepcopy(self.review)
        r['cases'][0]['evidence_anchors'][0]['excerpt_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'quoted review span'):
            self.check(r)

    def test_changed_status_and_symlinked_source_refused(self):
        self.map.execute('UPDATE originals SET status=?',('error',))
        with self.assertRaisesRegex(ValueError,'source-map/receipt'):
            self.check()
        self.map.execute('UPDATE originals SET status=?',('complete',))
        original=self.orig/self.path
        copy=self.root/'target.html';copy.write_bytes(original.read_bytes())
        original.unlink();original.symlink_to(copy)
        with self.assertRaisesRegex(ValueError,'symlink'):
            self.check()

    def turn_source_into_pdf(self):
        newpath="research/paper.pdf"
        (self.orig/self.path).rename(self.orig/newpath)
        original_sha=digest((self.orig/newpath).read_bytes())
        self.source=self.source.replace('HTML original publisher document','PDF page 2')
        d=self.deriv/self.derived_path
        d.write_bytes(self.source.encode())
        self.s1=digest(d.read_bytes())
        self.map.execute('UPDATE originals SET source_path=?,source_sha256=?,derived_sha256=?',
                         (newpath,original_sha,self.s1))
        self.receipts={newpath:original_sha}
        self.shadow['cases'][0]['raw_top_three']=[newpath]
        row=self.review['cases'][0]
        row['candidate_path']=newpath
        row['original_sha256']=original_sha
        row['derived_sha256']=self.s1
        phrase=row['evidence_anchors'][0]['exact_derived_excerpt']
        row['evidence_anchors'][0]['source_start_char']=self.source.index(phrase)
        return newpath

    def test_extractor_second_pdf_label_only_cannot_verify_review_quote(self):
        name=self.turn_source_into_pdf()
        actual=self.source
        second='[[SOURCE UNIT: PDF page 13]]'
        paragraph='This genuine second source page describes a different research claim.'
        self.source=actual+second+chr(10)+paragraph+chr(10)
        derivative=self.deriv/self.derived_path
        derivative.write_text(self.source)
        new_sha=digest(derivative.read_bytes())
        self.map.execute('UPDATE originals SET derived_sha256=? WHERE source_path=?',
                         (new_sha,name))
        quote=second
        r=deepcopy(self.review)
        r['cases'][0]['derived_sha256']=new_sha
        r['cases'][0]['evidence_anchors']=[
            {'source_start_char':self.source.index(quote),
             'exact_derived_excerpt':quote,
             'excerpt_sha256':digest(quote.encode())}]
        with self.assertRaisesRegex(ValueError,'publisher text body'):
            self.check(r)
        # A full source-unit boundary must not be crossed and described
        # as one original publisher quote either.
        tail='statement.'+chr(10)+second+chr(10)+'This genuine'
        r['cases'][0]['evidence_anchors']=[
            {'source_start_char':self.source.index(tail),
             'exact_derived_excerpt':tail,
             'excerpt_sha256':digest(tail.encode())}]
        with self.assertRaisesRegex(ValueError,'publisher text body'):
            self.check(r)
        # Positive second-page original body still works, by itself.
        r['cases'][0]['evidence_anchors']=[
            {'source_start_char':self.source.index(paragraph),
             'exact_derived_excerpt':paragraph,
             'excerpt_sha256':digest(paragraph.encode())}]
        result=self.check(r)
        self.assertEqual(1,result['reviewed'])
        self.assertEqual(paragraph,self.source[
            result['cases'][0]['source_text_anchors_sha_verified'][0]['start_char']:
            result['cases'][0]['source_text_anchors_sha_verified'][0]['end_char']])

    def test_unclosed_later_source_unit_cannot_prove_review_text(self):
        name=self.turn_source_into_pdf()
        self.source += '[[SOURCE UNIT: PDF page '+('X'*520)+chr(10)
        derivative=self.deriv/self.derived_path
        derivative.write_text(self.source)
        new_sha=digest(derivative.read_bytes())
        self.map.execute('UPDATE originals SET derived_sha256=? WHERE source_path=?',
                         (new_sha,name))
        r=deepcopy(self.review)
        r['cases'][0]['derived_sha256']=new_sha
        with self.assertRaisesRegex(ValueError,'incomplete or untrusted publisher source-unit'):
            self.check(r)

    def test_pdf_physical_page_proof_is_separate_from_relevance_grade(self):
        source=self.turn_source_into_pdf()
        mock_verifier=mock.Mock(return_value=[{'status':'verified','page':2,
            'page_text_sha256':'f'*64,'basis':'derivative_label_confirmed'}])
        data=self.check(verify_physical_pdf_pages=True,pdf_verifier=mock_verifier)
        self.assertEqual({'partial_support':1},data['grade_counts'])
        proof=data['cases'][0]['physical_original_pdf_pages'][0]
        self.assertEqual('verified',proof['status'])
        self.assertEqual(2,proof['page'])
        self.assertEqual(1,mock_verifier.call_count)
        self.assertEqual(self.orig/source,mock_verifier.call_args.args[0])
        self.assertEqual('PDF page 2',mock_verifier.call_args.args[2][0]['source_section'])
        self.assertTrue(data['physical_pdf_page_verification_requested'])

    def test_wrong_pdf_page_is_not_silently_promoted(self):
        self.turn_source_into_pdf()
        verifier=mock.Mock(return_value=[{'status':'unverified','reason':'excerpt_absent'}])
        data=self.check(verify_physical_pdf_pages=True,pdf_verifier=verifier)
        self.assertEqual('unverified',data['cases'][0]['physical_original_pdf_pages'][0]['status'])
        verifier.return_value=[{'status':'verified','page':0,'page_text_sha256':'f'*64}]
        with self.assertRaisesRegex(ValueError,'malformed verified PDF'):
            self.check(verify_physical_pdf_pages=True,pdf_verifier=verifier)
        verifier.return_value=[]
        with self.assertRaisesRegex(ValueError,'cardinality'):
            self.check(verify_physical_pdf_pages=True,pdf_verifier=verifier)
        verifier.side_effect=ValueError('source changed mid-verification')
        with self.assertRaisesRegex(ValueError,'source changed'):
            self.check(verify_physical_pdf_pages=True,pdf_verifier=verifier)

    def test_non_pdf_is_explicitly_not_a_physical_page_proof(self):
        data=self.check(verify_physical_pdf_pages=True)
        self.assertEqual('not_pdf',data['cases'][0]['physical_original_pdf_pages'][0]['status'])
        with self.assertRaisesRegex(ValueError,'boolean flag'):
            self.check(verify_physical_pdf_pages='yes')

    def test_invalid_review_labels_cannot_be_promoted_as_truth(self):
        for label in ('known_relevant', 'verified_true', 'fake', ''):
            r=deepcopy(self.review)
            r['cases'][0]['review_grade']=label
            with self.subTest(label=label),self.assertRaisesRegex(ValueError,'unsupported manual'):
                self.check(r)


if __name__=='__main__':
    unittest.main()
