"""Optional physical PDF preview verification must not imply answer verification."""
from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from file_system_catalog.dual_source_discovery import attach_verified_pdf_previews


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DualPdfPreviewProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.pdf=self.root/'originals/paper.pdf'
        self.pdf.parent.mkdir()
        self.pdf.write_bytes(b'%PDF-1.5 example preserved bytes')
        self.receipts={'originals/paper.pdf': digest(self.pdf.read_bytes())}
        self.preview='This is the original passage about implementation of the method.'
        self.result={
            'schema':'kmd.two_channel_original_discovery.v1',
            'summary_channel':[{'original_path':'originals/paper.pdf','summary_rank':1}],
            'original_passage_channel':[
                {'original_path':'plain/page.txt','original_source_rank':1,
                 'publisher_revision_verified':True,'passage_preview':'text','source_section':'Original plain text'},
                {'original_path':'originals/paper.pdf','original_source_rank':2,
                 'publisher_revision_verified':True,'passage_preview':self.preview,'source_section':'PDF page 3'},
            ],
        }
        self.proof={'status':'verified','page':3,'basis':'derivative_label_confirmed',
                    'page_text_sha256':'f'*64}
        self.verifier=Mock(return_value=[self.proof])

    def run_proof(self, **kwargs):
        return attach_verified_pdf_previews(
            self.result,original_root=self.root,original_receipts=self.receipts,
            max_pdf_previews=kwargs.pop('max_pdf_previews',2),verifier=self.verifier,**kwargs)

    def test_verified_pdf_matches_display_preview_only_and_keeps_rank(self):
        result=self.run_proof()
        text,pdf=result['original_passage_channel']
        self.assertEqual('not_pdf',text['physical_pdf_preview_check']['status'])
        self.assertEqual(2,pdf['original_source_rank'])
        self.assertEqual(3,pdf['physical_pdf_preview_check']['physical_pdf_page'])
        self.assertIn('displayed derivative excerpt',pdf['physical_pdf_preview_check']['scope'])
        self.assertEqual(1,result['physical_pdf_preview_audit']['original_pdf_previews_physically_verified'])
        self.assertEqual(1,result['physical_pdf_preview_audit']['original_pdf_previews_checked'])
        self.assertNotIn('physical_pdf_preview_check',self.result['original_passage_channel'][1])
        self.verifier.assert_called_once_with(self.pdf,self.receipts['originals/paper.pdf'],
            [{'source_section':'PDF page 3','source_excerpt':self.preview}])

    def test_non_matching_text_must_remain_unverified(self):
        self.verifier.return_value=[{'status':'unverified','reason':'excerpt_absent'}]
        result=self.run_proof()
        self.assertEqual('unverified',result['original_passage_channel'][1]['physical_pdf_preview_check']['status'])
        self.assertEqual('excerpt_absent',result['original_passage_channel'][1]['physical_pdf_preview_check']['reason'])
        self.assertEqual(0,result['physical_pdf_preview_audit']['original_pdf_previews_physically_verified'])

    def test_bounded_pdf_budget_marks_unchecked_without_false_page(self):
        second=self.root/'other.pdf';second.write_bytes(b'%PDF-1.5 second original')
        self.receipts['other.pdf']=digest(second.read_bytes())
        self.result['original_passage_channel'].append(dict(self.result['original_passage_channel'][1],original_path='other.pdf',original_source_rank=3))
        result=self.run_proof(max_pdf_previews=1)
        self.assertEqual('not_checked_limit',result['original_passage_channel'][2]['physical_pdf_preview_check']['status'])
        self.assertEqual(1,result['physical_pdf_preview_audit']['original_pdf_previews_checked'])
        self.assertEqual(1,self.verifier.call_count)

    def test_missing_or_changed_receipt_or_symlink_fail_closed(self):
        for receipts in ({}, {'originals/paper.pdf':'wrong'}):
            with self.subTest(receipts=receipts):
                self.receipts=receipts
                with self.assertRaises(ValueError):self.run_proof()
        self.receipts={'originals/paper.pdf':digest(self.pdf.read_bytes())}
        self.pdf.unlink()
        self.pdf.symlink_to(self.root/'other.pdf')
        with self.assertRaises(ValueError):self.run_proof()
        self.verifier.assert_not_called()

    def test_verifier_errors_or_invalid_proof_fail_closed(self):
        for output in ([],[{'status':'verified','page':0,'page_text_sha256':'f'*64}],
                       [{'status':'verified','page':3,'page_text_sha256':'bad'}],
                       [{'status':'maybe'}]):
            with self.subTest(output=output):
                self.verifier.return_value=output
                with self.assertRaises(ValueError):self.run_proof()
        self.verifier.side_effect=ValueError('publisher PDF changed during verification')
        with self.assertRaisesRegex(ValueError,'changed'):
            self.run_proof()

    def test_invalid_budget_and_unverified_upstream_proof_rejected(self):
        for v in (0,6,True):
            with self.subTest(value=v):
                with self.assertRaises(ValueError):self.run_proof(max_pdf_previews=v)
        self.result['original_passage_channel'][1]['publisher_revision_verified']=False
        with self.assertRaises(ValueError):self.run_proof()


if __name__=='__main__':
    unittest.main()
