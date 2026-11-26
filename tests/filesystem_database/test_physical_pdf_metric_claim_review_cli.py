"""Physical publisher PDF model/metric evidence must retain current native source identity."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch


def module():
    path=Path(__file__).resolve().parents[2]/'scripts/raspi_physical_pdf_metric_claim_review.py'
    scripts=str(path.parent)
    if scripts not in sys.path:
        sys.path.insert(0,scripts)
    spec=importlib.util.spec_from_file_location('physical_pdf_model_metric_cli_under_test',path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CurrentNativePdfModelMetricReview(TestCase):
    def setUp(self):
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.orig=self.root/'Originals';self.orig.mkdir()
        self.derived=self.root/'Text';self.derived.mkdir()
        self.path='papers/agentmem.pdf'
        orig=self.orig/self.path;orig.parent.mkdir(parents=True)
        orig.write_bytes(b'A completed approved publisher original PDF fixture')
        self.original_sha=hashlib.sha256(orig.read_bytes()).hexdigest()
        source=self.derived/'papers/agentmem.pdf.txt';source.parent.mkdir(parents=True)
        source.write_text('[[SOURCE UNIT: PDF page 8]]\nAgentMemRec and T-Mem on paper table.\n')
        self.derived_sha=hashlib.sha256(source.read_bytes()).hexdigest()
        self.ledger=self.root/'completed.json'
        self.ledger.write_text(json.dumps({'items':[{'path':self.path,'sha256':self.original_sha,'status':'complete'}]}))
        self.sm=self.root/'source-map.sqlite3'
        self.native=self.root/'filedb.sqlite3'
        with sqlite3.connect(self.sm) as con:
            con.execute('CREATE TABLE originals(source_path TEXT,source_sha256 TEXT,derived_sha256 TEXT,derived_path TEXT,status TEXT)')
            con.execute('INSERT INTO originals VALUES(?,?,?,?,?)',(self.path,self.original_sha,self.derived_sha,'papers/agentmem.pdf.txt','complete'))
        with sqlite3.connect(self.native) as con:
            con.execute('CREATE TABLE filesystem_entries(id INTEGER, relative_path_display TEXT, entry_type TEXT)')
            con.execute('INSERT INTO filesystem_entries VALUES (19,?,?)',(self.path,'file'))
        self.app=module()
        self.structure={'current_native_summary_nodes_fully_verified':2,'derived_sha256':self.derived_sha}
        self.generated='AgentMemRec has LoCoMo result 80.3% accuracy, while on another test it reports 83.2%.'
        self.native_doc={
            'schema':'kmd.current_original_root_claim_evidence_review.v1',
            'original_path':self.path,
            'original_sha256':self.original_sha,'derived_sha256':self.derived_sha,
            'current_native_root_summary_sha256':'b'*64,
            'verified_complete_native_source_tree':self.structure,
            'original_and_native_tree_checked_again_after_review':True,
            'all_native_node_claim_reviews':[
                {'node_id':'summary:leaf','level':0,
                 'original_source_start_char':0,'original_source_end_char':300,
                 'native_summary_sha256':'b'*64,
                 'claim_evidence':{'claims':[
                     {'claim_text':self.generated,'matched_value':'80.3%',
                      'claim_sha256':'c'*64,'source_numeric_text_match_kind':'bare_number_no_unit',
                      'source_evidence':[{'source_unit_label':'PDF page 8','matched_anchor_char':70}]},
                     {'claim_text':self.generated,'matched_value':'83.2%',
                      'claim_sha256':'c'*64,'source_numeric_text_match_kind':'bare_number_no_unit',
                      'source_evidence':[{'source_unit_label':'PDF page 8','matched_anchor_char':49}]},
                 ]}}
            ],
        }
        self.words=self.fixture_words()
        self.kw=dict(original_path=self.path,physical_page=8,focus_model='AgentMemRec',
                     other_models=['T-Mem','T-Mem, reprod.'],comparison_numbers=['77.0','77.4'],metric_headings=[],
                     original_root=self.orig,derived_root=self.derived,
                     source_map=self.sm,native_db=self.native,completed_ledger=self.ledger)

    def fixture_words(self):
        from file_system_catalog.physical_pdf_metric_review import Word
        return [Word('AgentMemRec',56,241.28,57.42,3.7,8),Word('83.2',136.65,241.28,14.23,3.7,8),
                Word('T-Mem',173.43,224.87,23.53,7.44,8),Word('80.3',277.22,226.8,14.23,3.7,8),
                Word('T-Mem,',173.43,254.76,25.28,7.44,8),Word('reprod.',200.71,254.76,23.36,7.44,8),
                Word('77.4',278.58,254.76,12.87,7.44,8),Word('AgentMemRec',173.43,271.16,57.42,3.7,8),
                Word('77.0',278.58,269.23,12.87,7.44,8)]

    def patched_call(self,*,after=None):
        return (
            patch.object(self.app,'reviewed_native_root',return_value=self.native_doc),
            patch.object(self.app,'audit_current_source_tree',return_value=self.structure if after is None else after),
            patch.object(self.app,'read_hashed_original_pdf_page_tsv',return_value=(self.words,
                {'publisher_physical_page':8,'pdf_tsv_sha256':'a'*64,
                 'publisher_original_sha256':self.original_sha,'source_file_same_open_fd_sha256_verified':True})),
        )

    def test_unique_original_and_real_geometry_compared_to_current_native_claim(self):
        from contextlib import ExitStack
        with ExitStack() as context:
            for mocker in self.patched_call():context.enter_context(mocker)
            report=self.app.review_current_pdf_summary_metrics(**self.kw)
        self.assertEqual(self.original_sha,report['approved_publisher_original_sha256'])
        self.assertEqual(self.derived_sha,report['current_extracted_source_sha256'])
        self.assertEqual(19,report['current_native_original_document_id'])
        self.assertEqual(2,report['num_current_native_nodes_reverified'])
        cells={x['number']:x for x in report['original_pdf_numeric_table_word_candidates']['numeric_cell_candidates']}
        self.assertEqual('T-Mem',cells['80.3']['unique_explicit_named_model_in_same_row_if_any'])
        self.assertEqual('AgentMemRec',cells['77.0']['unique_explicit_named_model_in_same_row_if_any'])
        self.assertEqual('AgentMemRec',cells['83.2']['unique_explicit_named_model_in_same_row_if_any'])
        self.assertEqual('T-Mem, reprod.',cells['77.4']['unique_explicit_named_model_in_same_row_if_any'])
        reviews={x['claimed_numeric_percent']:x for x in report['current_generated_claims_containing_focus_model']}
        self.assertEqual('numeric_value_near_other_named_model_but_not_focus_model_review_original_table',reviews['80.3%']['source_geometry_review_cue'])
        self.assertEqual('focus_model_appears_near_at_least_one_matching_number_review_metric_column',reviews['83.2%']['source_geometry_review_cue'])
        self.assertTrue(report['comparison_model_and_metric_truth_not_automatically_verified'])
        self.assertTrue(report['native_summary_vectors_and_original_source_files_unchanged'])
        self.assertTrue(report['source_and_entire_native_summary_graph_checked_after_physical_page_read'])

    def test_stale_revision_after_original_pdf_scan_aborts_before_result(self):
        from contextlib import ExitStack
        with ExitStack() as context:
            for patcher in self.patched_call(after={'current_native_summary_nodes_fully_verified':3}):
                context.enter_context(patcher)
            with self.assertRaisesRegex(ValueError,'source/native tree changed'):
                self.app.review_current_pdf_summary_metrics(**self.kw)

    def test_no_native_unique_entry_and_unapproved_original_abort(self):
        from contextlib import ExitStack
        with sqlite3.connect(self.native) as con:
            con.execute('DELETE FROM filesystem_entries')
        with ExitStack() as context:
            for patcher in self.patched_call():context.enter_context(patcher)
            with self.assertRaisesRegex(ValueError,'unique original native FileDB'):
                self.app.review_current_pdf_summary_metrics(**self.kw)
        self.ledger.write_text(json.dumps({'items':[{'path':self.path,'sha256':self.original_sha,'status':'error'}]}))
        with self.assertRaisesRegex(ValueError,'unique approved'):
            self.app.review_current_pdf_summary_metrics(**self.kw)

    def test_cli_does_not_replace_existing_private_evidence_or_modify_source_files(self):
        from contextlib import ExitStack
        result=self.root/'output.json'
        result.write_text('{"protected":"owner original evidence"}')
        argv=['metric_cli','--original-path',self.path,'--physical-page','8',
              '--focus-model','AgentMemRec','--output',str(result),
              '--original-root',str(self.orig),'--derived-root',str(self.derived),
              '--source-map',str(self.sm),'--native-db',str(self.native),
              '--completed-ledger',str(self.ledger)]
        original_sha=hashlib.sha256((self.orig/self.path).read_bytes()).hexdigest()
        with patch.object(sys,'argv',argv),self.assertRaisesRegex(ValueError,'fresh private JSON'):
            self.app.main()
        self.assertEqual('{"protected":"owner original evidence"}',result.read_text())
        self.assertEqual(original_sha,hashlib.sha256((self.orig/self.path).read_bytes()).hexdigest())
        result.unlink()
        with ExitStack() as context:
            for patcher in self.patched_call():context.enter_context(patcher)
            with patch.object(sys,'argv',argv):
                self.app.main()
        self.assertTrue(result.is_file())
        self.assertEqual(0o600,result.stat().st_mode&0o777)
        d=json.loads(result.read_text())
        self.assertEqual(2,len(d['current_generated_claims_containing_focus_model']))
        self.assertFalse(not d['native_summary_vectors_and_original_source_files_unchanged'])
        self.assertEqual(original_sha,hashlib.sha256((self.orig/self.path).read_bytes()).hexdigest())


if __name__=='__main__':
    import unittest
    unittest.main()
