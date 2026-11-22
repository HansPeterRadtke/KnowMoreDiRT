"""Optional source-level discovery never claims that candidate ranks prove facts."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import unittest
from unittest import mock
import sqlite3
from tempfile import TemporaryDirectory


def load_cli():
    p=Path(__file__).resolve().parents[2]/'scripts/raspi_three_channel_source_discovery.py'
    spec=importlib.util.spec_from_file_location('test_only_optional_source_discovery',p)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OptionalThreeChannelSourceDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.cli=load_cli()

    def test_query_and_result_limit_fail_before_model_session_open(self):
        with mock.patch.object(self.cli,'LocalE5',side_effect=AssertionError('model opened')):
            for query,top in ((' ',8),('q',10),('normal question',0),('normal question',11),('x'*601,8),('query',True)):
                with self.subTest(query=query[:20],top=top):
                    with self.assertRaises(ValueError):self.cli.read_only_source_discovery(query,top=top)

    def test_three_independent_source_channels_ambiguous_answer_scope(self):
        with TemporaryDirectory() as folder:
            db=sqlite3.connect(':memory:')
            db.execute('PRAGMA query_only=ON')
            self.addCleanup(db.close)
            path='known/source.html'
            snap={'source_revisions':{path:('a'*64,'b'*64,111)},
                  'metadata':{path:{}},'receipt':{path:{}},'catalog':{path:{}},
                  'load_counts':{'complete_original_paths':1}}
            responses={'original_passage':[path],'summary':[path],'lexical':[path]}
            recorded=[]
            def proof(entry,**kwargs):
                recorded.append(entry)
                return {'original_path':entry,'publisher_original_actual_sha256_verified':True,
                  'source_map_derivative_actual_sha256_verified':True,
                  'factual_answer_or_claim_supported_by_retrieval_alone':False}
            with mock.patch.object(self.cli,'LocalE5',return_value=object()), \
                 mock.patch.object(self.cli,'capture_source_channels',return_value=snap), \
                 mock.patch.object(self.cli,'title_summary_fts',return_value=(object(),1)), \
                 mock.patch.object(self.cli,'query_channels',return_value=responses), \
                 mock.patch.object(self.cli,'checked_sqlite',return_value=db), \
                 mock.patch.object(self.cli,'verify_selected_source',side_effect=proof):
                got=self.cli.read_only_source_discovery('Find the original publication.',top=1)
            self.assertEqual([path],recorded)
            self.assertEqual(1,got['all_returned_originals_physically_sha_verified'])
            self.assertTrue(got['separate_original_channels_no_rank_fusion_or_source_suppression'])
            self.assertTrue(got['source_navigation_only_not_a_generated_answer'])
            for name in ('original_passage','summary','lexical'):
                self.assertEqual(path,got['channels'][name][0]['source']['original_path'])
                self.assertFalse(got['channels'][name][0]['source']['factual_answer_or_claim_supported_by_retrieval_alone'])
            self.assertNotIn('answer',got)
            self.assertNotIn('confidence',got)

    def test_uncitable_source_aborts_all_results_not_a_rank_substitution(self):
        with TemporaryDirectory():
            db=sqlite3.connect(':memory:');db.execute('PRAGMA query_only=ON');self.addCleanup(db.close)
            path='one/paper.pdf';snap={'source_revisions':{},'metadata':{},'receipt':{},'catalog':{},
                                     'load_counts':{'complete_original_paths':1}}
            with mock.patch.object(self.cli,'LocalE5',return_value=object()), \
                 mock.patch.object(self.cli,'capture_source_channels',return_value=snap), \
                 mock.patch.object(self.cli,'title_summary_fts',return_value=(object(),1)), \
                 mock.patch.object(self.cli,'query_channels',return_value={'original_passage':[path],'summary':[],'lexical':[]}), \
                 mock.patch.object(self.cli,'checked_sqlite',return_value=db), \
                 mock.patch.object(self.cli,'verify_selected_source',side_effect=ValueError('stale source')):
                with self.assertRaisesRegex(ValueError,'stale source'):
                    self.cli.read_only_source_discovery('Find the original.',top=5)

if __name__=='__main__':unittest.main()
