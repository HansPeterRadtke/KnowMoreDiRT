"""Source-specific dual search may not silently substitute another original."""
from __future__ import annotations

from copy import deepcopy
import unittest

from file_system_catalog.dual_source_discovery import scope_explicit_publisher_query


class SourceScopedDualReaderTests(unittest.TestCase):
    def setUp(self):
        self.query='According to CiteGuard, what method is proposed?'
        self.path='research/citeguard.pdf'
        self.other='research/unrelated.pdf'
        self.base={'query':self.query,'documents':[
            {'original_path':self.path,'best_summary':'correct'},
            {'original_path':self.other,'best_summary':'other'},
        ]}

    def route(self,match=None,docs=None,**kw):
        payload=deepcopy(self.base)
        if match is not None:payload['source_name_match']=match
        if docs is not None:payload['documents']=docs
        return scope_explicit_publisher_query(payload,kw.get('query',self.query))

    def test_general_topic_keeps_both_independent_channels_unrestricted(self):
        original,scope,run,metadata=self.route()
        self.assertEqual(self.base,original)
        self.assertIsNone(scope)
        self.assertTrue(run)
        self.assertIsNone(metadata)

    def test_verified_explicit_original_never_keeps_unrelated_summary_documents(self):
        match={'status':'verified','kind':'catalog_title_lead',
               'matching_catalog_paths':1,'original_path':self.path}
        native,scope,run,selected=self.route(match)
        self.assertEqual(self.path,scope)
        self.assertTrue(run)
        self.assertEqual([self.path],[z['original_path'] for z in native['documents']])
        self.assertEqual(self.path,selected['original_path'])
        self.assertEqual('verified',selected['status'])
        self.assertNotIn(self.other,str(native))

    def test_named_unindexed_original_filters_all_summary_substitutes(self):
        match={'status':'no_verified_index_match','kind':'catalog_title_lead',
               'matching_catalog_paths':1,'original_path':self.path}
        native,scope,run,selected=self.route(match,docs=[{'original_path':self.other}])
        self.assertEqual([],native['documents'])
        self.assertEqual(self.path,scope)
        self.assertTrue(run)  # Original source passage can be independently indexed.
        self.assertEqual('no_verified_index_match',selected['status'])

    def test_ambiguous_and_missing_pub_do_not_run_any_global_original_vector_search(self):
        for match in (
            {'status':'ambiguous','matching_catalog_paths':3,
             'candidate_paths':['papers/a.pdf','papers/b.pdf','papers/c.pdf']},
            {'status':'no_catalog_match'},
        ):
            with self.subTest(status=match['status']):
                native,scope,run,selected=self.route(match)
                self.assertEqual([],native['documents'])
                self.assertIsNone(scope)
                self.assertFalse(run)
                self.assertEqual(match['status'],selected['status'])
        choices=[f'papers/{z:02d}.pdf' for z in range(8)]
        native,scope,run,selected=self.route({'status':'ambiguous',
                      'matching_catalog_paths':19,'candidate_paths':choices})
        self.assertTrue(selected['candidate_choices_truncated'])
        self.assertEqual(19,selected['matching_catalog_paths'])
        self.assertFalse(run)

    def test_malformed_reference_and_stale_source_revision_fail_closed(self):
        valid={'status':'verified','kind':'catalog_title_lead',
               'matching_catalog_paths':1,'original_path':self.path}
        for invalid in (
            {'status':'maybe'},
            {'status':'verified','matching_catalog_paths':1,'original_path':'../evil'},
            {'status':'verified','matching_catalog_paths':2,'original_path':self.path},
            {'status':'verified','matching_catalog_paths':1,'original_path':self.path},
            {'status':'ambiguous','matching_catalog_paths':1,'candidate_paths':['a.pdf']},
            {'status':'ambiguous','matching_catalog_paths':3,'candidate_paths':['a.pdf','a.pdf']},
        ):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                self.route(invalid,docs=[{'original_path':self.other}])
        with self.assertRaises(ValueError):
            self.route(valid,query='a changed query was substituted')
        with self.assertRaises(ValueError):
            scope_explicit_publisher_query({'query':self.query,'source_name_match':'fake'},self.query)


if __name__=='__main__':
    unittest.main()
