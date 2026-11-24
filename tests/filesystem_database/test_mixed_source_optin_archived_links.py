"""Private opt-in archived source references in existing native mixed search.

All fixtures are isolated; the live source-map, original collection and native
FileDB are never modified. Native PDF rank and normal search remain unchanged.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import json
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))

from test_mixed_source_evidence import MixedSourceEvidenceTests
import raspi_archived_html_follow_links as archived
import raspi_archived_markdown_follow_links as markdown


class NativeMixedArchivedReferenceTests(unittest.TestCase):
    def setUp(self):
        fixture=MixedSourceEvidenceTests(methodName='test_mixed_document_rank_distinguishes_derived_html_from_physically_verified_pdf')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.fixture=fixture;self.cli=fixture.cli
        self.args=fixture.args
        self.parent=fixture.paths['html']
        self.args.follow_archived_links=True
        self.args.linked_originals=2
        self.child='docs/publisher-approved-child.html'
        self.link={
          'original_path':self.parent,'parent_original_path':self.parent,
          'parent_original_sha256':'a'*64,'parent_derived_sha256':'b'*64,
          'linked_original_sources_checked':1,'linked_source_excerpt_candidates':1,
          'publisher_original_article_evidence_passages':1,
          'linked_existing_original_sources':[{
            'original_path':self.child,'original_sha256':'a'*64,
            'derived_sha256':'b'*64,'native_filedb_original_id':51,
            'passage_candidates':[{'extracted_text_excerpt_candidate':'Evidence in archived source.',
              'factual_answer_sufficiency_verified':False}],
            'link_provenance':{'publisher_article_link_text':'Official archived child',
               'candidate_not_answer':True}}],
          'parent_original_HTML_article_link_proof':{
            'same_origin_relevant_unarchived_or_unapproved_references':1,
            'unavailable_publisher_references_not_followed':[{
              'publisher_article_link_text':'Missing official guide',
              'publisher_destination_contents_unavailable':True,
              'not_a_source_excerpt_or_answer':True}]},
          'not_proof_of_correct_answer_or_atomic_cross_store_generation':True,
          'safe_to_cutover_live':False,
        }
        self.reader=mock.patch.object(archived,'find_archived_linked_source_passages',return_value=self.link)
        self.follow=self.reader.start();self.addCleanup(self.reader.stop)
        self.version=mock.patch.object(self.cli,'final_original_revision_guard',
            return_value={'latest_source_map_revision_and_native_file_id_rechecked':True})
        self.guard=self.version.start();self.addCleanup(self.version.stop)

    def invoke_once(self):
        with mock.patch('builtins.print') as printed:
            self.cli.find_source_evidence(self.args)
        return json.loads(printed.call_args.args[0])

    def test_native_selected_HTML_candidate_has_separate_linked_evidence_and_pdf_unchanged(self):
        out=self.invoke_once()
        self.assertEqual([self.parent,self.fixture.paths['pdf']],
            [doc['original_path'] for doc in out['documents']])
        self.assertEqual([1,2],[doc['rank'] for doc in out['documents']])
        self.assertEqual(1,out['verified_pdf_excerpts'])
        self.assertEqual(1,out['verified_non_pdf_derived_excerpts'])
        self.assertEqual(1,out['opt_in_archived_original_reference_navigation']['linked_original_sources_checked'])
        self.assertEqual(1,out['opt_in_archived_original_reference_navigation']['original_HTML_article_evidence_passages'])
        self.assertEqual(0,out['opt_in_archived_original_reference_navigation']['already_native_ranked_linked_reference_count'])
        self.assertEqual(1,out['opt_in_archived_original_reference_navigation']['distinct_new_linked_originals_not_in_native_selection'])
        self.assertEqual(1,out['opt_in_archived_original_reference_navigation']['missing_unapproved_original_references'])
        self.assertFalse(out['opt_in_archived_original_reference_navigation']['safe_to_cutover_live'])
        self.assertEqual(self.child,out['opt_in_archived_original_reference_navigation']['parents'][0]['linked_existing_original_sources'][0]['original_path'])
        self.follow.assert_called_once()
        self.assertEqual(self.parent,self.follow.call_args.args[0])
        self.assertEqual(2,self.follow.call_args.kwargs['limit'])
        self.assertEqual(1,self.fixture.mocks['append_query_audit'].call_count)
        audit=self.fixture.mocks['append_query_audit'].call_args.args[0]
        self.assertNotIn('Evidence in archived source.',str(audit))
        self.assertEqual(1,audit['opt_in_archived_original_references']['linked_original_sources_checked'])

    def test_native_mixed_structural_review_separately_tags_firstparty_children_without_new_ranks(self):
        self.args.assess_passage_utility=True
        direct={'reviewed_publisher_excerpt_candidates':2,
                'navigation_or_prompt_template_candidates':0,
                'reviewed_originals':[],'no_candidates_removed_or_reordered':True}
        children={'reviewed_linked_excerpt_candidates':1,
                  'navigation_or_prompt_linked_excerpt_candidates':1,
                  'original_linked_children_not_native_ranked':True}
        with mock.patch.object(self.cli,'review_source_documents',return_value=direct),\
             mock.patch.object(self.cli,'review_verified_linked_originals',return_value=children) as inspector:
            out=self.invoke_once()
        inspector.assert_called_once()
        self.assertEqual(self.args.query,inspector.call_args.args[0])
        observed=inspector.call_args.args[1]
        self.assertEqual(1,len(observed))
        self.assertEqual(self.child,observed[0]['original_path'])
        self.assertFalse(observed[0]['already_in_native_selected_documents'])
        self.assertIsNone(observed[0]['native_selected_original_rank_if_already_selected'])
        self.assertEqual(children,out['optional_source_passage_utility_review']['optional_linked_original_structural_review'])
        self.assertEqual([self.parent,self.fixture.paths['pdf']],[x['original_path'] for x in out['documents']])
        self.assertEqual([1,2],[x['rank'] for x in out['documents']])
        assert out['opt_in_archived_original_reference_navigation']['linked_original_sources_checked']==1
        audit=self.fixture.mocks['append_query_audit'].call_args.args[0]
        self.assertEqual(1,audit['optional_source_passage_utility_counts']['linked_excerpt_candidates'])
        self.assertEqual(1,audit['optional_source_passage_utility_counts']['linked_navigation_or_prompt_candidates'])
        self.assertNotIn('Evidence in archived source.',str(audit))

    def test_without_explicit_flag_normal_mixed_source_search_is_unchanged(self):
        self.args.follow_archived_links=False
        out=self.invoke_once()
        self.assertNotIn('opt_in_archived_original_reference_navigation',out)
        self.assertNotIn('opt_in_archived_original_references',self.fixture.mocks['append_query_audit'].call_args.args[0])
        self.follow.assert_not_called()

    def test_source_name_selection_still_uses_catalog_and_not_speculative_semantic_rank(self):
        manifest,ledger,source_map=self.cli._current_versions()
        self.args.source_name='Exact Synthetic Publisher HTML'
        with mock.patch.object(self.cli,'_current_versions',return_value=(manifest,ledger,source_map)), \
             mock.patch.object(self.cli,'resolve_source_name',return_value={'status':'unique','original_path':self.parent,'subject_query':self.args.query,'kind':'publisher_title'}):
            out=self.invoke_once()
        self.assertEqual('explicit_named_catalog_original',out['selection_basis'])
        self.assertEqual(1,len(out['documents']))
        self.assertEqual(1,out['opt_in_archived_original_reference_navigation']['linked_original_sources_checked'])
        self.follow.assert_called_once()

    def test_non_HTML_and_out_of_range_link_limits_refuse_before_any_source_follow(self):
        for value in (0,6,True,-1):
            self.args.linked_originals=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'archived'):
                self.invoke_once()
        self.follow.assert_not_called()

    def test_rfc_continuation_and_corrected_PDF_overlay_cannot_be_mixed_with_link_lookups(self):
        self.args.continue_section='next-page'
        with self.assertRaisesRegex(ValueError,'archived'):
            self.invoke_once()
        del self.args.continue_section
        self.args.private_staged_pdf_preflight=Path('/tmp/private-staged-pdf.json')
        with self.assertRaisesRegex(ValueError,'archived'):
            self.invoke_once()
        self.follow.assert_not_called()

    def test_stale_or_swapped_linked_original_results_rejected_as_a_group(self):
        for tamper in ('wrong_parent','wrong_parent_sha','wrong_child_sha','wrong_child_id'):
            changed={**self.link,'linked_existing_original_sources':[dict(self.link['linked_existing_original_sources'][0])]}
            if tamper=='wrong_parent':changed['parent_original_path']='docs/unrelated.html'
            elif tamper=='wrong_parent_sha':changed['parent_original_sha256']='f'*64
            elif tamper=='wrong_child_sha':changed['linked_existing_original_sources'][0]['original_sha256']='f'*64
            else:changed['linked_existing_original_sources'][0]['native_filedb_original_id']=0
            self.follow.return_value=changed
            with self.subTest(tamper=tamper),self.assertRaises(ValueError):
                self.invoke_once()
        self.assertEqual(4,self.fixture.mocks['append_query_audit'].call_count)
        self.assertFalse(self.fixture.mocks['append_query_audit'].call_args.args[0]['success'])

    def _catalog_includes_native_ranked_child(self):
        with sqlite3.connect(self.fixture.db) as db:
            db.execute("INSERT INTO filesystem_entries VALUES (51,'file',?)",(self.child,))
        m,l,c=self.cli._current_versions()
        manifest={**m,self.child:{'path':self.child,'kind':'html',
                                    'source_url':'https://publisher.example/approved-child'}}
        ledger={**l,self.child:{'sha256':'a'*64}}
        sources={**c,self.child:(self.child,'a'*64,'b'*64,self.child+'.txt','complete')}
        hits=[self.fixture.hit('html'),
              dict(self.fixture.hit('html'),original_path=self.child,
                   kmd_original_file_entry_id=51,
                   kmd_native_chunk_id='second-native-original-has-independent-rank',
                   semantic_similarity=0.83),
              self.fixture.hit('pdf')]
        return (manifest,ledger,sources),hits

    def test_linked_source_already_in_native_top_results_is_not_counted_as_new(self):
        versions,hits=self._catalog_includes_native_ranked_child()
        self.args.linked_originals=1
        with mock.patch.object(self.cli,'_current_versions',return_value=versions), \
             mock.patch.object(self.cli,'_native_passage_search',return_value={'results':hits}):
            result=self.invoke_once()
        overlay=result['opt_in_archived_original_reference_navigation']
        self.assertEqual(1,overlay['linked_original_sources_checked'])
        self.assertEqual(1,overlay['already_native_ranked_linked_reference_count'])
        self.assertEqual(0,overlay['distinct_new_linked_originals_not_in_native_selection'])
        self.assertEqual(1,overlay['distinct_linked_original_paths'])
        self.assertEqual([1,2,3],[doc['rank'] for doc in result['documents']])
        extra=overlay['parents'][0]['linked_existing_original_sources'][0]
        self.assertTrue(extra['already_in_native_selected_documents'])
        self.assertEqual(2,extra['native_selected_original_rank_if_already_selected'])
        audit=self.fixture.mocks['append_query_audit'].call_args.args[0]
        self.assertEqual(1,audit['opt_in_archived_original_references']['already_native_ranked_linked_references'])
        self.assertEqual(0,audit['opt_in_archived_original_references']['distinct_new_linked_originals'])

    def test_native_ranked_child_can_also_be_a_selected_parent_without_revision_confusion(self):
        versions,hits=self._catalog_includes_native_ranked_child()
        self.args.linked_originals=2
        second={**self.link,'parent_original_path':self.child,
                'linked_existing_original_sources':[],
                'linked_original_sources_checked':0,
                'linked_source_excerpt_candidates':0,
                'publisher_original_article_evidence_passages':0,
                'parent_original_HTML_article_link_proof':{
                  'same_origin_relevant_unarchived_or_unapproved_references':0,
                  'unavailable_publisher_references_not_followed':[]}}
        self.follow.side_effect=[self.link,second]
        with mock.patch.object(self.cli,'_current_versions',return_value=versions), \
             mock.patch.object(self.cli,'_native_passage_search',return_value={'results':hits}):
            result=self.invoke_once()
        overlay=result['opt_in_archived_original_reference_navigation']
        self.assertEqual(2,overlay['native_selected_HTML_parents_checked'])
        self.assertEqual(1,overlay['linked_original_sources_checked'])
        self.assertEqual(1,overlay['already_native_ranked_linked_reference_count'])
        self.assertEqual(0,overlay['distinct_new_linked_originals_not_in_native_selection'])
        self.assertEqual([self.parent,self.child],
                         [x['parent_original_path'] for x in overlay['parents']])
        self.assertEqual(2,self.follow.call_count)
        self.assertTrue(overlay['all_returned_native_selected_parent_and_child_current_source_revisions_rechecked'])

    def test_linked_original_identity_must_match_its_existing_native_rank(self):
        versions,hits=self._catalog_includes_native_ranked_child()
        self.args.linked_originals=1
        bad={**self.link,'linked_existing_original_sources':[{
             **self.link['linked_existing_original_sources'][0],
             'native_filedb_original_id':52}]}
        self.follow.return_value=bad
        with mock.patch.object(self.cli,'_current_versions',return_value=versions), \
             mock.patch.object(self.cli,'_native_passage_search',return_value={'results':hits}):
            with self.assertRaisesRegex(ValueError,'differs from already native-ranked'):
                self.invoke_once()
        self.assertFalse(self.fixture.mocks['append_query_audit'].call_args.args[0]['success'])

    def test_two_native_selected_HTML_originals_share_one_bounded_link_budget(self):
        other='docs/other-official-overview.html'
        path2='docs/other-archived-child.html'
        with sqlite3.connect(self.fixture.db) as db:
            db.execute("INSERT INTO filesystem_entries VALUES (13,'file',?)",(other,))
        versions=list(self.cli._current_versions())
        for mapping in versions[:2]:
            mapping=dict(mapping)
        versions[0]=dict(versions[0],**{other:{'path':other,'kind':'html','source_url':'https://publisher.example/other'}})
        versions[1]=dict(versions[1],**{other:{'sha256':'a'*64}})
        versions[2]=dict(versions[2],**{other:(other,'a'*64,'b'*64,other+'.txt','complete')})
        selected=[self.fixture.hit('html'),
                  dict(self.fixture.hit('html'),original_path=other,kmd_original_file_entry_id=13,
                       kmd_native_chunk_id='other-current-original-chunk',semantic_similarity=0.83),
                  self.fixture.hit('pdf')]
        second={**self.link,'parent_original_path':other,
                'linked_existing_original_sources':[{**self.link['linked_existing_original_sources'][0],
                     'original_path':path2,'native_filedb_original_id':52}],
                'parent_original_HTML_article_link_proof':{
                   'same_origin_relevant_unarchived_or_unapproved_references':0,
                   'unavailable_publisher_references_not_followed':[]}}
        self.args.linked_originals=2
        self.follow.side_effect=[self.link,second]
        with mock.patch.object(self.cli,'_current_versions',return_value=tuple(versions)), \
             mock.patch.object(self.cli,'_native_passage_search',return_value={'results':selected}):
            result=self.invoke_once()
        overlay=result['opt_in_archived_original_reference_navigation']
        self.assertEqual(2,overlay['linked_original_sources_checked'])
        self.assertEqual(2,overlay['native_selected_HTML_parents_checked'])
        self.assertEqual([1,2],[p['native_parent_rank'] for p in overlay['parents']])
        self.assertEqual([1,2,3],[d['rank'] for d in result['documents']])
        self.assertEqual(2,self.follow.call_count)
        self.assertEqual([2,1],[c.kwargs['limit'] for c in self.follow.call_args_list])
        self.assertTrue(overlay['all_returned_native_selected_parent_and_child_current_source_revisions_rechecked'])
        self.assertFalse(overlay['cross_store_immutable_generation_pinned'])

    def test_budget_cannot_be_exceeded_by_an_untrusted_link_reader_response(self):
        too_many={**self.link,'linked_original_sources_checked':3,
                  'linked_existing_original_sources':[self.link['linked_existing_original_sources'][0]]}
        self.follow.return_value=too_many
        with self.assertRaisesRegex(ValueError,'unbounded or inconsistent'):
            self.invoke_once()
        self.assertFalse(self.fixture.mocks['append_query_audit'].call_args.args[0]['success'])

    def test_late_native_revision_change_after_link_reader_blocks_combined_result(self):
        def stale(source,*,original_path,native_file_id):
            if original_path==self.child:
                raise ValueError('linked publisher derivative changed after child evidence')
            return {'latest_source_map_revision_and_native_file_id_rechecked':True}
        self.guard.side_effect=stale
        with self.assertRaisesRegex(ValueError,'linked publisher derivative changed'):
            self.invoke_once()
        self.assertFalse(self.fixture.mocks['append_query_audit'].call_args.args[0]['success'])

    def _native_approved_markdown_source_fixture(self):
        """Select a real-kind Markdown original and a PDF without changing global ranks."""
        parent='docs/official-data-overview.md'
        child='docs/official-transform-methods.md'
        with sqlite3.connect(self.fixture.db) as db:
            db.execute("INSERT INTO filesystem_entries VALUES (32,'file',?)",(parent,))
        manifest,ledger,source_map=self.cli._current_versions()
        versions=(
            {**manifest,parent:{'path':parent,'kind':'text',
                                'url':'https://docs.example.org/official-data-overview.md'}},
            {**ledger,parent:{'sha256':'a'*64}},
            {**source_map,parent:(parent,'a'*64,'b'*64,parent+'.txt','complete')},
        )
        source_hit={**self.fixture.hit('text'),
                    'original_path':parent,
                    'kmd_original_file_entry_id':32,
                    'kmd_native_chunk_id':'current-original-n8n-markdown-chunk',
                    'semantic_similarity':0.96}
        hits=[source_hit,self.fixture.hit('pdf')]
        link={
           'parent_original_path':parent,'parent_original_sha256':'a'*64,
           'parent_derived_sha256':'b'*64,
           'linked_original_sources_checked':1,
           'linked_derived_passage_candidates':2,
           'independently_publisher_original_plaintext_byte_attested_passages':2,
           'parent_original_Markdown_literal_byte_link_proof':{
              'missing_or_unapproved_official_reference_count':1,
              'original_Markdown_publisher_literal_reference_count':5,
              'selected_links':[{'publisher_markdown_link_text':'Transform data',
                'publisher_link_literal_start_byte':24,
                'publisher_link_literal_end_byte':52,
                'publisher_link_literal_sha256':'a'*64,
                'candidate_not_answer':True}]},
           'linked_approved_current_original_sources':[{
              'original_path':child,'original_sha256':'a'*64,
              'derived_sha256':'b'*64,'native_filedb_original_id':56,
              'publisher_source_link_proof':{'publisher_markdown_link_text':'Transform data',
                      'candidate_not_answer':True},
              'passage_candidates':[{'current_derived_text_excerpt':'Publisher source passage.',
                 'publisher_plaintext_byte_interval_independently_SHA_verified':True,
                 'factual_answer_sufficiency_verified':False}]}],
           'cross_store_immutable_generation_pinned':False,
           'safe_to_cutover_live':False,
        }
        return parent,child,versions,hits,link

    def test_native_discovery_provides_verified_table_columns_only_from_selected_markdown_original(self):
        parent,child,versions,hits,_link=self._native_approved_markdown_source_fixture()
        self.args.follow_archived_links=False
        self.args.inspect_original_tables=True
        table={'schema':'test-current-source-table','original_path':parent,
               'returned_original_publisher_table_row_candidates':1,
               'publisher_rows':[{'publisher_table_row_cells':['AI Transform node','Cloud only'],
                                  'publisher_table_columns':['Approach','Available on']}],
               'current_publisher_source_byte_proof_not_semantic_answer_correctness':True}
        original_checked=self.fixture.checked
        def checked(path,**kwargs):
            src=original_checked(path,**kwargs)
            return {**src,'publisher_text_bytes':b'| Approach | Available on |\n| --- | --- |\n| AI Transform node | Cloud only |\n'} if path==parent else src
        with mock.patch.object(self.cli,'_current_versions',return_value=versions),\
             mock.patch.object(self.cli,'_native_passage_search',return_value={'results':hits}),\
             mock.patch.object(self.cli,'checked_original',side_effect=checked),\
             mock.patch.object(self.cli,'verified_original_markdown_table_rows',return_value=table) as table_reader:
            result=self.invoke_once()
        self.assertEqual([parent,self.fixture.paths['pdf']],[x['original_path'] for x in result['documents']])
        self.assertEqual([1,2],[x['rank'] for x in result['documents']])
        tables=result['optional_original_markdown_table_evidence']
        self.assertEqual(1,tables['original_markdown_documents_inspected'])
        self.assertEqual(1,tables['original_publisher_table_row_candidates'])
        self.assertTrue(tables['native_ranked_documents_unchanged'])
        self.assertEqual(parent,tables['originals'][0]['original_path'])
        self.assertEqual(table,tables['originals'][0]['original_firstparty_markdown_table_evidence'])
        table_reader.assert_called_once()
        self.assertEqual(parent,table_reader.call_args.kwargs['original_path'])
        self.assertEqual(2,table_reader.call_args.kwargs['top'])
        audit=self.fixture.mocks['append_query_audit'].call_args.args[0]
        self.assertEqual(1,audit['opt_in_original_markdown_table_evidence_counts']['original_table_row_candidates'])
        self.assertNotIn('Cloud only',str(audit))

    def test_optional_markdown_tables_do_not_reinterpret_HTML_or_PDF_as_table_proofs(self):
        self.args.follow_archived_links=False
        self.args.inspect_original_tables=True
        with mock.patch.object(self.cli,'verified_original_markdown_table_rows',
                               side_effect=AssertionError('no Markdown source selected')) as reader:
            result=self.invoke_once()
        reader.assert_not_called()
        self.assertEqual(0,result['optional_original_markdown_table_evidence']['original_markdown_documents_inspected'])
        self.assertEqual(0,result['optional_original_markdown_table_evidence']['original_publisher_table_row_candidates'])
        self.assertEqual([1,2],[x['rank'] for x in result['documents']])

    def test_optional_original_table_reader_SHA_mutation_fails_whole_native_result(self):
        parent,child,versions,hits,_link=self._native_approved_markdown_source_fixture()
        self.args.follow_archived_links=False
        self.args.inspect_original_tables=True
        original_checked=self.fixture.checked
        def checked(path,**kwargs):
            src=original_checked(path,**kwargs)
            return {**src,'publisher_text_bytes':b'| Table | Source |'} if path==parent else src
        with mock.patch.object(self.cli,'_current_versions',return_value=versions),\
             mock.patch.object(self.cli,'_native_passage_search',return_value={'results':hits}),\
             mock.patch.object(self.cli,'checked_original',side_effect=checked),\
             mock.patch.object(self.cli,'verified_original_markdown_table_rows',
                side_effect=ValueError('original published table row SHA changed')):
            with self.assertRaisesRegex(ValueError,'original published table row SHA changed'):
                self.invoke_once()
        self.assertFalse(self.fixture.mocks['append_query_audit'].call_args.args[0]['success'])

    def test_native_selected_markdown_adds_separate_publisher_original_byte_evidence(self):
        parent,child,versions,hits,link=self._native_approved_markdown_source_fixture()
        self.args.linked_originals=2
        with mock.patch.object(self.cli,'_current_versions',return_value=versions), \
             mock.patch.object(self.cli,'_native_passage_search',return_value={'results':hits}), \
             mock.patch.object(markdown,'follow_current_archived_markdown_references',return_value=link) as reader:
            result=self.invoke_once()
        self.assertEqual([parent,self.fixture.paths['pdf']],
                         [row['original_path'] for row in result['documents']])
        self.assertEqual([1,2],[row['rank'] for row in result['documents']])
        overlay=result['opt_in_archived_original_reference_navigation']
        self.assertEqual(0,overlay['native_selected_HTML_parents_checked'])
        self.assertEqual(1,overlay['native_selected_Markdown_parents_checked'])
        self.assertEqual(0,overlay['original_HTML_article_evidence_passages'])
        self.assertEqual(2,overlay['original_Markdown_publisher_byte_evidence_passages'])
        self.assertEqual(1,overlay['linked_original_sources_checked'])
        self.assertEqual(1,overlay['distinct_new_linked_originals_not_in_native_selection'])
        self.assertEqual(1,overlay['missing_unapproved_original_references'])
        self.assertTrue(overlay['all_returned_native_selected_parent_and_child_current_source_revisions_rechecked'])
        candidate=overlay['parents'][0]['linked_existing_original_sources'][0]
        self.assertEqual(child,candidate['original_path'])
        self.assertTrue(candidate['passage_candidates'][0]['publisher_plaintext_byte_interval_independently_SHA_verified'])
        self.assertFalse(candidate['passage_candidates'][0]['factual_answer_sufficiency_verified'])
        self.assertEqual('original_Markdown',overlay['parents'][0]['parent_source_kind'])
        reader.assert_called_once()
        self.assertEqual(parent,reader.call_args.args[0])
        self.assertEqual(2,reader.call_args.kwargs['limit'])
        self.follow.assert_not_called()
        audit=self.fixture.mocks['append_query_audit'].call_args.args[0]
        self.assertEqual(2,audit['opt_in_archived_original_references']['original_Markdown_publisher_byte_evidence_passages'])
        self.assertNotIn('Publisher source passage.',str(audit))

    def test_native_markdown_optin_with_absent_official_link_does_not_substitute_source(self):
        parent,child,versions,hits,link=self._native_approved_markdown_source_fixture()
        empty={**link,'linked_original_sources_checked':0,'linked_derived_passage_candidates':0,
               'independently_publisher_original_plaintext_byte_attested_passages':0,
               'linked_approved_current_original_sources':[]}
        with mock.patch.object(self.cli,'_current_versions',return_value=versions), \
             mock.patch.object(self.cli,'_native_passage_search',return_value={'results':hits}), \
             mock.patch.object(markdown,'follow_current_archived_markdown_references',return_value=empty) as reader:
            result=self.invoke_once()
        overlay=result['opt_in_archived_original_reference_navigation']
        self.assertEqual(0,overlay['linked_original_sources_checked'])
        self.assertEqual(0,overlay['original_Markdown_publisher_byte_evidence_passages'])
        self.assertEqual([],overlay['parents'][0]['linked_existing_original_sources'])
        reader.assert_called_once()

    def test_mixed_markdown_changed_parent_OR_target_SHA_fail_closed(self):
        parent,child,versions,hits,link=self._native_approved_markdown_source_fixture()
        for target_key in ('parent_original_sha256','child_source_sha','child_ID'):
            changed={**link,'linked_approved_current_original_sources':[
                 dict(link['linked_approved_current_original_sources'][0])]}
            if target_key=='parent_original_sha256':changed[target_key]='f'*64
            elif target_key=='child_source_sha':changed['linked_approved_current_original_sources'][0]['original_sha256']='f'*64
            else:changed['linked_approved_current_original_sources'][0]['native_filedb_original_id']=0
            with self.subTest(target_key=target_key), \
                 mock.patch.object(self.cli,'_current_versions',return_value=versions), \
                 mock.patch.object(self.cli,'_native_passage_search',return_value={'results':hits}), \
                 mock.patch.object(markdown,'follow_current_archived_markdown_references',return_value=changed):
                with self.assertRaises(ValueError):self.invoke_once()
            self.assertFalse(self.fixture.mocks['append_query_audit'].call_args.args[0]['success'])

    def test_markdown_linked_source_already_native_ranked_is_not_counted_new(self):
        parent,child,versions,hits,link=self._native_approved_markdown_source_fixture()
        with sqlite3.connect(self.fixture.db) as db:
            db.execute("INSERT INTO filesystem_entries VALUES (56,'file',?)",(child,))
        hits=[hits[0],{**hits[0], 'original_path':child,
                         'kmd_original_file_entry_id':56,'semantic_similarity':0.84,
                         'kmd_native_chunk_id':'current-other-original-markdown'},hits[1]]
        versions=({**versions[0],child:{'path':child,'kind':'text',
                   'url':'https://docs.example.org/official-transform-methods.md'}},
                  {**versions[1],child:{'sha256':'a'*64}},
                  {**versions[2],child:(child,'a'*64,'b'*64,child+'.txt','complete')})
        self.args.linked_originals=1
        with mock.patch.object(self.cli,'_current_versions',return_value=versions), \
             mock.patch.object(self.cli,'_native_passage_search',return_value={'results':hits}), \
             mock.patch.object(markdown,'follow_current_archived_markdown_references',return_value=link) as reader:
            result=self.invoke_once()
        out=result['opt_in_archived_original_reference_navigation']
        self.assertEqual(1,out['linked_original_sources_checked'])
        self.assertEqual(1,out['already_native_ranked_linked_reference_count'])
        self.assertEqual(0,out['distinct_new_linked_originals_not_in_native_selection'])
        self.assertTrue(out['parents'][0]['linked_existing_original_sources'][0]['already_in_native_selected_documents'])
        self.assertEqual(2,out['parents'][0]['linked_existing_original_sources'][0]['native_selected_original_rank_if_already_selected'])
        reader.assert_called_once()

    def test_no_matching_official_link_returns_empty_source_list_not_a_guess(self):
        self.follow.return_value={**self.link,
            'linked_original_sources_checked':0,'linked_source_excerpt_candidates':0,
            'publisher_original_article_evidence_passages':0,
            'linked_existing_original_sources':[],
            'parent_original_HTML_article_link_proof':{
                'same_origin_relevant_unarchived_or_unapproved_references':0,
                'unavailable_publisher_references_not_followed':[]}}
        out=self.invoke_once()
        links=out['opt_in_archived_original_reference_navigation']
        self.assertEqual(0,links['linked_original_sources_checked'])
        self.assertEqual(0,links['original_HTML_article_evidence_passages'])
        self.assertEqual([],links['parents'][0]['linked_existing_original_sources'])
        self.assertEqual(1,out['verified_pdf_excerpts'])

if __name__=='__main__':unittest.main()
