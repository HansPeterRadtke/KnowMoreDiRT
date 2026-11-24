"""Main existing source-evidence CLI now exposes optional archived reference navigation.

The opt-in channel is a separate existing-SHA source link proof, not a
replacement for direct source excerpts, native semantic rank or answer truth.
"""
from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase,main,mock
import json
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))

from test_source_evidence import cli
import raspi_archived_html_follow_links as linked
import raspi_archived_markdown_follow_links as markdown_linked


class OptionalMainOriginalArchivedLinkTests(TestCase):
    def setUp(self):
        self.kmd=cli()
        self.rel='docs/official-overview.html'
        self.query='How do I avoid game stalls when loading texture and mesh data from disk?'
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.db=Path(self.tmp.name)/'native.sqlite3'
        with sqlite3.connect(self.db) as db:
            db.execute('CREATE TABLE filesystem_entries (id INTEGER,entry_type TEXT,relative_path_display TEXT)')
            db.execute("INSERT INTO filesystem_entries VALUES (3,'file',?)",(self.rel,))
        self.source={
            'original_path':self.rel,'original_sha256':'a'*64,'derived_sha256':'b'*64,
            'original_bytes':400,'title':'First-party Overview','derived_text':'overview source content',
        }
        self.link_result={
            'linked_original_sources_checked':1,
            'publisher_original_article_evidence_passages':1,
            'parent_original_HTML_article_link_proof':{
                'same_origin_relevant_unarchived_or_unapproved_references':1,
                'unavailable_publisher_references_not_followed':[
                    {'publisher_article_link_text':'Unavailable linked official guide',
                     'publisher_destination_contents_unavailable':True,
                     'not_a_source_excerpt_or_answer':True}],
            },
            'linked_existing_original_sources':[
                {'original_path':'docs/verified-linked-guide.html',
                 'passage_candidates':[{'factual_answer_sufficiency_verified':False,
                                       'extracted_text_excerpt_candidate':'A short source-linked candidate from a previously archived document.',
                                       'extracted_text_excerpt_is_not_a_literal_publisher_byte_quotation':True}],
                 'source_link_candidate_not_factual_answer':True}],
            'not_proof_of_correct_answer_or_atomic_cross_store_generation':True,
            'safe_to_cutover_live':False,
        }
        patches=[
            mock.patch.object(self.kmd,'DB',self.db),
            mock.patch.object(self.kmd,'checked_original',return_value=self.source),
            mock.patch.object(self.kmd,'_original_passages',return_value={
                'original_path':self.rel,'original_sha256':'a'*64,
                'derived_sha256':'b'*64,'passages':[], 'returned_passages':0,
                'evidence_warning':'No verified candidate passage in the selected original.'}),
            mock.patch.object(self.kmd,'final_original_revision_guard',return_value={
                'latest_source_map_revision_and_native_file_id_rechecked':True,
                'cross_store_immutable_generation_pinned':False,
                'safe_to_cutover_live':False}),
            mock.patch.object(self.kmd,'append_query_audit',return_value=Path(self.tmp.name)/'query-audit.jsonl'),
            mock.patch.object(linked,'find_archived_linked_source_passages',return_value=self.link_result),
        ]
        self.patched=[p.start() for p in patches]
        for p in patches:self.addCleanup(p.stop)
        self.audit=self.patched[-2]
        self.links=self.patched[-1]
        self.args=SimpleNamespace(query=self.query,original_path=self.rel,
                 top=2,snippet_characters=850,follow_archived_links=True,
                 linked_originals=3)

    def run_query(self):
        with mock.patch('builtins.print') as out:
            self.kmd.original_source_evidence(self.args)
        return json.loads(out.call_args.args[0])

    def test_empty_parent_source_receives_separate_verified_linked_evidence_only_optin(self):
        result=self.run_query()
        self.assertEqual(0,result['returned_passages'])
        self.assertEqual([],result['passages'])
        self.assertEqual(self.link_result,result['linked_archived_original_references'])
        candidate=result['linked_archived_original_references']['linked_existing_original_sources'][0]['passage_candidates'][0]
        self.assertIn('source-linked candidate',candidate['extracted_text_excerpt_candidate'])
        self.assertTrue(candidate['extracted_text_excerpt_is_not_a_literal_publisher_byte_quotation'])
        self.assertFalse(candidate['factual_answer_sufficiency_verified'])
        self.assertTrue(result['final_current_source_revision_check']['latest_source_map_revision_and_native_file_id_rechecked'])
        self.assertEqual(1,self.links.call_count)
        self.links.assert_called_once_with(self.rel,self.query,limit=3,passages=2,snippet_characters=850)
        receipt=self.audit.call_args.args[0]
        self.assertTrue(receipt['success'])
        self.assertEqual(0,receipt['count'])
        self.assertEqual(1,receipt['opt_in_archived_reference_follow']['linked_originals_checked'])
        self.assertEqual(1,receipt['opt_in_archived_reference_follow']['missing_unapproved_original_reference_count'])
        self.assertTrue(receipt['opt_in_archived_reference_follow']['not_a_verified_answer'])
        self.assertNotIn('linked_archived_original_references',receipt)

    def test_direct_source_evidence_structural_review_optin_only_after_final_revision_guard(self):
        self.args.follow_archived_links=False
        self.args.assess_passage_utility=True
        mocked={'reviewed_publisher_excerpt_candidates':0,
                'navigation_or_prompt_template_candidates':0,
                'reviewed_originals':[],
                'no_candidates_removed_or_reordered':True,
                'not_factual_answer_sufficiency_or_entailment_verification':True}
        with mock.patch.object(self.kmd,'review_source_documents',return_value=mocked) as review:
            result=self.run_query()
        self.assertEqual(mocked,result['optional_source_passage_utility_review'])
        review.assert_called_once_with(self.query,[{'original_path':self.rel,
                                                     'rank':1,'passages':[]}])
        audit=self.audit.call_args.args[0]
        self.assertEqual(0,audit['optional_source_passage_utility_counts']['reviewed_candidates'])
        self.assertTrue(audit['optional_source_passage_utility_counts']['not_answer_verification'])
        self.assertNotIn('optional_source_passage_utility_review',audit)
        self.links.assert_not_called()

    def test_review_source_query_rejects_oversized_or_continuing_sources_before_read(self):
        self.args.follow_archived_links=False
        self.args.assess_passage_utility=True
        for top,cursor in ((6,None),(50,None),(2,'continuation-scope')):
            with self.subTest(top=top,cursor=cursor):
                self.args.top=top
                self.args.continue_section=cursor
                with self.assertRaisesRegex(ValueError,'at most five original passages and no RFC continuation'):
                    self.kmd.original_source_evidence(self.args)
        self.patched[1].assert_not_called()

    def test_direct_source_structural_review_can_annotate_verified_linked_children_separately(self):
        self.args.assess_passage_utility=True
        direct={'reviewed_publisher_excerpt_candidates':0,
                'navigation_or_prompt_template_candidates':0,
                'reviewed_originals':[],'no_candidates_removed_or_reordered':True,
                'not_factual_answer_sufficiency_or_entailment_verification':True}
        linked={'reviewed_linked_excerpt_candidates':1,
                'navigation_or_prompt_linked_excerpt_candidates':1,
                'original_linked_children_not_native_ranked':True}
        with mock.patch.object(self.kmd,'review_source_documents',return_value=direct),\
             mock.patch.object(self.kmd,'review_verified_linked_originals',return_value=linked) as assessor:
            result=self.run_query()
        assessor.assert_called_once_with(self.query,self.link_result['linked_existing_original_sources'])
        self.assertEqual(linked,result['optional_source_passage_utility_review']['optional_linked_original_structural_review'])
        self.assertEqual(1,self.audit.call_args.args[0]['optional_source_passage_utility_counts']['linked_excerpt_candidates'])
        self.assertEqual(1,self.audit.call_args.args[0]['optional_source_passage_utility_counts']['linked_navigation_or_prompt_candidates'])
        self.assertNotIn('Evidence in archived source.',str(self.audit.call_args.args[0]))
        self.assertEqual(self.link_result,result['linked_archived_original_references'])

    def test_without_structural_flag_default_direct_source_audit_unmodified(self):
        self.args.follow_archived_links=False
        with mock.patch.object(self.kmd,'review_source_documents',side_effect=AssertionError('optin only')) as review:
            result=self.run_query()
        review.assert_not_called()
        self.assertNotIn('optional_source_passage_utility_review',result)
        self.assertNotIn('optional_source_passage_utility_counts',self.audit.call_args.args[0])

    def test_existing_default_source_command_never_imports_or_follows_links(self):
        self.args.follow_archived_links=False
        result=self.run_query()
        self.assertNotIn('linked_archived_original_references',result)
        self.assertNotIn('opt_in_archived_reference_follow',self.audit.call_args.args[0])
        self.links.assert_not_called()

    def test_wrong_source_types_and_unbounded_optin_fail_closed_without_child_query(self):
        for rel,cursor,size,query in [
            ('docs/official-original.pdf',None,2,self.query),
            ('docs/official-original.txt',None,2,self.query),
            (self.rel,'rfc_section_cursor',2,self.query),
            (self.rel,None,0,self.query),
            (self.rel,None,6,self.query),
            (self.rel,None,True,self.query),
            (self.rel,None,2,'Q'*301),
        ]:
            with self.subTest(rel=rel,cursor=cursor,size=size):
                with self.assertRaisesRegex(ValueError,'archived source following requires'):
                    self.kmd._optional_verified_archived_links(
                        SimpleNamespace(follow_archived_links=True,linked_originals=size,
                                        top=2,snippet_characters=850),
                        original_path=rel,evidence_query=query,continuation_cursor=cursor)
        self.links.assert_not_called()

    def test_error_in_linked_original_source_revision_refuses_entire_main_result(self):
        self.links.side_effect=ValueError('linked child current original SHA changed')
        with mock.patch('builtins.print') as out:
            with self.assertRaisesRegex(ValueError,'linked child current original SHA changed'):
                self.kmd.original_source_evidence(self.args)
        out.assert_not_called()
        receipt=self.audit.call_args.args[0]
        self.assertFalse(receipt['success'])
        self.assertIn('linked child current original SHA changed',receipt['error'])

    def _switch_to_markdown_original(self):
        prev=self.rel
        self.rel='docs/official-overview.md'
        self.args.original_path=self.rel
        self.source['original_path']=self.rel
        with sqlite3.connect(self.db) as db:
            db.execute('UPDATE filesystem_entries SET relative_path_display=? WHERE relative_path_display=?',
                       (self.rel,prev))
        self.patched[2].return_value={
            'original_path':self.rel,'original_sha256':'a'*64,
            'derived_sha256':'b'*64,'passages':[],'returned_passages':0,
            'evidence_warning':'No direct matching publisher passage in the selected overview.'}

    def test_main_source_evidence_direct_markdown_firstparty_link_navigation_has_distinct_byte_proof(self):
        self._switch_to_markdown_original()
        self.args.exact_link_text='Transform data'
        selected={'linked_original_sources_checked':1,
           'independently_publisher_original_plaintext_byte_attested_passages':2,
           'parent_original_Markdown_literal_byte_link_proof':{
                'missing_or_unapproved_official_reference_count':1},
           'linked_approved_current_original_sources':[{'original_path':'docs/transform-data.md',
               'passage_candidates':[{'publisher_plaintext_byte_interval_independently_SHA_verified':True,
                                      'factual_answer_sufficiency_verified':False}]}],
           'safe_to_cutover_live':False}
        with mock.patch.object(markdown_linked,'follow_current_archived_markdown_references',return_value=selected) as md:
            result=self.run_query()
        self.assertEqual(0,result['returned_passages'])
        self.assertEqual(selected,result['linked_archived_original_references'])
        self.assertEqual(1,md.call_count)
        self.assertEqual(self.rel,md.call_args.args[0])
        self.assertEqual('Transform data',md.call_args.kwargs['exact_link_text'])
        audit=self.audit.call_args.args[0]['opt_in_archived_reference_follow']
        self.assertEqual(0,audit['verified_publisher_HTML_article_candidate_passages'])
        self.assertEqual(2,audit['independently_publisher_original_plaintext_byte_attested_passages'])
        self.assertEqual(['docs/transform-data.md'],audit['candidate_original_paths'])
        self.assertEqual(1,audit['missing_unapproved_original_reference_count'])
        self.links.assert_not_called()

    def test_explicit_markdown_navigation_without_optin_or_on_HTML_refuses(self):
        self.args.follow_archived_links=False
        self.args.exact_link_text='Transform data'
        with self.assertRaisesRegex(ValueError,'requires explicit --follow-archived-links'):
            self.run_query()
        self.args.follow_archived_links=True
        with self.assertRaisesRegex(ValueError,'only supported for original Markdown'):
            self.run_query()
        self.links.assert_not_called()

    def test_markdown_source_nonexistent_exact_link_does_not_fabricate_relevance(self):
        self._switch_to_markdown_original()
        self.args.exact_link_text='Nonexistent official publication'
        result_data={'linked_original_sources_checked':0,
          'independently_publisher_original_plaintext_byte_attested_passages':0,
          'parent_original_Markdown_literal_byte_link_proof':{
             'missing_or_unapproved_official_reference_count':0},
          'linked_approved_current_original_sources':[],
          'safe_to_cutover_live':False}
        with mock.patch.object(markdown_linked,'follow_current_archived_markdown_references',return_value=result_data):
            result=self.run_query()
        self.assertEqual(0,result['linked_archived_original_references']['linked_original_sources_checked'])
        self.assertEqual([],result['linked_archived_original_references']['linked_approved_current_original_sources'])
        self.assertEqual(0,self.audit.call_args.args[0]['opt_in_archived_reference_follow']['linked_originals_checked'])

    def test_invalid_html_topic_without_query_terms_yields_source_error_not_fabricated_result(self):
        self.links.return_value={**self.link_result,
              'linked_original_sources_checked':0,
              'publisher_original_article_evidence_passages':0,
              'linked_existing_original_sources':[]}
        self.args.query='Unrelated neutrino and stellar dynamics phenomenon'
        result=self.run_query()
        self.assertEqual(0,result['returned_passages'])
        self.assertEqual([],result['linked_archived_original_references']['linked_existing_original_sources'])
        self.assertEqual(0,self.audit.call_args.args[0]['opt_in_archived_reference_follow']['linked_originals_checked'])


if __name__=='__main__':main()
