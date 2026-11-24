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
                 'passage_candidates':[{'factual_answer_sufficiency_verified':False}],
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
