"""Offline same-origin archived-reference passage navigation integration tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from file_system_catalog.archived_html_source_links import source_link_candidates


def sha(x):return hashlib.sha256(x).hexdigest()


def cli():
    path=Path(__file__).resolve().parents[2]/'scripts'/'raspi_archived_html_follow_links.py'
    spec=importlib.util.spec_from_file_location('kmd_offline_archived_link_cli_test',path)
    instance=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(instance)
    return instance


class ArchivedOriginalLinkRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.cli=cli()
        self.parent_path='docs/source-overview.html'
        self.child_path='docs/loading-mesh-data.html'
        self.parent_url='https://manual.example.org/engine/source-overview'
        self.child_url='https://manual.example.org/engine/loading-mesh-data'
        self.parent_raw=(b'<html><body><nav><a href="https://evil.example.org/claim">Mesh fake</a></nav>'
                         b'<article><h1>Overview</h1>'
                         b'<a href="loading-mesh-data">Texture and mesh loading</a>'
                         b'<a href="https://evil.example.org/claim">External mesh claim</a>'
                         b'</article></body></html>')
        self.statement=('Texture and mesh data are loaded asynchronously from disk to avoid pausing '
                        'the game while rendering textures or geometry, with source data at build time.')
        self.child_raw=('<html><body><article><h1>Texture and mesh loading</h1><p>'+
                         self.statement+'</p></article></body></html>').encode()
        self.derived=('KMD source provenance generated metadata must never count as source evidence.\n'
                      '[[SOURCE UNIT: HTML original publisher document]]\n'+
                      'Texture and mesh loading\n'+self.statement+'\n')
        self.catalog=[{'path':self.parent_path,'kind':'html','title':'Overview',
                       'url':self.parent_url},
                      {'path':self.child_path,'kind':'html','title':'Texture and mesh loading',
                       'url':self.child_url}]
        self.ledger={x['path']:{'path':x['path'],'title':x['title'],'url':x['url'],
                                'status':'complete','sha256':sha(self.parent_raw if x['path']==self.parent_path else self.child_raw),
                                'bytes_received':len(self.parent_raw if x['path']==self.parent_path else self.child_raw)} for x in self.catalog}
        self.parent={'original_path':self.parent_path,'title':'Overview','original_sha256':sha(self.parent_raw),
                     'derived_sha256':sha(b'overview derived'),'publisher_html_bytes':self.parent_raw}
        self.child={'original_path':self.child_path,'title':'Texture and mesh loading',
                    'original_sha256':sha(self.child_raw),'derived_sha256':sha(self.derived.encode()),
                    'publisher_html_bytes':self.child_raw,'derived_text':self.derived}
        self.query='How do I avoid stalls when loading texture and mesh data from disk?'
        self.checked_calls=[];self.fenced=[]
        def checked(path,**kw):
            self.checked_calls.append((path,kw))
            return self.parent if path==self.parent_path else self.child
        def guard(source,*,original_path,native_file_id):
            self.fenced.append((original_path,native_file_id))
            return {'latest_source_map_revision_and_native_file_id_rechecked':True}
        for name,func in [('_records',lambda:(self.catalog,{x['path']:x for x in self.catalog},self.ledger,set(),set())),
                          ('_native_file_id',lambda p:1 if p==self.parent_path else 2)]:
            patcher=patch.object(self.cli,name,side_effect=func)
            patcher.start();self.addCleanup(patcher.stop)
        for name,func in [('checked_original',checked),('final_original_revision_guard',guard)]:
            patcher=patch.object(self.cli.library,name,side_effect=func)
            patcher.start();self.addCleanup(patcher.stop)
        patcher=patch.object(self.cli,'publication_receipt_matches',return_value=True)
        patcher.start();self.addCleanup(patcher.stop)

    def test_end_to_end_publisher_article_link_target_provenance_and_exact_excerpt(self):
        r=self.cli.find_archived_linked_source_passages(self.parent_path,self.query)
        self.assertEqual(1,r['linked_original_sources_checked'])
        self.assertEqual(1,r['linked_source_excerpt_candidates'])
        self.assertEqual(1,r['publisher_original_article_evidence_passages'])
        child=r['linked_existing_original_sources'][0]
        self.assertEqual(self.child_path,child['original_path'])
        self.assertEqual(sha(self.child_raw),child['original_sha256'])
        self.assertEqual(2,child['native_filedb_original_id'])
        self.assertEqual('Texture and mesh loading',child['link_provenance']['publisher_article_link_text'])
        passage=child['passage_candidates'][0]
        source=passage['extracted_text_excerpt_candidate']
        self.assertTrue(source)
        self.assertEqual(source,self.derived[passage['extracted_text_excerpt_start_char']:passage['extracted_text_excerpt_end_char']])
        self.assertEqual(sha(source.encode()),passage['extracted_text_excerpt_sha256'])
        self.assertTrue(passage['extracted_text_excerpt_is_not_a_literal_publisher_byte_quotation'])
        self.assertTrue(passage['independent_original_article_text_match_verified'])
        self.assertEqual('article',passage['publisher_original_normalized_HTML_article_anchor']['container'])
        self.assertFalse(passage['literal_publisher_bytes_verified'])
        self.assertFalse(passage['factual_answer_sufficiency_verified'])
        self.assertTrue(r['no_new_source_downloads_or_external_navigation'])
        self.assertFalse(r['safe_to_cutover_live'])
        self.assertEqual([(self.child_path,2),(self.parent_path,1),
                          (self.child_path,2),(self.parent_path,1)],self.fenced)
        self.assertTrue(r['all_selected_destination_original_source_map_and_native_ID_rechecked_after_other_targets'])
        self.assertFalse(r['cross_store_immutable_generation_pinned'])
        self.assertTrue(all(kwargs['publication_record'] for _,kwargs in self.checked_calls))

    def test_target_mutation_after_passages_refuses_entire_combined_result(self):
        def guard(source,*,original_path,native_file_id):
            if original_path==self.child_path:
                raise ValueError('child source revision changed during proof')
            return {'latest_source_map_revision_and_native_file_id_rechecked':True}
        with patch.object(self.cli.library,'final_original_revision_guard',side_effect=guard):
            with self.assertRaisesRegex(ValueError,'child source revision changed'):
                self.cli.find_archived_linked_source_passages(self.parent_path,self.query)

    def test_mismatched_excerpt_source_text_and_sha_refuses_all_results(self):
        from file_system_catalog.source_evidence import source_evidence_candidates
        for tamper in ('wrong_excerpt_text','wrong_sha','wrong_offset'):
            def corrupt(**kwargs):
                matched=source_evidence_candidates(**kwargs)
                candidate=matched['passages'][0]
                if tamper=='wrong_excerpt_text':candidate['source_excerpt']='not the original source'
                elif tamper=='wrong_sha':candidate['excerpt_sha256']='f'*64
                else:candidate['excerpt_start_char']=candidate['excerpt_start_char']+1
                return matched
            with self.subTest(tamper=tamper),patch.object(self.cli,'source_evidence_candidates',side_effect=corrupt):
                with self.assertRaisesRegex(ValueError,'candidate excerpt does not match'):
                    self.cli.find_archived_linked_source_passages(self.parent_path,self.query)

    def test_destination_changed_only_after_initial_target_guard_still_fails_final_group(self):
        seen=[]
        def check(source,*,original_path,native_file_id):
            seen.append(original_path)
            if original_path==self.child_path and seen.count(self.child_path)>=2:
                raise ValueError('destination source revised while later source evidence assembled')
            return {'latest_source_map_revision_and_native_file_id_rechecked':True}
        with patch.object(self.cli.library,'final_original_revision_guard',side_effect=check):
            with self.assertRaisesRegex(ValueError,'destination source revised'):
                self.cli.find_archived_linked_source_passages(self.parent_path,self.query)
        self.assertEqual([self.child_path,self.parent_path,self.child_path],seen)

    def test_parent_replaced_after_following_child_refuses_combined_result(self):
        def guard(source,*,original_path,native_file_id):
            if original_path==self.parent_path:
                raise ValueError('parent publisher link source revision changed')
            return {'latest_source_map_revision_and_native_file_id_rechecked':True}
        with patch.object(self.cli.library,'final_original_revision_guard',side_effect=guard):
            with self.assertRaisesRegex(ValueError,'parent publisher link source revision changed'):
                self.cli.find_archived_linked_source_passages(self.parent_path,self.query)

    def test_unapproved_or_missing_archived_link_is_never_replaced_by_unrelated_source(self):
        no_receipt={self.parent_path:self.ledger[self.parent_path]}
        with patch.object(self.cli,'_records',return_value=(self.catalog,{x['path']:x for x in self.catalog},no_receipt,set(),set())):
            report=self.cli.find_archived_linked_source_passages(self.parent_path,self.query)
        self.assertEqual(0,report['linked_original_sources_checked'])
        self.assertEqual(0,report['linked_source_excerpt_candidates'])
        missing=report['parent_original_HTML_article_link_proof']['unavailable_publisher_references_not_followed']
        self.assertEqual(1,len(missing))
        self.assertEqual('catalogued_but_not_approved_completed_html_or_text',missing[0]['archive_status'])
        self.assertTrue(missing[0]['not_a_source_excerpt_or_answer'])
        self.assertEqual([(self.parent_path,1),(self.parent_path,1)],self.fenced)

    def test_bounded_and_non_html_parent_requests_refused_before_any_source_read(self):
        for path in ('../private.html','docs/other.pdf','/data/private.html'):
            with self.subTest(path=path),self.assertRaises(ValueError):
                self.cli.find_archived_linked_source_passages(path,self.query)
        for maxlinks in (0,6,True):
            with self.subTest(maxlinks=maxlinks),self.assertRaises(ValueError):
                self.cli.find_archived_linked_source_passages(self.parent_path,self.query,limit=maxlinks)
        self.assertEqual([],self.checked_calls)

    def test_fake_candidate_outside_exact_original_link_cannot_get_excerpt(self):
        corrupt=dict(self.child,original_sha256='f'*64)
        with patch.object(self.cli.library,'checked_original',side_effect=lambda path,**kw: self.parent if path==self.parent_path else corrupt):
            with self.assertRaisesRegex(ValueError,'linked original publisher file changed'):
                self.cli.find_archived_linked_source_passages(self.parent_path,self.query)

if __name__=='__main__':unittest.main()
