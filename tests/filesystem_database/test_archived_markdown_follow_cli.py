"""Read-only KMD original Markdown links + source bytes fail closed across races."""
import hashlib
import importlib.util
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))


def sha(v):return hashlib.sha256(v).hexdigest()

SPEC=importlib.util.spec_from_file_location('archived_markdown_cli_test',Path(__file__).resolve().parents[2]/'scripts'/'raspi_archived_markdown_follow_links.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)

class SourceAttestedMarkdownOriginalNavigationTests(unittest.TestCase):
    def setUp(self):
        self.parent='docs/overview.md';self.child='docs/methods.md';self.query='How can I transform data in nodes using expressions?'
        self.parent_url='https://docs.example.org/overview.md';self.child_url='https://docs.example.org/methods.md'
        self.parent_raw=b'# Overview\n[Transform data](/methods.md)\n'
        self.child_raw=b'# Transform data\n\nUse expressions to modify data in workflow nodes.\n'
        self.excerpt='Use expressions to modify data in workflow nodes.'
        self.derived='[[SOURCE UNIT: Original plain text]]\n'+self.excerpt+'\n'
        self.parent_record={'path':self.parent,'url':self.parent_url,'kind':'text','title':'Overview'}
        self.child_record={'path':self.child,'url':self.child_url,'kind':'text','title':'Transform data'}
        self.parent_receipt={'path':self.parent,'url':self.parent_url,'status':'complete','sha256':sha(self.parent_raw),'bytes_received':len(self.parent_raw)}
        self.child_receipt={'path':self.child,'url':self.child_url,'status':'complete','sha256':sha(self.child_raw),'bytes_received':len(self.child_raw)}
        self.parent_source={'original_path':self.parent,'title':'Overview','original_sha256':sha(self.parent_raw),'derived_sha256':sha(b'parent derived'),'publisher_text_bytes':self.parent_raw,'derived_text':'parent derived'}
        self.child_source={'original_path':self.child,'title':'Transform data','original_sha256':sha(self.child_raw),'derived_sha256':sha(self.derived.encode()),'publisher_text_bytes':self.child_raw,'derived_text':self.derived}
        self.records=([self.parent_record,self.child_record],
          {self.parent:self.parent_record,self.child:self.child_record},
          {self.parent:self.parent_receipt,self.child:self.child_receipt},set(),set())
        self.calls=[]
        self.patcher=[]
        def mockit(obj,name,**kw):
            p=patch.object(obj,name,**kw);p.start();self.addCleanup(p.stop)
            self.patcher.append(p)
        mockit(M,'_records',return_value=self.records)
        mockit(M,'_native_file_id',side_effect=lambda r:11 if r==self.parent else 12)
        mockit(M,'publication_receipt_matches',return_value=True)
        def checked(p,**kw):
            self.calls.append(('checked',p))
            return self.parent_source if p==self.parent else self.child_source
        mockit(M.library,'checked_original',side_effect=checked)
        self.passage={
          'source_excerpt':self.excerpt,
          'excerpt_start_char':self.derived.index(self.excerpt),
          'excerpt_end_char':self.derived.index(self.excerpt)+len(self.excerpt),
          'excerpt_sha256':sha(self.excerpt.encode()),
          'source_section':'Transform data',
          'matched_excerpt_terms':['expressions','data','nodes'],
          'publisher_excerpt':{'start_byte':self.child_raw.index(self.excerpt.encode()),
                               'end_byte':self.child_raw.index(self.excerpt.encode())+len(self.excerpt.encode()),
                               'sha256':sha(self.excerpt.encode()),'encoding':'utf-8'},
        }
        self.passages={'passages':[self.passage]}
        mockit(M,'source_evidence_candidates',return_value=self.passages)
        def fence(source,*,original_path,native_file_id):
            self.calls.append(('fence',original_path))
            return {'latest_source_map_revision_and_native_file_id_rechecked':True}
        mockit(M.library,'final_original_revision_guard',side_effect=fence)

    def test_exact_literal_parent_link_bytes_and_target_original_passage_are_verified(self):
        result=M.follow_current_archived_markdown_references(self.parent,self.query,exact_link_text='Transform data')
        self.assertEqual(1,result['linked_original_sources_checked'])
        self.assertEqual(1,result['independently_publisher_original_plaintext_byte_attested_passages'])
        child=result['linked_approved_current_original_sources'][0]
        self.assertEqual(self.child,child['original_path'])
        self.assertEqual(sha(self.child_raw),child['original_sha256'])
        p=child['passage_candidates'][0]
        self.assertTrue(p['publisher_plaintext_byte_interval_independently_SHA_verified'])
        self.assertEqual(self.excerpt,p['current_derived_text_excerpt'])
        link=child['publisher_source_link_proof']
        literal=self.parent_raw[link['publisher_link_literal_start_byte']:link['publisher_link_literal_end_byte']]
        self.assertEqual(b'[Transform data](/methods.md)',literal)
        self.assertEqual(sha(literal),link['publisher_link_literal_sha256'])
        self.assertEqual([('checked',self.parent),('checked',self.child),('fence',self.child),('fence',self.child),('fence',self.parent)],self.calls)
        self.assertFalse(result['safe_to_cutover_live'])
        self.assertTrue(result['no_external_network_or_new_source_acquisition'])

    def test_stale_target_original_SHA_after_lookup_is_rejected(self):
        bad={**self.child_source,'original_sha256':'f'*64}
        with patch.object(M.library,'checked_original',side_effect=lambda r,**kw: self.parent_source if r==self.parent else bad):
            with self.assertRaisesRegex(ValueError,'publisher SHA differs'):
                M.follow_current_archived_markdown_references(self.parent,self.query,exact_link_text='Transform data')

    def test_target_source_revision_change_after_initial_proof_rejects_group(self):
        def fence(src,*,original_path,native_file_id):
            if original_path==self.child:
                self.calls.append(('fence',self.child))
                if self.calls.count(('fence',self.child))==2:
                    raise ValueError('target source changed after another source')
            return {'latest_source_map_revision_and_native_file_id_rechecked':True}
        with patch.object(M.library,'final_original_revision_guard',side_effect=fence):
            with self.assertRaisesRegex(ValueError,'target source changed'):
                M.follow_current_archived_markdown_references(self.parent,self.query,exact_link_text='Transform data')

    def test_parent_source_mutation_after_target_check_rejects_response(self):
        def fence(src,*,original_path,native_file_id):
            if original_path==self.parent:raise ValueError('parent official Markdown original changed')
            return {'latest_source_map_revision_and_native_file_id_rechecked':True}
        with patch.object(M.library,'final_original_revision_guard',side_effect=fence):
            with self.assertRaisesRegex(ValueError,'parent official Markdown original changed'):
                M.follow_current_archived_markdown_references(self.parent,self.query,exact_link_text='Transform data')

    def test_modified_derived_excerpt_hash_fails_closed(self):
        fake={'passages':[{**self.passage,'excerpt_sha256':'f'*64}]}
        with patch.object(M,'source_evidence_candidates',return_value=fake):
            with self.assertRaisesRegex(ValueError,'does not match'):
                M.follow_current_archived_markdown_references(self.parent,self.query,exact_link_text='Transform data')

    def test_modified_publisher_original_byte_excerpt_hash_fails_closed(self):
        fake={'passages':[{**self.passage,'publisher_excerpt':{**self.passage['publisher_excerpt'],'sha256':'f'*64}}]}
        with patch.object(M,'source_evidence_candidates',return_value=fake):
            with self.assertRaisesRegex(ValueError,'publisher-original plaintext byte interval failed'):
                M.follow_current_archived_markdown_references(self.parent,self.query,exact_link_text='Transform data')

    def test_genuine_publisher_byte_interval_with_wrong_quote_text_refused(self):
        # A SHA-valid original publisher interval is not evidence unless it
        # decodes to the EXACT cited source excerpt, not unrelated original
        # header or nearby text, even when that other interval has a real SHA.
        other=self.child_raw[:3]
        wrong={'passages':[{**self.passage,'publisher_excerpt':{
           'start_byte':0,'end_byte':3,'sha256':sha(other),'encoding':'utf-8'}}]}
        with patch.object(M,'source_evidence_candidates',return_value=wrong):
            with self.assertRaisesRegex(ValueError,'does not quote the retrieved source excerpt'):
                M.follow_current_archived_markdown_references(self.parent,self.query,exact_link_text='Transform data')

    def test_invalid_publisher_byte_quote_encoding_refused(self):
        wrong={'passages':[{**self.passage,'publisher_excerpt':{
            **self.passage['publisher_excerpt'],'encoding':'ISO-8859-1'}}]}
        with patch.object(M,'source_evidence_candidates',return_value=wrong):
            with self.assertRaisesRegex(ValueError,'unsupported claimed publisher original quotation encoding'):
                M.follow_current_archived_markdown_references(self.parent,self.query,exact_link_text='Transform data')

    def test_missing_or_unapproved_child_is_not_quoted_or_substituted(self):
        receipt={**self.child_receipt,'status':'pending'}
        revised=(self.records[0],self.records[1],{self.parent:self.parent_receipt,self.child:receipt},set(),set())
        with patch.object(M,'_records',return_value=revised):
            result=M.follow_current_archived_markdown_references(self.parent,self.query,exact_link_text='Transform data')
        self.assertEqual([],result['linked_approved_current_original_sources'])
        self.assertEqual(1,result['parent_original_Markdown_literal_byte_link_proof']['missing_or_unapproved_official_reference_count'])
        self.assertNotIn(('checked',self.child),self.calls)

    def test_non_markdown_source_unsafe_path_and_unbounded_input_denied(self):
        for path in ('docs/other.html','../sensitive.md','/private/secret.md','docs/./original.md'):
            with self.subTest(path=path),self.assertRaises(ValueError):
                M.follow_current_archived_markdown_references(path,self.query)
        for limit in (0,6,True):
            with self.subTest(limit=limit),self.assertRaises(ValueError):
                M.follow_current_archived_markdown_references(self.parent,self.query,limit=limit)

if __name__=='__main__':unittest.main()
