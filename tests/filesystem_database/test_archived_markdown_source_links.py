"""Offline SHA-attested literal publisher Markdown navigation, never document fabrication."""
import hashlib
import unittest
from file_system_catalog.archived_markdown_source_links import markdown_link_candidates


def sha(s):return hashlib.sha256(s).hexdigest()

class PublisherMarkdownOriginalLinks(unittest.TestCase):
    def setUp(self):
        self.parent='docs/navigation.md'
        self.url='https://docs.example.org/navigation.md'
        self.good='docs/transforming.md';self.mapper='docs/mapper.md'
        self.catalog=[{'path':self.parent,'url':self.url,'kind':'text','title':'Navigation'},
           {'path':self.good,'url':'https://docs.example.org/transforming.md','kind':'text','title':'Transform data'},
           {'path':self.mapper,'url':'https://docs.example.org/mapper.md','kind':'text','title':'Map fields'}]
        self.ledger={p:{'path':p,'url':x['url'],'status':'complete','sha256':sha(p.encode()),'bytes_received':155} for x in self.catalog for p in [x['path']]}
        self.raw=('Café introductory source\n\n'
          '  * [Map fields](/mapper.md)\n'
          '  * [Transform data](/transforming.md)\n'
          '\n```markdown\n[Map fields](/mapper.md)\n```\n'
          '    [Map fields](/mapper.md)\n'
          'Inline `[Map fields](/mapper.md)` is code.\n'
          '[Outside](https://evil.example.org/outside.md)\n'
          '<!-- [Map fields](/mapper.md) -->\n').encode('utf8')

    def select(self,**kw):
        args={'raw_markdown':self.raw,'original_sha256':sha(self.raw),
              'publisher_url':self.url,'catalog':self.catalog,
              'completed_receipts':self.ledger,'query':'How do I map fields into n8n node parameters?',
              'original_path':self.parent,'limit':5}
        args.update(kw)
        return markdown_link_candidates(**args)

    def test_original_UTF8_literal_byte_coordinates_and_full_SHA_agree(self):
        x=self.select()
        self.assertEqual(1,x['approved_matched_catalog_link_candidates'])
        self.assertEqual(1,len(x['selected_links']))
        l=x['selected_links'][0]
        self.assertEqual(self.mapper,l['catalog_target_original_path'])
        literal=self.raw[l['publisher_link_literal_start_byte']:l['publisher_link_literal_end_byte']]
        self.assertEqual(b'[Map fields](/mapper.md)',literal)
        self.assertEqual(sha(literal),l['publisher_link_literal_sha256'])
        self.assertTrue(l['publisher_link_occurs_in_SHA_attested_original_Markdown_bytes'])
        self.assertFalse(l['target_original_bytes_verified_by_this_function'])
        self.assertTrue(x['not_an_answer_or_semantic_relevance_proof'])

    def test_explicit_first_party_navigation_can_select_label_without_query_overlap(self):
        x=self.select(exact_link_text='Transform data')
        self.assertEqual(1,len(x['selected_links']))
        l=x['selected_links'][0]
        self.assertEqual(self.good,l['catalog_target_original_path'])
        self.assertTrue(l['explicit_navigation_label_not_semantic_relevance'])
        self.assertEqual([],l['matched_query_terms'])
        self.assertTrue(l['candidate_not_answer'])

    def test_explicit_navigation_never_claims_semantic_relevance_even_when_words_overlap(self):
        x=self.select(exact_link_text='Map fields')
        self.assertEqual(1,len(x['selected_links']))
        self.assertTrue(x['selected_links'][0]['matched_query_terms'])
        self.assertTrue(x['selected_links'][0]['explicit_navigation_label_not_semantic_relevance'])
        self.assertTrue(x['selected_links'][0]['candidate_not_answer'])

    def test_code_fences_inline_indents_HTML_comments_and_external_urls_not_promoted(self):
        result=self.select()
        self.assertEqual(3,result['original_Markdown_publisher_literal_reference_count'])
        self.assertEqual(1,result['approved_matched_catalog_link_candidates'])
        self.assertEqual(1,len(result['selected_links']))

    def test_changed_ledger_after_approved_query_is_a_new_missing_status(self):
        before=self.select()
        self.assertEqual(1,len(before['selected_links']))
        self.ledger[self.mapper]['status']='pending'
        later=self.select()
        self.assertEqual([],later['selected_links'])
        self.assertEqual('catalogued_but_not_approved_completed_source',
                         later['unavailable_publisher_references_not_followed'][0]['archive_status'])

    def test_changed_catalog_URL_does_not_use_cached_approval(self):
        self.assertEqual(1,len(self.select()['selected_links']))
        self.catalog[2]['url']='https://docs.example.org/another.md'
        later=self.select()
        self.assertEqual([],later['selected_links'])
        self.assertEqual('not_in_existing_catalog',
                         later['unavailable_publisher_references_not_followed'][0]['archive_status'])

    def test_duplicate_publisher_URL_must_not_authorize_either_ambiguous_source(self):
        duplicated=self.catalog+[{'path':'another/alias.md','url':self.catalog[2]['url'],'kind':'text','title':'Map fields'}]
        out=self.select(catalog=duplicated)
        self.assertEqual([],out['selected_links'])

    def test_unapproved_unacquired_reference_is_missing_not_quoted(self):
        raw=b'[Map fields](/unavailable.md)'
        value=self.select(raw_markdown=raw,original_sha256=sha(raw))
        self.assertEqual(0,value['approved_matched_catalog_link_candidates'])
        self.assertEqual(1,value['missing_or_unapproved_official_reference_count'])
        self.assertTrue(value['unavailable_publisher_references_not_followed'][0]['publisher_destination_contents_unavailable'])

    def test_multiple_backtick_code_spans_never_promote_fabricated_original_links(self):
        samples=[
            b'Ignore ``[Map fields](/mapper.md)`` original example.',
            b'Ignore ```[Map fields](/mapper.md)``` original example.',
            b'Ignore ``code that contains [Map fields](/mapper.md) and more``.',
            b'Unclosed inline double ``\n[Map fields](/mapper.md)\n`` end.',
            b'Markdown prose <!-- [Map fields](/mapper.md) --> comment.',
            b'Open comment <!--\n[Map fields](/mapper.md)\n--> end.',
        ]
        for raw in samples:
            with self.subTest(raw=raw[:30]):
                result=self.select(raw_markdown=raw,original_sha256=sha(raw))
                self.assertEqual([],result['selected_links'])
                self.assertEqual(0,result['original_Markdown_publisher_literal_reference_count'])

    def test_accepted_original_link_after_closed_double_backtick_code_span(self):
        raw=(b'Ignore ``[Map fields](/mapper.md)`` but follow '
             b'[Map fields](/mapper.md) to real source.')
        result=self.select(raw_markdown=raw,original_sha256=sha(raw))
        self.assertEqual(1,result['original_Markdown_publisher_literal_reference_count'])
        self.assertEqual(1,len(result['selected_links']))
        ref=result['selected_links'][0]
        self.assertEqual(b'[Map fields](/mapper.md)',raw[ref['publisher_link_literal_start_byte']:ref['publisher_link_literal_end_byte']])

    def test_real_firstparty_markdown_link_after_multiline_code_span_close(self):
        raw=(b'Before multi ``\n[Map fields](/mapper.md)\n'
             b'`` the real [Map fields](/mapper.md) link follows.')
        result=self.select(raw_markdown=raw,original_sha256=sha(raw))
        self.assertEqual(1,result['original_Markdown_publisher_literal_reference_count'])
        self.assertEqual(1,len(result['selected_links']))

    def test_same_origin_requires_explicit_HTTPS_and_no_credentials(self):
        raw=(b'[Map fields](https://other.example.org/mapper.md)\n'
             b'[Map fields](http://docs.example.org/mapper.md)\n'
             b'[Map fields](javascript:alert%281%29)\n'
             b'[Map fields](https://user:password@docs.example.org/mapper.md)\n')
        out=self.select(raw_markdown=raw,original_sha256=sha(raw))
        self.assertEqual([],out['selected_links'])
        self.assertEqual(0,out['missing_or_unapproved_official_reference_count'])

    def test_bad_original_SHA_and_unsafe_path_refused(self):
        with self.assertRaisesRegex(ValueError,'whole original SHA'):
            self.select(original_sha256='0'*64)
        for path in ('../secret.md','/private/secret.md','docs/./malformed.md','docs/n8n.txt'):
            with self.subTest(path=path),self.assertRaises(ValueError):
                self.select(original_path=path)

    def test_unsafe_or_oversized_original_or_query_refused(self):
        for value in (b'',b'a'*(8*1024*1024+1)):
            with self.assertRaises(ValueError):self.select(raw_markdown=value,original_sha256=sha(value))
        with self.assertRaises(ValueError):self.select(limit=True)
        with self.assertRaises(ValueError):self.select(query='')

if __name__=='__main__':unittest.main()
