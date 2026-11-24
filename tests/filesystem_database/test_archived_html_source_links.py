"""Signed-off same-origin HTML article reference navigation fixtures."""
import hashlib
import unittest
from file_system_catalog.archived_html_source_links import source_link_candidates

def sha(b):return hashlib.sha256(b).hexdigest()

class ArchivedHtmlReferenceTests(unittest.TestCase):
    def setUp(self):
        self.original='manual/chapter.html'
        self.url='https://docs.example.org/engine/chapter'
        self.child='manual/child.html'
        self.catalog=[{'path':self.original,'kind':'html','url':self.url,'title':'Chapter Overview'},
                      {'path':self.child,'kind':'html','url':'https://docs.example.org/engine/chapter/mesh','title':'Texture and mesh loading'},
                      {'path':'other/unrelated.html','kind':'html','url':'https://evil.example.org/mesh','title':'Texture and mesh loading'}]
        self.ledger={self.child:{'path':self.child,'url':'https://docs.example.org/engine/chapter/mesh',
                                 'sha256':sha(b'exact target original'),'bytes_received':123,'status':'complete'}}
        self.raw=(b'<html><body><nav><a href="/engine/chapter/bad">Mesh spoof</a></nav>'
                  b'<article><h1>Overview</h1><a href="/engine/chapter/mesh">Texture and mesh loading</a>'
                  b'<a href="https://evil.example.org/mesh">Texture and mesh loading</a>'
                  b'<script><a href="/engine/chapter/mesh">Fake script mesh</a></script>'
                  b'</article></body></html>')

    def call(self,**kw):
        args={'raw_html':self.raw,'original_sha256':sha(self.raw),'publisher_url':self.url,
              'catalog':self.catalog,'completed_receipts':self.ledger,
              'query':'How do I load mesh data from disk without stalls?',
              'original_path':self.original,'limit':3}
        args.update(kw)
        return source_link_candidates(**args)

    def test_exact_first_party_original_article_link_to_existing_original_receipt(self):
        result=self.call()
        self.assertEqual(1,result['matching_verified_catalog_receipt_link_candidates'])
        self.assertEqual(1,len(result['selected_links']))
        link=result['selected_links'][0]
        self.assertEqual(self.child,link['catalog_target_original_path'])
        self.assertEqual(self.ledger[self.child]['sha256'],link['catalog_target_acquisition_sha256'])
        self.assertEqual(['mesh'],link['matched_query_terms'])
        self.assertTrue(link['source_link_occurs_in_original_HTML_article_or_main'])
        self.assertFalse(link['target_original_bytes_verified_by_this_function'])
        self.assertTrue(result['no_answer_truth_or_page_quote_proved'])

    def test_original_source_sha_or_ambiguous_catalog_url_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'source SHA'):
            self.call(original_sha256='0'*64)
        ambiguous=self.catalog+[{'path':'different/duplicate.html','kind':'html',
                                'url':self.catalog[1]['url'],'title':'Different edition same URL'}]
        self.assertEqual([],self.call(catalog=ambiguous)['selected_links'])

    def test_deny_external_non_https_script_navigation_and_unlisted_targets(self):
        raw=(b'<article><a href="javascript:alert(1)">Mesh data</a>'
             b'<a href="//evil.example.org/mesh">Texture and mesh loading</a>'
             b'<a href="/engine/chapter/unknown">Mesh file</a></article>')
        self.assertEqual([],self.call(raw_html=raw,original_sha256=sha(raw))['selected_links'])

    def test_relevant_missing_same_origin_original_is_reported_only_as_gap(self):
        raw=(b'<article><p><a href="/engine/mesh-compatibility">Make mesh compatible with asynchronous loading</a></p></article>')
        result=self.call(raw_html=raw,original_sha256=sha(raw),
                         query='How can I make a mesh compatible with asynchronous loading?')
        self.assertEqual([],result['selected_links'])
        self.assertEqual(1,result['same_origin_relevant_unarchived_or_unapproved_references'])
        gap=result['unavailable_publisher_references_not_followed'][0]
        self.assertEqual('not_in_existing_catalog',gap['archive_status'])
        self.assertTrue(gap['publisher_destination_contents_unavailable'])
        self.assertTrue(gap['no_download_or_external_navigation'])
        self.assertTrue(gap['not_a_source_excerpt_or_answer'])

    def test_listed_but_unapproved_original_is_separate_from_absent_catalog(self):
        raw=b'<article><a href="/engine/chapter/mesh">Texture and mesh loading</a></article>'
        ledger={self.child:dict(self.ledger[self.child],status='pending')}
        result=self.call(raw_html=raw,original_sha256=sha(raw),completed_receipts=ledger)
        self.assertEqual([],result['selected_links'])
        gap=result['unavailable_publisher_references_not_followed'][0]
        self.assertEqual('catalogued_but_not_approved_completed_html_or_text',gap['archive_status'])
        self.assertTrue(gap['publisher_destination_contents_unavailable'])

    def test_canonical_url_cache_does_not_cache_publisher_approval_or_ledger_status(self):
        # First populate the *pure string normalization* URL cache. Mutation
        # of the separately consulted current receipt must still fail closed.
        from file_system_catalog.archived_html_source_links import _canonical_url_string
        before=self.call()
        self.assertEqual(1,len(before['selected_links']))
        self.assertGreater(_canonical_url_string.cache_info().currsize,0)
        prior=dict(self.ledger[self.child])
        self.ledger[self.child]['status']='pending'
        try:
            after=self.call()
            self.assertEqual([],after['selected_links'])
            self.assertEqual('catalogued_but_not_approved_completed_html_or_text',
                             after['unavailable_publisher_references_not_followed'][0]['archive_status'])
        finally:
            self.ledger[self.child]=prior
        restored=self.call()
        self.assertEqual(before['selected_links'],restored['selected_links'])

    def test_cached_canonicalization_does_not_resurrect_changed_official_catalog_URL(self):
        before=self.call()
        self.assertEqual(1,len(before['selected_links']))
        prev=self.catalog[1]['url']
        self.catalog[1]['url']='https://docs.example.org/different-publisher-guide'
        try:
            after=self.call()
            self.assertEqual([],after['selected_links'])
            self.assertEqual('not_in_existing_catalog',
                             after['unavailable_publisher_references_not_followed'][0]['archive_status'])
        finally:
            self.catalog[1]['url']=prev
        self.assertEqual(1,len(self.call()['selected_links']))

    def test_canonical_URL_cache_rejects_unhashable_and_unsafe_strings_as_before(self):
        from file_system_catalog.archived_html_source_links import _canonical_url
        for value in ([],{},None,1,'https://evil.example.org/a?token=private',
                      'http://docs.example.org/a','javascript:alert(1)',
                      'https://user:password@docs.example.org/a'):
            with self.subTest(value=repr(value)):
                self.assertIsNone(_canonical_url(value))
        self.assertEqual('https://docs.example.org/source',_canonical_url('https://docs.example.org/source/'))

    def test_no_complete_archived_target_receipt_never_promoted(self):
        missing=dict(self.ledger[self.child],status='pending')
        self.assertEqual([],self.call(completed_receipts={self.child:missing})['selected_links'])
        tampered=dict(self.ledger[self.child],url='https://docs.example.org/other')
        self.assertEqual([],self.call(completed_receipts={self.child:tampered})['selected_links'])
        self.assertEqual([],self.call(completed_receipts={})['selected_links'])

    def test_main_fallback_when_no_article_and_no_navigation_sidebars(self):
        raw=(b'<main><nav><a href="/engine/chapter/mesh">FAKE mesh nav</a></nav>'
             b'<p><a href="/engine/chapter/mesh">Texture and mesh loading</a></p></main>')
        links=self.call(raw_html=raw,original_sha256=sha(raw))['selected_links']
        self.assertEqual(1,len(links))
        self.assertEqual('Texture and mesh loading',links[0]['publisher_article_link_text'])

    def test_zero_query_term_overlap_keeps_possible_source_unselected(self):
        r=self.call(query='separate astrophysics cosmology neutrino simulations')
        self.assertEqual([],r['selected_links'])

    def test_unsafe_original_path_or_unbounded_top_are_refused(self):
        for path in ('../book.html','/data/var/Knowledge Library/private.html','bad\\book.html'):
            with self.subTest(path=path),self.assertRaises(ValueError):
                self.call(original_path=path)
        for limit in (0,6,True):
            with self.subTest(limit=limit),self.assertRaises(ValueError):
                self.call(limit=limit)

if __name__=='__main__':unittest.main()
