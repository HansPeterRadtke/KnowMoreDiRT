"""Independent original-HTML text validation; no quote or answer claims."""
import hashlib
import unittest
from file_system_catalog.publisher_html_dom_evidence import attach_original_html_dom_anchors

def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()

def hit(text:str)->dict:
    return {'source_excerpt':text,'excerpt_sha256':sha(text.encode('utf8')),
            'status':'source_lexical_candidate_not_verified_answer'}

class PublisherHtmlDomAnchorTests(unittest.TestCase):
    def test_approved_article_excerpt_matches_normalized_original_with_entities(self):
        raw=b'<html><head><script>ignore this attacker generated script text</script></head><body><article><h2>Normal Map</h2><p>Set the RGB values of the normal map image to make the rendered mesh look bumpy under directional lighting without changing the actual mesh geometry.</p><p>Additional independent explanation of image texture mode and coordinate space.</p></article></body></html>'
        excerpt='Normal Map Set the RGB values of the normal map image to make the rendered mesh look bumpy under directional lighting without changing the actual mesh geometry.'
        r=hit(excerpt)
        self.assertEqual(1,attach_original_html_dom_anchors(raw,sha(raw),[r]))
        a=r['publisher_html_dom_anchor'];self.assertEqual('article',a['container'])
        self.assertEqual(sha(excerpt.encode('utf8')),a['normalized_excerpt_sha256'])
        self.assertFalse(a['literal_publisher_byte_interval_verified'])
        self.assertFalse(a['answer_factual_sufficiency_verified'])
        self.assertEqual('source_lexical_candidate_not_verified_answer',r['status'])

    def test_main_element_without_article_is_independent_publisher_proof(self):
        excerpt='Perspective Transform removes apparent perspective distortion from photographs of material surfaces and rectifies the photographed quadrilateral into a rectangle.'
        raw=('<html><body><main><h1>Guide</h1><p>'+excerpt+'</p></main></body></html>').encode()
        r=hit(excerpt)
        self.assertEqual(1,attach_original_html_dom_anchors(raw,sha(raw),[r]))
        self.assertEqual('main',r['publisher_html_dom_anchor']['container'])

    def test_malicious_script_content_does_not_count_as_original_publisher_article(self):
        excerpt='This long made up sentence appears only inside a script tag and cannot establish a visible publisher technical explanation.'
        raw=('<main><script>'+excerpt+'</script><p>Unrelated true publisher paragraph.</p></main>').encode()
        r=hit(excerpt)
        self.assertEqual(0,attach_original_html_dom_anchors(raw,sha(raw),[r]))
        self.assertNotIn('publisher_html_dom_anchor',r)

    def test_no_main_article_or_ambiguous_repeated_text_remains_unverified(self):
        excerpt='The original HTML author explains how normal mapping and shading depend on tangent space and image coordinates.'
        for raw in (('<div>'+excerpt+'</div>').encode(),
                    ('<main><p>'+excerpt+'</p><p>'+excerpt+'</p></main>').encode()):
            with self.subTest(raw=raw[:10]):
                r=hit(excerpt)
                self.assertEqual(0,attach_original_html_dom_anchors(raw,sha(raw),[r]))
                self.assertNotIn('publisher_html_dom_anchor',r)

    def test_original_sha_wrong_or_derived_excerpt_sha_wrong_refused(self):
        excerpt='The publication defines original geometry processing methods and discusses the costs of photorealistic reconstruction.'
        raw=('<main><p>'+excerpt+'</p></main>').encode();r=hit(excerpt)
        with self.assertRaisesRegex(ValueError,'original SHA'):
            attach_original_html_dom_anchors(raw,'0'*64,[r])
        self.assertNotIn('publisher_html_dom_anchor',r)
        r['excerpt_sha256']='a'*64
        with self.assertRaisesRegex(ValueError,'excerpt SHA'):
            attach_original_html_dom_anchors(raw,sha(raw),[r])

    def test_only_short_or_absent_source_fragment_refused_without_enlarging_it(self):
        raw=b'<main><p>The original publisher says this source has a very short title.</p></main>'
        excerpt='The original publisher says this source has a very short title.'
        r=hit(excerpt)
        self.assertEqual(0,attach_original_html_dom_anchors(raw,sha(raw),[r]))
        self.assertNotIn('publisher_html_dom_anchor',r)

if __name__=='__main__':unittest.main()
