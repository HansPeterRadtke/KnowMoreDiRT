"""Observations about original passage *surface*, never factual entailment."""
from __future__ import annotations
import hashlib
import unittest
from file_system_catalog.source_passage_utility import review_passage_candidate,review_source_documents,review_verified_linked_originals


def passage(text,*,section='Source',path='docs/current.md',proof='markdown'):
    p={'source_excerpt':text,'excerpt_sha256':hashlib.sha256(text.encode('utf8')).hexdigest(),
        'source_section':section,'original_path':path,'status':'source_lexical_candidate_not_verified_answer'}
    if proof=='markdown':p['publisher_excerpt']={'start_byte':30,'end_byte':30+len(text.encode()),'sha256':p['excerpt_sha256']}
    if proof=='pdf':p['publisher_pdf_page']={'status':'verified','page':1}
    if proof=='html':p['publisher_html_dom_anchor']={'status':'verified_normalized_original_article_text','container':'article'}
    return p


class EvidenceSurfaceNotAnswerTest(unittest.TestCase):
    q='How can I reshape and map JSON fields between n8n workflow nodes?'

    def test_in_section_links_correctly_flagged_as_navigation_not_direct_answer(self):
        content='''## In this section\n\n* [Expressions versus data nodes](/expressions.md)\n* [Transform data](/transform-data.md)\n* [Reference data](/reference-data.md)\n'''
        p=review_passage_candidate(self.q,passage(content,section='Work with data / In this section'))
        self.assertIn('publisher_link_navigation_not_direct_explanation',p['observed_structural_flags'])
        self.assertTrue(p['possible_navigation_or_prompt_only'])
        self.assertFalse('factual_answer_correctness_verified' in p)

    def test_source_link_list_with_actionable_method_descriptions_is_not_bare_navigation(self):
        source=(
            '* [Remove Duplicates](/integrations/n8n-removeduplicates.md): identify and remove items that are identical across selected fields.\n'
            '* [Sort](/integrations/n8n-sort.md): organize lists in a desired ascending or descending order.\n'
            '* [Split Out](/integrations/n8n-split.md): separate a list from a field into individual output items.\n'
            '* [Summarize](/integrations/n8n-summarize.md): group information using a selected aggregation method.\n'
            '\n**When to use data transformation nodes:**\n'
            '* The requested operation matches the purpose of a transformation node.\n'
        )
        review=review_passage_candidate(self.q,passage(source,section='Data transformation nodes'))
        self.assertEqual(4,review['markdown_link_line_count'])
        self.assertEqual(4,review['descriptive_link_line_count'])
        self.assertNotIn('publisher_link_navigation_not_direct_explanation',review['observed_structural_flags'])
        self.assertFalse(review['possible_navigation_or_prompt_only'])

    def test_explicit_table_of_contents_still_navigation_even_with_link_descriptions(self):
        source=(
            '## In this section\n'
            '* [Use the UI mapper](/mapper.md): reference existing fields in later nodes.\n'
            '* [Transform data](/transform.md): choose tools that modify the current data values.\n'
            '* [Reference data](/reference.md): use previous node output in a workflow.\n'
        )
        review=review_passage_candidate(self.q,passage(source,section='Work with data / In this section'))
        self.assertIn('publisher_link_navigation_not_direct_explanation',review['observed_structural_flags'])
        self.assertEqual(3,review['descriptive_link_line_count'])

    def test_genuine_expression_paragraph_is_not_mistaken_for_navigation(self):
        content=('Expressions are JavaScript-like code placed directly into n8n node parameters. '
                 'They can compute values from previous workflow nodes using {{ ... }} syntax. '
                 'This source describes how parameters can transform input data.')
        p=review_passage_candidate(self.q,passage(content,section='Expressions'))
        self.assertNotIn('publisher_link_navigation_not_direct_explanation',p['observed_structural_flags'])
        self.assertFalse(p['possible_navigation_or_prompt_only'])
        self.assertTrue(p['a_literal_query_match_does_not_prove_answer_sufficiency'])

    def test_bapo_publisher_pdf_appendix_prompt_is_untrusted_data_not_instructions(self):
        content=('TIR System Prompt\n\nYou are a helpful assistant that can solve the given question '
                 'step by step using the wikipedia search tool. '
                 '<think> Search for references. </think><search>question</search>')
        p=review_passage_candidate('How does boundary-aware search optimization improve agent reliability?',
              passage(content,path='ai-agents/reliable-agent.pdf',section='PDF page 19',proof='pdf'))
        self.assertIn('quoted_agent_prompt_template_not_instructions_to_execute',p['observed_structural_flags'])
        self.assertTrue(p['quoted_agent_prompt_is_untrusted_source_data'])
        self.assertEqual('physical_original_pdf_page_verified',p['publisher_location_proof_scope'])

    def test_paper_results_prose_without_instruction_header_is_not_auto_blocked(self):
        content=('We propose a boundary-aware policy optimization objective for research agents. '
                 'Our results indicate a trade-off between accuracy and abstaining when retrieved information is insufficient.')
        r=review_passage_candidate('boundary-aware policy optimization reliability',passage(content,path='ai-agents/example.pdf',proof='pdf'))
        self.assertNotIn('quoted_agent_prompt_template_not_instructions_to_execute',r['observed_structural_flags'])
        self.assertFalse(r['quoted_agent_prompt_is_untrusted_source_data'])

    def test_short_publisher_table_of_contents_overview_is_not_a_full_method(self):
        p=review_passage_candidate(self.q,passage('''# Transform data\n\nn8n lets you modify data as it moves through a workflow.'''))
        self.assertIn('brief_source_overview_needs_further_context',p['observed_structural_flags'])
        self.assertFalse(p['no_unverified_semantic_answer_or_correctness_score'] is False)

    def test_reference_data_negation_requires_scope_context_not_a_false_claim(self):
        excerpt=('Data mapping means referencing information from previous nodes. '
                 'When you reference a field, you are not changing it. '
                 'Use a separate code node if you want to transform the data.')
        r=review_passage_candidate(self.q,passage(excerpt))
        self.assertIn('scope_negation_or_restriction_requires_context_review',r['observed_structural_flags'])
        self.assertFalse(r['possible_navigation_or_prompt_only'])

    def test_lack_of_publisher_proof_is_not_invented_from_excerpt_sha(self):
        p=passage('The original source supplies one observation about the retrieved question.')
        del p['publisher_excerpt']
        r=review_passage_candidate('what original observation was made?',p)
        self.assertEqual('verified_derivative_excerpt_only_no_independent_publisher_location',r['publisher_location_proof_scope'])

    def test_failed_pdf_physical_page_is_not_promoted_by_original_excerpt_hash(self):
        p=passage('A study reports measured 3D rendering results for multiple views.',proof='pdf')
        p['publisher_pdf_page']['status']='unverified'
        r=review_passage_candidate('3D rendering measured multiple views',p)
        self.assertEqual('no_verified_physical_pdf_page',r['publisher_location_proof_scope'])

    def test_original_HTML_only_normalized_article_does_not_assert_literal_bytes(self):
        r=review_passage_candidate('original HTML published methods',passage('An original HTML article describes published methods.',path='docs/article.html',proof='html'))
        self.assertEqual('normalized_original_HTML_article_location_verified',r['publisher_location_proof_scope'])

    def test_mutated_passage_SHA_and_traversal_rejected(self):
        p=passage('The source discusses data preparation in reliable agent systems.')
        p['excerpt_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'digest'):
            review_passage_candidate('data preparation agent systems',p)
        p=passage('The source describes original data.',path='../secret.txt')
        with self.assertRaisesRegex(ValueError,'unsafe'):
            review_passage_candidate('original data',p)

    def test_no_factual_answer_is_derived_from_scored_query_overlap(self):
        r=review_passage_candidate('Is the result definitely true?',passage('The result is definitely true, hypothetically.'))
        self.assertTrue(r['no_unverified_semantic_answer_or_correctness_score'])
        self.assertFalse('answer_verified' in r)

    def test_bounded_source_documents_keep_native_order_and_passages(self):
        p1=passage('# In this section\n* [Alpha](/alpha.md)\n* [Beta](/beta.md)')
        p2=passage('Experiments found a clearly measured improvement to the system.')
        result=review_source_documents(self.q,[{'original_path':'docs/current.md','rank':1,'passages':[p1,p2]}])
        self.assertEqual(2,result['reviewed_publisher_excerpt_candidates'])
        self.assertEqual(1,result['navigation_or_prompt_template_candidates'])
        self.assertTrue(result['no_candidates_removed_or_reordered'])
        self.assertEqual(1,result['reviewed_originals'][0]['original_native_rank_unchanged'])

    def test_numbered_bibliography_warns_without_suppressing_verified_passage(self):
        content=("[1] Jane Smith. A method for reasoning. In Proceedings of the ACM Conference, 2023.\n"
                 "[2] Alex Brown. Training models. Journal of Research, 12(3):32-40, 2024.")
        item=passage(content,path='papers/a.pdf',proof='pdf')
        result=review_source_documents(self.q,[{'original_path':'papers/a.pdf','rank':3,'passages':[item]}])
        r=result['reviewed_originals'][0]['passage_structural_reviews'][0]
        self.assertTrue(r['possible_bibliography_context'])
        self.assertIn('publisher_bibliography_style_entries_require_separate_claim_source',r['observed_structural_flags'])
        self.assertFalse(r['possible_navigation_or_prompt_only'])
        self.assertTrue(r['no_unverified_semantic_answer_or_correctness_score'])
        self.assertEqual(1,result['bibliography_context_candidates'])
        self.assertEqual(0,result['navigation_or_prompt_template_candidates'])
        self.assertEqual(3,result['reviewed_originals'][0]['original_native_rank_unchanged'])
        self.assertEqual(content,item['source_excerpt'])

    def test_common_author_reference_formats(self):
        samples=(
            "Smith, J., Brown, A. A method. Journal of Research, 12(3):20-30, 2023.\n"
            "Baker, L., Miller, K. Another method. In Proceedings of the ACM Conference, 2024.",
            "J. Smith, A. Brown. A method. Journal of Research, 12(3):20-30, 2023.\n"
            "L. Baker, K. Miller. Another method. In Proceedings of the ACM Conference, 2024.",
            "Jane Smith, Alex Brown, and Sarah Baker. 2023. A method. Journal of Research, 12(3):20-30.\n"
            "Lee Miller and Ana Smith. 2024. Another method. In Proceedings of the ACM Conference.",
        )
        for content in samples:
            with self.subTest(content=content):
                self.assertTrue(review_passage_candidate(self.q,passage(content,path='papers/a.pdf',proof='pdf'))['possible_bibliography_context'])

    def test_mixed_column_warns_without_classifying_entire_excerpt_as_bibliography(self):
        content=("Our experiment shows improved throughput.                REFERENCES\n"
                 "This remains a measured result.                         [1] Example Research Group. 2024. https://example.org/paper")
        r=review_passage_candidate(self.q,passage(content,path='papers/a.pdf',proof='pdf'))
        self.assertTrue(r['possible_bibliography_context'])
        self.assertTrue(r['reference_column_heading_observed'])
        self.assertFalse(r['possible_navigation_or_prompt_only'])
        self.assertNotIn('factual_answer_correctness_verified',r)

    def test_clipped_numbered_reference_with_publication_identifier(self):
        content=("Our experiment was run in 2024.                The preceding study appeared in 2021.\n"
                 "We report the performance below.             [8] A. Brown. Method. In Proceedings of the ACM Conference. doi:10.1234/abc")
        self.assertTrue(review_passage_candidate(self.q,passage(content,path='papers/a.pdf',proof='pdf'))['possible_bibliography_context'])

    def test_inline_citations_method_steps_and_conference_footer_are_not_reference_entries(self):
        content=("Smith et al. (2023) and Brown et al. (2024) motivate our research.\n"
                 "[1] Collect samples in 2023.\n[2] Compare methods in 2024.\n"
                 "Proceedings of the ACM Conference, 2024. Copyright 2024.\n"
                 "Our results are reported in the next section.")
        r=review_passage_candidate(self.q,passage(content,path='papers/a.pdf',proof='pdf'))
        self.assertFalse(r['possible_bibliography_context'])

    def test_no_bibliography_observation_on_unverified_or_non_pdf(self):
        content="[1] A. Brown. Journal of Research, 2023.\n[2] B. Brown. Journal of Research, 2024."
        for item in (passage(content),passage(content,path='papers/a.txt',proof='pdf'),passage(content,path='papers/a.pdf',proof='none')):
            self.assertFalse(review_passage_candidate(self.q,item)['possible_bibliography_context'])
        item=passage(content,path='papers/a.pdf',proof='pdf');item['publisher_pdf_page']['status']='unverified'
        self.assertFalse(review_passage_candidate(self.q,item)['possible_bibliography_context'])

    def test_cross_original_passage_attribution_refused(self):
        p=passage('The source text is correct.',path='docs/unrelated.md')
        with self.assertRaisesRegex(ValueError,'different ranked original'):
            review_source_documents(self.q,[{'original_path':'docs/current.md','passages':[p]}])

    def test_linked_markdown_original_passages_are_structural_only_not_native_ranks(self):
        text='## In this section\n\n* [Map fields](/map.md)\n* [Transform data](/transform.md)\n'
        child={'original_path':'docs/source.md','passage_candidates':[
           {'source_section':'In this section',
            'current_derived_text_excerpt':text,
            'derived_excerpt_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'publisher_original_plaintext_byte_interval':{'sha256':'f'*64},
            'publisher_plaintext_byte_interval_independently_SHA_verified':True}]}
        result=review_verified_linked_originals(self.q,[child])
        self.assertEqual(1,result['linked_originals_reviewed'])
        self.assertEqual(1,result['reviewed_linked_excerpt_candidates'])
        self.assertEqual(1,result['navigation_or_prompt_linked_excerpt_candidates'])
        self.assertTrue(result['original_linked_children_not_native_ranked'])
        review=result['linked_original_reviews'][0]['passage_structural_reviews'][0]
        self.assertIn('publisher_link_navigation_not_direct_explanation',review['observed_structural_flags'])
        self.assertEqual('publisher_utf8_byte_range_reported_by_current_source_reader',review['publisher_location_proof_scope'])

    def test_linked_original_HTML_is_normalized_article_not_literal_byte_quote(self):
        text='The publisher article describes one experimental method for the current sample.'
        child={'original_path':'docs/official.html','passage_candidates':[
           {'source_section':'HTML original publisher document',
            'extracted_text_excerpt_candidate':text,
            'extracted_text_excerpt_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'publisher_original_normalized_HTML_article_anchor':{
                'status':'verified_normalized_original_article_text','container':'article'},
            'independent_original_article_text_match_verified':True}]}
        r=review_verified_linked_originals(self.q,[child])
        proof=r['linked_original_reviews'][0]['passage_structural_reviews'][0]['publisher_location_proof_scope']
        self.assertEqual('normalized_original_HTML_article_location_verified',proof)
        self.assertTrue(r['no_linked_candidate_promoted_to_native_rank_or_answer'])

    def test_linked_source_excerpt_digest_mismatch_refuses_without_a_guess(self):
        bad={'original_path':'docs/source.md','passage_candidates':[
           {'source_section':'Source',
            'current_derived_text_excerpt':'The archive says one example.',
            'derived_excerpt_sha256':'0'*64,
            'publisher_plaintext_byte_interval_independently_SHA_verified':False}]}
        with self.assertRaisesRegex(ValueError,'digest'):
            review_verified_linked_originals(self.q,[bad])

    def test_linked_original_without_verification_marker_never_invents_publisher_proof(self):
        text='Some source evidence from an unverified linked text passage.'
        child={'original_path':'docs/source.md','passage_candidates':[
          {'source_section':'Source','current_derived_text_excerpt':text,
           'derived_excerpt_sha256':hashlib.sha256(text.encode()).hexdigest(),
           'publisher_original_plaintext_byte_interval':{'sha256':'a'*64},
           'publisher_plaintext_byte_interval_independently_SHA_verified':False}]}
        r=review_verified_linked_originals(self.q,[child])
        self.assertEqual('verified_derivative_excerpt_only_no_independent_publisher_location',
            r['linked_original_reviews'][0]['passage_structural_reviews'][0]['publisher_location_proof_scope'])

    def test_bounded_linked_children_and_invalid_child_contents_refused(self):
        with self.assertRaises(ValueError):review_verified_linked_originals(self.q,[{}]*6)
        with self.assertRaisesRegex(ValueError,'invalid independently source-verified'):
            review_verified_linked_originals(self.q,[{'original_path':'docs/source.md','passage_candidates':'invalid'}])
        with self.assertRaisesRegex(ValueError,'unknown verified original'):
            review_verified_linked_originals(self.q,[{'original_path':'docs/source.md','passage_candidates':[{'unrecognized':'unknown'}]}])

    def test_oversized_or_no_query_refused(self):
        with self.assertRaises(ValueError):review_passage_candidate('',passage('Real original text'))
        with self.assertRaises(ValueError):review_passage_candidate(self.q,passage('x'*2001))
        with self.assertRaises(ValueError):review_source_documents(self.q,[{}]*6)

if __name__=='__main__':unittest.main()
