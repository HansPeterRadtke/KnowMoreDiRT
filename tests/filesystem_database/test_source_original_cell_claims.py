"""Exact literal publisher field claims, distinct from semantic truth."""
from __future__ import annotations
import hashlib
import unittest
from file_system_catalog.source_markdown_tables import (
    verified_original_markdown_table_cell,
    compare_claimed_literal_to_current_original_cell,
)


def sha(x):return hashlib.sha256(x).hexdigest()

class SourceCellClaimLiteralSafetyTests(unittest.TestCase):
    def setUp(self):
        self.text=('## Current archived availability\n'
                   '| Feature | Available on |\n'
                   '| --- | --- |\n'
                   '| AI Transform node | Cloud only |\n'
                   '| Code node | Cloud and Self-hosted |\n')
        raw=self.text.encode();derived='[[SOURCE UNIT: Original plain text]]\n'+self.text
        self.kw=dict(subject='AI Transform node',column='Available on',
            original_path='docs/n8n.md',original_sha256=sha(raw),
            derived_sha256=sha(derived.encode()),derived_text=derived,
            publisher_text_bytes=raw,filesystem_entry_id=7)

    def lookup(self,**kw):return verified_original_markdown_table_cell(**(self.kw|kw))
    def claim(self,proof,assertion):return compare_claimed_literal_to_current_original_cell(
        original_cell_result=proof,claimed_literal_value=assertion)

    def test_exact_unique_literal_supports_only_character_equality(self):
        c=self.claim(self.lookup(),'Cloud only')
        self.assertEqual('literal_match',c['status'])
        self.assertEqual(sha(b'Cloud only'),c['claimed_literal_sha256'])
        self.assertEqual(sha(b'Cloud only'),c['source_cell_sha256'])
        self.assertTrue(c['literal_value_match_only_if_unique'])
        self.assertFalse(c['literal_value_differs_only_if_unique'])
        self.assertTrue(c['literal_match_is_NOT_proof_publisher_information_is_current_beyond_pinned_original'])
        self.assertTrue(c['literal_difference_is_NOT_proof_semantic_contradiction_or_false_claim'])

    def test_different_value_is_only_literal_difference_not_semantic_falsehood(self):
        for value in ('Cloud and Self-hosted','cloud only','Cloud-only','✓'):
            with self.subTest(value=value):
                c=self.claim(self.lookup(),value)
                self.assertEqual('literal_different',c['status'])
                self.assertFalse(c['literal_value_match_only_if_unique'])
                self.assertTrue(c['literal_value_differs_only_if_unique'])
                self.assertTrue(c['literal_difference_is_NOT_proof_semantic_contradiction_or_false_claim'])

    def test_absent_named_source_row_cannot_claim_matching_value(self):
        result=self.claim(self.lookup(subject='Unknown new API'),'Cloud only')
        self.assertEqual('source_not_found_no_literal_comparison',result['status'])
        self.assertIsNone(result['source_cell_sha256'])
        self.assertTrue(result['not_found_is_not_proof_entire_archive_has_no_answer'])

    def test_duplicate_rows_are_ambiguous_no_claim_comparison_even_if_values_identical(self):
        text='''## A\n| Feature | Available on |\n| --- | --- |\n| AI Transform node | Cloud only |\n## B\n| Feature | Available on |\n| --- | --- |\n| AI Transform node | Self-hosted |\n'''
        raw=text.encode();derived='[[Original]]\n'+text
        original=self.lookup(publisher_text_bytes=raw,original_sha256=sha(raw),
            derived_text=derived,derived_sha256=sha(derived.encode()))
        self.assertEqual('ambiguous',original['status'])
        result=self.claim(original,'Cloud only')
        self.assertEqual('source_ambiguous_no_literal_comparison',result['status'])
        self.assertTrue(result['source_ambiguous_requires_more_specific_original_selection'])
        self.assertIsNone(result['source_cell_sha256'])

    def test_exact_section_revalidates_original_byte_proof_before_literal_comparison(self):
        c=self.claim(self.lookup(exact_section='Current archived availability'),'Cloud only')
        self.assertEqual('literal_match',c['status'])
        x=self.lookup(exact_section='Other document')
        self.assertEqual('source_not_found_no_literal_comparison',self.claim(x,'Cloud only')['status'])

    def test_native_selected_original_document_result_literal_match(self):
        row=self.lookup()['resolved_unique_literal_publisher_cell']
        native={'schema':'kmd.optin_native_selected_unique_publisher_markdown_cell.v1',
                'status':'resolved','matching_original_table_rows_across_selected_documents':1,
                'selected_original_source_cell':{'original_native_rank_prior_to_cell_lookup':2,
                                                 'original_path':'docs/n8n.md',
                                                 'original_verified_unique_table_cell':row}}
        self.assertEqual('literal_match',self.claim(native,'Cloud only')['status'])
        self.assertEqual('literal_different',self.claim(native,'On-premises')['status'])

    def test_native_duplicate_sources_are_ambiguous_even_when_top_rank_is_perfect(self):
        for state,count in (('ambiguous',2),('not_found_in_selected_native_originals',0)):
            with self.subTest(state=state):
                native={'schema':'kmd.optin_native_selected_unique_publisher_markdown_cell.v1',
                    'status':state,'matching_original_table_rows_across_selected_documents':count,
                    'selected_original_source_cell':None}
                c=self.claim(native,'Cloud only')
                expected='source_ambiguous_no_literal_comparison' if state=='ambiguous' else 'source_not_found_no_literal_comparison'
                self.assertEqual(expected,c['status'])
                self.assertIsNone(c['source_cell_sha256'])

    def test_tampered_publisher_verified_literal_sha_mismatch_fails_closed(self):
        proof=self.lookup()
        proof['resolved_unique_literal_publisher_cell']['original_table_cell_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'physical byte SHA'):
            self.claim(proof,'Cloud only')
        proof=self.lookup()
        proof['resolved_unique_literal_publisher_cell']['selected_original_table_column_value']='Forged fact'
        with self.assertRaisesRegex(ValueError,'physical byte SHA'):
            self.claim(proof,'Forged fact')

    def test_malformed_or_forged_status_and_source_shape_refused(self):
        original=self.lookup()
        for field,value in (('matching_original_table_rows',2),('matching_original_table_rows',True),
                            ('status','fake_verified'),('resolved_unique_literal_publisher_cell',None)):
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                self.claim(original|{field:value},'Cloud only')
        with self.assertRaises(ValueError):self.claim({'status':'resolved'},'Cloud only')
        with self.assertRaises(ValueError):self.claim([], 'Cloud only')

    def test_expected_literal_is_bounded_and_requires_no_trailing_whitespace(self):
        for value in ('','  ',' Cloud only','Cloud only ','Cloud only\x00',
                      'Cloud\nonly','Cloud\ronly','Cloud\tonly','Cloud\x1bonly',
                      'z'*1201,None,123):
            with self.subTest(value=repr(value)[:30]),self.assertRaises(ValueError):
                self.claim(self.lookup(),value)

    def test_agent_eval_persona_original_multi_column_claims_use_exact_quoted_fields(self):
        source=('## Personas\n| Behavior | **EXPERT** persona | **NOVICE** persona |\n'
                '| --- | --- | --- |\n| **Correct Agent Inaccuracies** | Yes | No |\n'
                '| **Troubleshoot Agent Errors** | Once | Never |\n')
        raw=source.encode();der='[SOURCE]\n'+source
        base=dict(publisher_text_bytes=raw,original_sha256=sha(raw),
                  derived_text=der,derived_sha256=sha(der.encode()))
        expert=self.lookup(subject='**Correct Agent Inaccuracies**',
            column='**EXPERT** persona',**base)
        novice=self.lookup(subject='**Correct Agent Inaccuracies**',
            column='**NOVICE** persona',**base)
        self.assertEqual('literal_match',self.claim(expert,'Yes')['status'])
        self.assertEqual('literal_match',self.claim(novice,'No')['status'])
        mismatch=self.claim(novice,'Yes')
        self.assertEqual('literal_different',mismatch['status'])
        self.assertTrue(mismatch['literal_difference_is_NOT_proof_semantic_contradiction_or_false_claim'])

    def test_untrusted_source_cell_text_is_data_never_a_command(self):
        # Neither lookups nor assertion checks should execute quoted text.
        text='| Error | Solution |\n| --- | --- |\n| Missing GPU | rm -rf /never-run-this-command |\n'
        raw=text.encode()
        source=self.lookup(subject='Missing GPU',column='Solution',
            publisher_text_bytes=raw,original_sha256=sha(raw),
            derived_text=text,derived_sha256=sha(raw))
        self.assertEqual('literal_match',self.claim(source,'rm -rf /never-run-this-command')['status'])
        self.assertTrue(self.claim(source,'Do nothing')['no_extra_source_acquisition_or_native_rank_modification'])

    def test_utf8_original_field_check_matches_literal_unicode_only(self):
        text='| Feature | Supported |\n| --- | --- |\n| QP | ✗ |\n| Routing | ✓ |\n'
        raw=text.encode()
        proof=self.lookup(subject='QP',column='Supported',publisher_text_bytes=raw,
            original_sha256=sha(raw),derived_text=text,derived_sha256=sha(raw))
        self.assertEqual('literal_match',self.claim(proof,'✗')['status'])
        self.assertEqual('literal_different',self.claim(proof,'✓')['status'])

if __name__=='__main__':unittest.main()
