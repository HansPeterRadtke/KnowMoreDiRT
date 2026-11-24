"""Precise original publisher Markdown row/column extraction, no inferred answer."""
from __future__ import annotations
import hashlib
import unittest
from file_system_catalog.source_markdown_tables import verified_original_markdown_table_cell

def sha(raw):return hashlib.sha256(raw).hexdigest()

class ExactOriginalPublisherCellTests(unittest.TestCase):
    def setUp(self):
        self.publisher='''# n8n features
## Availability
| Approach | Use when you need to... | Available on |
| --- | --- | --- |
| Expressions | Simple value references | Cloud and Self-hosted |
| Code node | Custom JavaScript or Python | Cloud and Self-hosted |
| AI Transform node | Generate code from natural language | Cloud only |
'''
        self.raw=self.publisher.encode('utf8')
        self.derived='[Original source metadata]\n'+self.publisher
        self.kw=dict(subject='AI Transform node',column='Available on',
             original_path='official/n8n-features.md',original_sha256=sha(self.raw),
             derived_sha256=sha(self.derived.encode()),
             derived_text=self.derived,publisher_text_bytes=self.raw,filesystem_entry_id=239)

    def cell(self,**changes):
        return verified_original_markdown_table_cell(**(self.kw|changes))

    def test_resolved_exact_source_cell_has_full_original_row_column_and_cell_byte_proof(self):
        r=self.cell();self.assertEqual('resolved',r['status'])
        self.assertEqual(1,r['matching_original_table_rows'])
        v=r['resolved_unique_literal_publisher_cell'];self.assertEqual('Cloud only',v['selected_original_table_column_value'])
        self.assertEqual('Available on',v['selected_original_table_column_name'])
        self.assertEqual('AI Transform node',v['original_table_row_subject'])
        self.assertEqual('n8n features / Availability',v['original_table_section_heading'])
        for prefix in ('original_table_header','original_table_row','original_table_cell'):
            start=v[f'{prefix}_byte_start'];end=v[f'{prefix}_byte_end'];physical=self.raw[start:end]
            self.assertEqual(sha(physical),v[f'{prefix}_sha256'])
            if prefix=='original_table_cell':self.assertEqual(b'Cloud only',physical)
        begin=v['derived_table_row_start_char'];end=v['derived_table_row_end_char']
        self.assertEqual(v['derived_table_row_sha256'],sha(self.derived[begin:end].encode()))
        self.assertTrue(v['column_value_is_original_exact_quoted_publisher_text'])
        self.assertTrue(v['literal_table_cell_does_not_prove_current_product_policy_or_semantic_answer_truth'])

    def test_literal_subject_and_column_are_case_insensitive_but_not_fuzzy(self):
        r=self.cell(subject='ai transform NODE',column='AVAILABLE ON')
        self.assertEqual('Cloud only',r['resolved_unique_literal_publisher_cell']['selected_original_table_column_value'])
        self.assertEqual('not_found',self.cell(subject='AI Transformation node')['status'])
        self.assertEqual('not_found',self.cell(column='Available on currently')['status'])

    def test_mathematical_omega_and_Omega_identifier_are_not_the_same_quoted_original_symbol(self):
        source=(
            '### Symbols\n'
            '| Symbol | Description |\n| --- | --- |\n'
            '| $\\omega_i$ | Direction from $p$ along which incident radiance arrives |\n'
            '| $\\Omega_i$ | Hemisphere of incident directions around surface normal |\n'
            '| $L_i$ | Incident radiance at $p$ along direction omega_i |\n'
            '| $l_i$ | Example lower-case l identifier, not incident radiance |\n'
        )
        raw=source.encode();derived='[[SOURCE]]\n'+source
        for symbol,want in ((r'$\omega_i$','Direction from $p$ along which incident radiance arrives'),
                            (r'$\Omega_i$','Hemisphere of incident directions around surface normal'),
                            ('$L_i$','Incident radiance at $p$ along direction omega_i'),
                            ('$l_i$','Example lower-case l identifier, not incident radiance')):
            with self.subTest(symbol=symbol):
                r=self.cell(subject=symbol,column='Description',publisher_text_bytes=raw,
                    original_sha256=sha(raw),derived_text=derived,derived_sha256=sha(derived.encode()))
                self.assertEqual('resolved',r['status'])
                self.assertEqual(1,r['matching_original_table_rows'])
                self.assertEqual(want,r['resolved_unique_literal_publisher_cell']['selected_original_table_column_value'])
        self.assertEqual('not_found',self.cell(subject='$\\OMEGA_i$',column='Description',
            publisher_text_bytes=raw,original_sha256=sha(raw),derived_text=derived,
            derived_sha256=sha(derived.encode()))['status'])

    def test_inline_code_identifiers_are_case_sensitive_even_when_natural_text_labels_are_not(self):
        source='| API | Default |\n| --- | --- |\n| `dataValue` | true |\n| `DataValue` | false |\n'
        raw=source.encode();derived=source
        for subject,value in [('`dataValue`','true'),('`DataValue`','false')]:
            with self.subTest(subject=subject):
                r=self.cell(subject=subject,column='Default',publisher_text_bytes=raw,
                    original_sha256=sha(raw),derived_text=derived,derived_sha256=sha(raw))
                self.assertEqual('resolved',r['status'])
                self.assertEqual(value,r['resolved_unique_literal_publisher_cell']['selected_original_table_column_value'])

    def test_not_found_never_invents_prose_or_answer(self):
        for change in (dict(subject='Teleportation engine'),dict(column='Very current market prices'),dict(exact_section='Wrong section')):
            with self.subTest(change=change):
                r=self.cell(**change)
                self.assertEqual('not_found',r['status'])
                self.assertEqual(0,r['matching_original_table_rows'])
                self.assertIsNone(r['resolved_unique_literal_publisher_cell'])
                self.assertTrue(r['not_found_does_not_fabricate_a_source_value'])

    def test_same_named_row_in_two_independent_tables_is_ambiguous_without_section(self):
        source='''# MaterialX
## Shader port
| Name | Default |
| --- | --- |
| normal | Nworld |
## Other shader port
| Name | Default |
| --- | --- |
| normal | ObjectSpace |
'''
        raw=source.encode();der='[[SOURCE]]\n'+source
        args=dict(subject='normal',column='Default',publisher_text_bytes=raw,original_sha256=sha(raw),derived_text=der,derived_sha256=sha(der.encode()))
        result=self.cell(**args)
        self.assertEqual('ambiguous',result['status'])
        self.assertEqual(2,result['matching_original_table_rows'])
        self.assertIsNone(result['resolved_unique_literal_publisher_cell'])
        self.assertTrue(result['multiple_matches_never_choose_a_source_value'])
        resolved=self.cell(**args,exact_section='MaterialX / Other shader port')
        self.assertEqual('resolved',resolved['status'])
        self.assertEqual('ObjectSpace',resolved['resolved_unique_literal_publisher_cell']['selected_original_table_column_value'])

    def test_ambiguous_duplicate_original_header_column_names_refused(self):
        src='| Name | Default | Default |\n| --- | --- | --- |\n| normal | Nworld | Oworld |\n'
        raw=src.encode()
        with self.assertRaisesRegex(ValueError,'ambiguous duplicate column'):
            self.cell(subject='normal',column='Default',publisher_text_bytes=raw,original_sha256=sha(raw),derived_text=src,derived_sha256=sha(raw))

    def test_original_and_derivative_sha_mismatch_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'original Markdown publisher SHA'):
            self.cell(original_sha256='a'*64)
        with self.assertRaisesRegex(ValueError,'derivative SHA'):
            self.cell(derived_sha256='b'*64)

    def test_selected_literal_row_missing_or_duplicated_in_derived_text_refused(self):
        row='| AI Transform node | Generate code from natural language | Cloud only |'
        for text in (self.derived.replace('Cloud only','Cloud conditional'),self.derived+'\n'+row+'\n'):
            with self.subTest(text=text[-90:]),self.assertRaisesRegex(ValueError,'not unique in current extracted'):
                self.cell(derived_text=text,derived_sha256=sha(text.encode()))

    def test_fenced_indented_or_HTML_comment_fake_original_tables_ignored(self):
        sources=[
            '```markdown\n| Product | Available on |\n| --- | --- |\n| AI Transform node | Cloud only |\n```\n',
            '    | Product | Available on |\n    | --- | --- |\n    | AI Transform node | Cloud only |\n',
            '<!--\n| Product | Available on |\n| --- | --- |\n| AI Transform node | Cloud only |\n-->\n',
        ]
        for src in sources:
            raw=src.encode()
            with self.subTest(fake=src[:15]):
                r=self.cell(publisher_text_bytes=raw,original_sha256=sha(raw),derived_text=src,derived_sha256=sha(raw))
                self.assertEqual('not_found',r['status'])

    def test_utf8_publisher_original_cell_positions_are_actual_bytes_not_characters(self):
        src='''## Cafés
| Plan | Description |
| --- | --- |
| Café Pro | Contrôle visuel — ✓ |
'''
        raw=src.encode();der=src
        r=self.cell(subject='Café Pro',column='Description',publisher_text_bytes=raw,
                    original_sha256=sha(raw),derived_text=der,derived_sha256=sha(raw))
        self.assertEqual('resolved',r['status'])
        v=r['resolved_unique_literal_publisher_cell'];data=raw[v['original_table_cell_byte_start']:v['original_table_cell_byte_end']]
        self.assertEqual('Contrôle visuel — ✓',data.decode())
        self.assertEqual(sha(data),v['original_table_cell_sha256'])

    def test_physical_CRLf_row_boundaries_and_padding_are_sha_attested(self):
        src='| Plan | Max cases |\r\n| --- | --- |\r\n| Self-hosted Business  |  3  |\r\n'
        raw=src.encode();der=src
        r=self.cell(subject='Self-hosted Business',column='Max cases',publisher_text_bytes=raw,
                    original_sha256=sha(raw),derived_text=der,derived_sha256=sha(raw))
        self.assertEqual('resolved',r['status'])
        v=r['resolved_unique_literal_publisher_cell'];self.assertEqual(b'3',raw[v['original_table_cell_byte_start']:v['original_table_cell_byte_end']])
        self.assertEqual(sha(raw[v['original_table_row_byte_start']:v['original_table_row_byte_end']]),v['original_table_row_sha256'])

    def test_invalid_subject_column_section_source_paths_and_types_rejected(self):
        for arg in (dict(subject='foo\nbar'),dict(subject=''),dict(subject=' subject '),dict(column='Available|Now'),
                    dict(column=''),dict(exact_section='x\nheading'),dict(original_path='../private.md'),
                    dict(original_path='/absolute/private.md'),dict(filesystem_entry_id=True),dict(publisher_text_bytes=b'')):
            with self.subTest(arg=arg),self.assertRaises(ValueError):self.cell(**arg)

if __name__=='__main__':unittest.main()
