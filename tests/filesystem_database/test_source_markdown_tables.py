"""Original first-party publisher Markdown table column/row/byte integrity."""
import hashlib
import unittest
from file_system_catalog.source_markdown_tables import verified_original_markdown_table_rows


def digest(x):return hashlib.sha256(x).hexdigest()

class OriginalMarkdownTableAnswerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.source=('''# Publisher methods\n\n'''
          '''| Approach | Use when you need to... | Available on |\n'''
          '''| --- | :---: | ---: |\n'''
          '''| Expressions | Small calculations | Cloud and Self-hosted |\n'''
          '''| Code node | Complex JavaScript or Python | Cloud and Self-hosted |\n'''
          '''| AI Transform node | Generate code from natural language | Cloud only |\n'''
          '''| Other data transformation nodes | Visual editing of item data | Cloud and Self-hosted |\n'''
          '''\nThe documentation discusses the difference between these operations.\n''')
        self.raw=self.source.encode('utf8')
        self.derivative='KMD source provenance test\n[[SOURCE UNIT: Original plain text]]\n'+self.source
        self.q='Is the AI Transform node available to self-hosted workflows?'
        self.args={'query':self.q,'original_path':'docs/official-publisher.md',
           'original_sha256':digest(self.raw),'derived_sha256':digest(self.derivative.encode()),
           'derived_text':self.derivative,'publisher_text_bytes':self.raw,
           'filesystem_entry_id':42,'top':2}

    def table(self,**changes):
        return verified_original_markdown_table_rows(**(self.args|changes))

    def test_named_AI_transform_row_reproduced_with_correct_column_and_unchanged_original_SHA(self):
        d=self.table();self.assertEqual(2,d['returned_original_publisher_table_row_candidates'])
        r=d['publisher_rows'][0]
        self.assertEqual('AI Transform node',r['publisher_table_row_cells'][0])
        self.assertEqual('Cloud only',dict(zip(r['publisher_table_columns'],r['publisher_table_row_cells']))['Available on'])
        self.assertEqual(['ai','node','transform'],r['matched_query_terms_in_first_identifier_column'])
        physical=self.raw[r['publisher_row_original_start_byte']:r['publisher_row_original_end_byte']]
        self.assertEqual(digest(physical),r['publisher_row_original_SHA256'])
        self.assertEqual(r['publisher_row_exact_text'],physical.decode().rstrip('\r\n'))
        header=self.raw[r['publisher_header_byte_start']:r['publisher_header_byte_end']]
        self.assertEqual(digest(header),r['publisher_header_SHA256'])
        self.assertEqual(self.derivative[r['current_derived_row_start_char']:r['current_derived_row_end_char']],r['publisher_row_exact_text'])
        self.assertTrue(d['current_publisher_source_byte_proof_not_semantic_answer_correctness'])
        self.assertTrue(d['no_native_rank_or_pdf_physical_page_claim'])

    def test_materialx_symbol_column_and_source_section_beat_unrelated_normal_input_port(self):
        source=(
           '## Physical Material Model\n### Scattering Framework\n### Symbols\n'
           '| Symbol | Description |\n| --- | --- |\n'
           '| $n$ | Unit surface normal vector at $p$ |\n'
           '| $\\theta$ | Elevation angle of a direction from the surface normal |\n'
           '\n### measured_edf\n'
           '| Port | Description | Type | Default |\n| --- | --- | --- | --- |\n'
           '| `normal` | Normal vector of the surface | vector3 | Nworld |\n'
        )
        raw=source.encode('utf8');derived='[[SOURCE UNIT: Original plain text]]\n'+source
        d=self.table(query='Which MaterialX PBR symbol represents the unit surface normal vector at p?',
              original_path='graphics/pbr.md',publisher_text_bytes=raw,original_sha256=digest(raw),
              derived_text=derived,derived_sha256=digest(derived.encode()))
        self.assertEqual('$n$',d['publisher_rows'][0]['publisher_table_row_cells'][0])
        self.assertEqual(['Symbol','Description'],d['publisher_rows'][0]['publisher_table_columns'])
        self.assertTrue(d['publisher_rows'][0]['publisher_table_section_heading'].endswith('Symbols'))
        self.assertEqual(['symbol'],d['publisher_rows'][0]['matched_query_terms_in_table_column_headings'])
        self.assertEqual('Symbol',d['publisher_rows'][-1]['publisher_table_columns'][0])
        self.assertEqual(2,d['returned_original_publisher_table_row_candidates'])

    def test_real_PBR_port_question_may_select_port_but_never_relabel_as_mathematical_symbol(self):
        source=(
           '### Symbols\n| Symbol | Description |\n| --- | --- |\n| $n$ | Unit surface normal vector at $p$ |\n'
           '### measured_edf\n| Port | Description | Default |\n| --- | --- | --- |\n'
           '| `normal` | Normal vector of the surface | Nworld |\n'
        )
        raw=source.encode();derived='[[SOURCE UNIT: Original plain text]]\n'+source
        result=self.table(query='Which measured_edf normal Port has default Nworld?',
               publisher_text_bytes=raw,original_sha256=digest(raw),
               derived_text=derived,derived_sha256=digest(derived.encode()),top=1)
        row=result['publisher_rows'][0]
        self.assertEqual('Port',row['publisher_table_columns'][0])
        self.assertEqual('`normal`',row['publisher_table_row_cells'][0])
        self.assertEqual('Nworld',row['publisher_table_row_cells'][-1])
        self.assertTrue(row['publisher_table_section_heading'].endswith('measured_edf'))

    def test_n8n_exact_selfhosted_business_plan_phrase_outweighs_general_n8n_brand_row(self):
        source=(
            '### Run test cases in parallel\n'
            '| Plan | Maximum parallel test cases |\n'
            '| --- | --- |\n'
            '| Self-hosted Community, n8n Cloud Pro | 1 (sequential) |\n'
            '| Self-hosted Business | 3 |\n'
            '| n8n Cloud Enterprise, self-hosted Enterprise | 5 |\n'
        )
        raw=source.encode('utf8');derived='[[SOURCE UNIT: Original plain text]]\n'+source
        d=self.table(query='For n8n self-hosted Business plan how many maximum parallel test cases can run?',
            publisher_text_bytes=raw,original_sha256=digest(raw),derived_text=derived,
            derived_sha256=digest(derived.encode()),top=2)
        row=d['publisher_rows'][0]
        self.assertEqual('Self-hosted Business',row['publisher_table_row_cells'][0])
        self.assertEqual('3',row['publisher_table_row_cells'][1])
        self.assertEqual(3,row['longest_exact_query_phrase_in_first_identifier_column'])
        self.assertEqual(['Plan','Maximum parallel test cases'],row['publisher_table_columns'])
        self.assertEqual('1 (sequential)',d['publisher_rows'][1]['publisher_table_row_cells'][1])
        self.assertEqual(2,d['publisher_rows'][1]['longest_exact_query_phrase_in_first_identifier_column'])

    def test_query_that_names_Code_node_prefers_Code_row(self):
        d=self.table(query='Can the Code node execute Python on self-hosted workflows?',top=1)
        self.assertEqual('Code node',d['publisher_rows'][0]['publisher_table_row_cells'][0])
        self.assertEqual('Cloud and Self-hosted',d['publisher_rows'][0]['publisher_table_row_cells'][-1])

    def test_wrong_publisher_entire_SHA_is_never_a_source_quote(self):
        with self.assertRaisesRegex(ValueError,'publisher SHA'):
            self.table(original_sha256='f'*64)

    def test_wrong_current_derivative_SHA_refuses_table(self):
        with self.assertRaisesRegex(ValueError,'derivative SHA'):
            self.table(derived_sha256='f'*64)

    def test_row_missing_from_current_derived_text_refuses_instead_of_misattributing(self):
        mutated=self.derivative.replace('Cloud only','Cloud optional')
        with self.assertRaisesRegex(ValueError,'not uniquely present'):
            self.table(derived_text=mutated,derived_sha256=digest(mutated.encode()))

    def test_ambiguous_repeated_original_table_rows_refused_without_choosing_one(self):
        repeated=self.derivative+self.source.splitlines()[6]+'\n'
        with self.assertRaisesRegex(ValueError,'not uniquely present'):
            self.table(derived_text=repeated,derived_sha256=digest(repeated.encode()))

    def test_fenced_and_indented_fake_table_rows_not_original_answers(self):
        for code in (
            '```md\n| Approach | Available on |\n| --- | --- |\n| AI Transform node | On premise |\n```\n',
            '    | Approach | Available on |\n    | --- | --- |\n    | AI Transform node | On premise |\n',
            '<!--\n| Approach | Available on |\n| --- | --- |\n| AI Transform node | On premise |\n-->\n',
        ):
            raw=code.encode()
            with self.subTest(code=code[:14]):
                d=self.table(publisher_text_bytes=raw,original_sha256=digest(raw),
                             derived_text=code,derived_sha256=digest(raw))
                self.assertEqual(0,d['returned_original_publisher_table_row_candidates'])

    def test_escaped_pipes_or_malformed_table_separator_are_not_parsed(self):
        for table in (
            '| Approach | Available on |\n| --- | - |\n| AI Transform node | Cloud only |\n',
            '| Approach | Available on |\n| --- | --- |\n| AI Transform \\| Node | Cloud only |\n',
            '| Approach | Available on |\n| --- | --- |\n| AI Transform node | Cloud only\n',
        ):
            raw=table.encode()
            with self.subTest(table=table[:22]):
                d=self.table(publisher_text_bytes=raw,original_sha256=digest(raw),
                            derived_text=table,derived_sha256=digest(raw))
                self.assertEqual(0,d['returned_original_publisher_table_row_candidates'])

    def test_no_query_terms_or_unrelated_subject_does_not_invent_table(self):
        with self.assertRaises(ValueError):self.table(query='')
        d=self.table(query='quantum submarine teleportation in moonlight')
        self.assertEqual([],d['publisher_rows'])

    def test_original_path_traversal_pdf_and_unsafe_size_refused(self):
        for path in ('../secret.md','/private/source.md','docs/example.pdf','docs/./duplicate.md'):
            with self.subTest(path=path),self.assertRaises(ValueError):self.table(original_path=path)
        with self.assertRaises(ValueError):self.table(top=True)
        with self.assertRaises(ValueError):self.table(publisher_text_bytes=b'',original_sha256=digest(b''))
        with self.assertRaises(ValueError):self.table(publisher_text_bytes=b'x'*(8*1024*1024+1),original_sha256=digest(b'x'*(8*1024*1024+1)))

    def test_output_caps_and_only_original_table_rows_not_an_answer_claim(self):
        d=self.table(top=1)
        self.assertEqual(1,len(d['publisher_rows']))
        self.assertTrue(d['publisher_rows'][0]['source_row_is_candidate_not_proof_of_factual_entailment'])
        self.assertTrue(d['read_only_current_approved_original_and_extracted_text'])

if __name__=='__main__':unittest.main()
