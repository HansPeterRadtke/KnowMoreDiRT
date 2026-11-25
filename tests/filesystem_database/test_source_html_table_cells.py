"""Strict original HTML table proofs, never browser or entailment claims."""
from copy import deepcopy
from hashlib import sha256
from unittest import TestCase
from file_system_catalog.source_html_table_cells import (
    verified_original_html_table_cell as lookup,
    compare_claimed_literal_to_original_html_cell as compare,
)

HEADER='<tr><th>Material</th><th>Value</th></tr>'
ROW='<tr><td>Example</td><td>52&nbsp;<b>%</b></td></tr>'
TABLE='<table><thead>'+HEADER+'</thead><tbody>'+ROW+'</tbody></table>'

def args(html='<main>'+TABLE+'</main>'):
    raw=html.encode()
    derived='private synthetic extraction'
    return dict(subject='Example',column='Value',original_path='fixture.html',
        original_sha256=sha256(raw).hexdigest(),derived_sha256=sha256(derived.encode()).hexdigest(),
        derived_text=derived,publisher_html_bytes=raw,filesystem_entry_id=19)

class OriginalHTMLCells(TestCase):
    def test_unique_normalized_text_and_raw_byte_offsets(self):
        a=args('<main>Ä ö ∑\r\n'+TABLE+'</main>')
        r=lookup(**a);self.assertEqual('resolved',r['status'])
        c=r['resolved_unique_literal_publisher_cell']
        self.assertEqual('52 %',c['selected_original_table_column_value'])
        self.assertEqual(sha256(b'52 %').hexdigest(),c['normalized_original_cell_text_sha256'])
        self.assertFalse(c['literal_publisher_text_byte_interval_verified'])
        self.assertFalse(r['browser_visibility_or_rendered_layout_verified'])
        for key,value in c.items():
            if key.endswith('_markup'):
                raw=a['publisher_html_bytes'][value['byte_start']:value['byte_end']]
                self.assertEqual(sha256(raw).hexdigest(),value['raw_markup_sha256'])
        i=c['original_selected_cell_markup']
        self.assertEqual(b'<td>52&nbsp;<b>%</b></td>',a['publisher_html_bytes'][i['byte_start']:i['byte_end']])
        self.assertEqual('literal_match',compare(original_cell_result=r,claimed_literal_value='52 %')['status'])
        d=compare(original_cell_result=r,claimed_literal_value='52%')
        self.assertEqual('literal_different',d['status'])
        self.assertTrue(d['literal_difference_is_NOT_proof_semantic_contradiction_or_false_claim'])

    def test_exact_casefold_selectors_not_fuzzy(self):
        a=args();a.update(subject='example',column='VALUE')
        self.assertEqual('resolved',lookup(**a)['status'])
        a['subject']='Exam';self.assertEqual('not_found',lookup(**a)['status'])

    def test_changed_hash_rejected(self):
        for field in ('original_sha256','derived_sha256'):
            a=args();a[field]='0'*64
            with self.assertRaisesRegex(ValueError,'SHA differs'):lookup(**a)

    def test_ambiguous_equal_values_never_resolved(self):
        for html in ('<main>'+TABLE+TABLE+'</main>',
                     '<main><table>'+HEADER+ROW+ROW+'</table></main>'):
            r=lookup(**args(html))
            self.assertEqual('ambiguous',r['status']);self.assertEqual(2,r['matching_original_table_rows'])
            self.assertIsNone(r['resolved_unique_literal_publisher_cell'])
            self.assertEqual('source_ambiguous_no_literal_comparison',
                compare(original_cell_result=r,claimed_literal_value='52 %')['status'])

    def test_scope_excludes_nav_and_noncontent(self):
        r=lookup(**args('<nav><main>'+TABLE+'</main></nav><main>'+TABLE+'</main>'))
        self.assertEqual('resolved',r['status']);self.assertEqual(1,r['tables_in_declared_scope'])
        r=lookup(**args('<body>'+TABLE+'</body>'))
        self.assertEqual('not_found',r['status'])
        self.assertIsNone(r['resolved_unique_literal_publisher_cell'])

    def test_unsupported_structure_prevents_any_unique_claim(self):
        variants=[
            TABLE.replace('<td>52','<td rowspan="2">52'),
            TABLE.replace('<td>52','<td colspan="2">52'),
            TABLE.replace('<td>52','<td hidden>52'),
            TABLE.replace('<td>52','<td style="display:none">52'),
            TABLE.replace('<td>52','<td aria-hidden="true">52'),
            TABLE.replace('<td>52','<td><script>52'),
            TABLE.replace('<td>52','<td><table>52'),
            TABLE.replace('<th>Value</th>','<th>Material</th>'),
            TABLE.replace('<td>Example</td>',''),
            TABLE.replace('</td>','',1),
            '<div hidden>'+TABLE+'</div>',
            TABLE.replace('</table>',''),
            TABLE.replace('<tbody>','<tbody><tr><th>A</th><th>B</th></tr>'),
        ]
        for other in variants:
            with self.subTest(other=other):
                r=lookup(**args('<main>'+TABLE+other+'</main>'))
                self.assertEqual('unsupported_table_structure',r['status'])
                self.assertFalse(r['all_tables_in_declared_scope_supported'])
                self.assertIsNone(r['resolved_unique_literal_publisher_cell'])
                self.assertEqual('source_unsupported_table_structure_no_literal_comparison',
                    compare(original_cell_result=r,claimed_literal_value='52 %')['status'])

    def test_plain_inline_and_publisher_custom_property(self):
        r=lookup(**args('<article>'+TABLE.replace('<table>','<table style="--columns-count: 2;">')+'</article>'))
        self.assertEqual('resolved',r['status'])

    def test_bad_lookup_parameters(self):
        for key,value in [('subject',' Example'),('column','Value\n'),('original_path','../fixture.html'),
                          ('original_path','fixture.md'),('filesystem_entry_id',True)]:
            a=args();a[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):lookup(**a)

    def test_tampered_normalized_digest_refused(self):
        r=lookup(**args())
        r['resolved_unique_literal_publisher_cell']['selected_original_table_column_value']='invented'
        with self.assertRaisesRegex(ValueError,'digest mismatch'):
            compare(original_cell_result=r,claimed_literal_value='invented')

    def test_tampered_uniqueness_and_empty_comparison_refused(self):
        r=lookup(**args())
        t=deepcopy(r);t['matching_original_table_rows']=2
        with self.assertRaises(ValueError):compare(original_cell_result=t,claimed_literal_value='52 %')
        with self.assertRaises(ValueError):compare(original_cell_result=r,claimed_literal_value='')
        t=deepcopy(r);t['status']='not_found';t['matching_original_table_rows']=0
        with self.assertRaises(ValueError):compare(original_cell_result=t,claimed_literal_value='52 %')
