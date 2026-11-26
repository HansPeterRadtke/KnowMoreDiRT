"""PDF table numbers and model row evidence must be physical and read-only."""
from __future__ import annotations
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
from file_system_catalog.physical_pdf_metric_review import (
    parse_physical_pdf_tsv,read_hashed_original_pdf_page_tsv,
    review_original_table_numeric_cells,validate_model_labels,validate_numeric_values,
)

HEAD='level\tpage_num\tpar_num\tblock_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n'

def tsv(rows,page=8):
    content=HEAD+f'1\t{page}\t0\t0\t0\t0\t0\t0\t612\t792\t-1\t###PAGE###\n'
    for i,(word,x,y,w,h) in enumerate(rows):
        content+=f'5\t{page}\t0\t0\t0\t{i}\t{x}\t{y}\t{w}\t{h}\t100\t{word}\n'
    return content.encode()


def fixture():
    return tsv([
        ('MemOS',56,224.87,26,7.4),('77.8',138,224.87,12,7.4),
        ('T-Mem',173.43,224.87,23.53,7.44),('∗',196.96,223.34,2.82,5.13),
        ('[16]',202.27,224.87,13.09,7.44),('80.3',277.22,226.8,14.23,3.7),
        ('AgentMemRec',56.39,241.28,57.42,3.7),('83.2',136.65,241.28,14.23,3.7),
        ('T-Mem,',173.43,254.76,25.28,7.44),
        ('reprod.',200.71,254.76,23.36,7.44),('77.4',278.58,254.76,12.87,7.44),
        ('AgentMemRec',173.43,271.16,57.42,3.7),('77.0',278.58,269.23,12.87,7.44),
        ('Another',440,224.87,33,7.44),('80.3',530,224.87,12,7.44),
    ])


class PhysicalPdfMetricTableSafety(unittest.TestCase):
    def test_realistic_two_parallel_columns_favor_nearest_model_and_expose_ambiguity(self):
        words,receipt=parse_physical_pdf_tsv(fixture(),page=8)
        self.assertEqual(8,receipt['publisher_physical_page'])
        self.assertEqual(612,receipt['page_width_pt'])
        self.assertEqual(15,receipt['word_boxes_parsed'])
        result=review_original_table_numeric_cells(words=words,page=8,
             model_labels=['AgentMemRec','T-Mem','T-Mem, reprod.','MemOS'],
             numeric_values=['80.3','83.2','77.0','77.4'])
        row80=next(x for x in result['numeric_cell_candidates']
                   if x['number']=='80.3' and x['numeric_cell_box']['left_pt']==277.22)
        self.assertEqual('T-Mem',row80['unique_explicit_named_model_in_same_row_if_any'])
        self.assertFalse(row80['source_table_column_heading_and_experiment_not_automatically_verified'] is False)
        self.assertEqual('NOT_ASSESSED',row80['numeric_cell_model_attribution_factual_verdict'])
        row83=next(x for x in result['numeric_cell_candidates'] if x['number']=='83.2')
        self.assertEqual('AgentMemRec',row83['unique_explicit_named_model_in_same_row_if_any'])
        row77=next(x for x in result['numeric_cell_candidates'] if x['number']=='77.0')
        self.assertEqual('AgentMemRec',row77['unique_explicit_named_model_in_same_row_if_any'])
        row774=next(x for x in result['numeric_cell_candidates'] if x['number']=='77.4')
        self.assertEqual('T-Mem, reprod.',row774['unique_explicit_named_model_in_same_row_if_any'])
        self.assertGreater(len(row774['geometrically_near_named_models']),1)
        other80=next(x for x in result['numeric_cell_candidates']
                     if x['number']=='80.3' and x['numeric_cell_box']['left_pt']==530)
        self.assertIsNone(other80['unique_explicit_named_model_in_same_row_if_any'])
        self.assertTrue(result['never_automatic_source_fact_or_corrected_native_summary'])
        # A number in a separate physical table on the same y-row is NOT
        # automatically assigned a model from the other table.
        self.assertIsNone(other80['unique_explicit_named_model_in_same_row_if_any'])
        self.assertGreaterEqual(len(other80['distant_same_row_named_models_not_attribution_proof']),1)

    def test_wide_single_table_row_reports_distant_model_hypothesis_without_false_verdict(self):
        words,_=parse_physical_pdf_tsv(tsv([
            ('LATMiX-LU',101.12,502.56,46.09,7.36),
            ('(Ours)',149.26,502.56,23.21,7.36),
            ('99.49',504.59,502.56,18.43,7.36),
        ],page=32),page=32)
        report=review_original_table_numeric_cells(words=words,page=32,
                model_labels=['LATMiX-LU','LATMiX-QR'],numeric_values=['99.49'])
        match=report['numeric_cell_candidates'][0]
        self.assertIsNone(match['unique_explicit_named_model_in_same_row_if_any'])
        self.assertEqual('LATMiX-LU',match['unique_distant_named_model_if_no_direct_match'])
        self.assertEqual([],match['geometrically_near_named_models'])
        self.assertTrue(match['distant_same_row_named_models_not_attribution_proof'][0][
            'same_horizontal_row_alone_cannot_prove_shared_table_or_metric'])
        self.assertEqual('NOT_ASSESSED',match['numeric_cell_model_attribution_factual_verdict'])

    def test_original_pdf_metric_column_headers_resolve_distinct_benchmark_columns(self):
        source=tsv([
            ('LongMemEval-S',74.19,167.96,58.89,3.7),
            ('LoCoMo',217.16,167.96,30.57,3.7),
            # Deliberate repeated paper intro mention: closer physical table
            # header must win over a similarly placed earlier narrative word.
            ('LoCoMo.',221.41,120.12,37.21,4.16),
            ('AgentMemRec',56.39,241.28,57.42,3.7),('83.2',136.65,241.28,14.23,3.7),
            ('T-Mem',173.43,224.87,23.53,7.44),('80.3',277.22,226.8,14.23,3.7),
            ('AgentMemRec',173.43,271.16,57.42,3.7),('77.0',278.58,269.23,12.87,7.44),
        ])
        words,_=parse_physical_pdf_tsv(source,page=8)
        rows=review_original_table_numeric_cells(
            words=words,page=8,model_labels=['T-Mem','AgentMemRec'],
            numeric_values=['80.3','83.2','77.0'],
            explicit_metric_headings=['LoCoMo','LongMemEval-S'])['numeric_cell_candidates']
        by_value={x['number']:x for x in rows}
        for value,expected in [('80.3','LoCoMo'),('77.0','LoCoMo'),('83.2','LongMemEval-S')]:
            cell=by_value[value]
            self.assertEqual(expected,cell['unique_nearest_explicit_metric_heading_geometric_candidate'])
            self.assertTrue(cell['metric_heading_closeness_does_NOT_prove_metric_or_experiment'])
            self.assertEqual('NOT_ASSESSED',cell['numeric_cell_model_attribution_factual_verdict'])
        found=by_value['80.3']['explicit_metric_heading_geometric_candidates']
        self.assertEqual(1,sum(x['explicit_metric_header']=='LoCoMo' for x in found))

    def test_wide_original_table_has_explicit_metric_heading_but_unproved_column_truth(self):
        words,_=parse_physical_pdf_tsv(tsv([
            ('Recovery',491.07,290.33,31.14,7.32),
            ('Accuracy',449.98,290.33,31.26,7.32),
            ('LATMiX-LU',101.12,502.56,46.09,7.36),
            ('99.49',504.59,502.56,18.43,7.36),
        ],page=32),page=32)
        row=review_original_table_numeric_cells(
            words=words,page=32,model_labels=['LATMiX-LU'],
            numeric_values=['99.49'],explicit_metric_headings=['Recovery','Accuracy'])[
                'numeric_cell_candidates'][0]
        self.assertEqual('LATMiX-LU',row['unique_distant_named_model_if_no_direct_match'])
        self.assertEqual('Recovery',row['unique_nearest_explicit_metric_heading_geometric_candidate'])
        self.assertIsNone(row['unique_explicit_named_model_in_same_row_if_any'])
        self.assertEqual('NOT_ASSESSED',row['numeric_cell_model_attribution_factual_verdict'])

    def test_no_generated_percentage_is_an_explicit_zero_review_not_falsehood(self):
        words,_=parse_physical_pdf_tsv(tsv([
            ('AgentMemRec',101,200,52,6),('No',173,200,16,6),
        ],page=8),page=8)
        report=review_original_table_numeric_cells(
            words=words,page=8,model_labels=['AgentMemRec'],
            numeric_values=[],explicit_metric_headings=[])
        self.assertEqual([],report['numeric_cell_candidates'])
        self.assertTrue(report['no_generated_percentage_or_explicit_numeric_cell_to_review'])
        self.assertTrue(report['never_automatic_source_fact_or_corrected_native_summary'])

    def test_user_model_label_and_metric_bounds_prevent_untrusted_matchers(self):
        for invalid in ([],[''],['***'],['x'*90],['AgentMemRec']*2,
                        ['Agent\nModel'],[' x'],['x '],['x','y']*9):
            with self.subTest(labels=invalid),self.assertRaisesRegex(ValueError,'model labels'):
                validate_model_labels(invalid)
        for invalid in ([],[''],['9999'],['80.3%'],['80.3;DROP'],['80.3','80.3'],
                        [str(i) for i in range(18)]):
            with self.subTest(values=invalid),self.assertRaisesRegex(ValueError,'numeric values'):
                validate_numeric_values(invalid)
        self.assertEqual(['AgentMemRec','T-Mem'],validate_model_labels(['AgentMemRec','T-Mem']))
        self.assertEqual(['80.3','77.0'],validate_numeric_values(['80.3','77.0']))

    def test_tsv_truncation_malformed_word_bounds_and_wrong_page_fail_closed(self):
        for bad in (b'bad\tTSV\n',tsv([('original',600,221,30,8)]),
                    tsv([('valid',100,100,20,8)],page=7),
                    tsv([('valid',100,100,20,8)])+b'1\t8\t0\t0\t0\t0\t0\t0\t612\t792\t-1\t###PAGE###\n',
                    tsv([('valid',100,100,20,8)])[:70]):
            with self.subTest(bad=bad[:40]),self.assertRaises(ValueError):
                parse_physical_pdf_tsv(bad,page=8)
        with self.assertRaises(ValueError):
            parse_physical_pdf_tsv(tsv([('word',100,100,20,8)]),page=8,max_words=0)
        with self.assertRaisesRegex(ValueError,'word count'):
            parse_physical_pdf_tsv(tsv([('word',100,100,20,8)]*2),page=8,max_words=1)

    def test_same_pdf_fd_source_sha_and_page_boundaries_are_verified(self):
        with TemporaryDirectory() as folder:
            p=Path(folder)/'original.pdf'
            p.write_bytes(b'ONLY A PRIVATE MOCK PDF SOURCE, not actual publisher bytes')
            sha=hashlib.sha256(p.read_bytes()).hexdigest()
            simulated_words=tsv([('AgentMemRec',173,271,50,5),('77.0',278,270,13,7)])
            calls=[]
            def fake_poppler(argv,fd,_deadline):
                calls.append((argv,fd))
                self.assertIn('/proc/self/fd/',argv[-1] if argv[0].endswith('pdfinfo') else argv[-2])
                return b'Pages:       8\n' if argv[0].endswith('pdfinfo') else simulated_words
            with patch('file_system_catalog.physical_pdf_metric_review._poppler',side_effect=fake_poppler):
                words,proof=read_hashed_original_pdf_page_tsv(p,source_sha256=sha,page=8)
            self.assertEqual(2,len(calls))
            self.assertEqual(sha,proof['publisher_original_sha256'])
            self.assertEqual(hashlib.sha256(simulated_words).hexdigest(),proof['pdf_tsv_sha256'])
            self.assertEqual(8,proof['original_physical_pdf_pages'])
            self.assertTrue(proof['source_file_same_open_fd_sha256_verified'])
            self.assertEqual(2,len(words))
            p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'SHA differs'):
                read_hashed_original_pdf_page_tsv(p,source_sha256=sha,page=8)
            p.unlink();p.symlink_to(Path(folder)/'different.pdf')
            with self.assertRaises(OSError):
                read_hashed_original_pdf_page_tsv(p,source_sha256=sha,page=8)

    def test_no_model_label_does_not_allow_an_auto_claim_of_falsehood(self):
        words,_=parse_physical_pdf_tsv(tsv([('OtherModel',100,200,40,6),('80.3',220,200,15,6)]),page=8)
        report=review_original_table_numeric_cells(words=words,page=8,
             model_labels=['AgentMemRec'],numeric_values=['80.3'])
        row=report['numeric_cell_candidates'][0]
        self.assertEqual([],row['geometrically_near_named_models'])
        self.assertIsNone(row['unique_explicit_named_model_in_same_row_if_any'])
        self.assertEqual('NOT_ASSESSED',row['numeric_cell_model_attribution_factual_verdict'])


if __name__=='__main__':unittest.main()
