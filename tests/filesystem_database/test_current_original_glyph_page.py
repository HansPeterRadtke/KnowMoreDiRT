"""Approved current original PDF physical glyph recovery negative/positive guards.

Synthetic publisher original, no OCR, network, model or PDF programs. Actual
original physical extractor and Poppler robustness are tested separately.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest import mock
import os
import sys

from file_system_catalog.current_original_glyph_page import (
 current_original_pdf_identity,candidate_current_original_physical_glyph_page,
)


def sha(x:bytes)->str:return sha256(x).hexdigest()


class UnstagedOriginalPhysicalPageTests(unittest.TestCase):
    def setUp(self):
        t=TemporaryDirectory();self.addCleanup(t.cleanup)
        self.root=Path(t.name)
        self.ori=self.root/'originals';self.der=self.root/'extracted'
        self.ori.mkdir();self.der.mkdir()
        self.path='science/paper.pdf'
        self.original_data=b'%PDF-1.7 fake-but-revision-pinned-current-publisher-original'
        self.derived_text='KMD source provenance test\n[[SOURCE UNIT: PDF page 1]]\n'+'Publisher original page text evidence is untrusted. '*12
        self.original=self.ori/self.path
        self.derived=self.der/(self.path+'.txt')
        for p in (self.original,self.derived):
            p.parent.mkdir(parents=True,exist_ok=True)
        self.original.write_bytes(self.original_data)
        self.derived.write_text(self.derived_text)
        self.sm=self.root/'map.sqlite3';self.db=self.root/'native.sqlite3'
        self.catalog=self.root/'catalog.json'
        self.ledger=self.root/'ledger.json'
        self.source_url='https://proceedings.example.test/papers/abcd.html'
        self.raw_url='https://download.example.test/abcd.pdf'
        self.title='A verified archived research original'
        with sqlite3.connect(self.sm) as c:
            c.execute('CREATE TABLE originals(source_path TEXT,source_sha256 TEXT,derived_sha256 TEXT,'
                      'derived_path TEXT,original_url TEXT,original_title TEXT,status TEXT,'
                      'original_bytes INTEGER,original_kind TEXT)')
            c.execute('INSERT INTO originals VALUES(?,?,?,?,?,?,?,?,?)',
                      (self.path,sha(self.original_data),sha(self.derived_text.encode()),
                       self.path+'.txt',self.source_url,self.title,'complete',
                       len(self.original_data),'pdf'))
        with sqlite3.connect(self.db) as c:
            c.execute('CREATE TABLE filesystem_entries(id INTEGER,entry_type TEXT,relative_path_display TEXT)')
            c.execute('INSERT INTO filesystem_entries VALUES(?,?,?)',(99,'file',self.path))
        self.catalog.write_text(json.dumps({'sources':[{
            'path':self.path,'kind':'pdf','title':self.title,
            'source_url':self.source_url,'url':self.raw_url}]}))
        self.ledger.write_text(json.dumps({'items':[{
            'path':self.path,'status':'complete','sha256':sha(self.original_data),
            'bytes_received':len(self.original_data)}]}))
        self.extractor_calls=[]
        self.searcher_calls=[]
        self.report_status='same_page_second_extractor_exact_phrase_corroborated'

    def page_bounds(self,original,text):
        self.assertEqual(self.original,original)
        self.assertEqual(self.derived_text,text)
        return {'status':'derived_page_labels_exceed_original_physical_pages',
                'original_physical_pdf_pages':3,'max_derived_page_label':5,
                'derived_pdf_page_map_suspect':True,'publisher_page_position_verified':False}

    def extractor(self,original,expected_sha,*,pages,expected_physical_page_count):
        self.extractor_calls.append((original,expected_sha,pages,expected_physical_page_count))
        self.assertEqual(self.original,original)
        self.assertEqual(sha(self.original_data),expected_sha)
        self.assertEqual([2],pages)
        self.assertEqual(3,expected_physical_page_count)
        return {'schema':'kmd.readonly_original_pdf_bounded_machine_text_candidates.v1',
                'original_pdf_sha256':expected_sha,
                'original_pdf_sha_verified_before_and_after_same_file_description':True,
                'all_page_numbers_from_original_pdfinfo_not_fake_derivative_labels':True,
                'machine_extracted_text_is_provisional_not_visually_verified':True,
                'pages':[{'physical_pdf_page':2,'status':'source_bounded_provisional_machine_text',
                          'text':'This is a machine text candidate from the source.',
                          'display_text_sha256':sha(b'This is a machine text candidate from the source.'),
                          'raw_machine_text_sha256':'a'*64}]}

    def searcher(self,query,pages,*,top,original_pdf,expected_original_sha256):
        self.searcher_calls.append((query,top))
        self.assertEqual(self.original,original_pdf)
        self.assertEqual(sha(self.original_data),expected_original_sha256)
        self.assertEqual(2,pages[0]['physical_pdf_page'])
        self.assertEqual(1,top)
        return {'schema':'kmd.ephemeral_provisional_alternate_pdf_physical_page_lexical_search.v1',
                'accepted_provisional_ranked_pages':[{
                    'physical_pdf_page':2,'status':self.report_status,
                    'independent_poppler_same_physical_page_passage_corroborated':(
                      self.report_status=='same_page_second_extractor_exact_phrase_corroborated'),
                    'corroborated_short_fragment_only':None,
                    'provisional_original_page_excerpt':'This is a machine text candidate from the source.',
                    'provisional_excerpt_sha256':sha(b'This is a machine text candidate from the source.'),
                    'original_pdf_page_text_cross_extractor_sha256':(
                        'b'*64 if self.report_status=='same_page_second_extractor_exact_phrase_corroborated'
                        else None),
                    'factual_answer_verified':False,
                    'publisher_original_visual_layout_correctness_verified':False,
                    'provisional_original_page_excerpt':'This is a machine text candidate from the source.'}],
                'no_factual_answers_asserted':True}

    def args(self):
        return {'original_root':self.ori,'derived_root':self.der,
                'source_map':self.sm,'native_db':self.db,
                'catalog':self.catalog,'completed_ledger':self.ledger}

    def evaluate_case(self,**changes):
        kw={**self.args(),'original_path':self.path,'page':2,
            'query':'machine source candidate question',
            'bounds_reader':self.page_bounds,
            'alternate_reader':self.extractor,
            'ranked_reader':self.searcher}
        kw.update(changes)
        return candidate_current_original_physical_glyph_page(**kw)

    def test_current_approved_original_distinguishes_publisher_from_download_url(self):
        x=current_original_pdf_identity(original_path=self.path,
                                        bounds_reader=self.page_bounds,**self.args())
        self.assertEqual(self.source_url,x['source_publisher_url'])
        self.assertEqual(self.raw_url,x['raw_pdf_acquisition_url_separate_from_publication'])
        self.assertEqual(99,x['current_native_filedb_original_id'])
        self.assertTrue(x['cached_page_labels_exceed_actual_pdf_physical_pages'])
        self.assertTrue(x['current_complete_original_and_derived_full_SHA_verified'])

    def test_successful_corrobation_is_boolean_json_and_source_revision_checked_twice(self):
        out=self.evaluate_case()
        self.assertTrue(out['exact_current_original_publisher_and_derived_SHA_and_native_ID_rechecked_twice'])
        self.assertTrue(out['no_new_source_acquisition_OCR_or_external_network_model'])
        self.assertTrue(out['no_original_or_extracted_source_file_modified'])
        self.assertEqual(2,out['physical_page_selected'])
        self.assertNotIn('text',out['source_page_machine_text_quality_metadata_only'])
        self.assertEqual(1,len(self.extractor_calls))
        self.assertEqual(1,len(self.searcher_calls))
        hit=out['ranked_same_page_source_candidates']['accepted_provisional_ranked_pages'][0]
        self.assertIs(hit['independent_poppler_same_physical_page_passage_corroborated'],True)
        self.assertFalse(hit['factual_answer_verified'])
        cloned=json.loads(json.dumps(out))
        self.assertIs(cloned['ranked_same_page_source_candidates']['accepted_provisional_ranked_pages'][0][
            'independent_poppler_same_physical_page_passage_corroborated'],True)
        self.assertEqual(sha(self.original_data),sha(self.original.read_bytes()))

    def test_no_source_stage_required_for_current_approved_original(self):
        self.report_status='provisional_alternate_machine_text_only_not_a_verified_quote'
        found=self.evaluate_case()
        hit=found['ranked_same_page_source_candidates']['accepted_provisional_ranked_pages'][0]
        self.assertFalse(hit['independent_poppler_same_physical_page_passage_corroborated'])
        self.assertTrue(found['full_corresponding_ghostscript_page_not_source_verified_by_one_match'])
        self.assertFalse(found['not_an_independent_physical_figure_visual_or_factual_truth_verdict'] is False)

    def test_unknown_duplicate_or_stale_original_not_reported_as_valid(self):
        for rel in ('../private.pdf','/etc/private.pdf','science/../pdf.pdf','science\x00/x.pdf','science/text.txt'):
            with self.subTest(path=rel),self.assertRaises(ValueError):
                self.evaluate_case(original_path=rel)
        with sqlite3.connect(self.db) as c:
            c.execute('INSERT INTO filesystem_entries VALUES (?,?,?)',(100,'file',self.path))
        with self.assertRaisesRegex(ValueError,'ambiguous original source identity'):
            self.evaluate_case()
        self.assertEqual([],self.extractor_calls)

    def test_missing_completed_acquisition_or_mismatched_publication_url_refuses(self):
        self.ledger.write_text(json.dumps({'items':[{
            'path':self.path,'status':'pending','sha256':sha(self.original_data)}]}))
        with self.assertRaisesRegex(ValueError,'not a completed'):
            self.evaluate_case()
        self.assertEqual([],self.extractor_calls)
        self.ledger.write_text(json.dumps({'items':[{
            'path':self.path,'status':'complete','sha256':sha(self.original_data),
            'bytes_received':len(self.original_data)}]}))
        rows=json.loads(self.catalog.read_bytes())
        rows['sources'][0]['source_url']='https://other.example.test/not-this-paper'
        self.catalog.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError,'metadata disagrees'):
            self.evaluate_case()
        self.assertEqual([],self.extractor_calls)

    def test_changed_original_and_derived_are_refused_before_pdf_extraction(self):
        self.original.write_bytes(b'updated publisher bytes not approved')
        with self.assertRaises(ValueError):self.evaluate_case()
        self.assertEqual([],self.extractor_calls)
        self.original.write_bytes(self.original_data)
        self.derived.write_bytes(b'mutated extracted source')
        with self.assertRaisesRegex(ValueError,'derived source bytes differ'):
            self.evaluate_case()
        self.assertEqual([],self.extractor_calls)

    def test_original_revisions_are_rechecked_after_machine_page_extraction(self):
        def mutate(original,sha256,*,pages,expected_physical_page_count):
            result=self.extractor(original,sha256,pages=pages,
                                  expected_physical_page_count=expected_physical_page_count)
            self.derived.write_bytes(self.derived.read_bytes()+b'untrusted late mutation')
            return result
        with self.assertRaises(ValueError):
            self.evaluate_case(alternate_reader=mutate)
        self.assertEqual(1,len(self.extractor_calls))

    def test_native_id_source_map_change_during_query_refused(self):
        def rewrite(*args,**kw):
            result=self.searcher(*args,**kw)
            with sqlite3.connect(self.db) as c:
                c.execute('UPDATE filesystem_entries SET id=? WHERE relative_path_display=?',
                          (101,self.path))
            return result
        with self.assertRaisesRegex(ValueError,'approved original source/native identity changed'):
            self.evaluate_case(ranked_reader=rewrite)

    def test_bounded_original_page_and_queries_fail_before_extraction(self):
        for value in (0,-1,4,451,True):
            with self.subTest(page=value),self.assertRaises(ValueError):
                self.evaluate_case(page=value)
        for query in ('','x','query\nunsafe','Q'*401):
            with self.subTest(query=query[:12]),self.assertRaises(ValueError):
                self.evaluate_case(query=query)
        self.assertEqual([],self.extractor_calls)

    def test_attempt_to_launder_source_truth_or_opaque_bools_refused(self):
        def invalid(*args,**kw):
            d=self.searcher(*args,**kw)
            d['accepted_provisional_ranked_pages'][0]['factual_answer_verified']=True
            return d
        with self.assertRaisesRegex(ValueError,'unverified original'):
            self.evaluate_case(ranked_reader=invalid)
        def wrong_page(*args,**kw):
            d=self.searcher(*args,**kw)
            d['accepted_provisional_ranked_pages'][0]['physical_pdf_page']=3
            return d
        with self.assertRaises(ValueError):
            self.evaluate_case(ranked_reader=wrong_page)

    def test_short_original_page_fragment_must_match_same_source_page_and_hash(self):
        self.report_status='provisional_alternate_machine_text_only_not_a_verified_quote'
        excerpt='machine text candidate from the source'
        def source_fragment(*args,**kw):
            response=self.searcher(*args,**kw)
            hit=response['accepted_provisional_ranked_pages'][0]
            hit['corroborated_short_fragment_only']={
                'text':excerpt,
                'fragment_sha256':sha(excerpt.encode()),
                'original_pdf_poppler_physical_page_sha256':'d'*64,
                'physical_pdf_page':2,
                'matched_query_terms_count':2,
                'basis':'two_independent_machine_extractors_exact_same_original_physical_page_phrase',
                'full_machine_page_and_visual_layout_not_verified':True}
            return response
        response=self.evaluate_case(ranked_reader=source_fragment)
        hit=response['ranked_same_page_source_candidates']['accepted_provisional_ranked_pages'][0]
        self.assertIs(hit['independent_poppler_same_physical_page_passage_corroborated'],False)
        self.assertEqual(excerpt,hit['corroborated_short_fragment_only']['text'])
        self.assertTrue(response['full_corresponding_ghostscript_page_not_source_verified_by_one_match'])
        self.assertFalse(hit['factual_answer_verified'])
        json.dumps(response)
        for field,value in (
             ('fragment_sha256','e'*64),('physical_pdf_page',3),
             ('original_pdf_poppler_physical_page_sha256','badsha'),
             ('basis','model_guessed'),('full_machine_page_and_visual_layout_not_verified',False)):
            with self.subTest(field=field):
                def forged(*args,**kw):
                    data=source_fragment(*args,**kw)
                    data['accepted_provisional_ranked_pages'][0]['corroborated_short_fragment_only'][field]=value
                    return data
                with self.assertRaisesRegex(ValueError,'short publisher-original physical-page fragment'):
                    self.evaluate_case(ranked_reader=forged)

    def test_mismatched_status_boolean_is_rejected_without_promotion(self):
        def false_claim(*args,**kw):
            data=self.searcher(*args,**kw)
            hit=data['accepted_provisional_ranked_pages'][0]
            hit['status']='provisional_alternate_machine_text_only_not_a_verified_quote'
            # Inconsistent full-source confirmation must never survive.
            return data
        with self.assertRaisesRegex(ValueError,'impossible or unverified'):
            self.evaluate_case(ranked_reader=false_claim)
        def changed_excerpt(*args,**kw):
            data=self.searcher(*args,**kw)
            data['accepted_provisional_ranked_pages'][0]['provisional_excerpt_sha256']='f'*64
            return data
        with self.assertRaisesRegex(ValueError,'impossible or unverified'):
            self.evaluate_case(ranked_reader=changed_excerpt)

    def test_unavailable_original_machine_page_cannot_be_laundered_into_a_ranked_citation(self):
        def unreadable(original,sha,*,pages,expected_physical_page_count):
            response=self.extractor(original,sha,pages=pages,
                                  expected_physical_page_count=expected_physical_page_count)
            response['pages'][0]={'physical_pdf_page':2,'status':'unavailable',
                                  'reason':'no genuine current publisher text'}
            return response
        with self.assertRaisesRegex(ValueError,'unavailable original physical PDF page'):
            self.evaluate_case(alternate_reader=unreadable)

    def test_machine_source_excerpt_and_digest_must_match_current_provisional_text(self):
        def changed_machine(original,sha,*,pages,expected_physical_page_count):
            data=self.extractor(original,sha,pages=pages,
                                expected_physical_page_count=expected_physical_page_count)
            data['pages'][0]['display_text_sha256']='a'*64
            return data
        with self.assertRaisesRegex(ValueError,'alternate page text changed'):
            self.evaluate_case(alternate_reader=changed_machine)
        def corrupted_glyph(original,sha,*,pages,expected_physical_page_count):
            data=self.extractor(original,sha,pages=pages,
                                expected_physical_page_count=expected_physical_page_count)
            text=data['pages'][0]['text']+'\x00'
            data['pages'][0]['text']=text
            data['pages'][0]['display_text_sha256']=sha256(text.encode()).hexdigest()
            return data
        with self.assertRaisesRegex(ValueError,'unreadable glyphs'):
            self.evaluate_case(alternate_reader=corrupted_glyph)

    def test_operator_cli_private_output_is_created_only_in_safe_audit_folder(self):
        scripts=str(Path(__file__).resolve().parents[2]/'scripts')
        sys.path.insert(0,scripts)
        try:
            import raspi_current_original_pdf_glyph_search as cli
            report=self.evaluate_case()
            audit=self.root/'search-audit'
            audit.mkdir(mode=0o700)
            target=audit/'bounded_private_original_page.json'
            args=['raspi_current_original_pdf_glyph_search.py',
                  'machine source candidate question',
                  '--original-path',self.path,
                  '--physical-page','2',
                  '--output',str(target),
                  '--original-root',str(self.ori),
                  '--derived-root',str(self.der),
                  '--source-map',str(self.sm),
                  '--native-db',str(self.db),
                  '--approved-catalog',str(self.catalog),
                  '--completed-ledger',str(self.ledger)]
            with mock.patch.object(sys,'argv',args),mock.patch.object(
                    cli,'candidate_current_original_physical_glyph_page',return_value=report) as invoked:
                cli.main()
            self.assertEqual(1,invoked.call_count)
            loaded=json.loads(target.read_bytes())
            self.assertEqual(report,loaded)
            self.assertEqual(0o600,target.stat().st_mode & 0o777)
            self.assertTrue(loaded['no_original_or_extracted_source_file_modified'])
            with mock.patch.object(sys,'argv',args),mock.patch.object(
                    cli,'candidate_current_original_physical_glyph_page') as retry:
                with self.assertRaisesRegex(ValueError,'overwrite'):
                    cli.main()
                retry.assert_not_called()
            self.assertEqual(sha(self.original_data),sha(self.original.read_bytes()))
            # Exfiltration into an arbitrary path outside this same
            # FileDB's owner-private report tree is never permitted, even
            # if that path is writable and source bytes are approved.
            with TemporaryDirectory() as destination:
                foreign=Path(destination)/'unsafe_output_outside_audit.json'
                alternate_args=list(args)
                alternate_args[alternate_args.index(str(target))]=str(foreign)
                with mock.patch.object(sys,'argv',alternate_args),mock.patch.object(
                        cli,'candidate_current_original_physical_glyph_page') as attempted:
                    with self.assertRaisesRegex(ValueError,'owner-private search-audit'):
                        cli.main()
                    attempted.assert_not_called()
                self.assertFalse(foreign.exists())
        finally:
            sys.path.remove(scripts)

    def test_original_and_derived_symlink_paths_never_become_approved_current_source(self):
        alt=self.root/'alternate-source.pdf'
        alt.write_bytes(self.original_data)
        self.original.unlink()
        self.original.symlink_to(alt)
        with self.assertRaisesRegex(ValueError,'symlinked'):
            self.evaluate_case()
        self.assertEqual([],self.extractor_calls)
        self.original.unlink()
        self.original.write_bytes(self.original_data)
        target=self.root/'outside-extracted.txt'
        target.write_text(self.derived_text)
        self.derived.unlink()
        self.derived.symlink_to(target)
        with self.assertRaisesRegex(ValueError,'symlinked'):
            self.evaluate_case()
        self.assertEqual([],self.extractor_calls)

    def test_duplicate_approved_catalog_path_and_unavailable_original_page_count_fail(self):
        doc=json.loads(self.catalog.read_text())
        doc['sources'].append(doc['sources'][0].copy())
        self.catalog.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError,'ambiguous approved original'):
            self.evaluate_case()
        self.assertEqual([],self.extractor_calls)
        self.catalog.write_text(json.dumps({'sources':doc['sources'][:1]}))
        def missing(original,text):
            return {'status':'physical_page_count_unavailable',
                    'publisher_page_position_verified':False}
        with self.assertRaisesRegex(ValueError,'physical page count unavailable'):
            self.evaluate_case(bounds_reader=missing)
        self.assertEqual([],self.extractor_calls)

    def test_suspect_independent_cross_reader_verification_object_never_leaks_into_output(self):
        import re
        def leaked_match_object(*args,**kwargs):
            old=self.searcher(*args,**kwargs)
            hit=old['accepted_provisional_ranked_pages'][0]
            hit['independent_poppler_same_physical_page_passage_corroborated']=re.fullmatch(
                '[a-f0-9]{64}','a'*64)
            return old
        with self.assertRaisesRegex(ValueError,'impossible or unverified original'):
            self.evaluate_case(ranked_reader=leaked_match_object)

    def test_original_machine_page_proof_cannot_be_forged_to_another_physical_page(self):
        def wrong_page(original,sha,*,pages,expected_physical_page_count):
            x=self.extractor(original,sha,pages=pages,
                             expected_physical_page_count=expected_physical_page_count)
            x['pages'][0]['physical_pdf_page']=3
            return x
        with self.assertRaisesRegex(ValueError,'physical-page machine extraction lacked current source proof'):
            self.evaluate_case(alternate_reader=wrong_page)
        def wrong_original(original,sha,*,pages,expected_physical_page_count):
            x=self.extractor(original,sha,pages=pages,
                             expected_physical_page_count=expected_physical_page_count)
            x['original_pdf_sha256']='a'*64
            return x
        with self.assertRaisesRegex(ValueError,'physical-page machine extraction lacked current source proof'):
            self.evaluate_case(alternate_reader=wrong_original)

    def test_unavailable_machine_page_fails_honestly_with_no_claimed_quote(self):
        def unreadable(original,sha,*,pages,expected_physical_page_count):
            d=self.extractor(original,sha,pages=pages,
                             expected_physical_page_count=expected_physical_page_count)
            d['pages'][0]={'physical_pdf_page':2,'status':'unavailable',
                           'reason':'no machine-readable source text'}
            return d
        def empty_rank(*args,**kwargs):
            return {'schema':'kmd.ephemeral_provisional_alternate_pdf_physical_page_lexical_search.v1',
                    'accepted_provisional_ranked_pages':[],
                    'no_factual_answers_asserted':True}
        got=self.evaluate_case(alternate_reader=unreadable,ranked_reader=empty_rank)
        self.assertEqual('unavailable',got['source_page_machine_text_quality_metadata_only']['status'])
        self.assertEqual([],got['ranked_same_page_source_candidates']['accepted_provisional_ranked_pages'])


if __name__=='__main__':
    unittest.main()
