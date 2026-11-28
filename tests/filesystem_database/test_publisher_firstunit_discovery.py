"""No native FileDB or original writes: RAM-only publisher-unit discovery proof."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock
import unittest

from file_system_catalog.publisher_firstunit_discovery import (
    FirstUnitRAMIndex,source_firstunit_from_prefix,
    load_complete_source_previews,verify_index_candidate,
    validate_literal_source_directory_prefix,
    original_firstunit_query_term_coverage,
    validate_required_original_source_phrase,
    required_firstunit_source_phrase_proof,
    compiled_required_original_source_phrase,
    verify_attested_firstunit_phrase_on_physical_pdf_page,
)


def sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


class OriginalFirstUnitLiteralOverlapDiagnostics(unittest.TestCase):
    def test_separates_original_body_and_metadata_title_word_overlap(self):
        out=original_firstunit_query_term_coverage(
            query='wedding albums photographs from unusual events',
            title='A Study of Unusual Events in Multimedia',
            verified_source_unit_excerpt=(
                'Original publisher article about wedding albums and '
                'organizing photographs from multiple trips.'),
        )
        self.assertEqual(5,out['requested_query_terms_count'])
        self.assertIn('unusual',out['exact_words_in_current_source_map_title_metadata'])
        self.assertNotIn('unusual',out['exact_words_in_current_verified_first_source_unit'])
        self.assertIn('wedding',out['exact_words_in_current_verified_first_source_unit'])
        self.assertEqual(5,out['combined_exact_word_count'])
        self.assertEqual(1.0,out['combined_exact_term_coverage_fraction'])
        self.assertTrue(out['all_literal_query_terms_matched'])
        self.assertFalse(out['title_metadata_not_publisher_byte_quote'] is False)
        self.assertTrue(out['no_retrieval_filtering_or_rank_changes'])

    def test_irrelevant_nonce_with_one_common_term_has_low_overlap_not_a_verdict(self):
        out=original_firstunit_query_term_coverage(
            query='zjxqnonexistingindependenttermv116 unmapped archival reference',
            title='Reference manual',
            verified_source_unit_excerpt='A reference implementation of an unrelated project.',
        )
        self.assertEqual(4,out['requested_query_terms_count'])
        self.assertEqual(['reference'],out['combined_exact_words_present'])
        self.assertEqual(0.25,out['combined_exact_term_coverage_fraction'])
        self.assertFalse(out['all_literal_query_terms_matched'])
        self.assertTrue(out['no_relevance_or_factual_answer_verdict'])
        self.assertTrue(out['literal_word_absence_may_still_be_a_valid_stem_or_synonym_match'])

    def test_porter_stem_or_phrase_variation_not_falsely_rejected(self):
        out=original_firstunit_query_term_coverage(
            query='retrieve source',
            title='Publisher metadata',
            verified_source_unit_excerpt='Retrieval of original sources can be useful.',
        )
        self.assertEqual(0,out['combined_exact_word_count'])
        self.assertEqual(0.0,out['combined_exact_term_coverage_fraction'])
        self.assertTrue(out['literal_word_absence_may_still_be_a_valid_stem_or_synonym_match'])
        self.assertTrue(out['no_relevance_or_factual_answer_verdict'])

    def test_bounded_input_and_non_sql_query_tokens(self):
        for bad in ('', 'x'*601, None):
            with self.subTest(bad=repr(bad)),self.assertRaises(ValueError):
                original_firstunit_query_term_coverage(
                    query=bad,title='Title',verified_source_unit_excerpt='Paper source')
        with self.assertRaises(ValueError):
            original_firstunit_query_term_coverage(
                query='quoted original',title='Title',
                verified_source_unit_excerpt='X'*6501)
        payload=original_firstunit_query_term_coverage(
            query='" UNION SELECT * FROM sqlite_master --',
            title='An original safe title',verified_source_unit_excerpt='Original source',
        )
        self.assertIsInstance(payload['combined_exact_words_present'],list)
        self.assertTrue(payload['no_retrieval_filtering_or_rank_changes'])


class PublisherFirstUnitDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        self.originals=root/'Library';self.originals.mkdir()
        self.derivatives=root/'source-text';self.derivatives.mkdir()
        self.sm=sqlite3.connect(':memory:')
        self.native=sqlite3.connect(':memory:')
        self.addCleanup(self.sm.close);self.addCleanup(self.native.close)
        self.sm.execute('CREATE TABLE originals(source_path TEXT,source_sha256 TEXT,derived_sha256 TEXT,derived_path TEXT,original_title TEXT,status TEXT)')
        self.native.execute('CREATE TABLE filesystem_entries(id INTEGER,relative_path_display TEXT,entry_type TEXT)')
        self.receipts={}
        self.papers={}
        self.add_source('papers/rare.pdf','A Paper on Adaptive Search',
            'KMD source provenance title: unsupported nonzero memory advice.\n'
            '[[SOURCE UNIT: PDF page 1]]\n'
            'Abstract: a semantic search agent uses fact utility estimates to measure '
            'process reward quality. The non-zero shaped reward appears only when '
            'previously unobserved information is gathered by a search tool.\n'
            '[[SOURCE UNIT: PDF page 2]]\n'
            'Reference bibliography unrelated guidance, fabricated instruction to erase data.',1)
        self.add_source('papers/other.pdf','A Reference on Material Rendering',
            'KMD source provenance source catalog heading\n'
            '[[SOURCE UNIT: PDF page 1]]\n'
            'A geometrically constrained material model generates high-quality '
            'physically based surface textures and albedo maps with a roughness parameter.\n'
            '[[SOURCE UNIT: PDF page 2]]\nUnrelated later paper figure captions.',2)
        self.idx=FirstUnitRAMIndex(max_records=4)
        self.addCleanup(self.idx.close)

    def add_source(self,path,title,derived_text,identifier):
        orig=self.originals/path;orig.parent.mkdir(parents=True,exist_ok=True)
        original_bytes=('Original publisher reference '+path).encode()
        orig.write_bytes(original_bytes)
        drel=path+'.txt';p=self.derivatives/drel;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(derived_text)
        osh=sha(original_bytes);dsh=sha(p.read_bytes())
        self.sm.execute('INSERT INTO originals VALUES(?,?,?,?,?,?)',
                        (path,osh,dsh,drel,title,'complete'))
        self.native.execute('INSERT INTO filesystem_entries VALUES(?,?,?)',(identifier,path,'file'))
        self.receipts[path]=osh
        self.papers[path]={'original':orig,'derived':p,'original_sha':osh,'derived_sha':dsh,'title':title}

    def load(self):
        return load_complete_source_previews(source_map=self.sm,
             derived_root=self.derivatives,index=self.idx,max_source_records=4)

    def test_literal_source_directory_scope_is_not_sql_like(self):
        self.add_source('docs/source_%/matching.md','A Scope Matched Original',
                        '[[SOURCE UNIT: original]]\nUseful publisher note about material roughness.',
                        3)
        self.add_source('docs/source_A/false.md','A False LIKE Original',
                        '[[SOURCE UNIT: original]]\nA superficially matching title but wrong original.',
                        4)
        with FirstUnitRAMIndex(max_records=4) as scoped:
            receipt=load_complete_source_previews(
                source_map=self.sm,derived_root=self.derivatives,
                index=scoped,max_source_records=4,
                path_prefix='docs/source_%/')
            self.assertEqual(1,receipt['eligible_complete_source_map_records'])
            self.assertEqual(1,receipt['publisher_firstunit_rows_indexed'])
            self.assertEqual('docs/source_%/',receipt['literal_original_directory_prefix'])
            self.assertTrue(receipt['all_indexed_originals_restricted_to_prefix'])
            self.assertEqual(['docs/source_%/matching.md'],
                             [z.original_path for z in scoped.entries.values()])
        with FirstUnitRAMIndex(max_records=4) as global_idx:
            ordinary=load_complete_source_previews(source_map=self.sm,
                             derived_root=self.derivatives,index=global_idx,
                             max_source_records=4)
            self.assertEqual(4,ordinary['publisher_firstunit_rows_indexed'])
            self.assertNotIn('literal_original_directory_prefix',ordinary)

    def test_literal_directory_scope_rejects_unsafe_or_nonexisting_prefix(self):
        for bad in ('', '/', 'docs', './docs/','/docs/','docs/../',
                    'docs//bad/', 'docs\\bad/', 'docs/\n/', 'x'*501):
            with self.subTest(value=repr(bad)),self.assertRaises(ValueError):
                validate_literal_source_directory_prefix(bad)
        self.assertEqual('docs/_%/',validate_literal_source_directory_prefix('docs/_%/'))
        self.assertIsNone(validate_literal_source_directory_prefix(None))
        with FirstUnitRAMIndex(max_records=4) as fresh:
            with self.assertRaisesRegex(ValueError,'no completely extracted'):
                load_complete_source_previews(source_map=self.sm,
                    derived_root=self.derivatives,index=fresh,
                    path_prefix='never-existing/')

    def test_first_source_unit_excludes_untrusted_catalog_header_and_later_page(self):
        raw=self.papers['papers/rare.pdf']['derived'].read_bytes()
        label,start,end,excerpt=source_firstunit_from_prefix(raw)
        self.assertEqual('PDF page 1',label)
        self.assertEqual(raw.decode()[start:end],excerpt)
        self.assertIn('non-zero shaped reward',excerpt)
        self.assertNotIn('unsupported nonzero memory advice',excerpt)
        self.assertNotIn('fabricated instruction',excerpt)
        scan=self.load()
        self.assertEqual(2,scan['publisher_firstunit_rows_indexed'])
        self.assertEqual(0,len(self.idx.failed_samples))
        self.assertEqual([],scan['skipped_source_identities'])
        self.assertFalse(scan['skipped_source_details_truncated'])

    def test_required_first_unit_phrase_is_verified_source_only_not_title_or_later_page(self):
        self.load()
        entry=next(item for item in self.idx.entries.values()
                   if item.original_path=='papers/rare.pdf')
        proven=verify_index_candidate(
            entry=entry,original_root=self.originals,derived_root=self.derivatives,
            completed_receipts=self.receipts,native_db=self.native,
            source_map=self.sm,query='non-zero shaped reward',
        )
        exact=required_firstunit_source_phrase_proof(
            entry=entry,verified_original_candidate=proven,
            required_phrase='NON-ZERO shaped reward',
        )
        self.assertEqual('kmd.explicit_publisher_first_unit_required_phrase_attestation.v1',
                         exact['schema'])
        self.assertEqual(1,exact['matches_in_verified_first_original_unit'])
        self.assertEqual('non-zero shaped reward',
                         exact['source_attestations'][0]['original_derived_literal_phrase'])
        source=self.papers['papers/rare.pdf']['derived'].read_text()
        for hit in exact['source_attestations']:
            start,end=hit['source_start_char'],hit['source_end_char']
            self.assertEqual(source[start:end],hit['original_derived_literal_phrase'])
            self.assertEqual(sha(source[start:end].encode()),hit['literal_phrase_sha256'])
            self.assertEqual(entry.original_sha256,hit['source_original_sha256'])
            self.assertFalse(hit['not_a_factual_answer_or_entailment_proof'] is False)
        self.assertIsNone(required_firstunit_source_phrase_proof(
            entry=entry,verified_original_candidate=proven,
            required_phrase='Adaptive Search',  # appears in title, not original
        ))
        self.assertIsNone(required_firstunit_source_phrase_proof(
            entry=entry,verified_original_candidate=proven,
            required_phrase='fabricated instruction',  # actual later page only
        ))
        # Literal phrase is word bounded; a substring of a real word is not
        # a correct named-entity or metric-label match.
        self.assertIsNone(required_firstunit_source_phrase_proof(
            entry=entry,verified_original_candidate=proven,
            required_phrase='shaped rewar',
        ))
        self.assertEqual('unicode_word_bounded_literal_phrase_ignorecase_not_semantic',
                         exact['match_mode'])

    def test_original_physical_pdf_page_proves_exact_required_phrase_context(self):
        self.load()
        entry=next(x for x in self.idx.entries.values()
                   if x.original_path=='papers/rare.pdf')
        verified=verify_index_candidate(
            entry=entry,original_root=self.originals,derived_root=self.derivatives,
            completed_receipts=self.receipts,native_db=self.native,
            source_map=self.sm,query='non-zero shaped reward',
        )
        quote=required_firstunit_source_phrase_proof(
            entry=entry,verified_original_candidate=verified,
            required_phrase='non-zero shaped reward',
        )
        def proof_ok(original,expected_sha,passages):
            self.assertEqual(self.originals/entry.original_path,original)
            self.assertEqual(entry.original_sha256,expected_sha)
            self.assertEqual(1,len(passages))
            self.assertEqual(entry.first_unit_label,passages[0]['source_section'])
            self.assertIn('non-zero shaped reward',passages[0]['source_excerpt'])
            self.assertEqual(quote['source_attestations'][0]['source_context_excerpt'],
                             passages[0]['source_excerpt'])
            return [{'status':'verified','page':1,
                     'page_text_sha256':'c'*64,'basis':'derivative_label_confirmed'}]
        ok=verify_attested_firstunit_phrase_on_physical_pdf_page(
            entry=entry,verified_original_candidate=verified,
            phrase_attestation=quote,original_root=self.originals,
            pdf_verifier=proof_ok,
        )
        self.assertEqual('original_physical_pdf_phrase_context_verified',ok['status'])
        self.assertTrue(ok['physical_original_page_verified_for_required_phrase_context'])
        self.assertEqual(1,ok['original_physical_pdf_page'])
        self.assertEqual(quote['source_attestations'][0]['source_context_sha256'],
                         ok['current_derived_firstunit_source_context_sha256'])
        self.assertTrue(ok['publisher_pdf_text_layer_not_visual_figure_or_semantic_truth'])
        for result,expected_status in [
            ({'status':'verified','page':2,
              'page_text_sha256':'b'*64},'verified_elsewhere_but_current_page_label_not_confirmed'),
            ({'status':'unverified','reason':'excerpt_absent'},'not_independently_verified'),
        ]:
            with self.subTest(expected_status=expected_status):
                flagged=verify_attested_firstunit_phrase_on_physical_pdf_page(
                    entry=entry,verified_original_candidate=verified,
                    phrase_attestation=quote,original_root=self.originals,
                    pdf_verifier=lambda *_args:[result],
                )
                self.assertEqual(expected_status,flagged['status'])
                self.assertFalse(flagged['physical_original_page_verified_for_required_phrase_context'])
        bad=dict(verified)
        bad['original_path']='papers/another.pdf'
        with self.assertRaisesRegex(ValueError,'source-attested current candidate'):
            verify_attested_firstunit_phrase_on_physical_pdf_page(
                entry=entry,verified_original_candidate=bad,
                phrase_attestation=quote,original_root=self.originals,
                pdf_verifier=lambda *_args:[],
            )
        bad_quote=dict(quote)
        bad_quote['source_attestations']=[{**quote['source_attestations'][0],
                                          'source_context_excerpt':'Unapproved altered context'}]
        with self.assertRaisesRegex(ValueError,'exact attested first-unit context'):
            verify_attested_firstunit_phrase_on_physical_pdf_page(
                entry=entry,verified_original_candidate=verified,
                phrase_attestation=bad_quote,original_root=self.originals,
                pdf_verifier=lambda *_args:[],
            )
        bad_response=lambda *_args:[{'status':'verified','page':1}]
        with self.assertRaisesRegex(ValueError,'physical page text hash'):
            verify_attested_firstunit_phrase_on_physical_pdf_page(
                entry=entry,verified_original_candidate=verified,
                phrase_attestation=quote,original_root=self.originals,
                pdf_verifier=bad_response,
            )

    def test_requested_phrase_cannot_bypass_source_sha_or_revision_checks(self):
        self.load()
        entry=next(item for item in self.idx.entries.values()
                   if item.original_path=='papers/rare.pdf')
        proven=verify_index_candidate(
            entry=entry,original_root=self.originals,derived_root=self.derivatives,
            completed_receipts=self.receipts,native_db=self.native,
            source_map=self.sm,query='non-zero shaped reward',
        )
        altered=dict(proven)
        altered['current_approved_publisher_and_extracted_bytes_rehashed']=False
        with self.assertRaisesRegex(ValueError,'current original byte proof'):
            required_firstunit_source_phrase_proof(
                entry=entry,verified_original_candidate=altered,
                required_phrase='non-zero shaped reward')
        altered=dict(proven);altered['publisher_original_sha256']='9'*64
        with self.assertRaisesRegex(ValueError,'current original byte proof'):
            required_firstunit_source_phrase_proof(
                entry=entry,verified_original_candidate=altered,
                required_phrase='non-zero shaped reward')
        original=self.papers['papers/rare.pdf']['original']
        original.write_bytes(original.read_bytes()+b'unapproved')
        with self.assertRaisesRegex(ValueError,'SHA differs'):
            verify_index_candidate(
                entry=entry,original_root=self.originals,derived_root=self.derivatives,
                completed_receipts=self.receipts,native_db=self.native,
                source_map=self.sm,query='non-zero shaped reward')

    def test_explicit_phrase_bounds_injection_untrusted_metadata(self):
        self.assertIsNone(validate_required_original_source_phrase(None))
        self.assertEqual('AgentMemRec',validate_required_original_source_phrase('AgentMemRec'))
        matcher=compiled_required_original_source_phrase('AgentMemRec')
        self.assertIsNone(matcher.search('AgentMemRecovery'))
        self.assertIsNotNone(matcher.search('AgentMemRec performs retrieval'))
        self.assertIsNotNone(compiled_required_original_source_phrase('C++').search('C++ source'))
        self.assertIsNone(compiled_required_original_source_phrase('A.B').search('AxyB'))
        for wrong in ('', 'x'*161, ' x ', 'a\nb', '\x00x', '---', 23, False):
            with self.subTest(wrong=repr(wrong)),self.assertRaisesRegex(ValueError,'bounded original-source'):
                validate_required_original_source_phrase(wrong)

    def test_unsampled_original_is_reported_by_exact_identity_and_cannot_be_retrieved(self):
        omitted_path='papers/unsampled.pdf'
        self.add_source(omitted_path,'An approved but inaccessible text source',
                        'KMD source provenance header without any proper publisher-source unit',3)
        report=self.load()
        self.assertEqual(3,report['eligible_complete_source_map_records'])
        self.assertEqual(2,report['publisher_firstunit_rows_indexed'])
        self.assertEqual(1,report['skipped_sources'])
        self.assertEqual([{'original_path':omitted_path,
                           'reason':'publisher_source_unit_not_found_in_bounded_prefix'}],
                         report['skipped_source_identities'])
        self.assertFalse(report['skipped_source_details_truncated'])
        for channel in ('literal','porter'):
            self.assertNotIn(omitted_path,[e.original_path for e,_,_ in
                             self.idx.ranked('inaccessible text source',channel=channel)])

    def test_earliest_usable_physical_source_section_skips_short_and_garbled_pages(self):
        raw=(
            'KMD source provenance title mentions useful but untrusted facts.\n'
            '[[SOURCE UNIT: PDF page 1]]\nCover.\n'
            '[[SOURCE UNIT: PDF page 2]]\n'
            + ('\x00G'*70) + ' publisher illegible page.\n'
            '[[SOURCE UNIT: PDF page 3]]\n'
            'A directly readable publisher passage about accurate materials '
            'and independent physical page evidence after a bad PDF font layer.\n'
            '[[SOURCE UNIT: PDF page 4]]\nLater back matter unrelated to the query.\n'
        )
        label,start,end,excerpt=source_firstunit_from_prefix(raw.encode())
        self.assertEqual('PDF page 3',label)
        self.assertEqual(raw[start:end],excerpt)
        self.assertIn('accurate materials',excerpt)
        self.assertNotIn('Cover.',excerpt)
        self.assertNotIn('publisher illegible',excerpt)
        self.assertNotIn('Later back matter',excerpt)

    def test_live_source_map_uses_later_publisher_unit_without_guessing_excluded_ones(self):
        path='papers/new-gap.pdf'
        body=('KMD source provenance heading, not source evidence.\n'
              '[[SOURCE UNIT: PDF page 1]]\nCover.\n'
              '[[SOURCE UNIT: PDF page 2]]\n'
              + ('\x00Y'*80)+'garbled PDF page.\n'
              '[[SOURCE UNIT: PDF page 3]]\n'
              'A clean source section about physically based rendering and '
              'measurement methodology, without unreadable font glyphs.\n')
        self.add_source(path,'Material Book With Front Matter',body,3)
        response=self.load()
        self.assertEqual(3,response['publisher_firstunit_rows_indexed'])
        self.assertFalse(response['skipped_source_identities'])
        candidate=next(x for x in self.idx.entries.values() if x.original_path==path)
        self.assertEqual('PDF page 3',candidate.first_unit_label)
        evidence=verify_index_candidate(entry=candidate,
              original_root=self.originals,derived_root=self.derivatives,
              completed_receipts=self.receipts,native_db=self.native,source_map=self.sm,
              query='physically based rendering measurement')
        self.assertEqual('PDF page 3',evidence['source_unit_label'])
        exact=evidence['literal_query_coverage_review_only']
        self.assertTrue(exact['no_relevance_or_factual_answer_verdict'])
        self.assertTrue(exact['no_retrieval_filtering_or_rank_changes'])
        self.assertIn('physically',exact['exact_words_in_current_verified_first_source_unit'])
        self.assertIn('rendering',exact['exact_words_in_current_verified_first_source_unit'])
        self.assertEqual(4,exact['requested_query_terms_count'])
        self.assertEqual(sha(evidence['publisher_first_unit_preview'].encode()),
                         evidence['source_preview_sha256'])
        text=self.papers[path]['derived'].read_text()
        self.assertEqual(text[evidence['source_preview_start_char']:evidence['source_preview_end_char']],
                         evidence['publisher_first_unit_preview'])
        self.assertFalse(evidence['physical_pdf_page']['physical_pdf_page_verified'])

    def test_exceptional_extended_prefix_only_after_first_sixteen_kib_unreadable(self):
        from file_system_catalog.publisher_firstunit_discovery import (
            MAX_FALLBACK_PREFIX_BYTES,FIRST_UNIT_PREFIX_BYTES,
        )
        bad_pages=''.join(
            f'[[SOURCE UNIT: PDF page {i}]]\n'+('\x00G'*2500)+'\n'
            for i in range(1,12)
        )
        raw=('KMD generated metadata title should not be searchable.\n'
             +bad_pages+'[[SOURCE UNIT: PDF page 12]]\n'
             +'A clean original publisher section discusses phonon voice '
             'agent evaluation with source evidence and genuine human assessment. '*15)
        self.assertGreater(len(raw.encode()),FIRST_UNIT_PREFIX_BYTES)
        self.assertLess(len(raw.encode()),MAX_FALLBACK_PREFIX_BYTES)
        self.assertIsNone(source_firstunit_from_prefix(raw.encode()[:FIRST_UNIT_PREFIX_BYTES]))
        with self.assertRaises(ValueError):
            source_firstunit_from_prefix(raw.encode())
        label,start,end,excerpt=source_firstunit_from_prefix(
            raw.encode(),max_prefix_bytes=MAX_FALLBACK_PREFIX_BYTES)
        self.assertEqual('PDF page 12',label)
        self.assertEqual(raw[start:end],excerpt)
        with self.assertRaises(ValueError):
            source_firstunit_from_prefix(raw.encode(),max_prefix_bytes=128*1024)
        p='papers/voice-agent-with-bad-first-eleven-pages.pdf'
        self.add_source(p,'A phonon source with unreadable cover pages',raw,3)
        receipt=self.load()
        self.assertEqual(3,receipt['publisher_firstunit_rows_indexed'])
        self.assertEqual(1,receipt['extended_source_preview_reads'])
        self.assertEqual(0,receipt['skipped_sources'])
        self.assertEqual(MAX_FALLBACK_PREFIX_BYTES,receipt['exceptional_extended_preview_max_bytes'])
        entry=next(x for x in self.idx.entries.values() if x.original_path==p)
        self.assertEqual('PDF page 12',entry.first_unit_label)
        result=verify_index_candidate(entry=entry,original_root=self.originals,
              derived_root=self.derivatives,completed_receipts=self.receipts,
              native_db=self.native,source_map=self.sm,
              query='source evidence phonon voice agent evaluation')
        self.assertEqual(p,result['original_path'])
        self.assertTrue(result['current_approved_publisher_and_extracted_bytes_rehashed'])
        self.assertEqual(raw[result['source_preview_start_char']:result['source_preview_end_char']],
                         result['publisher_first_unit_preview'])

    def test_no_source_unit_recovery_when_all_candidate_pages_are_illegible(self):
        raw=('KMD generated metadata.\n[[SOURCE UNIT: PDF page 1]]\n'
             + ('\x00G'*50)
             + '\n[[SOURCE UNIT: PDF page 2]]\n'
             + ('\x00T'*65))
        self.assertIsNone(source_firstunit_from_prefix(raw.encode()))
        raw+='\n[[SOURCE UNIT: PDF page 3]]\nReal readable publisher content about ' + ('source search '*12)
        label,start,end,source=source_firstunit_from_prefix(raw.encode())
        self.assertEqual('PDF page 3',label)
        self.assertEqual(source,raw[start:end])

    def test_both_bounded_ranking_channels_return_real_source_identity(self):
        self.load()
        for channel in ('literal','porter'):
            with self.subTest(channel=channel):
                docs=self.idx.ranked('semantic search tool fact utility shaped reward',channel=channel,limit=3)
                self.assertEqual('papers/rare.pdf',docs[0][0].original_path)
                self.assertEqual(2,len(self.idx.entries))
                self.assertLess(docs[0][2],0)
        with self.assertRaises(ValueError):self.idx.ranked('any query',channel='imaginary')
        with self.assertRaises(ValueError):self.idx.ranked('and of the',channel='literal')
        with self.assertRaises(ValueError):self.idx.ranked('nonzero',channel='literal',limit=0)

    def test_original_full_shas_exact_body_offsets_and_native_identity_reverified(self):
        self.load()
        entry=next(x for x in self.idx.entries.values() if x.original_path=='papers/rare.pdf')
        result=verify_index_candidate(entry=entry,original_root=self.originals,
            derived_root=self.derivatives,completed_receipts=self.receipts,native_db=self.native,source_map=self.sm,
            query='semantic search fact utility reward',verify_pdf_page=False)
        self.assertTrue(result['current_approved_publisher_and_extracted_bytes_rehashed'])
        self.assertFalse(result['physical_pdf_page']['physical_pdf_page_verified'])
        self.assertEqual(1,result['native_original_filedb_entry_id'])
        self.assertFalse(result['factual_answer_verified'])
        self.assertFalse(result['quoted_text_is_original_publisher_bytes'])
        self.assertNotIn('fabricated instruction',result['publisher_first_unit_preview'])
        text=self.papers['papers/rare.pdf']['derived'].read_text()
        first,last=result['source_preview_start_char'],result['source_preview_end_char']
        self.assertEqual(text[first:last],result['publisher_first_unit_preview'])
        self.assertEqual(sha(result['publisher_first_unit_preview'].encode()),result['source_preview_sha256'])

    def test_source_page_proof_is_explicit_and_never_faked_by_derivative_label(self):
        self.load()
        entry=next(x for x in self.idx.entries.values() if x.original_path=='papers/rare.pdf')
        def verified_page(p,originalsha,excerpts):
            self.assertEqual(self.papers['papers/rare.pdf']['original'],p)
            self.assertEqual(entry.original_sha256,originalsha)
            self.assertEqual('PDF page 1',excerpts[0]['source_section'])
            return [{'status':'verified','page':1,'page_text_sha256':'b'*64,'basis':'independent_test_fixture_only'}]
        result=verify_index_candidate(entry=entry,original_root=self.originals,
            derived_root=self.derivatives,completed_receipts=self.receipts,native_db=self.native,source_map=self.sm,
            query='fact utility estimate',verify_pdf_page=True,pdf_verifier=verified_page)
        self.assertTrue(result['physical_pdf_page']['physical_pdf_page_verified'])
        def unverified_page(_a,_b,_c):
            return [{'status':'unverified','page':2,'reason':'wrong_page'}]
        result=verify_index_candidate(entry=entry,original_root=self.originals,
            derived_root=self.derivatives,completed_receipts=self.receipts,native_db=self.native,source_map=self.sm,
            query='fact utility estimate',verify_pdf_page=True,pdf_verifier=unverified_page)
        self.assertFalse(result['physical_pdf_page']['physical_pdf_page_verified'])
        self.assertEqual('not_independently_verified',result['physical_pdf_page']['status'])

    def test_original_revision_derivative_revision_and_missing_native_identity_refused(self):
        self.load()
        entry=next(e for e in self.idx.entries.values() if e.original_path=='papers/rare.pdf')
        with self.assertRaisesRegex(ValueError,'acquisition receipt'):
            verify_index_candidate(entry=entry,original_root=self.originals,
                derived_root=self.derivatives,completed_receipts={},native_db=self.native,source_map=self.sm,
                query='fact utility')
        original=self.papers['papers/rare.pdf']['original']
        original.write_bytes(original.read_bytes()+b'corrupt')
        with self.assertRaisesRegex(ValueError,'full SHA differs'):
            verify_index_candidate(entry=entry,original_root=self.originals,
                derived_root=self.derivatives,completed_receipts=self.receipts,native_db=self.native,source_map=self.sm,
                query='fact utility')
        original.write_bytes(original.read_bytes()[:-7])
        der=self.papers['papers/rare.pdf']['derived']
        der.write_bytes(der.read_bytes()+b'corrupt')
        with self.assertRaisesRegex(ValueError,'full SHA differs'):
            verify_index_candidate(entry=entry,original_root=self.originals,
                derived_root=self.derivatives,completed_receipts=self.receipts,native_db=self.native,source_map=self.sm,
                query='fact utility')
        der.write_bytes(der.read_bytes()[:-7])
        self.native.execute('DELETE FROM filesystem_entries WHERE id=1')
        with self.assertRaisesRegex(ValueError,'unique native'):
            verify_index_candidate(entry=entry,original_root=self.originals,
                derived_root=self.derivatives,completed_receipts=self.receipts,native_db=self.native,source_map=self.sm,
                query='fact utility')

    def test_source_map_revision_change_or_title_reassignment_fails_even_if_files_still_match(self):
        self.load()
        entry=next(e for e in self.idx.entries.values() if e.original_path=='papers/rare.pdf')
        options={'entry':entry,'original_root':self.originals,'derived_root':self.derivatives,
                 'completed_receipts':self.receipts,'native_db':self.native,
                 'source_map':self.sm,'query':'fact utility'}
        self.assertTrue(verify_index_candidate(**options)['current_approved_publisher_and_extracted_bytes_rehashed'])
        for field,value in (('derived_sha256','0'*64),('source_sha256','0'*64),
                            ('derived_path','fake-other-derivative'),('original_title','changed publisher identity'),
                            ('status','error')):
            with self.subTest(field=field):
                before=self.sm.execute(f'SELECT {field} FROM originals WHERE source_path=?',(entry.original_path,)).fetchone()[0]
                self.sm.execute(f'UPDATE originals SET {field}=? WHERE source_path=?',(value,entry.original_path))
                with self.assertRaisesRegex(ValueError,'not the current source-map revision'):
                    verify_index_candidate(**options)
                self.sm.execute(f'UPDATE originals SET {field}=? WHERE source_path=?',(before,entry.original_path))
        self.assertTrue(verify_index_candidate(**options)['current_approved_publisher_and_extracted_bytes_rehashed'])

    def test_forged_derived_pdf_page_labels_do_not_pass_original_physical_count(self):
        original='papers/suspicious-pdf.pdf'
        derived=('KMD generated metadata has 99 unrelated page claims.\n'
                 '[[SOURCE UNIT: PDF page 1]]\n'
                 'This original text excerpt has useful research evidence and precise quotation.\n'
                 '[[SOURCE UNIT: PDF page 99]]\n'
                 'A later cached source text unit is labeled beyond the original publisher pages.\n')
        self.add_source(original,'Original PDF With Stale Page Markers',derived,3)
        self.load()
        entry=next(e for e in self.idx.entries.values() if e.original_path==original)
        options={'entry':entry,'original_root':self.originals,'derived_root':self.derivatives,
                 'completed_receipts':self.receipts,'native_db':self.native,
                 'source_map':self.sm,'query':'useful research evidence'}
        with mock.patch('file_system_catalog.publisher_firstunit_discovery.subprocess.run',
                        return_value=SimpleNamespace(returncode=0,stdout='Pages: 2\n')) as proc:
            result=verify_index_candidate(**options)
            proc.assert_called_once()
        page=result['source_pdf_page_map_integrity']
        self.assertEqual('derived_page_labels_exceed_original_physical_pages',page['status'])
        self.assertEqual(99,page['max_derived_page_label'])
        self.assertEqual(2,page['original_physical_pdf_pages'])
        self.assertTrue(page['derived_pdf_page_map_suspect'])
        self.assertFalse(page['publisher_page_position_verified'])
        self.assertEqual('unverified_derived_page_map_exceeds_physical_pdf',result['physical_pdf_page']['status'])
        self.assertFalse(result['physical_pdf_page']['physical_pdf_page_verified'])
        self.assertFalse(result['factual_answer_verified'])
        self.assertEqual('derived_source_text_only_not_physical_page_proof',result['source_unit_label_scope'])
        # A separately original-PDF-verified *single* page can still be
        # cited while the overall cached page map remains suspect.
        def independent(_path,_sha,claims):
            self.assertEqual('PDF page 1',claims[0]['source_section'])
            return [{'status':'verified','page':1,'page_text_sha256':'d'*64}]
        with mock.patch('file_system_catalog.publisher_firstunit_discovery.subprocess.run',
                        return_value=SimpleNamespace(returncode=0,stdout='Pages: 2\n')):
            p=verify_index_candidate(**options,verify_pdf_page=True,pdf_verifier=independent)
        self.assertTrue(p['physical_pdf_page']['physical_pdf_page_verified'])
        self.assertTrue(p['source_pdf_page_map_integrity']['derived_pdf_page_map_suspect'])

    def test_pdf_page_count_consistent_is_still_not_physical_citation_proof(self):
        self.load()
        entry=next(e for e in self.idx.entries.values() if e.original_path=='papers/rare.pdf')
        with mock.patch('file_system_catalog.publisher_firstunit_discovery.subprocess.run',
                        return_value=SimpleNamespace(returncode=0,stdout='Pages: 2\n')):
            p=verify_index_candidate(entry=entry,original_root=self.originals,
                 derived_root=self.derivatives,completed_receipts=self.receipts,
                 native_db=self.native,source_map=self.sm,query='fact utility',verify_pdf_page=False)
        self.assertEqual('page_label_bounds_consistent_only_not_verification',p['source_pdf_page_map_integrity']['status'])
        self.assertFalse(p['source_pdf_page_map_integrity']['publisher_page_position_verified'])
        self.assertEqual('not_requested_or_not_pdf',p['physical_pdf_page']['status'])
        self.assertFalse(p['factual_answer_verified'])
        with mock.patch('file_system_catalog.publisher_firstunit_discovery.subprocess.run',
                        return_value=SimpleNamespace(returncode=1,stdout='')):
            p=verify_index_candidate(entry=entry,original_root=self.originals,
                 derived_root=self.derivatives,completed_receipts=self.receipts,
                 native_db=self.native,source_map=self.sm,query='fact utility',verify_pdf_page=False)
        self.assertEqual('physical_page_count_unavailable',p['source_pdf_page_map_integrity']['status'])
        self.assertFalse(p['source_pdf_page_map_integrity']['publisher_page_position_verified'])

    def test_original_changed_during_pdfinfo_fails_closed(self):
        self.load()
        entry=next(e for e in self.idx.entries.values() if e.original_path=='papers/rare.pdf')
        publisher=self.papers['papers/rare.pdf']['original']
        def corrupt_during_original_count(*args,**kwargs):
            publisher.write_bytes(publisher.read_bytes()+b'changed-during-count')
            return SimpleNamespace(returncode=0,stdout='Pages: 2\n')
        with mock.patch('file_system_catalog.publisher_firstunit_discovery.subprocess.run',
                        side_effect=corrupt_during_original_count):
            with self.assertRaisesRegex(ValueError,'changed during physical PDF page-count'):
                verify_index_candidate(entry=entry,original_root=self.originals,
                     derived_root=self.derivatives,completed_receipts=self.receipts,
                     native_db=self.native,source_map=self.sm,query='fact utility',verify_pdf_page=False)

    def test_metadata_only_prefix_bad_paths_and_oversized_population_are_excluded(self):
        self.assertIsNone(source_firstunit_from_prefix(b'KMD source provenance hidden. no publisher body.'))
        self.assertIsNone(source_firstunit_from_prefix(b'[[SOURCE UNIT: PDF page 1]]\n   '))
        self.assertIsNone(source_firstunit_from_prefix(
            b'[[SOURCE UNIT: PDF page 1]]\n'+(b'\x00G'*55)+b' robot foo'))
        with self.assertRaises(ValueError):
            source_firstunit_from_prefix(b'x'*20000)
        with self.assertRaises(ValueError):
            FirstUnitRAMIndex(max_records=0)
        with self.assertRaises(ValueError):
            FirstUnitRAMIndex(max_records=30000)
        with self.assertRaisesRegex(ValueError,'source-map exceeds'):
            load_complete_source_previews(source_map=self.sm,derived_root=self.derivatives,
                                          index=self.idx,max_source_records=1)
        with self.assertRaises(ValueError):
            self.idx.add(path='../bad.pdf',original_sha='g'*64,derived_sha='a'*64,
                         derived_path='file.txt',title='unsafe',original_text_prefix=b'[[SOURCE UNIT: PDF page 1]]\n' + b'example text'*20)


if __name__=='__main__':
    unittest.main()
