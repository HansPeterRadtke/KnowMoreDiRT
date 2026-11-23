"""No-write, original physical PDF page verified staged text retrieval."""
from __future__ import annotations
import hashlib
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from file_system_catalog.staged_physical_page_search import (
    StagedPhysicalPageRAMIndex,source_units_from_corrected_text,verify_staged_page_result,
    attach_staged_physical_page_neighbors,
)


def digest(data:bytes):return hashlib.sha256(data).hexdigest()


class StagedPhysicalPageSearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary=TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.orig=self.root/'original';self.live=self.root/'live';self.stage=self.root/'private'
        for p in (self.orig,self.live,self.stage):p.mkdir()
        self.relative='ai-agents/known-source.pdf'
        self.original=b'%PDF-private-original-fixture'
        self.old=b'old cached inaccurate page labels'
        self.source=("KMD source provenance original fixture\n{}\n\n"
             "[[SOURCE UNIT: PDF page 1]]\nSpeech synthesis requires punctuation boundaries and model evaluation.\n"
             "[[SOURCE UNIT: PDF page 2]]\nVector retrieval must cite original publisher physical pages and preserve page provenance.\n")
        self.staged=self.source.encode()
        for base,content,suffix in ((self.orig,self.original,''),(self.live,self.old,'.txt'),(self.stage,self.staged,'.txt')):
            path=base/(self.relative+suffix);path.parent.mkdir();path.write_bytes(content)
        self.item={'original_path':self.relative,'original_sha256':digest(self.original),
                   'old_derived_sha256':digest(self.old),'new_staged_derived_sha256':digest(self.staged),
                   'staged_private_file':str(self.stage/(self.relative+'.txt')),
                   'native_original_filedb_id':33,'original_pdf_physical_pages':2,'new_max_staged_page_label':2}
        self.sm=sqlite3.connect(':memory:');self.addCleanup(self.sm.close)
        self.sm.execute('CREATE TABLE originals (source_sha256 TEXT,derived_sha256 TEXT,derived_path TEXT,status TEXT,source_path TEXT)')
        self.sm.execute('INSERT INTO originals VALUES (?,?,?,?,?)',(digest(self.original),digest(self.old),self.relative+'.txt','complete',self.relative))
        self.native=sqlite3.connect(':memory:');self.addCleanup(self.native.close)
        self.native.execute('CREATE TABLE filesystem_entries (id INTEGER,entry_type TEXT,relative_path_display TEXT)')
        self.native.execute('INSERT INTO filesystem_entries VALUES (?,?,?)',(33,'file',self.relative))
        self.sm.commit();self.native.commit()

    def test_both_page_units_without_catalog_header(self):
        units=list(source_units_from_corrected_text(self.source,2,2))
        self.assertEqual([1,2],[row[1] for row in units])
        self.assertNotIn('provenance original fixture',units[0][2])
        self.assertIn('physical pages',units[1][2])

    def test_out_of_bounds_or_repeated_pages_rejected(self):
        with self.assertRaisesRegex(ValueError,'out-of-bounds'):
            list(source_units_from_corrected_text(self.source,1,2))
        broken=self.source.replace('PDF page 2','PDF page 1')
        with self.assertRaisesRegex(ValueError,'repeated'):
            list(source_units_from_corrected_text(broken,2,1))

    def test_named_original_restricts_ranked_physical_pages_without_rewriting_global_order(self):
        with StagedPhysicalPageRAMIndex(2) as index:
            index.add(self.item,stage_root=self.stage)
            other=dict(self.item)
            other['original_path']='ai-agents/other.pdf'
            other_source=('KMD source provenance another original\n'
                 '[[SOURCE UNIT: PDF page 1]]\nPublisher physical page provenance and vectors.\n')
            p=self.stage/(other['original_path']+'.txt');p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(other_source)
            other.update(staged_private_file=str(p),
                         new_staged_derived_sha256=digest(other_source.encode()),
                         original_pdf_physical_pages=1,new_max_staged_page_label=1)
            index.add(other,stage_root=self.stage);index.commit()
            global_hits=index.ranked('publisher physical page provenance',limit=10)
            scoped=index.ranked('publisher physical page provenance',limit=10,
                                original_path=self.relative)
            self.assertTrue(global_hits)
            self.assertTrue(scoped)
            self.assertTrue(all(row[0][0]==self.relative for row in scoped))
            self.assertEqual(2,len({row[0][0] for row in global_hits}))
            with self.assertRaisesRegex(ValueError,'not present'):
                index.ranked('publisher physical page',original_path='not-in-index.pdf')

    def test_exact_verified_publisher_page_result(self):
        with StagedPhysicalPageRAMIndex(1) as index:
            index.add(self.item,stage_root=self.stage);index.commit()
            ranked=index.ranked('publisher physical page provenance')
            self.assertTrue(ranked)
            self.assertEqual(2,ranked[0][0][2])
            def verified(_source,_sha,passages):
                self.assertEqual('PDF page 2',passages[0]['source_section'])
                self.assertIn('preserve page provenance',passages[0]['source_excerpt'])
                return [{'status':'verified','page':2,'basis':'derivative_label_confirmed','page_text_sha256':'a'*64}]
            r=verify_staged_page_result(ranked[0],query='publisher physical page provenance',
                original_root=self.orig,derived_root=self.live,stage_root=self.stage,
                completed_receipts={self.relative:digest(self.original)},
                source_map=self.sm,native_db=self.native,verifier=verified)
            self.assertEqual('publisher_original_physical_pdf_page_verified',r['status'])
            self.assertEqual(2,r['physical_pdf_page'])
            self.assertFalse(r['factual_answer_verified'])
            self.assertTrue(r['source_map_native_id_and_three_file_sha_rechecked_after_original_pdf_page_proof'])

    def test_publisher_pdf_original_changed_while_verifying_page_fails_closed(self):
        with StagedPhysicalPageRAMIndex(1) as index:
            index.add(self.item,stage_root=self.stage);index.commit()
            hit=index.ranked('publisher physical page provenance')[0]
            def changed(*_):
                (self.orig/self.relative).write_bytes(b'changed during proof')
                return [{'status':'verified','page':2,'basis':'derivative_label_confirmed',
                         'page_text_sha256':'a'*64}]
            with self.assertRaisesRegex(ValueError,'changed during physical PDF page verification'):
                verify_staged_page_result(hit,query='publisher physical page provenance',
                    original_root=self.orig,derived_root=self.live,stage_root=self.stage,
                    completed_receipts={self.relative:digest(self.original)},
                    source_map=self.sm,native_db=self.native,verifier=changed)

    def test_live_derivative_changed_while_verifying_page_fails_closed(self):
        with StagedPhysicalPageRAMIndex(1) as index:
            index.add(self.item,stage_root=self.stage);index.commit()
            hit=index.ranked('publisher physical page provenance')[0]
            def changed(*_):
                (self.live/(self.relative+'.txt')).write_bytes(b'new extraction revision mid-query')
                return [{'status':'verified','page':2,'basis':'derivative_label_confirmed',
                         'page_text_sha256':'a'*64}]
            with self.assertRaisesRegex(ValueError,'changed during physical PDF page verification'):
                verify_staged_page_result(hit,query='publisher physical page provenance',
                    original_root=self.orig,derived_root=self.live,stage_root=self.stage,
                    completed_receipts={self.relative:digest(self.original)},
                    source_map=self.sm,native_db=self.native,verifier=changed)

    def test_private_corrected_derivative_changed_while_verifying_page_fails_closed(self):
        with StagedPhysicalPageRAMIndex(1) as index:
            index.add(self.item,stage_root=self.stage);index.commit()
            hit=index.ranked('publisher physical page provenance')[0]
            def changed(*_):
                (self.stage/(self.relative+'.txt')).write_bytes(self.staged+b' concurrent changed body')
                return [{'status':'verified','page':2,'basis':'derivative_label_confirmed',
                         'page_text_sha256':'a'*64}]
            with self.assertRaisesRegex(ValueError,'changed during physical PDF page verification'):
                verify_staged_page_result(hit,query='publisher physical page provenance',
                    original_root=self.orig,derived_root=self.live,stage_root=self.stage,
                    completed_receipts={self.relative:digest(self.original)},
                    source_map=self.sm,native_db=self.native,verifier=changed)

    def test_source_map_advances_during_page_verification_fails_closed(self):
        with StagedPhysicalPageRAMIndex(1) as index:
            index.add(self.item,stage_root=self.stage);index.commit()
            hit=index.ranked('publisher physical page provenance')[0]
            def changed(*_):
                self.sm.execute('UPDATE originals SET derived_sha256=? WHERE source_path=?',
                                ('0'*64,self.relative));self.sm.commit()
                return [{'status':'verified','page':2,'basis':'derivative_label_confirmed',
                         'page_text_sha256':'a'*64}]
            with self.assertRaisesRegex(ValueError,'source-map revision changed'):
                verify_staged_page_result(hit,query='publisher physical page provenance',
                    original_root=self.orig,derived_root=self.live,stage_root=self.stage,
                    completed_receipts={self.relative:digest(self.original)},
                    source_map=self.sm,native_db=self.native,verifier=changed)

    def test_native_filedb_identity_changes_during_page_proof_fails_closed(self):
        with StagedPhysicalPageRAMIndex(1) as index:
            index.add(self.item,stage_root=self.stage);index.commit()
            hit=index.ranked('publisher physical page provenance')[0]
            def changed(*_):
                self.native.execute('UPDATE filesystem_entries SET id=? WHERE relative_path_display=?',
                                    (300,self.relative));self.native.commit()
                return [{'status':'verified','page':2,'basis':'derivative_label_confirmed',
                         'page_text_sha256':'a'*64}]
            with self.assertRaisesRegex(ValueError,'FileDB identity changed'):
                verify_staged_page_result(hit,query='publisher physical page provenance',
                    original_root=self.orig,derived_root=self.live,stage_root=self.stage,
                    completed_receipts={self.relative:digest(self.original)},
                    source_map=self.sm,native_db=self.native,verifier=changed)

    def test_publisher_page_proof_with_malformed_sha_is_not_valid(self):
        with StagedPhysicalPageRAMIndex(1) as index:
            index.add(self.item,stage_root=self.stage);index.commit()
            hit=index.ranked('publisher physical page provenance')[0]
            with self.assertRaisesRegex(ValueError,'page proof lacks a valid text digest'):
                verify_staged_page_result(hit,query='publisher physical page provenance',
                    original_root=self.orig,derived_root=self.live,stage_root=self.stage,
                    completed_receipts={self.relative:digest(self.original)},
                    source_map=self.sm,native_db=self.native,
                    verifier=lambda *_:[{'status':'verified','page':2,
                        'basis':'derivative_label_confirmed','page_text_sha256':'not-a-sha'}])

    def test_wrong_original_page_never_verified(self):
        with StagedPhysicalPageRAMIndex(1) as index:
            index.add(self.item,stage_root=self.stage);index.commit()
            r=verify_staged_page_result(index.ranked('retrieval original publisher')[0],
                query='retrieval original publisher',original_root=self.orig,derived_root=self.live,
                stage_root=self.stage,completed_receipts={self.relative:digest(self.original)},
                source_map=self.sm,native_db=self.native,
                verifier=lambda *_:[{'status':'verified','page':1,'basis':'unique_across_pdf'}])
            self.assertEqual('not_independently_verified',r['status'])

    def test_replaced_derivative_or_staged_source_fails_closed(self):
        with StagedPhysicalPageRAMIndex(1) as index:
            index.add(self.item,stage_root=self.stage);index.commit()
            hit=index.ranked('publisher physical page')[0]
            (self.live/(self.relative+'.txt')).write_bytes(b'changed original live derivative')
            with self.assertRaisesRegex(ValueError,'revision differs'):
                verify_staged_page_result(hit,query='publisher physical page',original_root=self.orig,
                    derived_root=self.live,stage_root=self.stage,
                    completed_receipts={self.relative:digest(self.original)},source_map=self.sm,native_db=self.native)

    def _attested_stage_hit(self):
        with StagedPhysicalPageRAMIndex(1) as idx:
            idx.add(self.item,stage_root=self.stage);idx.commit()
            hit=idx.ranked('publisher physical page provenance')[0]
        verified=verify_staged_page_result(hit,query='publisher physical page provenance',
            original_root=self.orig,derived_root=self.live,stage_root=self.stage,
            completed_receipts={self.relative:digest(self.original)},
            source_map=self.sm,native_db=self.native,
            verifier=lambda *_:[{'status':'verified','page':2,
                'basis':'derivative_label_confirmed','page_text_sha256':'a'*64}])
        self.assertEqual('publisher_original_physical_pdf_page_verified',verified['status'])
        return hit,verified

    def _physical_neighbor_proof(self,*,page_sha='a'*64):
        return {'original_pdf_revision_verified_before_and_after':True,
           'required_center_excerpt_verified':True,
           'separate_original_pages_not_joined_into_one_quotation':True,
           'original_pdf_sha256':digest(self.original),
           'requested_center_physical_page':2,'neighbor_radius':1,
           'pages':[{'physical_page':1,'physical_page_poppler_sha256':'b'*64,
                     'status':'verified_original_page_text','extracted_page_text':'Earlier source context.'},
                    {'physical_page':2,'physical_page_poppler_sha256':page_sha,
                     'status':'verified_original_page_text','extracted_page_text':'Actual original publisher physical page text.'}]}

    def _neighbors(self,hit,proof,reader):
        return attach_staged_physical_page_neighbors(hit,proof,
            original_root=self.orig,derived_root=self.live,stage_root=self.stage,
            completed_receipts={self.relative:digest(self.original)},
            source_map=self.sm,native_db=self.native,neighbor_radius=1,reader=reader)

    def test_adjacent_physical_page_context_attests_two_separate_original_pages(self):
        hit,proof=self._attested_stage_hit()
        def reader(_pdf,_sha,*,requested_page,neighbor_radius,required_center_excerpt):
            self.assertEqual(2,requested_page)
            self.assertEqual(1,neighbor_radius)
            self.assertEqual(proof['publisher_source_excerpt'],required_center_excerpt)
            return self._physical_neighbor_proof()
        got=self._neighbors(hit,proof,reader)
        self.assertEqual([1,2],[r['physical_page'] for r in got['publisher_original_physical_page_neighbors']['pages']])
        self.assertTrue(got['center_physical_pdf_page_sha_independently_correlated_twice'])
        self.assertTrue(got['source_map_native_id_and_all_three_source_sha_rechecked_after_neighbor_extraction'])
        self.assertFalse(got['neighbor_context_factual_answer_verified'])
        self.assertFalse(got['factual_answer_verified'])

    def test_neighbor_page_digest_mismatch_refused_after_source_bound_reader(self):
        hit,proof=self._attested_stage_hit()
        with self.assertRaisesRegex(ValueError,'digest changed'):
            self._neighbors(hit,proof,lambda *_args,**_kw:self._physical_neighbor_proof(page_sha='c'*64))

    def test_unreadable_adjacent_page_is_explicitly_unknown_but_center_remains_proven(self):
        hit,proof=self._attested_stage_hit()
        context=self._physical_neighbor_proof()
        context['pages'][0]={'physical_page':1,'status':'adjacent_original_pdf_page_text_unavailable',
            'physical_page_poppler_sha256':'b'*64,'extracted_page_text':None,
            'reason':'unreadable original neighbor','requires_original_pdf_visual_or_authorized_ocr_review':True}
        context['unavailable_neighbor_text_pages']=1
        out=self._neighbors(hit,proof,lambda *_args,**_kw:context)
        self.assertTrue(out['center_physical_pdf_page_sha_independently_correlated_twice'])
        self.assertIsNone(out['publisher_original_physical_page_neighbors']['pages'][0]['extracted_page_text'])
        self.assertFalse(out['neighbor_context_factual_answer_verified'])

    def test_unreadable_neighbor_must_never_invent_its_publisher_text(self):
        hit,proof=self._attested_stage_hit()
        context=self._physical_neighbor_proof()
        context['pages'][0]['status']='adjacent_original_pdf_page_text_unavailable'
        with self.assertRaisesRegex(ValueError,'invented text'):
            self._neighbors(hit,proof,lambda *_args,**_kw:context)

    def test_neighbor_extraction_source_map_advancement_fails_closed(self):
        hit,proof=self._attested_stage_hit()
        def change(*_args,**_kw):
            self.sm.execute('UPDATE originals SET derived_sha256=? WHERE source_path=?',('0'*64,self.relative))
            self.sm.commit()
            return self._physical_neighbor_proof()
        with self.assertRaisesRegex(ValueError,'source-map revision changed during neighbor extraction'):
            self._neighbors(hit,proof,change)

    def test_neighbor_extraction_changes_private_staged_text_fails_closed(self):
        hit,proof=self._attested_stage_hit()
        def change(*_args,**_kw):
            (self.stage/(self.relative+'.txt')).write_bytes(b'private source changed after page hash')
            return self._physical_neighbor_proof()
        with self.assertRaisesRegex(ValueError,'changed during neighbor extraction'):
            self._neighbors(hit,proof,change)

    def test_neighbor_extraction_changes_old_live_source_fails_closed(self):
        hit,proof=self._attested_stage_hit()
        def change(*_args,**_kw):
            (self.live/(self.relative+'.txt')).write_bytes(b'old source revised mid-query')
            return self._physical_neighbor_proof()
        with self.assertRaisesRegex(ValueError,'changed during neighbor extraction'):
            self._neighbors(hit,proof,change)

    def test_neighbor_extraction_changes_original_publisher_file_fails_closed(self):
        hit,proof=self._attested_stage_hit()
        def change(*_args,**_kw):
            (self.orig/self.relative).write_bytes(b'publisher original replaced mid-query')
            return self._physical_neighbor_proof()
        with self.assertRaisesRegex(ValueError,'changed during neighbor extraction'):
            self._neighbors(hit,proof,change)

    def test_neighbor_extraction_changes_native_original_id_fails_closed(self):
        hit,proof=self._attested_stage_hit()
        def change(*_args,**_kw):
            self.native.execute('UPDATE filesystem_entries SET id=888 WHERE relative_path_display=?',(self.relative,))
            self.native.commit()
            return self._physical_neighbor_proof()
        with self.assertRaisesRegex(ValueError,'FileDB identity changed during neighbor extraction'):
            self._neighbors(hit,proof,change)

    def test_unverified_or_stale_page_cannot_be_promoted_to_neighbor_proof(self):
        hit,proof=self._attested_stage_hit()
        for bad in ({**proof,'status':'not_independently_verified'},
                    {**proof,'source_excerpt_sha256':'0'*64},
                    {**proof,'original_path':'different.pdf'},
                    {**proof,'page_text_sha256':'not-a-digest'}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self._neighbors(hit,bad,lambda *_args,**_kw:self._physical_neighbor_proof())

    def test_unbounded_or_fabricated_neighbor_page_set_rejected(self):
        hit,proof=self._attested_stage_hit()
        with self.assertRaisesRegex(ValueError,'invalid bounded physical PDF neighbor radius'):
            attach_staged_physical_page_neighbors(hit,proof,original_root=self.orig,
                derived_root=self.live,stage_root=self.stage,
                completed_receipts={self.relative:digest(self.original)},
                source_map=self.sm,native_db=self.native,neighbor_radius=2,
                reader=lambda *_args,**_kw:self._physical_neighbor_proof())
        damaged=self._physical_neighbor_proof()
        damaged['pages']=[{'physical_page':2,'physical_page_poppler_sha256':'a'*64}]*4
        with self.assertRaisesRegex(ValueError,'exceeds the physical page limit'):
            self._neighbors(hit,proof,lambda *_args,**_kw:damaged)
        malformed=self._physical_neighbor_proof()
        malformed['pages']=[{'physical_page':2,'physical_page_poppler_sha256':'a'*64},
                            {'physical_page':1,'physical_page_poppler_sha256':'b'*64}]
        with self.assertRaisesRegex(ValueError,'not distinct and ordered'):
            self._neighbors(hit,proof,lambda *_args,**_kw:malformed)

    def test_duplicate_source_entry_rejected(self):
        with StagedPhysicalPageRAMIndex(2) as index:
            index.add(self.item,stage_root=self.stage)
            with self.assertRaisesRegex(ValueError,'duplicate'):index.add(self.item,stage_root=self.stage)

    def test_filename_does_not_rank_without_publisher_body_word(self):
        with StagedPhysicalPageRAMIndex(1) as index:
            index.add(self.item,stage_root=self.stage);index.commit()
            self.assertEqual([],index.ranked('known'))

    def test_exact_publisher_page_but_irrelevant_excerpt_is_excluded(self):
        with StagedPhysicalPageRAMIndex(1) as index:
            index.add(self.item,stage_root=self.stage);index.commit()
            # The page contains 'publisher' but not multiple relevant terms.
            hit=index.ranked('publisher extraterrestrial ontology')[0]
            result=verify_staged_page_result(hit,query='publisher extraterrestrial ontology',
                original_root=self.orig,derived_root=self.live,stage_root=self.stage,
                completed_receipts={self.relative:digest(self.original)},
                source_map=self.sm,native_db=self.native,
                verifier=lambda *_:[{'status':'verified','page':2,'basis':'derivative_label_confirmed'}])
            self.assertEqual('insufficient_source_query_term_coverage',result['status'])

    def test_no_verified_original_page_without_proof(self):
        with StagedPhysicalPageRAMIndex(1) as index:
            index.add(self.item,stage_root=self.stage);index.commit()
            result=verify_staged_page_result(index.ranked('publisher page')[0],query='publisher page',
                original_root=self.orig,derived_root=self.live,stage_root=self.stage,
                completed_receipts={self.relative:digest(self.original)},
                source_map=self.sm,native_db=self.native,
                verifier=lambda *_:[{'status':'unverified','reason':'excerpt_absent'}])
            self.assertNotEqual('publisher_original_physical_pdf_page_verified',result['status'])


if __name__=='__main__':unittest.main()
