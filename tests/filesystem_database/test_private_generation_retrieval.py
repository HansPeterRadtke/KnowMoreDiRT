"""Pinned source-linked retrieval against isolated immutable PDF generations.

These tests never query, download, stage, rewrite or hold the live Knowledge
Library database or source collection. Generated PDF text is NOT a quote.
"""
from __future__ import annotations
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
import unittest
from unittest import mock

import test_private_source_generation as generation_fixture
from file_system_catalog.private_source_generation import publish_private_only
from file_system_catalog.private_source_admission import (
    admit_private_legacy_writer,publish_private_after_legacy_drain,
)
from file_system_catalog.private_generation_retrieval import retrieve_private_source_locators
import file_system_catalog.private_generation_retrieval as private_reader


class PrivateSourceLinkedRetrievalTests(unittest.TestCase):
    def setUp(self):
        f=generation_fixture.PrivateSourceGenerationTests();f.setUp()
        self.addCleanup(f.doCleanups)
        self.f=f
        self.first=f.records[0]
        self.second=f.records[1]

    def stage_two_originals_in_one_immutable_cohort(self):
        from file_system_catalog.private_source_generation import SCHEMA
        from file_system_catalog.source_page_reindex_release_gate import assess_corrected_native_release
        from file_system_catalog.source_review_audit import sha_file
        from file_system_catalog.summary_coverage import revision_identity
        folder=self.f.switch/'generations'/'gen-00000001'
        folder.mkdir()
        native=folder/'native.sqlite3';source_map=folder/'source-map.sqlite3'
        shutil.copy2(self.f.fixture.db,native)
        shutil.copy2(self.f.fixture.db,source_map)
        assets=[('native_sqlite',native),('source_map_sqlite',source_map)]
        for row in self.f.records:
            name=row['original_path'];text=folder/'text'/(name+'.txt')
            text.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(self.f.fixture.root/(name+'.txt'),text)
            filename=Path(name).name+'.'+revision_identity(
                name,row['original_sha256'],row['new_staged_derived_sha256'])[:24]+'.summary-tree.json'
            summary=folder/'summaries'/filename
            summary.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(self.f.fixture.summaries/filename,summary)
            assets.extend([('corrected_derived_text',text),('recursive_summary_tree',summary)])
        with sqlite3.connect(source_map.resolve().as_uri()+'?mode=ro',uri=True) as sm, \
             sqlite3.connect(native.resolve().as_uri()+'?mode=ro',uri=True) as db, \
             sqlite3.connect(self.f.fixture.fixture.sqlite.resolve().as_uri()+'?mode=ro',uri=True) as historic:
            for connection in (sm,db,historic):connection.execute('pragma query_only=on')
            ready=assess_corrected_native_release(records=self.f.records,
                source_map=sm,native=db,historical=historic,derived_root=folder/'text',
                summary_root=folder/'summaries',expected_count=2)
        self.assertEqual(2,ready['originals_with_native_full_source_roots_and_vectors'])
        ready['native_sqlite_sha256']=sha_file(native)
        ready['source_map_sqlite_sha256']=sha_file(source_map)
        receipt=folder/'proof.json';receipt.write_text(json.dumps(ready))
        manifest={'schema':SCHEMA,'generation':1,'cohort_originals':2,
            'native_gate':{'path':'proof.json','sha256':sha_file(receipt)},
            'assets':[{'kind':kind,'path':str(file.relative_to(folder)),
                       'sha256':sha_file(file),'bytes':file.stat().st_size}
                       for kind,file in assets]}
        (folder/'manifest.json').write_text(json.dumps(manifest))
        pub=publish_private_only(self.f.switch,self.f.private_root,1)
        self.assertEqual(1,pub['private_pointer_generation'])
        return folder

    def stage_one(self):
        path=self.f.stage(1,0)
        publish_private_only(self.f.switch,self.f.private_root,1)
        return path

    def lookup(self,path,query,top=2):
        return retrieve_private_source_locators(self.f.switch,self.f.private_root,path,query,top=top)

    def test_current_original_private_page_locator_has_exact_provisional_SHA_not_publisher_quote(self):
        folder=self.stage_one()
        first=self.first
        found=self.lookup(first['original_path'],'publisher evidence')
        self.assertEqual(1,found['generation'])
        self.assertEqual(first['original_path'],found['original_path'])
        self.assertEqual(first['original_sha256'],found['original_sha256'])
        self.assertEqual(first['new_staged_derived_sha256'],found['derived_sha256'])
        self.assertTrue(found['pinned_native_summary_root_id'])
        self.assertEqual(1,found['returned_source_locators'])
        hit=found['locators'][0]
        self.assertEqual('PDF page 1',hit['source_section'])
        raw=(folder/'text'/(first['original_path']+'.txt')).read_text('utf8')
        self.assertEqual(hashlib.sha256(raw[hit['excerpt_start_char']:hit['excerpt_end_char']].encode()).hexdigest(),
                         hit['excerpt_sha256'])
        self.assertEqual(['evidence','publisher'],hit['matched_source_terms'])
        self.assertNotIn('source_excerpt',hit,'provisional machine text must not be presented as a quote')
        self.assertFalse(hit['original_pdf_page_independently_verified'])
        self.assertFalse(hit['original_publisher_quote_verified'])
        self.assertFalse(found['source_factual_claims_verified'])
        self.assertFalse(found['safe_to_cutover_live'])

    def test_unknown_and_malformed_private_originals_cannot_escape_one_generation(self):
        self.stage_one()
        for path in ('../private.pdf','/data/var/Knowledge Library/private.pdf','publishers\\file.pdf','publishers/missing.pdf'):
            with self.subTest(path=path),self.assertRaisesRegex(ValueError,'private source|one safe|not in this'):
                self.lookup(path,'publisher evidence')
        for top in (0,6,True,100000):
            with self.subTest(top=top),self.assertRaisesRegex(ValueError,'bounded query'):
                self.lookup(self.first['original_path'],'publisher evidence',top=top)
        with self.assertRaises(ValueError):
            self.lookup(self.first['original_path'],'source\x00control')

    def test_negative_exact_source_term_request_yields_zero_locators_without_model_guess(self):
        self.stage_one()
        found=self.lookup(self.first['original_path'],'fabricated astrophysical cornflower 0000')
        self.assertEqual(0,found['returned_source_locators'])
        self.assertEqual([],found['locators'])
        self.assertTrue(found['all_returned_locations_current_derived_text_SHA_verified'])
        self.assertFalse(found['source_factual_claims_verified'])

    def test_old_private_reader_completes_during_new_generation_pointer_switch(self):
        self.stage_one()
        self.f.stage(2,1)
        underlying=private_reader.source_evidence_candidates
        switches=[]
        def switch_during_private_source_search(**kwargs):
            pub=publish_private_only(self.f.switch,self.f.private_root,2)
            switches.append(pub['private_pointer_generation'])
            return underlying(**kwargs)
        with mock.patch.object(private_reader,'source_evidence_candidates',side_effect=switch_during_private_source_search):
            old=self.lookup(self.first['original_path'],'publisher evidence')
        self.assertEqual([2],switches)
        self.assertEqual(1,old['generation'])
        self.assertEqual(1,old['returned_source_locators'])
        new=self.lookup(self.second['original_path'],'private forward')
        self.assertEqual(2,new['generation'])
        self.assertEqual(self.second['original_sha256'],new['original_sha256'])
        self.assertGreaterEqual(new['returned_source_locators'],1)
        with self.assertRaisesRegex(ValueError,'not in this'):
            self.lookup(self.first['original_path'],'publisher evidence')

    def test_generation_aware_private_lookup_works_only_after_cooperating_writer_drains(self):
        self.f.stage(1,0)
        with admit_private_legacy_writer(self.f.switch,self.f.private_root):
            with self.assertRaisesRegex(ValueError,'legacy reader or writer remains active'):
                publish_private_after_legacy_drain(self.f.switch,self.f.private_root,1)
        published=publish_private_after_legacy_drain(self.f.switch,self.f.private_root,1)
        self.assertEqual(1,published['private_pointer_generation'])
        self.assertTrue(published['cooperating_legacy_reader_admissions_sealed'])
        self.assertFalse(published['safe_to_cutover_live'])
        found=self.lookup(self.first['original_path'],'publisher evidence')
        self.assertEqual(1,found['generation'])
        self.assertEqual(1,found['returned_source_locators'])
        with self.assertRaisesRegex(ValueError,'legacy reader admission.*sealed'):
            with admit_private_legacy_writer(self.f.switch,self.f.private_root):pass

    def test_pinned_source_tamper_during_query_rejected_before_results_escape_context(self):
        folder=self.stage_one()
        underlying=private_reader.source_evidence_candidates
        path=folder/'text'/(self.first['original_path']+'.txt')
        def malicious_machine_content_mutation(**kwargs):
            candidates=underlying(**kwargs)
            path.write_bytes(path.read_bytes()+b'\nprivate test-only changes after retrieval\n')
            return candidates
        with mock.patch.object(private_reader,'source_evidence_candidates',side_effect=malicious_machine_content_mutation):
            with self.assertRaisesRegex(ValueError,'asset bytes changed'):
                self.lookup(self.first['original_path'],'publisher evidence')

    def test_two_originals_share_one_generation_but_never_mix_source_references(self):
        folder=self.stage_two_originals_in_one_immutable_cohort()
        first=self.lookup(self.first['original_path'],'publisher evidence')
        second=self.lookup(self.second['original_path'],'private forward')
        self.assertEqual(first['generation'],second['generation'])
        self.assertEqual(1,first['generation'])
        self.assertEqual(1,first['returned_source_locators'])
        self.assertGreaterEqual(second['returned_source_locators'],1)
        self.assertNotEqual(first['original_sha256'],second['original_sha256'])
        self.assertNotEqual(first['native_filedb_original_id'],second['native_filedb_original_id'])
        self.assertNotEqual(first['pinned_complete_summary_tree_sha256'],
                            second['pinned_complete_summary_tree_sha256'])
        self.assertTrue((folder/'source-map.sqlite3').is_file())
        for found,row in ((first,self.first),(second,self.second)):
            self.assertEqual(row['original_path'],found['original_path'])
            self.assertEqual(row['new_staged_derived_sha256'],found['derived_sha256'])
            self.assertFalse(found['original_publisher_pdf_physical_page_verified'])
            self.assertFalse(found['safe_to_cutover_live'])

    def test_native_index_mutated_during_private_lookup_rejected_after_source_query(self):
        folder=self.stage_one()
        native=folder/'native.sqlite3'
        underlying=private_reader.source_evidence_candidates
        def change_native_record_after_selection(**kwargs):
            result=underlying(**kwargs)
            with sqlite3.connect(native) as db:
                db.execute("UPDATE filesystem_entries SET relative_path_display=? WHERE id=?",
                           ('unrelated/other-source.pdf',self.first['native_original_filedb_id']))
            return result
        with mock.patch.object(private_reader,'source_evidence_candidates',
                               side_effect=change_native_record_after_selection):
            with self.assertRaisesRegex(ValueError,'asset bytes changed'):
                self.lookup(self.first['original_path'],'publisher evidence')

    def test_wrong_source_span_SHA_cannot_pass_even_if_excerpt_SHA_matches(self):
        self.stage_one()
        original=private_reader.source_evidence_candidates
        def corrupted_span(**kwargs):
            result=original(**kwargs)
            self.assertTrue(result['passages'])
            result['passages'][0]['source_span_sha256']='f'*64
            return result
        with mock.patch.object(private_reader,'source_evidence_candidates',side_effect=corrupted_span):
            with self.assertRaisesRegex(ValueError,'not exact current generated text'):
                self.lookup(self.first['original_path'],'publisher evidence')

    def test_pinned_summary_tree_tamper_during_query_is_rejected_before_result(self):
        folder=self.stage_one()
        tree=next((folder/'summaries').rglob('*.summary-tree.json'))
        underlying=private_reader.source_evidence_candidates
        def swapped_tree(**kwargs):
            candidates=underlying(**kwargs)
            tree.write_bytes(tree.read_bytes()+b'\nprivate test-only summary revision replacement\n')
            return candidates
        with mock.patch.object(private_reader,'source_evidence_candidates',side_effect=swapped_tree):
            with self.assertRaisesRegex(ValueError,'asset bytes changed'):
                self.lookup(self.first['original_path'],'publisher evidence')

    def test_private_lookup_still_refuses_the_live_kmd_root(self):
        if Path('/data/var/knowmoredirt/knowledge-library').is_dir():
            with self.assertRaisesRegex(ValueError,'private'):
                retrieve_private_source_locators(Path('/data/var/knowmoredirt/knowledge-library'),
                    Path('/data/var'),self.first['original_path'],'publisher evidence')
