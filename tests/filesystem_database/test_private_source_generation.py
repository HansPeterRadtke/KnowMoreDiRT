"""True private original/native/source-map generations, and atomic pointer crashes.

These use the independently tested two-PDF migration fixtures. Every positive
private source generation really passes the production-root and native vector
release gate. No live library files, workers, directories, or services touched.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest import mock
from tempfile import TemporaryDirectory
import unittest

import test_source_page_reindex_release_gate as source_fixtures
import file_system_catalog.private_source_generation as generation_module
from file_system_catalog.private_source_generation import (
    SCHEMA,POINTER_SCHEMA,read_pinned,verify_generation,publish_private_only,
)
from file_system_catalog.source_page_reindex_release_gate import assess_corrected_native_release
from file_system_catalog.source_review_audit import sha_file
from file_system_catalog.summary_coverage import revision_identity


def digest(raw:bytes):return hashlib.sha256(raw).hexdigest()


class PrivateSourceGenerationTests(unittest.TestCase):
    def setUp(self):
        t=TemporaryDirectory();self.addCleanup(t.cleanup)
        self.private_root=Path(t.name)
        self.switch=self.private_root/'isolated-generation-switch'
        self.switch.mkdir(mode=0o700)
        (self.switch/'generations').mkdir()
        self.fixture=source_fixtures.NativeReleaseGateTests()
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.fixture.build_all_correct_revisions()
        self.records=self.fixture.fixture.records

    def stage(self,generation:int,which:int):
        folder=self.switch/'generations'/f'gen-{generation:08d}'
        folder.mkdir()
        src=self.fixture.db
        row=self.records[which]
        original_path=row['original_path']; ident=row['native_original_filedb_id']
        native=folder/'native.sqlite3';sm=folder/'source-map.sqlite3'
        shutil.copy2(src,native)
        shutil.copy2(src,sm)
        with sqlite3.connect(native) as conn:
            conn.execute('DELETE FROM content_representations WHERE chunk_id IN ('
                 'SELECT chunk_id FROM content_chunks WHERE filesystem_entry_id!=?)',(ident,))
            conn.execute('DELETE FROM content_chunks WHERE filesystem_entry_id!=?',(ident,))
            conn.execute('DELETE FROM filesystem_entries WHERE id!=?',(ident,))
            conn.execute('DELETE FROM originals WHERE source_path!=?',(original_path,))
        with sqlite3.connect(sm) as conn:
            conn.execute('DELETE FROM originals WHERE source_path!=?',(original_path,))
        # The isolated source map and native FileDB are deliberately SEPARATE,
        # even though the initial migration test fixture has one combined DB.
        source_text=self.fixture.root/(original_path+'.txt')
        target_text=folder/'text'/(original_path+'.txt')
        target_text.parent.mkdir(parents=True)
        shutil.copy2(source_text,target_text)
        basename=Path(original_path).name+'.'+revision_identity(
                original_path,row['original_sha256'],row['new_staged_derived_sha256'])[:24]+'.summary-tree.json'
        summary=folder/'summaries'/basename
        summary.parent.mkdir()
        shutil.copy2(self.fixture.summaries/basename,summary)
        with sqlite3.connect(sm.resolve().as_uri()+'?mode=ro',uri=True) as conn_sm, \
             sqlite3.connect(native.resolve().as_uri()+'?mode=ro',uri=True) as conn_native, \
             sqlite3.connect(self.fixture.fixture.sqlite.resolve().as_uri()+'?mode=ro',uri=True) as hist:
            for c in (conn_sm,conn_native,hist):c.execute('PRAGMA query_only=ON')
            actual=assess_corrected_native_release(
                records=[row],source_map=conn_sm,native=conn_native,historical=hist,
                derived_root=folder/'text',summary_root=folder/'summaries',expected_count=1)
        self.assertTrue(actual['all_corrected_revision_search_evidence_complete'])
        self.assertFalse(actual['safe_to_cutover_live'])
        actual['native_sqlite_sha256']=sha_file(native)
        actual['source_map_sqlite_sha256']=sha_file(sm)
        receipt=folder/'proof.json';receipt.write_text(json.dumps(actual))
        paths=[('native_sqlite',native),('source_map_sqlite',sm),
               ('corrected_derived_text',target_text),('recursive_summary_tree',summary)]
        manifest={'schema':SCHEMA,'generation':generation,'cohort_originals':1,
                  'native_gate':{'path':'proof.json','sha256':sha_file(receipt)},
                  'assets':[{'kind':kind,'path':str(p.relative_to(folder)),
                    'sha256':sha_file(p),'bytes':p.stat().st_size} for kind,p in paths]}
        (folder/'manifest.json').write_text(json.dumps(manifest))
        return folder

    def test_private_generation_real_original_binding_acceptance_and_reader_pin(self):
        self.stage(1,0)
        with self.assertRaisesRegex(ValueError,'no private source generation'):
            read_pinned(self.switch,self.private_root)
        proof=verify_generation(self.switch,self.private_root,1)
        self.assertEqual(4,proof['candidate_files_byte_verified'])
        self.assertTrue(proof['private_generation_complete'])
        self.assertFalse(proof['safe_to_cutover_live'])
        published=publish_private_only(self.switch,self.private_root,1)
        self.assertFalse(published['safe_to_cutover_live'])
        pinned=read_pinned(self.switch,self.private_root)
        self.assertEqual(1,pinned.generation)
        self.assertEqual(1,pinned.verify_bytes(self.private_root)['generation'])
        self.assertTrue((self.switch/'generations/gen-00000001/manifest.json').exists())

    def test_failed_pointer_switch_retains_older_complete_generation(self):
        self.stage(1,0)
        publish_private_only(self.switch,self.private_root,1)
        pinned_old=read_pinned(self.switch,self.private_root)
        self.stage(2,1)
        def crash():raise RuntimeError('injected isolated process crash before pointer replace')
        with self.assertRaisesRegex(RuntimeError,'injected'):
            publish_private_only(self.switch,self.private_root,2,before_pointer_replace=crash)
        self.assertEqual(1,read_pinned(self.switch,self.private_root).generation)
        self.assertEqual(1,pinned_old.verify_bytes(self.private_root)['generation'])
        self.assertFalse((self.switch/'.private-generation-pointer-new-DOES-NOT-EXIST').exists())
        published=publish_private_only(self.switch,self.private_root,2)
        self.assertEqual(1,published['previous_private_generation'])
        self.assertFalse(published['safe_to_cutover_live'])
        self.assertEqual(2,read_pinned(self.switch,self.private_root).generation)
        self.assertEqual(1,pinned_old.verify_bytes(self.private_root)['generation'])
        self.assertTrue((self.switch/'generations/gen-00000001/manifest.json').exists())
        self.assertTrue((self.switch/'generations/gen-00000002/manifest.json').exists())

    def test_staging_mutated_during_last_preflight_cannot_publish_pointer(self):
        folder=self.stage(1,0)
        derived=next((folder/'text').rglob('*.txt'))
        def mutation():derived.write_bytes(derived.read_bytes()+b'private pre-publish tamper')
        with self.assertRaisesRegex(ValueError,'asset bytes changed'):
            publish_private_only(self.switch,self.private_root,1,before_pointer_replace=mutation)
        self.assertFalse((self.switch/'active-private-generation.json').exists())

    def test_crash_after_private_switch_requires_reading_current_pointer_not_replay(self):
        self.stage(1,0)
        publish_private_only(self.switch,self.private_root,1)
        first=read_pinned(self.switch,self.private_root)
        self.stage(2,1)
        def crash():raise RuntimeError('injected test-only crash after private pointer was switched')
        with self.assertRaisesRegex(RuntimeError,'after private pointer'):
            publish_private_only(self.switch,self.private_root,2,after_pointer_replace=crash)
        actual=read_pinned(self.switch,self.private_root)
        self.assertEqual(2,actual.generation)
        self.assertEqual(1,first.verify_bytes(self.private_root)['generation'])
        self.assertEqual(2,actual.verify_bytes(self.private_root)['generation'])
        with self.assertRaisesRegex(ValueError,'increment exactly one'):
            publish_private_only(self.switch,self.private_root,2)

    def test_wrong_or_replayed_ordinal_never_replaces_private_original_generation(self):
        self.stage(1,0)
        publish_private_only(self.switch,self.private_root,1)
        with self.assertRaisesRegex(ValueError,'increment exactly one'):
            publish_private_only(self.switch,self.private_root,1)
        self.stage(3,1)
        with self.assertRaisesRegex(ValueError,'increment exactly one'):
            publish_private_only(self.switch,self.private_root,3)
        self.assertEqual(1,read_pinned(self.switch,self.private_root).generation)

    def test_mutating_new_source_map_or_source_text_fails_closed(self):
        folder=self.stage(1,0)
        asset=next((folder/'text').rglob('*.txt'))
        asset.write_bytes(asset.read_bytes()+b'\nwrong document\n')
        with self.assertRaisesRegex(ValueError,'asset bytes changed'):
            publish_private_only(self.switch,self.private_root,1)
        self.assertFalse((self.switch/'active-private-generation.json').exists())

    def test_manifest_rebound_to_wrong_revision_is_rejected_even_if_all_shas_match(self):
        folder=self.stage(1,0)
        sm=folder/'source-map.sqlite3'
        with sqlite3.connect(sm) as db:
            db.execute('UPDATE originals SET derived_sha256=?',('a'*64,))
        # Keep the manifest and synthetic QA SHA self-consistent after the
        # wrong source-map rewrite, to test actual per-original binding.
        qa_file=folder/'proof.json'
        qa=json.loads(qa_file.read_text());qa['source_map_sqlite_sha256']=sha_file(sm)
        qa_file.write_text(json.dumps(qa))
        manifest_path=folder/'manifest.json';manifest=json.loads(manifest_path.read_text())
        manifest['native_gate']['sha256']=sha_file(qa_file)
        for item in manifest['assets']:
            if item['kind']=='source_map_sqlite':
                item['sha256']=sha_file(sm);item['bytes']=sm.stat().st_size
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'source map|summary tree|summary-tree'):
            publish_private_only(self.switch,self.private_root,1)

    def test_source_qa_rejection_preserves_no_active_generation(self):
        folder=self.stage(1,0)
        qa_file=folder/'proof.json'
        qa=json.loads(qa_file.read_text());qa['all_corrected_revision_search_evidence_complete']=False
        qa_file.write_text(json.dumps(qa))
        manifest_path=folder/'manifest.json';manifest=json.loads(manifest_path.read_text())
        manifest['native_gate']['sha256']=sha_file(qa_file)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'not completely rebuilt'):
            publish_private_only(self.switch,self.private_root,1)
        self.assertFalse((self.switch/'active-private-generation.json').exists())

    def test_old_pinned_reader_detects_original_source_tampering(self):
        folder=self.stage(1,0)
        publish_private_only(self.switch,self.private_root,1)
        pinned=read_pinned(self.switch,self.private_root)
        source=next((folder/'text').rglob('*.txt'))
        source.write_bytes(source.read_bytes()+b'injected')
        with self.assertRaises(ValueError):pinned.verify_bytes(self.private_root)
        with self.assertRaises(ValueError):read_pinned(self.switch,self.private_root)

    def test_read_pins_unchanged_complete_generation_during_concurrent_switch(self):
        self.stage(1,0)
        self.stage(2,1)
        publish_private_only(self.switch,self.private_root,1)
        reader_entered=threading.Event();reader_release=threading.Event()
        original=generation_module.verify_generation
        def hold_verification(base,allowed_root,generation):
            if threading.current_thread().name.startswith('PinnedReader') and generation==1:
                reader_entered.set()
                if not reader_release.wait(timeout=3):
                    raise AssertionError('timed out while holding old source snapshot reader lock')
            return original(base,allowed_root,generation)
        with mock.patch.object(generation_module,'verify_generation',side_effect=hold_verification):
            with ThreadPoolExecutor(max_workers=2,thread_name_prefix='PinnedReader') as pool:
                old_reader=pool.submit(read_pinned,self.switch,self.private_root)
                self.assertTrue(reader_entered.wait(timeout=3))
                # Publish from the other thread while old read holds LOCK_SH;
                # even after its preflight, marker must remain gen-1.
                publisher=pool.submit(publish_private_only,self.switch,self.private_root,2)
                self.assertFalse(publisher.done())
                self.assertEqual(1,json.loads((self.switch/'active-private-generation.json').read_text())['generation'])
                reader_release.set()
                old=old_reader.result(timeout=3)
                outcome=publisher.result(timeout=3)
        self.assertEqual(1,old.generation)
        self.assertEqual(2,outcome['private_pointer_generation'])
        self.assertEqual(2,read_pinned(self.switch,self.private_root).generation)
        self.assertEqual(1,old.verify_bytes(self.private_root)['generation'])

    def test_coherent_private_two_database_reader_keeps_old_pair_across_new_pointer(self):
        folder1=self.stage(1,0)
        folder2=self.stage(2,1)
        publish_private_only(self.switch,self.private_root,1)
        old=self.records[0]['original_path'];new=self.records[1]['original_path']
        # A real reader must hold one old native/SM pair, never combine two
        # different generation pointers when another thread publishes later.
        with generation_module.open_pinned_private_read_view(self.switch,self.private_root) as view:
            self.assertEqual(1,view.generation)
            self.assertEqual([old],[x[0] for x in view.source_map.execute(
                'SELECT source_path FROM originals ORDER BY source_path')])
            self.assertEqual([old],[x[0] for x in view.native.execute(
                'SELECT relative_path_display FROM filesystem_entries ORDER BY id')])
            with ThreadPoolExecutor(max_workers=1) as pool:
                result=pool.submit(publish_private_only,self.switch,self.private_root,2).result(timeout=10)
            self.assertEqual(2,result['private_pointer_generation'])
            self.assertEqual(2,read_pinned(self.switch,self.private_root).generation)
            self.assertEqual([old],[x[0] for x in view.native.execute(
                'SELECT relative_path_display FROM filesystem_entries ORDER BY id')])
            self.assertEqual([old],[x[0] for x in view.source_map.execute(
                'SELECT source_path FROM originals ORDER BY source_path')])
            asset='text/'+old+'.txt'
            data=view.read_source_asset('corrected_derived_text',asset)
            digest_source=view.source_map.execute('SELECT derived_sha256 FROM originals').fetchone()[0]
            self.assertEqual(digest(data),digest_source)
            self.assertIn(b'[[SOURCE UNIT:',data)
            summary_assets=[name for kind,name,_ in view.pin.assets if kind=='recursive_summary_tree']
            self.assertEqual(1,len(summary_assets))
            tree=json.loads(view.read_source_asset('recursive_summary_tree',summary_assets[0]))
            self.assertEqual(old,tree['original_path'])
            self.assertEqual(digest_source,tree['derived_sha256'])
            self.assertTrue(view.verify_bytes(self.private_root)['private_generation_complete'])
        with generation_module.open_pinned_private_read_view(self.switch,self.private_root) as fresh:
            self.assertEqual(2,fresh.generation)
            self.assertEqual([new],[x[0] for x in fresh.source_map.execute(
                'SELECT source_path FROM originals ORDER BY source_path')])
            self.assertEqual([new],[x[0] for x in fresh.native.execute(
                'SELECT relative_path_display FROM filesystem_entries ORDER BY id')])
        self.assertTrue(folder1.exists() and folder2.exists())

    def test_old_private_dual_database_reader_refuses_tamper_at_context_exit(self):
        folder=self.stage(1,0)
        publish_private_only(self.switch,self.private_root,1)
        with self.assertRaisesRegex(ValueError,'asset bytes changed'):
            with generation_module.open_pinned_private_read_view(self.switch,self.private_root) as view:
                self.assertEqual(1,view.generation)
                self.assertTrue(view.native.execute('SELECT 1').fetchone())
                # It must reject a modified old pinned view even if pointer
                # remains the same; never return a passed final view check.
                asset=next((folder/'text').rglob('*.txt'))
                asset.write_bytes(asset.read_bytes()+b'\ninjected private byte change\n')
        with self.assertRaisesRegex(ValueError,'asset bytes changed'):
            read_pinned(self.switch,self.private_root)

    def test_private_two_database_view_does_not_allow_sqlite_writes_or_live_roots(self):
        self.stage(1,0)
        publish_private_only(self.switch,self.private_root,1)
        with generation_module.open_pinned_private_read_view(self.switch,self.private_root) as view:
            self.assertFalse(view.safe_to_cutover_live)
            self.assertEqual(1,view.generation)
            for db in (view.native,view.source_map):
                with self.assertRaises(sqlite3.OperationalError):
                    db.execute('DELETE FROM originals')
        if Path('/data/var/knowmoredirt/knowledge-library').exists():
            with self.assertRaises(ValueError):
                with generation_module.open_pinned_private_read_view(
                     Path('/data/var/knowmoredirt/knowledge-library'),Path('/data/var')):
                    pass

    def test_private_reader_refuses_unlisted_wrong_role_tampered_and_unbounded_asset(self):
        folder=self.stage(1,0)
        publish_private_only(self.switch,self.private_root,1)
        rel='text/'+self.records[0]['original_path']+'.txt'
        with generation_module.open_pinned_private_read_view(self.switch,self.private_root) as view:
            good=view.read_source_asset('corrected_derived_text',rel)
            self.assertTrue(good)
            for kind,path,limit in [('native_sqlite',rel,100000),
                                    ('recursive_summary_tree',rel,100000),
                                    ('corrected_derived_text','../escape.txt',100000),
                                    ('corrected_derived_text',rel,1),
                                    ('corrected_derived_text',rel,999999999)]:
                with self.subTest(kind=kind,path=path,limit=limit),self.assertRaises(ValueError):
                    view.read_source_asset(kind,path,max_bytes=limit)
        (folder/rel).write_bytes((folder/rel).read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'asset bytes changed'):
            with generation_module.open_pinned_private_read_view(self.switch,self.private_root):
                pass

    def test_private_commit_directory_is_synced_before_post_switch_crash_hook(self):
        self.stage(1,0)
        syncs=[];actual=generation_module.os.fsync
        def seen(fd):
            mode=__import__('os').fstat(fd).st_mode
            if __import__('stat').S_ISDIR(mode):syncs.append('directory_synced')
            return actual(fd)
        def crash_after_switch():
            self.assertEqual(1,json.loads((self.switch/'active-private-generation.json').read_text())['generation'])
            self.assertTrue(syncs,'published pointer must be fsynced in its parent directory before crash hook')
            raise RuntimeError('private post-fsync crash simulation')
        with mock.patch.object(generation_module.os,'fsync',side_effect=seen):
            with self.assertRaisesRegex(RuntimeError,'post-fsync crash'):
                publish_private_only(self.switch,self.private_root,1,after_pointer_replace=crash_after_switch)
        self.assertEqual(1,read_pinned(self.switch,self.private_root).generation)

    def test_private_only_root_enforcement_refuses_real_live_kmd_directories(self):
        from file_system_catalog.private_source_generation import _validate_private_root
        live=Path('/data/var/knowmoredirt/knowledge-library')
        if live.is_dir():
            with self.assertRaisesRegex(ValueError,'private'):
                _validate_private_root(live,Path('/data/var'))
        originals=Path('/data/var/Knowledge Library')
        if originals.is_dir():
            with self.assertRaisesRegex(ValueError,'private'):
                _validate_private_root(originals,Path('/data/var'))

    def test_symlinked_or_world_readable_generation_lock_never_admits_reader(self):
        self.stage(1,0)
        lock=self.switch/'.private-generation-pointer.lock'
        external=self.private_root/'external-lock.txt'
        external.write_text('outside')
        lock.symlink_to(external)
        with self.assertRaises(OSError):publish_private_only(self.switch,self.private_root,1)
        self.assertFalse((self.switch/'active-private-generation.json').exists())
        lock.unlink()
        publish_private_only(self.switch,self.private_root,1)
        lock.chmod(0o644)
        with self.assertRaisesRegex(ValueError,'inaccessible to other users'):
            read_pinned(self.switch,self.private_root)
        lock.chmod(0o600)
        self.assertEqual(1,read_pinned(self.switch,self.private_root).generation)

    def test_manifest_pointer_does_not_trust_unexpected_sha_or_symlink(self):
        self.stage(1,0)
        publish_private_only(self.switch,self.private_root,1)
        pointer=self.switch/'active-private-generation.json'
        record=json.loads(pointer.read_text());record['manifest_sha256']='f'*64
        pointer.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'modified source-generation manifest'):
            read_pinned(self.switch,self.private_root)
        pointer.unlink()
        target=self.switch/'pointer-outside.json'
        target.write_text(json.dumps(record))
        pointer.symlink_to(target)
        with self.assertRaises(ValueError):read_pinned(self.switch,self.private_root)

if __name__=='__main__':unittest.main()
