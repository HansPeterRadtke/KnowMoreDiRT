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

    def test_two_original_pdfs_share_one_pinned_private_native_and_source_map_generation(self):
        # The prior crash and pointer tests use one original per generation.
        # A real migration stage also needs MULTIPLE current original rows,
        # matching derivatives, and recursive summary roots in ONE snapshot.
        folder=self.switch/'generations'/'gen-00000001'
        folder.mkdir()
        native=folder/'native.sqlite3';source_map=folder/'source-map.sqlite3'
        shutil.copy2(self.fixture.db,native);shutil.copy2(self.fixture.db,source_map)
        assets=[('native_sqlite',native),('source_map_sqlite',source_map)]
        for row in self.records:
            name=row['original_path'];target=folder/'text'/(name+'.txt')
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(self.fixture.root/(name+'.txt'),target)
            base=Path(name).name+'.'+revision_identity(name,row['original_sha256'],
                row['new_staged_derived_sha256'])[:24]+'.summary-tree.json'
            summary=folder/'summaries'/base
            summary.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(self.fixture.summaries/base,summary)
            assets.extend([('corrected_derived_text',target),('recursive_summary_tree',summary)])
        with sqlite3.connect(source_map.resolve().as_uri()+'?mode=ro',uri=True) as sm, \
             sqlite3.connect(native.resolve().as_uri()+'?mode=ro',uri=True) as db, \
             sqlite3.connect(self.fixture.fixture.sqlite.resolve().as_uri()+'?mode=ro',uri=True) as historic:
            for con in (sm,db,historic):con.execute('PRAGMA query_only=ON')
            ready=assess_corrected_native_release(records=self.records,source_map=sm,
                native=db,historical=historic,derived_root=folder/'text',
                summary_root=folder/'summaries',expected_count=2)
        self.assertEqual(2,ready['originals_with_native_full_source_roots_and_vectors'])
        ready['native_sqlite_sha256']=sha_file(native)
        ready['source_map_sqlite_sha256']=sha_file(source_map)
        proof=folder/'proof.json';proof.write_text(json.dumps(ready))
        manifest={'schema':SCHEMA,'generation':1,'cohort_originals':2,
            'native_gate':{'path':'proof.json','sha256':sha_file(proof)},
            'assets':[{'kind':kind,'path':str(file.relative_to(folder)),
                       'sha256':sha_file(file),'bytes':file.stat().st_size}
                      for kind,file in assets]}
        (folder/'manifest.json').write_text(json.dumps(manifest))
        published=publish_private_only(self.switch,self.private_root,1)
        self.assertEqual(1,published['private_pointer_generation'])
        with generation_module.open_pinned_private_read_view(self.switch,self.private_root) as view:
            native_paths={r[0] for r in view.native.execute(
                'SELECT relative_path_display FROM filesystem_entries')}
            map_sources={r[0]:(r[1],r[2]) for r in view.source_map.execute(
                'SELECT source_path,source_sha256,derived_sha256 FROM originals')}
            self.assertEqual(native_paths,set(map_sources))
            self.assertEqual(native_paths,{r['original_path'] for r in self.records})
            for path,(original_sha,derived_sha) in map_sources.items():
                source=view.read_source_asset('corrected_derived_text','text/'+path+'.txt')
                self.assertEqual(derived_sha,digest(source))
                self.assertTrue(any(path==r['original_path'] and original_sha==r['original_sha256']
                                    for r in self.records))
            self.assertTrue(view.verify_bytes(self.private_root)['private_generation_complete'])
        self.assertEqual(2,len(map_sources))
        self.assertFalse(ready['safe_to_cutover_live'])

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

    def test_every_staged_asset_and_nested_directory_fsynced_before_private_pointer_commit(self):
        import os
        folder=self.stage(1,0)
        expected={folder/'manifest.json',folder/'proof.json',folder/'native.sqlite3',
                  folder/'source-map.sqlite3',folder,folder/'text',folder/'summaries',
                  self.switch/'generations',self.switch}
        expected.update((folder/'text').rglob('*.txt'))
        expected.update((folder/'summaries').rglob('*.summary-tree.json'))
        for file in (folder/'text').rglob('*.txt'):
            expected.update(parent for parent in file.parents if parent.is_relative_to(folder))
        synced=set();real=os.fsync
        def record(fd):
            target=Path(os.readlink(f'/proc/self/fd/{fd}'))
            synced.add(target)
            return real(fd)
        def after_preflight_and_durable_assets():
            missing=expected-synced
            self.assertEqual(set(),missing,'private generation files/directories are not durably staged before pointer switch')
            raise RuntimeError('test-only exit after durable assets before private pointer replace')
        with mock.patch.object(generation_module.os,'fsync',side_effect=record):
            with self.assertRaisesRegex(RuntimeError,'after durable assets'):
                publish_private_only(self.switch,self.private_root,1,
                     before_pointer_replace=after_preflight_and_durable_assets)
        self.assertFalse((self.switch/'active-private-generation.json').exists())
        self.assertTrue(synced.issuperset(expected))

    def test_failed_staged_file_or_directory_fsync_keeps_old_private_generation(self):
        import os
        self.stage(1,0)
        publish_private_only(self.switch,self.private_root,1)
        self.stage(2,1)
        actual=os.fsync
        for failed in ('native.sqlite3','gen-00000002'):
            with self.subTest(failed=failed):
                def inject(fd):
                    target=Path(os.readlink(f'/proc/self/fd/{fd}'))
                    if (failed=='native.sqlite3' and target.name==failed and
                            'gen-00000002' in str(target)) or (
                            failed=='gen-00000002' and target.name==failed):
                        raise OSError('isolated private staged fsync failed')
                    return actual(fd)
                with mock.patch.object(generation_module.os,'fsync',side_effect=inject):
                    with self.assertRaisesRegex(OSError,'staged fsync failed'):
                        publish_private_only(self.switch,self.private_root,2)
                self.assertEqual(1,read_pinned(self.switch,self.private_root).generation)
                self.assertEqual(1,json.loads((self.switch/'active-private-generation.json').read_text())['generation'])
        self.assertEqual(2,publish_private_only(self.switch,self.private_root,2)['private_pointer_generation'])
        self.assertEqual(2,read_pinned(self.switch,self.private_root).generation)

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

    def test_recreated_pointer_lock_filename_does_not_bypass_pinned_selection(self):
        import os
        self.stage(1,0);self.stage(2,1)
        publish_private_only(self.switch,self.private_root,1)
        entered=threading.Event();release=threading.Event();publisher_waiting=threading.Event()
        actual=generation_module.verify_generation
        actual_flock=generation_module.fcntl.flock
        def hold_reader(base,allowed,generation):
            if threading.current_thread().name=='private-pinned-reader' and generation==1:
                entered.set()
                if not release.wait(timeout=8):raise RuntimeError('test reader release timeout')
            return actual(base,allowed,generation)
        def observe_flock(fd,op):
            if op==generation_module.fcntl.LOCK_EX and os.readlink('/proc/self/fd/'+str(fd))==str(self.switch/'generations'):
                publisher_waiting.set()
            return actual_flock(fd,op)
        with mock.patch.object(generation_module,'verify_generation',side_effect=hold_reader), \
             mock.patch.object(generation_module.fcntl,'flock',side_effect=observe_flock):
            reader_result=[];reader_errors=[]
            def reader():
                try:reader_result.append(read_pinned(self.switch,self.private_root))
                except Exception as exc:reader_errors.append(exc)
            t=threading.Thread(target=reader,name='private-pinned-reader')
            t.start()
            try:
                self.assertTrue(entered.wait(timeout=5))
                lock=self.switch/'.private-generation-pointer.lock'
                displaced=self.switch/'old-test-pointer-lock'
                lock.rename(displaced)
                fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pending=pool.submit(publish_private_only,self.switch,self.private_root,2)
                    try:
                        self.assertTrue(publisher_waiting.wait(timeout=5))
                        self.assertFalse(pending.done())
                        self.assertEqual(1,json.loads((self.switch/'active-private-generation.json').read_text())['generation'])
                    finally:release.set()
                    self.assertEqual(2,pending.result(timeout=10)['private_pointer_generation'])
            finally:
                release.set();t.join(timeout=10)
        self.assertFalse(t.is_alive())
        self.assertEqual([],reader_errors)
        self.assertEqual(1,reader_result[0].generation)
        self.assertEqual(2,read_pinned(self.switch,self.private_root).generation)
        self.assertTrue(displaced.exists())

    def test_abandoned_pid_named_pointer_temp_preserved(self):
        import os
        self.stage(1,0)
        orphan=self.switch/('.private-generation-pointer-new-'+str(os.getpid()))
        orphan.write_bytes(b'unknown prior private temporary bytes')
        self.assertEqual(1,publish_private_only(self.switch,self.private_root,1)['private_pointer_generation'])
        self.assertEqual(b'unknown prior private temporary bytes',orphan.read_bytes())

    def test_symlinked_generations_parent_refuses_before_opening_manifest(self):
        self.stage(1,0)
        moved=self.private_root/'other-generations'
        (self.switch/'generations').rename(moved)
        (self.switch/'generations').symlink_to(moved,target_is_directory=True)
        with mock.patch.object(generation_module,'_read_json_regular',side_effect=AssertionError('read escaped root')):
            with self.assertRaisesRegex(ValueError,'parent missing or symlinked'):
                verify_generation(self.switch,self.private_root,1)

    def test_sqlite_rollback_journal_and_dangling_wal_sidecars_refused(self):
        folder=self.stage(1,0)
        for suffix in ('-journal','-wal'):
            with self.subTest(suffix=suffix):
                sidecar=folder/('native.sqlite3'+suffix)
                sidecar.symlink_to(folder/'absent-test-only-sidecar')
                try:
                    with self.assertRaisesRegex(ValueError,'journal sidecars'):
                        verify_generation(self.switch,self.private_root,1)
                finally:sidecar.unlink()

    def _rebind_test_receipt(self,folder,role):
        # Only synthetic, deliberately inconsistent private QA fixtures.
        # Re-hash both receipt and manifest so failures cannot be ascribed
        # merely to an ordinary byte-digest mismatch.
        manifest_path=folder/'manifest.json'
        manifest=json.loads(manifest_path.read_text())
        asset=next(a for a in manifest['assets'] if a['kind']==role)
        path=folder/asset['path'];actual=sha_file(path)
        asset.update(sha256=actual,bytes=path.stat().st_size)
        proof_path=folder/'proof.json';proof=json.loads(proof_path.read_text())
        proof[role+'_sha256']=actual
        proof_path.write_text(json.dumps(proof))
        manifest['native_gate']['sha256']=sha_file(proof_path)
        manifest_path.write_text(json.dumps(manifest))

    def test_current_root_and_rebound_receipt_cannot_hide_missing_direct_vector(self):
        folder=self.stage(1,0)
        with sqlite3.connect(folder/'native.sqlite3') as db:
            db.execute("UPDATE content_chunks SET embedding_blob=NULL,embedding_sha256=NULL WHERE collection_id LIKE '%:overview'")
        self._rebind_test_receipt(folder,'native_sqlite')
        with self.assertRaisesRegex(ValueError,'direct-source vector'):
            publish_private_only(self.switch,self.private_root,1)
        self.assertFalse((self.switch/'active-private-generation.json').exists())

    def test_rebound_native_receipt_rejects_stale_span_and_summary_vector(self):
        for kind in ('span','summary'):
            with self.subTest(kind=kind):
                folder=self.stage(1 if kind=='span' else 2,0)
                with sqlite3.connect(folder/'native.sqlite3') as db:
                    if kind=='span':
                        db.execute("UPDATE content_chunks SET text_sha256=? WHERE collection_id LIKE '%:overview'",('a'*64,))
                    else:
                        bad=b'\x00'*768
                        db.execute("UPDATE content_representations SET embedding_blob=?,embedding_sha256=?",(bad,digest(bad)))
                self._rebind_test_receipt(folder,'native_sqlite')
                with self.assertRaisesRegex(ValueError,'wrong source span|summary vector'):
                    verify_generation(self.switch,self.private_root,1 if kind=='span' else 2)

    def test_rebound_source_map_receipt_rejects_incorrect_character_and_page_accounting(self):
        for generation,sql in ((1,'UPDATE originals SET text_characters=text_characters+1'),
                               (2,'UPDATE originals SET extracted_pages=extracted_pages+1,nonempty_pages=nonempty_pages+1')):
            with self.subTest(generation=generation):
                folder=self.stage(generation,0)
                with sqlite3.connect(folder/'source-map.sqlite3') as db:db.execute(sql)
                self._rebind_test_receipt(folder,'source_map_sqlite')
                with self.assertRaisesRegex(ValueError,'body character count|physical source-page identities'):
                    verify_generation(self.switch,self.private_root,generation)

    def test_rebound_receipt_rejects_unexpected_original_or_orphan_representation(self):
        for generation,sql in ((1,"INSERT INTO filesystem_entries VALUES(99999,'unexpected.pdf','file')"),
                               (2,"INSERT INTO content_representations(representation_id,chunk_id) VALUES('orphan','absent')")):
            with self.subTest(generation=generation):
                folder=self.stage(generation,0)
                with sqlite3.connect(folder/'native.sqlite3') as db:db.execute(sql)
                self._rebind_test_receipt(folder,'native_sqlite')
                with self.assertRaisesRegex(ValueError,'original file set differs|orphan or non-original'):
                    verify_generation(self.switch,self.private_root,generation)

    def test_current_vector_counts_independently_match_acceptance_receipt(self):
        folder=self.stage(1,0)
        r=verify_generation(self.switch,self.private_root,1)
        self.assertEqual(2,r['current_native_source_chunks_reverified'])
        self.assertEqual(1,r['current_native_summary_representations_reverified'])
        p=folder/'proof.json';proof=json.loads(p.read_text())
        proof['new_corrected_native_source_chunks_validated']+=1
        p.write_text(json.dumps(proof))
        self._rebind_test_receipt(folder,'native_sqlite')
        with self.assertRaisesRegex(ValueError,'native counts differ'):
            verify_generation(self.switch,self.private_root,1)

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
