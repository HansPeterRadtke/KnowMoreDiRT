"""Current source-known original SHA validation in isolated SQLite/byte fixtures."""
from __future__ import annotations
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from test_frozen_reference_budget import fixture
from file_system_catalog.frozen_reference_budget import compare_frozen_reference_budgets

SCRIPT=Path(__file__).resolve().parents[2]/'scripts'/'raspi_frozen_reference_budget_eval.py'
SPEC=importlib.util.spec_from_file_location('kmd_original_fixed_budget_cli_test',SCRIPT)
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def sha(raw):return hashlib.sha256(raw).hexdigest()


class PhysicalOriginalCandidateBudgetVerifierTests(unittest.TestCase):
    def setUp(self):
        tmp=TemporaryDirectory(prefix='kmd-frozen-quality-fixture-');self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name);self.original=self.root/'original';self.derived=self.root/'derived'
        self.original.mkdir();self.derived.mkdir();self.private=self.root/'private';self.private.mkdir()
        self.catalog=self.root/'catalog.json';self.ledger=self.root/'status.json'
        self.sm=self.root/'source.sqlite3';self.native=self.root/'native.sqlite3'
        self.contents={};self.derivatives={};self.sources={}
        self.n,self.h,self.md=fixture()
        gold={x for row in self.n['cases'] for x in row['known_originals']}
        parent=self.h['cases'][0]['approved_source_link_proofs'][0]['parent_original_path']
        original_paths=gold|{parent}
        for id,path in enumerate(sorted(original_paths),1):
            raw=('Original first-party technical reference: '+path+'\n').encode('utf8')
            derivative=('[[SOURCE UNIT: Original plain text]]\nThis source discusses '+path+'.\n').encode('utf8')
            orig=self.original/path;orig.parent.mkdir(parents=True,exist_ok=True);orig.write_bytes(raw)
            derived=self.derived/(path+'.txt');derived.parent.mkdir(parents=True,exist_ok=True);derived.write_bytes(derivative)
            record={'path':path,'url':'https://publisher.example/'+path,'kind':'text','title':path}
            receipt={'path':path,'url':record['url'],'sha256':sha(raw),'bytes_received':len(raw),'status':'complete'}
            self.sources[path]=(record,receipt)
            self.contents[path]=raw;self.derivatives[path]=derivative
        self.catalog.write_text(json.dumps({'sources':[pair[0] for pair in self.sources.values()]}))
        self.ledger.write_text(json.dumps({'items':[pair[1] for pair in self.sources.values()]}))
        for row in self.n['cases']:
            for path in row['known_originals']:
                row['source_revisions'][path]={
                    'original_sha256':sha(self.contents[path]),
                    'derived_sha256':sha(self.derivatives[path])}
        link=self.h['cases'][0]['approved_source_link_proofs'][0]
        link['source_link_original_SHA']=sha(self.contents[parent])
        link['target_original_receipt_SHA']=sha(self.contents[link['target_original_path']])
        with sqlite3.connect(self.sm) as db:
            db.execute('CREATE TABLE originals (source_path TEXT,source_sha256 TEXT,derived_sha256 TEXT,derived_path TEXT,status TEXT)')
            db.executemany('INSERT INTO originals VALUES (?,?,?,?,?)',[(p,sha(raw),sha(self.derivatives[p]),p+'.txt','complete') for p,raw in self.contents.items()])
        with sqlite3.connect(self.native) as db:
            db.execute('CREATE TABLE filesystem_entries (id INTEGER PRIMARY KEY, entry_type TEXT, relative_path_display TEXT)')
            db.executemany('INSERT INTO filesystem_entries VALUES (?,?,?)',[(i,'file',p) for i,p in enumerate(self.contents,1)])
        self.paths={'CORPUS':self.original,'DERIVED':self.derived,'SOURCE_MAP':self.sm,
                    'NATIVE':self.native,'CATALOG':self.catalog,'LEDGER':self.ledger,
                    'PRIVATE':self.private}
        self.patches=[mock.patch.object(M,key,p) for key,p in self.paths.items()]
        for patch in self.patches:patch.start();self.addCleanup(patch.stop)
        self.npath=self.private/'native.json'
        self.native_raw=(json.dumps(self.n,ensure_ascii=False)+'\n').encode('utf8')
        self.npath.write_bytes(self.native_raw)
        self.h['frozen_original_results_SHA256']=sha(self.native_raw)
        self.md['source_frozen_rank_report_sha256']=sha(self.native_raw)
        self.hpath=self.private/'HTML.json';self.hpath.write_text(json.dumps(self.h))
        self.mdpath=self.private/'Markdown.json';self.mdpath.write_text(json.dumps(self.md))
        self.result=compare_frozen_reference_budgets(self.n,self.h,self.md,
                    native_input_sha256=sha(self.native_raw))

    def test_entire_original_bytes_derivatives_and_native_IDs_verified_current(self):
        data=M.revalidate_current_source_paths(self.n,self.h,self.md,self.result)
        self.assertEqual(len(self.sources),data['current_full_source_publisher_and_derived_sha_verified'])
        self.assertEqual(len(self.sources),data['unique_source_IDs_verified'])
        self.assertTrue(data['current_catalog_ledger_source_map_and_native_IDs_rechecked_after_full_hashes'])
        self.assertTrue(data['cross_store_atomic_generation_reader_admission_not_claimed'])

    def test_end_to_end_private_one_shot_CLI_with_exact_no_source_mutation(self):
        output=self.private/'new-proof.json'
        args=['cli','--native',str(self.npath),'--html',str(self.hpath),
              '--markdown',str(self.mdpath),'--private-output',str(output)]
        with mock.patch.object(sys,'argv',args),mock.patch('builtins.print'):
            M.main()
        self.assertTrue(output.is_file())
        data=json.loads(output.read_text())
        self.assertEqual(73,data['frozen_original_questions'])
        self.assertEqual(len(self.sources),data['current_source_and_link_candidate_reverification']['current_full_source_publisher_and_derived_sha_verified'])
        self.assertEqual(10,data['group_scores']['all']['10']['maximum_source_candidate_slots_per_question'])
        self.assertFalse(data['live_KMD_native_ranking_changed'])
        self.assertTrue(data['no_source_download_or_model_inference'])
        self.assertFalse(data['original_publisher_or_derived_source_files_written'])

    def test_changed_publisher_original_bytes_after_Frozen_source_report_fail(self):
        path=next(iter(self.contents))
        (self.original/path).write_bytes(self.contents[path]+b'unauthorized-revision')
        with self.assertRaisesRegex(ValueError,'size'):
            M.revalidate_current_source_paths(self.n,self.h,self.md,self.result)

    def test_changed_source_derived_bytes_refuses_native_original_evidence(self):
        path=next(iter(self.contents))
        (self.derived/(path+'.txt')).write_bytes(self.derivatives[path]+b'other-content')
        with self.assertRaisesRegex(ValueError,'derived original contents'):
            M.revalidate_current_source_paths(self.n,self.h,self.md,self.result)

    def test_unapproved_OR_different_receipt_SHA_after_study_refused(self):
        rows=json.loads(self.ledger.read_text());rows['items'][0]['status']='pending'
        self.ledger.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError,'acquisition receipt'):
            M.revalidate_current_source_paths(self.n,self.h,self.md,self.result)

    def test_native_identity_removed_after_link_report_is_not_certified(self):
        with sqlite3.connect(self.native) as db:
            db.execute('DELETE FROM filesystem_entries WHERE id=1')
        with self.assertRaisesRegex(ValueError,'native FileDB identity not current|unique current native FileDB identity'):
            M.revalidate_current_source_paths(self.n,self.h,self.md,self.result)

    def test_current_source_map_revision_changed_after_frozen_benchmark_refused(self):
        with sqlite3.connect(self.sm) as db:
            db.execute("UPDATE originals SET derived_sha256=? WHERE source_path=?",('f'*64,next(iter(self.contents))))
        with self.assertRaisesRegex(ValueError,'derived original contents'):
            M.revalidate_current_source_paths(self.n,self.h,self.md,self.result)

    def test_symlinked_original_denied_without_following_external_file(self):
        path=next(iter(self.contents))
        f=self.original/path;f.unlink();f.symlink_to(self.ledger)
        with self.assertRaisesRegex(ValueError,'symlinked'):
            M.revalidate_current_source_paths(self.n,self.h,self.md,self.result)
        self.assertTrue(self.ledger.is_file())

    def test_mismatched_frozen_input_SHA_aborts_prior_to_original_read(self):
        self.h['frozen_original_results_SHA256']='b'*64
        with self.assertRaisesRegex(ValueError,'different frozen'):
            compare_frozen_reference_budgets(self.n,self.h,self.md,native_input_sha256=sha(self.native_raw))

    def test_unrelated_native_indexer_commit_does_not_stale_all_selected_current_originals(self):
        """Resident workers continually add OTHER documents while we hash."""
        real=M.sha_file;inserted=[False]
        def during_hash(path):
            result=real(path)
            if not inserted[0]:
                inserted[0]=True
                with sqlite3.connect(self.native) as db:
                    db.execute('INSERT INTO filesystem_entries VALUES (9999,?,?)',
                               ('file','unrelated/new-indexed-later-source.html'))
            return result
        with mock.patch.object(M,'sha_file',side_effect=during_hash):
            state=M.revalidate_current_source_paths(self.n,self.h,self.md,self.result)
        self.assertTrue(inserted[0])
        self.assertEqual(len(self.sources),state['current_full_source_publisher_and_derived_sha_verified'])
        self.assertTrue(state['current_catalog_ledger_source_map_and_native_IDs_rechecked_after_full_hashes'])

    def test_unrelated_source_extractor_commit_does_not_stale_other_completed_sources(self):
        real=M.sha_file;inserted=[False]
        def during_hash(path):
            result=real(path)
            if not inserted[0]:
                inserted[0]=True
                with sqlite3.connect(self.sm) as db:
                    db.execute('INSERT INTO originals VALUES (?,?,?,?,?)',
                        ('unrelated/just-extracted.md','c'*64,'d'*64,
                         'unrelated/just-extracted.md.txt','complete'))
            return result
        with mock.patch.object(M,'sha_file',side_effect=during_hash):
            state=M.revalidate_current_source_paths(self.n,self.h,self.md,self.result)
        self.assertTrue(inserted[0])
        self.assertEqual(len(self.sources),state['unique_source_IDs_verified'])

    def test_selected_original_native_ID_change_during_hash_is_still_refused(self):
        real=M.sha_file;changed=[False]
        def during_hash(path):
            result=real(path)
            if not changed[0]:
                changed[0]=True
                with sqlite3.connect(self.native) as db:
                    db.execute('UPDATE filesystem_entries SET id=6000 WHERE id=1')
            return result
        with mock.patch.object(M,'sha_file',side_effect=during_hash):
            with self.assertRaisesRegex(ValueError,'no longer current|native FileDB identity'):
                M.revalidate_current_source_paths(self.n,self.h,self.md,self.result)

    def test_selected_original_source_map_revision_change_during_hash_refused(self):
        real=M.sha_file;changed=[False]
        def during_hash(path):
            result=real(path)
            if not changed[0]:
                changed[0]=True
                with sqlite3.connect(self.sm) as db:
                    db.execute('UPDATE originals SET derived_sha256=? WHERE source_path=?',
                      ('e'*64,next(iter(self.contents))))
            return result
        with mock.patch.object(M,'sha_file',side_effect=during_hash):
            with self.assertRaisesRegex(ValueError,'no longer current|derived original contents'):
                M.revalidate_current_source_paths(self.n,self.h,self.md,self.result)

    def test_external_input_and_preexisting_private_output_path_are_refused(self):
        other=self.root/'foreign.json';other.write_text('{}')
        with self.assertRaisesRegex(ValueError,'bounded private'):
            M._load_input(other)
        out=self.private/'existing.json';out.write_text('not-overwritten')
        args=['cli','--native',str(self.npath),'--html',str(self.hpath),
              '--markdown',str(self.mdpath),'--private-output',str(out)]
        with mock.patch.object(sys,'argv',args),self.assertRaisesRegex(ValueError,'unsafe or already occupied'):
            M.main()
        self.assertEqual('not-overwritten',out.read_text())

if __name__=='__main__':unittest.main()
