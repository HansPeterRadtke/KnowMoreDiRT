"""Real SQLite fixtures: full publisher + recursive native source graph checks."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from file_system_catalog.recent_source_tree_audit import audit_current_source_tree


def sha(v:bytes)->str:
    return hashlib.sha256(v).hexdigest()


class RecursiveTreeOriginalProofTests(unittest.TestCase):
    def setUp(self):
        temporary=TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root=Path(temporary.name)
        self.original_root=root/'orig';self.original_root.mkdir()
        self.derivative_root=root/'text';self.derivative_root.mkdir()
        self.path='papers/original.html'
        orig=self.original_root/self.path;orig.parent.mkdir()
        orig.write_text('<html><p>Official publisher text, with independent original SHA.</p></html>')
        self.o_sha=sha(orig.read_bytes())
        self.dpath='papers/original.txt'
        der=self.derivative_root/self.dpath;der.parent.mkdir()
        self.text='[[SOURCE UNIT: HTML original publisher document]]\n'+('A measured original-source statement about shading and material parameters. ' * 6)
        der.write_text(self.text)
        self.d_sha=sha(der.read_bytes())
        self.receipts={self.path:self.o_sha}
        self.sm=sqlite3.connect(':memory:')
        self.db=sqlite3.connect(':memory:')
        self.addCleanup(self.sm.close);self.addCleanup(self.db.close)
        self.sm.execute('CREATE TABLE originals(source_path TEXT,source_sha256 TEXT,derived_sha256 TEXT,derived_path TEXT,status TEXT,original_bytes INTEGER)')
        self.sm.execute('INSERT INTO originals VALUES (?,?,?,?,?,?)',(self.path,self.o_sha,self.d_sha,self.dpath,'complete',orig.stat().st_size))
        self.db.execute('CREATE TABLE filesystem_entries(id INTEGER,relative_path_display TEXT,entry_type TEXT)')
        self.db.execute('INSERT INTO filesystem_entries VALUES(?,?,?)',(7,self.path,'file'))
        self.db.execute('CREATE TABLE content_chunks(chunk_id TEXT,filesystem_entry_id INTEGER,collection_id TEXT,start_char INTEGER,end_char INTEGER,text_sha256 TEXT,content_sha256 TEXT)')
        self.db.execute('CREATE TABLE content_representations(chunk_id TEXT,prompt_version TEXT,analysis_status TEXT,representation_kind TEXT,representation_text TEXT,representation_text_sha256 TEXT,embedding_blob BLOB,embedding_sha256 TEXT,generation_json TEXT,embedding_model TEXT,embedding_dimension INTEGER,embedding_dtype TEXT,embedding_norm REAL)')
        cut=len(self.text)//2
        self.add_node('leaf1',0,cut,'summary',0,[])
        self.add_node('leaf2',cut,len(self.text),'summary',0,[])
        self.add_node('root',0,len(self.text),'summary_long',1,['leaf1','leaf2'])

    def add_node(self,name,start,end,kind,level,children,*,origin=None,derived=None):
        text='Source-linked summary '+name
        embed=bytes([len(name)])*40
        g={'original_sha256':origin or self.o_sha,'derived_sha256':derived or self.d_sha,
           'source_tree_sha256':derived or self.d_sha,'source_start_char':start,'source_end_char':end,
           'source_passage_sha256':sha(self.text[start:end].encode()),'summary_sha256':sha(text.encode()),
           'node_id':name,'summary_level':level,'children':children}
        self.db.execute('INSERT INTO content_chunks VALUES (?,?,?,?,?,?,?)',(
            name,7,'native:summary',start,end,sha(self.text[start:end].encode()),origin or self.o_sha))
        self.db.execute('INSERT INTO content_representations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',(
            name,'kmd-hierarchical-source-summary-v1','complete',kind,text,sha(text.encode()),embed,sha(embed),json.dumps(g),
            'independent-test-embedding',20,'float16',1.0))

    def check(self,**options):
        return audit_current_source_tree(original_path=options.pop('original_path',self.path),
                                         source_map=self.sm,native_db=self.db,
                                         original_root=self.original_root,derivative_root=self.derivative_root,
                                         completed_receipts=options.pop('completed_receipts',self.receipts),
                                         **options)

    def update_generation(self,node,mutate):
        raw=self.db.execute('SELECT generation_json FROM content_representations WHERE chunk_id=?',(node,)).fetchone()[0]
        g=json.loads(raw);mutate(g)
        self.db.execute('UPDATE content_representations SET generation_json=? WHERE chunk_id=?',(json.dumps(g),node))

    def test_real_publisher_byte_shas_native_source_children_coverage_and_embeddings(self):
        result=self.check()
        self.assertEqual(self.o_sha,result['publisher_original_sha256_reverified'])
        self.assertEqual(self.d_sha,result['derived_original_sha256_reverified'])
        self.assertEqual(len(self.text),result['derived_chars_checked'])
        self.assertEqual(3,result['current_native_summary_nodes_fully_verified'])
        self.assertEqual(1,result['maximum_recursive_level'])
        self.assertTrue(result['all_original_text_covered_by_recursive_summary_graph'])
        self.assertTrue(result['all_summary_nodes_source_spans_and_embeddings_reverified'])
        self.assertTrue(result['no_index_or_source_mutation'])
        self.assertIn('not model-summary factual accuracy',result['truth_scope'])

    def test_original_or_derived_full_bytes_changed_fail_closed(self):
        orig=self.original_root/self.path
        orig.write_bytes(orig.read_bytes()+b'new unverified paragraph')
        with self.assertRaisesRegex(ValueError,'bytes differ'):
            self.check()
        # Restore correctly rather than accepting any unrelated revision.
        orig.write_text('<html><p>Official publisher text, with independent original SHA.</p></html>')
        derivative=self.derivative_root/self.dpath
        derivative.write_text(self.text+'unverified trailing text')
        with self.assertRaisesRegex(ValueError,'extracted text differs'):
            self.check()
        derivative.write_text(self.text)
        self.assertEqual(3,self.check()['current_native_summary_nodes_fully_verified'])

    def test_changed_source_receipt_and_missing_map_source_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'approved receipt'):
            self.check(completed_receipts={})
        self.sm.execute("UPDATE originals SET status='error'")
        with self.assertRaisesRegex(ValueError,'unique current completed'):
            self.check()

    def test_missing_child_or_disconnected_root_is_not_complete(self):
        self.db.execute("DELETE FROM content_representations WHERE chunk_id='leaf2'")
        with self.assertRaisesRegex(ValueError,'missing child'):
            self.check()
        self.db.execute("DELETE FROM content_chunks WHERE chunk_id='leaf2'")
        self.update_generation('root',lambda g:g.update(children=['leaf1']))
        with self.assertRaisesRegex(ValueError,'do not cover complete parent'):
            self.check()

    def test_wrong_subtree_order_or_gap_invalidates_root(self):
        self.update_generation('root',lambda g:g.update(children=['leaf2','leaf1']))
        with self.assertRaisesRegex(ValueError,'not in source order'):
            self.check()
        self.update_generation('root',lambda g:g.update(children=['leaf1','leaf1']))
        with self.assertRaisesRegex(ValueError,'invalid, duplicated'):
            self.check()

    def test_altered_offsets_and_source_span_sha_are_rejected(self):
        self.update_generation('leaf1',lambda g:g.update(source_end_char=g['source_end_char']-1))
        with self.assertRaisesRegex(ValueError,'source span'):
            self.check()
        self.update_generation('leaf1',lambda g:g.update(source_end_char=len(self.text)//2))
        self.db.execute("UPDATE content_chunks SET text_sha256=? WHERE chunk_id='leaf1'",('0'*64,))
        with self.assertRaisesRegex(ValueError,'source span'):
            self.check()

    def test_embedding_and_generated_summary_mutation_fail_closed(self):
        self.db.execute("UPDATE content_representations SET embedding_blob=? WHERE chunk_id='leaf1'",(b'evilbytes',))
        with self.assertRaisesRegex(ValueError,'embedding digest/dimension/dtype/norm'):
            self.check()
        self.db.execute("UPDATE content_representations SET embedding_blob=? WHERE chunk_id='leaf1'",(bytes([5])*40,))
        self.db.execute("UPDATE content_representations SET representation_text=? WHERE chunk_id='root'",('altered',))
        with self.assertRaisesRegex(ValueError,'native summary content'):
            self.check()

    def test_embedding_dimension_dtype_norm_and_model_must_match_blob(self):
        for field,value in (('embedding_dimension',33),('embedding_dtype','float32'),
                            ('embedding_norm',-1.0),('embedding_norm',float('inf')),
                            ('embedding_model','')):
            with self.subTest(field=field,value=value):
                old=self.db.execute(f'SELECT {field} FROM content_representations WHERE chunk_id=?',('leaf1',)).fetchone()[0]
                self.db.execute(f'UPDATE content_representations SET {field}=? WHERE chunk_id=?',(value,'leaf1'))
                with self.assertRaisesRegex(ValueError,'embedding digest/dimension/dtype/norm'):
                    self.check()
                self.db.execute(f'UPDATE content_representations SET {field}=? WHERE chunk_id=?',(old,'leaf1'))
        self.assertEqual(3,self.check()['current_native_summary_nodes_fully_verified'])

    def test_full_root_missing_or_truncated_fails(self):
        self.db.execute("DELETE FROM content_representations WHERE chunk_id='root'")
        with self.assertRaisesRegex(ValueError,'complete-span root'):
            self.check()

    def test_only_completed_source_current_revision_counted_and_stale_skipped(self):
        self.add_node('older',0,len(self.text),'summary_long',1,[],origin='f'*64)
        result=self.check()
        self.assertEqual(3,result['current_native_summary_nodes_fully_verified'])
        self.assertEqual(1,result['historical_revision_nodes_not_counted_as_current'])

    def test_explicit_extended_original_SHA_guard_preserves_default_derived_memory_bound(self):
        from file_system_catalog.recent_source_tree_audit import (
            MAX_SOURCE_BYTES,MAX_EXTENDED_SOURCE_BYTES,MAX_DERIVED_BYTES,
        )
        self.assertEqual(64*1024*1024,MAX_SOURCE_BYTES)
        self.assertEqual(256*1024*1024,MAX_EXTENDED_SOURCE_BYTES)
        self.assertEqual(64*1024*1024,MAX_DERIVED_BYTES)
        # A caller may deliberately hash a larger publisher PDF using a
        # streaming descriptor, but the *derived* source read_bytes remains
        # bounded at the previous strict 64-MiB limit.
        proven=self.check(max_original_bytes=128*1024*1024)
        self.assertEqual(self.o_sha,proven['publisher_original_sha256_reverified'])
        self.assertTrue(proven['all_original_text_covered_by_recursive_summary_graph'])
        for invalid in (True,0,MAX_EXTENDED_SOURCE_BYTES+1,MAX_EXTENDED_SOURCE_BYTES*10):
            with self.subTest(max_original_bytes=invalid),self.assertRaisesRegex(ValueError,'limits required'):
                self.check(max_original_bytes=invalid)
        for invalid in (True,0,MAX_DERIVED_BYTES+1):
            with self.subTest(max_derived_bytes=invalid),self.assertRaisesRegex(ValueError,'limits required'):
                self.check(max_derived_bytes=invalid)

    def test_unsafe_original_path_symlink_and_bounded_size(self):
        with self.assertRaises(ValueError):
            self.check(original_path='../original.html')
        with self.assertRaises(ValueError):
            self.check(max_original_bytes=10)
        with self.assertRaises(ValueError):
            self.check(max_derived_bytes=5)
        original=self.original_root/self.path
        target=self.original_root/'target.html';target.write_bytes(original.read_bytes())
        original.unlink();original.symlink_to(target)
        with self.assertRaisesRegex(ValueError,'symlink'):
            self.check()


    def test_read_only_current_root_claim_review_uses_verified_graph(self):
        from scripts.raspi_summary_claim_evidence import reviewed_native_root
        review = reviewed_native_root(
            original_path=self.path, source_map=self.sm, native_db=self.db,
            original_root=self.original_root, derived_root=self.derivative_root,
            completed_receipts=self.receipts,
        )
        self.assertEqual(self.o_sha,review["original_sha256"])
        self.assertEqual(self.d_sha,review["derived_sha256"])
        self.assertEqual("root",review["current_native_root_id"])
        self.assertEqual(3,review["verified_complete_native_source_tree"][
            "current_native_summary_nodes_fully_verified"])
        self.assertTrue(review["original_and_native_tree_checked_again_after_review"])
        self.assertEqual(0,review["candidate_review"]["review_candidates"])
        self.assertFalse(review["source_visual_figure_or_table_reviewed"])
        self.assertTrue(review["cannot_automatically_establish_factual_truth"])

    def test_current_root_claim_review_rejects_changed_original_revision(self):
        from scripts.raspi_summary_claim_evidence import reviewed_native_root
        (self.original_root/self.path).write_text("UNAPPROVED CURRENT ORIGINAL")
        with self.assertRaisesRegex(ValueError,"bytes differ"):
            reviewed_native_root(
                original_path=self.path, source_map=self.sm, native_db=self.db,
                original_root=self.original_root,
                derived_root=self.derivative_root,
                completed_receipts=self.receipts,
            )


    def test_read_only_all_native_summary_nodes_preserve_exact_source_spans(self):
        from scripts.raspi_summary_claim_evidence import reviewed_native_root
        import hashlib
        report = reviewed_native_root(
            original_path=self.path, source_map=self.sm, native_db=self.db,
            original_root=self.original_root, derived_root=self.derivative_root,
            completed_receipts=self.receipts, all_nodes=True,
        )
        nodes = report["all_native_node_claim_reviews"]
        self.assertEqual(3, len(nodes))
        self.assertEqual(0, report["all_native_node_review_candidate_count"])
        self.assertEqual([0,0,1], [n["level"] for n in nodes])
        self.assertEqual(["summary","summary","summary_long"],[n["kind"] for n in nodes])
        for node in nodes:
            start,end=node["original_source_start_char"],node["original_source_end_char"]
            self.assertEqual(hashlib.sha256(self.text[start:end].encode()).hexdigest(),
                             node["original_source_span_sha256"])
            self.assertEqual((start,end),(node["claim_evidence"]["reviewed_source_start_char"],
                                          node["claim_evidence"]["reviewed_source_end_char"]))
        with self.assertRaisesRegex(ValueError,"exceeds bounded review"):
            reviewed_native_root(
                original_path=self.path, source_map=self.sm, native_db=self.db,
                original_root=self.original_root, derived_root=self.derivative_root,
                completed_receipts=self.receipts, all_nodes=True, maximum_reviewed_nodes=2,
            )


    def test_claim_review_rechecks_original_if_source_changes_during_scan(self):
        from unittest.mock import patch
        from file_system_catalog.summary_claim_evidence import scan_summary_claims
        from scripts.raspi_summary_claim_evidence import reviewed_native_root
        def mutate_after_first_verified_read(**kwargs):
            (self.derivative_root/self.dpath).write_text(self.text+"new unapproved trailing content")
            return scan_summary_claims(**kwargs)
        with patch("scripts.raspi_summary_claim_evidence.scan_summary_claims",
                   side_effect=mutate_after_first_verified_read):
            with self.assertRaisesRegex(ValueError,"extracted text differs"):
                reviewed_native_root(
                    original_path=self.path,source_map=self.sm,native_db=self.db,
                    original_root=self.original_root,derived_root=self.derivative_root,
                    completed_receipts=self.receipts,
                )

    def test_claim_review_rechecks_native_summary_if_root_changes_during_scan(self):
        from unittest.mock import patch
        from file_system_catalog.summary_claim_evidence import scan_summary_claims
        from scripts.raspi_summary_claim_evidence import reviewed_native_root
        def mutate_native_root(**kwargs):
            self.db.execute("UPDATE content_representations SET representation_text=? "
                            "WHERE chunk_id='root'",("UNVERIFIED MODEL OUTPUT",))
            return scan_summary_claims(**kwargs)
        with patch("scripts.raspi_summary_claim_evidence.scan_summary_claims",
                   side_effect=mutate_native_root):
            with self.assertRaisesRegex(ValueError,"native summary content differs"):
                reviewed_native_root(
                    original_path=self.path,source_map=self.sm,native_db=self.db,
                    original_root=self.original_root,derived_root=self.derivative_root,
                    completed_receipts=self.receipts,
                )


if __name__=='__main__':
    unittest.main()
