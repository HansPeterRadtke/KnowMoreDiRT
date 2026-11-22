"""Release may only accept fully regenerated originals AND all hierarchy vectors."""
from __future__ import annotations
from dataclasses import asdict
import hashlib,json,sqlite3
from pathlib import Path
import unittest
import test_source_page_forward_rehearsal as forward_fixture
from file_system_catalog.hierarchical_summary import SummaryNode,_VERSION
from file_system_catalog.source_page_reindex_release_gate import assess_corrected_native_release
from file_system_catalog.summary_coverage import revision_identity


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


class NativeReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.fixture=forward_fixture.PrivateFullForwardTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.do_forward()
        self.db=self.fixture.output/'private_after_transition_not_search_ready.sqlite3'
        self.root=self.fixture.output/'corrected_source_text'
        self.summaries=self.fixture.root/'new-revision-fully-generated-trees'
        self.summaries.mkdir()
        self._add_native_columns_for_minimal_rehearsal_fixture()
        with sqlite3.connect(self.fixture.sqlite) as baseline:
            baseline.execute('ALTER TABLE content_chunks ADD COLUMN collection_id TEXT')
            baseline.execute('UPDATE content_chunks SET collection_id=?',
                             ('raspi-original-books-semantic-content-v1:overview',))
            baseline.execute('ALTER TABLE content_representations ADD COLUMN representation_kind TEXT')
            baseline.execute("UPDATE content_representations SET representation_kind='summary_long'")

    def _add_native_columns_for_minimal_rehearsal_fixture(self):
        # The source-page shadow's minimal SQL fixture deliberately excludes
        # production-native summary columns. Add the real read fields once.
        with sqlite3.connect(self.db) as conn:
            for table,extra in (
                ('content_chunks',{'collection_id':'TEXT','content_sha256':'TEXT',
                  'content_object_id':'TEXT','start_char':'INTEGER','end_char':'INTEGER',
                  'text_sha256':'TEXT','embedding_sha256':'TEXT',
                  'embedding_dimension':'INTEGER','embedding_dtype':'TEXT',
                  'embedding_model':'TEXT','embedding_model_revision':'TEXT'}),
                ('content_representations',{'representation_kind':'TEXT','representation_text':'TEXT',
                  'representation_text_sha256':'TEXT','generation_json':'TEXT',
                  'embedding_sha256':'TEXT','embedding_dimension':'INTEGER',
                  'embedding_dtype':'TEXT','embedding_model':'TEXT',
                  'embedding_model_revision':'TEXT','prompt_version':'TEXT','analysis_status':'TEXT'})):
                for column,decl in extra.items():
                    conn.execute(f'ALTER TABLE {table} ADD COLUMN {column} {decl}')
            conn.commit()

    def check(self):
        with sqlite3.connect(self.db.resolve().as_uri()+'?mode=ro',uri=True) as source:
            source.execute('PRAGMA query_only=ON')
            with sqlite3.connect(self.fixture.sqlite.resolve().as_uri()+'?mode=ro',uri=True) as historical:
                historical.execute('PRAGMA query_only=ON')
                return assess_corrected_native_release(records=self.fixture.records,
                    source_map=source,native=source,historical=historical,
                    derived_root=self.root,summary_root=self.summaries,expected_count=2)

    def build_all_correct_revisions(self):
        with sqlite3.connect(self.db) as conn:
            for num,receipt in enumerate(self.fixture.records,1):
                path=receipt['original_path'];original=receipt['original_sha256']
                derived=receipt['new_staged_derived_sha256'];ident=receipt['native_original_filedb_id']
                text=(self.root/(path+'.txt')).read_text()
                summary='Accurate, complete summary for original publisher paper '+str(num)
                node=SummaryNode(node_id='summary:whole_'+str(num),
                     level=0,start_char=0,end_char=len(text),source_label='whole PDF',
                     summary=summary,summary_sha256=digest(summary.encode()),child_ids=(),
                     source_sha256=derived,prompt_tokens=100,model_context_tokens=4096,
                     allocated_output_tokens=200,prompt_version=_VERSION,
                     prompt_policy='source_grounded',native_analysis=None)
                tree_name=(Path(path).name+'.'+revision_identity(path,original,derived)[:24]+'.summary-tree.json')
                tree={'schema':'kmd-complete-hierarchical-summary-tree-v1',
                      'original_path':path,'original_sha256':original,'derived_sha256':derived,
                      'tree':{'source_sha256':derived,'original_characters':len(text),
                              'root_id':node.node_id,'nodes':[asdict(node)],'leaf_sections':2}}
                (self.summaries/tree_name).write_text(json.dumps(tree))
                conn.execute('UPDATE originals SET status=? WHERE source_path=?',('complete',path))
                vector=b'\x00\x3c'+b'\x00\x00'*383  # normalized little-endian float16
                for chunk_id,category in ((f'overview{num}','overview'),(f'summary{num}','summary')):
                    blob=vector if category=='overview' else None
                    conn.execute('''INSERT INTO content_chunks
                      (chunk_id,filesystem_entry_id,embedding_blob,collection_id,content_sha256,
                       content_object_id,start_char,end_char,text_sha256,embedding_sha256,
                       embedding_dimension,embedding_dtype,embedding_model,embedding_model_revision)
                      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                       (chunk_id,ident,blob,'raspi-original-books-semantic-content-v1:'+category,
                        original,'sha256:'+derived,0,len(text),digest(text.encode()),
                        digest(vector) if blob else None,384 if blob else None,'float16' if blob else None,
                        'fixture-e5' if blob else None,'fixture-revision' if blob else None))
                meta={'original_sha256':original,'derived_sha256':derived,
                      'source_tree_sha256':derived,'source_start_char':0,
                      'source_end_char':len(text),'source_passage_sha256':digest(text.encode()),
                      'node_id':node.node_id,'summary_sha256':node.summary_sha256,
                      'summary_level':0}
                conn.execute('''INSERT INTO content_representations
                   (representation_id,chunk_id,embedding_blob,representation_kind,representation_text,
                    representation_text_sha256,generation_json,embedding_sha256,embedding_dimension,
                    embedding_dtype,embedding_model,embedding_model_revision,prompt_version,analysis_status)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                    (f'rep_new_{num}',f'summary{num}',vector,'summary_long',summary,
                     node.summary_sha256,json.dumps(meta),digest(vector),384,'float16',
                     'fixture-e5','fixture-revision',_VERSION,'complete'))
            conn.commit()

    def test_privately_staged_source_is_not_searchable(self):
        result=self.check()
        self.assertEqual(2,result['originals_with_rechecked_source_derivative_sha'])
        self.assertEqual(0,result['originals_with_native_full_source_roots_and_vectors'])
        self.assertFalse(result['full_corpus_new_revision_search_ready'])
        self.assertFalse(result['safe_to_cutover_live'])
        self.assertEqual(2,result['failures_by_reason']['source-map corrected revision is not fully published and complete'])

    def test_all_corrected_source_vectors_and_full_recursion_are_required(self):
        self.build_all_correct_revisions()
        result=self.check()
        self.assertEqual(2,result['originals_with_native_full_source_roots_and_vectors'])
        self.assertEqual(2,result['originals_with_verified_complete_recursion_tree'])
        self.assertEqual(4,result['new_corrected_native_source_chunks_validated'])
        self.assertEqual(2,result['new_corrected_native_summary_representations_validated'])
        self.assertTrue(result['full_corpus_new_revision_search_ready'])
        self.assertFalse(result['safe_to_cutover_live'])
        self.assertEqual({},result['failures_by_reason'])

    def test_old_or_missing_native_source_overview_rejected(self):
        self.build_all_correct_revisions()
        with sqlite3.connect(self.db) as db:
            db.execute("DELETE FROM content_chunks WHERE chunk_id='overview1'")
        result=self.check()
        self.assertFalse(result['full_corpus_new_revision_search_ready'])
        self.assertEqual(1,result['originals_with_native_full_source_roots_and_vectors'])

    def test_historically_present_full_content_channel_must_be_rebuilt(self):
        # A current root alone must not silently delete a previously searchable
        # full-text passage channel from these original publisher PDFs.
        self.build_all_correct_revisions()
        with sqlite3.connect(self.fixture.sqlite) as old:
            old.execute('INSERT INTO content_chunks (chunk_id,filesystem_entry_id,embedding_blob,collection_id) VALUES(?,?,?,?)',
                        ('historical_full_passage_1',self.fixture.records[0]['native_original_filedb_id'],b'digest',
                         'raspi-original-books-semantic-content-v1:full'))
        rejected=self.check()
        self.assertFalse(rejected['full_corpus_new_revision_search_ready'])
        self.assertEqual(1,rejected['originals_with_native_full_source_roots_and_vectors'])
        receipt=self.fixture.records[0]
        text=(self.root/(receipt['original_path']+'.txt')).read_text()
        vector=b'\x00\x3c'+b'\x00\x00'*383
        with sqlite3.connect(self.db) as conn:
            conn.execute('''INSERT INTO content_chunks (chunk_id,filesystem_entry_id,embedding_blob,
               collection_id,content_sha256,content_object_id,start_char,end_char,
               text_sha256,embedding_sha256,embedding_dimension,embedding_dtype,
               embedding_model,embedding_model_revision)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
              ('full1',receipt['native_original_filedb_id'],vector,'raspi-original-books-semantic-content-v1:full',
               receipt['original_sha256'],'sha256:'+receipt['new_staged_derived_sha256'],
               0,len(text),digest(text.encode()),digest(vector),384,'float16',
               'fixture-e5','fixture-revision'))
        accepted=self.check()
        self.assertTrue(accepted['full_corpus_new_revision_search_ready'])
        self.assertFalse(accepted['safe_to_cutover_live'])

    def test_all_facets_must_match_corrected_revision_even_if_root_is_current(self):
        self.build_all_correct_revisions()
        with sqlite3.connect(self.db) as conn:
            previous=conn.execute('SELECT generation_json,embedding_blob FROM content_representations WHERE representation_id=?',
                                  ('rep_new_1',)).fetchone()
            gen=json.loads(previous[0]);gen['derived_sha256']='a'*64
            keyword='stale publisher keyword'
            conn.execute('''INSERT INTO content_representations (
               representation_id,chunk_id,embedding_blob,representation_kind,representation_text,
               representation_text_sha256,generation_json,embedding_sha256,
               embedding_dimension,embedding_dtype,embedding_model,embedding_model_revision,
               prompt_version,analysis_status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
              ('invalid_facet','summary1',previous[1],'keyword',keyword,digest(keyword.encode()),
               json.dumps(gen),digest(previous[1]),384,'float16',
               'fixture-e5','fixture-revision',_VERSION,'complete'))
        result=self.check()
        self.assertFalse(result['full_corpus_new_revision_search_ready'])
        self.assertEqual(1,result['originals_with_native_full_source_roots_and_vectors'])

    def test_original_historical_keyword_search_channel_must_not_silently_disappear(self):
        self.build_all_correct_revisions()
        original=self.fixture.records[0]
        old_ident=original['native_original_filedb_id']
        with sqlite3.connect(self.fixture.sqlite) as historical:
            old_chunk=historical.execute(
                'SELECT chunk_id FROM content_chunks WHERE filesystem_entry_id=? LIMIT 1',
                (old_ident,)).fetchone()[0]
            historical.execute('INSERT INTO content_representations '
                               '(representation_id,chunk_id,embedding_blob,representation_kind) '
                               'VALUES(?,?,?,?)',
                               ('old_keyword_term',old_chunk,b'digest','keyword'))
        rejected=self.check()
        self.assertEqual(1,rejected['originals_with_native_full_source_roots_and_vectors'])
        self.assertFalse(rejected['full_corpus_new_revision_search_ready'])
        self.assertTrue(any('keyword' in k for k in rejected['failures_by_reason']))
        with sqlite3.connect(self.db) as new:
            previous=new.execute('SELECT generation_json,embedding_blob '
                     'FROM content_representations WHERE representation_id=?',
                     ('rep_new_1',)).fetchone()
            provenance=json.loads(previous[0]);provenance['native_facet_kind']='keyphrase'
            keyphrase='corrected original physical PDF publisher page'
            new.execute('''INSERT INTO content_representations (
                representation_id,chunk_id,embedding_blob,representation_kind,representation_text,
                representation_text_sha256,generation_json,embedding_sha256,embedding_dimension,
                embedding_dtype,embedding_model,embedding_model_revision,
                prompt_version,analysis_status)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                ('new_keyword_term','summary1',previous[1],'keyphrase',keyphrase,
                 digest(keyphrase.encode()),json.dumps(provenance),digest(previous[1]),
                 384,'float16','fixture-e5','fixture-revision',_VERSION,'complete'))
        accepted=self.check()
        self.assertTrue(accepted['full_corpus_new_revision_search_ready'])
        self.assertFalse(accepted['safe_to_cutover_live'])

    def test_source_chunk_excerpt_stale_from_previous_derived_revision_rejected(self):
        self.build_all_correct_revisions()
        with sqlite3.connect(self.db) as db:
            db.execute("UPDATE content_chunks SET text_sha256=? WHERE chunk_id='overview1'",('a'*64,))
        self.assertFalse(self.check()['full_corpus_new_revision_search_ready'])

    def test_zero_native_vector_with_matching_sha_is_not_semantic_ready(self):
        self.build_all_correct_revisions()
        zero=b'\x00\x00'*384
        with sqlite3.connect(self.db) as db:
            db.execute('UPDATE content_representations SET embedding_blob=?,embedding_sha256=? WHERE representation_id=?',
                       (zero,digest(zero),'rep_new_1'))
        self.assertFalse(self.check()['full_corpus_new_revision_search_ready'])

    def test_nonfinite_native_vector_with_matching_sha_is_not_semantic_ready(self):
        self.build_all_correct_revisions()
        # little-endian float16 canonical NaN at first position
        nan=b'\x00\x7e'+b'\x00\x00'*383
        with sqlite3.connect(self.db) as db:
            db.execute('UPDATE content_chunks SET embedding_blob=?,embedding_sha256=? WHERE chunk_id=?',
                       (nan,digest(nan),'overview1'))
        self.assertFalse(self.check()['full_corpus_new_revision_search_ready'])

    def test_summary_native_blob_sha_tamper_is_rejected(self):
        self.build_all_correct_revisions()
        with sqlite3.connect(self.db) as db:
            db.execute("UPDATE content_representations SET embedding_sha256=? WHERE representation_id='rep_new_1'",('0'*64,))
        self.assertFalse(self.check()['full_corpus_new_revision_search_ready'])

    def test_corrected_source_map_old_revision_is_rejected(self):
        self.build_all_correct_revisions()
        with sqlite3.connect(self.db) as db:
            db.execute("UPDATE originals SET derived_sha256=? WHERE source_path=?",('b'*64,self.fixture.records[0]['original_path']))
        self.assertFalse(self.check()['full_corpus_new_revision_search_ready'])

    def test_native_summary_tree_old_revision_or_missing_node_rejected(self):
        self.build_all_correct_revisions()
        tree=next(self.summaries.glob('*paper-1.*.summary-tree.json'))
        obj=json.loads(tree.read_text())
        obj['derived_sha256']='c'*64
        tree.write_text(json.dumps(obj))
        self.assertFalse(self.check()['full_corpus_new_revision_search_ready'])

    def test_current_summary_tree_source_span_must_cover_complete_text(self):
        self.build_all_correct_revisions()
        tree=next(self.summaries.glob('*paper-1.*.summary-tree.json'))
        obj=json.loads(tree.read_text())
        obj['tree']['nodes'][0]['end_char']=5
        tree.write_text(json.dumps(obj))
        self.assertFalse(self.check()['full_corpus_new_revision_search_ready'])

    def test_staged_original_sha_mismatch_rejected(self):
        receipt=self.fixture.records[0]
        target=self.root/(receipt['original_path']+'.txt')
        target.write_bytes(target.read_bytes()+b'not-the-same')
        self.assertFalse(self.check()['full_corpus_new_revision_search_ready'])

    def test_invalid_reader_write_permissions_rejected(self):
        with sqlite3.connect(self.db) as db:
            with self.assertRaisesRegex(ValueError,'query-only'):
                assess_corrected_native_release(records=self.fixture.records,
                    source_map=db,native=db,historical=db,derived_root=self.root,
                    summary_root=self.summaries,expected_count=2)

if __name__=='__main__':unittest.main()
