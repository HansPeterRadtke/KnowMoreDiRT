"""Pure fixed source-rank evaluation and unmodified previous lexical rules."""
from __future__ import annotations
import hashlib
import unittest
import numpy as np
from file_system_catalog.frozen_same_snapshot_rank import (
    STOP,frozen_lexical_terms,rank_max_similarity,reciprocal_rank_fusion,label_coverage,verified_native_vector,
)


class FrozenSameSnapshotRankingTests(unittest.TestCase):
    def test_prior_frozen_lexical_policy_not_tuned_for_gold(self):
        q='The mesh needs to look bumpy under light without changing geometry, using values encoded into an RGB image.'
        self.assertEqual(['the','mesh','needs','look','bumpy','under','light','without','changing','geometry','values','encoded','into','rgb','image'],frozen_lexical_terms(q))
        self.assertEqual(['blender','texture','nodes'],frozen_lexical_terms('Blender texture nodes: Blender!'))

    def test_rank_uses_best_original_unit_without_duplicate_document_hits(self):
        a=np.zeros((3,384),dtype='<f4');a[0,0]=0.30;a[0,1]=0.95
        a[1,0]=1.0;a[2,0]=0.7;a[2,1]=.6
        q=np.zeros(384,dtype='<f4');q[0]=1.0
        out=rank_max_similarity(a,np.array([0,0,1]),q,['doc-a','doc-b'])
        self.assertEqual(['doc-a','doc-b'],out)
        self.assertEqual(out,rank_max_similarity(a,np.array([0,0,1]),q,['doc-a','doc-b']))

    def test_rrf_never_excludes_single_channel_unique_relevant_original(self):
        c={'original_passage':['a','b','c'],'summary':['b','x','y'],
           'lexical':['c','a','z']}
        self.assertEqual('a',reciprocal_rank_fusion(c)[0])  # tied RRF scores, stable path tie-break
        r=label_coverage(['a','z'],c,top=2)
        self.assertEqual(1,r['original_passage_hits'])
        self.assertEqual(1,r['lexical_hits'])
        self.assertEqual(1,r['either_channel_hits'])
        self.assertFalse(r['all_known_sources_found_by_some_channel'])

    def test_reject_duplicate_source_false_independent_vote(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            reciprocal_rank_fusion({'summary':['same','same']})
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            label_coverage(['x','x'],{'summary':['x']})

    def test_native_vector_requires_exact_model_revision_sha_and_finite_norm(self):
        original=np.zeros(384,dtype='<f2');original[0]=1;original[1]=1
        blob=original.tobytes();digest=hashlib.sha256(blob).hexdigest()
        base=dict(expected_model='native-e5',expected_revision='pinned-sha')
        def verify(**kw):
            arguments=dict(blob=blob,digest=digest,model='native-e5',
                           revision='pinned-sha',dimension=384,dtype='float16',**base)
            arguments.update(kw)
            return verified_native_vector(**arguments)
        vector=verify()
        self.assertAlmostEqual(1.0,float(np.linalg.norm(vector)),places=5)
        self.assertAlmostEqual(.7071067,float(vector[0]),places=4)
        self.assertIsNone(verify(digest='0'*64))
        self.assertIsNone(verify(model='wrong-private-model'))
        self.assertIsNone(verify(revision='wrong-revision'))
        self.assertIsNone(verify(dimension=512))
        self.assertIsNone(verify(dtype='float32'))
        self.assertIsNone(verify(blob=bytes(len(blob))))
        invalid=original.copy();invalid[0]=np.nan
        self.assertIsNone(verify(blob=invalid.tobytes(),digest=hashlib.sha256(invalid.tobytes()).hexdigest()))

    def test_ties_and_empty_or_invalid_channel_inputs_are_not_truth_scores(self):
        self.assertEqual(['a','b'],reciprocal_rank_fusion({'lexical':['a','b']}))
        with self.assertRaisesRegex(ValueError,'invalid frozen'):
            reciprocal_rank_fusion({'lexical':['a']},k=42)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            reciprocal_rank_fusion({'lexical':['a','a']})
        with self.assertRaisesRegex(ValueError,'invalid frozen'):
            reciprocal_rank_fusion({'arbitrary_trained_model':['a']})

    def test_fail_closed_wrong_model_embedding_dimension_and_nonfinite(self):
        v=np.zeros((1,384),dtype='<f4');q=np.zeros(384,dtype='<f4');q[0]=1
        with self.assertRaises(ValueError):rank_max_similarity(v,np.array([0]),np.zeros(384,dtype='<f4'),['a'])
        with self.assertRaises(ValueError):rank_max_similarity(v[:,:10],np.array([0]),q,['a'])
        v[0,1]=np.nan
        with self.assertRaises(ValueError):rank_max_similarity(v,np.array([0]),q,['a'])
        v[0,1]=0
        with self.assertRaises(ValueError):rank_max_similarity(v,np.array([2]),q,['a'])

if __name__=='__main__':unittest.main()
