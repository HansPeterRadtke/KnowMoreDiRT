"""No source access or model inference required for frozen candidate-recall audit."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[2] / 'scripts' / 'raspi_dual_channel_gold_audit.py'
spec = importlib.util.spec_from_file_location('raspi_dual_channel_gold_audit', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def example():
    return {'schema': 'kmd.read_only_raw_overview_vs_hierarchical_summary_shadow.v1',
            'cases': [
                {'id': 'a', 'family': 'technical', 'known_relevant_count': 2,
                 'known_originals': ['publisher/a', 'publisher/b'],
                 'raw_ranks': [1, 44], 'summary_ranks': [None, 2], 'fusion_ranks': [7, 8]},
                {'id': 'b', 'family': 'technical', 'known_relevant_count': 1,
                 'known_originals': ['publisher/c'],
                 'raw_ranks': [None], 'summary_ranks': [None], 'fusion_ranks': [None]},
            ]}


class FrozenDualChannelGoldAuditTests(unittest.TestCase):
    def test_union_keeps_both_channel_candidates_without_fusing(self):
        x = module.audit(example(), top=10)
        family = x['families']['technical']
        self.assertEqual((1, 1, 2, 2), (
            family['raw_hits'], family['summary_hits'], family['either_channel_hits'],
            family['fusion_hits']))
        self.assertEqual(1, family['queries_with_all_known_sources_in_either_channel'])
        self.assertEqual('publisher/c', family['missed_known_sources'][0]['original_path'])
        self.assertTrue(x['no_relevance_rank_fusion_applied'])

    def test_strict_input_rejects_ambiguous_labels_and_invalid_rank_types(self):
        for mutate in (
            lambda x: x['cases'][0].update(known_originals=['a', 'a']),
            lambda x: x['cases'][0].update(raw_ranks=[True, 4]),
            lambda x: x['cases'][0].update(summary_ranks=[0, 2]),
            lambda x: x['cases'].append(dict(x['cases'][0])),
            lambda x: x.update(schema='other'),
        ):
            with self.subTest(mutate=mutate):
                data = example()
                mutate(data)
                with self.assertRaises(ValueError):
                    module.audit(data)

    def test_top_parameter_changes_only_coverage_not_known_sources(self):
        data = example()
        data['cases'][0]['raw_ranks'][1] = 12
        at10 = module.audit(data, top=10)['totals']
        at20 = module.audit(data, top=20)['totals']
        self.assertEqual(at10['known_labels'], at20['known_labels'])
        self.assertGreaterEqual(at20['either_channel_hits'], at10['either_channel_hits'])
        for top in (0, 31, True):
            with self.assertRaises(ValueError):
                module.audit(data, top=top)


if __name__ == '__main__':
    unittest.main()
