"""Shard assignment mathematics and safety without opening original FileDB."""
from __future__ import annotations
from copy import deepcopy
import math
import unittest

from file_system_catalog.summary_shard_planner import compare_assignments


class SummaryShardAllocationTests(unittest.TestCase):
    def setUp(self):
        self.shards={
            0:{'pending':1155,'pending_characters':562638840},
            1:{'pending':2662,'pending_characters':636080803},
            2:{'pending':2603,'pending_characters':588098933},
            3:{'pending':2640,'pending_characters':638826286},
        }
        self.rates={'coder':652000.0,'q4':795000.0}
        self.current={'coder':[1,2,3],'q4':[0]}

    def evaluate(self, **kwargs):
        return compare_assignments(kwargs.get('shard_state',self.shards),
                                   chars_per_hour=kwargs.get('chars_per_hour',self.rates),
                                   current=kwargs.get('current',self.current))

    def test_all_shards_covered_once_in_every_selected_assignment(self):
        report=self.evaluate()
        self.assertEqual(14,report['all_disjoint_allocations_evaluated'])
        self.assertEqual(4,report['shard_count'])
        for name in ('current','best_over_all_disjoint_allocations','best_with_at_most_one_shard_reassigned'):
            assignment=report[name]
            self.assertIsNotNone(assignment)
            coder=set(assignment['coder_shards'])
            q4=set(assignment['q4_shards'])
            self.assertEqual(set(),coder&q4)
            self.assertEqual(set(range(4)),coder|q4)
            self.assertEqual(9060,assignment['coder_pending_originals']+assignment['q4_pending_originals'])
        self.assertFalse('restart' in report['schema'])

    def test_safe_lower_churn_alternative_reduces_sample_makespan(self):
        report=self.evaluate()
        original=report['current']['sensitivity_hours_at_observed_rates']['makespan']
        candidate=report['best_with_at_most_one_shard_reassigned']
        self.assertEqual([0,3],candidate['q4_shards'])
        self.assertEqual([1,2],candidate['coder_shards'])
        self.assertEqual(1,candidate['shards_changing_worker'])
        self.assertLess(candidate['sensitivity_hours_at_observed_rates']['makespan'],original)
        self.assertGreater(report['source_characters_remaining'],2_400_000_000)
        self.assertIn('Not an ETA',report['warning'])

    def test_baseline_preserves_observed_current_ownership(self):
        report=self.evaluate()
        old=report['current']
        self.assertEqual([0],old['q4_shards'])
        self.assertEqual([1,2,3],old['coder_shards'])
        self.assertEqual(0,old['shards_changing_worker'])
        self.assertEqual(1155,old['q4_pending_originals'])
        self.assertEqual(7905,old['coder_pending_originals'])
        self.assertEqual(1,report['best_with_at_most_one_shard_reassigned']['shards_changing_worker'])

    def test_invalid_load_rates_and_allocation_are_rejected(self):
        for rates in ({'coder':0,'q4':1},{'coder':math.nan,'q4':1},
                      {'coder':1,'q4':math.inf},{'coder':True,'q4':1},
                      {'coder':1},{'coder':1,'q4':2,'other':3}):
            with self.subTest(rates=rates),self.assertRaises(ValueError):
                self.evaluate(chars_per_hour=rates)
        for assignment in ({'coder':[1,2,3,0],'q4':[0]},
                           {'coder':[1,2],'q4':[0]},
                           {'coder':[0,1,2,3],'q4':[]},
                           {'coder':[1,2,3,3],'q4':[0]},
                           {'coder':['1',2,3],'q4':[0]}):
            with self.subTest(assignment=assignment),self.assertRaises(ValueError):
                self.evaluate(current=assignment)

    def test_invalid_shard_stats_are_rejected(self):
        for mutation in (
            lambda x:x.pop(2),
            lambda x:x[1].update(pending=-1),
            lambda x:x[2].update(pending_characters=3.0),
            lambda x:x[3].update(pending_characters=-50),
        ):
            data=deepcopy(self.shards);mutation(data)
            with self.assertRaises(ValueError):
                self.evaluate(shard_state=data)

    def test_two_shards_cannot_have_overlap_or_empty_owners(self):
        rates={'coder':1e6,'q4':1e6}
        shard={0:{'pending':2,'pending_characters':500},
               1:{'pending':2,'pending_characters':500}}
        cur={'coder':[0],'q4':[1]}
        report=self.evaluate(shard_state=shard,chars_per_hour=rates,current=cur)
        self.assertEqual(2,report['all_disjoint_allocations_evaluated'])
        self.assertEqual(0,report['best_with_at_most_one_shard_reassigned']['shards_changing_worker'])


if __name__=='__main__':
    unittest.main()
