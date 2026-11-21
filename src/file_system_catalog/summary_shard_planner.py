"""Read-only, source-disjoint assignment comparison for recursive summary workers.

Estimated hours are sensitivity calculations from an observed throughput window,
not a reliable forecast or permission to modify services. No source, workers,
models, native FileDB or installed systemd units are changed by this module.
"""
from __future__ import annotations

from itertools import combinations
import math
from typing import Mapping


def compare_assignments(shard_state: Mapping[int,dict], *,
                        chars_per_hour: Mapping[str,float],
                        current: Mapping[str,list[int]]) -> dict:
    """Compare all source-disjoint two-worker allocations of existing hash shards.

    The number of original source characters is a cost proxy; measured model
    speed per source character may change sharply for longer documents.
    """
    if set(chars_per_hour) != {"coder","q4"} or any(
            not isinstance(v,(int,float)) or isinstance(v,bool)
            or not math.isfinite(v) or v<=0 for v in chars_per_hour.values()):
        raise ValueError("positive measured character throughput required for both workers")
    if (not shard_state or len(shard_state)>16
            or set(shard_state) != set(range(len(shard_state)))):
        raise ValueError("shard keys must cover one bounded contiguous partition")
    for shard,stats in shard_state.items():
        if (not isinstance(stats,dict)
                or type(stats.get('pending')) is not int or stats['pending']<0
                or type(stats.get('pending_characters')) is not int or stats['pending_characters']<0):
            raise ValueError('invalid source-map shard backlog')
    if set(current) != {"coder","q4"}:
        raise ValueError("require exactly the two current workers")
    all_shards=set(shard_state)
    for worker,assigned in current.items():
        if not isinstance(assigned,list) or len(assigned)!=len(set(assigned)):
            raise ValueError('current worker assignments duplicated')
        if not all(type(z) is int and z in all_shards for z in assigned):
            raise ValueError('invalid current worker shard')
    q4_old=set(current['q4']);coder_old=set(current['coder'])
    if not q4_old or not coder_old or q4_old & coder_old or q4_old|coder_old != all_shards:
        raise ValueError("current source shard coverage is not disjoint/complete")
    if len(all_shards)<2:
        raise ValueError("cannot assign one shard disjointly to two workers")

    def score(q4_shards: set[int]):
        coder_shards=all_shards-q4_shards
        if not q4_shards or not coder_shards:
            raise ValueError('empty model shard assignment')
        q4_chars=sum(shard_state[z]['pending_characters'] for z in q4_shards)
        coder_chars=sum(shard_state[z]['pending_characters'] for z in coder_shards)
        q4_docs=sum(shard_state[z]['pending'] for z in q4_shards)
        coder_docs=sum(shard_state[z]['pending'] for z in coder_shards)
        times={'coder':coder_chars/chars_per_hour['coder'],
               'q4':q4_chars/chars_per_hour['q4']}
        return {
            'coder_shards':sorted(coder_shards),'q4_shards':sorted(q4_shards),
            'coder_pending_originals':coder_docs,'q4_pending_originals':q4_docs,
            'coder_pending_source_characters':coder_chars,
            'q4_pending_source_characters':q4_chars,
            'shards_changing_worker':len(q4_shards^q4_old),
            'sensitivity_hours_at_observed_rates':{
                'coder':round(times['coder'],2),'q4':round(times['q4'],2),
                'makespan':round(max(times.values()),2),
            },
        }
    baseline=score(q4_old)
    options=[]
    for n in range(1,len(all_shards)):
        for subset in combinations(sorted(all_shards),n):
            options.append(score(set(subset)))
    # Favor minimum churn if projected times are exactly tied.
    sort_key=lambda x:(x['sensitivity_hours_at_observed_rates']['makespan'],
                        x['shards_changing_worker'],x['q4_shards'])
    best=min(options,key=sort_key)
    # Relocating only one shard is an operationally simpler boundary handoff,
    # not an authorization to issue signals or edit live systemd units.
    least_churn=min((x for x in options if x['shards_changing_worker']<=1),
                    key=sort_key,default=None)
    return {
        'schema':'kmd.read_only_source_shard_sensitivity.v1',
        'current':baseline,'best_over_all_disjoint_allocations':best,
        'best_with_at_most_one_shard_reassigned':least_churn,
        'all_disjoint_allocations_evaluated':len(options),
        'shard_count':len(all_shards),'originals_remaining':sum(x['pending'] for x in shard_state.values()),
        'source_characters_remaining':sum(x['pending_characters'] for x in shard_state.values()),
        'rate_assumption_chars_per_hour':dict(chars_per_hour),
        'warning':'Not an ETA or validated live performance gain: model availability, source length/tokenization, quality, active document completion, and shared-slot contention vary. No live assignment changed.',
    }
