#!/usr/bin/env python3
"""Read-only frozen-gold recall audit for *separate* original and summary ranks.

The shadow input is a candidate-vector experiment, not publisher-page proof.
The two-channel union counts a known source if either native channel finds it
within its own top-k; it does not merge scores or claim answer correctness.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

MODES = ('raw', 'summary', 'fusion')


def within(rank: object, top: int) -> bool:
    return type(rank) is int and 1 <= rank <= top


def audit(payload: dict, *, top: int = 10) -> dict:
    if type(top) is not int or not 1 <= top <= 30:
        raise ValueError('top must be an integer from 1 to 30')
    if payload.get('schema') != 'kmd.read_only_raw_overview_vs_hierarchical_summary_shadow.v1':
        raise ValueError('unrecognized frozen shadow evaluation schema')
    cases = payload.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('missing frozen cases')
    families: dict[str, dict] = {}
    seen = set()
    for row in cases:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not isinstance(row.get('family'), str):
            raise ValueError('invalid case identity')
        key = row['family'], row['id']
        if key in seen:
            raise ValueError('duplicated frozen question')
        seen.add(key)
        n = row.get('known_relevant_count')
        paths = row.get('known_originals')
        if type(n) is not int or n < 1 or not isinstance(paths, list) or len(paths) != n or len(set(paths)) != n:
            raise ValueError('ambiguous frozen source labels')
        ranks = {}
        for mode in MODES:
            value = row.get(mode + '_ranks')
            if not isinstance(value, list) or len(value) != n or any(r is not None and (type(r) is not int or r <= 0) for r in value):
                raise ValueError('invalid frozen rank array')
            ranks[mode] = value
        family = families.setdefault(row['family'], {
            'queries': 0, 'known_labels': 0, 'raw_hits': 0, 'summary_hits': 0,
            'fusion_hits': 0, 'either_channel_hits': 0, 'queries_with_all_known_sources_in_either_channel': 0,
            'missed_known_sources': [],
        })
        family['queries'] += 1
        family['known_labels'] += n
        all_found = True
        for index, path in enumerate(paths):
            flags = {mode: within(ranks[mode][index], top) for mode in MODES}
            for mode in MODES:
                family[mode + '_hits'] += int(flags[mode])
            union = flags['raw'] or flags['summary']
            family['either_channel_hits'] += int(union)
            if not union:
                all_found = False
                family['missed_known_sources'].append({
                    'question_id': row['id'], 'original_path': path,
                    'raw_rank': ranks['raw'][index], 'summary_rank': ranks['summary'][index],
                    'fusion_rank': ranks['fusion'][index],
                })
        family['queries_with_all_known_sources_in_either_channel'] += int(all_found)
    totals = {name: sum(f[name] for f in families.values()) for name in (
        'queries', 'known_labels', 'raw_hits', 'summary_hits', 'fusion_hits',
        'either_channel_hits', 'queries_with_all_known_sources_in_either_channel')}
    return {
        'schema': 'kmd.read_only_dual_channel_frozen_gold_audit.v1',
        'top_per_channel': top,
        'source_schema': payload['schema'],
        'source_gold_sha256': payload.get('gold_sha256'),
        'candidate_only_not_publisher_passage_verification': True,
        'no_relevance_rank_fusion_applied': True,
        'warning': 'Known relevant source recall is not complete relevance, passage truth, or answer correctness.',
        'families': families, 'totals': totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frozen_shadow', type=Path)
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--expected-sha256', help='Reject a changed frozen shadow JSON')
    args = parser.parse_args()
    if args.frozen_shadow.is_symlink() or not args.frozen_shadow.is_file() or args.frozen_shadow.stat().st_size > 5 * 1024 * 1024:
        raise ValueError('unsafe or oversized frozen shadow')
    data = args.frozen_shadow.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if args.expected_sha256 and args.expected_sha256 != digest:
        raise ValueError('frozen shadow digest changed')
    output = audit(json.loads(data), top=args.top)
    output['frozen_shadow_sha256'] = digest
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
