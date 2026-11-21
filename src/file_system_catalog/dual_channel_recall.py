"""Read-only exact-source recall for independently attested multi-source gold.

This evaluator only counts predeclared exact original identities. Publisher
source and derivative bytes must be independently attested by the caller.
A verified candidate is a navigation hit, not an answer or ground truth proof.
"""
from __future__ import annotations

import hashlib

from .retrieval_regression import validated_gold


def evaluate_verified_dual_case(case: dict, payload: dict, *, top: int) -> dict:
    if type(top) is not int or not 1 <= top <= 20:
        raise ValueError('invalid dual source recall top-k')
    gold=validated_gold([case])[0]
    query=gold['query']
    if (not isinstance(payload,dict)
            or payload.get('schema')!='kmd.two_channel_original_discovery.v1'
            or payload.get('query')!=query
            or payload.get('no_database_or_original_mutation') is not True
            or payload.get('no_ranking_fusion_or_score_threshold_applied') is not True):
        raise ValueError('native two-channel source/revision contract changed')
    seen={}
    for key,ordinal,name in (
        ('summary_channel','summary_rank','summary'),
        ('original_passage_channel','original_source_rank','original'),
    ):
        entries=payload.get(key)
        if not isinstance(entries,list) or len(entries)>top:
            raise ValueError('unbounded native source channel')
        ranks={}
        for index,record in enumerate(entries,1):
            if (not isinstance(record,dict)
                    or record.get('publisher_revision_verified') is not True
                    or record.get(ordinal)!=index):
                raise ValueError('ranked source is not current and source-provenance-verified')
            path=record.get('original_path')
            if not isinstance(path,str) or not path or path in ranks:
                raise ValueError('duplicate or invalid native publisher original')
            if key=='original_passage_channel':
                preview=record.get('passage_preview')
                digest=record.get('passage_preview_sha256')
                if (not isinstance(preview,str) or not preview
                        or not isinstance(digest,str)
                        or hashlib.sha256(preview.encode()).hexdigest()!=digest):
                    raise ValueError('displayed original excerpt is missing its exact SHA')
            ranks[path]=index
        seen[name]=ranks
    label_rows=[]
    for path in gold['relevant_paths']:
        r=seen['original'].get(path)
        s=seen['summary'].get(path)
        label_rows.append({'known_original_path':path,
                           'original_passage_rank':r,
                           'hierarchical_summary_rank':s,
                           'raw_topk':r is not None,
                           'summary_topk':s is not None,
                           'either_topk':r is not None or s is not None})
    return {
        'id':gold['id'],
        'category':gold['category'],
        'query_sha256':hashlib.sha256(query.encode()).hexdigest(),
        'known_labels':len(label_rows),
        'raw_topk_hits':sum(r['raw_topk'] for r in label_rows),
        'summary_topk_hits':sum(r['summary_topk'] for r in label_rows),
        'either_channel_topk_hits':sum(r['either_topk'] for r in label_rows),
        'complete_known_labels_in_either_channel':all(r['either_topk'] for r in label_rows),
        'known_original_ranks':label_rows,
        'native_source_summary_candidates':len(seen['summary']),
        'native_original_passage_candidates':len(seen['original']),
        'uncitable_passages_omitted':payload.get('uncitable_native_passage_candidates_omitted',0),
    }


def aggregate_verified_dual_cases(rows: list[dict], *, top: int) -> dict:
    if (not isinstance(rows,list) or not 1<=len(rows)<=200 or
            type(top) is not int or not 1<=top<=20):
        raise ValueError('invalid bounded two-channel evaluation set')
    ids=set()
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get('id'),str) or row['id'] in ids:
            raise ValueError('duplicate case in source-label recall')
        ids.add(row['id'])
        count=row.get('known_labels')
        for field in ('raw_topk_hits','summary_topk_hits','either_channel_topk_hits'):
            value=row.get(field)
            if type(value) is not int or type(count) is not int or not 0<=value<=count or count<1:
                raise ValueError('untrusted or inconsistent source-label hit counts')
        if (row['either_channel_topk_hits']<row['raw_topk_hits']
                or row['either_channel_topk_hits']<row['summary_topk_hits']):
            raise ValueError('union recall is inconsistent with its native channels')
    labels=sum(x['known_labels'] for x in rows)
    return {
        'schema':'kmd.attested_two_channel_source_recall.v1',
        'top_per_channel':top,'cases':len(rows),'known_source_labels':labels,
        'raw_source_hits':sum(x['raw_topk_hits'] for x in rows),
        'summary_source_hits':sum(x['summary_topk_hits'] for x in rows),
        'either_channel_source_hits':sum(x['either_channel_topk_hits'] for x in rows),
        'cases_with_all_known_sources_in_either_channel':sum(x['complete_known_labels_in_either_channel'] for x in rows),
        'raw_source_recall':sum(x['raw_topk_hits'] for x in rows)/labels,
        'summary_source_recall':sum(x['summary_topk_hits'] for x in rows)/labels,
        'either_source_recall':sum(x['either_channel_topk_hits'] for x in rows)/labels,
        'notes':'Exact predeclared known original labels only. Not exhaustive relevance, answer correctness, or calibrated confidence. Raw and summary rank lists stay separate; this is set union only, not rank fusion.',
    }
