"""Compare two privately captured, predeclared source-navigation snapshots.

This is *label recall*, never answer accuracy. Compare only matching frozen
questions, source revisions, and ranking/encoder contracts. A different
original or generated derivative revision is explicitly incomparable instead
of being misreported as ranking regression or improvement.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

SCHEMA='kmd.actual_current_same_read_snapshot_triple_channel_predeclared_gold_recall.v1'
REPORT_SCHEMA='kmd.frozen_two_era_source_navigation_rank_drift.v1'
CHANNELS=('original_passage','summary','lexical','rrf_two','rrf_three')
TOPS=(1,5,10,20,30)
SHA=re.compile(r'^[a-f0-9]{64}$')


def _checked_sha(value):
    if not isinstance(value,str) or not SHA.fullmatch(value):
        raise ValueError('invalid frozen source or query SHA256')
    return value


def _require_snapshot(doc):
    if not isinstance(doc,dict) or doc.get('schema')!=SCHEMA:
        raise ValueError('not a signed-off same-snapshot source navigation result')
    if not doc.get('all_search_channels_and_questions_use_one_captured_native_in_memory_embedding_set'):
        raise ValueError('native source channels did not use one recorded run snapshot')
    if not doc.get('nonproduction_original_chunk_ranking_excludes_original_analysis_facet_vectors'):
        raise ValueError('unrecognized native retrieval ranking source scope')
    gold=doc.get('frozen_gold_file_sha256')
    if not isinstance(gold,dict) or set(gold)!=set(('historical','independent','multiple','heldout')):
        raise ValueError('missing or changed predeclared gold files')
    for digest in gold.values():_checked_sha(digest)
    if not doc.get('gold_original_bytes',{}).get('all_frozen_known_original_revisions_match_current_source_map_and_download_ledger'):
        raise ValueError('frozen known original publisher revisions not independently current')
    count=doc.get('native_snapshot',{}).get('complete_original_paths')
    if type(count) is not int or count<1:
        raise ValueError('invalid native original snapshot size')
    cases=doc.get('cases')
    if not isinstance(cases,list) or not 1<=len(cases)<=150:
        raise ValueError('empty or unbounded source-known frozen labels')
    result={}
    for item in cases:
        if not isinstance(item,dict):raise ValueError('non-object retrieval case')
        id=item.get('id');group=item.get('group');family=item.get('family')
        if (not isinstance(id,str) or not 1<=len(id)<=120 or id in result
                or not isinstance(group,str) or not group
                or not isinstance(family,str) or not family):
            raise ValueError('duplicate or ambiguous frozen query identity')
        _checked_sha(item.get('query_sha256'))
        known=item.get('known_originals');ranks=item.get('label_ranks');source=item.get('source_revisions')
        if (not isinstance(known,list) or not 1<=len(known)<=6 or len(known)!=len(set(known))
                or not isinstance(ranks,dict) or set(ranks)!=set(CHANNELS)
                or not isinstance(source,dict) or set(source)!=set(known)):
            raise ValueError('unknown, duplicate or missing frozen original source labels')
        for path in known:
            if (not isinstance(path,str) or not path or path.startswith('/') or '\\' in path
                    or any(seg in ('','.','..') for seg in path.split('/'))
                    or not isinstance(source[path],dict)
                    or set(source[path])!=set(('original_sha256','derived_sha256'))):
                raise ValueError('invalid original and derived source revision identity')
            _checked_sha(source[path]['original_sha256']);_checked_sha(source[path]['derived_sha256'])
        for name,values in ranks.items():
            if not isinstance(values,list) or len(values)!=len(known):
                raise ValueError('mismatched source ranks in a search channel')
            if any(r is not None and (type(r) is not int or not 1<=r<=30) for r in values):
                raise ValueError('invalid or unbounded source label ranking')
        result[id]=item
    return result


def compare_frozen_source_ranks(old:dict,new:dict)->dict:
    """Compare attested label ranks; flag changed source revisions as incomparable."""
    before=_require_snapshot(old);after=_require_snapshot(new)
    if (old['frozen_gold_file_sha256']!=new['frozen_gold_file_sha256']
            or set(before)!=set(after)
            or old.get('query_encoder_revision')!=new.get('query_encoder_revision')
            or old.get('lexical_policy')!=new.get('lexical_policy')
            or old.get('rrf_policy')!=new.get('rrf_policy')):
        raise ValueError('different frozen query gold or retrieval policy; rank comparison prohibited')
    score={channel:{str(n):{'before_hits':0,'after_hits':0,'gains':0,'regressions':0}
                    for n in TOPS} for channel in CHANNELS}
    delta=[];excluded=[];families={}
    for id in sorted(before):
        older=before[id];newer=after[id]
        if (older['query_sha256']!=newer['query_sha256'] or older['known_originals']!=newer['known_originals']
            or older['group']!=newer['group'] or older['family']!=newer['family']):
            raise ValueError('frozen question or original-label identity changed')
        family=older['family']
        if family not in families:families[family]={'comparable_labels':0,'excluded_labels':0}
        for i,path in enumerate(older['known_originals']):
            if older['source_revisions'][path]!=newer['source_revisions'][path]:
                excluded.append({'id':id,'original_path':path,'reason':'source_or_derived_revision_changed'})
                families[family]['excluded_labels']+=1
                continue
            families[family]['comparable_labels']+=1
            item={'id':id,'family':family,'original_path':path,'rank_changed':{},
                  'top10_navigation_changes':{}}
            for channel in CHANNELS:
                previous=older['label_ranks'][channel][i]
                current=newer['label_ranks'][channel][i]
                if previous!=current:
                    item['rank_changed'][channel]={'before':previous,'after':current}
                for k in TOPS:
                    value=score[channel][str(k)]
                    was=previous is not None and previous<=k
                    now=current is not None and current<=k
                    value['before_hits']+=int(was)
                    value['after_hits']+=int(now)
                    value['gains']+=int(now and not was)
                    value['regressions']+=int(was and not now)
                    if k==10 and now!=was:
                        item['top10_navigation_changes'][channel]='gain' if now else 'regression'
            if item['rank_changed'] or item['top10_navigation_changes']:
                delta.append(item)
    return {'schema':REPORT_SCHEMA,
        'older_snapshot_utc':old.get('captured_utc'),
        'newer_snapshot_utc':new.get('captured_utc'),
        'older_complete_original_roots':old['native_snapshot']['complete_original_paths'],
        'newer_complete_original_roots':new['native_snapshot']['complete_original_paths'],
        'same_predeclared_gold_queries':len(before),
        'independently_rehashed_known_original_revisions_in_both_reports':True,
        'comparable_source_labels':sum(v['comparable_labels'] for v in families.values()),
        'incomparable_source_labels':len(excluded),
        'changed_source_labels_not_scored':excluded,
        'families':families,'channel_scores':score,
        'per_source_rank_changes':delta,
        'changed_rank_case_count':len({v['id'] for v in delta}),
        'identical_top10_3channel_source_label_recall':(
            score['rrf_three']['10']['gains']==0 and score['rrf_three']['10']['regressions']==0),
        'not_answer_accuracy_or_original_pdf_physical_page_truth':True,
        'not_proof_native_and_source_map_are_one_atomic_generation':True,
        'gold_labels_not_an_automatic_production_rank_change_authority':True}


def _digest(raw:bytes):return hashlib.sha256(raw).hexdigest()


def report_from_files(older:Path,newer:Path)->dict:
    for p in (older,newer):
        if (not isinstance(p,Path) or p.is_symlink() or not p.is_file()
                or not 100<=p.stat().st_size<=4_000_000):
            raise ValueError('missing or unbounded private frozen retrieval evidence')
    old_raw=older.read_bytes();new_raw=newer.read_bytes()
    report=compare_frozen_source_ranks(json.loads(old_raw),json.loads(new_raw))
    report['older_private_report_sha256']=_digest(old_raw)
    report['newer_private_report_sha256']=_digest(new_raw)
    return report
