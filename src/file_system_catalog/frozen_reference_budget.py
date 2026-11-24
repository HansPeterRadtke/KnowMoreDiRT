"""Source-known, fixed-slot, SHA-scoped *navigation* benchmark.

An optional publisher link is not a better source merely because it links to
an original, and adding it on top of the normal ten results is NOT a fair
comparison. This pure evaluator verifies previous immutable source-label
identity and computes the precise known-label gains AND losses for a fixed
maximum ten candidate slots, without modifying native ranking or source data.
Not a factual answer/edition/page verifier and not a production reranker.
"""
from __future__ import annotations
import hashlib
import re

_N='kmd.actual_current_same_read_snapshot_triple_channel_predeclared_gold_recall.v1'
_H='kmd.private_unseen_73_frozen_same_snapshot_retrieval_official_HTML_archive_link_supplementary_recall.v1'
_M='kmd.private_frozen_73_original_label_approved_markdown_source_reference_navigation.v1'
_SHA=re.compile('^[0-9a-f]{64}$')
_POLICIES=(10,9,8,7,5)


def _safe_path(path:str)->str:
    if (not isinstance(path,str) or not path or path.startswith('/')
            or '\\' in path or any(segment in ('','.','..') for segment in path.split('/'))):
        raise ValueError('invalid archived original source path in frozen reference evaluation')
    return path


def _digest(value)->str:
    if not isinstance(value,str) or not _SHA.fullmatch(value):
        raise ValueError('missing official original revision source SHA')
    return value


def _index(rows,*,source:str):
    if not isinstance(rows,list) or not 1<=len(rows)<=150:
        raise ValueError('missing or unbounded '+source+' frozen source rows')
    ret={}
    for r in rows:
        if (not isinstance(r,dict) or not isinstance(r.get('id'),str)
                or not 1<=len(r['id'])<=120 or r['id'] in ret):
            raise ValueError('duplicate or invalid '+source+' frozen source ID')
        ret[r['id']]=r
    return ret


def compare_frozen_reference_budgets(native:dict,html:dict,markdown:dict,
                                     *,native_input_sha256:str)->dict:
    """Count true source-known gains/losses, never inferred answer correctness.

    Native ranks from one frozen run; links from that SAME run's explicitly
    SHA-attested top-five parent source originals. The number of reserved
    slots never exceeds ten. Source full-byte SHA and current native DB
    revision are independently checked by calling acquisition audit; this
    helper works only on already verified frozen evidence/identities.
    """
    if (not isinstance(native,dict) or native.get('schema')!=_N
            or not isinstance(html,dict) or html.get('schema')!=_H
            or not isinstance(markdown,dict) or markdown.get('schema')!=_M
            or not native.get('gold_original_bytes',{}).get('all_frozen_known_original_revisions_match_current_source_map_and_download_ledger')
            or not native.get('all_search_channels_and_questions_use_one_captured_native_in_memory_embedding_set')
            or html.get('no_live_source_index_writes_or_acquisition') is not True
            or markdown.get('no_source_download_or_live_native_index_change') is not True):
        raise ValueError('unrecognized or unsupported frozen original source-truth evidence')
    _digest(native_input_sha256)
    if (html.get('frozen_original_results_SHA256')!=native_input_sha256
            or markdown.get('source_frozen_rank_report_sha256')!=native_input_sha256):
        raise ValueError('official publisher linked original labels refer to a different frozen native ranking')
    base=_index(native.get('cases'),source='native')
    links_H=_index(html.get('cases'),source='HTML')
    links_M=_index(markdown.get('cases'),source='Markdown')
    if (set(base)!=set(links_H) or set(base)!=set(links_M)
            or len(base)!=73 or html.get('frozen_question_count')!=73
            or markdown.get('frozen_gold_case_count')!=73
            or html.get('frozen_gold_attested_labels')!=79
            or markdown.get('known_original_label_count')!=79):
        raise ValueError('missing or changed frozen question/label population')
    results=[]
    for id,case in base.items():
        gold=case.get('known_originals');ranks=case.get('label_ranks',{}).get('rrf_three');
        topfive=case.get('original_top5',{}).get('rrf_three')
        if (not isinstance(gold,list) or not 1<=len(gold)<=6
                or len(set(gold))!=len(gold) or not isinstance(ranks,list)
                or len(ranks)!=len(gold) or not isinstance(topfive,list)
                or len(topfive)!=5 or len(set(topfive))!=5
                or not isinstance(case.get('group'),str) or case['group'] not in ('previous_46','independent_27')
                or links_H[id].get('known_originals')!=gold
                or any(r is not None and (type(r) is not int or not 1<=r<=30) for r in ranks)):
            raise ValueError('source-original frozen labels or native ranking shape changed')
        for path in gold+topfive:_safe_path(path)
        for path in gold:
            proof=case.get('source_revisions',{}).get(path)
            if not isinstance(proof,dict):raise ValueError('missing original+derived frozen source SHA')
            _digest(proof.get('original_sha256'));_digest(proof.get('derived_sha256'))
        groups={parent:[] for parent in topfive}
        for l in links_H[id].get('approved_source_link_proofs',[]):
            if not isinstance(l,dict):raise ValueError('invalid official HTML reference proof')
            parent=_safe_path(l.get('parent_original_path'));target=_safe_path(l.get('target_original_path'))
            _digest(l.get('source_link_original_SHA'))
            _digest(l.get('target_original_receipt_SHA'))
            if parent not in groups:raise ValueError('HTML link parent was not in previous native top five')
            groups[parent].append(target)
        for l in links_M[id].get('firstparty_approved_link_candidates',[]):
            if not isinstance(l,dict):raise ValueError('invalid official Markdown reference proof')
            parent=_safe_path(l.get('parent_original'));target=_safe_path(l.get('target_original'))
            _digest(l.get('publisher_original_SHA256'))
            _digest(l.get('target_receipt_SHA256'))
            _digest(l.get('link_original_byte_sha256'))
            if (type(l.get('original_markdown_start_byte')) is not int or l['original_markdown_start_byte']<0
                    or parent not in groups):
                raise ValueError('Markdown original literal proof or frozen parent invalid')
            groups[parent].append(target)
        ordered_links=[]
        seen=set(topfive)
        for parent in topfive:
            for dest in groups[parent]:
                if dest not in seen:
                    seen.add(dest);ordered_links.append(dest)
        methods={}
        for k in _POLICIES:
            added=[]
            for target in ordered_links:
                if len(added)>=10-k:break
                added.append(target)
            # The baseline's full known-source native ranks allow unbiased
            # label scoring without recovering private prose or all top-ten
            # paths. Duplicate native rank 6-10 destinations can only consume
            # a slot: this evaluation NEVER counts >10 source slots.
            found=[(rank is not None and rank<=k) or original in added
                   for original,rank in zip(gold,ranks)]
            methods[str(k)]={'source_known_label_hits':sum(found),
                'retrieved_known_labels_boolean':found,
                'added_original_link_candidates':added,
                'reserved_native_slots':k,
                'reserved_link_slots':10-k,
                'maximum_total_source_candidate_slots':10}
        results.append({'id':id,'group':case['group'],'family':case.get('family'),
                        'query_sha256':_digest(case.get('query_sha256')),
                        'known_originals':gold,'known_rrf_three_ranks':ranks,
                        'available_distinct_same_origin_link_destinations':len(ordered_links),
                        'methods':methods})
    group_summaries={}
    for group in ('previous_46','independent_27','all'):
        cases=[row for row in results if group=='all' or row['group']==group]
        n=sum(len(r['known_originals']) for r in cases)
        group_summaries[group]={}
        for k in _POLICIES:
            baseline=sum(r['methods']['10']['source_known_label_hits'] for r in cases)
            now=sum(r['methods'][str(k)]['source_known_label_hits'] for r in cases)
            gains=[];losses=[]
            for row in cases:
                for i,p in enumerate(row['known_originals']):
                    old=row['methods']['10']['retrieved_known_labels_boolean'][i]
                    new=row['methods'][str(k)]['retrieved_known_labels_boolean'][i]
                    if new and not old:gains.append({'id':row['id'],'original_path':p})
                    if old and not new:losses.append({'id':row['id'],'original_path':p})
            group_summaries[group][str(k)]={'frozen_known_original_labels':n,
                'native_ten_source_baseline_hits':baseline,
                'source_known_label_hits':now,
                'net_delta_hits':now-baseline,
                'gains':gains,'losses':losses,
                'maximum_source_candidate_slots_per_question':10}
    return {'schema':'kmd.private_fixed_ten_original_source_slots_publisher_reference_recall_guard.v1',
        'frozen_original_questions':len(results),
        'source_known_original_labels':sum(len(r['known_originals']) for r in results),
        'validated_previous_native_input_sha256':native_input_sha256,
        'chosen_native_reserved_slots':list(_POLICIES),
        'source_nav_candidates_prior_native_topfive_parents_only':True,
        'ten_slot_count_at_most_never_exceeded':True,
        'not_a_pristine_holdout':True,
        'not_a_precise_equal_unique_document_count_if_duplicate_link_or_fewer_links':True,
        'not_a_factual_answer_quality_or_page_truth_measure':True,
        'source_approval_and_full_original_hash_claims_only_from_input_frozen_receipts':True,
        'not_an_authority_to_change_installed_source_rankings':True,
        'group_scores':group_summaries,
        'cases':results}
