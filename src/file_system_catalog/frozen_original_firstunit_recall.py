"""Pure frozen-gold, original-source first-unit candidate-recall attestation.

This is explicitly a label-navigation benchmark, never an answer-truth test
or a new production rank. Keep independently ranked literal and Porter
FTS results separate, retaining the known-source multi-positive denominator.
The caller additionally verifies the publisher original/derived full-byte
and unique native FileDB identities for every known positive.
"""
from __future__ import annotations
import hashlib
import json
import re

_SHA=re.compile(r'[a-f0-9]{64}\Z')
_QUERY_ID=re.compile(r'[A-Za-z0-9_.-]{1,80}\Z')
_SCOPE='derived_source_text_only_not_physical_page_proof'
_CHANNELS=('literal','porter')


def _sha_text(txt:str)->str:
    return hashlib.sha256(txt.encode('utf8')).hexdigest()


def _source_label(path:str)->str:
    if (not isinstance(path,str) or not 1<=len(path)<=1600
            or path.startswith('/') or '\\' in path or '\x00' in path
            or any(z in ('','.','..') for z in path.split('/'))):
        raise ValueError('invalid frozen source-original relative path')
    return path


def _gold_rows(independent:list[dict],multi:list[dict])->list[dict]:
    if not isinstance(independent,list) or len(independent)!=27:
        raise ValueError('expected exact 27 predeclared independent original-source questions')
    if not isinstance(multi,list) or len(multi)!=5:
        raise ValueError('expected exact five frozen independently attested multi-relevant questions')
    found=set();rows=[]
    for group,items in (('independent_27',independent),('multiple_5',multi)):
        for row in items:
            if not isinstance(row,dict):
                raise ValueError('invalid frozen original-source question')
            ident=row.get('id')
            query=row.get('query')
            if (not isinstance(ident,str) or not _QUERY_ID.fullmatch(ident)
                    or ident in found or not isinstance(query,str)
                    or not 3<=len(query)<=600 or not query.strip()):
                raise ValueError('duplicate or invalid original predeclared question')
            if group=='independent_27':
                gold=[_source_label(row.get('expected_path'))]
            else:
                labels=row.get('relevant_paths')
                if (not isinstance(labels,list) or not 2<=len(labels)<=20):
                    raise ValueError('unbounded multi-original positive source population')
                gold=[_source_label(p) for p in labels]
                if len(gold)!=len(set(gold)):
                    raise ValueError('duplicated multi-original source labels')
            found.add(ident)
            rows.append({'id':ident,'group':group,
                         'query_sha256':_sha_text(query),
                         'query':query,'known_originals':gold})
    if len(rows)!=32 or sum(len(x['known_originals']) for x in rows)!=38:
        raise ValueError('frozen thirty-two question source label cardinality mismatch')
    return rows


def _one_verified_channel(data:dict,*,max_rank:int, query:str)->tuple[list[str],list[dict]]:
    if (not isinstance(data,dict) or type(data.get('returned_current_originals')) is not int
            or not isinstance(data.get('source_originals'),list)
            or data['returned_current_originals']!=len(data['source_originals'])
            or len(data['source_originals'])>max_rank):
        raise ValueError('unverified or too large original first-unit source channel')
    paths=[]
    slim=[]
    for rank,original in enumerate(data['source_originals'],1):
        if (not isinstance(original,dict) or original.get('source_rank')!=rank
                or type(original['source_rank']) is not int
                or original.get('current_approved_publisher_and_extracted_bytes_rehashed') is not True
                or original.get('quoted_text_is_original_publisher_bytes') is not False
                or original.get('factual_answer_verified') is not False
                or original.get('source_unit_label_scope')!=_SCOPE
                or type(original.get('native_original_filedb_entry_id')) is not int
                or original['native_original_filedb_entry_id']<=0
                or not isinstance(original.get('publisher_first_unit_preview'),str)
                or not isinstance(original.get('source_preview_sha256'),str)
                or _SHA.fullmatch(original['source_preview_sha256']) is None
                or original['source_preview_sha256']!=_sha_text(original['publisher_first_unit_preview'])
                or type(original.get('source_preview_start_char')) is not int
                or type(original.get('source_preview_end_char')) is not int
                or not 0<=original['source_preview_start_char']<original['source_preview_end_char']
                or original['source_preview_end_char']-original['source_preview_start_char']!=len(
                    original['publisher_first_unit_preview'])):
            raise ValueError('missing or forged selected publisher-original source proof')
        for k in ('publisher_original_sha256','current_derivative_sha256'):
            if (not isinstance(original.get(k),str)
                    or _SHA.fullmatch(original[k]) is None):
                raise ValueError('original or derived source revision is not verified')
        path=_source_label(original.get('original_path'))
        if path in paths:
            raise ValueError('duplicate source original in one original lexical channel')
        paths.append(path)
        slim.append({
            'original_path':path,
            'source_rank':rank,
            'source_original_sha256':original['publisher_original_sha256'],
            'current_derived_sha256':original['current_derivative_sha256'],
            'native_original_filedb_id':original['native_original_filedb_entry_id'],
            'source_excerpt_sha256':original['source_preview_sha256'],
            'physical_original_PDF_page_verification_not_implied':True,
        })
    return paths,slim


def evaluate_frozen_firstunit_original_recall(
    *,
    independent:list[dict],
    multi:list[dict],
    queries_file:dict,
    source_report:dict,
    expected_current_native_sha_by_original:dict[str,tuple[str,str,int]],
)->dict:
    """Attest immutable input questions and two independent source-rank lists.

    expected_current_native_sha_by_original is an independently and freshly
    rehashed caller-provided current (original SHA, derived SHA, native ID)
    for *every positive label*, even labels absent from any ranked list.
    """
    gold=_gold_rows(independent,multi)
    if (not isinstance(queries_file,dict) or set(queries_file)!={'queries'}
            or not isinstance(queries_file['queries'],list)
            or len(queries_file['queries'])!=len(gold)
            or not isinstance(source_report,dict)
            or source_report.get('schema')!='kmd.readonly_publisher_first_source_unit_dual_lexical_discovery.v1'
            or source_report.get('query_count')!=len(gold)
            or source_report.get('independent_rank_channels_no_fusion') is not True
            or source_report.get('native_filedb_and_source_map_read_only') is not True
            or source_report.get('no_persistent_second_database_created') is not True
            or source_report.get('all_displayed_candidates_original_and_full_derivative_sha_reverified') is not True
            or source_report.get('originals_downloaded_or_modified')!=0
            or not isinstance(source_report.get('queries'),list)
            or len(source_report['queries'])!=len(gold)):
        raise ValueError('not a complete current first-unit, source-verified dual lexical report')
    scope=source_report.get('source_scan')
    if (not isinstance(scope,dict)
            or type(scope.get('eligible_complete_source_map_records')) is not int
            or type(scope.get('publisher_firstunit_rows_indexed')) is not int
            or scope['publisher_firstunit_rows_indexed']<=0
            or scope['publisher_firstunit_rows_indexed']>scope['eligible_complete_source_map_records']
            or scope.get('skipped_sources')!=0):
        raise ValueError('incomplete or unverified first-unit global original-source coverage')
    if source_report.get('literal_source_directory_prefix') is not None:
        raise ValueError('global frozen benchmark cannot claim a publisher or path-scoped source subset')
    positive={p for row in gold for p in row['known_originals']}
    if (not isinstance(expected_current_native_sha_by_original,dict)
            or positive!=set(expected_current_native_sha_by_original)):
        raise ValueError('not all predeclared positive publisher originals have current source bytes')
    for p,(sha,der,ident) in expected_current_native_sha_by_original.items():
        if (not isinstance(sha,str) or not _SHA.fullmatch(sha)
                or not isinstance(der,str) or not _SHA.fullmatch(der)
                or type(ident) is not int or ident<=0):
            raise ValueError('frozen positive original source revision/identity unattested')
    details=[]
    counts={group:{ch:{'known_originals':0,'known_originals_at1':0,
                       'known_originals_at5':0,'known_originals_at10':0,
                       'questions_with_any_known_source_at10':0,
                       'questions_with_every_known_source_at10':0}
                    for ch in _CHANNELS}
            for group in ('independent_27','multiple_5')}
    union_metrics={
        group:{'questions':0,'known_positive_labels':0,
               'known_positive_labels_found_in_either_source_channel_at10':0,
               'questions_with_any_positive_in_either_source_channel_at10':0,
               'questions_with_all_positives_in_either_source_channel_at10':0,
               'known_positives_present_exclusively_in_literal_channel':0,
               'known_positives_present_exclusively_in_porter_channel':0}
        for group in ('independent_27','multiple_5')
    }
    recovered_twice=0
    for i,case in enumerate(gold):
        input_row=queries_file['queries'][i]
        row=source_report['queries'][i]
        if (not isinstance(input_row,dict) or set(input_row)!={'id','query'}
                or input_row['id']!=case['id'] or input_row['query']!=case['query']
                or not isinstance(row,dict) or row.get('id')!=case['id']
                or row.get('query_sha256')!=case['query_sha256']
                or row.get('query')!=case['query']
                or row.get('rank_fusion_applied') is not False
                or not isinstance(row.get('independent_rank_channels'),dict)
                or set(row['independent_rank_channels'])!=set(_CHANNELS)):
            raise ValueError('changed previously frozen source query or fabricated ranking record')
        channels={}
        slim={}
        for ch in _CHANNELS:
            paths,proofs=_one_verified_channel(
                row['independent_rank_channels'][ch],query=case['query'],max_rank=10)
            by_path={p['original_path']:p for p in proofs}
            ranks={}
            for original in case['known_originals']:
                position=paths.index(original)+1 if original in paths else None
                ranks[original]=position
                if position is not None:
                    attested=by_path[original]
                    sha,derived,ident=expected_current_native_sha_by_original[original]
                    if ((attested['source_original_sha256'],attested['current_derived_sha256'],
                         attested['native_original_filedb_id'])!=(sha,derived,ident)):
                        raise ValueError('ranked golden original no longer agrees with its current verified source identity')
                    recovered_twice+=1
            stats=counts[case['group']][ch]
            stats['known_originals']+=len(ranks)
            stats['known_originals_at1']+=sum(r==1 for r in ranks.values())
            stats['known_originals_at5']+=sum(r is not None and r<=5 for r in ranks.values())
            stats['known_originals_at10']+=sum(r is not None for r in ranks.values())
            stats['questions_with_any_known_source_at10']+=int(any(r is not None for r in ranks.values()))
            stats['questions_with_every_known_source_at10']+=int(all(r is not None for r in ranks.values()))
            channels[ch]=ranks
            slim[ch]=len(paths)
        # A union here means presence in EITHER verified source candidate
        # list, NEVER an interleaved, ranked or score-calibrated top ten.
        union=union_metrics[case['group']]
        union['questions']+=1
        union['known_positive_labels']+=len(case['known_originals'])
        seen_by_any={
            name:(channels['literal'][name] is not None or
                  channels['porter'][name] is not None)
            for name in case['known_originals']
        }
        union['known_positive_labels_found_in_either_source_channel_at10']+=sum(seen_by_any.values())
        union['questions_with_any_positive_in_either_source_channel_at10']+=int(any(seen_by_any.values()))
        union['questions_with_all_positives_in_either_source_channel_at10']+=int(all(seen_by_any.values()))
        union['known_positives_present_exclusively_in_literal_channel']+=sum(
            channels['literal'][name] is not None and channels['porter'][name] is None
            for name in case['known_originals'])
        union['known_positives_present_exclusively_in_porter_channel']+=sum(
            channels['porter'][name] is not None and channels['literal'][name] is None
            for name in case['known_originals'])
        details.append({'id':case['id'],'group':case['group'],
                        'question_sha256':case['query_sha256'],
                        'positive_original_ranks_in_independent_channels':channels,
                        'source_original_results_by_channel':slim,
                        'positive_originals_present_in_either_channel_without_a_combined_rank':
                            [name for name,present in seen_by_any.items() if present],
                        'not_a_factual_answer_ground_truth_or_exhaustive_relevance_gold':True})
    return {
        'schema':'kmd.frozen_source_verified_dual_firstunit_recall_evaluation.v1',
        'known_gold_questions':len(gold),
        'known_positive_originals':len(positive),
        'known_positive_labels_including_repeated_paths':sum(len(z['known_originals']) for z in gold),
        'source_preview_indexed_original_count':scope['publisher_firstunit_rows_indexed'],
        'all_source_gold_originals_original_and_derived_current_full_sha_verified_externally':True,
        'source_query_text_frozen_before_retrieval':True,
        'no_score_fusion_original_raw_sources_or_native_ranking_change':True,
        'only_first_current_source_unit_per_original_indexed':True,
        'source_relevance_labels_not_assumed_exhaustive':True,
        'publisher_visual_PDF_page_or_factual_answer_verification_not_claimed':True,
        'ranked_positive_source_occurrences_cross_checked_against_current_native_ID':recovered_twice,
        'separate_source_channel_metrics':counts,
        'distinct_channel_candidate_union_presence_only_not_ranked_top10':union_metrics,
        'no_union_confidence_or_fused_rank_defined':True,
        'case_ranks':details,
    }


def complementary_multi_original_source_presence(
    *, frozen_multi:list[dict], frozen_multi_full_file_sha256:str,
    verified_firstunit_summary:dict, current_native_report:dict,
)->dict:
    """Compare original-source *presence* in two existing separate searches.

    The existing native hierarchical summary top ten and first-unit FTS top
    ten do not share a calibrated score or rank. A union is an UNRANKED
    navigation pool; do NOT call it recall@10 of a single ranked reader.
    This five-case fixture was predeclared independently of both methods.
    """
    if (not isinstance(frozen_multi,list) or len(frozen_multi)!=5
            or not isinstance(frozen_multi_full_file_sha256,str)
            or _SHA.fullmatch(frozen_multi_full_file_sha256) is None
            or not isinstance(verified_firstunit_summary,dict)
            or verified_firstunit_summary.get('schema')!='kmd.frozen_source_verified_dual_firstunit_recall_evaluation.v1'
            or verified_firstunit_summary.get('no_score_fusion_original_raw_sources_or_native_ranking_change') is not True
            or not isinstance(current_native_report,dict)
            or current_native_report.get('schema')!='kmd.predeclared_original_retrieval_drift.v1'
            or current_native_report.get('fixed_gold_sha256')!=frozen_multi_full_file_sha256
            or current_native_report.get('complete_fixed_population') is not True
            or current_native_report.get('all_publisher_gold_bytes_rechecked') is not True
            or current_native_report.get('scope')!='only exact path predeclared gold; historical negatives are not labeled unanswerable'):
        raise ValueError('source presence comparison lacks matching full frozen multi-original publisher provenance')
    multi_by_id={x['id']:x for x in frozen_multi}
    source_by_id={x['id']:x for x in verified_firstunit_summary.get('case_ranks',[])
                  if x.get('group')=='multiple_5'}
    native_rows=current_native_report.get('cases')
    if (len(multi_by_id)!=5 or len(source_by_id)!=5
            or not isinstance(native_rows,list) or len(native_rows)!=5):
        raise ValueError('native or original first-unit multi-source question population incomplete')
    native_by_id={z.get('id'):z for z in native_rows if isinstance(z,dict)}
    if len(native_by_id)!=5 or set(native_by_id)!=set(multi_by_id) or set(source_by_id)!=set(multi_by_id):
        raise ValueError('source identity and question set changed between two methods')
    cases=[]
    aggregate={'positive_originals':0,'native_top10_source_positives':0,
               'original_firstunit_either_channel_top10_source_positives':0,
               'unranked_source_presence_union_positives':0,
               'source_labels_found_exclusively_in_native':0,
               'source_labels_found_exclusively_in_original_firstunit':0,
               'questions_with_all_known_positive_originals_in_unranked_union':0}
    for entry in frozen_multi:
        label=entry['id']
        paths=entry['relevant_paths']
        source=source_by_id[label]
        native=native_by_id[label]
        if (not isinstance(paths,list) or len(set(paths))!=len(paths)
                or source.get('question_sha256')!=_sha_text(entry['query'])
                or native.get('query_sha256')!=_sha_text(entry['query'])
                or native.get('known_relevant_original_paths')!=paths
                or source.get('positive_original_ranks_in_independent_channels') is None
                or not isinstance(native.get('known_relevant_ranks'),dict)
                or set(native['known_relevant_ranks'])!=set(paths)):
            raise ValueError('frozen question, native ranks or original-source positive labels changed')
        literal=source['positive_original_ranks_in_independent_channels']['literal']
        porter=source['positive_original_ranks_in_independent_channels']['porter']
        if set(literal)!=set(paths) or set(porter)!=set(paths):
            raise ValueError('missing literal/Porter source-attested positive original rank')
        found=[]
        for path in paths:
            n=native['known_relevant_ranks'][path]
            l=literal[path]
            p=porter[path]
            if not all(v is None or type(v) is int and 1<=v<=10 for v in (n,l,p)):
                raise ValueError('current native or first-unit rank is outside verified top ten')
            raw=n is not None
            source_seen=l is not None or p is not None
            aggregate['positive_originals']+=1
            aggregate['native_top10_source_positives']+=int(raw)
            aggregate['original_firstunit_either_channel_top10_source_positives']+=int(source_seen)
            aggregate['unranked_source_presence_union_positives']+=int(raw or source_seen)
            aggregate['source_labels_found_exclusively_in_native']+=int(raw and not source_seen)
            aggregate['source_labels_found_exclusively_in_original_firstunit']+=int(source_seen and not raw)
            found.append(raw or source_seen)
        aggregate['questions_with_all_known_positive_originals_in_unranked_union']+=int(all(found))
        cases.append({
            'id':label,'query_sha256':_sha_text(entry['query']),
            'frozen_source_positive_count':len(paths),
            'known_positive_originals_present_somewhere_in_unranked_union':sum(found),
            'all_frozen_positive_originals_present_in_unranked_union':all(found),
        })
    if aggregate['positive_originals']!=11:
        raise ValueError('unexpected frozen multi-original eleven-label population')
    return {
        'schema':'kmd.private_source_verified_complementary_native_and_firstunit_5multi_positive_presence.v1',
        'frozen_source_original_gold_sha256':frozen_multi_full_file_sha256,
        'source_provenance_from_current_native_publisher_gold_and_original_firstunit':True,
        'each_individual_source_query_unchanged_between_methods':True,
        'no_single_top10_source_union_rank_defined':True,
        'not_a_global_rank_quality_or_answer_accuracy_result':True,
        'original_source_labels_are_known_relevant_not_exhaustive':True,
        'scores':aggregate,
        'cases':cases,
    }
