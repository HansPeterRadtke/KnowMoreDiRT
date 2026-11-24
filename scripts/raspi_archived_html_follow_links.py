#!/usr/bin/env python3
"""Follow source-attested HTML article links into ALREADY archived KMD originals.

This command cannot browse or download; it reads original publisher and
verified extracted bytes through canonical KnowMoreDiRT source checks. The
first original path is selected explicitly; this is NOT general corpus ranking
or automatic answer generation. All resulting passages remain candidates.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
import raspi_hierarchical_library as library
from file_system_catalog.archived_html_source_links import source_link_candidates
from file_system_catalog.publication_identifier import publication_receipt_matches
from file_system_catalog.source_evidence import source_evidence_candidates

AUDIT_DIR=Path('/data/var/knowmoredirt/knowledge-library/search-audit')


def _records():
    catalog=json.loads(library.SOURCES.read_text('utf8'))['sources']
    ledger=json.loads(library.LEDGER.read_text('utf8'))['items']
    if not isinstance(catalog,list) or not isinstance(ledger,list) or len(catalog)>50000 or len(ledger)>50000:
        raise ValueError('invalid bounded original publisher catalog or acquisition ledger')
    bypath={};duplicates=set();completed={};duplicate_receipts=set()
    for entry in catalog:
        p=entry.get('path')
        if p in bypath:duplicates.add(p)
        bypath[p]=entry
    for rec in ledger:
        p=rec.get('path')
        if p in completed:duplicate_receipts.add(p)
        completed[p]=rec
    return catalog,bypath,completed,duplicates,duplicate_receipts


def _native_file_id(path:str):
    with sqlite3.connect(f'file:{library.DB.resolve()}?mode=ro',uri=True,
                         timeout=library.NATIVE_READER_WAIT_SECONDS) as db:
        db.execute('pragma query_only=on')
        ids=db.execute("SELECT id FROM filesystem_entries WHERE relative_path_display=? "
                       "AND entry_type='file'",(path,)).fetchall()
    if len(ids)!=1 or type(ids[0][0]) is not int or ids[0][0]<1:
        raise ValueError('no unique original native FileDB identity in current source generation')
    return ids[0][0]


def find_archived_linked_source_passages(original_path:str,query:str,*,limit:int=3,
                                        passages:int=2,snippet_characters:int=850):
    if (not isinstance(original_path,str) or not original_path.lower().endswith(('.html','.htm'))
            or not isinstance(query,str) or not 1<=len(query)<=300
            or type(limit) is not int or not 1<=limit<=5
            or type(passages) is not int or not 1<=passages<=5
            or type(snippet_characters) is not int or not 160<=snippet_characters<=2000):
        raise ValueError('invalid bounded archived original-link retrieval request')
    catalog,bypath,receipts,duplicated_catalog,duplicated_receipts=_records()
    if original_path in duplicated_catalog or original_path in duplicated_receipts:
        raise ValueError('ambiguous source publisher catalog/receipt original path')
    record=bypath.get(original_path);receipt=receipts.get(original_path)
    if (not isinstance(record,dict) or record.get('kind')!='html'
            or not receipt or receipt.get('status')!='complete'
            or not publication_receipt_matches(record,receipt)):
        raise ValueError('original source has no uniquely approved completed HTML publisher receipt')
    parent=library.checked_original(original_path,include_publisher_text=True,
                                    publication_record=record)
    raw=parent.get('publisher_html_bytes')
    if raw is None:
        raise ValueError('original publisher HTML too large for bounded source link proof')
    links=source_link_candidates(raw,parent['original_sha256'],record['url'],catalog,
                                 receipts,query,original_path=original_path,limit=limit)
    parent_id=_native_file_id(original_path)
    candidates=[]
    inspected_source_versions=[]
    for link in links['selected_links']:
        path=link['catalog_target_original_path']
        if path in duplicated_catalog or path in duplicated_receipts:
            raise ValueError('ambiguous archived target catalog/receipt original path')
        row=bypath.get(path);receipt=receipts.get(path)
        if (not row or not receipt or not publication_receipt_matches(row,receipt)):
            raise ValueError('linked target current original receipt no longer matches catalog')
        # Independently full-SHA verify original publisher and current source
        # map derivative for every linked original. No blind link traversal.
        target=library.checked_original(path,include_publisher_text=True,
                                        publication_record=row)
        if target['original_sha256']!=link['catalog_target_acquisition_sha256']:
            raise ValueError('linked original publisher file changed during source navigation')
        native_id=_native_file_id(path)
        passages_raw=source_evidence_candidates(
            original_path=path,original_title=target['title'],
            filesystem_entry_id=native_id,
            original_sha256=target['original_sha256'],
            derived_sha256=target['derived_sha256'],
            derived_text=target['derived_text'],
            query=query,top=passages,snippet_characters=snippet_characters,
            publisher_html_bytes=target.get('publisher_html_bytes'),
            publisher_text_bytes=target.get('publisher_text_bytes'))
        verified=[]
        for passage in passages_raw['passages']:
            start=passage['excerpt_start_char']
            end=passage['excerpt_end_char']
            excerpt=passage['source_excerpt']
            # A candidate's text can be useful immediately, rather than
            # requiring a second manual lookup by character offsets. Recheck
            # exact source span and UTF-8 digest ourselves before including it.
            if (type(start) is not int or type(end) is not int
                    or not 0<=start<end<=len(target['derived_text'])
                    or end-start>snippet_characters
                    or not isinstance(excerpt,str)
                    or target['derived_text'][start:end]!=excerpt
                    or hashlib.sha256(excerpt.encode('utf8')).hexdigest()!=passage['excerpt_sha256']):
                raise ValueError('linked source candidate excerpt does not match the current verified derived original')
            anchor=passage.get('publisher_html_dom_anchor')
            verified.append({
                'source_section':passage['source_section'],
                'extracted_text_excerpt_candidate':excerpt,
                'extracted_text_excerpt_is_not_a_literal_publisher_byte_quotation':True,
                'extracted_text_excerpt_start_char':passage['excerpt_start_char'],
                'extracted_text_excerpt_end_char':passage['excerpt_end_char'],
                'extracted_text_excerpt_sha256':passage['excerpt_sha256'],
                'matched_source_terms':passage['matched_excerpt_terms'],
                'publisher_original_normalized_HTML_article_anchor':anchor,
                'independent_original_article_text_match_verified':bool(anchor),
                'literal_publisher_bytes_verified':False,
                'original_pdf_physical_page_verified':False,
                'factual_answer_sufficiency_verified':False,
            })
        final=library.final_original_revision_guard(target,original_path=path,native_file_id=native_id)
        if not final['latest_source_map_revision_and_native_file_id_rechecked']:
            raise ValueError('linked target original/source-map/native revision not current')
        inspected_source_versions.append((path,target,native_id))
        candidates.append({'original_path':path,'original_sha256':target['original_sha256'],
            'derived_sha256':target['derived_sha256'],'native_filedb_original_id':native_id,
            'link_provenance':link,'passage_candidates':verified,
            'returned_passage_candidates':len(verified),
            'publisher_HTML_article_evidence_passages':sum(bool(x['independent_original_article_text_match_verified']) for x in verified),
            'final_source_revision_checked_after_passage_selection':True,
            'source_link_candidate_not_factual_answer':True})
    # Reopen parent original/source-map/native AFTER searching all destinations.
    # The source where an article link was observed must still be current.
    parent_fence=library.final_original_revision_guard(parent,original_path=original_path,
                                                       native_file_id=parent_id)
    if not parent_fence['latest_source_map_revision_and_native_file_id_rechecked']:
        raise ValueError('original linked source changed during target passage retrieval')
    # A prior linked destination can change WHILE a later destination is being
    # checked. Reopen and full-SHA reverify ALL returned originals only after
    # the last evidence was assembled, then recheck the parent one more time.
    # This rejects observed stale target versions; it does NOT create an
    # immutable atomic generation across the independently updated databases.
    for path,target,native_id in inspected_source_versions:
        last=library.final_original_revision_guard(target,original_path=path,native_file_id=native_id)
        if not last['latest_source_map_revision_and_native_file_id_rechecked']:
            raise ValueError('linked target revision changed after later source results')
    last_parent=library.final_original_revision_guard(parent,original_path=original_path,native_file_id=parent_id)
    if not last_parent['latest_source_map_revision_and_native_file_id_rechecked']:
        raise ValueError('parent source link revision changed after final target check')
    return {'schema':'kmd.existing_archived_original_html_first_party_reference_follow_v1',
        'query':query,'parent_original_path':original_path,
        'parent_original_sha256':parent['original_sha256'],
        'parent_derived_sha256':parent['derived_sha256'],
        'parent_original_HTML_article_link_proof':links,
        'linked_existing_original_sources':candidates,
        'linked_original_sources_checked':len(candidates),
        'linked_source_excerpt_candidates':sum(x['returned_passage_candidates'] for x in candidates),
        'publisher_original_article_evidence_passages':sum(x['publisher_HTML_article_evidence_passages'] for x in candidates),
        'parent_current_original_source_map_and_native_ID_rechecked_after_targets':True,
        'all_selected_destination_original_source_map_and_native_ID_rechecked_after_other_targets':True,
        'cross_store_immutable_generation_pinned':False,
        'no_new_source_downloads_or_external_navigation':True,
        'no_original_or_native_index_mutation':True,
        'not_general_semantic_document_discovery':True,
        'not_proof_of_correct_answer_or_atomic_cross_store_generation':True,
        'safe_to_cutover_live':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('query')
    p.add_argument('--original-path',required=True,help='Exact archived HTML page explicitly selected by user or verified source catalogue')
    p.add_argument('--links',type=int,default=3)
    p.add_argument('--passages',type=int,default=2)
    p.add_argument('--snippet-characters',type=int,default=850)
    p.add_argument('--private-output',type=Path,required=True,help='New file under existing private KMD search audit directory')
    args=p.parse_args()
    output=args.private_output
    if (not output.is_absolute() or output.is_symlink() or output.exists()
            or not output.parent.is_dir() or output.parent.is_symlink()
            or not output.resolve().is_relative_to(AUDIT_DIR.resolve())):
        raise ValueError('refusing to overwrite or redirect private source evidence')
    stamp=time.monotonic()
    result=find_archived_linked_source_passages(args.original_path,args.query,
                      limit=args.links,passages=args.passages,snippet_characters=args.snippet_characters)
    result['query_executed_utc']=datetime.now(timezone.utc).isoformat()
    result['read_only_query_duration_seconds']=round(time.monotonic()-stamp,3)
    raw=(json.dumps(result,indent=2,ensure_ascii=False)+'\n').encode()
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    print('MATCHED_ARCHIVED_SOURCE_LINKS',result['linked_original_sources_checked'])
    print('VERIFIED_EXTRACTED_PASSAGE_CANDIDATES',result['linked_source_excerpt_candidates'])
    print('VERIFIED_PUBLISHER_ORIGINAL_ARTICLE_CANDIDATES',result['publisher_original_article_evidence_passages'])
    print('NOT_AN_ANSWER_OR_PRODUCTION_GENERATION_SWITCH',True)
    print('PRIVATE_OUTPUT',str(output))

if __name__=='__main__':main()
