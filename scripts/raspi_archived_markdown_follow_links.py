#!/usr/bin/env python3
"""Verify first-party Markdown links to ALREADY archived originals and passages.

Read-only source navigation; never fetch documents, modify the native index,
change default search rankings, or claim that a matching excerpt is a correct
answer. Original Markdown link literal byte coordinates are independently
SHA-attested, distinct from extracted-text quotation byte coordinates.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from raspi_archived_html_follow_links import _records,_native_file_id,library,AUDIT_DIR
from file_system_catalog.archived_markdown_source_links import markdown_link_candidates
from file_system_catalog.publication_identifier import publication_receipt_matches
from file_system_catalog.source_evidence import source_evidence_candidates


def follow_current_archived_markdown_references(original_path:str,query:str,*,
            limit:int=3,passages:int=2,snippet_characters:int=850,
            exact_link_text:str|None=None)->dict:
    if (not isinstance(original_path,str) or not original_path.endswith('.md')
            or not isinstance(query,str) or not 1<=len(query)<=300
            or type(limit) is not int or not 1<=limit<=5
            or type(passages) is not int or not 1<=passages<=5
            or type(snippet_characters) is not int or not 160<=snippet_characters<=2000):
        raise ValueError('invalid bounded official Markdown original-link request')
    catalog,by_path,receipts,duplicates,dup_receipts=_records()
    if original_path in duplicates or original_path in dup_receipts:
        raise ValueError('ambiguous original publisher catalogue or acquisition receipt')
    original=by_path.get(original_path);receipt=receipts.get(original_path)
    if (not isinstance(original,dict) or original.get('kind')!='text'
            or not receipt or receipt.get('status')!='complete'
            or not publication_receipt_matches(original,receipt)):
        raise ValueError('not a unique acquired and approved original Markdown publisher source')
    parent=library.checked_original(original_path,include_publisher_text=True,
                                    publication_record=original)
    raw=parent.get('publisher_text_bytes')
    if not isinstance(raw,bytes):
        raise ValueError('full original Markdown bytes unavailable for exact link proof')
    links=markdown_link_candidates(raw,parent['original_sha256'],original['url'],
             catalog,receipts,query,original_path=original_path,limit=limit,
             exact_link_text=exact_link_text)
    parent_id=_native_file_id(original_path)
    children=[];scoped=[]
    for linked in links['selected_links']:
        path=linked['catalog_target_original_path']
        if path in duplicates or path in dup_receipts:
            raise ValueError('ambiguous linked original publisher identity')
        record=by_path.get(path);ledger=receipts.get(path)
        if not record or not ledger or not publication_receipt_matches(record,ledger):
            raise ValueError('linked original publisher acquisition metadata no longer matches')
        # Recheck the LITERAL Markdown source link byte interval independently.
        start,end=linked['publisher_link_literal_start_byte'],linked['publisher_link_literal_end_byte']
        if (type(start) is not int or type(end) is not int
                or not 0<=start<end<=len(raw)
                or hashlib.sha256(raw[start:end]).hexdigest()!=linked['publisher_link_literal_sha256']):
            raise ValueError('publisher Markdown literal link changed during original reference selection')
        target=library.checked_original(path,include_publisher_text=True,
                                        publication_record=record)
        if target['original_sha256']!=linked['catalog_target_acquisition_sha256']:
            raise ValueError('linked target original publisher SHA differs from completed receipt')
        native_id=_native_file_id(path)
        ranked=source_evidence_candidates(
            original_path=path,original_title=target['title'],
            filesystem_entry_id=native_id,
            original_sha256=target['original_sha256'],
            derived_sha256=target['derived_sha256'],
            derived_text=target['derived_text'],query=query,
            top=passages,snippet_characters=snippet_characters,
            publisher_html_bytes=target.get('publisher_html_bytes'),
            publisher_text_bytes=target.get('publisher_text_bytes'))
        verified=[]
        for passage in ranked['passages']:
            excerpt=passage['source_excerpt']
            pos=passage['excerpt_start_char'];endchar=passage['excerpt_end_char']
            if (not isinstance(excerpt,str) or type(pos) is not int or type(endchar) is not int
                or not 0<=pos<endchar<=len(target['derived_text'])
                or len(excerpt)>snippet_characters
                or target['derived_text'][pos:endchar]!=excerpt
                or hashlib.sha256(excerpt.encode('utf8')).hexdigest()!=passage['excerpt_sha256']):
                raise ValueError('linked passage text does not match the current original-derived SHA interval')
            pub=passage.get('publisher_excerpt')
            original_bytes=target.get('publisher_text_bytes')
            text_byte_verified=False
            if pub is not None:
                if (not isinstance(original_bytes,bytes) or not isinstance(pub,dict)
                    or type(pub.get('start_byte')) is not int or type(pub.get('end_byte')) is not int
                    or not 0<=pub['start_byte']<pub['end_byte']<=len(original_bytes)
                    or hashlib.sha256(original_bytes[pub['start_byte']:pub['end_byte']]).hexdigest()!=pub.get('sha256')):
                    raise ValueError('claimed publisher-original plaintext byte interval failed independent proof')
                # Matching a digest only proves the claimed publisher byte
                # interval exists. Also check that the bytes ACTUALLY QUOTE
                # this derived-text candidate, allowing only the explicit
                # historical universal-newline normalization. Otherwise an
                # unrelated but genuine publisher span could be mislabeled
                # as support for the retrieved excerpt.
                publisher_span=original_bytes[pub['start_byte']:pub['end_byte']]
                if pub.get('encoding')!='utf-8' or pub.get('newline_normalization') not in (None,'universal'):
                    raise ValueError('unsupported claimed publisher original quotation encoding')
                try:
                    actual=publisher_span.decode('utf8','strict')
                except UnicodeDecodeError as exc:
                    raise ValueError('publisher original quote bytes are not valid UTF-8') from exc
                if pub.get('newline_normalization')=='universal':
                    actual=actual.replace('\r\n','\n').replace('\r','\n')
                if actual!=excerpt:
                    raise ValueError('publisher original byte interval does not quote the retrieved source excerpt')
                text_byte_verified=True
            html_anchor=passage.get('publisher_html_dom_anchor')
            verified.append({'source_section':passage['source_section'],
                'current_derived_text_excerpt':excerpt,
                'derived_excerpt_start_char':pos,'derived_excerpt_end_char':endchar,
                'derived_excerpt_sha256':passage['excerpt_sha256'],
                'matched_excerpt_query_terms':passage['matched_excerpt_terms'],
                'publisher_original_plaintext_byte_interval':pub,
                'publisher_plaintext_byte_interval_independently_SHA_verified':text_byte_verified,
                'publisher_HTML_normalized_article_anchor':html_anchor,
                'publisher_literal_byte_quote_claimed':text_byte_verified,
                'factual_answer_sufficiency_verified':False})
        guard=library.final_original_revision_guard(target,original_path=path,native_file_id=native_id)
        if not guard['latest_source_map_revision_and_native_file_id_rechecked']:
            raise ValueError('linked original source revision changed before result assembly')
        scoped.append((path,target,native_id))
        children.append({'original_path':path,'original_sha256':target['original_sha256'],
              'derived_sha256':target['derived_sha256'],'native_filedb_original_id':native_id,
              'publisher_source_link_proof':linked,
              'passage_candidates':verified,
              'current_source_revision_guard_after_excerpts':True,
              'candidate_is_not_a_verified_answer':True})
    # Rehash every referenced publisher original/derivative and native ID at
    # the END of all queries. An observed stale result fails the whole call.
    for path,target,native_id in scoped:
        latest=library.final_original_revision_guard(target,original_path=path,native_file_id=native_id)
        if not latest['latest_source_map_revision_and_native_file_id_rechecked']:
            raise ValueError('linked original source changed after later retrieved source')
    parent_guard=library.final_original_revision_guard(parent,original_path=original_path,
                                                       native_file_id=parent_id)
    if not parent_guard['latest_source_map_revision_and_native_file_id_rechecked']:
        raise ValueError('parent Markdown link source changed after child passage retrieval')
    return {'schema':'kmd.private_acquired_original_markdown_first_party_link_evidence.v1',
       'query':query,'parent_original_path':original_path,
       'parent_original_sha256':parent['original_sha256'],
       'parent_derived_sha256':parent['derived_sha256'],
       'explicit_publisher_link_name_for_navigation':exact_link_text,
       'parent_original_Markdown_literal_byte_link_proof':links,
       'linked_approved_current_original_sources':children,
       'linked_original_sources_checked':len(children),
       'linked_derived_passage_candidates':sum(len(x['passage_candidates']) for x in children),
       'independently_publisher_original_plaintext_byte_attested_passages':sum(
           sum(p['publisher_plaintext_byte_interval_independently_SHA_verified'] for p in x['passage_candidates']) for x in children),
       'all_parent_and_target_full_source_revision_checks_after_search':True,
       'cross_store_immutable_generation_pinned':False,
       'safe_to_cutover_live':False,
       'not_general_semantic_original_discovery_or_factual_answer':True,
       'no_original_native_vectors_source_map_index_writes':True,
       'no_external_network_or_new_source_acquisition':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query')
    parser.add_argument('--original-path',required=True)
    parser.add_argument('--links',type=int,default=3)
    parser.add_argument('--passages',type=int,default=2)
    parser.add_argument('--snippet-characters',type=int,default=850)
    parser.add_argument('--exact-link-text',help='Literal official publisher link text; navigation only, not semantic relevance proof')
    parser.add_argument('--private-output',type=Path,required=True)
    opts=parser.parse_args()
    output=opts.private_output
    if (not output.is_absolute() or output.exists() or output.is_symlink()
        or not output.parent.is_dir() or output.parent.is_symlink()
        or not output.resolve().is_relative_to(AUDIT_DIR.resolve())):
        raise ValueError('refusing unsafe or duplicate private Markdown source proof')
    start=time.monotonic()
    data=follow_current_archived_markdown_references(opts.original_path,opts.query,
        limit=opts.links,passages=opts.passages,
        snippet_characters=opts.snippet_characters,exact_link_text=opts.exact_link_text)
    data['checked_utc']=datetime.now(timezone.utc).isoformat()
    data['read_only_query_seconds']=round(time.monotonic()-start,3)
    blob=(json.dumps(data,indent=2,ensure_ascii=False)+'\n').encode('utf8')
    fd=os.open(output,os.O_CREAT|os.O_WRONLY|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:
        f.write(blob);f.flush();os.fsync(f.fileno())
    print('ORIGINAL_MARKDOWN_LITERALS_VERIFIED',len(data['parent_original_Markdown_literal_byte_link_proof']['selected_links']))
    print('LINKED_CURRENT_ACQUIRED_ORIGINALS',data['linked_original_sources_checked'])
    print('INDEPENDENT_PUBLISHER_ORIGINAL_PLAINTEXT_BYTES_PASSED',data['independently_publisher_original_plaintext_byte_attested_passages'])
    print('THIS_IS_NOT_FACTUAL_ANSWER_OR_LIVE_SOURCE_MIGRATION',True)
    print('PRIVATE_ORIGINAL_EVIDENCE_RECEIPT',output)

if __name__=='__main__':main()
