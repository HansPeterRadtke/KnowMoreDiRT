"""Retrieve one exact publisher-source locator from a PRIVATE pinned generation.

No external network, no shared model, no live FileDB/source-map connection,
no production cutover, and NO independently verified physical PDF quotation.
The caller obtains source-linked machine-text locators only after both immutable
SQLite views and all pinned derivative/summary assets are rechecked on exit.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

from .private_source_generation import open_pinned_private_read_view
from .source_evidence import source_evidence_candidates
from .summary_coverage import revision_identity

_SOURCE_SHA=re.compile(r'^[0-9a-f]{64}$')


def _sha(blob:bytes)->str:
    return hashlib.sha256(blob).hexdigest()


def retrieve_private_source_locators(base:Path,allowed_root:Path,original_path:str,
                                    query:str,*,top:int=3)->dict:
    """One original's current private summaries and exact machine page locators.

    A completed return implies the pinned read view exited cleanly and
    independently verified all old-generation assets *after* searching.
    Original publisher visual page identity remains unverified by this API.
    """
    if (not isinstance(original_path,str) or not 1<=len(original_path)<=1200
            or original_path.startswith('/') or '\\' in original_path
            or any(p in ('','.','..') for p in original_path.split('/'))
            or not original_path.endswith('.pdf')
            or not isinstance(query,str) or not 1<=len(query)<=240
            or type(top) is not int or not 1<=top<=5):
        raise ValueError('private source lookup requires one safe publisher PDF and bounded query')
    if any(ord(ch)<32 for ch in original_path+query):
        raise ValueError('private source lookup contains control characters')
    with open_pinned_private_read_view(base,allowed_root) as pinned:
        # Both connections are read-only from one pinned generation, not live.
        records=pinned.source_map.execute(
            'SELECT source_path,source_sha256,derived_sha256,derived_path,status,original_kind '
            'FROM originals WHERE source_path=?',(original_path,)).fetchall()
        if len(records)!=1:
            raise ValueError('private original is not in this selected source generation')
        _,source_sha,derived_sha,derived_path,status,kind=records[0]
        if (status!='complete' or kind!='pdf' or derived_path!=original_path+'.txt'
                or not isinstance(source_sha,str) or not _SOURCE_SHA.fullmatch(source_sha)
                or not isinstance(derived_sha,str) or not _SOURCE_SHA.fullmatch(derived_sha)):
            raise ValueError('private selected original source-map revision invalid')
        ids=pinned.native.execute('SELECT id FROM filesystem_entries '
                 "WHERE relative_path_display=? AND entry_type='file'",(original_path,)).fetchall()
        if len(ids)!=1 or type(ids[0][0]) is not int or ids[0][0]<1:
            raise ValueError('private original has no unique canonical native FileDB ID')
        file_id=ids[0][0]
        derived_bytes=pinned.read_source_asset('corrected_derived_text','text/'+derived_path)
        if _sha(derived_bytes)!=derived_sha:
            raise ValueError('private original derivative differs from pinned source-map revision')
        text=derived_bytes.decode('utf8','strict')
        basename=re.sub(r'[^A-Za-z0-9_.-]+','-',Path(original_path).name[:64])
        expected='summaries/'+basename+'.'+revision_identity(
                     original_path,source_sha,derived_sha)[:24]+'.summary-tree.json'
        tree_bytes=pinned.read_source_asset('recursive_summary_tree',expected)
        tree=json.loads(tree_bytes)
        if (tree.get('original_path')!=original_path
                or tree.get('original_sha256')!=source_sha
                or tree.get('derived_sha256')!=derived_sha
                or not isinstance(tree.get('tree'),dict)
                or not isinstance(tree['tree'].get('root_id'),str)):
            raise ValueError('private recursive summary tree belongs to wrong source revision')
        root=tree['tree']['root_id']
        # One current native root summary is independently attached to the
        # exact source. verify_generation already checks ALL tree nodes and
        # valid native vector dimensions, revisions and SHA before pinning.
        native_roots=pinned.native.execute('''SELECT r.generation_json,r.representation_text_sha256
               FROM content_representations r JOIN content_chunks c ON r.chunk_id=c.chunk_id
               WHERE c.filesystem_entry_id=? AND r.representation_kind='summary_long'
                 AND r.analysis_status='complete' ''',(file_id,)).fetchall()
        if len(native_roots)!=1:
            raise ValueError('private native root summary missing or duplicated')
        try:root_meta=json.loads(native_roots[0][0])
        except (ValueError,TypeError) as exc:
            raise ValueError('private native root summary metadata invalid') from exc
        if (not isinstance(root_meta,dict) or root_meta.get('node_id')!=root
                or root_meta.get('original_sha256')!=source_sha
                or root_meta.get('derived_sha256')!=derived_sha
                or root_meta.get('summary_sha256')!=native_roots[0][1]):
            raise ValueError('private native root summary belongs to another source revision')
        candidates=source_evidence_candidates(
            original_path=original_path,original_title=Path(original_path).name,
            filesystem_entry_id=file_id,original_sha256=source_sha,
            derived_sha256=derived_sha,derived_text=text,query=query,top=top)
        locators=[]
        for c in candidates['passages']:
            begin,end=c['excerpt_start_char'],c['excerpt_end_char']
            start,stop=c['source_start_char'],c['source_end_char']
            if (c['original_path']!=original_path or c['filesystem_entry_id']!=file_id
                    or c['original_sha256']!=source_sha or c['derived_sha256']!=derived_sha
                    or type(start) is not int or type(stop) is not int
                    or type(begin) is not int or type(end) is not int
                    or not 0<=start<=begin<end<=stop<=len(text)
                    or not isinstance(c['source_section'],str)
                    or not re.fullmatch(r'PDF page [1-9][0-9]{0,3}',c['source_section'])
                    or _sha(text[start:stop].encode('utf8'))!=c['source_span_sha256']
                    or _sha(text[begin:end].encode('utf8'))!=c['excerpt_sha256']):
                raise ValueError('private source page locator is not exact current generated text')
            locators.append({'source_section':c['source_section'],
                'excerpt_start_char':begin,'excerpt_end_char':end,
                'excerpt_sha256':c['excerpt_sha256'],
                'matched_source_terms':c['matched_source_terms'],
                'original_pdf_page_independently_verified':False,
                'original_publisher_quote_verified':False})
        result={'generation':pinned.generation,'original_path':original_path,
                'original_sha256':source_sha,'derived_sha256':derived_sha,
                'native_filedb_original_id':file_id,
                'pinned_complete_summary_tree_sha256':_sha(tree_bytes),
                'pinned_native_summary_root_id':root,
                'locators':locators,'returned_source_locators':len(locators),
                'all_returned_locations_current_derived_text_SHA_verified':True,
                'source_location_is_private_provisional_machine_text':True,
                'original_publisher_pdf_physical_page_verified':False,
                'source_factual_claims_verified':False,
                'private_generation_only':True,
                'all_live_readers_and_writers_enrolled':False,
                'safe_to_cutover_live':False}
    # The context's final full-generation check has now completed; nothing
    # is surfaced from partially verified private data or temporary SQLite.
    return result
