"""Read-only approved first-party links with original publisher Markdown byte proof.

Unlike parsed HTML DOM evidence, every accepted Markdown link identifies the
literal `[...]()` byte interval in the ORIGINAL acquired source, not derived
text or model-generated content. Excludes fenced/indented/inline code, images,
external origins and unapproved/ambiguous destination receipts. Never fetches.
"""
from __future__ import annotations
import hashlib
import re
from urllib.parse import urljoin,urlsplit

from .archived_html_source_links import _canonical_url,_SHA
from .hybrid_search import query_terms

# Intentionally narrow: supports ordinary inline [label](href) links from
# first-party Markdown. Nested brackets, reference-style links, malformed
# escapes, scripts and fragment-only links are NOT silently interpreted.
_LINK=re.compile(r'(?<!\\)(?<!!)\[([^\]\r\n]{2,240})\]\(([^\s)\r\n]{1,2048})\)')
_FENCE=re.compile(r'^[ \t]{0,3}(`{3,}|~{3,})')
_MAX_BYTES=8*1024*1024


def _sha(blob:bytes)->str:return hashlib.sha256(blob).hexdigest()


def _literal_links(raw:bytes):
    try:decoded=raw.decode('utf8','strict')
    except UnicodeDecodeError as e:raise ValueError('publisher Markdown not valid UTF-8') from e
    offset=0;fence=None;inline_backtick_run=None;html_comment=False;results=[]
    for line in decoded.splitlines(keepends=True):
        current=line.lstrip(' \t')
        marker=_FENCE.match(line)
        if marker:
            symbol=marker[1][0];run=len(marker[1])
            if fence is None:fence=(symbol,run)
            elif fence[0]==symbol and run>=fence[1]:fence=None
            offset+=len(line.encode('utf8'))
            continue
        if fence is not None or line.startswith(('    ','\t')):
            offset+=len(line.encode('utf8'));continue
        # Ignore any line intersecting an HTML comment, including inline
        # comments and comments spanning multiple lines. This intentionally
        # forgoes valid adjacent links rather than promoting a fake source.
        if html_comment or '<!--' in line or '-->' in line:
            if html_comment:
                if '-->' in line:html_comment=False
            elif '<!--' in line and ('-->' not in line or line.index('-->')<line.index('<!--')):
                html_comment=True
            offset+=len(line.encode('utf8'));continue
        # Markdown permits double/triple backtick inline code spans, and
        # inline code may span lines. Counting all backticks modulo two was
        # incorrect: two opening backticks could promote an example link as
        # a real publisher hyperlink. Match the exact delimiter run length,
        # even when the closing run occurs on a later line.
        events=[(hit.start(),1,hit) for hit in _LINK.finditer(line)]
        events.extend((run.start(),0,run) for run in re.finditer(r'`+',line))
        events.sort(key=lambda item:(item[0],item[1]))
        for _position,kind,hit in events:
            if kind==0:
                length=len(hit[0])
                if inline_backtick_run is None:
                    inline_backtick_run=length
                elif inline_backtick_run==length:
                    inline_backtick_run=None
                continue
            if inline_backtick_run is not None:
                continue
            label=hit[1].strip();href=hit[2]
            if not label or label.startswith(('!','[')) or len(label)>240:
                continue
            left=offset+len(line[:hit.start()].encode('utf8'))
            literal=hit[0].encode('utf8');right=left+len(literal)
            if raw[left:right]!=literal:
                raise ValueError('publisher Markdown original literal offset mismatch')
            results.append((href,label,left,right,_sha(literal)))
            if len(results)>250:raise ValueError('unbounded publisher Markdown link count')
        offset+=len(line.encode('utf8'))
    return results


def markdown_link_candidates(raw_markdown:bytes,original_sha256:str,publisher_url:str,
                             catalog:list[dict],completed_receipts:dict[str,dict],
                             query:str,*,original_path:str,limit:int=3,
                             exact_link_text:str|None=None)->dict:
    """Choose link candidates; selected targets need later full original SHA checks.

    `exact_link_text` is navigation to an explicit publisher anchor, NOT a
    claimed semantic query hit or sufficient factual answer.
    """
    if (not isinstance(raw_markdown,bytes) or not 1<=len(raw_markdown)<=_MAX_BYTES
        or not isinstance(original_sha256,str) or not _SHA.fullmatch(original_sha256)
        or not isinstance(original_path,str) or not original_path.endswith('.md')
        or original_path.startswith('/') or '\\' in original_path
        or any(t in ('','.','..') for t in original_path.split('/'))
        or type(limit) is not int or not 1<=limit<=5
        or not isinstance(catalog,list) or not 1<=len(catalog)<=50000
        or not isinstance(completed_receipts,dict)
        or not isinstance(query,str) or not 1<=len(query)<=300
        or exact_link_text is not None and (not isinstance(exact_link_text,str)
            or not 2<=len(exact_link_text)<=240)):
        raise ValueError('invalid bounded exact original Markdown reference selection')
    parent_url=_canonical_url(publisher_url)
    if parent_url is None or _sha(raw_markdown)!=original_sha256:
        raise ValueError('publisher Markdown source URL or whole original SHA invalid')
    references=_literal_links(raw_markdown)
    permitted={};ambiguous=set();known=set()
    for entry in catalog:
        if isinstance(entry,dict):
            x=_canonical_url(entry.get('url'))
            if x:known.add(x)
    for entry in catalog:
        if not isinstance(entry,dict):continue
        path=entry.get('path');url=_canonical_url(entry.get('url'))
        if (not isinstance(path,str) or not path or path.startswith('/') or '\\' in path
                or any(t in ('','.','..') for t in path.split('/'))
                or entry.get('kind') not in ('html','text') or url is None):continue
        if url in permitted and permitted[url]['path']!=path:ambiguous.add(url)
        else:permitted[url]=entry
    terms=set(query_terms(query))
    if not terms and exact_link_text is None:
        raise ValueError('Markdown source navigation needs informative query terms or an exact link name')
    candidates={};unavailable={}
    for index,(href,label,start,end,literal_sha) in enumerate(references):
        try:target=_canonical_url(urljoin(publisher_url,href))
        except ValueError:continue
        if (target is None or urlsplit(target).hostname!=urlsplit(parent_url).hostname
                or target==parent_url or target in ambiguous):continue
        entry=permitted.get(target)
        if entry and entry['path']==original_path:continue
        title=(entry or {}).get('title') or ''
        if not isinstance(title,str) or len(title)>500:continue
        matching=sorted(terms.intersection(query_terms(label+' '+title)))
        explicitly_selected=exact_link_text is not None and label.casefold()==exact_link_text.casefold()
        if exact_link_text is not None and not explicitly_selected:continue
        if not matching and not explicitly_selected:continue
        receipt=completed_receipts.get(entry['path']) if entry else None
        approved=(isinstance(receipt,dict) and receipt.get('status')=='complete'
            and receipt.get('path')==entry['path']
            and _canonical_url(receipt.get('url'))==target
            and isinstance(receipt.get('sha256'),str) and _SHA.fullmatch(receipt['sha256'])
            and type(receipt.get('bytes_received')) is int and receipt['bytes_received']>0)
        common={'original_link_index':index,'source_original_path':original_path,
                'source_original_sha256':original_sha256,
                'publisher_markdown_link_text':label,
                'publisher_link_literal_start_byte':start,
                'publisher_link_literal_end_byte':end,
                'publisher_link_literal_sha256':literal_sha,
                'publisher_url':target,'matched_query_terms':matching,
                'explicit_navigation_label_not_semantic_relevance':bool(explicitly_selected),
                'publisher_link_occurs_in_SHA_attested_original_Markdown_bytes':True}
        if not approved:
            item={**common,'archive_status':('not_in_existing_catalog' if target not in known else 'catalogued_but_not_approved_completed_source'),
                  'publisher_destination_contents_unavailable':True,
                  'not_a_source_excerpt_or_answer':True,'no_download_or_external_navigation':True}
            previous=unavailable.get(target)
            if previous is None or len(matching)>len(previous['matched_query_terms']):unavailable[target]=item
            continue
        value={**common,'catalog_target_original_path':entry['path'],
               'catalog_target_title':title,'catalog_target_acquisition_sha256':receipt['sha256'],
               'candidate_not_answer':True,'target_original_bytes_verified_by_this_function':False}
        previous=candidates.get(entry['path'])
        if previous is None or (len(matching),-index)>(len(previous['matched_query_terms']),-previous['original_link_index']):
            candidates[entry['path']]=value
    ordered=sorted(candidates.values(),key=lambda x:(-len(x['matched_query_terms']),x['original_link_index'],x['catalog_target_original_path']))
    missing=sorted(unavailable.values(),key=lambda x:(-len(x['matched_query_terms']),x['original_link_index'],x['publisher_url']))
    return {'schema':'kmd.exact_acquired_publisher_markdown_official_reference_candidates.v1',
        'original_path':original_path,'original_sha256':original_sha256,
        'original_Markdown_publisher_literal_reference_count':len(references),
        'approved_matched_catalog_link_candidates':len(ordered),'selected_links':ordered[:limit],
        'missing_or_unapproved_official_reference_count':len(missing),
        'unavailable_publisher_references_not_followed':missing[:limit],
        'original_publisher_entire_byte_SHA_verified':True,
        'destination_source_bytes_not_yet_verified':True,
        'no_network_or_new_source_acquisition':True,
        'not_an_answer_or_semantic_relevance_proof':True}
