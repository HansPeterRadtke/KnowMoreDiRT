"""Read-only, SHA-attested first-party source links into the existing archive.

Links must occur in the original HTML article/main text, resolve to one exact
same-origin approved publisher URL, and have one completed archive receipt.
No downloading, browsing, network calls, new indexes, or factual-answer claims.
"""
from __future__ import annotations
import hashlib
from html.parser import HTMLParser
from functools import lru_cache
import re
from urllib.parse import urljoin, urlsplit, urlunsplit

from .hybrid_search import query_terms

_SHA=re.compile(r'^[a-f0-9]{64}$')
_MAX_HTML=8*1024*1024


def _sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()


def _text(s:str)->str:return ' '.join(s.split())


def _canonical_url(url:str)->str|None:
    # The approved URL set is reloaded and validated for every lookup, but
    # canonicalization itself is a pure string operation. In multi-source
    # discovery, the same ~18k original publisher URLs were reparsed for
    # every parent HTML source. Cache ONLY canonicalization, never approved
    # catalog/ledger identities, live files, source-map versions or receipts.
    if not isinstance(url,str) or not 8<=len(url)<=2400 or any(ord(c)<33 for c in url):
        return None
    return _canonical_url_string(url)


@lru_cache(maxsize=32768)
def _canonical_url_string(url:str)->str|None:
    try:
        u=urlsplit(url)
    except ValueError:return None
    if (u.scheme!='https' or not u.hostname or u.username is not None
            or u.password is not None or u.port not in (None,443)
            or u.query or '\\' in u.path):
        return None
    return urlunsplit(('https',u.hostname.lower(),u.path.rstrip('/') or '/', '', ''))


class _Anchors(HTMLParser):
    _EMPTY=frozenset(('area','base','br','col','embed','hr','img','input',
                      'link','meta','param','source','track','wbr'))
    _INERT=frozenset(('script','style','noscript','template','svg','canvas','nav','aside','footer'))

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack=[]
        self.active=[]
        self.groups={'article':[],'main':[]}

    def handle_starttag(self,tag,attrs):
        if tag=='a':
            href=dict(attrs).get('href')
            if (isinstance(href,str) and not any(x in self.stack for x in self._INERT)
                    and not any(x['tag']=='a' for x in self.active)):
                role='article' if 'article' in self.stack else 'main' if 'main' in self.stack else None
                if role:self.active.append({'tag':'a','role':role,'href':href,'parts':[]})
        if tag not in self._EMPTY:self.stack.append(tag)

    def handle_startendtag(self,tag,attrs):return

    def handle_data(self,text):
        if not text or any(t in self.stack for t in self._INERT):return
        if self.active:self.active[-1]['parts'].append(text)

    def handle_endtag(self,tag):
        if tag=='a' and self.active:
            item=self.active.pop()
            label=_text(' '.join(item['parts']))
            if label and 2<=len(label)<=300:
                self.groups[item['role']].append((item['href'],label))
        if tag in self.stack:
            while self.stack:
                name=self.stack.pop()
                if name==tag:break


def source_link_candidates(raw_html:bytes,original_sha256:str,publisher_url:str,
                          catalog:list[dict],completed_receipts:dict[str,dict],
                          query:str,*,original_path:str,limit:int=3)->dict:
    """Find *possible* already archived follow-up pages from one original.

    This validates the source HTML bytes and catalogue/receipt URL metadata.
    Caller MUST full-SHA verify every destination's actual original and derived
    bytes and its canonical native ID before treating it as source evidence.
    """
    if (not isinstance(raw_html,bytes) or not 1<=len(raw_html)<=_MAX_HTML
            or not isinstance(original_sha256,str) or not _SHA.fullmatch(original_sha256)
            or not isinstance(original_path,str) or not original_path
            or original_path.startswith('/') or '\\' in original_path
            or any(p in ('','.','..') for p in original_path.split('/'))
            or type(limit) is not int or not 1<=limit<=5
            or not isinstance(catalog,list) or not 1<=len(catalog)<=50_000
            or not isinstance(completed_receipts,dict)
            or not isinstance(query,str) or not 1<=len(query)<=300):
        raise ValueError('invalid bounded original HTML archive link selection')
    current_url=_canonical_url(publisher_url)
    if current_url is None or _sha(raw_html)!=original_sha256:
        raise ValueError('original publisher HTML URL or complete source SHA invalid')
    try:html=raw_html.decode('utf8','strict')
    except UnicodeDecodeError as e:raise ValueError('original HTML is not UTF-8') from e
    parser=_Anchors();parser.feed(html);parser.close()
    references=parser.groups['article'] or parser.groups['main']
    if len(references)>200:
        raise ValueError('unbounded original HTML article references')
    permitted={}
    ambiguous=set()
    known_catalog_urls=set()
    for entry in catalog:
        if isinstance(entry,dict):
            known_url=_canonical_url(entry.get('url'))
            if known_url:known_catalog_urls.add(known_url)
    for entry in catalog:
        if not isinstance(entry,dict):continue
        path=entry.get('path');url=_canonical_url(entry.get('url'))
        if (not isinstance(path,str) or not path or path.startswith('/')
                or '\\' in path or any(x in ('','.','..') for x in path.split('/'))
                or entry.get('kind') not in ('html','text')
                or url is None):continue
        if url in permitted and permitted[url].get('path')!=path:
            ambiguous.add(url)
        else:permitted[url]=entry
    terms=set(query_terms(query))
    if not terms:raise ValueError('no informative question terms for archived link candidates')
    candidates={}
    missing={}
    for index,(href,label) in enumerate(references):
        try:target=_canonical_url(urljoin(publisher_url,href))
        except ValueError:continue
        if target is None or urlsplit(target).hostname!=urlsplit(current_url).hostname:
            continue
        if target==current_url or target in ambiguous:continue
        entry=permitted.get(target)
        if entry and entry['path']==original_path:continue
        title=(entry or {}).get('title') or ''
        if not isinstance(title,str) or len(title)>500:continue
        found=sorted(terms.intersection(query_terms(label+' '+title)))
        if not found:continue
        receipt=completed_receipts.get(entry['path']) if entry else None
        complete=(isinstance(receipt,dict) and receipt.get('status')=='complete'
                and receipt.get('path')==entry['path']
                and _canonical_url(receipt.get('url'))==target
                and isinstance(receipt.get('sha256'),str)
                and _SHA.fullmatch(receipt['sha256'])
                and type(receipt.get('bytes_received')) is int
                and receipt['bytes_received']>0)
        if not complete:
            # The original first-party page really contains this reference,
            # but the destination is NOT admitted as archived source evidence.
            # In particular, never download a missing article as a side effect.
            item={'original_link_index':index,'source_original_path':original_path,
                  'source_original_sha256':original_sha256,
                  'publisher_article_link_text':label,'publisher_url':target,
                  'matched_query_terms':found,
                  'archive_status':('not_in_existing_catalog' if target not in known_catalog_urls
                       else 'catalogued_but_not_approved_completed_html_or_text'),
                  'publisher_destination_contents_unavailable':True,
                  'no_download_or_external_navigation':True,
                  'not_a_source_excerpt_or_answer':True}
            prev=missing.get(target)
            if prev is None or (len(found),-index)>(len(prev['matched_query_terms']),-prev['original_link_index']):
                missing[target]=item
            continue
        value={
            'original_link_index':index,
            'source_original_path':original_path,
            'source_original_sha256':original_sha256,
            'publisher_article_link_text':label,
            'publisher_link_text_sha256':_sha(label.encode('utf8')),
            'publisher_url':target,
            'catalog_target_original_path':entry['path'],
            'catalog_target_title':title,
            'catalog_target_acquisition_sha256':receipt['sha256'],
            'matched_query_terms':found,
            'candidate_not_answer':True,
            'source_link_occurs_in_original_HTML_article_or_main':True,
            'target_original_bytes_verified_by_this_function':False,
        }
        existing=candidates.get(entry['path'])
        if existing is None or (len(value['matched_query_terms']),-index)>(len(existing['matched_query_terms']),-existing['original_link_index']):
            candidates[entry['path']]=value
    ordered=sorted(candidates.values(),key=lambda x:(-len(x['matched_query_terms']),x['original_link_index'],x['catalog_target_original_path']))
    unavailable=sorted(missing.values(),key=lambda x:(-len(x['matched_query_terms']),x['original_link_index'],x['publisher_url']))
    return {'original_path':original_path,'original_sha256':original_sha256,
            'original_archived_catalog_reference_count':len(references),
            'matching_verified_catalog_receipt_link_candidates':len(ordered),
            'selected_links':ordered[:limit],
            'same_origin_relevant_unarchived_or_unapproved_references':len(unavailable),
            'unavailable_publisher_references_not_followed':unavailable[:limit],
            'no_external_source_fetch_or_download':True,
            'no_target_original_or_derived_bytes_yet_verified':True,
            'no_answer_truth_or_page_quote_proved':True}
