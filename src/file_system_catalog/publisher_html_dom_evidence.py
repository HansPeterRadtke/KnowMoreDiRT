"""Independently anchor an extracted HTML source excerpt to publisher DOM text.

One SHA-attested original HTML and one already source-hashed derived excerpt are
required. Only bounded matching text in original article/main elements qualifies.
Whitespace and HTML character references are normalized. This does NOT prove
literal publisher byte coordinates, page layout, visibility or answer truth.
No external requests, full browser, generated HTML or persistent index needed.
"""
from __future__ import annotations
import hashlib
from html.parser import HTMLParser
import re

MAX_HTML_BYTES=8*1024*1024
_SHA=re.compile(r'^[a-f0-9]{64}$')


def _sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def _collapse(text:str)->str:
    return ' '.join(text.split())


class _PublisherArticleParser(HTMLParser):
    """Independent parser for main/article visible-ish text; scripts excluded."""
    _VOID=frozenset(('area','base','br','col','embed','hr','img','input','link',
                     'meta','param','source','track','wbr'))
    _EXCLUDE=frozenset(('script','style','noscript','template','svg','canvas'))

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags=[]
        self.text={'article':[],'main':[]}

    def handle_starttag(self,tag,attrs):
        if tag not in self._VOID:
            self.tags.append(tag)

    def handle_startendtag(self,tag,attrs):
        return

    def handle_endtag(self,tag):
        if tag in self.tags:
            while self.tags:
                popped=self.tags.pop()
                if popped==tag:break

    def handle_data(self,value):
        if not value or any(tag in self.tags for tag in self._EXCLUDE):
            return
        for container in self.text:
            if container in self.tags:self.text[container].append(value)


def attach_original_html_dom_anchors(raw:bytes,expected_original_sha256:str,passages:list[dict])->int:
    """Attach *only* uniquely verified normalized article/main text anchors.

    This checks the complete original SHA again, then parses independent
    publisher HTML. It never invents a byte span or promotes a lexical match
    into a correct answer. Results without unique full normalized DOM matches
    remain extracted-text candidates (no anchor field).
    """
    if (not isinstance(raw,bytes) or not 1<=len(raw)<=MAX_HTML_BYTES
            or not isinstance(expected_original_sha256,str)
            or not _SHA.fullmatch(expected_original_sha256)
            or not isinstance(passages,list) or len(passages)>50):
        raise ValueError('invalid bounded publisher HTML provenance input')
    if _sha(raw)!=expected_original_sha256:
        raise ValueError('publisher HTML original SHA differs from approved current source')
    try:
        content=raw.decode('utf8','strict')
    except UnicodeError as exc:
        raise ValueError('publisher HTML not safely UTF-8 decodable') from exc
    parser=_PublisherArticleParser()
    parser.feed(content)
    parser.close()
    containers={key:_collapse(' '.join(values)) for key,values in parser.text.items()}
    accepted=0
    for row in passages:
        if not isinstance(row,dict):raise ValueError('invalid extracted source passage row')
        excerpt=row.get('source_excerpt')
        digest=row.get('excerpt_sha256')
        if (not isinstance(excerpt,str) or not excerpt.strip()
                or not isinstance(digest,str) or not _SHA.fullmatch(digest)
                or _sha(excerpt.encode('utf8'))!=digest):
            raise ValueError('derived source excerpt SHA inconsistent before original HTML anchor')
        normalized=_collapse(excerpt)
        if not 80<=len(normalized)<=2000:
            continue  # Too little publisher-specific visible text to anchor.
        for role in ('article','main'):
            body=containers[role]
            if not body:continue
            if body.count(normalized)!=1:
                continue
            start=body.find(normalized)
            if start<0:continue
            row['publisher_html_dom_anchor']={
                'status':'verified_normalized_original_article_text',
                'container':role,
                'original_sha256':expected_original_sha256,
                'original_normalized_text_start_char':start,
                'original_normalized_text_end_char':start+len(normalized),
                'normalized_excerpt_sha256':_sha(normalized.encode('utf8')),
                'normalization':'HTML character references decoded and whitespace collapsed',
                'literal_publisher_byte_interval_verified':False,
                'original_pdf_page_verified':False,
                'answer_factual_sufficiency_verified':False,
            }
            accepted+=1
            break
    return accepted
