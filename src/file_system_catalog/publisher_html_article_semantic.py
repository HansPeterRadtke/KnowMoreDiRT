"""Optional private CPU semantic discovery over current original HTML articles.

Only a short-lived RAM index is built; callers must independently verify
the ORIGINAL publisher bytes, extracted source revision, and native FileDB
identity of each returned candidate before displaying it. A normalized
HTML text span is not a literal publisher-byte quote or a factual answer.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any

import numpy as np

from .publisher_html_article_discovery import ArticleEntry

MAX_SEMANTIC_DOCUMENTS = 512
MAX_SEMANTIC_WINDOWS = 2048
MAX_WINDOWS_PER_ORIGINAL = 32
MAX_SOURCE_ARTICLE_TOKENS = 400
MAX_ORIGINAL_WORDS = 300000
_WORD = re.compile(r'\S+')


@dataclass(frozen=True)
class ArticleSpan:
    original_path: str
    source_start_char: int
    source_end_char: int
    embedded_text: str


def bounded_semantic_article_windows(model: Any, entry: ArticleEntry) -> list[ArticleSpan]:
    """Cover full normalized publisher article, never silently clip at 512.

    The title is explicitly marked as source-map metadata, not original
    article words; its characters are outside the returned original span.
    Every original article word is represented in exactly one window, and
    every nonempty returned span is an actual substring of entry.text.
    """
    if (not isinstance(entry, ArticleEntry) or not isinstance(entry.path, str)
            or not entry.path or not isinstance(entry.text, str)
            or not entry.text or not isinstance(entry.title, str)
            or len(entry.title) > 1000 or not callable(getattr(model, 'token_count', None))):
        raise ValueError('invalid current original article semantic input')
    words = list(_WORD.finditer(entry.text))
    if not words or len(words) > MAX_ORIGINAL_WORDS:
        raise ValueError('current publisher article has invalid word population')
    if ' '.join(m.group() for m in words) != entry.text:
        raise ValueError('semantic original article is not normalized')
    metadata_prefix = '[Catalog title metadata] ' + entry.title + '\n[Original publisher article] '
    if model.token_count(metadata_prefix) > MAX_SOURCE_ARTICLE_TOKENS // 2:
        raise ValueError('catalog title itself exceeds original semantic window bound')
    windows=[]
    left=0
    while left < len(words):
        # Bounded binary search: choose the longest word-aligned span that
        # fits the actual pinned model tokenizer, including its E5 prefix.
        lower=left+1
        upper=min(len(words),left+600)
        best=None
        while lower<=upper:
            middle=(lower+upper)//2
            a=words[left].start()
            b=words[middle-1].end()
            trial=metadata_prefix+entry.text[a:b]
            if model.token_count(trial) <= MAX_SOURCE_ARTICLE_TOKENS:
                best=(middle,a,b,trial)
                lower=middle+1
            else:
                upper=middle-1
        if best is None:
            raise ValueError('one publisher original word exceeds trained semantic context')
        next_word,a,b,trial=best
        if not (0<=a<b<=len(entry.text)):
            raise ValueError('original semantic span out of bounds')
        windows.append(ArticleSpan(entry.path,a,b,trial))
        if len(windows)>MAX_WINDOWS_PER_ORIGINAL:
            raise ValueError('original HTML requires more than bounded per-source semantic windows')
        left=next_word
    actual_words=[m.group() for w in windows for m in _WORD.finditer(
        entry.text[w.source_start_char:w.source_end_char])]
    if actual_words != [m.group() for m in words]:
        raise ValueError('bounded semantic windows dropped or duplicated original words')
    return windows


def _strict_normalized_matrix(rows: list[Any], length: int, dimension: int) -> np.ndarray:
    if not isinstance(rows,list) or len(rows)!=length or not 1<=dimension<=4096:
        raise ValueError('model returned invalid semantic batch shape')
    matrix=np.asarray(rows,dtype=np.float32)
    if matrix.shape!=(length,dimension) or not np.isfinite(matrix).all():
        raise ValueError('model returned nonfinite or incorrect semantic vector')
    norms=np.linalg.norm(matrix,axis=1)
    if not np.isfinite(norms).all() or (norms<0.99).any() or (norms>1.01).any():
        raise ValueError('E5 model semantic vectors are not unit-normalized')
    return matrix


class OriginalArticleSemanticIndex:
    """Two independent local E5 discovery channels with real publisher spans.

    Neither cross-document scores nor the title-only channel may be claimed
    as verified answers. The caller controls source verification/rechecks.
    """

    def __init__(self, entries: list[ArticleEntry], model: Any):
        if (not isinstance(entries,list) or not 1<=len(entries)<=MAX_SEMANTIC_DOCUMENTS
                or not isinstance(getattr(model,'embedding_dimension',None),int)
                or not callable(getattr(model,'embed',None))
                or not callable(getattr(model,'embed_queries',None))):
            raise ValueError('invalid bounded original article semantic model or corpus')
        if len(set(e.path for e in entries))!=len(entries):
            raise ValueError('ambiguous duplicate original source article identity')
        self.entries={e.path:e for e in entries}
        self.model=model
        self.windows=[]
        for entry in entries:
            self.windows+=bounded_semantic_article_windows(model,entry)
            if len(self.windows)>MAX_SEMANTIC_WINDOWS:
                raise ValueError('source corpus exceeds bounded ephemeral semantic window budget')
        dim=model.embedding_dimension
        self.article_vectors=_strict_normalized_matrix(
            model.embed([w.embedded_text for w in self.windows]),len(self.windows),dim)
        self.title_vectors=_strict_normalized_matrix(
            model.embed(['[Catalog title metadata] '+e.title for e in entries]),len(entries),dim)
        self.title_paths=[e.path for e in entries]
        self.dimension=dim

    def ranked(self, query: str, *, channel: str='article', limit: int=10) -> list[dict]:
        if (not isinstance(query,str) or not 3<=len(query.strip())<=600
                or channel not in ('article','catalog_title')
                or type(limit) is not int or not 1<=limit<=10):
            raise ValueError('invalid bounded original semantic retrieval query')
        vector=_strict_normalized_matrix(self.model.embed_queries([query]),1,self.dimension)[0]
        best={}
        if channel=='article':
            scores=self.article_vectors@vector
            for window,score in zip(self.windows,scores):
                identity=window.original_path
                prior=best.get(identity)
                if prior is None or float(score)>prior[0]:
                    best[identity]=(float(score),window.source_start_char,window.source_end_char)
        else:
            scores=self.title_vectors@vector
            for source,score in zip(self.title_paths,scores):
                prior=best.get(source)
                if prior is not None:
                    raise ValueError('ambiguous original title identity')
                best[source]=(float(score),0,min(len(self.entries[source].text),600))
        ordered=sorted(best,key=lambda key:(-best[key][0],key))[:limit]
        out=[]
        for path in ordered:
            score,start,end=best[path]
            entry=self.entries[path]
            source=entry.text[start:end]
            if not source or not math.isfinite(score):
                raise ValueError('original semantic excerpt unavailable')
            out.append({
                'original_path':path,
                'semantic_similarity_not_calibrated_probability':round(score,6),
                'rank_channel':('local_E5_original_article_window' if channel=='article'
                                else 'local_E5_catalog_metadata_title'),
                'source_excerpt':source,
                'source_excerpt_sha256':hashlib.sha256(source.encode()).hexdigest(),
                'normalized_original_article_span_start_char':start,
                'normalized_original_article_span_end_char':end,
                'source_excerpt_is_whitespace_normalized_publisher_article_not_literal_HTML_bytes':True,
                'title_only_rank_is_not_original_article_content_evidence':channel=='catalog_title',
                'factual_answer_verified':False,
            })
        return out
