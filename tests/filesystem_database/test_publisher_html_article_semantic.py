"""Source-window evidence integrity for optional, in-memory local article E5."""
from __future__ import annotations
import hashlib
import math
import unittest
from pathlib import Path
import numpy as np

from file_system_catalog.publisher_html_article_discovery import ArticleEntry
from file_system_catalog.publisher_html_article_semantic import (
    OriginalArticleSemanticIndex,bounded_semantic_article_windows,
    MAX_WINDOWS_PER_ORIGINAL,
)


class FakeLocalE5:
    embedding_dimension=3
    model='mock-local-e5-for-unit-tests-only'
    revision='mocked-input-no-network'
    def token_count(self,s):
        return len(s.split())+4
    def _vec(self,s):
        low=s.lower()
        vals=[1.0 if 'roughness' in low else 0.0,
              1.0 if 'sprite' in low else 0.0,
              1.0 if 'pixel' in low else 0.0]
        if sum(vals)==0:vals[2]=0.1
        row=np.asarray(vals,dtype=np.float32)
        return row / np.linalg.norm(row)
    def embed(self,ss):
        return [self._vec(z) for z in ss]
    def embed_queries(self,ss):
        return [self._vec(z) for z in ss]


def article(path,title,text):
    url='https://manual.example.org/4.5/'+path
    return ArticleEntry((path,'a'*64,'b'*64,path+'.txt','html',url,title,
                         len(text),len(text),'complete'),
                        'c'*64,'d'*64,url,'article',text)


class BoundedPublisherArticleSemanticTests(unittest.TestCase):
    def test_every_full_original_publisher_word_is_covered_without_truncation(self):
        text=' '.join('sentence%d' % i for i in range(940))
        e=article('page.html','Short reference',text)
        windows=bounded_semantic_article_windows(FakeLocalE5(),e)
        self.assertGreater(len(windows),1)
        self.assertLessEqual(len(windows),MAX_WINDOWS_PER_ORIGINAL)
        self.assertEqual(text.split(), [
            word for window in windows
            for word in text[window.source_start_char:window.source_end_char].split()])
        self.assertTrue(all(window.embedded_text.startswith('[Catalog title metadata]')
                            for window in windows))
        self.assertTrue(all(FakeLocalE5().token_count(w.embedded_text)<=400 for w in windows))

    def test_source_article_similarity_and_metadata_title_are_separate_channels(self):
        a=article('roughness.html','Surface materials','Microfacet roughness modifies reflections and light scatter.')
        b=article('sprite.html','Text information','In Unity sprite assets supply rich image labels.')
        model=FakeLocalE5()
        e=OriginalArticleSemanticIndex([a,b],model)
        gold=e.ranked('What is a sprite asset?',channel='article',limit=2)
        self.assertEqual('sprite.html',gold[0]['original_path'])
        self.assertTrue(gold[0]['source_excerpt_is_whitespace_normalized_publisher_article_not_literal_HTML_bytes'])
        self.assertFalse(gold[0]['factual_answer_verified'])
        start=gold[0]['normalized_original_article_span_start_char']
        end=gold[0]['normalized_original_article_span_end_char']
        self.assertEqual(b.text[start:end],gold[0]['source_excerpt'])
        self.assertEqual(hashlib.sha256(gold[0]['source_excerpt'].encode()).hexdigest(),
                         gold[0]['source_excerpt_sha256'])
        title=e.ranked('material surface',channel='catalog_title',limit=2)
        self.assertTrue(all(x['title_only_rank_is_not_original_article_content_evidence'] for x in title))
        self.assertTrue(all(x['rank_channel']=='local_E5_catalog_metadata_title' for x in title))

    def test_bad_source_text_fails_closed_instead_of_silent_drop(self):
        model=FakeLocalE5()
        for text in ('','  paragraph   with  double  spaces','word '*20000):
            with self.subTest(size=len(text)),self.assertRaises(ValueError):
                bounded_semantic_article_windows(model,article('a.html','Original',text))
        with self.assertRaises(ValueError):
            OriginalArticleSemanticIndex([
                article('same.html','Title','Normal publisher text'),
                article('same.html','Different','Other publisher text'),
            ],model)

    def test_invalid_query_and_non_normalized_model_vectors_refused(self):
        a=article('doc.html','A guide','Valid publisher sprite source data.')
        model=FakeLocalE5()
        idx=OriginalArticleSemanticIndex([a],model)
        for q in ('','??','x'*601):
            with self.subTest(query=q[:15]),self.assertRaises(ValueError):
                idx.ranked(q)
        class BadE5(FakeLocalE5):
            def embed(self,ss):
                return [np.asarray([float('nan'),0.0,0.0],dtype=np.float32) for _ in ss]
        with self.assertRaises(ValueError):
            OriginalArticleSemanticIndex([a],BadE5())

    def test_no_cross_source_excerpt_in_ranked_results(self):
        a=article('a.html','A source','sprite representations in data')
        b=article('b.html','B source','roughness parameters in shader')
        idx=OriginalArticleSemanticIndex([a,b],FakeLocalE5())
        for channel in ('article','catalog_title'):
            for hit in idx.ranked('shader roughness',channel=channel,limit=2):
                corresponding=a if hit['original_path']=='a.html' else b
                self.assertIn(hit['source_excerpt'],corresponding.text)
                self.assertFalse(hit['factual_answer_verified'])
                self.assertFalse(hit['semantic_similarity_not_calibrated_probability']<0)
