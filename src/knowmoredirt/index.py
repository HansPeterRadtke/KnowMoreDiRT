"""Lightweight lexical index over raw text."""

from __future__ import annotations

from collections import Counter, defaultdict

from .models import Sentence
from .text import content_tokens, normalize, tokenize


class LexicalIndex:
    def __init__(self, sentences: list[Sentence]) -> None:
        self.sentences = sentences
        self._tokens_by_sentence: dict[str, Counter[str]] = {}
        self._postings: dict[str, set[str]] = defaultdict(set)
        self._by_id = {sentence.sentence_id: sentence for sentence in sentences}
        for sentence in sentences:
            counts = Counter(tokenize(sentence.text))
            self._tokens_by_sentence[sentence.sentence_id] = counts
            for token in counts:
                self._postings[token].add(sentence.sentence_id)

    def search(self, query: str, limit: int = 12, required: list[str] | None = None) -> list[tuple[Sentence, float]]:
        query_tokens = content_tokens(query)
        required_tokens = [normalize(item) for item in required or [] if normalize(item)]
        if not query_tokens and not required_tokens:
            return []

        candidate_ids: set[str] = set()
        for token in query_tokens + required_tokens:
            candidate_ids.update(self._postings.get(token, set()))

        scored: list[tuple[Sentence, float]] = []
        query_counter = Counter(query_tokens)
        for sentence_id in candidate_ids:
            sentence = self._by_id[sentence_id]
            haystack = normalize(sentence.text)
            if required_tokens and not all(item in haystack for item in required_tokens):
                continue
            counts = self._tokens_by_sentence[sentence_id]
            score = 0.0
            for token, weight in query_counter.items():
                score += min(counts.get(token, 0), weight)
            for item in required_tokens:
                if item in haystack:
                    score += 4.0
            if score:
                scored.append((sentence, score))
        scored.sort(key=lambda item: (-item[1], item[0].rel_path, item[0].order))
        return scored[:limit]

    def all_sentences_containing(self, terms: list[str]) -> list[Sentence]:
        required = [normalize(term) for term in terms if normalize(term)]
        out: list[Sentence] = []
        for sentence in self.sentences:
            text = normalize(sentence.text)
            if all(term in text for term in required):
                out.append(sentence)
        return out

