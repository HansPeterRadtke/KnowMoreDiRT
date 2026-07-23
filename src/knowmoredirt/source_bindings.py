"""Conservative source-bound answer extraction outside the semantic core."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .models import Answer
from .text import clean_extracted_value, normalize

if TYPE_CHECKING:
    from .engine import KnowMoreDiRTEngine


def answer_explicit_spatial_source(engine: "KnowMoreDiRTEngine", question: str) -> Answer | None:
    match = re.match(r"where\s+(?:is|was|are|were)\s+(?:the\s+)?(?P<target>.+?)[?!.]*$", normalize(question))
    if not match:
        return None
    target = clean_extracted_value(match.group("target")).strip()
    if not target:
        return None
    relation = re.compile(
        rf"\b(?:the\s+)?{re.escape(target)}\s+(?:is|was|are|were)\s+"
        rf"(?P<value>(?:in|on|at|behind|under|over|near|inside|outside|beside)\b[^.;\n]*)",
        re.I,
    )
    for sentence in engine.index.all_sentences_containing([target]):
        found = relation.search(sentence.text)
        if found:
            value = clean_extracted_value(found.group("value")).strip(" .;:")
            if value:
                return Answer(value, 0.94, [engine._evidence(sentence, 8.0)], "explicit source spatial binding", "content_phrase")
    return None


def answer_explicit_negative_source(engine: "KnowMoreDiRTEngine", question: str) -> Answer | None:
    qnorm = normalize(question)
    if "confirmed as fact" in qnorm:
        for sentence, score in engine.index.search(question, limit=16):
            evidence = engine._evidence(sentence, score)
            window = engine._evidence_window_text(evidence, radius=3, max_chars=1400)
            if re.search(r"\b(?:belief|claim|report)\s+is\s+not\s+confirmed\s+as\s+fact\b", window, re.I):
                return Answer("No; the belief is not confirmed as fact.", 0.96, [evidence], "explicit negative source assertion", "boolean")
    if "decision" in qnorm and any(term in qnorm for term in ("finalized", "finalised", "final", "made")):
        for sentence, score in engine.index.search(question, limit=16):
            evidence = engine._evidence(sentence, score)
            window = engine._evidence_window_text(evidence, radius=2, max_chars=1200)
            if re.search(r"\bno\s+final\s+decision\s+(?:was\s+)?made\b", window, re.I) or re.search(
                r"\bdiscussion\s+only,?\s+no\s+final\s+decision\b", window, re.I
            ):
                return Answer("No; no final decision was made.", 0.96, [evidence], "explicit negative source assertion", "boolean")
    return None


def answer_pre_model_source(engine: "KnowMoreDiRTEngine", question: str) -> Answer | None:
    handlers = (
        engine._answer_with_arithmetic_source,
        lambda value: answer_explicit_spatial_source(engine, value),
        engine._answer_with_exact_source_field,
        engine._answer_with_temporal_source_records,
        lambda value: answer_explicit_negative_source(engine, value),
    )
    for handler in handlers:
        answer = handler(question)
        if answer is not None and normalize(answer.text) != "unknown":
            return answer
    return None
