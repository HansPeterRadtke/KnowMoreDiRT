"""Conservative source recovery after a model returns no grounded answer."""
from __future__ import annotations

import re
from typing import Any

from .models import Answer
from .text import content_tokens, normalize


def recover_after_unknown(engine: Any, question: str, prior_answer: Answer | None = None) -> Answer | None:
    qnorm = normalize(question)
    if re.search(r"\b(?:person|actor|badge|case|parcel|asset|audit)\s+id\b", qnorm):
        answer = engine._answer_with_labeled_attribute_source(question)
        if answer is not None:
            return answer
    if "current state" in qnorm or "final state" in qnorm:
        answer = engine._answer_with_temporal_source_records(question, prior_answer)
        if answer is not None:
            return answer
    if not re.match(r"^(?:is|was|were|did|does|do|has|have|had)\b", qnorm):
        return None
    belief_name_match = re.search(r"^(?:is|was)\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+belief\b", question, re.I)
    if belief_name_match:
        name = normalize(belief_name_match.group("name"))
    else:
        name_match = re.search(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", question)
        name = normalize(name_match.group(1)) if name_match else ""
    if "belief" in qnorm and "confirmed as fact" in qnorm:
        for sentence in engine.sentences:
            material = normalize(sentence.text)
            if "belief is not confirmed as fact" not in material:
                continue
            same_source = " ".join(
                normalize(other.text)
                for other in engine.sentences
                if other.rel_path == sentence.rel_path
            )
            if name and name not in same_source:
                continue
            if "belief is not confirmed as fact" in material:
                return Answer(
                    "No; the belief is not confirmed as fact.",
                    0.95,
                    [engine._evidence(sentence, 1.0)],
                    "explicit negative belief confirmation source",
                    "boolean",
                )
    if "decision" in qnorm and "finalized" in qnorm:
        target_terms = [
            term
            for term in content_tokens(question)
            if term not in {"was", "the", "decision", "finalized", "archive"}
        ]
        for sentence in engine.sentences:
            material = normalize(sentence.text)
            if target_terms and not all(term in material for term in target_terms):
                continue
            if "no final decision" in material:
                return Answer(
                    "No; no final decision was made.",
                    0.95,
                    [engine._evidence(sentence, 1.0)],
                    "explicit no-final-decision source",
                    "boolean",
                )
    return engine._answer_with_boolean_source_explanation(question, prior_answer)
