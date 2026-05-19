"""Internal query planning helpers for bounded DSPG retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .extractors import capitalized_phrases, identifiers, urls
from .text import normalize


PREDICATE_TERMS = {
    "author": "author",
    "authored": "author",
    "draft": "author",
    "drafted": "author",
    "review": "review",
    "reviewed": "review",
    "approve": "approve",
    "approved": "approve",
    "merge": "merge",
    "merged": "merge",
    "open": "open",
    "opened": "open",
    "close": "close",
    "closed": "close",
    "reopen": "reopen",
    "reopened": "reopen",
    "report": "report",
    "reported": "report",
    "request": "request",
    "requested": "request",
    "fix": "fix",
    "fixed": "fix",
    "delete": "delete",
    "deleted": "delete",
    "believe": "believe",
    "believes": "believe",
    "allege": "allege",
    "alleges": "allege",
    "own": "own",
    "owns": "own",
    "owned": "own",
    "test": "test",
    "tested": "test",
    "manage": "manage",
    "manages": "manage",
}


@dataclass(frozen=True)
class QueryPlan:
    """Small internal plan used to bound retrieval before answering."""

    anchors: tuple[str, ...]
    predicates: tuple[str, ...]
    wants_current_state: bool = False


def plan_question(question: str) -> QueryPlan:
    qnorm = normalize(question)
    anchors: list[str] = []
    anchors.extend(urls(question))
    anchors.extend(identifiers(question))
    anchors.extend(capitalized_phrases(question))

    seen: set[str] = set()
    unique_anchors: list[str] = []
    for anchor in anchors:
        key = normalize(anchor)
        if key and key not in seen:
            seen.add(key)
            unique_anchors.append(anchor)

    predicates: list[str] = []
    for token in re.findall(r"[a-z]+", qnorm):
        predicate = PREDICATE_TERMS.get(token)
        if predicate and predicate not in predicates:
            predicates.append(predicate)

    wants_current_state = any(term in qnorm for term in ["current state", "final state", "latest state"])
    return QueryPlan(tuple(unique_anchors), tuple(predicates), wants_current_state)
