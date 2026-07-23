from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Sequence

@dataclass(frozen=True)
class Candidate:
    referent_id: str
    entity_type: str
    aliases: frozenset[str] = frozenset()
    attributes: Mapping[str, str] = field(default_factory=dict)
    active_from: int | None = None
    active_to: int | None = None
    vector_score: float = 0.0
    local_recency: float = 0.0

@dataclass(frozen=True)
class Mention:
    text: str
    entity_type: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)
    year: int | None = None
    explicit_new: bool = False

@dataclass(frozen=True)
class Decision:
    state: str
    referent_id: str | None
    score: float
    margin: float
    reasons: tuple[str, ...]

EXACT_ATTRIBUTE_WEIGHTS = {
    'email': 1.00,
    'account_id': 1.00,
    'employee_id': 1.00,
    'organization': 0.25,
    'product': 0.20,
    'city': 0.12,
    'role': 0.10,
}

def incompatible(mention: Mention, candidate: Candidate) -> tuple[bool, list[str]]:
    reasons=[]
    if mention.entity_type and candidate.entity_type != mention.entity_type:
        reasons.append('entity_type_conflict')
    for key in ('email','account_id','employee_id'):
        mv=mention.attributes.get(key);cv=candidate.attributes.get(key)
        if mv and cv and mv != cv: reasons.append(f'{key}_conflict')
    if mention.year is not None:
        if candidate.active_from is not None and mention.year < candidate.active_from: reasons.append('before_active_interval')
        if candidate.active_to is not None and mention.year > candidate.active_to: reasons.append('after_active_interval')
    return bool(reasons),reasons

def score_candidate(mention: Mention, candidate: Candidate) -> tuple[float, list[str]]:
    bad,reasons=incompatible(mention,candidate)
    if bad:return float('-inf'),reasons
    score=0.55*candidate.vector_score+0.10*candidate.local_recency
    normalized=mention.text.casefold().strip(' .,:;')
    if normalized and normalized in {x.casefold() for x in candidate.aliases}:
        score+=0.80;reasons.append('exact_alias')
    for key,value in mention.attributes.items():
        if value and candidate.attributes.get(key)==value:
            score+=EXACT_ATTRIBUTE_WEIGHTS.get(key,0.08);reasons.append(f'{key}_match')
    return score,reasons

def resolve(mention: Mention, candidates: Sequence[Candidate], *, accept_score: float=0.86, accept_margin: float=0.16) -> Decision:
    if mention.explicit_new:return Decision('new',None,1.0,1.0,('explicit_new',))
    ranked=[]
    for c in candidates:
        s,reasons=score_candidate(mention,c)
        if s!=float('-inf'):ranked.append((s,c,reasons))
    if not ranked:return Decision('new',None,0.0,0.0,('no_compatible_candidate',))
    ranked.sort(key=lambda x:x[0],reverse=True)
    best=ranked[0];second=ranked[1][0] if len(ranked)>1 else 0.0;margin=best[0]-second
    if best[0]>=accept_score and margin>=accept_margin:
        return Decision('resolved',best[1].referent_id,best[0],margin,tuple(best[2]))
    return Decision('ambiguous',None,best[0],margin,tuple(best[2]))
