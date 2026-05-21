"""Generic local-model query planning primitives.

This module keeps the optional local-model path generic: it asks a localhost
model to classify a question into broad raw-folder knowledge-system operations.
It does not contain external-evaluation names, special input markers, hidden
answer labels, or dataset-group routing.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any

from .model import LocalModelClient
from .text import normalize


INTENT_GRAMMAR = r'''
root ::= "{" ws "\"query_plan\"" ws ":" ws "{" ws "\"intent\"" ws ":" ws intent ws "," ws "\"target_surface\"" ws ":" ws string ws "," ws "\"answer_role\"" ws ":" ws role ws "," ws "\"requires_asserted\"" ws ":" ws bool ws "}" ws "}"
intent ::= "\"role_lookup\"" | "\"reference_lookup\"" | "\"url_lookup\"" | "\"file_lookup\"" | "\"state_lookup\"" | "\"context_lookup\"" | "\"identity_lookup\"" | "\"grouped_search\"" | "\"unknown\""
role ::= "\"actor\"" | "\"author\"" | "\"reviewer\"" | "\"approver\"" | "\"owner\"" | "\"reporter\"" | "\"assignee\"" | "\"organization\"" | "\"reference\"" | "\"state\"" | "\"unknown\""
bool ::= "true" | "false"
string ::= "\"" chars "\""
chars ::= ([^"\\] | "\\" ["\\/bfnrt])*
ws ::= [ \t\n\r]*
'''

EVIDENCE_EXTRACTION_GRAMMAR = r'''
root ::= "{" ws "\"answer\"" ws ":" ws "{" ws "\"sufficient_evidence\"" ws ":" ws bool ws "," ws "\"answer_type\"" ws ":" ws answer_type ws "," ws "\"answer\"" ws ":" ws string ws "," ws "\"evidence_span\"" ws ":" ws string ws "}" ws "}"
answer_type ::= "\"person\"" | "\"actor\"" | "\"organization\"" | "\"identifier\"" | "\"url\"" | "\"file_path\"" | "\"count\"" | "\"state\"" | "\"date_time\"" | "\"boolean\"" | "\"content_phrase\"" | "\"metadata_value\"" | "\"unknown\""
bool ::= "true" | "false"
string ::= "\"" chars "\""
chars ::= ([^"\\] | "\\" ["\\/bfnrt])*
ws ::= [ \t\n\r]*
'''

REFERENCE_PATTERNS = [
    r"\b[A-Z][A-Z0-9]{1,12}(?:-[A-Z0-9]{2,12})*-\d+[A-Z0-9-]*\b",
    r"https?://[^\s\]\)\"']+",
    r"\b[A-Za-z0-9_./-]+\.(?:cpp|tmp|py|js|md|txt|json|yaml|yml|csv|tsv|pdf)\b",
]

GENERIC_NAMED_ANCHORS = {
    "Find", "What", "Which", "Who", "Where", "When", "How", "Can", "Could",
    "Document", "Report", "Note", "Record", "Guide", "Manual", "IDs", "ID",
}


@dataclass
class ModelQueryTrace:
    enabled: bool = False
    call_count: int = 0
    parsed_count: int = 0
    accepted_count: int = 0
    model_answer_count: int = 0
    evidence_call_count: int = 0
    evidence_parsed_count: int = 0
    evidence_accepted_count: int = 0
    evidence_rejected_count: int = 0
    prompt_hashes: list[str] | None = None
    response_hashes: list[str] | None = None
    last_plan: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "call_count": self.call_count,
            "parsed_count": self.parsed_count,
            "accepted_count": self.accepted_count,
            "model_answer_count": self.model_answer_count,
            "evidence_call_count": self.evidence_call_count,
            "evidence_parsed_count": self.evidence_parsed_count,
            "evidence_accepted_count": self.evidence_accepted_count,
            "evidence_rejected_count": self.evidence_rejected_count,
            "prompt_hashes": self.prompt_hashes or [],
            "response_hashes": self.response_hashes or [],
            "last_plan": self.last_plan,
        }


def visible_reference_anchor(question: str) -> str:
    for pattern in REFERENCE_PATTERNS:
        match = re.search(pattern, question, re.I)
        if match:
            return match.group(0)
    return ""


def visible_named_anchors(question: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    pattern = re.compile(r"\b[A-Z][A-Za-z0-9_-]+(?:\s+[A-Z][A-Za-z0-9_-]+){0,4}\b")
    for match in pattern.finditer(question):
        value = match.group(0).strip()
        parts = value.split()
        if all(part in GENERIC_NAMED_ANCHORS for part in parts):
            continue
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def preferred_target_surface(question: str) -> str:
    reference = visible_reference_anchor(question)
    if reference:
        return reference
    anchors = visible_named_anchors(question)
    return " ".join(anchors[:3])


def _role_object_anchor(question: str, role_words: list[str]) -> str:
    role_alt = "|".join(re.escape(word) for word in role_words)
    for pattern in [rf"\b(?:{role_alt})\s+(?:the\s+)?([^?]+?)(?:\?|$)", rf"\b(?:{role_alt})\s+(?:for|of)\s+(?:the\s+)?([^?]+?)(?:\?|$)"]:
        match = re.search(pattern, question, re.I)
        if match:
            value = re.sub(r"\b(?:after|before|during|for|by)\b.*$", "", match.group(1), flags=re.I)
            return value.strip(" .?,")
    return ""


def _text_has_target(text: str, terms: list[str]) -> bool:
    if not terms:
        return True
    low = normalize(text)
    hits = sum(1 for term in terms if term and term in low)
    if len(terms) >= 4:
        return hits >= 3
    if len(terms) >= 3:
        return hits >= 2
    return hits >= 1


def deterministic_plan(question: str) -> dict[str, Any]:
    q = normalize(question)
    qtokens = set(re.findall(r"[a-z0-9_-]+", q))
    target = preferred_target_surface(question)
    plan: dict[str, Any] = {
        "intent": "unknown",
        "target_surface": target,
        "answer_role": "unknown",
        "requires_asserted": True,
        "source": "deterministic",
    }
    wants_identifier_answer = re.search(r"\bids?\b|\bidentifiers?\b|\breferences?\b", q) is not None or (
        q.startswith(("what ", "which ")) and any(token in qtokens for token in {"raw", "json", "record"})
    )
    wants_metadata_answer = any(word in q for word in ["modified", "created", "mtime", "ctime", "size", "hash", "suffix", "extension", "encoding", "line count", "word count"])
    if wants_metadata_answer:
        plan.update({"intent": "context_lookup", "answer_role": "state", "target_surface": target or question.strip(" ?"), "requires_asserted": False})
    elif wants_identifier_answer and target:
        plan.update({"intent": "reference_lookup", "answer_role": "reference", "target_surface": target, "requires_asserted": True})
    elif any(phrase in q for phrase in ["who owns", "who owned", "who is owner", "which owner"]):
        plan.update({"intent": "role_lookup", "answer_role": "owner", "target_surface": target or _role_object_anchor(question, ["owns", "owned", "owner"]), "requires_asserted": True})
    elif any(phrase in q for phrase in ["who authored", "who wrote", "who created", "who assembled", "responsible for", "carried", "actor behind"]):
        plan.update({"intent": "role_lookup", "answer_role": "author", "target_surface": target or _role_object_anchor(question, ["authored", "wrote", "created", "assembled", "responsible for", "carried"]), "requires_asserted": True})
    elif any(phrase in q for phrase in ["who reviewed", "reviewer", "looked over", "checked", "inspected", "signed off"]):
        plan.update({"intent": "role_lookup", "answer_role": "reviewer", "target_surface": target or _role_object_anchor(question, ["reviewed", "looked over", "checked", "inspected", "signed off"]), "requires_asserted": True})
    elif "approved" in q:
        plan.update({"intent": "role_lookup", "answer_role": "approver", "target_surface": target, "requires_asserted": True})
    elif any(phrase in q for phrase in ["which organization", "what organization", "which company", "what company", "which group", "what group"]):
        plan.update({"intent": "role_lookup", "answer_role": "organization", "target_surface": target or question.strip(" ?"), "requires_asserted": True})
    elif any(phrase in q for phrase in ["who reported", "who requested", "who claimed", "who alleged", "which account"]):
        plan.update({"intent": "role_lookup", "answer_role": "reporter", "target_surface": target or question.strip(" ?"), "requires_asserted": True})
    elif qtokens.intersection({"url", "urls", "link", "links", "runbook", "manual", "guide"}):
        plan.update({"intent": "url_lookup", "answer_role": "reference", "target_surface": target or question.strip(" ?"), "requires_asserted": True})
    elif qtokens.intersection({"file", "files", "path", "paths"}):
        plan.update({"intent": "file_lookup", "answer_role": "reference", "target_surface": target or question.strip(" ?"), "requires_asserted": True})
    elif qtokens.intersection({"reference", "references", "identifier", "identifiers", "id", "ids", "case"}):
        plan.update({"intent": "reference_lookup", "answer_role": "reference", "target_surface": target or question.strip(" ?"), "requires_asserted": True})
    elif any(phrase in q for phrase in ["final state", "left in", "ended up", "at the end", "current state"]):
        plan.update({"intent": "state_lookup", "answer_role": "state", "target_surface": target or question.strip(" ?"), "requires_asserted": True})
    elif any(word in q for word in ["measurement", "measured", "modified", "archived", "valid", "validity", "effective"]):
        plan.update({"intent": "context_lookup", "answer_role": "state", "target_surface": question.strip(" ?"), "requires_asserted": False})
    elif any(word in q for word in ["asserted", "alleged", "reported", "quoted", "fictional", "dream"]):
        plan.update({"intent": "context_lookup", "answer_role": "unknown", "target_surface": target or question.strip(" ?"), "requires_asserted": False})
    elif (q.startswith("are ") and " same " in q) or ("same person" in q and "identify" in q):
        plan.update({"intent": "identity_lookup", "answer_role": "unknown", "target_surface": question.strip(" ?"), "requires_asserted": False})
    return plan


def normalize_model_plan(question: str, model: dict[str, Any] | None, det: dict[str, Any]) -> dict[str, Any] | None:
    if not model or not model.get("accepted"):
        return model
    plan = dict(model)
    plan["query_text"] = question
    q = normalize(question)
    qtokens = set(re.findall(r"[a-z0-9_-]+", q))
    target = preferred_target_surface(question)
    target_terms = [term for term in re.findall(r"[a-z0-9_-]+", normalize(str(plan.get("target_surface") or ""))) if len(term) > 1]
    target_is_generic_reference = bool(target_terms) and len(target_terms) <= 3 and any(term in {"id", "ids", "identifier", "identifiers", "reference", "references"} for term in target_terms)
    if target and (target_is_generic_reference or not _text_has_target(str(plan.get("target_surface", "")), [normalize(term) for term in target.split()])):
        plan["target_surface"] = target
    wants_identifier_answer = re.search(r"\bids?\b|\bidentifiers?\b|\breferences?\b", q) is not None
    wants_metadata_answer = any(word in q for word in ["modified", "created", "mtime", "ctime", "size", "hash", "suffix", "extension", "encoding", "line count", "word count"])
    if wants_metadata_answer:
        plan.update({"intent": "context_lookup", "answer_role": "state", "target_surface": target or question.strip(" ?"), "requires_asserted": False})
    elif wants_identifier_answer and target:
        plan.update({"intent": "reference_lookup", "answer_role": "reference", "target_surface": target, "requires_asserted": True})
    elif any(phrase in q for phrase in ["who owns", "who owned", "who is owner", "which owner"]):
        plan.update({"intent": "role_lookup", "answer_role": "owner"})
    elif any(phrase in q for phrase in ["who authored", "who wrote", "who created", "who assembled", "responsible for", "carried", "actor behind"]):
        plan.update({"intent": "role_lookup", "answer_role": "author"})
    elif any(phrase in q for phrase in ["who reviewed", "reviewer", "looked over", "checked", "inspected", "signed off"]):
        plan.update({"intent": "role_lookup", "answer_role": "reviewer"})
    elif "approved" in q:
        plan.update({"intent": "role_lookup", "answer_role": "approver"})
    elif any(phrase in q for phrase in ["which organization", "what organization", "which company", "what company", "which group", "what group"]):
        plan.update({"intent": "role_lookup", "answer_role": "organization"})
    elif any(phrase in q for phrase in ["who reported", "who requested", "who claimed", "who alleged", "which account"]):
        plan.update({"intent": "role_lookup", "answer_role": "reporter"})
    elif qtokens.intersection({"url", "urls", "link", "links", "runbook", "manual", "guide"}):
        plan.update({"intent": "url_lookup", "answer_role": "reference"})
    elif qtokens.intersection({"file", "files", "path", "paths"}):
        plan.update({"intent": "file_lookup", "answer_role": "reference"})
    elif qtokens.intersection({"reference", "references", "identifier", "identifiers", "id", "ids", "case"}):
        plan.update({"intent": "reference_lookup", "answer_role": "reference"})
    elif any(phrase in q for phrase in ["final state", "left in", "ended up", "at the end", "current state"]):
        plan.update({"intent": "state_lookup", "answer_role": "state", "requires_asserted": True})
    elif any(word in q for word in ["measurement", "measured", "modified", "archived", "valid", "validity", "effective", "asserted", "alleged", "reported", "quoted", "fictional", "dream"]):
        plan.update({"intent": "context_lookup", "requires_asserted": False})
    if not str(plan.get("target_surface") or "").strip() and det.get("target_surface"):
        plan["target_surface"] = det["target_surface"]
    if det.get("intent") == "identity_lookup" and plan.get("intent") != "identity_lookup":
        plan = dict(det)
        plan["source"] = "deterministic_identity_guard"
    if plan.get("intent") in {"role_lookup", "reference_lookup", "url_lookup", "file_lookup", "state_lookup"}:
        plan["requires_asserted"] = True
    return plan


def build_query_plan_prompt(question: str) -> str:
    return (
        "JSON only. Convert the question into a generic raw-text knowledge query plan; do not answer it. "
        "Use role_lookup for questions asking who performed a role or action. "
        "Use reference_lookup for IDs or named references, url_lookup for links, file_lookup for file paths, "
        "state_lookup for final/current state, context_lookup for assertion/report/quote/validity/time context, "
        "identity_lookup for same-entity questions, grouped_search for broad grouped retrieval, unknown otherwise. "
        "target_surface must be an exact visible anchor from the question when present."
        + json.dumps({"question": question}, ensure_ascii=False)
    )


def call_model_query_plan(question: str, client: LocalModelClient, *, n_predict: int = 96) -> dict[str, Any]:
    prompt = build_query_plan_prompt(question)
    start = time.time()
    try:
        parsed = client.complete_json(prompt, n_predict=n_predict, grammar=INTENT_GRAMMAR)
    except Exception as exc:
        return {
            "intent": "unknown",
            "target_surface": "",
            "answer_role": "unknown",
            "requires_asserted": True,
            "source": "model",
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "grammar_hash": hashlib.sha256(INTENT_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    plan = parsed.get("query_plan") if isinstance(parsed, dict) else None
    raw = str(parsed.get("_model_raw") or "") if isinstance(parsed, dict) else ""
    if not isinstance(plan, dict):
        return {
            "intent": "unknown",
            "target_surface": "",
            "answer_role": "unknown",
            "requires_asserted": True,
            "source": "model",
            "accepted": False,
            "reason": "invalid_json",
            "raw_text": raw,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "grammar_hash": hashlib.sha256(INTENT_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    return {
        **plan,
        "source": "model",
        "accepted": True,
        "raw_text": raw,
        "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
        "stop_reason": "parsed_json",
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
        "grammar_hash": hashlib.sha256(INTENT_GRAMMAR.encode()).hexdigest(),
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }


def build_evidence_extraction_prompt(question: str, expected_answer_type: str, evidence_items: list[dict[str, str]]) -> str:
    return (
        "JSON only. Answer the question only from the provided raw-text evidence. "
        "Return sufficient_evidence=false and answer='unknown' when the evidence does not state a complete answer. "
        "The answer must be the shortest exact grounded value compatible with the expected answer type. "
        "Do not use outside knowledge."
        + json.dumps(
            {
                "question": question,
                "expected_answer_type": expected_answer_type,
                "evidence": evidence_items,
            },
            ensure_ascii=False,
        )
    )


def call_model_evidence_answer(
    question: str,
    expected_answer_type: str,
    evidence_items: list[dict[str, str]],
    client: LocalModelClient,
    *,
    n_predict: int = 160,
) -> dict[str, Any]:
    prompt = build_evidence_extraction_prompt(question, expected_answer_type, evidence_items)
    start = time.time()
    try:
        parsed = client.complete_json(prompt, n_predict=n_predict, grammar=EVIDENCE_EXTRACTION_GRAMMAR)
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "grammar_hash": hashlib.sha256(EVIDENCE_EXTRACTION_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    answer = parsed.get("answer") if isinstance(parsed, dict) else None
    raw = str(parsed.get("_model_raw") or "") if isinstance(parsed, dict) else ""
    if not isinstance(answer, dict):
        return {
            "accepted": False,
            "reason": "invalid_json",
            "raw_text": raw,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "grammar_hash": hashlib.sha256(EVIDENCE_EXTRACTION_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    return {
        **answer,
        "source": "model_evidence_extraction",
        "accepted": True,
        "raw_text": raw,
        "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
        "stop_reason": "parsed_json",
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
        "grammar_hash": hashlib.sha256(EVIDENCE_EXTRACTION_GRAMMAR.encode()).hexdigest(),
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }
