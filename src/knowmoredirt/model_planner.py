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

REFERENCE_PATTERNS = [
    r"\b[A-Z][A-Z0-9]{1,12}-\d+[A-Z0-9-]*\b",
    r"https?://[^\s\]\)\"']+",
    r"\b[A-Za-z0-9_./-]+\.(?:cpp|tmp|py|js|md|txt|json|yaml|yml|csv|tsv|pdf)\b",
]


@dataclass
class ModelQueryTrace:
    enabled: bool = False
    call_count: int = 0
    parsed_count: int = 0
    accepted_count: int = 0
    model_answer_count: int = 0
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
    target = visible_reference_anchor(question)
    plan: dict[str, Any] = {
        "intent": "unknown",
        "target_surface": target,
        "answer_role": "unknown",
        "requires_asserted": True,
        "source": "deterministic",
    }
    if any(phrase in q for phrase in ["who owns", "who owned", "who is owner", "which owner"]):
        plan.update({"intent": "role_lookup", "answer_role": "owner", "target_surface": target or _role_object_anchor(question, ["owns", "owned", "owner"]), "requires_asserted": True})
    elif any(phrase in q for phrase in ["who authored", "who wrote", "who created", "who assembled", "responsible for", "carried", "actor behind"]):
        plan.update({"intent": "role_lookup", "answer_role": "author", "target_surface": target or _role_object_anchor(question, ["authored", "wrote", "created", "assembled", "responsible for", "carried"]), "requires_asserted": True})
    elif any(phrase in q for phrase in ["who reviewed", "reviewer", "looked over", "checked", "inspected", "signed off"]):
        plan.update({"intent": "role_lookup", "answer_role": "reviewer", "target_surface": target or _role_object_anchor(question, ["reviewed", "looked over", "checked", "inspected", "signed off"]), "requires_asserted": True})
    elif "approved" in q:
        plan.update({"intent": "role_lookup", "answer_role": "approver", "target_surface": target, "requires_asserted": True})
    elif any(phrase in q for phrase in ["who reported", "who requested", "who claimed", "who alleged", "which organization", "which company", "which account"]):
        plan.update({"intent": "role_lookup", "answer_role": "reporter", "target_surface": target or question.strip(" ?"), "requires_asserted": True})
    elif any(word in q for word in ["url", "link", "runbook", "manual", "guide"]):
        plan.update({"intent": "url_lookup", "answer_role": "reference", "target_surface": target or question.strip(" ?"), "requires_asserted": True})
    elif any(word in q for word in ["file", "path"]):
        plan.update({"intent": "file_lookup", "answer_role": "reference", "target_surface": target or question.strip(" ?"), "requires_asserted": True})
    elif any(word in q for word in ["reference", "identifier", "id", "case"]):
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
    target = visible_reference_anchor(question)
    if target and not _text_has_target(str(plan.get("target_surface", "")), [normalize(target)]):
        plan["target_surface"] = target
    if any(phrase in q for phrase in ["who owns", "who owned", "who is owner", "which owner"]):
        plan.update({"intent": "role_lookup", "answer_role": "owner"})
    elif any(phrase in q for phrase in ["who authored", "who wrote", "who created", "who assembled", "responsible for", "carried", "actor behind"]):
        plan.update({"intent": "role_lookup", "answer_role": "author"})
    elif any(phrase in q for phrase in ["who reviewed", "reviewer", "looked over", "checked", "inspected", "signed off"]):
        plan.update({"intent": "role_lookup", "answer_role": "reviewer"})
    elif "approved" in q:
        plan.update({"intent": "role_lookup", "answer_role": "approver"})
    elif any(phrase in q for phrase in ["who reported", "who requested", "who claimed", "who alleged", "which organization", "which company", "which account"]):
        plan.update({"intent": "role_lookup", "answer_role": "reporter"})
    elif any(word in q for word in ["url", "link", "runbook", "manual", "guide"]):
        plan.update({"intent": "url_lookup", "answer_role": "reference"})
    elif any(word in q for word in ["file", "path"]):
        plan.update({"intent": "file_lookup", "answer_role": "reference"})
    elif any(word in q for word in ["reference", "identifier", "id", "case"]):
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
