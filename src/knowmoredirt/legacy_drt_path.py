"""Migrated old DRT model-query planning primitives.

This module is a cleaned port of the high-performing DRT_tests
``scripts/dspg_query.py`` model-query path. It contains only generic query
planning, prompt construction, normalization, and trace helpers. It deliberately
contains no benchmark-specific names, prepared-corpus markers, gold labels, or
adapter logic.
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
intent ::= "\"who_author\"" | "\"who_opened\"" | "\"who_review\"" | "\"who_approved\"" | "\"who_commented\"" | "\"who_merged\"" | "\"who_assigned\"" | "\"who_owns\"" | "\"who_reported\"" | "\"which_customer\"" | "\"which_pr\"" | "\"which_ticket\"" | "\"which_issue\"" | "\"which_url\"" | "\"which_file\"" | "\"final_state\"" | "\"scope_status\"" | "\"identity_status\"" | "\"context_time\"" | "\"broad_search_grouped\"" | "\"unknown\""
role ::= "\"agent\"" | "\"author\"" | "\"reviewer\"" | "\"approver\"" | "\"customer\"" | "\"theme\"" | "\"artifact\"" | "\"state\"" | "\"unknown\""
bool ::= "true" | "false"
string ::= "\"" chars "\""
chars ::= ([^"\\] | "\\" ["\\/bfnrt])*
ws ::= [ \t\n\r]*
'''

ARTIFACT_PATTERNS = [
    r"\bPR-\d+\b",
    r"\b(?:BUG|ISSUE)-\d+\b",
    r"\b(?:SUP|TICKET)-\d+\b",
    r"https?://[^\s\]\)\"']+",
    r"\b[A-Za-z0-9_./-]+\.(?:cpp|tmp|py|js|md|txt|json|yaml|yml)\b",
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


def visible_artifact_anchor(question: str) -> str:
    for pattern in ARTIFACT_PATTERNS:
        match = re.search(pattern, question, re.I)
        if match:
            return match.group(0)
    return ""


def deterministic_plan(question: str) -> dict[str, Any]:
    q = normalize(question)
    target = visible_artifact_anchor(question)
    if ("authored" in q or "opened" in q or "which engineer" in q or "carried" in q) and target:
        return {"intent": "who_author", "target_surface": target, "answer_role": "author", "requires_asserted": True, "source": "deterministic"}
    if any(phrase in q for phrase in ["who owns", "who owned", "who is owner", "which owner"]) and target:
        return {"intent": "who_owns", "target_surface": target, "answer_role": "owner", "requires_asserted": True, "source": "deterministic"}
    if any(phrase in q for phrase in ["responsible for", "carried the fix", "engineer behind", "authored the fix"]) and target:
        return {"intent": "who_author", "target_surface": target, "answer_role": "author", "requires_asserted": True, "source": "deterministic"}
    if ("reviewed" in q or "reviewer" in q) and target:
        return {"intent": "who_review", "target_surface": target, "answer_role": "reviewer", "requires_asserted": True, "source": "deterministic"}
    if any(phrase in q for phrase in ["looked over", "checked", "inspected", "signed off"]) and target:
        return {"intent": "who_review", "target_surface": target, "answer_role": "reviewer", "requires_asserted": True, "source": "deterministic"}
    if "approved" in q and target:
        return {"intent": "who_approved", "target_surface": target, "answer_role": "approver", "requires_asserted": True, "source": "deterministic"}
    if ("which customer" in q or "which company" in q or "which account" in q) and target:
        return {"intent": "which_customer", "target_surface": target, "answer_role": "customer", "requires_asserted": True, "source": "deterministic"}
    if "which pr" in q and target:
        return {"intent": "which_pr", "target_surface": target, "answer_role": "artifact", "requires_asserted": True, "source": "deterministic"}
    if "which ticket" in q and target:
        return {"intent": "which_ticket", "target_surface": target, "answer_role": "artifact", "requires_asserted": True, "source": "deterministic"}
    if "which issue" in q and target:
        return {"intent": "which_issue", "target_surface": target, "answer_role": "artifact", "requires_asserted": True, "source": "deterministic"}
    if "which url" in q:
        return {"intent": "which_url", "target_surface": question.strip(" ?"), "answer_role": "artifact", "requires_asserted": True, "source": "deterministic"}
    if q.startswith("where ") and any(word in q for word in ["catalog", "located", "link", "url", "runbook", "manual", "guide", "document", "drawing"]):
        return {"intent": "which_url", "target_surface": question.strip(" ?"), "answer_role": "artifact", "requires_asserted": True, "source": "deterministic"}
    if "which file" in q:
        return {"intent": "which_file", "target_surface": target or question.strip(" ?"), "answer_role": "artifact", "requires_asserted": True, "source": "deterministic"}
    if ("final state" in q or "left" in q or "ended up" in q or "at the end" in q) and target:
        return {"intent": "final_state", "target_surface": target, "answer_role": "state", "requires_asserted": True, "source": "deterministic"}
    if any(word in q for word in ["measurement year", "measurement date", "measured", "source modified", "file modified", "modified time", "archived as of", "valid until", "validity"]):
        return {"intent": "context_time", "target_surface": question.strip(" ?"), "answer_role": "state", "requires_asserted": False, "source": "deterministic"}
    if ("asserted" in q or "alleged" in q or "reported" in q or "assertion status" in q) and target:
        return {"intent": "scope_status", "target_surface": target, "answer_role": "unknown", "requires_asserted": False, "source": "deterministic"}
    if (q.startswith("are ") and " same " in q) or ("same person" in q and "identify" in q):
        return {"intent": "identity_status", "target_surface": question.strip(" ?"), "answer_role": "unknown", "requires_asserted": False, "source": "deterministic"}
    return {"intent": "unknown", "target_surface": target, "answer_role": "unknown", "requires_asserted": True, "source": "deterministic"}


def _customer_anchor(question: str) -> str:
    match = re.search(r"\b(?:customer|account|company)\s+([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,5})", question)
    return match.group(1).strip(" .?,") if match else ""


def _role_object_anchor(question: str, role_words: list[str]) -> str:
    role_alt = "|".join(re.escape(word) for word in role_words)
    for pattern in [rf"\b(?:{role_alt})\s+(?:the\s+)?([^?]+?)(?:\?|$)", rf"\b(?:{role_alt})\s+(?:for|of)\s+(?:the\s+)?([^?]+?)(?:\?|$)"]:
        match = re.search(pattern, question, re.I)
        if match:
            value = re.sub(r"\b(?:after|before|during|for customer|by customer)\b.*$", "", match.group(1), flags=re.I)
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


def normalize_model_plan(question: str, model: dict[str, Any] | None, det: dict[str, Any]) -> dict[str, Any] | None:
    if not model or not model.get("accepted"):
        return model
    plan = dict(model)
    plan["query_text"] = question
    q = normalize(question)
    target = visible_artifact_anchor(question)
    if target and not _text_has_target(str(plan.get("target_surface", "")), [normalize(target)]):
        plan["target_surface"] = target
    customer = _customer_anchor(question)
    if any(word in q for word in ["which account", "which customer", "which company", "what account", "what customer", "what company"]):
        plan["intent"] = "which_customer"
        plan["answer_role"] = "customer"
        if target:
            plan["target_surface"] = target
    if customer and any(phrase in q for phrase in ["for customer", "for account", "customer ", "account ", "company "]):
        generic_target = normalize(str(plan.get("target_surface") or ""))
        if not generic_target or generic_target in {"engineer", "customer", "account", "company", "person", "fix"}:
            plan["target_surface"] = customer
    if plan.get("intent") in {"who_author", "who_opened", "who_reported", "unknown"} and (plan.get("answer_role") == "customer" or any(word in q for word in ["which account", "which customer", "which company", "account sounded", "company flagged", "account raised", "customer escalated"])):
        plan["intent"] = "which_customer"
        plan["answer_role"] = "customer"
    if any(phrase in q for phrase in ["who authored", "who wrote", "who created", "who prepared", "which engineer", "engineer behind", "carried the fix", "responsible for", "authored the fix"]):
        plan["intent"] = "who_author"
        plan["answer_role"] = "author"
        if not target and not customer:
            obj = _role_object_anchor(question, ["authored", "wrote", "created", "prepared", "carried", "responsible for"])
            if obj:
                plan["target_surface"] = obj
    if any(phrase in q for phrase in ["who reviewed", "who looked over", "looked over", "checked", "inspected", "signed off"]):
        plan["intent"] = "who_review"
        plan["answer_role"] = "reviewer"
        if not target:
            obj = _role_object_anchor(question, ["reviewed", "looked over", "checked", "inspected", "signed off"])
            if obj:
                plan["target_surface"] = obj
    if any(phrase in q for phrase in ["who owns", "who owned", "who is owner", "which owner"]):
        plan["intent"] = "who_owns"
        plan["answer_role"] = "owner"
        if target:
            plan["target_surface"] = target
    if plan.get("intent") == "scope_status" and q.startswith("where") and any(word in q for word in ["cataloged", "link", "url", "drawing", "runbook", "manual", "document"]):
        plan["intent"] = "which_url"
        plan["answer_role"] = "artifact"
    if plan.get("intent") in {"unknown", "scope_status", "broad_search_grouped"} and any(word in q for word in ["guide", "runbook", "manual", "document", "drawing"]) and any(phrase in q for phrase in ["point me", "show me", "locate", "where"]):
        plan["intent"] = "which_url"
        plan["answer_role"] = "artifact"
        if target:
            plan["target_surface"] = target
    if plan.get("intent") in {"unknown", "scope_status"} and any(word in q for word in ["measurement year", "measurement date", "measured", "source modified", "file modified", "valid until", "archived as of"]):
        plan["intent"] = "context_time"
        plan["answer_role"] = "state"
    if plan.get("intent") == "context_time" and not str(plan.get("target_surface") or "").strip():
        plan["target_surface"] = question.strip(" ?")
    if any(phrase in q for phrase in ["final state", "left in", "ended up", "at the end"]):
        plan["intent"] = "final_state"
        plan["answer_role"] = "state"
        if target:
            plan["target_surface"] = target
        plan["requires_asserted"] = True
    if plan.get("intent") in {"who_author", "who_opened", "who_review", "who_approved", "who_commented", "who_merged", "who_assigned", "who_owns", "which_customer", "which_pr", "which_ticket", "which_issue", "which_url", "which_file", "final_state"}:
        plan["requires_asserted"] = True
    if det.get("intent") == "identity_status" and plan.get("intent") != "identity_status":
        plan = dict(det)
        plan["source"] = "deterministic_identity_guard"
    return plan


def build_query_plan_prompt(question: str) -> str:
    return (
        "JSON only. Convert the question into a graph query plan, do not answer it. "
        "Use the intent enum. Map implicit responsibility for a change/fix/patch to who_author or who_opened. "
        "Map responsible owner, engineer behind, carried, landed, or authored the fix to who_author unless the question asks ownership directly. "
        "Map reviewed/looked over/checked/inspected/signed-off review to who_review. "
        "Map account/customer/company raised/flagged/escalated/reported/affected/suffered/experienced/requested refund to customer/report intents. "
        "Map runbook/design/manual/document/drawing link/cataloged/location questions to which_url. "
        "Map final state/left/ended up/after timeline/at the end to final_state. "
        "Map asserted/reported/quoted/alleged/fictional status questions to scope_status. "
        "Map measurement/source-modified/validity/archive time questions to context_time. "
        "target_surface must be an exact visible anchor from the question when present.\n"
        + json.dumps({"question": question}, ensure_ascii=False)
    )


def call_model_query_plan(question: str, client: LocalModelClient, *, n_predict: int = 128) -> dict[str, Any]:
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
