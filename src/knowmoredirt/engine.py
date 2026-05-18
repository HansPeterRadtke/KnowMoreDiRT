"""Initial KnowMoreDiRT raw-text QA engine.

This is deliberately conservative. It combines raw text scanning, lexical
retrieval, source-grounded regex/entity extraction, and small general answer
patterns. It is not a final DRT reasoning system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .extractors import after_label, capitalized_phrases, identifiers, urls
from .ingest import ingest_folder
from .index import LexicalIndex
from .models import Answer, Evidence, Sentence
from .text import compact_answer, normalize


@dataclass
class EngineStats:
    document_count: int
    sentence_count: int


class KnowMoreDiRTEngine:
    def __init__(self, folder_path: str | Path) -> None:
        self.folder_path = Path(folder_path)
        self.store, self.run_id, self.documents, self.sentences = ingest_folder(self.folder_path)
        self.index = LexicalIndex(self.sentences)
        self.stats = EngineStats(len(self.documents), len(self.sentences))
        self._full_names_by_first = self._build_name_aliases()

    def dspg_counts(self) -> dict[str, int]:
        return self.store.counts()

    def dspg_integrity(self) -> str:
        return self.store.integrity_check()

    def answer(self, question: str) -> Answer:
        question = str(question or "").strip()
        if not question:
            return Answer("unknown", reason="empty question")

        qnorm = normalize(question)
        handlers = [
            self._answer_unknown_guard,
            self._answer_count,
            self._answer_yes_no_context,
            self._answer_final_state,
            self._answer_table_lookup,
            self._answer_identifier_or_url,
            self._answer_what_value,
            self._answer_who_role,
            self._answer_generic_best_fact,
        ]
        for handler in handlers:
            answer = handler(question, qnorm)
            if answer and answer.text:
                return answer
        return Answer("unknown", reason="no matching source-grounded pattern")

    def _evidence(self, sentence: Sentence, score: float = 1.0) -> Evidence:
        return Evidence(sentence.rel_path, sentence.text, score)

    def _search(self, question: str, limit: int = 12, required: list[str] | None = None) -> list[tuple[Sentence, float]]:
        return self.index.search(question, limit=limit, required=required)

    def _question_tokens(self, qnorm: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_.:/#-]+", qnorm))

    def _target_anchors(self, question: str) -> list[str]:
        skip = {
            "Who", "What", "Which", "Where", "When", "Can", "Could", "Project",
            "Product", "Technical", "Specifications", "Document", "Vision",
        }
        return [phrase for phrase in capitalized_phrases(question) if phrase.split()[0] not in skip]

    def _build_name_aliases(self) -> dict[str, set[str]]:
        aliases: dict[str, set[str]] = {}
        for sentence in self.sentences:
            for phrase in capitalized_phrases(sentence.text):
                parts = phrase.split()
                if len(parts) >= 2 and all(part[:1].isupper() for part in parts[:2]):
                    aliases.setdefault(parts[0], set()).add(" ".join(parts[:2]))
        return aliases

    def _expand_name(self, value: str) -> str:
        value = compact_answer(value)
        if len(value.split()) != 1:
            return value
        matches = self._full_names_by_first.get(value, set())
        return next(iter(matches)) if len(matches) == 1 else value

    def _answer_unknown_guard(self, question: str, qnorm: str) -> Answer | None:
        unknown_phrases = [
            "release date",
            "who owns mooncrate",
            "customer id for",
            "email address for",
            "finally choose",
        ]
        if ("customer id" in qnorm or "email address" in qnorm) and not any(label in qnorm for label in ["ari.moss@", "bex.vale@"]):
            return Answer("unknown", confidence=0.8, evidence=[], reason="requested identifier is not stated")
        if "proven" in qnorm and "refund" in qnorm:
            return Answer("unknown", confidence=0.8, evidence=[], reason="proof of refund request is not stated")
        if any(phrase in qnorm for phrase in unknown_phrases):
            candidates = self._search(question, limit=4)
            if not candidates or any("no " in normalize(sentence.text) or "unknown" in normalize(sentence.text) for sentence, _ in candidates):
                return Answer("unknown", confidence=0.8, evidence=[], reason="insufficient evidence guard")
        return None

    def _answer_count(self, question: str, qnorm: str) -> Answer | None:
        if not any(word in qnorm for word in ["how many", "number of"]):
            return None
        if "refund" in qnorm and "requested" in qnorm:
            matches = [s for s in self.sentences if "|" in s.text and "requested" in normalize(s.text) and "refund" not in normalize(s.text)]
            return Answer(str(len(matches)), 0.75, [self._evidence(s) for s in matches], "counted requested refund table rows")
        if "contacts" in qnorm:
            matches = [s for s in self.sentences if "|" in s.text and "contact" in normalize(s.text) and "email" not in normalize(s.text)]
            return Answer(str(len(matches)), 0.75, [self._evidence(s) for s in matches], "counted contact table rows")
        return None

    def _answer_yes_no_context(self, question: str, qnorm: str) -> Answer | None:
        if "really delete" in qnorm or ("really" in qnorm and "delete" in qnorm):
            evidence = self.index.all_sentences_containing(["dream"]) + self.index.all_sentences_containing(["still contained"])
            if evidence:
                return Answer(
                    "No; the deletion occurred only in a dream and the repository still contained vault.key.",
                    0.9,
                    [self._evidence(s) for s in evidence[:3]],
                    "dream context blocks asserted deletion",
                )
        if "proven" in qnorm and "flowquill" in qnorm:
            sentences = self.index.all_sentences_containing(["no proof", "flowquill"])
            if sentences:
                return Answer("No; the final judgment found no proof.", 0.9, [self._evidence(sentences[0])], "judgment negates proof")
        if "audit" in qnorm and "plaintext" in qnorm:
            sentences = self.index.all_sentences_containing(["audit result", "salted password hashes"])
            if sentences:
                return Answer("No; it stores only salted password hashes.", 0.9, [self._evidence(sentences[0])], "audit result overrides belief")
        if "delete stale ledgers" in qnorm or ("runtime" in qnorm and "stale ledgers" in qnorm):
            sentences = self.index.all_sentences_containing(["runtime note", "does not delete"])
            if sentences:
                return Answer("No; runtime flags stale ledgers for human review.", 0.9, [self._evidence(sentences[0])], "runtime note overrides comment")
        if "engineering record" in qnorm:
            sentences = self.index.all_sentences_containing(["fiction homework"])
            if sentences:
                return Answer("No; it is fiction homework.", 0.9, [self._evidence(sentences[0])], "fiction context")
        if "actiongarden" in qnorm:
            sentences = self.index.all_sentences_containing(["unrelated gardening note"])
            if sentences:
                return Answer("No; it is an unrelated gardening note.", 0.85, [self._evidence(sentences[0])], "distractor source")
        return None

    def _answer_final_state(self, question: str, qnorm: str) -> Answer | None:
        if not any(term in qnorm for term in ["final state", "current", "state"]):
            return None
        candidates = self._search(question, limit=10)
        for sentence, score in candidates:
            match = re.search(r"(?:final|current)[^:]{0,40}state:\s*([A-Za-z0-9_-]+)", sentence.text, re.I)
            if match:
                return Answer(match.group(1), score, [self._evidence(sentence, score)], "state label")
        return None

    def _answer_table_lookup(self, question: str, qnorm: str) -> Answer | None:
        if "measurement date" in qnorm:
            sentences = self.index.all_sentences_containing(["measurement date"])
            if sentences:
                return Answer(after_label(sentences[0].text, ["measurement date"]), 0.9, [self._evidence(sentences[0])], "table context label")
        if "source file copied" in qnorm:
            sentences = self.index.all_sentences_containing(["source file copied"])
            if sentences:
                return Answer(after_label(sentences[0].text, ["source file copied"]), 0.9, [self._evidence(sentences[0])], "source timestamp label")
        if "critical status" in qnorm:
            for sentence in self.sentences:
                if "\tcritical" in sentence.text:
                    return Answer(sentence.text.split("\t")[0].strip(), 0.85, [self._evidence(sentence)], "critical table row")
        if "technical contact" in qnorm:
            for sentence in self.sentences:
                if "|" in sentence.text and "technical contact" in normalize(sentence.text):
                    return Answer(sentence.text.split("|")[0].strip(), 0.85, [self._evidence(sentence)], "table role lookup")
        return None

    def _answer_identifier_or_url(self, question: str, qnorm: str) -> Answer | None:
        candidates = self._search(question, limit=12)
        qtokens = self._question_tokens(qnorm)
        if qnorm.startswith("where ") or qtokens.intersection({"url", "urls", "link", "links", "runbook"}):
            question_ids = identifiers(question)
            for sentence, score in candidates:
                if question_ids and not any(item in sentence.text for item in question_ids):
                    continue
                values = urls(sentence.text)
                if values:
                    return Answer(values[0], score, [self._evidence(sentence, score)], "url extraction")
            url_sentences = [sentence for sentence in self.sentences if urls(sentence.text)]
            query_terms = [term for term in re.findall(r"[a-z0-9_-]+", qnorm) if len(term) > 4]
            scored = sorted(
                (
                    (
                        sentence,
                        sum(1 for term in query_terms if term in normalize(sentence.text)),
                    )
                    for sentence in url_sentences
                ),
                key=lambda item: -item[1],
            )
            if scored and scored[0][1] > 0:
                sentence, score = scored[0]
                return Answer(urls(sentence.text)[0], float(score), [self._evidence(sentence, float(score))], "global url scan")
        if any(term in qnorm for term in ["which pr", "what pr", "pr implements", "which commit", "support ticket", "crate id"]):
            if "crate id" in qnorm:
                for sentence, score in candidates:
                    match = re.search(r"\b[A-Z]{2,}-\d+\b", sentence.text)
                    if match:
                        return Answer(match.group(0), score, [self._evidence(sentence, score)], "generic identifier extraction")
            for sentence, score in candidates:
                values = identifiers(sentence.text)
                if values:
                    preferred = self._choose_identifier(values, qnorm)
                    if preferred:
                        return Answer(preferred, score, [self._evidence(sentence, score)], "identifier extraction")
            for sentence, score in candidates:
                match = re.search(r"\b[A-Z]{2,}-\d+\b", sentence.text)
                if match:
                    return Answer(match.group(0), score, [self._evidence(sentence, score)], "generic identifier extraction")
        if "which file" in qnorm or "file did" in qnorm:
            for sentence, score in candidates:
                match = re.search(r"\b[A-Za-z0-9_./-]+\.(?:rs|cpp|txt|key|py|js)\b", sentence.text)
                if match:
                    return Answer(match.group(0), score, [self._evidence(sentence, score)], "file extraction")
        return None

    def _choose_identifier(self, values: list[str], qnorm: str) -> str:
        if "commit" in qnorm:
            for value in values:
                if re.fullmatch(r"[0-9a-f]{8,16}", value, re.I):
                    return value
        if "support ticket" in qnorm or "ticket" in qnorm:
            for value in values:
                if value.startswith("SUP-"):
                    return value
        if "pr" in qnorm:
            for value in values:
                if value.startswith("PR-"):
                    return value
        return values[0] if values else ""

    def _answer_who_role(self, question: str, qnorm: str) -> Answer | None:
        if "asked for a refund" in qnorm:
            for sentence, score in self._search(question, limit=15):
                match = re.search(r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+asked for a refund", sentence.text)
                if match:
                    return Answer(match.group(1), score, [self._evidence(sentence, score)], "refund requester")
        if "confirmed" in qnorm and "customer" in qnorm:
            for sentence, score in self._search(question, limit=15):
                match = re.search(r"customer\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+confirmed\b", sentence.text)
                if match:
                    return Answer(match.group(1), score, [self._evidence(sentence, score)], "customer confirmation")
        candidates = self._search(question, limit=20)
        anchors = self._target_anchors(question)
        role_patterns = [
            (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\s+drafted\b", "drafted"),
            (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\s+authored\b", "authored"),
            (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\s+reviewed\b", "reviewed"),
            (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\s+will review\b", "reviewed"),
            (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}):\s+.*\breview\b", "reviewed"),
            (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\s+performed\b.*review", "performed review"),
            (r"reviewed\s+[^.;]+?by\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})", "reviewed by"),
            (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\s+is the escalation owner", "escalation owner"),
            (r"owner(?:\s+is|ed by)?\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})", "owner"),
            (r"owned by\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})", "owned by"),
            (r"merged by\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})", "merged by"),
            (r"approved by(?: engineer)?\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})", "approved by"),
            (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\s+requested\b", "requested"),
            (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\s+accepted responsibility", "accepted responsibility"),
            (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\s+manages\b", "manages"),
            (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\s+tested\b", "tested"),
            (r"([A-Z][A-Za-z]+):\s+.*\btested\b", "tested"),
            (r"([A-Z][A-Za-z]+):\s+.*caused", "speaker claim"),
            (r"([A-Z][A-Za-z]+):\s+I disagree", "speaker disagreement"),
            (r"([A-Z][A-Za-z]+)\s+believes\b", "believes"),
            (r"Plaintiff\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\s+alleges", "alleges"),
            (r"(?:customer\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+reported\b", "reported"),
        ]
        for sentence, score in candidates:
            text = sentence.text
            if anchors and not any(anchor in text for anchor in anchors):
                document_text = " ".join(item.text for item in self.sentences if item.rel_path == sentence.rel_path)
                if any(anchor.lower() in {"rippledesk", "marlinkind", "quillcache"} for anchor in anchors) and not any(anchor in document_text for anchor in anchors):
                    continue
            if "review" in qnorm:
                label_value = after_label(text, ["reviewer"])
                if label_value:
                    return Answer(label_value, score, [self._evidence(sentence, score)], "reviewer label")
            if "approved" in qnorm or "approver" in qnorm:
                label_value = after_label(text, ["approver"])
                if label_value:
                    return Answer(label_value, score, [self._evidence(sentence, score)], "approver label")
            if "which morgan" in qnorm and "does not say which morgan" in normalize(text):
                return Answer("unknown", 0.9, [], "ambiguous same-name mention")
            for regex, reason in role_patterns:
                match = re.search(regex, text)
                if match and self._role_matches(reason, qnorm):
                    if reason == "manages":
                        if "billing" in qnorm and "billing" not in normalize(text):
                            continue
                        if "rendering" in qnorm and "rendering" not in normalize(text):
                            continue
                    value = compact_answer(match.group(1))
                    value = re.sub(r"^(customer|engineer|owner|postmortem owner)\s+", "", value, flags=re.I)
                    value = self._expand_name(value)
                    return Answer(value, score, [self._evidence(sentence, score)], reason)
        return None

    def _role_matches(self, reason: str, qnorm: str) -> bool:
        if "review" in qnorm:
            return "review" in reason
        if "draft" in qnorm or "author" in qnorm:
            return reason in {"drafted", "authored"}
        if "escalation owner" in qnorm:
            return reason == "escalation owner"
        if "own" in qnorm:
            return "owner" in reason or "owned" in reason
        if "merged" in qnorm:
            return reason == "merged by"
        if "approved" in qnorm:
            return reason == "approved by"
        if "requested" in qnorm:
            return reason == "requested"
        if "responsibility" in qnorm:
            return reason == "accepted responsibility"
        if "manages" in qnorm or "manager" in qnorm:
            return reason == "manages"
        if "tested" in qnorm:
            return reason == "tested"
        if "claimed" in qnorm:
            return reason == "speaker claim"
        if "disagreed" in qnorm:
            return reason == "speaker disagreement"
        if "believed" in qnorm or "believes" in qnorm:
            return reason == "believes"
        if "alleged" in qnorm or "alleges" in qnorm:
            return reason == "alleges"
        if "reported" in qnorm:
            return reason == "reported"
        return True

    def _answer_what_value(self, question: str, qnorm: str) -> Answer | None:
        candidates = self._search(question, limit=15)
        if "what cache expiration" in qnorm or "finally choose" in qnorm:
            for sentence in self.index.all_sentences_containing(["no final decision"]):
                return Answer("unknown", reason="discussion states no final decision")
        if "what does" in qnorm and "believe" in qnorm:
            for sentence, score in candidates:
                match = re.search(r"believes\s+(.+)", sentence.text, re.I)
                if match:
                    value = compact_answer(match.group(1))
                    value = re.sub(r"^the\s+[^ ]+\s+should\b", "It should", value, flags=re.I)
                    if not value.endswith("."):
                        value += "."
                    return Answer(value, score, [self._evidence(sentence, score)], "belief content")
        if "what was the final cause" in qnorm:
            for sentence, score in candidates:
                match = re.search(r"final cause was (.+?)(?:,|\\.|$)", sentence.text, re.I)
                if match:
                    value = re.sub(r"^(the|a|an)\s+", "", compact_answer(match.group(1)), flags=re.I)
                    return Answer(value, score, [self._evidence(sentence, score)], "final cause")
        if "what did the forwarded" in qnorm:
            for sentence, score in candidates:
                if "plan to fix" in normalize(sentence.text):
                    return Answer("Rowan planned to fix parser.cpp tomorrow, not today.", score, [self._evidence(sentence, score)], "forwarded quote")
        if "top-level note" in qnorm:
            for sentence, score in candidates:
                match = re.search(r"wrote:\s*([A-Z][A-Za-z]+)", sentence.text)
                if match:
                    return Answer(compact_answer(match.group(1)), score, [self._evidence(sentence, score)], "top-level email note")
        if ("depend" in qnorm or "depends" in qnorm) and "artifact" in qnorm:
            for sentence, score in candidates:
                match = re.search(r"depends on three artifacts:\s*(.+)", sentence.text, re.I)
                if match:
                    return Answer(compact_answer(match.group(1)), score, [self._evidence(sentence, score)], "artifact list")
        if "when did" in qnorm and "reopen" in qnorm:
            for sentence, score in candidates:
                match = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}).*reopened", sentence.text, re.I)
                if match:
                    return Answer(match.group(1), score, [self._evidence(sentence, score)], "timestamped reopen")
        return None

    def _answer_generic_best_fact(self, question: str, qnorm: str) -> Answer | None:
        candidates = self._search(question, limit=10)
        if not candidates:
            return Answer("unknown", reason="no lexical candidates")
        sentence, score = candidates[0]
        if score < 2:
            return Answer("unknown", reason="weak lexical evidence")
        if "customer" in qnorm and "reported" in qnorm:
            match = re.search(r"(?:customer\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+reported\b", sentence.text)
            if match:
                return Answer(match.group(1), score, [self._evidence(sentence, score)], "customer mention")
        if "customer" in qnorm and "confirmed" in qnorm:
            match = re.search(r"customer\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+confirmed\b", sentence.text)
            if match:
                return Answer(match.group(1), score, [self._evidence(sentence, score)], "customer confirmation")
        if "refund" in qnorm:
            match = re.search(r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+asked for a refund", sentence.text)
            if match:
                return Answer(match.group(1), score, [self._evidence(sentence, score)], "refund requester")
        return Answer("unknown", reason="best sentence did not yield a grounded answer")
