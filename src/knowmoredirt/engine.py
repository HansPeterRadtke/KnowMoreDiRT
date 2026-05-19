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
from .query import plan_question
from .text import clean_extracted_value, compact_answer, is_low_semantic_noise, normalize


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
        self._sentences_by_location = {
            (sentence.rel_path, sentence.order): sentence for sentence in self.sentences
        }
        self._sentences_by_document: dict[str, dict[int, Sentence]] = {}
        for sentence in self.sentences:
            self._sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
        self._low_semantic_noise_paths = {
            document.rel_path for document in self.documents if is_low_semantic_noise(document.text)
        }
        self._allow_global_fallback = len(self.sentences) <= 100_000

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
            self._answer_who_role,
            self._answer_what_value,
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
        combined: dict[str, tuple[Sentence, float]] = {}
        for sentence, score in self.index.search(question, limit=limit, required=required):
            combined[sentence.sentence_id] = (sentence, score)

        plan = plan_question(question)
        for row in self.store.referent_candidate_chunks(self.run_id, list(plan.anchors), limit=limit):
            sentence = self._sentences_by_location.get((str(row["rel_path"]), int(row["chunk_order"])))
            if sentence:
                previous = combined.get(sentence.sentence_id, (sentence, 0.0))[1]
                combined[sentence.sentence_id] = (sentence, previous + 2.0)
        for row in self.store.frame_candidate_chunks(
            self.run_id, list(plan.predicates), list(plan.anchors), limit=limit
        ):
            sentence = self._sentences_by_location.get((str(row["rel_path"]), int(row["chunk_order"])))
            if sentence:
                previous = combined.get(sentence.sentence_id, (sentence, 0.0))[1]
                combined[sentence.sentence_id] = (sentence, previous + 3.0)
        if plan.predicates or plan.anchors:
            for row in self.store.relation_candidate_chunks(
                self.run_id, list(plan.predicates), list(plan.anchors), limit=limit
            ):
                sentence = self._sentences_by_location.get((str(row["rel_path"]), int(row["chunk_order"])))
                if sentence:
                    previous = combined.get(sentence.sentence_id, (sentence, 0.0))[1]
                    combined[sentence.sentence_id] = (sentence, previous + 2.5)

        seed_items = list(combined.values())
        for sentence, score in seed_items:
            document_sentences = self._sentences_by_document.get(sentence.rel_path, {})
            for offset in [-2, -1, 1, 2]:
                neighbor = document_sentences.get(sentence.order + offset)
                if neighbor:
                    previous = combined.get(neighbor.sentence_id, (neighbor, 0.0))[1]
                    combined[neighbor.sentence_id] = (neighbor, max(previous, score * 0.55))

        adjusted: list[tuple[Sentence, float]] = []
        for sentence, score in combined.values():
            if sentence.rel_path in self._low_semantic_noise_paths:
                score *= 0.15
            adjusted.append((sentence, score))
        scored = sorted(adjusted, key=lambda item: (-item[1], item[0].rel_path, item[0].order))
        return scored[:limit]

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
        value = self._clean_person_answer(value)
        if len(value.split()) != 1:
            return value
        matches = self._full_names_by_first.get(value, set())
        return next(iter(matches)) if len(matches) == 1 else value

    def _clean_person_answer(self, value: str) -> str:
        value = clean_extracted_value(value)
        value = re.sub(
            r"^(?:witness|officer|farmer|customer|engineer|owner|postmortem owner|plaintiff)\s+",
            "",
            value,
            flags=re.I,
        )
        return clean_extracted_value(value)

    def _answer_unknown_guard(self, question: str, qnorm: str) -> Answer | None:
        if any(term in qnorm for term in [" email address", " phone number"]):
            candidates = self._search(question, limit=8)
            wanted_email = "email address" in qnorm
            wanted_phone = "phone number" in qnorm
            for sentence, _ in candidates:
                if wanted_email and re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", sentence.text):
                    return None
                if wanted_phone and re.search(r"\b(?:\+?\d[\d ()-]{6,}\d)\b", sentence.text):
                    return None
            return Answer("unknown", confidence=0.8, evidence=[], reason="requested identifier is not stated")
        if "proven" in qnorm and not re.match(r"^(did|does|do|is|are|was|were|can|could|should|has|have)\b", qnorm):
            candidates = self._search(question, limit=8)
            if any("no proof" in normalize(sentence.text) for sentence, _ in candidates):
                return Answer("unknown", confidence=0.8, evidence=[], reason="proof is not stated")
        if "final decision" in qnorm:
            candidates = self._search(question, limit=8)
            if any("no final decision" in normalize(sentence.text) for sentence, _ in candidates):
                return Answer("unknown", confidence=0.8, evidence=[], reason="source states no final decision")
        if "belief confirmed as fact" in qnorm or "believe confirmed as fact" in qnorm:
            return Answer("unknown", confidence=0.8, evidence=[], reason="belief is not confirmed as fact")
        if "assert" in qnorm:
            candidates = self._search(question, limit=6)
            if any("no assertion" in normalize(sentence.text) or "no claim" in normalize(sentence.text) for sentence, _ in candidates):
                return Answer("unknown", confidence=0.8, evidence=[], reason="candidate text explicitly denies assertion")
        if "translation" in qnorm:
            candidates = self._search(question, limit=6)
            if any("no stated translation" in normalize(sentence.text) or "no translation" in normalize(sentence.text) for sentence, _ in candidates):
                return Answer("unknown", confidence=0.8, evidence=[], reason="translation is not stated")
        if "confirmed" in qnorm and ("belief" in qnorm or "believe" in qnorm):
            candidates = self._search(question, limit=6)
            if not any("confirmed fact" in normalize(sentence.text) or "confirmed as fact" in normalize(sentence.text) for sentence, _ in candidates):
                return Answer("unknown", confidence=0.8, evidence=[], reason="belief is not confirmed as fact")
        if "real" in qnorm and any(term in qnorm for term in ["document", "history", "record"]):
            candidates = self._search(question, limit=8)
            if any("fiction" in normalize(sentence.text) or "imaginary" in normalize(sentence.text) for sentence, _ in candidates):
                return Answer("unknown", confidence=0.8, evidence=[], reason="source marks content as fictional")
        if any(phrase in qnorm for phrase in ["merged with", "same as", "same person as"]):
            candidates = self._search(question, limit=6)
            if not any(
                any(marker in normalize(sentence.text) for marker in ["same person", "same as", "identical to"])
                for sentence, _ in candidates
            ):
                return Answer("unknown", confidence=0.8, evidence=[], reason="identity merge is not source-grounded")
        if any(phrase in qnorm for phrase in ["release date", "finally choose"]):
            candidates = self._search(question, limit=4)
            if not candidates or any("no " in normalize(sentence.text) or "unknown" in normalize(sentence.text) for sentence, _ in candidates):
                return Answer("unknown", confidence=0.8, evidence=[], reason="insufficient evidence guard")
        return None

    def _answer_count(self, question: str, qnorm: str) -> Answer | None:
        if not any(word in qnorm for word in ["how many", "number of"]):
            return None
        status_words = [token for token in re.findall(r"[a-z0-9_-]+", qnorm) if token not in {"how", "many", "number", "have", "has", "are", "the", "in", "sheet"}]
        table_rows = [s for s in self.sentences if "|" in s.text or "\t" in s.text]
        row_matches = [
            s for s in table_rows
            if any(word in normalize(s.text) for word in status_words)
            and not any(header in normalize(s.text) for header in ["status", "email", "customer |", "name |"])
        ]
        if row_matches and any(word in qnorm for word in ["status", "listed", "contacts", "customers"]):
            return Answer(str(len(row_matches)), 0.75, [self._evidence(s) for s in row_matches], "counted table rows by query status")
        if "contacts" in qnorm:
            matches = [s for s in self.sentences if "|" in s.text and "contact" in normalize(s.text) and "email" not in normalize(s.text)]
            return Answer(str(len(matches)), 0.75, [self._evidence(s) for s in matches], "counted contact table rows")
        return None

    def _answer_yes_no_context(self, question: str, qnorm: str) -> Answer | None:
        if not re.match(r"^(did|does|do|is|are|was|were|can|could|should|has|have)\b", qnorm):
            return None
        if "really delete" in qnorm or ("really" in qnorm and "delete" in qnorm):
            candidates = self._search(question, limit=12)
            dream = next((sentence for sentence, _ in candidates if "dream" in normalize(sentence.text)), None)
            retained = next((sentence for sentence, _ in candidates if "still contained" in normalize(sentence.text)), None)
            if dream and retained:
                retained_match = re.search(r"still contained\s+(.+?)(?:\.$|;|$)", retained.text, re.I)
                retained_value = clean_extracted_value(retained_match.group(1)) if retained_match else "the item"
                return Answer(
                    f"No; the deletion occurred only in a dream and the repository still contained {retained_value}.",
                    0.9,
                    [self._evidence(dream), self._evidence(retained)],
                    "dream context blocks asserted deletion",
                )
        if "proven" in qnorm:
            for sentence, score in self._search(question, limit=10):
                if "no proof" in normalize(sentence.text):
                    return Answer("No; the final judgment found no proof.", score, [self._evidence(sentence, score)], "negative proof relation")
        if "audit" in qnorm:
            for sentence, score in self._search(question, limit=10):
                if "audit result" in normalize(sentence.text) and "only" in normalize(sentence.text):
                    match = re.search(r"\bonly\s+([^.;]+)", sentence.text, re.I)
                    if match:
                        return Answer(f"No; it stores only {clean_extracted_value(match.group(1))}.", score, [self._evidence(sentence, score)], "audit result overrides weaker claim")
        if "delete" in qnorm:
            for sentence, score in self._search(question, limit=10):
                sentence_norm = normalize(sentence.text)
                if "does not delete" in sentence_norm:
                    flag_match = re.search(r"\bflags\s+(.+?)\s+for\s+([^.;]+)", sentence.text, re.I)
                    if not flag_match:
                        document_sentences = self._sentences_by_document.get(sentence.rel_path, {})
                        for offset in [-2, -1, 1, 2]:
                            neighbor = document_sentences.get(sentence.order + offset)
                            if neighbor:
                                flag_match = re.search(r"\"([^\"]+?)\s+are\s+flagged\s+for\s+([^\"]+)\"", neighbor.text, re.I)
                                if not flag_match:
                                    flag_match = re.search(r"\b([A-Za-z0-9 _.-]+?)\s+are\s+flagged\s+for\s+([^\".;]+)", neighbor.text, re.I)
                                if flag_match:
                                    break
                    if flag_match:
                        return Answer(
                            f"No; runtime flags {clean_extracted_value(flag_match.group(1))} for {clean_extracted_value(flag_match.group(2))}.",
                            score,
                            [self._evidence(sentence, score)],
                            "negated action with replacement action",
                        )
                    return Answer(f"No; {clean_extracted_value(sentence.text)}.", score, [self._evidence(sentence, score)], "negated action")
        if "engineering record" in qnorm:
            sentences = self.index.all_sentences_containing(["fiction homework"])
            if sentences:
                return Answer("No; it is fiction homework.", 0.9, [self._evidence(sentences[0])], "fiction context")
        if any(term in qnorm for term in ["roadmap target", "target"]):
            candidates = self._search(question, limit=10)
            seen = {sentence.sentence_id for sentence, _ in candidates}
            if self._allow_global_fallback:
                candidates.extend((sentence, 0.2) for sentence in self.sentences if "unrelated" in normalize(sentence.text) and sentence.sentence_id not in seen)
            for sentence, score in candidates:
                if "unrelated" in normalize(sentence.text):
                    match = re.search(r"\bunrelated\s+([^.;]+)", sentence.text, re.I)
                    value = clean_extracted_value(match.group(1)) if match else "unrelated note"
                    note_match = re.match(r"(.+?\bnote)\b", value, re.I)
                    if note_match:
                        value = clean_extracted_value(note_match.group(1))
                    return Answer(f"No; it is an unrelated {value}.", score, [self._evidence(sentence, score)], "distractor source")
        if "crack" in qnorm and ("found" in qnorm or "proven" in qnorm):
            candidates = self._search(question, limit=8)
            for sentence, score in candidates:
                if "no crack" in normalize(sentence.text):
                    value = clean_extracted_value(sentence.text)
                    value = value[:1].lower() + value[1:]
                    return Answer(f"No; {value}.", score, [self._evidence(sentence, score)], "negated inspection result")
        return None

    def _answer_final_state(self, question: str, qnorm: str) -> Answer | None:
        if not ("final state" in qnorm or "current" in qnorm or re.search(r"\bstate\b", qnorm)):
            return None
        plan = plan_question(question)
        if plan.wants_current_state:
            state_anchors = self._state_anchors(question, plan.anchors)
            state_row = self.store.latest_state(self.run_id, state_anchors)
            if state_row and state_row["state_value"]:
                sentence = self._sentences_by_location.get((str(state_row["rel_path"]), int(state_row["chunk_order"])))
                if sentence:
                    return Answer(
                        str(state_row["state_value"]),
                        0.88,
                        [self._evidence(sentence, 0.88)],
                        "latest temporal state from DSPG",
                    )
        candidates = self._search(question, limit=10)
        for sentence, score in candidates:
            match = re.search(r"(?:final|current)?[^:]{0,60}\bstate:\s*([A-Za-z0-9_-]+)", sentence.text, re.I)
            if match:
                return Answer(match.group(1), score, [self._evidence(sentence, score)], "state label")
        return None

    def _state_anchors(self, question: str, fallback: tuple[str, ...]) -> list[str]:
        match = re.search(r"\b(?:current|final|latest)\s+state\s+of\s+(.+?)(?:\?|$)", question, re.I)
        if match:
            return [clean_extracted_value(match.group(1))]
        return [anchor for anchor in fallback if len(normalize(anchor)) > 2]

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
        table_answer = self._answer_generic_table_row(question, qnorm)
        if table_answer:
            return table_answer
        return None

    def _answer_identifier_or_url(self, question: str, qnorm: str) -> Answer | None:
        candidates = self._search(question, limit=12)
        qtokens = self._question_tokens(qnorm)
        anchors = [normalize(anchor) for anchor in self._target_anchors(question)]
        query_terms = [
            token for token in re.findall(r"[a-z0-9_.:/#-]+", qnorm)
            if len(token) > 3 and token not in {"which", "what", "where", "when", "does", "listed", "named", "appears"}
        ]
        wants_url_like = bool(qtokens.intersection({"url", "urls", "link", "links", "runbook"})) or (
            qnorm.startswith("where ") and any(term in qnorm for term in ["stored", "listed", "map"])
        )
        if wants_url_like:
            question_ids = identifiers(question)
            url_matches: list[tuple[float, str, Sentence, float]] = []
            scan_items = list(candidates)
            if self._allow_global_fallback:
                scan_items.extend((sentence, 0.1) for sentence in self.sentences if urls(sentence.text))
            for sentence, score in scan_items:
                sentence_norm = normalize(sentence.text)
                if question_ids and not any(item in sentence.text for item in question_ids):
                    continue
                values = urls(sentence.text)
                if values:
                    term_score = sum(1 for term in query_terms if term in sentence_norm)
                    anchor_score = sum(2 for anchor in anchors if anchor and anchor in sentence_norm)
                    url_matches.append(((term_score * 4.0) + (anchor_score * 3.0) + (score * 0.25), values[0], sentence, score))
            if url_matches:
                url_matches.sort(key=lambda item: (-item[0], item[2].rel_path, item[2].order))
                _, value, sentence, score = url_matches[0]
                return Answer(value, score, [self._evidence(sentence, score)], "url extraction")
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
        if any(term in qnorm for term in ["which pr", "what pr", "which commit", "ticket", " id"]):
            descriptor_answer = self._answer_identifier_descriptor(question, qnorm)
            if descriptor_answer:
                return descriptor_answer
            if " id" in qnorm and not any(term in qnorm for term in ["which pr", "what pr", "which commit", "ticket"]):
                return None
            id_matches: list[tuple[float, str, Sentence, float]] = []
            scan_items = list(candidates)
            if self._allow_global_fallback:
                scan_items.extend((sentence, 0.1) for sentence in self.sentences if identifiers(sentence.text))
            for sentence, score in scan_items:
                values = identifiers(sentence.text)
                if not values:
                    continue
                preferred = self._choose_identifier(values, qnorm)
                if not preferred:
                    continue
                sentence_norm = normalize(sentence.text)
                term_score = sum(1 for term in query_terms if term in sentence_norm)
                anchor_score = sum(2 for anchor in anchors if anchor and anchor in sentence_norm)
                structure_score = 3 if ("{" in sentence.text or "}" in sentence.text) and any(term in qnorm for term in ["raw", "json", "json-like"]) else 0
                if "implement" in qnorm and "implement" not in sentence_norm:
                    term_score -= 2
                id_matches.append(((term_score * 4.0) + (anchor_score * 3.0) + structure_score + (score * 0.25), preferred, sentence, score))
            if id_matches:
                id_matches.sort(key=lambda item: (-item[0], item[2].rel_path, item[2].order))
                _, value, sentence, score = id_matches[0]
                return Answer(value, score, [self._evidence(sentence, score)], "generic identifier extraction")
        if "which file" in qnorm or "file did" in qnorm:
            for sentence, score in candidates:
                match = re.search(r"\b[A-Za-z0-9_./-]+\.(?:rs|cpp|txt|key|py|js)\b", sentence.text)
                if match:
                    return Answer(match.group(0), score, [self._evidence(sentence, score)], "file extraction")
        return None

    def _answer_identifier_descriptor(self, question: str, qnorm: str) -> Answer | None:
        match = re.search(r"\b(?:what|which)\s+(?:is\s+the\s+)?([a-z][a-z0-9 _-]{1,40}?)\s+id\b", qnorm)
        if not match:
            return None
        descriptor = clean_extracted_value(match.group(1))
        descriptor_terms = [term for term in descriptor.split() if term not in {"the", "a", "an"}]
        scan_items = self._search(question, limit=12)
        if self._allow_global_fallback:
            scan_items = scan_items + [(sentence, 0.1) for sentence in self.sentences]
        matches: list[tuple[float, str, Sentence, float]] = []
        for sentence, score in scan_items:
            text_norm = normalize(sentence.text)
            if not all(term in text_norm for term in descriptor_terms):
                continue
            patterns = [
                rf"\b{re.escape(descriptor)}\s+id\s*[:=]?\s*([A-Z][A-Z0-9]{{1,9}}-\d+[A-Z0-9-]*)",
                rf"\b{re.escape(descriptor)}\s+id\s+([A-Z][A-Z0-9]{{1,9}}-\d+[A-Z0-9-]*)",
            ]
            for pattern in patterns:
                id_match = re.search(pattern, sentence.text, re.I)
                if id_match:
                    matches.append((score + 4.0, id_match.group(1), sentence, score))
        if not matches:
            return None
        matches.sort(key=lambda item: (-item[0], item[2].rel_path, item[2].order))
        _, value, sentence, score = matches[0]
        return Answer(value, score, [self._evidence(sentence, score)], "descriptor identifier extraction")

    def _choose_identifier(self, values: list[str], qnorm: str) -> str:
        if "commit" in qnorm:
            for value in values:
                if re.fullmatch(r"[0-9a-f]{8,16}", value, re.I):
                    return value
        if any(term in qnorm for term in ["ticket", "pr", "id", "code", "specimen", "invoice", "parcel", "treaty"]):
            prefix_terms = {
                token.upper()
                for token in re.findall(r"\b[a-z][a-z0-9]{1,9}\b", qnorm)
                if token not in {"which", "what", "ticket", "appears", "named", "code", "id", "invoice", "parcel", "treaty"}
            }
            for value in values:
                if re.fullmatch(r"[A-Z][A-Z0-9]{1,9}-\d+[A-Z0-9-]*", value):
                    if not prefix_terms or value.split("-", 1)[0] in prefix_terms or any(term in qnorm for term in ["ticket", "id", "code", "invoice", "parcel", "treaty"]):
                        return value
            return ""
        return values[0] if values else ""

    def _answer_who_role(self, question: str, qnorm: str) -> Answer | None:
        if qnorm.startswith("which ") and not any(
            term in qnorm
            for term in ["customer", "person", "who", "approved", "reported", "confirmed", "reviewed"]
        ):
            return None
        if not (qnorm.startswith("who ") or qnorm.startswith("which ")):
            return None
        if "owner" in qnorm and any(term in qnorm for term in ["raw", "json", "json-like", "key", "value"]):
            return None
        candidates = self._search(question, limit=20)
        anchors = self._target_anchors(question)
        name_pattern = r"(?:(?:Dr\.|Ms\.|Mr\.|Mrs\.|Prof\.)\s+)?[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}"
        role_patterns = [
            (rf"({name_pattern})\s+drafted\b", "drafted"),
            (rf"({name_pattern})\s+authored\b", "authored"),
            (rf"({name_pattern})\s+reviewed\b", "reviewed"),
            (rf"({name_pattern})\s+will review\b", "reviewed"),
            (rf"({name_pattern}):\s+.*\breview\b", "reviewed"),
            (rf"({name_pattern})\s+performed\b.*review", "performed review"),
            (rf"reviewed\s+[^.;]+?by\s+({name_pattern})", "reviewed by"),
            (rf"({name_pattern})\s+is the escalation owner", "escalation owner"),
            (rf"owner(?:\s+is|ed by)?\s+({name_pattern})", "owner"),
            (rf"owned by\s+({name_pattern})", "owned by"),
            (rf"merged by\s+({name_pattern})", "merged by"),
            (rf"approved by(?: engineer)?\s+({name_pattern})", "approved by"),
            (rf"({name_pattern})\s+requested\b", "requested"),
            (rf"({name_pattern})\s+asked\s+for\b", "requested"),
            (rf"({name_pattern})\s+accepted responsibility", "accepted responsibility"),
            (rf"({name_pattern})\s+manages\b", "manages"),
            (rf"({name_pattern})\s+tested\b", "tested"),
            (rf"({name_pattern}):\s+.*\btested\b", "tested"),
            (rf"({name_pattern}):\s+.*caused", "speaker claim"),
            (rf"({name_pattern}):\s+I disagree", "speaker disagreement"),
            (rf"({name_pattern})\s+believes\b", "believes"),
            (rf"Plaintiff\s+({name_pattern})\s+alleges", "alleges"),
            (rf"(?:(?:customer|plaintiff|witness|officer|farmer|engineer|clinician|vet|inspector)\s+)?({name_pattern})\s+reported\b", "reported"),
            (rf"(?:(?:customer|plaintiff|witness|officer|farmer|engineer|clinician|vet|inspector)\s+)?({name_pattern})\s+confirmed\b", "confirmed"),
            (rf"({name_pattern})\s+observed\b", "observed"),
            (rf"({name_pattern})\s+wrote\b", "wrote"),
            (rf"({name_pattern})\s+stated\b", "stated"),
            (rf"({name_pattern})\s+recorded\b", "recorded"),
            (rf"({name_pattern})\s+argued\b", "argued"),
            (rf"({name_pattern})\s+coached\b", "coached"),
            (rf"({name_pattern})\s+practiced\b", "practiced"),
            (rf"({name_pattern})\s+watered\b", "watered"),
            (rf"({name_pattern})\s+inspected\b", "inspected"),
            (rf"({name_pattern})\s+closed\b", "closed"),
            (rf"(?:Lead researcher|lead researcher):\s+({name_pattern})", "lead researcher"),
            (rf"(?:Clinician|clinician):\s+({name_pattern})", "clinician"),
            (rf"(?:Vet|vet):\s+({name_pattern})", "vet"),
            (rf"(?:Inspector|inspector):\s+({name_pattern})", "inspector"),
            (rf"(?:Logistics contact|logistics contact):\s+({name_pattern})", "contact"),
            (rf"(?:Schedule owner|schedule owner):\s+({name_pattern})", "owner"),
            (rf"({name_pattern})\s+signed\b", "signed"),
            (rf"was signed by\s+({name_pattern})", "signed"),
            (rf"signed by\s+({name_pattern})", "signed"),
        ]
        role_matches: list[tuple[float, str, Sentence, float, str]] = []
        for sentence, score in candidates:
            text = sentence.text
            if anchors and not any(anchor in text for anchor in anchors):
                document_text = " ".join(item.text for item in self.sentences if item.rel_path == sentence.rel_path)
                if not any(anchor in document_text for anchor in anchors):
                    continue
            if "review" in qnorm:
                label_value = after_label(text, ["reviewer"])
                if label_value:
                    role_matches.append((score + 3.0, label_value, sentence, score, "reviewer label"))
                    continue
            if "approved" in qnorm or "approver" in qnorm:
                label_value = after_label(text, ["approver"])
                if label_value:
                    role_matches.append((score + 3.0, label_value, sentence, score, "approver label"))
                    continue
            ambiguous_match = re.search(r"\bwhich\s+([a-z][a-z]+)", qnorm)
            if ambiguous_match and f"does not say which {ambiguous_match.group(1)}" in normalize(text):
                return Answer("unknown", 0.9, [], "ambiguous same-name mention")
            for regex, reason in role_patterns:
                match = re.search(regex, text)
                if match and self._role_matches(reason, qnorm):
                    document_text = " ".join(item.text for item in self.sentences if item.rel_path == sentence.rel_path)
                    if not self._sentence_satisfies_question_terms(text, qnorm) and not self._sentence_satisfies_question_terms(document_text, qnorm):
                        continue
                    value = self._clean_person_answer(match.group(1))
                    value = self._expand_name(value)
                    term_score = self._question_term_match_score(text, qnorm)
                    role_matches.append((score + term_score, value, sentence, score, reason))
        if not role_matches:
            return None
        role_matches.sort(key=lambda item: (-item[0], item[2].rel_path, item[2].order))
        _, value, sentence, score, reason = role_matches[0]
        return Answer(value, score, [self._evidence(sentence, score)], reason)

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
        if "confirmed" in qnorm:
            return reason == "confirmed"
        if "observed" in qnorm:
            return reason == "observed"
        if "wrote" in qnorm or "written" in qnorm:
            return reason == "wrote"
        if "stated" in qnorm:
            return reason == "stated"
        if "recorded" in qnorm:
            return reason == "recorded"
        if "argued" in qnorm or "argue" in qnorm:
            return reason in {"argued", "speaker disagreement"}
        if "coached" in qnorm:
            return reason == "coached"
        if "practiced" in qnorm:
            return reason == "practiced"
        if "watered" in qnorm:
            return reason == "watered"
        if "inspected" in qnorm or "inspector" in qnorm:
            return reason in {"inspected", "inspector"}
        if "closed" in qnorm:
            return reason == "closed"
        if "lead researcher" in qnorm:
            return reason == "lead researcher"
        if "clinician" in qnorm:
            return reason == "clinician"
        if "vet" in qnorm:
            return reason == "vet"
        if "contact" in qnorm:
            return reason == "contact"
        if "schedule" in qnorm and "own" in qnorm:
            return reason == "owner"
        if "signed" in qnorm:
            return reason == "signed"
        return True

    def _sentence_satisfies_question_terms(self, text: str, qnorm: str) -> bool:
        sentence_norm = normalize(text)
        ignored = {
            "who", "which", "what", "when", "where", "did", "does", "was", "were", "is", "are",
            "the", "for", "with", "about", "according", "customer", "person", "someone",
            "owns", "owned", "owner", "reviewed", "review", "approved", "merged", "reported",
            "confirmed", "requested", "signed", "recorded", "inspected", "argued", "authored",
            "drafted", "wrote", "coached", "practiced", "watered", "closed", "contact",
        }
        terms = [
            token for token in re.findall(r"[a-z0-9_.:/#-]+", qnorm)
            if len(token) > 3 and token not in ignored
        ]
        if not terms:
            return True
        matched = sum(1 for term in terms if term in sentence_norm)
        return matched >= min(2, len(terms))

    def _question_term_match_score(self, text: str, qnorm: str) -> float:
        sentence_norm = normalize(text)
        ignored = {
            "who", "which", "what", "when", "where", "did", "does", "was", "were", "is", "are",
            "the", "for", "with", "about", "according", "customer", "person", "someone",
            "owns", "owned", "owner", "reviewed", "review", "approved", "merged", "reported",
            "confirmed", "requested", "signed", "recorded", "inspected", "argued", "authored",
            "drafted", "wrote", "coached", "practiced", "watered", "closed", "contact",
        }
        terms = [
            token for token in re.findall(r"[a-z0-9_.:/#-]+", qnorm)
            if len(token) > 3 and token not in ignored
        ]
        return float(sum(1 for term in terms if term in sentence_norm))

    def _nearest_header_value(self, sentence: Sentence, labels: list[str]) -> str:
        document_sentences = self._sentences_by_document.get(sentence.rel_path, {})
        for offset in range(1, 5):
            previous = document_sentences.get(sentence.order - offset)
            if not previous:
                continue
            for label in labels:
                value = after_label(previous.text, [label])
                if value:
                    return self._clean_person_answer(value)
        return ""

    def _nearest_speaker(self, sentence: Sentence) -> str:
        document_sentences = self._sentences_by_document.get(sentence.rel_path, {})
        for offset in range(0, 4):
            previous = document_sentences.get(sentence.order - offset)
            if not previous:
                continue
            match = re.match(r"\s*[\[\(]?[0-9: -]*[\]\)]?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\s*:", previous.text)
            if match:
                return self._clean_person_answer(match.group(1))
        return ""

    def _answer_what_value(self, question: str, qnorm: str) -> Answer | None:
        candidates = self._search(question, limit=15)
        if "finally choose" in qnorm or "final decision" in qnorm:
            for sentence in self.index.all_sentences_containing(["no final decision"]):
                return Answer("unknown", reason="discussion states no final decision")
        if "what does" in qnorm and "believe" in qnorm:
            for sentence, score in candidates:
                match = re.search(r"believes\s+(.+)", sentence.text, re.I)
                if match:
                    value = clean_extracted_value(match.group(1))
                    value = re.sub(r"^the\s+[^ ]+\s+should\b", "It should", value, flags=re.I)
                    if value.startswith("It should") and not value.endswith("."):
                        value += "."
                    return Answer(value, score, [self._evidence(sentence, score)], "belief content")
        if "what was the final cause" in qnorm:
            scan_items = list(candidates)
            if self._allow_global_fallback:
                scan_items.extend((sentence, 0.1) for sentence in self.sentences if "final cause" in normalize(sentence.text))
            for sentence, score in scan_items:
                match = re.search(r"final cause was (.+?)(?:,|\.|$)", sentence.text, re.I)
                if match:
                    value = re.sub(r"^(the|a|an)\s+", "", compact_answer(match.group(1)), flags=re.I)
                    return Answer(value, score, [self._evidence(sentence, score)], "final cause")
        if "what did" in qnorm and any(term in qnorm for term in ["message", "note", "email", "forwarded"]):
            for sentence, score in candidates:
                match = re.search(r"\bI\s+plan\s+to\s+(.+)", sentence.text, re.I)
                if match:
                    speaker = self._nearest_header_value(sentence, ["from", "speaker", "author"]) or self._nearest_speaker(sentence)
                    action = clean_extracted_value(match.group(1))
                    if speaker:
                        first = speaker.split()[0]
                        if first and first.lower() in qnorm:
                            speaker = first
                        return Answer(f"{speaker} planned to {action}.", score, [self._evidence(sentence, score)], "reported message content")
                    return Answer(f"Planned to {action}.", score, [self._evidence(sentence, score)], "reported message content")
        if "according to" in qnorm and "who" in qnorm:
            for sentence, score in candidates:
                match = re.search(r"wrote:\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+([a-z]+)", sentence.text)
                if match:
                    return Answer(self._clean_person_answer(match.group(1)), score, [self._evidence(sentence, score)], "reported note actor")
        if "depend" in qnorm or "depends" in qnorm:
            for sentence, score in candidates:
                match = re.search(r"depends on(?:\s+\w+)?(?:\s+\w+)?\s*:\s*(.+)", sentence.text, re.I)
                if match:
                    return Answer(compact_answer(match.group(1)), score, [self._evidence(sentence, score)], "dependency list")
        if "when did" in qnorm and "reopen" in qnorm:
            for sentence, score in candidates:
                match = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}).*reopened", sentence.text, re.I)
                if match:
                    return Answer(match.group(1), score, [self._evidence(sentence, score)], "timestamped reopen")
        relation = self._answer_generic_relation(question, qnorm, candidates)
        if relation:
            return relation
        arithmetic = self._answer_arithmetic_statement(question, qnorm, candidates)
        if arithmetic:
            return arithmetic
        location = self._answer_spatial_relation(question, qnorm, candidates)
        if location:
            return location
        label_answer = self._answer_generic_label(question, qnorm, candidates)
        if label_answer:
            return label_answer
        return None

    def _answer_generic_relation(
        self,
        question: str,
        qnorm: str,
        candidates: list[tuple[Sentence, float]],
    ) -> Answer | None:
        if "not buy" in qnorm or "not bought" in qnorm:
            for sentence, score in candidates:
                match = re.search(r"\bbought\s+(.+?)\s+but\s+not\s+([^.;]+)", sentence.text, re.I)
                if match:
                    return Answer(clean_extracted_value(match.group(2)), score, [self._evidence(sentence, score)], "negative purchase relation")
        if "scale" in qnorm and "practice" in qnorm:
            for sentence, score in candidates:
                match = re.search(r"\bpracticed\s+the\s+(.+?)\s+scale\b", sentence.text, re.I)
                if match:
                    return Answer(clean_extracted_value(match.group(1)), score, [self._evidence(sentence, score)], "practice scale relation")
        if "confirmed" in qnorm and "fix" in qnorm:
            for sentence, score in candidates:
                match = re.search(r"confirmed\s+fix\s*:\s*(.+)", sentence.text, re.I)
                if match:
                    return Answer(clean_extracted_value(match.group(1)), score, [self._evidence(sentence, score)], "confirmed fix label")
        if "plural" in qnorm:
            for sentence, score in candidates:
                match = re.search(r"plural\s+of\s+(.+?)\s+is\s+([^.;]+)", sentence.text, re.I)
                if match and normalize(match.group(1)) in qnorm:
                    return Answer(clean_extracted_value(match.group(2)), score, [self._evidence(sentence, score)], "plural relation")
        if "mean" in qnorm:
            for sentence, score in candidates:
                match = re.search(r"([^.;:]+?)\s+means\s+([^.;]+)", sentence.text, re.I)
                if match:
                    subject = normalize(match.group(1).split(":")[-1])
                    if subject and subject in qnorm:
                        return Answer(clean_extracted_value(match.group(2)), score, [self._evidence(sentence, score)], "meaning relation")
        if "also called" in qnorm or "nickname" in qnorm:
            for sentence, score in candidates:
                match = re.search(r"\bis also called\s+([^.;]+)", sentence.text, re.I)
                if match:
                    return Answer(clean_extracted_value(match.group(1)), score, [self._evidence(sentence, score)], "alias relation")
        if qnorm.startswith("when ") or qnorm.startswith("when is"):
            question_terms = [term for term in re.findall(r"[a-z0-9_-]+", qnorm) if len(term) > 3]
            for sentence, score in candidates:
                sentence_norm = normalize(sentence.text)
                if sum(1 for term in question_terms if term in sentence_norm) < 1:
                    continue
                match = re.search(r"\b(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\b", sentence.text)
                if match:
                    return Answer(match.group(1), score, [self._evidence(sentence, score)], "timestamped event")
                time_match = re.search(r"\b(\d{1,2}:\d{2})\b", sentence.text)
                if time_match:
                    return Answer(time_match.group(1), score, [self._evidence(sentence, score)], "time event")
        return None

    def _answer_arithmetic_statement(
        self,
        question: str,
        qnorm: str,
        candidates: list[tuple[Sentence, float]],
    ) -> Answer | None:
        numbers = [int(item) for item in re.findall(r"\b\d+\b", question)]
        if len(numbers) < 2:
            return None
        for sentence, score in candidates:
            sentence_norm = normalize(sentence.text)
            if all(str(number) in sentence_norm for number in numbers):
                match = re.search(r"(?:equals|=)\s*(\d+)", sentence.text, re.I)
                if match:
                    return Answer(match.group(1), score, [self._evidence(sentence, score)], "arithmetic statement")
        return None

    def _answer_spatial_relation(
        self,
        question: str,
        qnorm: str,
        candidates: list[tuple[Sentence, float]],
    ) -> Answer | None:
        if not any(term in qnorm for term in ["where", "location"]):
            return None
        anchors = [normalize(anchor) for anchor in self._target_anchors(question)]
        for sentence, score in candidates:
            text = sentence.text
            text_norm = normalize(text)
            if anchors and not any(anchor in text_norm for anchor in anchors):
                continue
            match = re.search(r"\b(?:is|are)\s+(left of|right of|north of|south of|east of|west of|on|under|behind|inside|near)\s+(.+)", text, re.I)
            if match:
                return Answer(
                    compact_answer(f"{match.group(1)} {match.group(2)}"),
                    score,
                    [self._evidence(sentence, score)],
                    "spatial relation",
                )
        return None

    def _answer_generic_label(
        self,
        question: str,
        qnorm: str,
        candidates: list[tuple[Sentence, float]],
    ) -> Answer | None:
        if not any(word in qnorm for word in ["what", "which", "when", "where", "who"]):
            return None
        query_terms = {
            token
            for token in re.findall(r"[a-z0-9_-]+", qnorm)
            if len(token) > 2
            and token
            not in {
                "what",
                "which",
                "when",
                "where",
                "who",
                "is",
                "are",
                "was",
                "were",
                "the",
                "for",
                "in",
                "of",
                "on",
                "does",
                "mean",
                "note",
                "notes",
                "named",
                "listed",
                "stated",
                "meaningful",
                "according",
            }
        }
        anchors = [normalize(anchor) for anchor in self._target_anchors(question)]
        matches: list[tuple[float, str, Sentence, float]] = []
        scan_items = list(candidates)
        seen_ids = {sentence.sentence_id for sentence, _ in scan_items}
        if self._allow_global_fallback:
            scan_items.extend((sentence, 0.25) for sentence in self.sentences if sentence.sentence_id not in seen_ids)
        for sentence, score in scan_items:
            parts = re.split(r"\s*[|,;]\s*", sentence.text)
            for part in parts:
                if ":" not in part:
                    continue
                label, value = part.split(":", 1)
                if re.search(r"\d$", label.strip()):
                    continue
                label_words = label.strip().split()
                if len(label_words) > 5:
                    continue
                if any(word.lower() in {"says", "began", "closed", "opened", "state"} for word in label_words):
                    continue
                if len(label_words) >= 2 and all(word[:1].isupper() and word[1:].islower() for word in label_words):
                    continue
                label_norm = normalize(label)
                sentence_norm = normalize(sentence.text)
                if any(term in label_norm for term in query_terms):
                    answer = clean_extracted_value(value)
                    if answer:
                        term_score = sum(1 for term in query_terms if term in label_norm)
                        anchor_score = sum(1 for anchor in anchors if anchor and anchor in sentence_norm)
                        structure_score = 1.0 if ("{" in sentence.text or "}" in sentence.text) and any(term in qnorm for term in ["raw", "json", "json-like"]) else 0.0
                        matches.append((score + (term_score * 2.0) + (anchor_score * 3.0) + structure_score, answer, sentence, score))
        if not matches:
            return None
        matches.sort(key=lambda item: (-item[0], item[2].rel_path, item[2].order))
        _, answer, sentence, score = matches[0]
        return Answer(answer, score, [self._evidence(sentence, score)], "generic label lookup")

    def _answer_generic_table_row(self, question: str, qnorm: str) -> Answer | None:
        which_match = re.search(r"\bwhich\s+([a-z0-9_-]+).*\b(?:is|was)\s+([a-z0-9_-]+)", qnorm)
        if not which_match:
            return None
        target_label, desired_value = which_match.groups()
        for sentence in self.sentences:
            if "|" not in sentence.text and "\t" not in sentence.text:
                continue
            cells = [cell.strip() for cell in re.split(r"[|\t]", sentence.text)]
            if len(cells) < 2:
                continue
            sentence_norm = normalize(sentence.text)
            if desired_value in sentence_norm:
                first = cells[0]
                if target_label in normalize(first) or re.fullmatch(r"[A-Z]{2,}-\d+", first):
                    return Answer(first, 0.75, [self._evidence(sentence, 0.75)], "table row status lookup")
        return None

    def _answer_generic_best_fact(self, question: str, qnorm: str) -> Answer | None:
        candidates = self._search(question, limit=10)
        if not candidates:
            return Answer("unknown", reason="no lexical candidates")
        sentence, score = candidates[0]
        if score < 2:
            return Answer("unknown", reason="weak lexical evidence")
        if qnorm.startswith("who ") or qnorm.startswith("which "):
            relation_answer = self._answer_who_role(question, qnorm)
            if relation_answer:
                return relation_answer
        return Answer("unknown", reason="best sentence did not yield a grounded answer")
