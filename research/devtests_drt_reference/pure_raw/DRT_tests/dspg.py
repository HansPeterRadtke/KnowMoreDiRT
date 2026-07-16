#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ALLOWED_REF_TYPES = {
    "person",
    "employee",
    "customer",
    "company",
    "product",
    "system",
    "concept",
    "object",
    "file",
    "pr",
    "bug",
    "issue",
    "ticket",
    "url",
    "artifact",
    "event",
    "time",
    "proposition",
    "context",
    "data",
    "unknown",
}

ALLOWED_MENTION_TYPES = {
    "person",
    "employee",
    "customer",
    "company",
    "product",
    "system",
    "concept",
    "object",
    "file",
    "pr",
    "bug",
    "issue",
    "ticket",
    "url",
    "artifact",
    "event",
    "time",
    "data",
    "pronoun",
    "descriptor",
    "unknown",
}

ALLOWED_CONTEXT_KINDS = {
    "asserted",
    "negated",
    "conditional_antecedent",
    "conditional_consequent",
    "believed",
    "reported",
    "quoted",
    "dreamed",
    "possible",
    "uncertain",
    "hypothetical",
}

ALLOWED_HYPOTHESIS_STATUSES = {
    "accepted",
    "candidate",
    "rejected",
    "ambiguous",
}


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    text = value.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


@dataclass
class SourceSpan:
    source_file: str
    char_start: int = 0
    char_end: int = 0
    line_start: int | None = None
    line_end: int | None = None
    text: str = ""


@dataclass
class SourceChunk:
    id: str
    source_name: str
    text: str
    order: int = 0
    token_estimate: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Mention:
    id: str
    text: str
    mention_type: str
    chunk_id: str
    sentence_index: int = 0
    order: int = 0
    source_span: SourceSpan | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Referent:
    id: str
    type: str
    label: str
    aliases: list[str] = field(default_factory=list)
    source_spans: list[SourceSpan] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    def all_labels(self) -> list[str]:
        labels = [self.label] + list(self.aliases)
        seen: set[str] = set()
        out: list[str] = []
        for item in labels:
            text = item.strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
        return out


@dataclass
class Proposition:
    id: str
    predicate: str
    context_id: str
    confidence: float = 1.0
    surface: str = ""
    source_spans: list[SourceSpan] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    id: str
    kind: str
    parent_id: str | None = None
    holder: str | None = None
    source_spans: list[SourceSpan] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    id: str
    source_id: str
    target_id: str
    type: str
    confidence: float = 1.0
    context_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class IdentityLink:
    source_label: str
    target_ref_id: str
    confidence: float
    reason: str
    source_ref_id: str | None = None


@dataclass
class IdentityHypothesis:
    id: str
    mention_id: str
    referent_id: str
    confidence: float
    status: str
    reason: str
    rank: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)


class Graph:
    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self.chunks: dict[str, SourceChunk] = {}
        self.mentions: dict[str, Mention] = {}
        self.referents: dict[str, Referent] = {}
        self.propositions: dict[str, Proposition] = {}
        self.contexts: dict[str, Context] = {}
        self.relations: dict[str, Relation] = {}
        self.identity_links: list[IdentityLink] = []
        self.identity_hypotheses: dict[str, IdentityHypothesis] = {}
        self.metadata: dict[str, Any] = {"name": name}
        self._counters = {
            "chunk": 0,
            "mention": 0,
            "referent": 0,
            "proposition": 0,
            "context": 0,
            "relation": 0,
            "hypothesis": 0,
        }

    def _next_id(self, prefix: str) -> str:
        self._counters[prefix] += 1
        return f"{prefix[0]}{self._counters[prefix]:04d}"

    def ensure_root_context(self) -> str:
        for context in self.contexts.values():
            if context.kind == "asserted" and context.parent_id is None:
                return context.id
        return self.add_context("asserted")

    def add_chunk(
        self,
        source_name: str,
        text: str,
        *,
        chunk_id: str | None = None,
        order: int = 0,
        token_estimate: int | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        chunk_id = chunk_id or self._next_id("chunk")
        chunk = SourceChunk(
            id=chunk_id,
            source_name=source_name,
            text=text,
            order=order,
            token_estimate=int(token_estimate if token_estimate is not None else max(1, len(text.split()))),
            attributes=attributes or {},
        )
        self.chunks[chunk.id] = chunk
        return chunk.id

    def add_mention(
        self,
        text: str,
        mention_type: str,
        chunk_id: str,
        *,
        mention_id: str | None = None,
        sentence_index: int = 0,
        order: int = 0,
        source_span: SourceSpan | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        mention_id = mention_id or self._next_id("mention")
        normalized_type = normalize_text(mention_type).replace(" ", "_")
        if normalized_type not in ALLOWED_MENTION_TYPES:
            normalized_type = "unknown"
        mention = Mention(
            id=mention_id,
            text=text.strip() or mention_id,
            mention_type=normalized_type,
            chunk_id=chunk_id,
            sentence_index=sentence_index,
            order=order,
            source_span=source_span,
            attributes=attributes or {},
        )
        self.mentions[mention.id] = mention
        return mention.id

    def add_context(
        self,
        kind: str,
        *,
        context_id: str | None = None,
        parent_id: str | None = None,
        holder: str | None = None,
        source_span: SourceSpan | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        context_id = context_id or self._next_id("context")
        normalized_kind = normalize_text(kind).replace(" ", "_")
        if normalized_kind not in ALLOWED_CONTEXT_KINDS:
            normalized_kind = "asserted" if not normalized_kind else normalized_kind
        context = Context(
            id=context_id,
            kind=normalized_kind,
            parent_id=parent_id,
            holder=holder,
            source_spans=[source_span] if source_span else [],
            attributes=attributes or {},
        )
        self.contexts[context.id] = context
        return context.id

    def add_referent(
        self,
        ref_type: str,
        label: str,
        *,
        referent_id: str | None = None,
        source_span: SourceSpan | None = None,
        aliases: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        referent_id = referent_id or self._next_id("referent")
        normalized_type = normalize_text(ref_type).replace(" ", "_")
        if normalized_type not in ALLOWED_REF_TYPES:
            normalized_type = "unknown"
        referent = Referent(
            id=referent_id,
            type=normalized_type,
            label=label.strip() or referent_id,
            aliases=aliases or [],
            source_spans=[source_span] if source_span else [],
            attributes=attributes or {},
        )
        self.referents[referent.id] = referent
        return referent.id

    def add_proposition(
        self,
        predicate: str,
        context_id: str,
        *,
        proposition_id: str | None = None,
        source_span: SourceSpan | None = None,
        confidence: float = 1.0,
        surface: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> str:
        proposition_id = proposition_id or self._next_id("proposition")
        proposition = Proposition(
            id=proposition_id,
            predicate=normalize_text(predicate).replace(" ", "_"),
            context_id=context_id,
            confidence=float(confidence),
            surface=surface,
            source_spans=[source_span] if source_span else [],
            attributes=attributes or {},
        )
        self.propositions[proposition.id] = proposition
        return proposition.id

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        *,
        relation_id: str | None = None,
        confidence: float = 1.0,
        context_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        relation_id = relation_id or self._next_id("relation")
        relation = Relation(
            id=relation_id,
            source_id=source_id,
            target_id=target_id,
            type=normalize_text(relation_type).replace(" ", "_"),
            confidence=float(confidence),
            context_id=context_id,
            attributes=attributes or {},
        )
        self.relations[relation.id] = relation
        return relation.id

    def add_identity_link(
        self,
        source_label: str,
        target_ref_id: str,
        confidence: float,
        reason: str,
        *,
        source_ref_id: str | None = None,
    ) -> None:
        self.identity_links.append(
            IdentityLink(
                source_label=source_label,
                target_ref_id=target_ref_id,
                confidence=float(confidence),
                reason=reason,
                source_ref_id=source_ref_id,
            )
        )

    def add_identity_hypothesis(
        self,
        mention_id: str,
        referent_id: str,
        confidence: float,
        status: str,
        reason: str,
        *,
        hypothesis_id: str | None = None,
        rank: int = 0,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        hypothesis_id = hypothesis_id or self._next_id("hypothesis")
        normalized_status = normalize_text(status).replace(" ", "_")
        if normalized_status not in ALLOWED_HYPOTHESIS_STATUSES:
            normalized_status = "candidate"
        hypothesis = IdentityHypothesis(
            id=hypothesis_id,
            mention_id=mention_id,
            referent_id=referent_id,
            confidence=float(confidence),
            status=normalized_status,
            reason=reason,
            rank=int(rank),
            attributes=attributes or {},
        )
        self.identity_hypotheses[hypothesis.id] = hypothesis
        return hypothesis.id

    def accepted_identity_hypotheses(self) -> list[IdentityHypothesis]:
        return [item for item in self.identity_hypotheses.values() if item.status == "accepted"]

    def hypotheses_for_mention(self, mention_id: str) -> list[IdentityHypothesis]:
        items = [item for item in self.identity_hypotheses.values() if item.mention_id == mention_id]
        return sorted(items, key=lambda item: (-item.confidence, item.rank, item.id))

    def accepted_referent_for_mention(self, mention_id: str) -> str | None:
        for item in self.hypotheses_for_mention(mention_id):
            if item.status == "accepted":
                return item.referent_id
        return None

    def unresolved_mentions(self) -> list[Mention]:
        output: list[Mention] = []
        for mention in self.mentions.values():
            if self.accepted_referent_for_mention(mention.id) is None:
                output.append(mention)
        return output

    def get_context_chain(self, context_id: str | None) -> list[Context]:
        chain: list[Context] = []
        seen: set[str] = set()
        current = context_id
        while current and current in self.contexts and current not in seen:
            context = self.contexts[current]
            chain.append(context)
            seen.add(current)
            current = context.parent_id
        return chain

    def proposition_has_context_kind(self, proposition_id: str, kind: str) -> bool:
        normalized_kind = normalize_text(kind).replace(" ", "_")
        proposition = self.propositions[proposition_id]
        return any(context.kind == normalized_kind for context in self.get_context_chain(proposition.context_id))

    def propositions_with_context_kind(self, kind: str) -> list[Proposition]:
        return [
            proposition
            for proposition in self.propositions.values()
            if self.proposition_has_context_kind(proposition.id, kind)
        ]

    def asserted_propositions(self) -> list[Proposition]:
        blocked = {
            "negated",
            "conditional_antecedent",
            "conditional_consequent",
            "believed",
            "reported",
            "quoted",
            "dreamed",
            "possible",
            "uncertain",
        }
        output: list[Proposition] = []
        for proposition in self.propositions.values():
            kinds = {context.kind for context in self.get_context_chain(proposition.context_id)}
            if "asserted" in kinds and not (blocked & kinds):
                output.append(proposition)
        return output

    def validate(self) -> list[str]:
        errors: list[str] = []
        for mention in self.mentions.values():
            if mention.chunk_id not in self.chunks:
                errors.append(f"mention {mention.id} missing chunk {mention.chunk_id}")
        for proposition in self.propositions.values():
            if proposition.context_id not in self.contexts:
                errors.append(f"proposition {proposition.id} missing context {proposition.context_id}")
        valid_ids = set(self.referents) | set(self.propositions) | set(self.contexts)
        for relation in self.relations.values():
            if relation.source_id not in valid_ids:
                errors.append(f"relation {relation.id} missing source {relation.source_id}")
            if relation.target_id not in valid_ids:
                errors.append(f"relation {relation.id} missing target {relation.target_id}")
            if relation.context_id and relation.context_id not in self.contexts:
                errors.append(f"relation {relation.id} missing context {relation.context_id}")
        for hypothesis in self.identity_hypotheses.values():
            if hypothesis.mention_id not in self.mentions:
                errors.append(f"identity hypothesis {hypothesis.id} missing mention {hypothesis.mention_id}")
            if hypothesis.referent_id not in self.referents:
                errors.append(f"identity hypothesis {hypothesis.id} missing referent {hypothesis.referent_id}")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metadata": self.metadata,
            "chunks": [self._dump_dataclass(item) for item in self.chunks.values()],
            "mentions": [self._dump_dataclass(item) for item in self.mentions.values()],
            "referents": [self._dump_dataclass(item) for item in self.referents.values()],
            "propositions": [self._dump_dataclass(item) for item in self.propositions.values()],
            "contexts": [self._dump_dataclass(item) for item in self.contexts.values()],
            "relations": [self._dump_dataclass(item) for item in self.relations.values()],
            "identity_links": [self._dump_dataclass(item) for item in self.identity_links],
            "identity_hypotheses": [self._dump_dataclass(item) for item in self.identity_hypotheses.values()],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Graph":
        graph = cls(payload.get("name", "graph"))
        graph.metadata = payload.get("metadata", {})
        for raw in payload.get("chunks", []):
            graph.chunks[raw["id"]] = SourceChunk(
                id=raw["id"],
                source_name=raw["source_name"],
                text=raw["text"],
                order=int(raw.get("order", 0)),
                token_estimate=int(raw.get("token_estimate", 0)),
                attributes=raw.get("attributes", {}),
            )
        for raw in payload.get("mentions", []):
            span = SourceSpan(**raw["source_span"]) if raw.get("source_span") else None
            graph.mentions[raw["id"]] = Mention(
                id=raw["id"],
                text=raw["text"],
                mention_type=raw["mention_type"],
                chunk_id=raw["chunk_id"],
                sentence_index=int(raw.get("sentence_index", 0)),
                order=int(raw.get("order", 0)),
                source_span=span,
                attributes=raw.get("attributes", {}),
            )
        for raw in payload.get("contexts", []):
            spans = [SourceSpan(**span) for span in raw.get("source_spans", [])]
            graph.contexts[raw["id"]] = Context(
                id=raw["id"],
                kind=raw["kind"],
                parent_id=raw.get("parent_id"),
                holder=raw.get("holder"),
                source_spans=spans,
                attributes=raw.get("attributes", {}),
            )
        for raw in payload.get("referents", []):
            spans = [SourceSpan(**span) for span in raw.get("source_spans", [])]
            graph.referents[raw["id"]] = Referent(
                id=raw["id"],
                type=raw["type"],
                label=raw["label"],
                aliases=raw.get("aliases", []),
                source_spans=spans,
                attributes=raw.get("attributes", {}),
            )
        for raw in payload.get("propositions", []):
            spans = [SourceSpan(**span) for span in raw.get("source_spans", [])]
            graph.propositions[raw["id"]] = Proposition(
                id=raw["id"],
                predicate=raw["predicate"],
                context_id=raw["context_id"],
                confidence=float(raw.get("confidence", 1.0)),
                surface=raw.get("surface", ""),
                source_spans=spans,
                attributes=raw.get("attributes", {}),
            )
        for raw in payload.get("relations", []):
            graph.relations[raw["id"]] = Relation(
                id=raw["id"],
                source_id=raw["source_id"],
                target_id=raw["target_id"],
                type=raw["type"],
                confidence=float(raw.get("confidence", 1.0)),
                context_id=raw.get("context_id"),
                attributes=raw.get("attributes", {}),
            )
        for raw in payload.get("identity_links", []):
            graph.identity_links.append(IdentityLink(**raw))
        for raw in payload.get("identity_hypotheses", []):
            graph.identity_hypotheses[raw["id"]] = IdentityHypothesis(**raw)
        return graph

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def human_summary(self) -> str:
        lines = [
            f"Graph: {self.name}",
            f"Chunks: {len(self.chunks)}",
            f"Mentions: {len(self.mentions)}",
            f"Referents: {len(self.referents)}",
            f"Contexts: {len(self.contexts)}",
            f"Propositions: {len(self.propositions)}",
            f"Relations: {len(self.relations)}",
            f"Identity links: {len(self.identity_links)}",
            f"Identity hypotheses: {len(self.identity_hypotheses)}",
            "",
            "Contexts:",
        ]
        for context in self.contexts.values():
            holder = f", holder={context.holder}" if context.holder else ""
            parent = f", parent={context.parent_id}" if context.parent_id else ""
            lines.append(f"- {context.id}: {context.kind}{parent}{holder}")
        lines.append("")
        lines.append("Propositions:")
        for proposition in self.propositions.values():
            roles = [
                f"{relation.type}={self.label_for(relation.target_id)}"
                for relation in self.relations.values()
                if relation.source_id == proposition.id
            ]
            lines.append(
                f"- {proposition.id}: {proposition.predicate} [{proposition.context_id}]"
                + (f" ({', '.join(roles)})" if roles else "")
            )
        if self.mentions:
            lines.append("")
            lines.append("Mentions:")
            for mention in sorted(self.mentions.values(), key=lambda item: (item.chunk_id, item.sentence_index, item.order, item.id)):
                accepted = self.accepted_referent_for_mention(mention.id)
                suffix = f" -> {self.label_for(accepted)}" if accepted else ""
                lines.append(f"- {mention.id}: {mention.text} [{mention.mention_type}]{suffix}")
        return "\n".join(lines)

    def label_for(self, node_id: str) -> str:
        if node_id in self.mentions:
            return self.mentions[node_id].text
        if node_id in self.referents:
            return self.referents[node_id].label
        if node_id in self.propositions:
            return self.propositions[node_id].predicate
        if node_id in self.contexts:
            return self.contexts[node_id].kind
        if node_id in self.chunks:
            return self.chunks[node_id].source_name
        return node_id

    @staticmethod
    def _dump_dataclass(item: Any) -> dict[str, Any]:
        return asdict(item)
