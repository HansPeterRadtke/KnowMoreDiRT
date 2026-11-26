"""Bounded, read-only evidence candidates for risky generated summary claims.

Generated summaries are retrieval hypotheses, NOT publisher facts. This module
locates source text for quantitative claims and possible prediction-to-result
drift, with verifiable original character offsets and hashes. It does not
infer scientific correctness or approve/rewrite native summary records.
"""
from __future__ import annotations

import hashlib
from itertools import islice
import re
from typing import Any

_NUMBER = re.compile(r"(?<![\w.])(?P<number>\d{1,3}(?:[.,]\d{1,3})?)\s*%", re.I)
_GPU = re.compile(r"\bgpus?\b", re.I)
_EFFECT = re.compile(r"\b(?:boosts?|improves?|increases?|accelerates?|enhances?|outperforms?)\b", re.I)
_MODAL = re.compile(r"\b(?:would|could|might|may|potential(?:ly)?|expected|predict(?:s|ed)?|"
                    r"propos(?:e|es|ed)|future|hypothes(?:is|ized))\b", re.I)
_QUALIFIERS = re.compile(
    r"\b(?:often|sometimes|estimated|estimate|up to|at most|conditional|"
    r"model[- ]scored|based on|compared (?:to|with)|may|might|could|would|"
    r"predict(?:s|ed)?|potential(?:ly)?|upper[- ]bound)\b", re.I,
)
_WORDS = re.compile(r"[a-z]{3,}", re.I)
_STOP = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "they", "their",
    "are", "was", "were", "into", "have", "has", "its", "but", "while",
    "across", "than", "can", "may", "would", "could", "often", "might",
    "under", "over", "use", "uses", "using", "shows", "show", "which",
    "some", "more", "about", "these", "those", "being", "been",
})


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sentences(summary: str):
    """Split only punctuation followed by whitespace and a capital letter.

    Decimal metrics such as 25.3% are kept intact. All retained offsets refer
    to the unmodified input string, not a normalized surrogate.
    """
    start = 0
    for marker in re.finditer(r"(?<=[.!?])\s+(?=[A-Z])", summary):
        end = marker.start()
        if summary[start:end].strip():
            yield start, end, summary[start:end]
        start = marker.end()
    if summary[start:].strip():
        yield start, len(summary), summary[start:]


def _tokens(text: str) -> set[str]:
    return set(w.casefold() for w in _WORDS.findall(text) if w.casefold() not in _STOP)


def _page_label(source: str, at: int) -> str | None:
    left = source.rfind("[[SOURCE UNIT:", 0, at + 1)
    if left == -1:
        return None
    close = source.find("]]", left, min(len(source), left + 400))
    return source[left + len("[[SOURCE UNIT:"):close].strip() if close != -1 else None


def _evidence(source: str, sentence: str, locations: list[int], *,
              context_chars: int, maximum: int, region_start: int, region_end: int) -> list[dict[str, Any]]:
    ranking = []
    claim_words = _tokens(sentence)
    for position in locations[:5000]:
        left = max(region_start, position - context_chars // 2)
        right = min(region_end, position + context_chars // 2)
        excerpt = source[left:right]
        shared = len(claim_words & _tokens(excerpt))
        # A matching conditional source is especially useful when assessing
        # an unqualified performance claim, but is NOT proof of a mismatch.
        conditional = bool(_MODAL.search(excerpt))
        ranking.append((-(shared * 3 + int(conditional)), position, left, right, excerpt))
    ranking.sort(key=lambda x: (x[0], x[2], x[3]))
    kept = []
    used = set()
    for _score, position, left, right, excerpt in ranking:
        if (left, right) in used:
            continue
        used.add((left, right))
        # A wide evidence view may contain other claims. Keep qualifier cues
        # close to the *matched* anchor; leave the full view available separately.
        qualifier_matches = [match for match in _QUALIFIERS.finditer(excerpt)
                             if abs(left + match.start() - position) <= 240]
        # This is a lexical *review cue* only. Its position and hash make it
        # possible to inspect the relevant original words independently.
        qualifiers = [
            {
                "source_start_char": left + match.start(),
                "source_end_char": left + match.end(),
                "original_phrase": match.group(0),
                "original_phrase_sha256": _sha(match.group(0)),
                "phrase_occurs_in_summary_sentence": bool(re.search(
                    r"(?<!\w)" + re.escape(match.group(0)) + r"(?!\w)",
                    sentence, re.I,
                )),
                "distance_from_matched_evidence_anchor_chars": abs(
                    left + match.start() - position
                ),
                "semantic_relevance_to_claim": "NOT_ASSESSED",
            }
            for match in qualifier_matches
        ]
        qualifiers.sort(key=lambda x: (
            x["distance_from_matched_evidence_anchor_chars"],
            x["source_start_char"],
        ))
        kept.append({
            "start_char": left, "end_char": right,
            "source_excerpt": excerpt, "excerpt_sha256": _sha(excerpt),
            "source_unit_label": _page_label(source, position),
            "matched_anchor_char": position,
            "source_qualifiers_observed": sorted({
                match.group(0).casefold() for match in qualifier_matches
            }),
            "possible_missing_qualification_review_cues": [
                item for item in qualifiers if not item["phrase_occurs_in_summary_sentence"]
            ][:12],
            "qualification_cues_do_not_prove_a_claim_incorrect": True,
            "qualifier_cue_distance_limit_chars": 240,
            "text_is_original_extracted_bytes": True,
            "semantic_support_verdict": "NOT_ASSESSED",
        })
        if len(kept) >= maximum:
            break
    return kept


def scan_summary_claims(
    *, summary: str, source: str, maximum_claims: int = 24,
    context_chars: int = 2000, maximum_evidence: int = 3,
    source_start_char: int = 0, source_end_char: int | None = None,
) -> dict[str, Any]:
    """Screen numerical claims and unqualified GPU performance statements.

    A review candidate is NOT an error verdict. In particular, the source
    passage may describe a different experiment, and facts printed solely
    inside figures may not be present in the extracted text layer.
    """
    if (not isinstance(summary, str) or not summary.strip() or len(summary) > 200_000
            or not isinstance(source, str) or not source.strip()
            or len(source.encode("utf-8")) > 64 * 1024 * 1024):
        raise ValueError("invalid or unbounded summary/source evidence")
    if (type(maximum_claims) is not int or not 1 <= maximum_claims <= 40
            or type(context_chars) is not int or not 160 <= context_chars <= 2000
            or type(maximum_evidence) is not int or not 1 <= maximum_evidence <= 6):
        raise ValueError("invalid bounded claim/evidence request")
    if source_end_char is None:
        source_end_char = len(source)
    if (type(source_start_char) is not int or type(source_end_char) is not int
            or not 0 <= source_start_char < source_end_char <= len(source)):
        raise ValueError("invalid verified original source interval")
    # Search only the exact source interval that generated this node, never
    # coincidentally matching a number in a different chapter or child.
    scoped_source = source[source_start_char:source_end_char]
    claims: list[dict[str, Any]] = []
    seen = set()
    for start, end, sentence in _sentences(summary):
        for percent in _NUMBER.finditer(sentence):
            value = percent.group("number").replace(",", ".")
            token = re.compile(r"(?<![\w.])" + re.escape(value) + r"\s*%", re.I)
            positions = [source_start_char + m.start() for m in islice(token.finditer(scoped_source), 5000)]
            key = (start, "numeric", value)
            if key in seen:
                continue
            seen.add(key)
            evidence = _evidence(
                source, sentence, positions, context_chars=context_chars,
                maximum=maximum_evidence, region_start=source_start_char,
                region_end=source_end_char,
            )
            claims.append({
                "review_kind": "quantified_summary_claim",
                "claim_start_char": start, "claim_end_char": end,
                "claim_text": sentence.strip(),
                "claim_sha256": _sha(sentence),
                "matched_value": value + "%",
                "evidence_status": ("exact_numeric_text_occurs_in_extracted_source"
                                    if evidence else "not_found_in_extracted_text_check_original_figures"),
                "source_evidence": evidence,
                "review_prompt": ("Check task/metric labels, sample, comparison,"
                                  " reporting attribution and conditions; matching a"
                                  " number alone does not establish support."),
                "automatic_truth_verdict": "NONE",
            })
        if _GPU.search(sentence) and _EFFECT.search(sentence) and not _MODAL.search(sentence):
            locations = [source_start_char + match.start() for match in islice(_GPU.finditer(scoped_source), 5000)]
            # Only potential source conditional evidence warrants this narrow
            # flag; observed GPU usage elsewhere is not a benchmark by itself.
            candidates = _evidence(
                source, sentence, locations, context_chars=context_chars,
                maximum=max(12, maximum_evidence),
                region_start=source_start_char, region_end=source_end_char,
            )
            conditional = [e for e in candidates if
                           _MODAL.search(e["source_excerpt"])
                           and _EFFECT.search(e["source_excerpt"])]
            if conditional:
                claims.append({
                    "review_kind": "possible_forecast_reported_as_result",
                    "claim_start_char": start, "claim_end_char": end,
                    "claim_text": sentence.strip(), "claim_sha256": _sha(sentence),
                    "evidence_status": "conditional_language_in_candidate_source_passage",
                    "source_evidence": conditional[:maximum_evidence],
                    "review_prompt": ("Verify whether the original actually benchmarks"
                                      " this performance claim or only predicts a benefit."
                                      " No error is established by lexical matching alone."),
                    "automatic_truth_verdict": "NONE",
                })
        if len(claims) >= maximum_claims:
            break
    return {
        "schema": "kmd.read_only_summary_claim_evidence_candidates.v1",
        "summary_sha256": _sha(summary), "extracted_source_sha256": _sha(source),
        "summary_characters": len(summary), "source_characters": len(source),
        "reviewed_source_start_char": source_start_char,
        "reviewed_source_end_char": source_end_char,
        "claims": claims[:maximum_claims],
        "review_candidates": len(claims[:maximum_claims]),
        "source_evidence_is_an_exact_text_locator_not_semantic_support": True,
        "generated_summary_is_not_an_original_source": True,
        "automatic_factual_accuracy_or_fidelity_verdict": "NOT_PROVIDED",
        "figure_and_table_visual_verification": "NOT_PERFORMED",
        "does_not_change_native_summary_or_corpus": True,
    }
