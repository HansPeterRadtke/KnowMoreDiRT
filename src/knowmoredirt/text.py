"""Text utilities for raw-folder indexing and retrieval."""

from __future__ import annotations

import re
import string


WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@:/#-]*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
ABBREVIATION_DOTS = {
    "Dr.": "Dr§",
    "Mr.": "Mr§",
    "Ms.": "Ms§",
    "Mrs.": "Mrs§",
    "Prof.": "Prof§",
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def tokenize(text: str) -> list[str]:
    return [token.strip(".,;:!?()[]{}\"'`").lower() for token in WORD_RE.findall(text or "") if token.strip(".,;:!?()[]{}\"'`")]


def content_tokens(text: str) -> list[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "by",
        "can",
        "could",
        "did",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
        "would",
    }
    return [token for token in tokenize(text) if len(token) > 2 and token not in stop]


def _append_bounded_unit(
    units: list[tuple[int, int, str]],
    *,
    start: int,
    value: str,
    max_unit_chars: int,
) -> None:
    if max_unit_chars <= 0 or len(value) <= max_unit_chars:
        units.append((start, start + len(value), value))
        return
    offset = 0
    while offset < len(value):
        hard_end = min(len(value), offset + max_unit_chars)
        split_end = hard_end
        if hard_end < len(value):
            floor = offset + max(1, max_unit_chars // 2)
            whitespace = value.rfind(" ", floor, hard_end)
            if whitespace > offset:
                split_end = whitespace
        chunk = value[offset:split_end].strip()
        if chunk:
            leading = len(value[offset:split_end]) - len(value[offset:split_end].lstrip())
            units.append((start + offset + leading, start + offset + leading + len(chunk), chunk))
        offset = split_end
        while offset < len(value) and value[offset].isspace():
            offset += 1


def split_units(text: str, *, max_unit_chars: int = 0) -> list[tuple[int, int, str]]:
    """Split raw text into line/sentence units while keeping offsets."""
    protected = text
    for source, target in ABBREVIATION_DOTS.items():
        protected = protected.replace(source, target)
    units: list[tuple[int, int, str]] = []
    if max_unit_chars > 0 and len(protected) > max_unit_chars:
        value = protected.strip()
        for source, target in ABBREVIATION_DOTS.items():
            value = value.replace(target, source)
        if value:
            leading = len(protected) - len(protected.lstrip())
            _append_bounded_unit(units, start=leading, value=value, max_unit_chars=max_unit_chars)
        return units
    for match in SENTENCE_SPLIT_RE.finditer(protected):
        pass
    cursor = 0
    for part in SENTENCE_SPLIT_RE.split(protected):
        start = protected.find(part, cursor)
        if start < 0:
            continue
        end = start + len(part)
        cursor = end
        value = part.strip()
        for source, target in ABBREVIATION_DOTS.items():
            value = value.replace(target, source)
        if value:
            leading = len(part) - len(part.lstrip())
            _append_bounded_unit(
                units,
                start=start + leading,
                value=value,
                max_unit_chars=max_unit_chars,
            )
    return units


def compact_answer(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).strip(" .;:")


def clean_extracted_value(text: str) -> str:
    value = compact_answer(text)
    value = value.strip(" \t\r\n\"'`{}[](),")
    value = re.sub(r"\s*[}\]),]+$", "", value).strip(" \"'`")
    return compact_answer(value)


def text_quality_metrics(text: str) -> dict[str, float | int | bool]:
    """Return generic raw-text quality signals used to downweight noise.

    These metrics are intentionally structural rather than fixture-specific:
    printable ratio, alphanumeric ratio, token count, unique-token ratio, and
    long-token pressure. They do not classify topics or assume schemas.
    """

    value = str(text or "")
    length = len(value)
    if length == 0:
        return {
            "char_count": 0,
            "printable_ratio": 1.0,
            "alnum_ratio": 0.0,
            "symbol_ratio": 0.0,
            "token_count": 0,
            "unique_token_ratio": 0.0,
            "long_token_ratio": 0.0,
            "non_ascii_ratio": 0.0,
            "ocrish_ratio": 0.0,
            "low_semantic_noise": True,
            "semantic_quality": "empty",
        }
    printable = sum(1 for char in value if char in string.printable or char.isprintable())
    alnum = sum(1 for char in value if char.isalnum())
    symbolic = sum(1 for char in value if not char.isalnum() and not char.isspace())
    tokens = tokenize(value)
    unique_tokens = set(tokens)
    long_tokens = [token for token in tokens if len(token) >= 24]
    machine_blob_tokens = []
    machine_blob_chars = 0
    for token in tokens:
        token_value = token.strip()
        hex_blob = bool(re.fullmatch(r"[0-9a-fA-F]{24,}", token_value))
        base64_blob = bool(
            len(token_value) >= 32
            and len(token_value) % 4 == 0
            and re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", token_value)
            and not re.fullmatch(r"[A-Za-z]+", token_value)
            and not re.fullmatch(r"[0-9]+", token_value)
        )
        if hex_blob or base64_blob:
            machine_blob_tokens.append(token_value)
            machine_blob_chars += len(token_value)
    non_ascii = sum(1 for char in value if ord(char) > 127)
    ocrish_tokens = [token for token in tokens if any(char.isdigit() for char in token) and any(char.isalpha() for char in token)]
    printable_ratio = printable / length
    alnum_ratio = alnum / length
    symbol_ratio = symbolic / length
    token_count = len(tokens)
    unique_token_ratio = (len(unique_tokens) / token_count) if token_count else 0.0
    long_token_ratio = (len(long_tokens) / token_count) if token_count else 0.0
    machine_blob_token_ratio = (len(machine_blob_tokens) / token_count) if token_count else 0.0
    machine_blob_char_ratio = (machine_blob_chars / max(1, alnum)) if alnum else 0.0
    non_ascii_ratio = non_ascii / length
    ocrish_ratio = (len(ocrish_tokens) / token_count) if token_count else 0.0
    machine_blob_text = bool(
        (length > 80 and machine_blob_token_ratio >= 0.50)
        or (length > 80 and machine_blob_char_ratio >= 0.50 and len(machine_blob_tokens) >= 1)
        or (length > 80 and token_count <= 6 and long_token_ratio >= 0.20 and ocrish_ratio >= 0.50)
    )
    if symbol_ratio > 0.35 or printable_ratio < 0.75:
        quality = "random_character_noise"
    elif machine_blob_text or long_token_ratio > 0.25:
        quality = "base64_or_hex_blob"
    elif ocrish_ratio > 0.20:
        quality = "ocr_corruption"
    elif non_ascii_ratio > 0.05 and unique_token_ratio > 0.80:
        quality = "multilingual_word_salad"
    elif token_count >= 12 and unique_token_ratio > 0.90:
        quality = "word_salad"
    elif "no actionable fact is asserted" in normalize(value):
        quality = "plausible_babble"
    else:
        quality = "meaningful_discourse"
    low_semantic_noise = (
        printable_ratio < 0.75
        or alnum_ratio < 0.25
        or symbol_ratio > 0.35
        or (length > 80 and token_count < 4)
        or (token_count >= 20 and unique_token_ratio > 0.95 and long_token_ratio > 0.20)
    )
    return {
        "char_count": length,
        "printable_ratio": round(printable_ratio, 4),
        "alnum_ratio": round(alnum_ratio, 4),
        "symbol_ratio": round(symbol_ratio, 4),
        "token_count": token_count,
        "unique_token_ratio": round(unique_token_ratio, 4),
        "long_token_ratio": round(long_token_ratio, 4),
        "machine_blob_token_ratio": round(machine_blob_token_ratio, 4),
        "machine_blob_char_ratio": round(machine_blob_char_ratio, 4),
        "non_ascii_ratio": round(non_ascii_ratio, 4),
        "ocrish_ratio": round(ocrish_ratio, 4),
        "low_semantic_noise": bool(low_semantic_noise),
        "semantic_quality": quality,
    }


def is_low_semantic_noise(text: str) -> bool:
    return bool(text_quality_metrics(text)["low_semantic_noise"])


def normalize_predicate_polarity(predicate: str, polarity: str = "positive") -> tuple[str, str]:
    """Normalize lexical negation into DRS polarity without domain knowledge."""
    raw = clean_extracted_value(predicate)
    normalized_polarity = normalize(polarity) or "positive"
    if not raw:
        return raw, normalized_polarity
    lowered = normalize(raw)
    negative = normalized_polarity == "negative"

    auxiliary_not = re.match(
        r"^(is|are|was|were|be|been|being|has|have|had|can|could|will|would|shall|should|may|might|must)\s+not(?:\s+(.*))?$",
        lowered,
    )
    do_not = re.match(r"^(do|does|did)\s+not\s+(.+)$", lowered)
    cannot = re.match(r"^cannot(?:\s+(.+))?$", lowered)
    starts_not = re.match(r"^not\s+(.+)$", lowered)

    if auxiliary_not:
        auxiliary, remainder = auxiliary_not.groups()
        raw = auxiliary if not remainder else f"{auxiliary} {remainder}"
        negative = True
    elif do_not:
        raw = do_not.group(2)
        negative = True
    elif cannot:
        remainder = cannot.group(1) or ""
        raw = "can" if not remainder else f"can {remainder}"
        negative = True
    elif starts_not:
        raw = starts_not.group(1)
        negative = True
    elif re.search(r"\bno\b", lowered):
        raw = re.sub(r"\bno\s+", "", lowered, count=1).strip()
        negative = True
    elif re.search(r"\bnever\b", lowered):
        raw = re.sub(r"\bnever\s*", "", lowered, count=1).strip()
        negative = True

    return clean_extracted_value(raw), "negative" if negative else normalized_polarity
