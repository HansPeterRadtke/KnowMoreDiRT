"""Bounded read-only original-source passage locator, not answer verification.

This is deliberately a second stage *after* document discovery or an
explicit original-file choice. It searches the verified current extracted
source through a temporary RAM-only FTS5 index and returns traceable
source spans/page labels, not semantic answer confidence or invented quotes.

No new persistent vector/metadata DB, no source file modification.
"""
from __future__ import annotations

from bisect import bisect_left
import hashlib
import re

from .publisher_excerpt import attach_publisher_excerpts
from .hybrid_search import BoundedPassageIndex, lexical_token_spans, query_terms
from .hierarchical_summary import Section, split_sections

_UNIT_PREFIX = re.compile(r"\A\[\[SOURCE UNIT: [^\]\n]{1,400}\]\][ \t]*(?:\r?\n|$)")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _garbled_pdf_excerpt(text: str) -> bool:
    """Detect text-layer encoding failures, not a semantic quality judgment.

    Some publisher PDFs expose glyph codes as NUL-interleaved strings. Such
    spans can exactly match the PDF bytes yet are unusable as quoted evidence.
    Do not repair/normalize the source text or invent replacement glyphs.
    """
    nul = text.count("\x00")
    return nul >= 12 and nul * 50 >= max(1, len(text))


_SOURCE_UNIT = re.compile(r"(?m)^\[\[SOURCE UNIT: ([^\]\r\n]{1,400})\]\][ \t]*(?:\r?\n|$)")
_FRONT_OPEN = re.compile(r"---[ \t]*\r?\n")
_FRONT_END = re.compile(r"(?m)^(?:---|\.\.\.)[ \t]*(?:\r?\n|$)")
_FRONT_FIELD = re.compile(r"(?m)^[A-Za-z_][\w-]*[ \t]*:")
_ATX = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+)")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def _markdown_sections(text: str, start: int):
    """Keep publisher headings and character offsets; ignore heading syntax inside code or YAML."""
    initial = start
    front = _FRONT_OPEN.match(text, start)
    if front:
        end = _FRONT_END.search(text, front.end())
        if end and _FRONT_FIELD.search(text, front.end(), end.start()):
            initial = end.end()
    label = "Original plain text"
    opened = ""
    headings = []
    for line in re.compile(r"(?m)^.*$").finditer(text, initial):
        value = line.group().rstrip("\r")
        fence = _FENCE.fullmatch(value)
        if opened:
            if fence and fence[1].startswith(opened) and not fence[2].strip():
                opened = ""
            continue
        if fence and (fence[1][0] == "~" or "`" not in fence[2]):
            opened = fence[1]
            continue
        heading = _ATX.match(value)
        if not heading:
            continue
        title = re.sub(r"[ \t]+#+[ \t]*$", "", heading[2]).strip()
        if not title:
            continue
        if start < line.start():
            yield Section(start, line.start(), label, text[start:line.start()])
        level = len(heading[1])
        while headings and headings[-1][0] >= level:
            headings.pop()
        headings.append((level, title))
        # Name the current heading even when its full ancestor path is long.
        label = " / ".join(name[:110] for _, name in headings[-2:])[:230]
        start = line.start()
    if start < len(text):
        yield Section(start, len(text), label, text[start:])


def _publisher_sections(text: str, markdown: bool):
    generated_header = text.startswith("KMD source provenance ")
    first = _SOURCE_UNIT.search(text) if generated_header or text.startswith("[[SOURCE UNIT:") else None
    if generated_header and first is None:
        # A damaged wrapper gives us no boundary between catalog metadata and the publisher.
        return
    if markdown and (first is None or first[1] == "Original plain text"):
        yield from _markdown_sections(text, first.end() if first else 0)
    else:
        for section in split_sections(text):
            if first is None or section.start_char >= first.start():
                yield section


def _best_excerpt(
    passage: str, terms: tuple[str, ...], *,
    length: int,
) -> tuple[int, int, tuple[str, ...]]:
    """Prefer dense exact query-term coverage within a bounded source span."""
    if not 160 <= length <= 2000:
        raise ValueError("invalid excerpt length")
    positions = {term: [] for term in terms}
    for term, start, end in lexical_token_spans(passage):
        if term in positions:
            positions[term].append((start, end))
    if len(passage) <= length:
        return 0, len(passage), tuple(sorted(term for term in terms if positions[term]))

    options = {0}
    for occurrences in positions.values():
        for start, _ in occurrences[:50]:
            for offset in (length // 3, length // 2):
                options.add(max(0, min(len(passage) - length, start - offset)))

    best = (0, length, ())
    best_score = (-1, -1)
    for start in sorted(options):
        end = start + length
        covered = []
        for term, occurrences in positions.items():
            i = bisect_left(occurrences, (start,))
            if i < len(occurrences) and occurrences[i][1] <= end:
                covered.append(term)
        # Exact identifiers take precedence over surrounding generic question words.
        score = (sum("_" in term for term in covered), len(covered))
        if score > best_score:
            best_score = score
            best = (start, end, tuple(sorted(covered)))
    return best


def _regular_word_forms(terms):
    forms = {}
    for word in terms:
        if len(word) < 4 or len(word) > 46 or not word.isascii() or not word.isalpha():
            continue
        if word.endswith("s"):
            if len(word) > 4 and not word.endswith(("ss", "us", "is", "ies")):
                forms[word] = word[:-1]
        elif not word.endswith(("x", "z", "ch", "sh", "y")):
            forms[word] = word + "s"
    return {word: form for word, form in forms.items() if form not in terms}


def _passage_matches(index, query, terms, *, allow_forms, limit):
    for row in index.search(query, limit=limit, require_identifier_in_passage=True):
        yield row, {}
    if allow_forms:
        forms = _regular_word_forms(terms)
        alternative = " ".join(forms.get(term, term) for term in terms)
        if forms and len(alternative) <= 1000:
            for row in index.search(alternative, limit=limit):
                yield row, forms


def source_evidence_candidates(
    *,
    original_path: str,
    original_title: str,
    filesystem_entry_id: int,
    original_sha256: str,
    derived_sha256: str,
    derived_text: str,
    query: str,
    top: int = 5,
    snippet_characters: int = 900,
    max_source_characters: int = 20_000_000,
    max_passages: int = 30_000,
    publisher_text_bytes: bytes | None = None,
) -> dict:
    """Return exact original-source lexical candidates with stable char spans.

    Preconditions:
      caller has already checked the publisher original file SHA and byte
      size against the accepted source ledger and checked this derivative
      against source_map for the exact original path.

    The derivative text itself is *also* hashed here so direct callers
    cannot accidentally bypass the revision precondition.
    """
    if not 1<=top<=50:
        raise ValueError("invalid bounded requested source passage count")
    if not 160<=snippet_characters<=2000:
        raise ValueError("invalid source snippet size")
    if not 1<=max_source_characters<=100_000_000 or not 1<=max_passages<=100_000:
        raise ValueError("unbounded source evidence indexing")
    if not isinstance(derived_text,str) or not derived_text.strip():
        raise ValueError("missing extracted original source text")
    if len(derived_text)>max_source_characters:
        raise ValueError("source exceeds explicit RAM-only FTS bound; no silent sampling")
    if _sha(derived_text)!=derived_sha256:
        raise ValueError("source text digest differs from required derivative revision")
    terms=query_terms(query)
    if not terms:
        raise ValueError("no informative source search terms")
    identifiers = {term for term in terms if "_" in term}
    markdown = original_path.casefold().endswith((".md", ".markdown"))
    sections = _publisher_sections(derived_text, markdown)
    source_units = 0
    offsets={}
    chunks=0
    with BoundedPassageIndex(
        max_passages=max_passages,max_characters=max_source_characters
    ) as idx:
        for section in sections:
            source_units += 1
            # >65k-sized sections are divided without changing source
            # character indices; the model's source unit labels stay intact.
            part=0
            for rel in range(0,len(section.text),60_000):
                text=section.text[rel:rel+60_000]
                if not text.strip():
                    continue
                part+=1
                label=section.label if len(section.text)<=60_000 else (
                    f"{section.label} [part {part}]"
                )
                start=section.start_char+rel
                end=start+len(text)
                digest=_sha(text)
                key=(label,digest)
                if key in offsets:
                    # Identical source units can occur; preserve distinct
                    # offset identities in the label instead of mixing them.
                    label=f"{label} [source offset {start}]"
                    key=(label,digest)
                offsets[key]=(start,end)
                heading = _ATX.match(text.split("\n", 1)[0]) if markdown and rel == 0 else None
                idx.add_passage(
                    filesystem_entry_id=filesystem_entry_id,
                    original_path=original_path,
                    original_title=original_title,
                    source_section=label,
                    passage=text,
                    publisher_heading=heading[2][:300] if heading else "",
                    original_sha256=original_sha256,
                    derived_sha256=derived_sha256,
                )
                chunks+=1
        # Oversample a small bounded group then rank by literal evidence
        # found in the source *body*. A matched document title alone is
        # not a supporting source passage.
        out = []
        accepted = set()
        unreadable_pdf_excerpts_dropped = 0
        candidates = _passage_matches(idx, query, terms, allow_forms=markdown and not identifiers,
                                      limit=min(2000, max(100, top * 12)))
        for item, forms in candidates:
            passage=item["passage"]
            marker = _UNIT_PREFIX.match(passage)
            body_start = marker.end() if marker else 0
            body = passage[body_start:]
            start,end=offsets[(item["source_section"],item["passage_sha256"])]
            if (start, end) in accepted:
                continue
            words = {term for term, _, _ in lexical_token_spans(body)}
            matches = set(terms).intersection(words)
            matched_forms = {q: form for q, form in forms.items() if form in words and q not in matches}
            if forms and not matched_forms:
                continue
            if identifiers and identifiers.isdisjoint(matches):
                continue
            has_specific_match = identifiers or item["matched_publisher_heading_terms"]
            if not has_specific_match and len(matches) + len(matched_forms) < min(2, len(terms)):
                continue
            small_start,small_end,covered=_best_excerpt(
                body, terms + tuple(matched_forms.values()), length=snippet_characters
            )
            small_start += body_start
            small_end += body_start
            excerpt=passage[small_start:small_end]
            identifier_words = set()
            if matched_forms:
                matched_forms = {q: form for q, form in matched_forms.items() if form in covered}
                if not matched_forms:
                    continue
                # A morphological fallback for a multi-term question needs a second
                # concept in the actual excerpt. Filenames can spell it inside an identifier.
                identifier_words = set(terms).intersection(
                    word for token, _, _ in lexical_token_spans(excerpt) if "_" in token
                    for word in token.split("_")
                )
                exact_concepts = set(terms).intersection(covered) | identifier_words
                if len(terms) > 1 and not exact_concepts.difference(matched_forms):
                    continue
            # Physical PDF verification establishes location, not readability.
            # Reject heavily NUL-interleaved font extraction from evidence, but
            # keep the original and derivative intact for independent inspection.
            if original_path.casefold().endswith(".pdf") and _garbled_pdf_excerpt(excerpt):
                unreadable_pdf_excerpts_dropped += 1
                continue
            accepted.add((start, end))
            out.append({
                "original_path":original_path,
                "filesystem_entry_id":filesystem_entry_id,
                "original_sha256":original_sha256,
                "derived_sha256":derived_sha256,
                "source_section":item["source_section"],
                "source_start_char":start,
                "source_end_char":end,
                "source_span_sha256":item["passage_sha256"],
                "excerpt_start_char":start+small_start,
                "excerpt_end_char":start+small_end,
                "excerpt_sha256":_sha(excerpt),
                "source_excerpt":excerpt,
                "lexical_bm25":item["lexical_bm25"],
                "matched_source_terms":sorted(matches),
                **({"matched_word_forms": matched_forms} if matched_forms else {}),
                **({"matched_identifier_words": sorted(identifier_words)} if identifier_words else {}),
                **({"matched_publisher_heading_terms": item["matched_publisher_heading_terms"]}
                   if item["matched_publisher_heading_terms"] else {}),
                "matched_excerpt_terms":list(covered),
                "total_informative_query_terms":len(terms),
                "status":"source_lexical_candidate_not_verified_answer",
            })
            if len(out)>=top:
                break
    if out and publisher_text_bytes is not None:
        first = _SOURCE_UNIT.search(derived_text)
        if first and first[1] == "Original plain text":
            attach_publisher_excerpts(publisher_text_bytes, original_sha256, derived_text, out,
                                      body_start=first.end())
    return {
        "source_coordinates": "zero_based_half_open_character_offsets_in_verified_derivative",
        "original_path":original_path,
        "original_sha256":original_sha256,
        "derived_sha256":derived_sha256,
        "source_characters":len(derived_text),
        "source_units":source_units,
        "ephemeral_fts_passages":chunks,
        "returned_passages":len(out),
        "unreadable_pdf_excerpts_dropped":unreadable_pdf_excerpts_dropped,
        "passages":out,
        "evidence_warning":(
            "These are literal matches in the verified original source, "
            "not proof the passage answers the question. Source scope, "
            "negation, date and contradictions require separate validation."
            + (" Heavily NUL-interleaved PDF text-layer excerpts were excluded; "
               "consult the publisher file for their content."
               if unreadable_pdf_excerpts_dropped else "")
        ),
    }
