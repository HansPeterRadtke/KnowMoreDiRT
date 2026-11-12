"""Read explicit CCS subject labels from a PDF's first extracted publisher page."""
import re


CCS_EXTRACTOR = "publisher-ccs-v1"
# Four UTF-8 bytes per character suffice for this prefix, including the page marker.
PREFIX_CHARS = 65_536
_UNIT = re.compile(r"(?m)^\[\[SOURCE UNIT: ([^\]\r\n]{1,400})\]\][ \t]*\r?$")
_HEADING = re.compile(r"(?m)^[ \t]*CCS Concepts[ \t]*\r?\n")
# A complete single-column entry. Decline wrapped or multi-column text.
_ENTRY = re.compile(
    r"[ \t]*•[ \t]+[\w][\w &/(),’'-]{0,119}?[ \t]+→[ \t]+"
    r"(?P<term>[\w][\w &/(),’'-]{0,119}?)[ \t]*;[ \t]*\r?\n"
)


def first_page_ccs(text: str) -> dict[str, dict]:
    """Map each printed subject to its exact half-open derivative character interval."""
    stop = min(len(text), PREFIX_CHARS)
    first = _UNIT.search(text, 0, stop)
    if first is None or first[1] != "PDF page 1":
        return {}
    second = _UNIT.search(text, first.end(), stop)
    if second:
        stop = second.start()
    heading = _HEADING.search(text, first.end(), stop)
    if heading is None or _HEADING.search(text, heading.end(), stop):
        return {}
    line = heading.end()
    while line < stop and text[line] in " \t\r\n":
        line += 1
    subjects = {}
    while entry := _ENTRY.match(text, line, stop):
        term = entry["term"]
        if "  " in term:
            break
        subjects.setdefault(term, {"scheme": "CCS Concepts", "source_section": "PDF page 1",
                                   "start_char": entry.start("term"), "end_char": entry.end("term")})
        line = entry.end()
    return subjects
