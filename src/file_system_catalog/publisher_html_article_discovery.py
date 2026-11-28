"""Bounded, read-only search of approved archived HTML in an explicit URL scope.

This optional RAM index supplements native KMD retrieval. It never changes a
source, derivative, persistent index or running worker. Parsed article text is
not a browser rendering, literal HTML byte quotation, or verified answer.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlsplit

from .hybrid_search import query_terms
from .source_review_audit import safe_file

MAX_METADATA_BYTES = 64 * 1024 * 1024
MAX_METADATA_ROWS = 25000
MAX_HTML_BYTES = 8 * 1024 * 1024
MAX_DERIVATIVE_BYTES = 16 * 1024 * 1024
MAX_SCOPE_BYTES = 128 * 1024 * 1024
MAX_SCOPE_RECORDS = 512
MAX_ARTICLE_CHARS = 2 * 1024 * 1024
MAX_SELECTED_SOURCES = 128
SHA = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_COLUMNS = (
    "source_path", "source_sha256", "derived_sha256", "derived_path",
    "original_kind", "original_url", "original_title", "original_bytes",
    "text_characters", "status",
)
SELECT_SOURCE = "SELECT " + ",".join(SOURCE_COLUMNS) + " FROM originals"


def publisher_url(value: str, *, directory: bool = False) -> tuple[str, str]:
    """Conservative HTTPS origin/path identity; unsupported encodings refuse."""
    if (not isinstance(value, str) or not 1 <= len(value) <= 2048
            or not value.isascii() or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value)
            or any(c in value for c in ("\\", "%", "?", "#"))):
        raise ValueError("publisher URL must be unambiguous ASCII HTTPS without query or fragment")
    parts = urlsplit(value)
    host = parts.hostname
    if (parts.scheme != "https" or not host
            or parts.netloc.lower() != host
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", host)
            or ".." in host or "." not in host
            or not parts.path.startswith("/") or "//" in parts.path
            or any(p in (".", "..") for p in parts.path.split("/"))
            or (directory and not parts.path.endswith("/"))):
        raise ValueError("publisher scope requires a plain HTTPS host and directory boundary")
    return host, parts.path


@dataclass(frozen=True)
class PublisherScope:
    url: str
    host: str
    path: str

    @classmethod
    def parse(cls, value: str) -> "PublisherScope":
        host, path = publisher_url(value, directory=True)
        return cls("https://" + host + path, host, path)

    def contains(self, value: str) -> bool:
        host, path = publisher_url(value)
        return host == self.host and path.startswith(self.path)


def _fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_identity(path: Path) -> tuple[int, ...]:
    s = path.stat(follow_symlinks=False)
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def read_bounded(path: Path, limit: int) -> bytes:
    if type(limit) is not int or not 1 <= limit <= MAX_METADATA_BYTES:
        raise ValueError("invalid source byte read bound")
    if path.is_symlink() or not path.is_file():
        raise ValueError("source input is missing or symlinked")
    before = file_identity(path)
    if before[2] > limit:
        raise ValueError("source input exceeds explicit byte bound")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit or before != file_identity(path) or len(raw) != before[2]:
        raise ValueError("source input changed or exceeded its bound while reading")
    return raw


def load_records(path: Path, key: str) -> dict[str, dict]:
    doc = json.loads(read_bounded(path, MAX_METADATA_BYTES))
    rows = doc.get(key) if isinstance(doc, dict) else None
    if not isinstance(rows, list) or len(rows) > MAX_METADATA_ROWS:
        raise ValueError("invalid or oversized source metadata population")
    result = {}
    for row in rows:
        name = row.get("path") if isinstance(row, dict) else None
        if (not isinstance(name, str) or not 1 <= len(name) <= 1600
                or name in result or name.startswith("/") or "\\" in name or "\0" in name
                or any(p in ("", ".", "..") for p in name.split("/"))):
            raise ValueError("invalid or duplicate source metadata path")
        result[name] = row
    return result


def readonly(path: Path) -> sqlite3.Connection:
    if path.is_symlink() or not path.is_file():
        raise ValueError("source database is missing or symlinked")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    return db


class _ArticleParser(HTMLParser):
    """Select a single explicitly closed article, else a single closed main.

    Text is whitespace-normalized. Known non-content/explicitly hidden HTML
    is excluded; external CSS, images, scripts and browser layout are not read.
    """
    VOID = frozenset(("area", "base", "br", "col", "embed", "hr", "img", "input",
                      "link", "meta", "param", "source", "track", "wbr"))
    EXCLUDED = frozenset(("nav", "aside", "footer", "script", "style", "noscript",
                          "template", "svg", "canvas", "button"))
    ROLES = frozenset(("navigation", "search", "complementary", "contentinfo"))

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.containers = []
        self.tags = 0

    def handle_starttag(self, tag, attrs):
        self.tags += 1
        if self.tags > 200000 or len(self.stack) >= 512:
            raise ValueError("HTML structure exceeds parsing bound")
        attrs = {k.lower(): (v or "") for k, v in attrs}
        hidden = ("hidden" in attrs or "inert" in attrs
                  or attrs.get("aria-hidden", "").strip().lower() == "true"
                  or bool(set(attrs.get("role", "").lower().split()) & self.ROLES)
                  or bool(re.search(r"(?:^|;)\s*(?:display\s*:\s*none|visibility\s*:\s*hidden)\s*(?:!important\s*)?(?:;|$)",
                                    attrs.get("style", ""), re.I)))
        excluded = bool(self.stack and self.stack[-1][1]) or tag in self.EXCLUDED or hidden
        container = None
        if tag in ("article", "main"):
            if len(self.containers) >= 16:
                raise ValueError("ambiguous or excessive HTML article containers")
            container = {"tag": tag, "text": [], "chars": 0, "closed": False,
                         "malformed": False, "excluded": excluded}
            self.containers.append(container)
        if tag not in self.VOID:
            self.stack.append((tag, excluded, container))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                for _tag, _excluded, container in self.stack[i + 1:]:
                    if container is not None:
                        container["malformed"] = True
                container = self.stack[i][2]
                if container is not None:
                    container["closed"] = True
                del self.stack[i:]
                return

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)

    def handle_data(self, data):
        if not self.stack or self.stack[-1][1]:
            return
        for _tag, _excluded, container in self.stack:
            if container is not None:
                container["chars"] += len(data)
                if container["chars"] > MAX_ARTICLE_CHARS:
                    raise ValueError("HTML article exceeds character bound")
                container["text"].append(data)

    def result(self):
        articles = [c for c in self.containers if c["tag"] == "article"]
        choices = articles or [c for c in self.containers if c["tag"] == "main"]
        if len(choices) != 1:
            raise ValueError("no unique article or main container")
        c = choices[0]
        if not c["closed"] or c["malformed"] or c["excluded"]:
            raise ValueError("HTML article is unclosed, ambiguous or explicitly hidden")
        text = " ".join(" ".join(c["text"]).split())
        if not 25 <= len(text) <= MAX_ARTICLE_CHARS or "\0" in text:
            raise ValueError("HTML article lacks bounded usable text")
        return c["tag"], text


class _PBRTPretextParser(HTMLParser):
    """Read only actual first-party PBRT book chapter fragments.

    PBRT fourth edition uses div.maincontainer containing many sibling
    div.pretext-layout-root prose/math fragments, NOT <article>/<main>.
    Selecting the entire body would incorrectly mix navigation and footer
    with quoted physics. This strict fallback is admitted only for an
    explicit current pbr-book.org/4ed/Reflection_Models/ publisher URL.
    """
    MAX_FRAGMENTS = 128

    def __init__(self, *, expected_heading:str="h2"):
        if expected_heading not in ("h1","h2"):
            raise ValueError("PBRT source page must specify chapter or section layout")
        super().__init__(convert_charrefs=True)
        self.expected_heading=expected_heading
        self.stack = []
        self.tags = 0
        self.main_count = 0
        self.main_closed = 0
        self.fragments = []
        self.h1_count = 0
        self.h2_count = 0
        self.chars = 0
        self.invalid_structure = False

    def handle_starttag(self, tag, attrs):
        self.tags += 1
        if self.tags > 200000 or len(self.stack) >= 512:
            raise ValueError("PBRT original chapter HTML exceeds bounded parsing work")
        data = {k.lower(): (v or "") for k,v in attrs}
        classes = set(data.get("class", "").split())
        hidden = ("hidden" in data or "inert" in data
                or data.get("aria-hidden","").strip().lower() == "true"
                or bool(set(data.get("role","").lower().split()) & _ArticleParser.ROLES)
                or bool(re.search(r"(?:^|;)\s*(?:display\s*:\s*none|visibility\s*:\s*hidden)\s*(?:!important\s*)?(?:;|$)",
                                  data.get("style",""),re.I)))
        excluded = (bool(self.stack and self.stack[-1]["excluded"])
                    or tag in _ArticleParser.EXCLUDED or hidden)
        in_main = bool(self.stack and self.stack[-1]["in_main"])
        active = next((frame["fragment"] for frame in reversed(self.stack)
                       if frame["fragment"] is not None),None)
        main = tag=="div" and "maincontainer" in classes
        if main:
            self.main_count += 1
            if self.main_count > 1 or excluded or in_main:
                raise ValueError("duplicate, nested or hidden PBRT book maincontainer")
            in_main = True
        fragment = None
        if tag=="div" and "pretext-layout-root" in classes:
            if not in_main or excluded or active is not None:
                raise ValueError("unbounded, hidden or nested PBRT book chapter fragment")
            if len(self.fragments) >= self.MAX_FRAGMENTS:
                raise ValueError("excessive PBRT book source chapter fragments")
            fragment={"text":[],"closed":False}
            self.fragments.append(fragment)
        if tag in ("h1","h2") and in_main and not excluded and (active is not None or fragment is not None):
            if tag=="h1":self.h1_count+=1
            else:self.h2_count+=1
        if tag not in _ArticleParser.VOID:
            self.stack.append({"tag":tag,"excluded":excluded,
                               "in_main":in_main,"is_main":main,"fragment":fragment})

    def handle_endtag(self,tag):
        for i in range(len(self.stack)-1,-1,-1):
            if self.stack[i]["tag"]==tag:
                if (any(x["fragment"] is not None for x in self.stack[i+1:])
                        or any(x["is_main"] for x in self.stack[i+1:])):
                    self.invalid_structure=True
                ending=self.stack[i]
                if ending["fragment"] is not None:ending["fragment"]["closed"]=True
                if ending["is_main"]:self.main_closed += 1
                del self.stack[i:]
                return

    def handle_startendtag(self,tag,attrs):
        self.handle_starttag(tag,attrs)
        if tag not in _ArticleParser.VOID:self.handle_endtag(tag)

    def handle_data(self,data):
        if not self.stack or self.stack[-1]["excluded"]:
            return
        for item in reversed(self.stack):
            frag=item["fragment"]
            if frag is not None:
                self.chars+=len(data)
                if self.chars>MAX_ARTICLE_CHARS:
                    raise ValueError("PBRT book chapter exceeds text bounds")
                frag["text"].append(data)
                break

    def result(self):
        heading_ok=(self.h1_count==1 and self.h2_count==0 if self.expected_heading=="h1"
                    else self.h2_count==1 and self.h1_count==0)
        if (self.main_count!=1 or self.main_closed!=1
                or self.invalid_structure or not heading_ok
                or not 2<=len(self.fragments)<=self.MAX_FRAGMENTS
                or any(not f["closed"] for f in self.fragments)):
            raise ValueError("PBRT original source has untrusted chapter container boundaries")
        parts=[" ".join(" ".join(f["text"]).split()) for f in self.fragments]
        text=" ".join(p for p in parts if p)
        if (not 80<=len(text)<=MAX_ARTICLE_CHARS or "\x00" in text):
            raise ValueError("PBRT original source chapter text empty or unusable")
        return "pbrt4_pretext_layout",text


def extract_article(raw: bytes, *, publisher_original_url: str | None = None) -> tuple[str, str]:
    """Select exact current first-party article, not arbitrary navigation text.

    Only the publisher's explicitly scoped PBRT fourth-edition Reflection
    Models pages may use the alternative many-fragment book layout. Default
    strict article/main behavior remains byte-equivalent for other sources.
    """
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= MAX_HTML_BYTES:
        raise ValueError("invalid or oversized original HTML")
    body=raw.decode("utf8", "strict")
    parser = _ArticleParser()
    parser.feed(body)
    parser.close()
    try:
        return parser.result()
    except ValueError as error:
        if str(error)!="no unique article or main container":
            raise
        if (not isinstance(publisher_original_url,str)
                or not PublisherScope.parse(
                    "https://pbr-book.org/4ed/").contains(publisher_original_url)):
            raise
        _host, page_path=publisher_url(publisher_original_url)
        if not page_path.endswith(".html"):
            raise
        # Original PBRT chapter overview URLs have exactly two segments
        # /4ed/Reflection_Models.html (h1); sections have >=3 segments
        # /4ed/Reflection_Models/Roughness_Using_Microfacet_Theory.html (h2).
        segments=page_path.strip("/").split("/")
        if not 2<=len(segments)<=3 or segments[0]!="4ed":
            raise
        fallback=_PBRTPretextParser(expected_heading=("h1" if len(segments)==2 else "h2"))
        fallback.feed(body)
        fallback.close()
        return fallback.result()


@dataclass(frozen=True)
class ArticleEntry:
    source_map_row: tuple
    catalog_sha256: str
    ledger_sha256: str
    url: str
    selector: str
    text: str

    @property
    def path(self):
        return self.source_map_row[0]

    @property
    def title(self):
        return self.source_map_row[6] or ""


def _metadata_identity(row: tuple, catalog: dict, ledger: dict,
                       scope: PublisherScope) -> tuple[str, str, str]:
    path, original_sha, derived_sha, derived_path, kind, url, title, size, chars, status = row
    c = catalog.get(path)
    receipt = ledger.get(path)
    if not isinstance(c, dict) or not isinstance(receipt, dict):
        raise ValueError("source missing from unique catalog or acquisition ledger")
    urls = [c.get(k) for k in ("source_url", "url") if c.get(k)]
    if not urls or any(publisher_url(u) != publisher_url(url) for u in urls):
        raise ValueError("catalog and source-map publisher URLs disagree")
    if not scope.contains(url):
        raise ValueError("source is outside requested publisher directory")
    if (status != "complete" or kind != "html" or c.get("kind") != "html"
            or derived_path != path + ".txt"
            or not isinstance(original_sha, str) or not SHA.fullmatch(original_sha)
            or not isinstance(derived_sha, str) or not SHA.fullmatch(derived_sha)
            or not isinstance(title, str) or not 1 <= len(title) <= 1000
            or type(size) is not int or size <= 0
            or type(chars) is not int or chars <= 0
            or receipt.get("status") != "complete"
            or receipt.get("sha256") != original_sha
            or type(receipt.get("bytes_received")) is not int
            or receipt["bytes_received"] != size):
        raise ValueError("incomplete or inconsistent approved HTML source identity")
    return _fingerprint(c), _fingerprint(receipt), "https://" + publisher_url(url)[0] + publisher_url(url)[1]


def assert_entry_metadata(entry: ArticleEntry, *, source_map: sqlite3.Connection,
                          catalog: dict, ledger: dict, scope: PublisherScope) -> None:
    if source_map.execute("PRAGMA query_only").fetchone() != (1,):
        raise ValueError("source checks require a query-only database")
    rows = source_map.execute(SELECT_SOURCE + " WHERE source_path=?", (entry.path,)).fetchall()
    if rows != [entry.source_map_row]:
        raise ValueError("source-map revision or publisher metadata changed")
    identity = _metadata_identity(rows[0], catalog, ledger, scope)
    if identity != (entry.catalog_sha256, entry.ledger_sha256, entry.url):
        raise ValueError("catalog, acquisition receipt or source URL changed")


def native_identity(db: sqlite3.Connection, path: str) -> int:
    if db.execute("PRAGMA query_only").fetchone() != (1,):
        raise ValueError("native source checks require a query-only database")
    rows = db.execute("SELECT id FROM filesystem_entries WHERE relative_path_display=? "
                      "AND entry_type='file'", (path,)).fetchall()
    if len(rows) != 1 or type(rows[0][0]) is not int:
        raise ValueError("original has no unique native FileDB identity")
    return rows[0][0]


class ArticleRAMIndex:
    def __init__(self):
        self.db = sqlite3.connect(":memory:")
        self.entries = {}
        self.ids = {}
        for channel, tokenizer in (("literal", "unicode61"), ("porter", "porter unicode61")):
            self.db.execute(f"CREATE VIRTUAL TABLE {channel} USING "
                            f"fts5(title,source_topic,source_body,tokenize='{tokenizer} remove_diacritics 2 tokenchars _')")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.db.close()

    def add(self, entry: ArticleEntry):
        if entry.path in self.ids or len(self.entries) >= MAX_SCOPE_RECORDS:
            raise ValueError("duplicate or excessive scoped article")
        ident = len(self.entries) + 1
        self.entries[ident] = entry
        self.ids[entry.path] = ident
        for channel in ("literal", "porter"):
            self.db.execute(f"INSERT INTO {channel}(rowid,title,source_topic,source_body) VALUES(?,?,?,?)",
                            (ident, entry.title, entry.path.split("/", 1)[0], entry.text))

    @staticmethod
    def clause(query: str) -> str:
        if not isinstance(query, str) or not 3 <= len(query) <= 600:
            raise ValueError("unbounded article query")
        terms = query_terms(query)
        if not terms:
            raise ValueError("query has no informative terms")
        return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)

    def ranked(self, query: str, *, channel: str, limit: int = 50):
        if channel not in ("literal", "porter") or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("invalid article rank channel or limit")
        rows = self.db.execute(f"SELECT rowid,bm25({channel},3.0,0.4,1.0) FROM {channel} "
                               f"WHERE {channel} MATCH ? ORDER BY 2,rowid LIMIT ?",
                               (self.clause(query), limit)).fetchall()
        return [(self.entries[ident], ident, float(score)) for ident, score in rows]

    def excerpt(self, entry: ArticleEntry, query: str, *, channel: str, length: int = 520):
        """Use FTS's actual matched forms, including Porter inflections."""
        if channel not in ("literal", "porter") or type(length) is not int or not 160 <= length <= 2000:
            raise ValueError("invalid article excerpt request")
        opening, closing = "\x1eKMD_START\x1e", "\x1eKMD_END\x1e"
        if opening in entry.text or closing in entry.text:
            raise ValueError("article contains reserved evidence marker")
        row = self.db.execute(f"SELECT highlight({channel},2,?,?) FROM {channel} "
                              f"WHERE rowid=? AND {channel} MATCH ?",
                              (opening, closing, self.ids[entry.path], self.clause(query))).fetchone()
        if row is None:
            raise ValueError("article no longer matches query")
        marked = row[0]
        plain = []
        spans = []
        cursor = total = 0
        while True:
            a = marked.find(opening, cursor)
            if a < 0:
                plain.append(marked[cursor:])
                break
            prefix = marked[cursor:a]
            plain.append(prefix)
            total += len(prefix)
            b = marked.find(closing, a + len(opening))
            if b < 0:
                raise ValueError("invalid FTS article match markers")
            token = marked[a + len(opening):b]
            spans.append((total, total + len(token), token.casefold()))
            plain.append(token)
            total += len(token)
            cursor = b + len(closing)
        if "".join(plain) != entry.text:
            raise ValueError("FTS article evidence differs from original parsed text")
        options = {0}
        for a, _b, _token in spans[:500]:
            for offset in (length // 3, length // 2):
                options.add(max(0, min(len(entry.text) - length, a - offset)))
        start = max(sorted(options), key=lambda p: len({t for a, b, t in spans if p <= a and b <= p + length}))
        end = min(len(entry.text), start + length)
        excerpt = entry.text[start:end]
        return {
            "source_excerpt": excerpt,
            "normalized_article_start_char": start,
            "normalized_article_end_char": end,
            "source_excerpt_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
            "matched_article_forms": sorted({t for a, b, t in spans if start <= a and b <= end})[:100],
        }


def article_context(entry: ArticleEntry, excerpt: dict, *, characters: int) -> dict:
    """Return separate bounded original-text spans; never infer a section.

    The caller's normal source checks attest current file bytes. This helper
    verifies the supplied excerpt against that entry and records exact
    normalized character coverage, including every omitted interval.
    """
    if type(characters) is not int or not 80 <= characters <= 2000:
        raise ValueError("article context requires eighty through two thousand characters per neighbor")
    if not isinstance(entry, ArticleEntry) or not isinstance(entry.text, str):
        raise ValueError("article context requires a current parsed article entry")
    if not isinstance(excerpt, dict):
        raise ValueError("article context requires a verified excerpt mapping")
    start = excerpt.get("normalized_article_start_char")
    end = excerpt.get("normalized_article_end_char")
    text = excerpt.get("source_excerpt")
    digest = excerpt.get("source_excerpt_sha256")
    if (type(start) is not int or type(end) is not int
            or not 0 <= start < end <= len(entry.text)
            or not isinstance(text, str) or text != entry.text[start:end]
            or not isinstance(digest, str) or not SHA.fullmatch(digest)
            or hashlib.sha256(text.encode()).hexdigest() != digest):
        raise ValueError("article context excerpt does not match its normalized original coordinates and digest")

    before = max(0, start - characters)
    after = min(len(entry.text), end + characters)
    ranges = (
        ("article_opening", 0, min(characters, before)),
        ("preceding", before, start),
        ("following", end, after),
    )
    spans = {}
    covered = [(start, end)]
    for role, left, right in ranges:
        if left == right:
            continue
        value = entry.text[left:right]
        spans[role] = {
            "start_char": left, "end_char": right, "text": value,
            "sha256": hashlib.sha256(value.encode()).hexdigest(),
        }
        covered.append((left, right))
    omitted = []
    cursor = 0
    for left, right in sorted(covered):
        if left > cursor:
            omitted.append({"start_char": cursor, "end_char": left})
        cursor = right
    if cursor < len(entry.text):
        omitted.append({"start_char": cursor, "end_char": len(entry.text)})
    return {
        "basis": "separate_bounded_whitespace_normalized_original_article_spans",
        "requested_characters_per_neighbor": characters,
        "original_path": entry.path,
        "publisher_original_sha256": entry.source_map_row[1],
        "normalized_article_sha256": hashlib.sha256(entry.text.encode()).hexdigest(),
        "article_selector": entry.selector,
        "spans": spans,
        "omitted_normalized_article_ranges": omitted,
        "covered_normalized_article_characters": sum(right - left for left, right in covered),
        "entire_selected_container_covered": not omitted,
        "governing_section_identified": False,
        "factual_answer_verified": False,
    }


def load_scoped_articles(*, source_map: sqlite3.Connection, original_root: Path,
                         catalog: dict, ledger: dict, index: ArticleRAMIndex,
                         scope: PublisherScope, max_records: int = MAX_SCOPE_RECORDS,
                         max_total_bytes: int = MAX_SCOPE_BYTES) -> dict:
    if (type(max_records) is not int or not 1 <= max_records <= MAX_SCOPE_RECORDS
            or type(max_total_bytes) is not int or not 1 <= max_total_bytes <= MAX_SCOPE_BYTES
            or source_map.execute("PRAGMA query_only").fetchone() != (1,)):
        raise ValueError("invalid bounded scoped source scan")
    rows = source_map.execute(SELECT_SOURCE + " ORDER BY source_path LIMIT ?",
                              (MAX_METADATA_ROWS + 1,)).fetchall()
    if len(rows) > MAX_METADATA_ROWS:
        raise ValueError("source map exceeds metadata cardinality limit")
    by_path = {}
    for row in rows:
        if row[0] in by_path:
            raise ValueError("duplicate source-map path")
        by_path[row[0]] = row
    scoped = []
    for path, c in catalog.items():
        urls = [c.get(k) for k in ("source_url", "url") if c.get(k)]
        matches = False
        for url in urls:
            try:
                matches = matches or scope.contains(url)
            except ValueError:
                continue
        if matches:
            scoped.append((path, c))
    html = [(p, c) for p, c in scoped if c.get("kind") == "html"]
    if len(html) > max_records:
        raise ValueError("publisher scope exceeds explicit HTML cardinality limit; choose a narrower directory")
    skipped = []
    total = 0
    for path, c in sorted(html):
        try:
            row = by_path.get(path)
            if row is None:
                raise ValueError("catalog source has no source-map row")
            catalog_sha, ledger_sha, url = _metadata_identity(row, catalog, ledger, scope)
            if row[7] > MAX_HTML_BYTES:
                raise ValueError("original HTML exceeds per-file byte bound")
            if total + row[7] > max_total_bytes:
                raise RuntimeError("publisher scope exceeds total byte budget; choose a narrower directory")
            original = safe_file(original_root, path)
            # Metadata is not an I/O budget authority. Refuse a size mismatch
            # before reading, and reserve the bounded read even if it fails.
            if original.stat().st_size != row[7]:
                raise ValueError("original HTML size differs from approved revision")
            remaining = max_total_bytes - total
            total += row[7]
            raw = read_bounded(original, min(MAX_HTML_BYTES, row[7], remaining))
            if len(raw) != row[7] or hashlib.sha256(raw).hexdigest() != row[1]:
                raise ValueError("original HTML bytes differ from approved revision")
            selector, body = extract_article(raw,publisher_original_url=url)
            index.add(ArticleEntry(tuple(row), catalog_sha, ledger_sha, url, selector, body))
        except (ValueError, OSError, UnicodeError) as error:
            skipped.append({"original_path": path, "reason": str(error)[:200]})
    index.db.commit()
    return {
        "publisher_url_prefix": scope.url,
        "catalog_records_matching_scope": len(scoped),
        "eligible_html_records": len(html),
        "non_html_records_excluded": len(scoped) - len(html),
        "verified_original_articles_indexed": len(index.entries),
        "original_html_byte_budget_used": total,
        "max_html_bytes": MAX_HTML_BYTES,
        "max_scope_bytes": max_total_bytes,
        "max_scope_records": max_records,
        "skipped_html_records": len(skipped),
        "skipped_sources": skipped[:150],
        "skip_details_truncated": len(skipped) > 150,
        "all_scoped_html_articles_available": len(html) == len(index.entries),
        "scope_status": ("no_archived_sources_in_requested_scope" if not scoped else
                         "no_usable_archived_html_articles" if not index.entries else
                         "partial_html_coverage" if skipped else "scoped_html_ready"),
        "out_of_scope_fallback_allowed": False,
    }


def verify_entry(entry: ArticleEntry, *, source_map: sqlite3.Connection,
                 native_db: sqlite3.Connection, catalog: dict, ledger: dict,
                 original_root: Path, derived_root: Path, scope: PublisherScope) -> dict:
    assert_entry_metadata(entry, source_map=source_map, catalog=catalog, ledger=ledger, scope=scope)
    original = safe_file(original_root, entry.path)
    derived = safe_file(derived_root, entry.source_map_row[3])
    raw = read_bounded(original, MAX_HTML_BYTES)
    if len(raw) != entry.source_map_row[7] or hashlib.sha256(raw).hexdigest() != entry.source_map_row[1]:
        raise ValueError("publisher original changed after article indexing")
    if extract_article(raw,publisher_original_url=entry.url) != (entry.selector, entry.text):
        raise ValueError("parsed publisher article changed after indexing")
    derivative = read_bounded(derived, MAX_DERIVATIVE_BYTES)
    if hashlib.sha256(derivative).hexdigest() != entry.source_map_row[2]:
        raise ValueError("current derivative bytes differ from source-map revision")
    return {
        "original_path": entry.path,
        "original_title": entry.title,
        "source_url": entry.url,
        "publisher_url_prefix": scope.url,
        "native_original_filedb_entry_id": native_identity(native_db, entry.path),
        "publisher_original_sha256": entry.source_map_row[1],
        "current_derivative_sha256": entry.source_map_row[2],
        "catalog_record_sha256": entry.catalog_sha256,
        "acquisition_receipt_sha256": entry.ledger_sha256,
        "original_file_identity": file_identity(original),
        "derivative_file_identity": file_identity(derived),
        "original_and_derivative_full_bytes_rehashed": True,
        "article_selector": entry.selector,
        "normalized_article_characters": len(entry.text),
        "normalized_article_sha256": hashlib.sha256(entry.text.encode()).hexdigest(),
        "source_excerpt_coordinate_system": "whitespace_normalized_selected_original_html_container",
        "quoted_text_is_original_publisher_bytes": False,
        "browser_visibility_or_rendering_verified": False,
        "factual_answer_verified": False,
    }


def final_metadata_fence(entries: dict[str, ArticleEntry], proofs: dict[str, dict], *,
                         source_map: sqlite3.Connection, native_db: sqlite3.Connection,
                         catalog: dict, ledger: dict, scope: PublisherScope,
                         original_root: Path, derived_root: Path) -> None:
    """Fresh metadata/native/file-identity check after all selected byte reads.

    Cooperating stores are checked, not atomically pinned. A subsequent mutation
    remains possible until immutable cross-store generations are deployed.
    """
    if set(entries) != set(proofs) or len(entries) > MAX_SELECTED_SOURCES:
        raise ValueError("unbounded or incomplete final article evidence set")
    for path, entry in entries.items():
        assert_entry_metadata(entry, source_map=source_map, catalog=catalog, ledger=ledger, scope=scope)
        p = proofs[path]
        if (native_identity(native_db, path) != p["native_original_filedb_entry_id"]
                or file_identity(safe_file(original_root, path)) != p["original_file_identity"]
                or file_identity(safe_file(derived_root, entry.source_map_row[3])) != p["derivative_file_identity"]):
            raise ValueError("native identity or source file changed before evidence emission")
