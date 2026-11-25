"""Exact simple-table lookup from SHA-attested publisher HTML source.

Only explicitly closed rectangular tables in parser-recognized main/article
content are supported. Raw markup intervals and normalized DOM text are
distinct proofs. No CSS visibility, browser layout or factual truth is
asserted. Unsupported tables fail closed instead of silently resolving an
apparently unique cell from an incomplete table scan.
"""
from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import re

MAX_HTML_BYTES=8*1024*1024
MAX_DERIVED_CHARS=16*1024*1024
MAX_TABLES=64
MAX_ROWS=2000
MAX_COLUMNS=64
_SHA=re.compile(r"[0-9a-f]{64}\Z")
_VOID=frozenset(("area","base","br","col","embed","hr","img","input","link",
                 "meta","param","source","track","wbr"))
_EXCLUDE=frozenset(("script","style","noscript","template","svg","canvas","nav","aside","footer"))
_INLINE=frozenset(("a","span","b","strong","i","em","code","kbd","samp","small","sup","sub","abbr"))


def _sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()


def _collapse(text:str)->str:return " ".join(text.split())


def _hidden_or_complex_attributes(attrs):
    names=[k for k,_ in attrs]
    if len(names)!=len(set(names)):return True
    d=dict(attrs)
    if "hidden" in d or ("aria-hidden" in d and d["aria-hidden"]!="false"):
        return True
    if any(k in d and d[k]!="1" for k in ("rowspan","colspan")):
        return True
    # Allow the harmless numeric custom-property form used by the archived
    # Epic publisher tables. Arbitrary inline CSS is not interpreted.
    if "style" in d and not re.fullmatch(r"\s*--columns-count\s*:\s*[1-9][0-9]*\s*;?\s*",d["style"] or ""):
        return True
    return False


class UnsupportedOriginalTable(ValueError):
    pass


class _LocatedParser(HTMLParser):
    def __init__(self,text):
        super().__init__(convert_charrefs=True)
        self.source=text
        if text.count("\n")>250000:raise UnsupportedOriginalTable("HTML line count exceeds bound")
        self.line_starts=[0]+[m.end() for m in re.finditer("\n",text)]

    def pos(self):
        line,col=self.getpos()
        return self.line_starts[line-1]+col

    def closing_end(self,tag):
        start=self.pos()
        m=re.compile(r"</"+re.escape(tag)+r"\s*>",re.I).match(self.source,start)
        if not m:raise UnsupportedOriginalTable("nonliteral closing table markup")
        return m.end()


class _MainTables(_LocatedParser):
    """Locate complete table element intervals without inferring HTML5 repair."""
    def __init__(self,text):
        super().__init__(text);self.stack=[];self.depth=0
        self.active=None;self.tables=[]

    def handle_starttag(self,tag,attrs):
        if tag=="table":
            if self.depth==0:
                tags=[t for t,_ in self.stack]
                eligible=any(t in ("article","main") for t in tags) and not any(t in _EXCLUDE for t in tags)
                self.active=(self.pos(),any(bad for _,bad in self.stack)) if eligible else None
            self.depth+=1
        if tag not in _VOID:
            # For ancestors, explicit hidden attributes matter. Other
            # attributes and external CSS cannot prove browser visibility.
            a=dict(attrs)
            bad=("hidden" in a or ("aria-hidden" in a and a["aria-hidden"]!="false"))
            self.stack.append((tag,bad))
            if len(self.stack)>256:raise UnsupportedOriginalTable("HTML nesting exceeds bound")

    def handle_startendtag(self,tag,attrs):
        if tag=="table":
            tags=[t for t,_ in self.stack]
            if self.depth or (any(t in ("article","main") for t in tags) and not any(t in _EXCLUDE for t in tags)):
                raise UnsupportedOriginalTable("self-closing table is not supported")

    def handle_endtag(self,tag):
        if tag=="table" and self.depth:
            self.depth-=1
            if self.depth==0 and self.active is not None:
                start,hidden=self.active
                self.tables.append((start,self.closing_end("table"),hidden))
                self.active=None
                if len(self.tables)>MAX_TABLES:
                    raise UnsupportedOriginalTable("too many publisher content tables")
        for i in range(len(self.stack)-1,-1,-1):
            if self.stack[i][0]==tag:
                del self.stack[i:];break

    def finish(self):
        self.feed(self.source);self.close()
        if self.depth and self.active is not None:
            raise UnsupportedOriginalTable("unclosed publisher content table")
        return self.tables


class _RectangularTable(_LocatedParser):
    """Conservative exact markup parser for explicitly closed simple tables."""
    def __init__(self,text):
        super().__init__(text);self.stack=[];self.rows=[]
        self.row=None;self.cell=None;self.seen_root=False

    def handle_starttag(self,tag,attrs):
        parent=self.stack[-1] if self.stack else None
        if _hidden_or_complex_attributes(attrs):
            raise UnsupportedOriginalTable("hidden, styled or spanning table markup")
        if tag=="table" and parent is None and not self.seen_root:
            self.seen_root=True
        elif tag in ("thead","tbody","caption") and parent=="table":
            pass
        elif tag=="tr" and parent in ("table","thead","tbody"):
            if self.row is not None:raise UnsupportedOriginalTable("nested table row")
            self.row={"start":self.pos(),"cells":[],"parent":parent}
        elif tag in ("td","th") and parent=="tr":
            if self.row is None or self.cell is not None:raise UnsupportedOriginalTable("misnested cell")
            self.cell={"start":self.pos(),"kind":tag,"parts":[]}
        elif (tag in _INLINE or tag in ("br","wbr")) and (
                self.cell is not None or "caption" in self.stack):
            if tag=="br" and self.cell is not None:self.cell["parts"].append(" ")
        else:
            raise UnsupportedOriginalTable("complex or misnested publisher table markup")
        if tag not in ("br","wbr"):
            self.stack.append(tag)
            if len(self.stack)>64:raise UnsupportedOriginalTable("table markup nesting exceeds bound")

    def handle_startendtag(self,tag,attrs):
        if tag not in ("br","wbr"):raise UnsupportedOriginalTable("self-closing table structure")
        self.handle_starttag(tag,attrs)

    def handle_data(self,text):
        if self.cell is not None:
            self.cell["parts"].append(text)
        elif "caption" not in self.stack and text.strip():
            raise UnsupportedOriginalTable("noncell table text")

    def handle_endtag(self,tag):
        if not self.stack or self.stack[-1]!=tag:
            raise UnsupportedOriginalTable("nonexplicit or misnested table closure")
        self.stack.pop()
        if tag in ("td","th"):
            c=self.cell
            if c is None:raise UnsupportedOriginalTable("missing cell start")
            value=_collapse("".join(c.pop("parts")))
            if len(value)>1200:raise UnsupportedOriginalTable("oversized normalized table cell")
            c.update(end=self.closing_end(tag),text=value)
            self.row["cells"].append(c);self.cell=None
            if len(self.row["cells"])>MAX_COLUMNS:raise UnsupportedOriginalTable("table width exceeds bound")
        elif tag=="tr":
            if self.row is None or not self.row["cells"]:raise UnsupportedOriginalTable("missing table row cells")
            self.row["end"]=self.closing_end(tag);self.rows.append(self.row);self.row=None
            if len(self.rows)>MAX_ROWS+1:raise UnsupportedOriginalTable("table rows exceed bound")

    def finish(self):
        self.feed(self.source);self.close()
        if self.stack or self.cell is not None or self.row is not None or not self.seen_root or not self.rows:
            raise UnsupportedOriginalTable("incomplete table structure")
        header=self.rows[0];cols=header["cells"]
        if len(cols)<2 or any(c["kind"]!="th" or not c["text"] for c in cols):
            raise UnsupportedOriginalTable("one explicit header row required")
        if len({c["text"].casefold() for c in cols})!=len(cols):
            raise UnsupportedOriginalTable("ambiguous duplicate column header")
        for r in self.rows[1:]:
            cells=r["cells"]
            if (len(cells)!=len(cols) or r["parent"]=="thead" or not cells[0]["text"]
                    or any(c["kind"]!="td" for c in cells[1:])):
                raise UnsupportedOriginalTable("nonrectangular or multiheader table")
        return self.rows


def _raw_interval(raw,source,base,node):
    start=base+node["start"];end=base+node["end"]
    if not 0<=start<end<=len(source):raise ValueError("invalid original HTML markup interval")
    bs=len(source[:start].encode("utf8"));be=bs+len(source[start:end].encode("utf8"))
    segment=raw[bs:be]
    if segment!=source[start:end].encode("utf8"):raise ValueError("original HTML byte interval mismatch")
    return {"byte_start":bs,"byte_end":be,"raw_markup_sha256":_sha(segment)}


def verified_original_html_table_cell(*,subject,column,original_path,original_sha256,
                                      derived_sha256,derived_text,publisher_html_bytes,
                                      filesystem_entry_id):
    for name,value in (("subject",subject),("column",column)):
        if (not isinstance(value,str) or not 1<=len(value)<=180 or value!=_collapse(value)
                or any(ord(ch)<32 or ord(ch)==127 for ch in value)):
            raise ValueError("invalid exact normalized HTML table "+name)
    if (not isinstance(original_path,str) or not original_path.lower().endswith((".html",".htm"))
            or original_path.startswith("/") or "\\" in original_path or "\x00" in original_path
            or any(p in ("",".","..") for p in original_path.split("/"))
            or not isinstance(original_sha256,str) or not _SHA.fullmatch(original_sha256)
            or not isinstance(derived_sha256,str) or not _SHA.fullmatch(derived_sha256)
            or type(filesystem_entry_id) is not int or filesystem_entry_id<1
            or not isinstance(derived_text,str) or not 1<=len(derived_text)<=MAX_DERIVED_CHARS
            or not isinstance(publisher_html_bytes,bytes) or not 1<=len(publisher_html_bytes)<=MAX_HTML_BYTES):
        raise ValueError("invalid bounded current original HTML table source")
    raw=publisher_html_bytes
    if _sha(raw)!=original_sha256 or _sha(derived_text.encode("utf8"))!=derived_sha256:
        raise ValueError("original HTML or current derivative SHA differs")
    source=raw.decode("utf8","strict")
    common={"schema":"kmd.optin_unique_original_html_table_cell_normalized_proof.v1",
            "requested_first_column_subject":subject,"requested_exact_column":column,
            "original_sha256":original_sha256,"derived_sha256":derived_sha256,
            "native_original_id":filesystem_entry_id,
            "scope":"parser-recognized main/article tables excluding navigation, aside, footer and noncontent elements",
            "selector_normalization":"collapsed whitespace, case-insensitive exact equality",
            "cell_normalization":"HTML character references decoded, inline text concatenated, whitespace collapsed",
            "browser_visibility_or_rendered_layout_verified":False,
            "no_automated_semantic_entailment_or_answer_correctness_claim":True,
            "no_original_or_native_rank_mutation":True}
    matches=[];count=0
    try:
        locations=_MainTables(source).finish()
        for table_index,(start,end,hidden) in enumerate(locations,1):
            if hidden:raise UnsupportedOriginalTable("content table has explicitly hidden ancestor")
            rows=_RectangularTable(source[start:end]).finish()
            count+=len(rows)-1
            if count>MAX_ROWS:raise UnsupportedOriginalTable("publisher total table rows exceed bound")
            header=rows[0];headers=[c["text"] for c in header["cells"]]
            found=[i for i,h in enumerate(headers) if h.casefold()==column.casefold()]
            if not found:continue
            idx=found[0]
            for row_index,r in enumerate(rows[1:],1):
                if r["cells"][0]["text"].casefold()!=subject.casefold():continue
                cell=r["cells"][idx]
                # Save only references during matching. Compute the relatively
                # expensive UTF-8 byte intervals only after uniqueness is known.
                matches.append((table_index,row_index,idx,start,header,r,cell,headers))
    except UnsupportedOriginalTable as error:
        return {**common,"status":"unsupported_table_structure","reason":str(error),
                "matching_original_table_rows":None,"resolved_unique_literal_publisher_cell":None,
                "all_tables_in_declared_scope_supported":False}
    status="resolved" if len(matches)==1 else "ambiguous" if matches else "not_found"
    chosen=None
    if status=="resolved":
        table_index,row_index,idx,start,header,r,cell,headers=matches[0]
        chosen={"original_source_path":original_path,
                "table_ordinal_in_content_scope":table_index,"body_row_ordinal":row_index,
                "column_number":idx+1,"original_table_column_headers":headers,
                "original_table_row_subject":r["cells"][0]["text"],
                "selected_original_table_column_name":headers[idx],
                "selected_original_table_column_value":cell["text"],
                "original_header_row_markup":_raw_interval(raw,source,start,header),
                "original_body_row_markup":_raw_interval(raw,source,start,r),
                "original_subject_cell_markup":_raw_interval(raw,source,start,r["cells"][0]),
                "original_header_cell_markup":_raw_interval(raw,source,start,header["cells"][idx]),
                "original_selected_cell_markup":_raw_interval(raw,source,start,cell),
                "normalized_original_cell_text_sha256":_sha(cell["text"].encode("utf8")),
                "raw_html_markup_byte_interval_verified":True,
                "literal_publisher_text_byte_interval_verified":False,
                "cell_value_is_normalized_original_html_dom_text":True}
    return {**common,"status":status,"matching_original_table_rows":len(matches),
            "tables_in_declared_scope":len(locations),"body_rows_examined":count,
            "all_tables_in_declared_scope_supported":True,
            "resolved_unique_literal_publisher_cell":chosen,
            "not_found_limited_to_declared_table_scope":True}


def compare_claimed_literal_to_original_html_cell(*,original_cell_result,claimed_literal_value):
    if (not isinstance(claimed_literal_value,str) or not 1<=len(claimed_literal_value)<=1200
            or claimed_literal_value!=_collapse(claimed_literal_value)
            or any(ord(ch)<32 or ord(ch)==127 for ch in claimed_literal_value)):
        raise ValueError("comparison requires exact normalized nonempty literal")
    r=original_cell_result
    if not isinstance(r,dict) or r.get("schema")!="kmd.optin_unique_original_html_table_cell_normalized_proof.v1":
        raise ValueError("verified original HTML table proof required")
    status=r.get("status");field=r.get("resolved_unique_literal_publisher_cell");count=r.get("matching_original_table_rows")
    if status=="resolved":
        if (type(count) is not int or count!=1 or not isinstance(field,dict)
                or r.get("all_tables_in_declared_scope_supported") is not True
                or field.get("cell_value_is_normalized_original_html_dom_text") is not True
                or field.get("raw_html_markup_byte_interval_verified") is not True
                or not isinstance(field.get("selected_original_table_column_value"),str)):
            raise ValueError("nonunique or unverified normalized HTML cell")
        value=field["selected_original_table_column_value"]
        digest=_sha(value.encode("utf8"))
        if digest!=field.get("normalized_original_cell_text_sha256"):
            raise ValueError("normalized HTML cell text digest mismatch")
        outcome="literal_match" if value==claimed_literal_value else "literal_different"
    else:
        if (field is not None or status not in ("ambiguous","not_found","unsupported_table_structure")
                or (status=="ambiguous" and (type(count) is not int or count<2))
                or (status=="not_found" and (type(count) is not int or count!=0))
                or (status=="unsupported_table_structure" and count is not None)):
            raise ValueError("invalid unresolved original HTML cell")
        digest=None;outcome="source_"+status+"_no_literal_comparison"
    return {"schema":"kmd.original_html_cell_normalized_literal_assertion_not_semantic_truth.v1",
            "status":outcome,"original_lookup_status":status,
            "claimed_literal_sha256":_sha(claimed_literal_value.encode("utf8")),
            "normalized_source_cell_sha256":digest,
            "literal_difference_is_NOT_proof_semantic_contradiction_or_false_claim":True,
            "literal_match_is_NOT_proof_publisher_information_is_current_beyond_pinned_original":True}
