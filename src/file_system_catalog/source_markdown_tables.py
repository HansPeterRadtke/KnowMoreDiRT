"""Read-only original-byte-verified Markdown table row evidence.

Publisher source tables have structured fields that can be lost when an FTS
snippet truncates a wide row. This helper preserves the exact original
UTF-8 row, original header/column names, original and derivative revisions,
and a publisher-original byte SHA for cited evidence. It does NOT decide the
factual truth of the user's question, and never re-ranks native documents.

Only simple pipe-bordered Markdown tables are admitted. Fenced code, HTML
comments, escaped pipes, malformed headers or ambiguous derived occurrences
are excluded rather than guessed. No download, source or index mutation.
"""
from __future__ import annotations
import hashlib
import re

from .hybrid_search import query_terms,lexical_token_spans

_SHA=re.compile(r'^[0-9a-f]{64}$')
_FENCE=re.compile(r'^ {0,3}(`{3,}|~{3,})')
_SEPARATOR=re.compile(r':?-{3,}:?')
_HEADING=re.compile(r'^ {0,3}(#{1,6})[ \t]+(.+)$')
_MAX_SOURCE_BYTES=8*1024*1024


def _sha(blob:bytes)->str:return hashlib.sha256(blob).hexdigest()


def _parse_row(line:str):
    line=line.rstrip('\r\n')
    if (not line.startswith('|') or not line.endswith('|')
            or '\\|' in line or len(line)>8000):return None
    cells=[x.strip() for x in line[1:-1].split('|')]
    if not 2<=len(cells)<=12 or any(not x or len(x)>2000 for x in cells):return None
    return cells


def _longest_exact_subject_phrase(subject_words:list[str],query_words:list[str])->int:
    """Prefer specific original row identifiers over shared product names.

    "n8n self-hosted Business" names the Business row more specifically
    than another publisher row "Self-hosted Community, n8n Cloud Pro" even
    though both contain the same number of query tokens. Only exact
    contiguous whole-word forms in the publisher first cell are considered.
    """
    best=0
    for start in range(len(subject_words)):
        for begin in range(len(query_words)):
            n=0
            while (start+n<len(subject_words) and begin+n<len(query_words)
                   and subject_words[start+n]==query_words[begin+n]):
                n+=1
            best=max(best,n)
    return best


def _table_rows(raw:bytes):
    """Yield physical publisher Markdown rows with original byte coordinates."""
    try:text=raw.decode('utf8','strict')
    except UnicodeDecodeError as exc:raise ValueError('original Markdown table is not valid UTF8') from exc
    pos=0;fence=None;comment=False;headings=[]
    physical=[]
    for line in text.splitlines(keepends=True):
        old=pos;part=line.encode('utf8');pos+=len(part)
        marker=_FENCE.match(line)
        if marker:
            sign,count=marker[1][0],len(marker[1])
            if fence is None:fence=(sign,count)
            elif sign==fence[0] and count>=fence[1]:fence=None
            physical.append((None,old,pos,None,' / '.join(x[1] for x in headings[-3:])));continue
        if fence is not None or line.startswith(('    ','\t')):
            physical.append((None,old,pos,None,' / '.join(x[1] for x in headings[-3:])));continue
        if comment or '<!--' in line or '-->' in line:
            if comment:
                if '-->' in line:comment=False
            elif '<!--' in line and ('-->' not in line or line.index('-->')<line.index('<!--')):comment=True
            physical.append((None,old,pos,None,' / '.join(x[1] for x in headings[-3:])));continue
        heading=_HEADING.match(line.rstrip('\r\n'))
        if heading:
            title=re.sub(r'[ \t]+#+[ \t]*$','',heading[2]).strip()
            if title:
                level=len(heading[1])
                while headings and headings[-1][0]>=level:headings.pop()
                headings.append((level,title[:170]))
        cells=_parse_row(line)
        physical.append((line.rstrip('\r\n') if cells is not None else None,old,pos,cells,
                         ' / '.join(x[1] for x in headings[-3:])[:400]))
    assert len(raw)==pos
    for i in range(len(physical)-2):
        header,hstart,hend,cols,section=physical[i]
        if header is None:continue
        _sep,_,_,delimiter,_unused=physical[i+1]
        if (delimiter is None or len(delimiter)!=len(cols)
                or not all(_SEPARATOR.fullmatch(c) for c in delimiter)):
            continue
        # Only a contiguous run of matching physical pipe rows following a
        # proper Markdown header/separator is accepted. The literal byte
        # ranges are preserved, not re-rendered.
        j=i+2
        while j<len(physical):
            line,start,end,values,_row_section=physical[j]
            if values is None or len(values)!=len(cols):break
            yield cols,header,hstart,hend,values,line,start,end,section
            j+=1


def verified_original_markdown_table_rows(*,query:str,original_path:str,
        original_sha256:str,derived_sha256:str,derived_text:str,
        publisher_text_bytes:bytes,filesystem_entry_id:int,top:int=2)->dict:
    """Source-verbatim table candidates; no entailment/availability inference."""
    if (not isinstance(query,str) or not 2<=len(query)<=300
            or not isinstance(original_path,str) or not original_path.lower().endswith(('.md','.markdown'))
            or original_path.startswith('/') or '\\' in original_path
            or any(x in ('','.','..') for x in original_path.split('/'))
            or not isinstance(original_sha256,str) or not _SHA.fullmatch(original_sha256)
            or not isinstance(derived_sha256,str) or not _SHA.fullmatch(derived_sha256)
            or not isinstance(derived_text,str) or not derived_text
            or not isinstance(publisher_text_bytes,bytes)
            or not 1<=len(publisher_text_bytes)<=_MAX_SOURCE_BYTES
            or type(filesystem_entry_id) is not int or filesystem_entry_id<1
            or type(top) is not int or not 1<=top<=5):
        raise ValueError('invalid bounded first-party publisher Markdown table source')
    if _sha(publisher_text_bytes)!=original_sha256:
        raise ValueError('entire original Markdown publisher SHA differs from completed receipt')
    if _sha(derived_text.encode('utf8'))!=derived_sha256:
        raise ValueError('current derivative SHA differs from current original source-map')
    query_words=list(query_terms(query))
    terms=set(query_words)
    if not terms:raise ValueError('source table lookup requires informative question terms')
    physical_rows=[]
    for columns,header,hstart,hend,values,line,start,end,section in _table_rows(publisher_text_bytes):
        if len(physical_rows)>=2000:raise ValueError('too many publisher table rows for bounded original QA')
        row_words={w for w,_,_ in lexical_token_spans(line)}
        first_sequence=[w for w,_,_ in lexical_token_spans(values[0])]
        first_words=set(first_sequence)
        subject=sorted(terms&first_words)
        matching=sorted(terms&row_words)
        column_terms={w for col in columns for w,_,_ in lexical_token_spans(col)}
        header_matches=sorted(terms&column_terms)
        section_matches=sorted(terms&{w for w,_,_ in lexical_token_spans(section)})
        if not matching or (not subject and len(matching)<2):continue
        # Reject ambiguous copies of a row in the derived extraction.
        original_literal=publisher_text_bytes[start:end].decode('utf8','strict').rstrip('\r\n')
        source_literal=line
        if original_literal!=source_literal:
            raise ValueError('publisher table original byte interval is inconsistent')
        deriv_pos=derived_text.find(source_literal)
        if deriv_pos<0 or derived_text.find(source_literal,deriv_pos+1)>=0:
            raise ValueError('original publisher table row not uniquely present in current derivative')
        header_offset=derived_text.find(header)
        if header_offset<0 or derived_text.find(header,header_offset+1)>=0:
            raise ValueError('original publisher table column heading missing or ambiguous in derivative')
        # Table COLUMN meaning is part of a question: a request for a
        # mathematical *symbol* should prefer the publisher's Symbols table
        # over an unrelated shader *Port* named `normal`. Specific first
        # column identifiers then beat incidental matches in other fields.
        # Heading is context, NOT independent answer evidence.
        phrase_specificity=_longest_exact_subject_phrase(first_sequence,query_words)
        # The user may mention a direction as context while asking for a
        # different quantity. In MaterialX, `omega_i` is the *direction* of
        # incoming radiance, whereas `$L_i$` is the incident radiance itself.
        # Compare exact adjacent informative query terms INSIDE EACH publisher
        # description cell, without inventing continuity across table cells.
        explanatory_cells=(cell for heading,cell in zip(columns[1:],values[1:])
                          if re.search(r'(?i)description|use when|meaning|definition|details',heading))
        descriptive_phrase_specificity=max((
            _longest_exact_subject_phrase(list(query_terms(cell,max_terms=100)),query_words)
            for cell in explanatory_cells),default=0)
        score=(len(header_matches),descriptive_phrase_specificity,
               phrase_specificity,len(subject),len(section_matches),len(matching))
        physical_rows.append({'score':score,
            'publisher_original_path':original_path,
            'publisher_original_sha256':original_sha256,
            'current_derivative_sha256':derived_sha256,
            'unique_native_original_filedb_id':filesystem_entry_id,
            'publisher_table_columns':columns,
            'publisher_table_section_heading':section,
            'matched_query_terms_in_table_column_headings':header_matches,
            'matched_query_terms_in_original_heading_context':section_matches,
            'publisher_table_row_cells':values,
            'publisher_header_exact_text':header,
            'publisher_header_byte_start':hstart,
            'publisher_header_byte_end':hend,
            'publisher_header_SHA256':_sha(publisher_text_bytes[hstart:hend]),
            'publisher_row_exact_text':line,
            'publisher_row_original_start_byte':start,
            'publisher_row_original_end_byte':end,
            'publisher_row_original_SHA256':_sha(publisher_text_bytes[start:end]),
            'current_derived_row_start_char':deriv_pos,
            'current_derived_row_end_char':deriv_pos+len(line),
            'current_derived_row_SHA256':_sha(line.encode('utf8')),
            'matched_query_terms_in_row':matching,
            'matched_query_terms_in_first_identifier_column':subject,
            'longest_exact_query_phrase_in_first_identifier_column':phrase_specificity,
            'longest_exact_query_phrase_in_one_description_column':descriptive_phrase_specificity,
            'source_row_is_candidate_not_proof_of_factual_entailment':True,
            'original_publisher_exact_table_row_utf8_byte_interval_verified':True})
    physical_rows.sort(key=lambda x:(-x['score'][0],-x['score'][1],-x['score'][2],-x['score'][3],
                                     -x['score'][4],-x['score'][5],x['publisher_row_original_start_byte']))
    selected=[]
    for row in physical_rows[:top]:
        score=row.pop('score')
        selected.append(row)
    return {'schema':'kmd.optin_firstparty_original_markdown_table_row_source_evidence.v1',
        'original_path':original_path,'original_sha256':original_sha256,
        'current_derivative_sha256':derived_sha256,
        'original_native_filedb_id':filesystem_entry_id,
        'returned_original_publisher_table_row_candidates':len(selected),
        'publisher_table_row_candidates_with_matching_terms':len(physical_rows),
        'publisher_rows':selected,
        'current_publisher_source_byte_proof_not_semantic_answer_correctness':True,
        'no_native_rank_or_pdf_physical_page_claim':True,
        'read_only_current_approved_original_and_extracted_text':True}
