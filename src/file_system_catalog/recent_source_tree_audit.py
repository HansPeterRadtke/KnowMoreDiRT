"""Read-only full-source and native recursive-summary graph integrity audit.

Use as a bounded independent test of source-linked recursive indexing. A
complete root alone does not prove leaf coverage: verify SHA fingerprints,
original/derivative bytes, native span offsets, embeddings and all ancestor
and leaf relationships, without indexing or changing the one canonical FileDB.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3

from .source_review_audit import safe_file, sha_file

SHA = re.compile(r"[0-9a-f]{64}\Z")
PROMPT = 'kmd-hierarchical-source-summary-v1'
MAX_SOURCE_BYTES = 64 * 1024 * 1024  # Default ordinary original-PDF byte audit cap.
MAX_EXTENDED_SOURCE_BYTES = 256 * 1024 * 1024  # Explicit opt-in only, source hash streams.
MAX_DERIVED_BYTES = 64 * 1024 * 1024  # Entire derivative is inspected in memory.
MAX_NODES = 4096


def audit_current_source_tree(
    *, original_path: str, source_map: sqlite3.Connection,
    native_db: sqlite3.Connection, original_root: Path,
    derivative_root: Path, completed_receipts: dict[str,str],
    max_original_bytes: int = MAX_SOURCE_BYTES,
    max_derived_bytes: int = MAX_DERIVED_BYTES,
) -> dict:
    """Audit an entire currently completed publisher original's native tree.

    All official source, derived-text and KMD byte digests are actually
    recomputed. All graph links refer to generated node IDs and exact spans
    in the current derived original; this verifies structure and provenance,
    **not** factual accuracy of generated summaries.
    """
    # The publisher PDF is independently SHA-hashed via safe_file/sha_file's
    # bounded streaming descriptor (not loaded into RAM), so a larger real
    # publisher scan can be admitted deliberately, without expanding the
    # 64-MiB in-memory *derived source text* cap or changing normal defaults.
    if (type(max_original_bytes) is not int or not 1<=max_original_bytes<=MAX_EXTENDED_SOURCE_BYTES
            or type(max_derived_bytes) is not int or not 1<=max_derived_bytes<=MAX_DERIVED_BYTES):
        raise ValueError('bounded publisher source and derivative limits required')
    original = safe_file(original_root, original_path)
    current = source_map.execute(
        "SELECT source_sha256, derived_sha256, derived_path, status, original_bytes "
        "FROM originals WHERE source_path=?",(original_path,),
    ).fetchall()
    if len(current)!=1 or current[0][3]!='complete':
        raise ValueError('source-map lacks unique current completed original')
    original_sha,derivative_sha,relative_derived,_,expected_bytes = current[0]
    if (not isinstance(original_sha,str) or not SHA.fullmatch(original_sha)
            or not isinstance(derivative_sha,str) or not SHA.fullmatch(derivative_sha)
            or completed_receipts.get(original_path)!=original_sha):
        raise ValueError('original source revision differs from approved receipt')
    derivative=safe_file(derivative_root,relative_derived)
    if (original.stat().st_size>max_original_bytes
            or derivative.stat().st_size>max_derived_bytes):
        raise ValueError('original or derivative exceeds independent byte audit limit')
    if (type(expected_bytes) is not int or original.stat().st_size!=expected_bytes
            or sha_file(original)!=original_sha):
        raise ValueError('current publisher-original bytes differ from recorded revision')
    raw=derivative.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=derivative_sha:
        raise ValueError('current extracted text differs from recorded source revision')
    text=raw.decode('utf-8','strict')
    if not text or '[[SOURCE UNIT:' not in text:
        raise ValueError('source derivative lacks authentic page/source unit boundary')
    identities=native_db.execute(
        "SELECT id FROM filesystem_entries WHERE relative_path_display=? AND entry_type='file'",
        (original_path,),
    ).fetchall()
    if len(identities)!=1:
        raise ValueError('native FileDB contains absent or ambiguous publisher original')
    entry_id=identities[0][0]
    rows=native_db.execute('''
        SELECT c.start_char, c.end_char, c.text_sha256,c.content_sha256,
               r.representation_kind,r.representation_text,r.representation_text_sha256,
               r.embedding_blob,r.embedding_sha256,r.generation_json,
               r.embedding_model,r.embedding_dimension,r.embedding_dtype,r.embedding_norm
        FROM content_chunks c JOIN content_representations r ON r.chunk_id=c.chunk_id
        WHERE c.filesystem_entry_id=? AND c.collection_id LIKE '%:summary'
          AND r.prompt_version=? AND r.analysis_status='complete'
          AND r.representation_kind IN ('summary','summary_long')
    ''',(entry_id,PROMPT)).fetchall()
    if not rows or len(rows)>MAX_NODES:
        raise ValueError('native recursive-summary node cardinality missing or excessive')
    nodes={}
    stale=0
    for (start,end,span_sha,chunk_original,kind,summary,summary_sha,embedding,
         embedding_sha,generation,embedding_model,embedding_dim,embedding_dtype,embedding_norm) in rows:
        try:
            props=json.loads(generation)
        except (ValueError,TypeError) as exc:
            raise ValueError('invalid native summary node generation provenance') from exc
        if not isinstance(props,dict):
            raise ValueError('native node generation is not a source dictionary')
        if (props.get('original_sha256')!=original_sha
                or props.get('derived_sha256')!=derivative_sha
                or chunk_original!=original_sha):
            stale+=1
            continue  # A separate historical revision can coexist; it is not current.
        node_id=props.get('node_id')
        children=props.get('children')
        level=props.get('summary_level')
        if (not isinstance(node_id,str) or not 4<=len(node_id)<=160
                or node_id in nodes or not isinstance(children,list)
                or len(children)>MAX_NODES
                or any(not isinstance(v,str) or not v for v in children)
                or len(children)!=len(set(children))
                or type(level) is not int or not 0<=level<64):
            raise ValueError('invalid, duplicated or contradictory summary node identity')
        if (type(start) is not int or type(end) is not int
                or not 0<=start<end<=len(text)
                or not isinstance(span_sha,str) or not SHA.fullmatch(span_sha)
                or hashlib.sha256(text[start:end].encode('utf-8')).hexdigest()!=span_sha
                or props.get('source_start_char')!=start
                or props.get('source_end_char')!=end
                or props.get('source_passage_sha256')!=span_sha
                or props.get('source_tree_sha256')!=derivative_sha):
            raise ValueError('source span or generated source provenance does not match extracted original')
        if (not isinstance(summary,str) or not summary.strip()
                or not isinstance(summary_sha,str) or not SHA.fullmatch(summary_sha)
                or hashlib.sha256(summary.encode('utf-8')).hexdigest()!=summary_sha
                or props.get('summary_sha256')!=summary_sha):
            raise ValueError('native summary content differs from generated SHA')
        width={'float16':2,'float32':4,'float64':8,'int8':1}.get(embedding_dtype)
        if (not isinstance(embedding,(bytes,bytearray,memoryview))
                or not isinstance(embedding_sha,str) or not SHA.fullmatch(embedding_sha)
                or hashlib.sha256(embedding).hexdigest()!=embedding_sha
                or not isinstance(embedding_model,str) or not embedding_model.strip()
                or type(embedding_dim) is not int or not 1<=embedding_dim<=16384
                or width is None or len(embedding)!=embedding_dim*width
                or not isinstance(embedding_norm,(float,int)) or isinstance(embedding_norm,bool)
                or not math.isfinite(embedding_norm) or not 0<embedding_norm<=10000):
            raise ValueError('native summary embedding digest/dimension/dtype/norm inconsistent')
        nodes[node_id]=(start,end,kind,level,children)
    if not nodes:
        raise ValueError('no current native summary nodes from this original revision')
    roots=[identifier for identifier,(start,end,kind,level,children) in nodes.items()
           if kind=='summary_long' and start==0 and end==len(text)]
    if len(roots)!=1:
        raise ValueError('current original lacks exactly one complete-span root summary')
    root_id=roots[0]
    visited=set()
    active=set()
    def traverse(node_id):
        if node_id in active:
            raise ValueError('recursive summary node graph contains cycle')
        if node_id in visited:
            raise ValueError('recursive summary node graph contains duplicate child ownership')
        if node_id not in nodes:
            raise ValueError('recursive summary node graph references missing child')
        active.add(node_id)
        start,end,kind,level,children=nodes[node_id]
        if children:
            if any(child not in nodes for child in children):
                raise ValueError('recursive summary node graph references missing child')
            ordered=sorted(children,key=lambda child:nodes[child][0])
            if ordered!=children:
                raise ValueError('source summary children are not in source order')
            next_start=start
            for child in children:
                if child not in nodes:
                    raise ValueError('recursive summary node graph references missing child')
                cstart,cend,ckind,clevel,_=nodes[child]
                if cstart!=next_start or cend>end or clevel>=level:
                    raise ValueError('recursive summary children do not partition parent source span')
                traverse(child)
                next_start=cend
            if next_start!=end:
                raise ValueError('recursive summary children do not cover complete parent span')
        elif kind=='summary_long' and len(nodes)>1:
            raise ValueError('root appears complete but has no recursively linked source children')
        active.remove(node_id)
        visited.add(node_id)
    traverse(root_id)
    if len(visited)!=len(nodes):
        raise ValueError('recursive summary has orphaned or disconnected original-linked nodes')
    return {
        'original_path':original_path,'publisher_original_sha256_reverified':original_sha,
        'derived_original_sha256_reverified':derivative_sha,
        'original_bytes_rehashed':original.stat().st_size,
        'derived_bytes_rehashed':len(raw),'derived_chars_checked':len(text),
        'current_native_summary_nodes_fully_verified':len(nodes),
        'current_root_id_sha256':hashlib.sha256(root_id.encode('utf-8')).hexdigest(),
        'maximum_recursive_level':max(x[3] for x in nodes.values()),
        'historical_revision_nodes_not_counted_as_current':stale,
        'all_summary_nodes_source_spans_and_embeddings_reverified':True,
        'all_original_text_covered_by_recursive_summary_graph':True,
        'no_index_or_source_mutation':True,
        'truth_scope':'Structural, original-byte and embedding-byte provenance only; not model-summary factual accuracy.',
    }
