"""Read-only content and recursive-summary fidelity gate for corrected PDF cohorts.

A private forward snapshot is NOT a searchable index merely because its new
source text, source-map revision and native original IDs are internally valid.
Before a future controlled release, each original must have current native
source passages, valid embedding blobs, an entire correct-revision summary
tree, and a complete root/child vector hierarchy. This gate cannot publish
or promote a generation and never authorizes a live multi-store cutover.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
import struct
from pathlib import Path
import re
import sqlite3

from .hierarchical_summary import SummaryNode,SummaryTree,_VERSION
from .hierarchical_summary_store import validate_complete_summary_tree
from .source_page_shadow_cohort import _require_read_only
from .source_revision_audit import _verified_derived_body_count
from .source_review_audit import safe_file,sha_file
from .summary_coverage import current_completed_originals,revision_identity
from .staged_physical_page_search import MARKER,source_units_from_corrected_text

_MAX_TREE_BYTES=32*1024*1024
_ALLOWED_CHUNK_TYPES=('overview','full','summary')
_SHA=re.compile(r'^[a-f0-9]{64}$')


def _text_sha(raw: str) -> str:
    return hashlib.sha256(raw.encode('utf8')).hexdigest()


def _valid_native_embedding(blob, digest, dimension, dtype, model, revision) -> bool:
    # A self-consistent SHA over all-zero, NaN or wrong-width bytes is still
    # NOT a valid searchable cosine vector. Reuse the native 384-float contract.
    if (not isinstance(blob,bytes) or dimension!=384 or
            dtype not in ('float16','float32') or
            not isinstance(model,str) or not model.strip() or
            not isinstance(revision,str) or not revision.strip()):
        return False
    size=2 if dtype=='float16' else 4
    if len(blob)!=dimension*size or hashlib.sha256(blob).hexdigest()!=digest:
        return False
    values=struct.iter_unpack('<e' if size==2 else '<f',blob)
    norm2=0.0
    for (value,) in values:
        if not math.isfinite(value):return False
        norm2+=value*value
    return 0.98<=norm2<=1.02


def _summary_tree(*,root:Path,path:str,original_sha:str,derived_sha:str,
                  expected_text_len:int) -> SummaryTree:
    prefix=re.sub(r'[^A-Za-z0-9_.-]+','-',Path(path).name[:64])
    name=prefix+'.'+revision_identity(path,original_sha,derived_sha)[:24]+'.summary-tree.json'
    location=safe_file(root,name)
    if location.stat().st_size>_MAX_TREE_BYTES:
        raise ValueError('unbounded source-revision summary-tree file')
    data=json.loads(location.read_bytes())
    if (data.get('schema')!='kmd-complete-hierarchical-summary-tree-v1'
            or data.get('original_path')!=path
            or data.get('original_sha256')!=original_sha
            or data.get('derived_sha256')!=derived_sha):
        raise ValueError('summary tree revision or original source identity differs')
    body=data['tree']
    if (body.get('source_sha256')!=derived_sha or
            body.get('original_characters')!=expected_text_len or
            not isinstance(body.get('nodes'),list) or len(body['nodes'])>50000):
        raise ValueError('summary tree does not cover corrected original derivative')
    nodes=tuple(SummaryNode(**{**node,'child_ids':tuple(node['child_ids'])})
                for node in body['nodes'])
    tree=SummaryTree(source_sha256=body['source_sha256'],
         original_characters=body['original_characters'],root_id=body['root_id'],
         nodes=nodes,leaf_sections=body['leaf_sections'])
    validate_complete_summary_tree(tree)
    for node in tree.nodes:
        if (node.prompt_version!=_VERSION or node.summary_sha256!=_text_sha(node.summary)
                or node.native_analysis is not None and
                    node.native_analysis.get('document_summary')!=node.summary):
            raise ValueError('summary node SHA or native facets inconsistent')
    return tree


def _native_vectors_for_one(db:sqlite3.Connection,*,file_id:int,original_sha:str,
                            derived_sha:str,source_text:str,tree:SummaryTree|None):
    """Validate every original-linked chunk/representation, not only one root.

    Native :overview/:full embeddings are direct source passages and do not
    carry summary generation JSON. They still require source-span hash checks
    against the *new* corrected derivative. All :summary representations have
    full generated-revision metadata and valid embeddings for each tree node.
    """
    cursor=db.execute('''SELECT c.chunk_id,c.collection_id,c.content_sha256,c.content_object_id,
             c.start_char,c.end_char,c.text_sha256,c.embedding_blob,c.embedding_sha256,
             c.embedding_dimension,c.embedding_dtype,c.embedding_model,c.embedding_model_revision,
             r.representation_kind,r.representation_text,r.representation_text_sha256,
             r.generation_json,r.embedding_blob,r.embedding_sha256,
             r.embedding_dimension,r.embedding_dtype,r.embedding_model,r.embedding_model_revision,
             r.prompt_version,r.analysis_status
        FROM content_chunks c LEFT JOIN content_representations r ON r.chunk_id=c.chunk_id
        WHERE c.filesystem_entry_id=? ORDER BY c.chunk_id,r.representation_id''',(file_id,))
    chunks=set();kinds=Counter();reps=0
    summary_nodes={}
    node_by_id={} if tree is None else {node.node_id:node for node in tree.nodes}
    for row in cursor:
        (chunk_id,collection,orig,object_id,start,end,source_segment_sha,
         chunk_blob,chunk_digest,chunk_dim,chunk_dtype,chunk_model,chunk_revision,
         kind,text,text_digest,generation,blob,blob_digest,
         dim,dtype,model,model_revision,prompt,analysis)=row
        if chunk_id not in chunks:
            chunks.add(chunk_id)
            category=collection.rsplit(':',1)[-1] if isinstance(collection,str) else None
            if (category not in _ALLOWED_CHUNK_TYPES
                    or orig!=original_sha or type(start) is not int or type(end) is not int
                    or not 0<=start<end<=len(source_text)
                    or _text_sha(source_text[start:end])!=source_segment_sha):
                raise ValueError('native original chunk is stale, unrecognized or wrong source span')
            kinds[category]+=1
            if category=='summary':
                if (object_id!='sha256:'+derived_sha or
                        chunk_blob is not None or chunk_digest is not None):
                    raise ValueError('summary node source derivative identity or vector placement wrong')
            else:
                if not _valid_native_embedding(chunk_blob,chunk_digest,chunk_dim,
                                             chunk_dtype,chunk_model,chunk_revision):
                    raise ValueError('corrected original direct-source vector is missing or tampered')
        category=collection.rsplit(':',1)[-1]
        if kind is None:
            if category=='summary':
                raise ValueError('summary source chunk has no native representation')
            continue
        if category!='summary':
            raise ValueError('direct publisher content unexpectedly has unverified dependent representation')
        if (not isinstance(text,str) or not text.strip() or text_digest!=_text_sha(text)
                or not _valid_native_embedding(blob,blob_digest,dim,dtype,model,model_revision)
                or prompt!=_VERSION or analysis!='complete'):
            raise ValueError('native summary vector, status or text digest is unverified')
        try:
            meta=json.loads(generation)
        except (ValueError,TypeError) as exc:
            raise ValueError('native summary representation metadata invalid') from exc
        if (not isinstance(meta,dict) or
            meta.get('original_sha256')!=original_sha or
            meta.get('derived_sha256')!=derived_sha or
            meta.get('source_tree_sha256')!=derived_sha or
            meta.get('source_start_char')!=start or
            meta.get('source_end_char')!=end or
            meta.get('source_passage_sha256')!=source_segment_sha):
            raise ValueError('stale native representation belongs to another extracted-text revision')
        node_id=meta.get('node_id')
        if (not isinstance(node_id,str) or not node_id
                or node_id not in node_by_id):
            raise ValueError('native representation does not match a complete summary-tree node')
        node=node_by_id[node_id]
        if (node.start_char!=start or node.end_char!=end
                or meta.get('summary_sha256')!=node.summary_sha256
                or kind in ('summary','summary_long') and
                   (text!=node.summary or text_digest!=node.summary_sha256)):
            raise ValueError('native summary/facet does not match its current tree node')
        if kind in ('summary','summary_long'):
            if kind!=('summary_long' if node.node_id==tree.root_id else 'summary'):
                raise ValueError('complete-root and leaf native representation kind inconsistent')
            if node.node_id in summary_nodes:
                raise ValueError('duplicate native summary vector for one complete tree node')
            summary_nodes[node_id]=True
        reps+=1
    if not kinds['overview'] or not kinds['summary'] or not reps:
        raise ValueError('corrected source lacks direct original and hierarchical summary vectors')
    if tree is None or set(summary_nodes)!={n.node_id for n in tree.nodes}:
        raise ValueError('corrected source tree nodes have missing native representations')
    return {'direct_overview_chunks':kinds['overview'],'direct_full_chunks':kinds['full'],
            'recursive_summary_chunks':kinds['summary'],'native_representations':reps,
            'native_tree_nodes_matched':len(summary_nodes)}


def assess_corrected_native_release(*,records:list[dict],source_map:sqlite3.Connection,
             native:sqlite3.Connection,historical:sqlite3.Connection,
             derived_root:Path,summary_root:Path,
             expected_count:int=327) -> dict:
    """Fail-closed, read-only acceptance of *new corrected semantic content*.

    Even if 327/327 pass, this does not cover other readers/writers or make
    source-map/FileDB/filesystem publication atomic. Never mutate live files.
    """
    _require_read_only(source_map);_require_read_only(native);_require_read_only(historical)
    if (type(expected_count) is not int or not 1<=expected_count<=327
            or not isinstance(records,list) or len(records)!=expected_count
            or not all(isinstance(root,Path) and root.is_absolute()
                 and root.is_dir() and not root.is_symlink()
                 for root in (derived_root,summary_root))):
        raise ValueError('invalid bounded private publisher cohort or root')
    paths=set();identities=set();failed=Counter();examples=[];passed=0
    source_verified=0;roots_verified=0;chunks_verified=0;reps_verified=0
    sources={}
    for item in records:
        if not isinstance(item,dict):raise ValueError('untrusted original release receipt')
        path=item.get('original_path');ident=item.get('native_original_filedb_id')
        original_sha=item.get('original_sha256');derived_sha=item.get('new_staged_derived_sha256')
        if (not isinstance(path,str) or path in paths or not path.endswith('.pdf')
                or path.startswith('/') or '\\' in path or
                any(z in ('','.','..') for z in path.split('/'))
                or type(ident) is not int or ident<=0 or ident in identities or
                not isinstance(original_sha,str) or not _SHA.fullmatch(original_sha)
                or not isinstance(derived_sha,str) or not _SHA.fullmatch(derived_sha)):
            raise ValueError('duplicate, unsafe or unattested original identity')
        paths.add(path);identities.add(ident)
        why=None
        try:
            source=safe_file(derived_root,path+'.txt')
            if source.stat().st_size>32*1024*1024 or sha_file(source)!=derived_sha:
                raise ValueError('corrected derivative SHA differs from pinned source')
            text=source.read_text('utf8','strict')
            chars=_verified_derived_body_count(text,original_path=path,
                    original_sha=original_sha,source_kind='pdf')
            pages=item.get('original_pdf_physical_pages')
            markers=list(MARKER.finditer(text))
            if (type(pages) is not int or not 1<=pages<=450 or
                not markers or set(int(m[2]) for m in markers)!=set(range(1,pages+1))
                or len(markers)!=item.get('new_staged_original_page_unit_count')):
                raise ValueError('corrected PDF physical source-page identity incomplete')
            # Enforce parser page ordering/continuations independently; this
            # is structural validity, not an asserted verified quotation.
            list(source_units_from_corrected_text(text,pages,pages))
            source_verified+=1
            history=historical.execute('''SELECT original_sha256,old_derived_sha256,
                      new_staged_derived_sha256,native_file_id FROM snapshot_revisions
                      WHERE original_path=?''',(path,)).fetchall()
            if len(history)!=1 or history[0]!=(original_sha,item.get('old_derived_sha256'),
                                                      derived_sha,ident):
                raise ValueError('corrected source differs from sealed historical original revision and identity')
            old_types={category:count for category,count in historical.execute(
                    '''SELECT SUBSTR(collection_id,INSTR(collection_id,':')+1),COUNT(*)
                       FROM content_chunks WHERE filesystem_entry_id=? GROUP BY 1''',(ident,))}
            if old_types.get('overview',0)<1:
                raise ValueError('historical original lacks native direct-source overview baseline')
            row=source_map.execute('''SELECT source_sha256,derived_sha256,derived_path,
                         original_kind,extracted_pages,nonempty_pages,text_characters,status
                         FROM originals WHERE source_path=?''',(path,)).fetchall()
            if len(row)!=1 or row[0]!=(original_sha,derived_sha,path+'.txt','pdf',
                                      pages,pages,chars,'complete'):
                raise ValueError('source-map corrected revision is not fully published and complete')
            native_id=native.execute('''SELECT id,relative_path_display,entry_type
                   FROM filesystem_entries WHERE id=?''',(ident,)).fetchall()
            if native_id!=[(ident,path,'file')]:
                raise ValueError('native original file identity differs from publisher source receipt')
            tree=_summary_tree(root=summary_root,path=path,original_sha=original_sha,
                       derived_sha=derived_sha,expected_text_len=len(text))
            roots_verified+=1
            summary=_native_vectors_for_one(native,file_id=ident,original_sha=original_sha,
                    derived_sha=derived_sha,source_text=text,tree=tree)
            if old_types.get('full',0)>0 and summary['direct_full_chunks']<1:
                raise ValueError('new original omits a historically present full-document native content channel')
            chunks_verified+=sum(summary[k] for k in ('direct_overview_chunks','direct_full_chunks','recursive_summary_chunks'))
            reps_verified+=summary['native_representations']
            sources[path]=(original_sha,derived_sha,chars)
            passed+=1
        except (ValueError,OSError,KeyError,TypeError,UnicodeError,sqlite3.Error,StopIteration) as error:
            why=str(error)[:140]
            failed[why]+=1
            if len(examples)<12:
                examples.append({'source_path':path,'reason':why})
    # Reuse the production-native root definition, not the existence of any
    # vector, leaf, stale source root, or filename alone.
    native_complete=current_completed_originals(native,sources)
    if len(native_complete)!=passed:
        failed['native corrected full-document root not accepted by production status reader']+=passed-len(native_complete)
        passed=len(native_complete)
    ready=(passed==expected_count and not failed)
    return {'schema':'kmd.corrected_pdf_native_source_revision_release_fidelity_gate.v1',
         'cohort_expected_originals':expected_count,
         'originals_with_rechecked_source_derivative_sha':source_verified,
         'originals_with_verified_complete_recursion_tree':roots_verified,
         'originals_with_native_full_source_roots_and_vectors':passed,
         'new_corrected_native_source_chunks_validated':chunks_verified,
         'new_corrected_native_summary_representations_validated':reps_verified,
         'failures_by_reason':dict(sorted(failed.items())),
         'bounded_failed_source_examples':examples,
         'all_corrected_revision_search_evidence_complete':ready,
         'full_corpus_new_revision_search_ready':ready,
         'live_multi_store_atomicity_or_quiesced_reader_barrier_verified':False,
         'safe_to_cutover_live':False,
         'all_inputs_read_only':True,
         'pdf_page_excerpt_claims_require_independent_original_publisher_file_verification':True}
