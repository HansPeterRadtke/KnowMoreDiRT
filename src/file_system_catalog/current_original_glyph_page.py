"""Read-only, explicitly selected physical PDF page search without a prior stage.

No source acquisition, OCR, model, native FileDB write or implicit source
recovery. Ghostscript text is only a provisional candidate. A literal same-
physical-page Poppler match may corroborate an *excerpt* or a short fragment,
NEVER the whole machine transcript, rendered image or scientific conclusion.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Callable

from .alternate_pdf_page_text import extract_alternate_original_pdf_pages
from .alternate_pdf_page_search import provisional_alternate_query_hits
from .publisher_firstunit_discovery import publisher_pdf_page_map_bounds
from .source_review_audit import safe_file,sha_file

_MAX_ORIGINAL_BYTES=50*1024*1024
_MAX_DERIVED_BYTES=32*1024*1024
_MAX_ORIGINAL_PAGES=450
_MAX_LEDGER_BYTES=64*1024*1024
_MAX_CATALOG_BYTES=32*1024*1024
_HEX=re.compile(r'[0-9a-f]{64}\Z')


def _checked_json(path:Path,bound:int)->dict:
    if (not isinstance(path,Path) or path.is_symlink()
            or not path.is_file() or not 0<path.stat().st_size<=bound):
        raise ValueError('missing/untrusted/oversized approved publisher metadata file')
    result=json.loads(path.read_bytes())
    if not isinstance(result,dict):
        raise ValueError('invalid approved publisher source registry')
    return result


def _readonly(db:Path):
    if not isinstance(db,Path) or db.is_symlink() or not db.is_file():
        raise ValueError('current native FileDB/source-map missing or untrusted')
    con=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True,timeout=35)
    con.execute('PRAGMA query_only=ON')
    return con


def _one(rows:list[dict],path:str,field:str)->dict:
    found=[z for z in rows if isinstance(z,dict) and z.get(field)==path]
    if len(found)!=1:
        raise ValueError('missing/ambiguous approved original publisher catalog or acquisition receipt')
    return found[0]


def current_original_pdf_identity(*,original_path:str,original_root:Path,
                                  derived_root:Path,source_map:Path,
                                  native_db:Path,catalog:Path,
                                  completed_ledger:Path,
                                  bounds_reader:Callable|None=None)->dict:
    """Rehash original/derivative, verify the current unique source identity.

    Only an already-acquired explicit original PDF with completed acquisition
    receipt is eligible. Different raw-download and official publication URLs
    are valid and MUST remain distinct in source provenance.
    """
    if (not isinstance(original_path,str) or not 1<=len(original_path)<=1600
            or not original_path.lower().endswith('.pdf')
            or original_path.startswith('/') or '\\' in original_path
            or any(part in ('','.','..') for part in original_path.split('/'))):
        raise ValueError('expected exact safe original publisher PDF relative path')
    original=safe_file(original_root,original_path)
    if not 0<original.stat().st_size<=_MAX_ORIGINAL_BYTES:
        raise ValueError('publisher original PDF exceeds safe full-byte audit bound')
    catalog_doc=_checked_json(catalog,_MAX_CATALOG_BYTES)
    ledger_doc=_checked_json(completed_ledger,_MAX_LEDGER_BYTES)
    if not isinstance(catalog_doc.get('sources'),list) or not isinstance(ledger_doc.get('items'),list):
        raise ValueError('missing original approved source catalog or completed receipt rows')
    source=_one(catalog_doc['sources'],original_path,'path')
    acquired=_one(ledger_doc['items'],original_path,'path')
    if source.get('kind')!='pdf' or acquired.get('status')!='complete':
        raise ValueError('source is not a completed, approved publisher PDF acquisition')
    raw_url=source.get('url')
    landing=source.get('source_url')
    if (not isinstance(raw_url,str) or not raw_url.startswith('https://')
            or not isinstance(landing,str) or not landing.startswith('https://')):
        raise ValueError('original publication and acquisition URLs not verified')
    with closing(_readonly(source_map)) as sm,closing(_readonly(native_db)) as native:
        rows=sm.execute('SELECT source_sha256,derived_sha256,derived_path,original_url,'
                        'original_title,status,original_bytes,original_kind '
                        'FROM originals WHERE source_path=?',(original_path,)).fetchall()
        if len(rows)!=1 or rows[0][5]!='complete' or rows[0][7]!='pdf':
            raise ValueError('current source-map missing complete original PDF revision')
        sha,dsha,drel,url,title,_state,bytes_expected,_kind=rows[0]
        ids=native.execute("SELECT id FROM filesystem_entries "
                           "WHERE relative_path_display=? AND entry_type='file'",
                           (original_path,)).fetchall()
        if len(ids)!=1 or type(ids[0][0]) is not int or ids[0][0]<=0:
            raise ValueError('current native FileDB has missing/ambiguous original source identity')
    if (not isinstance(sha,str) or _HEX.fullmatch(sha) is None
            or not isinstance(dsha,str) or _HEX.fullmatch(dsha) is None
            or drel!=original_path+'.txt'
            or not isinstance(title,str) or not title
            or title!=source.get('title')
            or url!=landing
            or acquired.get('sha256')!=sha
            or acquired.get('bytes_received')!=bytes_expected
            or bytes_expected!=original.stat().st_size):
        raise ValueError('current publisher acquisition/source-map metadata disagrees')
    derived=safe_file(derived_root,drel)
    if not 0<derived.stat().st_size<=_MAX_DERIVED_BYTES:
        raise ValueError('original extracted PDF text exceeds bounded full-SHA audit limit')
    if sha_file(original)!=sha or sha_file(derived)!=dsha:
        raise ValueError('current publisher original or derived source bytes differ from approved revision')
    text=derived.read_bytes()
    if hashlib.sha256(text).hexdigest()!=dsha:
        raise ValueError('current original PDF derivative changed after whole-file hash')
    decoded=text.decode('utf8','strict')
    reader=bounds_reader or publisher_pdf_page_map_bounds
    bounds=reader(original,decoded)
    if (not isinstance(bounds,dict) or bounds.get('status') not in
            ('page_label_bounds_consistent_only_not_verification',
             'derived_page_labels_exceed_original_physical_pages')
            or type(bounds.get('original_physical_pdf_pages')) is not int
            or not 1<=bounds['original_physical_pdf_pages']<=_MAX_ORIGINAL_PAGES):
        raise ValueError('actual publisher PDF physical page count unavailable from source')
    return {
        'original_path':original_path,
        'source_publisher_url':landing,
        'raw_pdf_acquisition_url_separate_from_publication':raw_url,
        'original_title_metadata_not_PDF_quote':title,
        'publisher_original_sha256':sha,'current_derived_source_sha256':dsha,
        'current_native_filedb_original_id':ids[0][0],
        'publisher_actual_pdf_physical_pages':bounds['original_physical_pdf_pages'],
        'cached_extracted_page_label_bounds_not_physical_page_citation':bounds,
        'cached_page_labels_exceed_actual_pdf_physical_pages':bool(
            bounds.get('derived_pdf_page_map_suspect')),
        'current_complete_original_and_derived_full_SHA_verified':True,
        'current_acquisition_receipt_and_unique_native_identity_verified':True,
        'source_material_downloaded_before_this_query':True,
    }


def candidate_current_original_physical_glyph_page(*,original_path:str,page:int,
          query:str,original_root:Path,derived_root:Path,source_map:Path,
          native_db:Path,catalog:Path,completed_ledger:Path,
          bounds_reader:Callable|None=None,
          alternate_reader:Callable|None=None,
          ranked_reader:Callable|None=None)->dict:
    """Return a current original-bound provisional page candidate, if any.

    Only a query that explicitly names *one* publisher original PDF and one
    existing physical page is allowed. Even a second-reader successful short
    excerpt never authorizes native FileDB source/vector modification.
    """
    if type(page) is not int or not 1<=page<=_MAX_ORIGINAL_PAGES:
        raise ValueError('exact bounded current original physical PDF page is required')
    if (not isinstance(query,str) or not 3<=len(query)<=400
            or not query.strip() or any(ord(c)<32 or ord(c)==127 for c in query)):
        raise ValueError('invalid bounded source-retrieval question')
    before=current_original_pdf_identity(
        original_path=original_path,original_root=original_root,
        derived_root=derived_root,source_map=source_map,native_db=native_db,
        catalog=catalog,completed_ledger=completed_ledger,
        bounds_reader=bounds_reader)
    if page>before['publisher_actual_pdf_physical_pages']:
        raise ValueError('requested page does not exist in the current original PDF')
    original=safe_file(original_root,original_path)
    extractor=alternate_reader or extract_alternate_original_pdf_pages
    alternate=extractor(original,before['publisher_original_sha256'],pages=[page],
               expected_physical_page_count=before['publisher_actual_pdf_physical_pages'])
    if (not isinstance(alternate,dict)
            or alternate.get('schema')!='kmd.readonly_original_pdf_bounded_machine_text_candidates.v1'
            or alternate.get('original_pdf_sha256')!=before['publisher_original_sha256']
            or alternate.get('original_pdf_sha_verified_before_and_after_same_file_description') is not True
            or alternate.get('all_page_numbers_from_original_pdfinfo_not_fake_derivative_labels') is not True
            or alternate.get('machine_extracted_text_is_provisional_not_visually_verified') is not True
            or not isinstance(alternate.get('pages'),list) or len(alternate['pages'])!=1
            or alternate['pages'][0].get('physical_pdf_page')!=page):
        raise ValueError('original PDF physical-page machine extraction lacked current source proof')
    machine=alternate['pages'][0]
    if machine.get('status') not in ('source_bounded_provisional_machine_text','unavailable'):
        raise ValueError('unknown original machine extractor page status')
    if machine['status']=='source_bounded_provisional_machine_text':
        machine_text=machine.get('text')
        machine_sha=machine.get('display_text_sha256')
        if (not isinstance(machine_text,str) or not machine_text.strip()
                or '\x00' in machine_text or '\ufffd' in machine_text
                or not isinstance(machine_sha,str)
                or _HEX.fullmatch(machine_sha) is None
                or hashlib.sha256(machine_text.encode('utf8')).hexdigest()!=machine_sha):
            raise ValueError('current original PDF alternate page text changed or contains unreadable glyphs')
    elif machine.get('text') is not None:
        raise ValueError('unavailable original PDF page cannot contain candidate publisher text')
    ranker=ranked_reader or provisional_alternate_query_hits
    result=ranker(query,alternate['pages'],top=1,original_pdf=original,
                  expected_original_sha256=before['publisher_original_sha256'])
    if (not isinstance(result,dict)
            or result.get('schema')!='kmd.ephemeral_provisional_alternate_pdf_physical_page_lexical_search.v1'
            or not isinstance(result.get('accepted_provisional_ranked_pages'),list)
            or len(result['accepted_provisional_ranked_pages'])>1
            or result.get('no_factual_answers_asserted') is not True):
        raise ValueError('untrusted current original physical-page lexical candidate record')
    hits=result['accepted_provisional_ranked_pages']
    if machine['status']=='unavailable' and hits:
        raise ValueError('unavailable original physical PDF page cannot have a verified citation')
    for hit in hits:
        confirmed=hit.get('independent_poppler_same_physical_page_passage_corroborated')
        status=hit.get('status')
        excerpt=hit.get('provisional_original_page_excerpt')
        excerpt_sha=hit.get('provisional_excerpt_sha256')
        if (hit.get('physical_pdf_page')!=page
                or type(confirmed) is not bool
                or hit.get('factual_answer_verified') is not False
                or hit.get('publisher_original_visual_layout_correctness_verified') is not False
                or status not in
                ('same_page_second_extractor_exact_phrase_corroborated',
                 'provisional_alternate_machine_text_only_not_a_verified_quote')
                or confirmed != (status=='same_page_second_extractor_exact_phrase_corroborated')
                or not isinstance(excerpt,str) or not 35<=len(excerpt)<=500
                or '\x00' in excerpt or '\ufffd' in excerpt
                or not isinstance(excerpt_sha,str) or _HEX.fullmatch(excerpt_sha) is None
                or hashlib.sha256(excerpt.encode('utf8')).hexdigest()!=excerpt_sha
                or machine['status']!='source_bounded_provisional_machine_text'
                or excerpt not in machine['text']):
            raise ValueError('impossible or unverified original publisher page evidence promoted')
        physical_sha=hit.get('original_pdf_page_text_cross_extractor_sha256')
        fragment=hit.get('corroborated_short_fragment_only')
        if confirmed:
            if (fragment is not None or not isinstance(physical_sha,str)
                    or _HEX.fullmatch(physical_sha) is None):
                raise ValueError('confirmed original physical PDF excerpt lacks independent same-page SHA')
        else:
            if physical_sha is not None:
                raise ValueError('provisional original PDF excerpt cannot claim independent physical-page SHA')
            if fragment is not None:
                if (not isinstance(fragment,dict)
                        or fragment.get('physical_pdf_page')!=page
                        or fragment.get('basis')!='two_independent_machine_extractors_exact_same_original_physical_page_phrase'
                        or fragment.get('full_machine_page_and_visual_layout_not_verified') is not True
                        or not isinstance(fragment.get('text'),str)
                        or not 35<=len(fragment['text'])<=480
                        or fragment['text'] not in machine['text']
                        or not isinstance(fragment.get('fragment_sha256'),str)
                        or _HEX.fullmatch(fragment['fragment_sha256']) is None
                        or hashlib.sha256(fragment['text'].encode('utf8')).hexdigest()!=fragment['fragment_sha256']
                        or not isinstance(fragment.get('original_pdf_poppler_physical_page_sha256'),str)
                        or _HEX.fullmatch(fragment['original_pdf_poppler_physical_page_sha256']) is None):
                    raise ValueError('unverified or changed short publisher-original physical-page fragment')
    after=current_original_pdf_identity(
        original_path=original_path,original_root=original_root,
        derived_root=derived_root,source_map=source_map,native_db=native_db,
        catalog=catalog,completed_ledger=completed_ledger,
        bounds_reader=bounds_reader)
    if before!=after:
        raise ValueError('approved original source/native identity changed during physical page query')
    quality={k:v for k,v in machine.items() if k!='text'}
    output={
        'schema':'kmd.optional_current_approved_unstaged_original_pdf_glyph_query.v1',
        'original':before,'physical_page_selected':page,
        'query_sha256':hashlib.sha256(query.encode()).hexdigest(),
        'source_page_machine_text_quality_metadata_only':quality,
        'ranked_same_page_source_candidates':result,
        'exact_current_original_publisher_and_derived_SHA_and_native_ID_rechecked_twice':True,
        'full_corresponding_ghostscript_page_not_source_verified_by_one_match':True,
        'possible_original_PDF_cross_reader_verification_only_for_individual_excerpt':True,
        'not_an_independent_physical_figure_visual_or_factual_truth_verdict':True,
        'not_an_approved_source_cutover_or_new_native_index':True,
        'no_new_source_acquisition_OCR_or_external_network_model':True,
        'no_original_or_extracted_source_file_modified':True,
        'no_atomic_generation_across_independent_live_sqlite_stores':True,
    }
    # Catch any accidental regex Match/Path or other opaque object leaking
    # through successful source corroboration before it reaches the CLI file.
    json.dumps(output,ensure_ascii=False)
    return output
