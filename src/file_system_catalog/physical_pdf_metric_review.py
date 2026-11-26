"""Physical original-PDF numeric table-cell and nearby model-label review.

Read-only evidence. Geometric proximity is NOT a statistical attribution,
experiment, semantic support, or scientific correctness verdict. The source
must already be an approved, current native original before this is called.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Any

from .publisher_pdf_page import _poppler

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_ALLOWED_VALUE = re.compile(r"[0-9]{1,3}(?:\.[0-9]{1,3})?\Z")
_TSV_HEADERS = ("level", "page_num", "par_num", "block_num", "line_num", "word_num",
                "left", "top", "width", "height", "conf", "text")
_STRIP = "*∗†,.;:()[]"


@dataclass(frozen=True)
class Word:
    text: str
    x: float
    y: float
    width: float
    height: float
    page: int

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def right(self) -> float:
        return self.x + self.width


def validate_model_labels(labels: list[str]) -> list[str]:
    if (not isinstance(labels, list) or not 1 <= len(labels) <= 12
            or any(not isinstance(s, str) or not 2 <= len(s) <= 85
                   or s != s.strip() or "  " in s or not any(c.isalnum() for c in s)
                   or any(ord(c) < 32 or ord(c) == 127 for c in s)
                   for s in labels)):
        raise ValueError("invalid bounded explicit model labels")
    if len({label.casefold() for label in labels}) != len(labels):
        raise ValueError("duplicate explicit model labels")
    return labels


def validate_numeric_values(values: list[str]) -> list[str]:
    if (not isinstance(values, list) or not 1 <= len(values) <= 16
            or any(not isinstance(v, str) or _ALLOWED_VALUE.fullmatch(v) is None
                   for v in values)
            or len(set(values)) != len(values)):
        raise ValueError("invalid bounded exact table numeric values")
    return values


def parse_physical_pdf_tsv(raw: bytes, *, page: int, max_words: int = 25_000) -> tuple[list[Word], dict]:
    """Bounded original Poppler word positions, no inferred row semantics."""
    if (not isinstance(raw, bytes) or len(raw) > 8*1024*1024
            or not isinstance(page, int) or isinstance(page, bool)
            or not 1 <= page <= 200_000 or not 1 <= max_words <= 25_000):
        raise ValueError("invalid bounded physical PDF TSV")
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise ValueError("publisher PDF TSV is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if tuple(reader.fieldnames or ()) != _TSV_HEADERS:
        raise ValueError("unrecognized publisher PDF TSV word schema")
    words: list[Word] = []
    size = None
    for row in reader:
        if not row or any(key not in row or row[key] is None for key in _TSV_HEADERS):
            raise ValueError("malformed physical PDF TSV row")
        if row["level"] == "1":
            if size is not None or row["page_num"] != str(page) or row["text"] != "###PAGE###":
                raise ValueError("physical PDF TSV has multiple or wrong page markers")
            w, h = float(row["width"]), float(row["height"])
            if not all(map(math.isfinite, (w,h))) or not 50 <= w <= 20_000 or not 50 <= h <= 20_000:
                raise ValueError("invalid publisher page bounds")
            size = (w,h)
        elif row["level"] == "5":
            if size is None or row["page_num"] != str(page):
                raise ValueError("word precedes publisher PDF page identity")
            if len(words) >= max_words:
                raise ValueError("publisher page exceeds bounded word count")
            x,y,w,h = (float(row[k]) for k in ("left", "top", "width", "height"))
            if (not all(map(math.isfinite,(x,y,w,h)))
                    or not 0 <= x <= size[0] or not 0 <= y <= size[1]
                    or not 0 < w <= size[0] or not 0 < h <= size[1]
                    or x+w > size[0]+2 or y+h > size[1]+2
                    or not row["text"] or len(row["text"]) > 400):
                raise ValueError("invalid bounded publisher word geometry")
            words.append(Word(row["text"], x,y,w,h,page))
        elif row["level"] not in ("3", "4"):
            raise ValueError("unexpected publisher PDF TSV level")
    if size is None:
        raise ValueError("missing original physical page marker")
    return words, {"publisher_physical_page":page,"page_width_pt":size[0],
                   "page_height_pt":size[1],"pdf_tsv_sha256":hashlib.sha256(raw).hexdigest(),
                   "word_boxes_parsed":len(words),"word_text_is_untrusted_source_data":True}


def read_hashed_original_pdf_page_tsv(path: Path, *, source_sha256: str, page: int) -> tuple[list[Word], dict]:
    """Hash the same opened PDF file descriptor used by bounded Poppler."""
    if (not isinstance(path,Path) or not isinstance(source_sha256,str)
            or _HEX.fullmatch(source_sha256) is None
            or type(page) is not int or not 1 <= page <= 200_000
            or path.suffix.lower() != ".pdf"):
        raise ValueError("invalid SHA-pinned original physical PDF request")
    flags=os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
    with os.fdopen(os.open(path,flags),"rb") as file:
        before=os.fstat(file.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_size > 256*1024*1024
                or hashlib.file_digest(file,"sha256").hexdigest()!=source_sha256):
            raise ValueError("publisher original PDF SHA differs from current approved revision")
        fd=file.fileno()
        name=f"/proc/self/fd/{fd}"
        deadline=time.monotonic()+20
        info=_poppler(["/usr/bin/pdfinfo",name],fd,deadline)
        counts=re.findall(rb"(?m)^Pages:[ \t]+([1-9][0-9]{0,5})[ \t]*$",info or b"")
        if len(counts)!=1:
            raise ValueError("publisher physical PDF page count unavailable")
        pages=int(counts[0])
        if page>pages:
            raise ValueError("requested physical PDF page exceeds current original")
        data=_poppler(["/usr/bin/pdftotext","-f",str(page),"-l",str(page),
                       "-tsv","-enc","UTF-8",name,"-"],fd,deadline)
        if data is None:
            raise ValueError("bounded original PDF TSV extraction unavailable")
        words,geometry=parse_physical_pdf_tsv(data,page=page)
        after=path.stat(follow_symlinks=False)
        if (not os.path.samestat(before,after) or before.st_size!=after.st_size
                or before.st_mtime_ns!=after.st_mtime_ns or before.st_ctime_ns!=after.st_ctime_ns):
            raise ValueError("publisher original PDF changed during physical page review")
    return words,{**geometry,"publisher_original_sha256":source_sha256,
                  "original_physical_pdf_pages":pages,
                  "source_file_same_open_fd_sha256_verified":True,
                  "all_source_table_text_from_original_physical_pdf_text_layer":True}


def _name_tokens(name: str) -> list[str]:
    return [word.casefold().strip(_STRIP) for word in name.split()]


def _locate_names(words:list[Word],names:list[str])->list[dict]:
    output=[]
    # Scan original reading order but require physical proximity between
    # adjacent words. Poppler can split words across blocks, so do not rely
    # on its block-number to prove table attribution.
    for name in names:
        tokens=_name_tokens(name)
        for i in range(len(words)-len(tokens)+1):
            cand=words[i:i+len(tokens)]
            if [w.text.casefold().strip(_STRIP) for w in cand]!=tokens:
                continue
            if any(abs(a.center_y-b.center_y)>6.5 or not -1<=b.x-a.right<=20
                   for a,b in zip(cand,cand[1:])):
                continue
            output.append({'model':name,
                           'original_source_label_as_printed':' '.join(w.text for w in cand),
                           'model_box_left_pt':round(cand[0].x,3),
                           'model_box_right_pt':round(cand[-1].right,3),
                           'model_box_center_y_pt':round(sum(w.center_y for w in cand)/len(cand),3),
                           'source_model_label_original_text_sha256':hashlib.sha256(
                               ' '.join(w.text for w in cand).encode('utf8')).hexdigest()})
    return output


def review_original_table_numeric_cells(*,words:list[Word],page:int,
                                        model_labels:list[str],numeric_values:list[str],
                                        explicit_metric_headings:list[str]|None=None,
                                        max_row_gap_pt:float=8.5,
                                        max_left_distance_pt:float=155)->dict[str,Any]:
    """Spatial association candidates only; never an automatic false-claim verdict."""
    validate_model_labels(model_labels)
    if not isinstance(numeric_values,list) or len(numeric_values)>16:
        raise ValueError("invalid bounded exact table numeric values")
    if numeric_values:
        validate_numeric_values(numeric_values)
    if explicit_metric_headings is None:
        explicit_metric_headings=[]
    elif explicit_metric_headings:
        validate_model_labels(explicit_metric_headings)
    elif not isinstance(explicit_metric_headings,list):
        raise ValueError("invalid bounded source column headers")
    if (not isinstance(words,list) or len(words)>25_000
            or type(page) is not int or not 1<=page<=200_000
            or not 4<=max_row_gap_pt<=15
            or not 30<=max_left_distance_pt<=200):
        raise ValueError("unbounded original numeric table geometry")
    if any(not isinstance(w,Word) or w.page!=page for w in words):
        raise ValueError("physical table word/page identity mismatch")
    models=_locate_names(words,model_labels)
    headers=_locate_names(words,explicit_metric_headings)
    output=[]
    for value in numeric_values:
        cells=[w for w in words if w.text.strip(_STRIP)==value]
        for cell in cells[:12]:
            spatial=[]
            for m in models:
                dy=abs(cell.center_y-m['model_box_center_y_pt'])
                gap=cell.x-m['model_box_right_pt']
                if dy>max_row_gap_pt or not 0<=gap<=max_left_distance_pt:
                    continue
                spatial.append({**m,
                                'vertical_center_gap_pt':round(dy,3),
                                'left_label_to_numeric_horizontal_gap_pt':round(gap,3),
                                'same_physical_pdf_row_geometrically_plausible':True,
                                'does_not_prove_same_metric_or_column_heading':True})
            spatial.sort(key=lambda x:(x['vertical_center_gap_pt'],x['left_label_to_numeric_horizontal_gap_pt'],x['model']))
            distant=[]
            for m in models:
                dy=abs(cell.center_y-m['model_box_center_y_pt'])
                gap=cell.x-m['model_box_right_pt']
                if dy>max_row_gap_pt or not max_left_distance_pt<gap<=2000:
                    continue
                distant.append({**m,
                                'vertical_center_gap_pt':round(dy,3),
                                'left_label_to_numeric_horizontal_gap_pt':round(gap,3),
                                'model_is_too_far_left_for_direct_cell_attribution':True,
                                'same_horizontal_row_alone_cannot_prove_shared_table_or_metric':True})
            distant.sort(key=lambda x:(x['vertical_center_gap_pt'],x['left_label_to_numeric_horizontal_gap_pt'],x['model']))
            # Ignore a broad alias if the same row has a more precise
            # longer label whose printed source text begins with the alias,
            # e.g. T-Mem vs T-Mem, reprod. Keep the original candidates
            # visible to expose why a selection was preferred.
            most_specific=[m for m in spatial if not any(
                n['model']!=m['model'] and
                n['model'].casefold().startswith(m['model'].casefold()) and
                n['model_box_left_pt']==m['model_box_left_pt'] and
                abs(n['model_box_center_y_pt']-m['model_box_center_y_pt'])<2.5
                for n in spatial)]
            unique_names={m['model'] for m in most_specific}
            heading_candidates=[]
            for heading in headers:
                cell_x=cell.x+cell.width/2
                head_x=(heading['model_box_left_pt']+heading['model_box_right_pt'])/2
                dy=cell.y-heading['model_box_center_y_pt']
                dx=abs(cell_x-head_x)
                if not 0<=dy<=300 or dx>125:
                    continue
                heading_candidates.append({
                    'explicit_metric_header':heading['model'],
                    'original_pdf_heading_as_printed':heading['original_source_label_as_printed'],
                    'heading_source_text_sha256':heading['source_model_label_original_text_sha256'],
                    'heading_box_center_x_pt':round(head_x,3),
                    'heading_box_center_y_pt':heading['model_box_center_y_pt'],
                    'header_to_number_horizontal_center_gap_pt':round(dx,3),
                    'header_above_numeric_cell_vertical_gap_pt':round(dy,3),
                    'geometric_header_not_verified_metric_column_truth':True,
                })
            # Repeated words in the abstract and the actual table can share
            # a header name. Prefer the closest preceding occurrence in Y;
            # retain only its strongest candidate for each EXPLICIT label.
            # This is a geometric hint, not a metric-unit assertion.
            per_heading={}
            for h in heading_candidates:
                key=h['explicit_metric_header']
                rank=(h['header_above_numeric_cell_vertical_gap_pt'],
                      h['header_to_number_horizontal_center_gap_pt'])
                if key not in per_heading or rank < per_heading[key][0]:
                    per_heading[key]=(rank,h)
            heading_candidates=[pair[1] for pair in per_heading.values()]
            heading_candidates.sort(key=lambda h:(h['header_above_numeric_cell_vertical_gap_pt'],
                                                  h['header_to_number_horizontal_center_gap_pt'],
                                                  h['explicit_metric_header']))
            best_heading=(heading_candidates[0]['explicit_metric_header'] if heading_candidates
                          and (len(heading_candidates)==1 or
                               (heading_candidates[1]['header_above_numeric_cell_vertical_gap_pt']
                                -heading_candidates[0]['header_above_numeric_cell_vertical_gap_pt']>12
                                or heading_candidates[1]['header_to_number_horizontal_center_gap_pt']
                                   -heading_candidates[0]['header_to_number_horizontal_center_gap_pt']>12))
                          else None)
            output.append({'number':value,'source_number_text':cell.text,
                           'source_number_text_sha256':hashlib.sha256(cell.text.encode('utf8')).hexdigest(),
                           'physical_original_pdf_page':page,
                           'numeric_cell_box':{'left_pt':round(cell.x,3),'top_pt':round(cell.y,3),
                                               'width_pt':round(cell.width,3),'height_pt':round(cell.height,3)},
                           'geometrically_near_named_models':spatial[:8],
                           'most_specific_named_model_candidates':most_specific[:8],
                           'distant_same_row_named_models_not_attribution_proof':distant[:8],
                           'unique_distant_named_model_if_no_direct_match':(
                               distant[0]['model']
                               if not spatial and len({m['model'] for m in distant})==1
                               else None),
                           'unique_explicit_named_model_in_same_row_if_any':(
                               next(iter(unique_names)) if len(unique_names)==1 else None),
                           'explicit_metric_heading_geometric_candidates':heading_candidates[:12],
                           'unique_nearest_explicit_metric_heading_geometric_candidate':best_heading,
                           'metric_heading_closeness_does_NOT_prove_metric_or_experiment':True,
                           'numeric_cell_model_attribution_factual_verdict':'NOT_ASSESSED',
                           'source_table_column_heading_and_experiment_not_automatically_verified':True,
                           'geometrically_near_label_is_a_review_hint_only':True})
    return {'schema':'kmd.physical_publisher_pdf_table_numeric_row_geometry_review.v1',
            'physical_original_pdf_page':page,
            'reviewed_model_label_candidates':model_labels,
            'reviewed_explicit_metric_heading_candidates':explicit_metric_headings,
            'numeric_values_queried':numeric_values,
            'no_generated_percentage_or_explicit_numeric_cell_to_review':not numeric_values,
            'model_label_boxes_present':len(models),
            'numeric_cells_found':len(output),
            'numeric_cell_candidates':output,
            'all_proximities_unverified_semantic_correspondence':True,
            'never_automatic_source_fact_or_corrected_native_summary':True,
            'source_text_does_not_include_original_figures_or_full_visual_table_grid':True}
