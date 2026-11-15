"""Private All2Text image crops link to the one native KMD original and text."""
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from file_system_catalog.pdf_visual_link import (
    link_pdf_regions, _caption_anchor, _fold_with_positions,
    _safe_rel, safe_private_result_path,
)


def sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path:Path)->str:
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


class NativeVisualSourceLinkTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.source=self.root/"original"
        self.source.mkdir()
        self.relative="papers/example.pdf"
        self.original=self.source/self.relative
        self.original.parent.mkdir()
        self.original.write_bytes(b"%PDF-1.4\nLocally verified mock original\n%%EOF\n")
        self.source_sha=hash_file(self.original)
        self.text_root=self.root/"derived"
        self.text_root.mkdir()
        self.derived_rel="papers/example.pdf.txt"
        self.derived=self.text_root/self.derived_rel
        self.derived.parent.mkdir()
        self.caption=("Figure 9: Two rendered material comparison panels with "
                      "roughness increasing from left to right.")
        self.body=("[[SOURCE UNIT: PDF page 1]]\n"
                   "This is the publisher source's first physical page.\n"
                   + self.caption+"\n")
        self.derived.write_text(self.body,encoding="utf-8")
        self.derived_sha=hash_file(self.derived)
        self.mapdb=self.root/"source-map.sqlite3"
        self.filedb=self.root/"native.sqlite3"
        with sqlite3.connect(self.mapdb) as db:
            db.execute("CREATE TABLE originals (source_path TEXT,status TEXT,source_sha256 TEXT,derived_sha256 TEXT,derived_path TEXT,error TEXT)")
            db.execute("INSERT INTO originals VALUES (?,?,?,?,?,NULL)",
                 (self.relative,"complete",self.source_sha,self.derived_sha,self.derived_rel))
        with sqlite3.connect(self.filedb) as db:
            db.execute("CREATE TABLE filesystem_entries (id INTEGER,relative_path_display TEXT,entry_type TEXT)")
            db.execute("CREATE TABLE content_chunks (filesystem_entry_id INTEGER,collection_id TEXT,embedding_blob BLOB)")
            db.execute("INSERT INTO filesystem_entries VALUES (101,?, 'file')",(self.relative,))
            db.execute("INSERT INTO content_chunks VALUES (101,'semantic:overview',?)",(b'fake-vector',))
        self.output=self.root/"all2text_output"
        self.output.mkdir()
        (self.output/"regions").mkdir()
        (self.output/"rendered_pages").mkdir()
        p=Image.new("RGB",(200,200),"white")
        dr=ImageDraw.Draw(p);dr.rectangle((10,10,110,110),fill="red")
        self.page=self.output/"rendered_pages/page_0001.png"
        self.page_crop=self.output/"regions/page_0001_region_01.png"
        p.save(self.page)
        p.crop((10,10,110,110)).save(self.page_crop)
        self.region={"source_relative_path":self.relative,
             "original_pdf_sha256":self.source_sha,"physical_pdf_page":1,
             "region_path":"regions/page_0001_region_01.png",
             "region_png_sha256":hash_file(self.page_crop),
             "rendered_page_path":"rendered_pages/page_0001.png",
             "rendered_page_sha256":hash_file(self.page),
             "raster_region_pixel_bbox_top_left":[10,10,110,110],
             "region_role":"figure_crop",
             "caption_text_for_prompt":self.caption,
             "caption_preview_sha256":sha(self.caption.encode())}
        self.region_file=self.output/"regions.json"
        self.context_file=self.output/"visual_context.json"
        self.write_files()

    def write_files(self):
        self.region_file.write_text(json.dumps({
          "schema":"all2text.verified_pdf_visual_regions.v1",
          "original":{"relative_path":self.relative,
                      "source_sha256":self.source_sha,
                      "download_receipt_sha256":self.source_sha},
          "pages":[{"physical_pdf_page":1}],
          "regions":[self.region]}))
        self.context_file.write_text(json.dumps({
            "schema":"all2text.visual_context.v1",
            "entries":{self.page_crop.name:{
                "image_sha256":hash_file(self.page_crop),
                "parent_pdf_sha256":self.source_sha,
                "physical_page":1,"region_role":"figure_crop"}}}))

    def link(self,**changes):
        args={"region_manifest":self.region_file,
              "context_manifest":self.context_file,
              "source_root":self.source,
              "source_map_db":self.mapdb,
              "filedb":self.filedb,
              "derived_root":self.text_root,
              "independently_rerender":False}
        args.update(changes)
        return link_pdf_regions(**args)

    def test_exact_page_caption_derivative_and_native_original_identity(self):
        before=(hash_file(self.mapdb),hash_file(self.filedb),hash_file(self.original))
        report=self.link()
        self.assertEqual(self.relative,report["source_relative_path"])
        self.assertEqual(self.source_sha,report["source_original_sha256"])
        self.assertEqual(self.derived_sha,report["current_derived_sha256"])
        self.assertEqual("complete",report["source_map_status"])
        self.assertEqual(101,report["native_filesystem_entry_id"])
        self.assertEqual([{"collection_id":"semantic:overview","chunk_count":1,
                           "embedded_chunk_count":1}],report["native_indexed_collections"])
        proof=report["regions"][0]["caption_anchor"]
        self.assertEqual("verified_original_derived_page_caption_prefix",proof["status"])
        self.assertEqual("exact_current_extracted_text_span_not_original_publisher_bytes",
                         proof["evidence_scope"])
        span=self.body[proof["derived_start_char"]:proof["derived_end_char"]]
        self.assertEqual(sha(span.encode()),proof["derived_prefix_sha256"])
        self.assertEqual(self.caption.casefold()[:96],proof["normalized_prefix"])
        self.assertEqual(before,(hash_file(self.mapdb),hash_file(self.filedb),
                                 hash_file(self.original)))
        self.assertFalse(report["regions"][0]["independent_physical_pdf_page_raster_verified"])

    def test_rerender_called_independently_in_production_mode(self):
        with patch("file_system_catalog.pdf_visual_link._rerender_original_page",
                   return_value=True) as render:
            result=self.link(independently_rerender=True)
        self.assertEqual(1,render.call_count)
        self.assertTrue(result["regions"][0]["independent_physical_pdf_page_raster_verified"])
        with patch("file_system_catalog.pdf_visual_link._rerender_original_page",
                   return_value=False):
            with self.assertRaisesRegex(ValueError,"did not reproduce"):
                self.link(independently_rerender=True)

    def test_wrong_page_cannot_inherit_exact_caption_from_another_page(self):
        self.body=("[[SOURCE UNIT: PDF page 2]]\n"+self.caption+"\n")
        self.derived.write_text(self.body)
        with sqlite3.connect(self.mapdb) as c:
            c.execute("UPDATE originals SET derived_sha256=?",(hash_file(self.derived),))
        r=self.link()["regions"][0]["caption_anchor"]
        self.assertEqual("physical_page_marker_absent_or_ambiguous",r["status"])

    def test_duplicated_caption_in_page_abstains(self):
        self.body=("[[SOURCE UNIT: PDF page 1]]\n"+self.caption+"\n"+
                   "Other explanation. "+self.caption+"\n")
        self.derived.write_text(self.body)
        with sqlite3.connect(self.mapdb) as c:
            c.execute("UPDATE originals SET derived_sha256=?",(hash_file(self.derived),))
        anchor=self.link()["regions"][0]["caption_anchor"]
        self.assertEqual("caption_prefix_missing_or_ambiguous_in_page",anchor["status"])
        self.assertEqual(2,anchor["occurrence_count"])

    def test_scanned_source_original_can_link_without_completed_text(self):
        with sqlite3.connect(self.mapdb) as c:
            c.execute("UPDATE originals SET status='error',derived_sha256=NULL,derived_path=NULL")
        self.region["caption_text_for_prompt"]=""
        self.region["caption_preview_sha256"]=None
        self.region["region_role"]="full_document_page"
        self.write_files()
        payload=json.loads(self.context_file.read_text())
        payload["entries"][self.page_crop.name]["region_role"]="full_document_page"
        self.context_file.write_text(json.dumps(payload))
        report=self.link()
        self.assertEqual("error",report["source_map_status"])
        self.assertIsNone(report["current_derived_sha256"])
        self.assertEqual(101,report["native_filesystem_entry_id"])
        self.assertEqual("no_completed_text_extraction",report["regions"][0]["caption_anchor"]["status"])

    def test_revision_mismatch_and_crop_tampering_rejected(self):
        self.original.write_bytes(self.original.read_bytes()+b'altered')
        with self.assertRaisesRegex(ValueError,"original revision"):
            self.link()
        self.original.write_bytes(self.original.read_bytes()[:-7])
        assert hash_file(self.original)==self.source_sha
        self.page_crop.write_bytes(self.page_crop.read_bytes()+b'BAD')
        with self.assertRaisesRegex(ValueError,"crop hash"):
            self.link()

    def test_derived_revision_mismatch_and_invented_native_identity_rejected(self):
        self.derived.write_text(self.body+"changed")
        with self.assertRaisesRegex(ValueError,"derived source SHA"):
            self.link()
        self.derived.write_text(self.body)
        with sqlite3.connect(self.filedb) as db:
            db.execute("INSERT INTO filesystem_entries VALUES (102,?,'file')",(self.relative,))
        with self.assertRaisesRegex(ValueError,"not unique"):
            self.link()

    def test_caption_tampering_and_wrong_context_rejected(self):
        self.region["caption_text_for_prompt"]="invented source statement"
        self.write_files()
        with self.assertRaisesRegex(ValueError,"caption preview"):
            self.link()
        self.region["caption_text_for_prompt"]=self.caption
        self.write_files()
        payload=json.loads(self.context_file.read_text())
        payload["entries"][self.page_crop.name]["physical_page"]=2
        self.context_file.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError,"another original/page"):
            self.link()

    def test_output_guards_keep_reports_outside_originals_and_database(self):
        audit=self.root/"runtime"/"search-audit"/"proofs"
        audit.mkdir(parents=True)
        native_dir=self.root/"runtime"
        real_filedb=native_dir/"filedb.sqlite3"
        source_map=native_dir/"source-map.sqlite3"
        real_filedb.write_bytes(b"database file")
        source_map.write_bytes(b"source map")
        def checked(where):
            return safe_private_result_path(
                where,source_root=self.source,
                derived_root=self.text_root,native_filedb=real_filedb,
                source_map_db=source_map,
            )
        self.assertEqual(audit/"new.json",checked(audit/"new.json"))
        with self.assertRaisesRegex(ValueError,"inside protected"):
            checked(self.source/"unwanted.json")
        with self.assertRaisesRegex(ValueError,"inside protected"):
            checked(self.text_root/"unwanted.json")
        with self.assertRaisesRegex(ValueError,"inside protected"):
            checked(native_dir/"unwanted.json")
        alias=self.root/"symlink-to-originals"
        alias.symlink_to(self.source,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,"inside protected"):
            checked(alias/"unwanted.json")
        existing=audit/"existing.json"
        existing.write_text("original result")
        with self.assertRaisesRegex(ValueError,"overwrite"):
            checked(existing)
        with self.assertRaisesRegex(ValueError,"missing"):
            checked(self.root/"unknown"/"unwanted.json")
        self.assertEqual("original result",existing.read_text())

    def test_safe_relative_path_and_fold_map(self):
        for bad in ('../x.pdf','/etc/password.pdf','one\\two.pdf','x/../a.pdf'):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                _safe_rel(bad,'.pdf')
        literal="FIGURE   2:  Some\nCaption  Data."
        folded,offsets=_fold_with_positions(literal)
        self.assertEqual("figure 2: some caption data.",folded)
        self.assertEqual(len(folded),len(offsets))
        self.assertEqual(literal[offsets[0]],"F")


if __name__=="__main__":
    unittest.main()
