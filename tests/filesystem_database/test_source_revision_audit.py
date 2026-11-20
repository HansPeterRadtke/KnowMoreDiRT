"""Reconcile complete native metadata and selected byte revisions without writes."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from file_system_catalog.source_revision_audit import (
    _verified_derived_body_count, _path, _digest_stable,
    deterministic_sample, audit, private_audit_destination,
)


def hash_bytes(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


class TestSourceRevisionAudit(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)
        self.root=root
        self.original=root/"originals";self.original.mkdir()
        self.derived=root/"derived";self.derived.mkdir()
        self.mapdb=root/"source-map.sqlite3"
        self.filedb=root/"filesystem.sqlite3"
        self.catalog=root/"approved.json"
        self.ledger=root/"completed.json"
        self.cats=[]
        self.receipts=[]
        self.sources=[]
        with sqlite3.connect(self.mapdb) as c:
            c.execute("CREATE TABLE originals (source_path TEXT,original_kind TEXT,status TEXT,source_sha256 TEXT,original_bytes INTEGER,derived_path TEXT,derived_sha256 TEXT,text_characters INTEGER,original_title TEXT,original_url TEXT)")
        with sqlite3.connect(self.filedb) as c:
            c.execute("CREATE TABLE filesystem_entries (id INTEGER, relative_path_display TEXT, size_bytes INTEGER, content_sha256 TEXT, hash_status TEXT, entry_type TEXT)")
            c.execute("CREATE TABLE content_chunks (filesystem_entry_id INTEGER,collection_id TEXT)")
        self.add("ref/excerpt.html","html",b"<html>Source verified evidence.</html>",
                 "HTML original publisher document",status="complete",
                 summary=True)
        self.add("ref/methods.pdf","pdf",b"%PDF-1.7\nPUBLISHER ORIGINAL\n%%EOF",
                 "PDF page 1",status="complete",summary=False)
        self.add("ref/scan.pdf","pdf",b"%PDF-1.7\nSCANNED PAGE\n%%EOF",
                 "PDF page 1",status="error",summary=False)
        self.write_inputs()

    def add(self,name,kind,source,unit,*,status="complete",summary=False):
        p=self.original/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(source)
        sha=hash_bytes(source)
        title="Original "+name.split("/")[-1]
        url="https://publisher.example/"+name
        self.cats.append({"path":name,"kind":kind,"title":title,"url":url})
        self.receipts.append({"path":name,"status":"complete","sha256":sha,
                              "bytes_received":len(source)})
        derivative=None
        derivative_sha=None
        characters=0
        if status=="complete":
            body="\n\n[[SOURCE UNIT: "+unit+"]]\n"+"Quoted source text here.\n"
            header={
                "source_path":name,"source_sha256":sha,"title":title,
                "publisher_url":url,"kind":kind,"publication_year":2026,
                "rights_note":"For personal study",
            }
            synthetic="KMD source provenance (the original PDF/HTML/ZIP/XLSX, not a new publisher book):\n"
            synthetic+=json.dumps(header,ensure_ascii=False)+"\n\n"+body
            derivative=name+".txt"
            file=self.derived/derivative
            file.parent.mkdir(parents=True,exist_ok=True)
            file.write_bytes(synthetic.encode())
            derivative_sha=hash_bytes(file.read_bytes())
            characters=len(body)
        self.sources.append({"source_path":name,"kind":kind,"status":status,
                             "source_sha256":sha,"original_bytes":len(source),
                             "derived_path":derivative,"derived_sha256":derivative_sha,
                             "text_characters":characters,"title":title,"url":url,
                             "summarized":summary})
        with sqlite3.connect(self.mapdb) as c:
            c.execute("INSERT INTO originals VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (name,kind,status,sha,len(source),derivative,derivative_sha,
                       characters,title,url))
        nativeid=len(self.sources)
        with sqlite3.connect(self.filedb) as c:
            c.execute("INSERT INTO filesystem_entries VALUES (?,?,?,?,?,?)",
                      (nativeid,name,len(source),None,"skipped_large","file"))
            if status=="complete":
                c.execute("INSERT INTO content_chunks VALUES (?,?)",
                          (nativeid,"collection:overview"))
                if summary:
                    c.execute("INSERT INTO content_chunks VALUES (?,?)",
                              (nativeid,"collection:summary"))

    def write_inputs(self):
        self.catalog.write_text(json.dumps({"sources":self.cats},ensure_ascii=False))
        self.ledger.write_text(json.dumps({"items":self.receipts},ensure_ascii=False))

    def run_audit(self,**kwargs):
        params=dict(source_root=self.original,derived_root=self.derived,
                    approved_catalog=self.catalog,completed_ledger=self.ledger,
                    source_map_db=self.mapdb,native_filedb=self.filedb,
                    quotas={"html":1,"pdf":1},max_original_bytes=1024,
                    max_derived_bytes=4096,max_total_bytes=10000)
        params.update(kwargs)
        return audit(**params)

    def test_complete_metadata_and_file_byte_checks_are_independent(self):
        before=[hash_bytes(x.read_bytes()) for x in
                (self.catalog,self.ledger,self.mapdb,self.filedb)]
        result=self.run_audit()
        m=result["summary"]
        self.assertEqual(3,m["curated_source_map_originals"])
        self.assertEqual(3,m["catalog_entries"])
        self.assertEqual(3,m["ledger_entries"])
        self.assertEqual(0,m["metadata_problem_categories"])
        self.assertEqual(0,m["selected_originals_with_failures"])
        self.assertEqual(3,m["deterministic_selected_originals"])
        self.assertEqual(3,m["originals_independently_hashed"])
        self.assertEqual(2,m["current_derivatives_independently_hashed"])
        self.assertEqual(2,m["publisher_and_derivative_jointly_byte_verified"])
        self.assertEqual(0,m["derivatives_verified_without_original_rehash"])
        self.assertEqual(2,m["native_overview_originals"])
        self.assertEqual(1,m["native_summary_originals"])
        self.assertEqual("ok",m["filedb_quick_check"])
        self.assertEqual(0,m["filedb_foreign_key_fault_count"])
        self.assertTrue(result["full_metadata_reconciled_not_full_file_content_hashed"])
        self.assertEqual(before,[hash_bytes(x.read_bytes()) for x in
                               (self.catalog,self.ledger,self.mapdb,self.filedb)])
        for row in result["samples"]:
            self.assertFalse(row["failures"])

    def test_embedded_metadata_preamble_is_not_publisher_text(self):
        p=self.derived/"ref/excerpt.html.txt"
        text=p.read_bytes().decode()
        src=next(x for x in self.sources if x["source_path"]=="ref/excerpt.html")
        count=_verified_derived_body_count(
            text,original_path=src["source_path"],
            original_sha=src["source_sha256"],source_kind="html",
        )
        self.assertEqual(src["text_characters"],count)
        self.assertGreater(len(text),count)
        with self.assertRaisesRegex(ValueError,"different source revision"):
            _verified_derived_body_count(
                text,original_path="ref/another.html",
                original_sha=src["source_sha256"],source_kind="html",
            )
        with self.assertRaisesRegex(ValueError,"missing canonical"):
            _verified_derived_body_count("just publisher text",
                original_path=src["source_path"],original_sha=src["source_sha256"],
                source_kind="html")

    def test_changed_original_file_is_reported_not_repaired(self):
        p=self.original/"ref/methods.pdf"
        p.write_bytes(p.read_bytes()+b"CHANGED")
        data=self.run_audit()
        affected=[x for x in data["samples"] if x["path"]=="ref/methods.pdf"]
        self.assertEqual(1,len(affected))
        self.assertFalse(affected[0]["original_file_revision_verified"])
        self.assertIn("publisher_sha_or_completed_receipt_mismatch",affected[0]["failures"])
        self.assertEqual(1,data["summary"]["selected_originals_with_failures"])
        self.assertEqual(0,data["summary"]["metadata_problem_categories"])

    def test_changed_derivative_byte_and_changed_embedded_publisher_identity(self):
        p=self.derived/"ref/excerpt.html.txt"
        original=p.read_bytes()
        p.write_bytes(original+b"CHANGED")
        report=self.run_audit()
        affected=next(x for x in report["samples"] if x["path"]=="ref/excerpt.html")
        self.assertIn("derived_sha_differs_from_source_map",affected["failures"])
        # A derivative whose outer checksum was also refreshed, but whose
        # internal publisher SHA was forged, MUST still fail the check.
        p.write_bytes(original.replace(b"ref/excerpt.html",b"ref/otherxx.html",1))
        dsha=hash_bytes(p.read_bytes())
        with sqlite3.connect(self.mapdb) as c:
            c.execute("UPDATE originals SET derived_sha256=? WHERE source_path='ref/excerpt.html'",
                      (dsha,))
        changed=self.run_audit()
        row=next(x for x in changed["samples"] if x["path"]=="ref/excerpt.html")
        self.assertIn("derived_revision_unverifiable:ValueError",row["failures"])

    def test_missing_native_row_and_mismatched_url_are_detected_before_sampling(self):
        with sqlite3.connect(self.filedb) as c:
            c.execute("DELETE FROM filesystem_entries WHERE relative_path_display='ref/excerpt.html'")
        self.cats[1]["url"]="https://wrong.example.com/methods.pdf"
        self.write_inputs()
        report=self.run_audit()
        issues=report["summary"]["metadata_problems"]
        self.assertEqual(1,issues["native_file_row_not_unique"]["count"])
        self.assertEqual(1,issues["catalog_publisher_url_differs_from_source_map"]["count"])

    def test_sample_stratification_is_stable_in_two_index_stages(self):
        rows=[dict(path=f"paper/{i}.pdf",kind="pdf",status="complete",
                   source_bytes=500,summarized=(i%2==0)) for i in range(20)]
        rows.append(dict(path="paper/bad.pdf",kind="pdf",status="error",
                         source_bytes=1000000,summarized=False))
        first=deterministic_sample(rows,seed="fixed",quotas={"pdf":6},
                                   max_original_bytes=600)
        second=deterministic_sample(list(reversed(rows)),seed="fixed",
                                    quotas={"pdf":6},max_original_bytes=600)
        self.assertEqual([x["path"] for x in first],[x["path"] for x in second])
        self.assertEqual(7,len(first))
        self.assertEqual(3,sum(x["summarized"] for x in first))
        self.assertIn("paper/bad.pdf",[x["path"] for x in first])

    def test_small_byte_budget_never_counts_unchecked_original_as_verified(self):
        res=self.run_audit(max_total_bytes=27)
        self.assertGreater(res["summary"]["skipped_due_budgets"],0)
        self.assertLess(res["summary"]["originals_independently_hashed"],3)
        self.assertLessEqual(res["summary"]["publisher_and_derivative_jointly_byte_verified"],
                             res["summary"]["originals_independently_hashed"])
        self.assertEqual(res["summary"]["current_derivatives_independently_hashed"],
                         res["summary"]["publisher_and_derivative_jointly_byte_verified"]+
                         res["summary"]["derivatives_verified_without_original_rehash"])
        self.assertGreaterEqual(res["summary"]["selected_originals_with_failures"],0)

    def test_output_destination_protects_original_derivative_and_database(self):
        base=self.root/"search-audit"
        base.mkdir()
        def checked(p):
            return private_audit_destination(
                p,source_root=self.original,derived_root=self.derived,
                native_filedb=self.filedb,source_map_db=self.mapdb)
        self.assertEqual(base/"new.json",checked(base/"new.json"))
        for candidate in (self.original/"unwanted.json",self.derived/"unwanted.json",
                          self.root/"unwanted.json"):
            with self.subTest(path=candidate),self.assertRaisesRegex(ValueError,"protected"):
                checked(candidate)
        existing=base/"existing.json"
        existing.write_text("original report")
        with self.assertRaisesRegex(ValueError,"overwrite"):
            checked(existing)
        alias=self.root/"source-alias"
        alias.symlink_to(self.original,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,"protected"):
            checked(alias/"unwanted.json")
        self.assertEqual("original report",existing.read_text())

    def test_symlink_and_path_traversal_originals_fail_closed(self):
        external=self.root/"external.txt";external.write_text("private!")
        alias=self.original/"alias.pdf";alias.symlink_to(external)
        for name in ("../external.txt", "/etc/passwd", "ref/../external.txt","alias.pdf"):
            with self.subTest(name=name),self.assertRaises(ValueError):
                _path(name,self.original)
        with self.assertRaisesRegex(ValueError,"not regular"):
            _digest_stable(self.original,500)


if __name__=="__main__":
    unittest.main()
