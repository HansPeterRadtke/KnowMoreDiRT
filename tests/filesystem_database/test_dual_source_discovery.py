"""Dual read-only KMD channels: exact revision, no unverified rank fusion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from file_system_catalog.dual_source_discovery import (
    safe_corpus_relative, existing_file, verify_current_source,
    combined_dual_discovery,
)


def sha(content: bytes)->str:
    return hashlib.sha256(content).hexdigest()


class SourceBoundDualSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)
        self.originals=root/"originals"
        self.derivatives=root/"derived"
        self.originals.mkdir()
        self.derivatives.mkdir()
        self.map=sqlite3.connect(":memory:")
        self.addCleanup(self.map.close)
        self.map.execute(
            "CREATE TABLE originals (source_path TEXT,status TEXT,source_sha256 TEXT,"
            "derived_sha256 TEXT,derived_path TEXT)"
        )
        self.receipts={}
        self.add("official/material.html","An authentic material microfacet reflectance statement")
        self.add("research/other.html","The second original has a camera shutter example")
        self.query="Which original source discusses microfacet reflectance?"
        self.summary={
            "query":self.query,
            "provenance_policy":"publisher_and_derived_sha256_checked;summary_is_not_primary_evidence",
            "documents":[self.summarized("official/material.html")],
        }
        self.raw={
            "query":self.query,
            "retrieval_mode":"overview",
            "results":[self.raw_node("official/material.html")],
        }

    def add(self,path,text):
        p=self.originals/path
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(f"<html><p>{text}</p></html>",encoding="utf-8")
        d=self.derivatives/(path+".txt")
        d.parent.mkdir(parents=True,exist_ok=True)
        d.write_text(f"[[SOURCE UNIT: HTML original publisher document]]\n{text}\n",encoding="utf-8")
        self.receipts[path]=sha(p.read_bytes())
        self.map.execute("INSERT INTO originals VALUES(?,?,?,?,?)",(
            path,"complete",sha(p.read_bytes()),sha(d.read_bytes()),path+".txt",
        ))
        self.payload_text={**getattr(self,"payload_text",{}),path:text}

    def summarized(self,path):
        return {
            "original_path":path,"best_summary":"Model-generated candidate text",
            "similarity":0.92,"matched_summary_nodes":[
                {"original_sha256":self.receipts[path],"summary_kind":"summary"}
            ],
        }

    def raw_node(self,path):
        text=self.payload_text[path]
        return {
            "original_path":path,"original_source_sha256":self.receipts[path],
            "exact_passage_hash_verified":True,
            "source_passage":text,
            "source_passage_sha256":sha(text.encode()),
            "source_excerpt":text,"semantic_similarity":0.95,
            "source_section":"HTML original publisher document",
            "publisher_url":"https://example.invalid/source",
            "source_page_citation_warning":None,
        }

    def combine(self,summary=None,raw=None,**kwargs):
        return combined_dual_discovery(
            summary_payload=summary or self.summary,
            original_payload=raw or self.raw,
            source_map=self.map,
            original_root=self.originals,
            derivative_root=self.derivatives,
            original_receipts=self.receipts,
            max_originals=kwargs.get("max_originals",5),
        )

    def test_both_independent_ranks_preserved_without_fusion(self):
        before=self.map.total_changes
        orig_bytes=(self.originals/"official/material.html").read_bytes()
        result=self.combine()
        self.assertEqual(1,result["summary_original_count"])
        self.assertEqual(1,result["raw_original_count"])
        self.assertEqual(1,result["union_original_count"])
        self.assertTrue(result["no_ranking_fusion_or_score_threshold_applied"])
        self.assertTrue(result["no_database_or_original_mutation"])
        self.assertEqual(1,result["summary_channel"][0]["summary_rank"])
        self.assertEqual(1,result["original_passage_channel"][0]["original_source_rank"])
        self.assertEqual("An authentic material microfacet reflectance statement",
                         result["original_passage_channel"][0]["passage_preview"])
        self.assertEqual(before,self.map.total_changes)
        self.assertEqual(orig_bytes,(self.originals/"official/material.html").read_bytes())

    def test_new_raw_only_original_is_not_discarded(self):
        raw=dict(self.raw)
        raw["results"]=[self.raw_node("research/other.html"),self.raw_node("official/material.html")]
        result=self.combine(raw=raw)
        self.assertEqual(2,result["union_original_count"])
        self.assertEqual(1,result["originals_found_only_via_original_passages"])
        self.assertEqual(
            [1,2],
            [r["original_source_rank"] for r in result["original_passage_channel"]],
        )
        self.assertIsNone(
            next(x for x in result["unique_originals_alphabetically_for_navigation_not_relevance"]
                 if x["original_path"]=="research/other.html")["summary_rank"]
        )

    def test_duplicate_raw_passage_cannot_inflate_document_count(self):
        raw=dict(self.raw)
        raw["results"]=[self.raw_node("official/material.html"),
                        self.raw_node("official/material.html"),
                        self.raw_node("research/other.html")]
        result=self.combine(raw=raw)
        self.assertEqual(2,result["raw_original_count"])
        self.assertEqual(2,result["union_original_count"])

    def test_changed_publisher_and_derivative_fail_closed(self):
        p=self.originals/"official/material.html"
        p.write_bytes(p.read_bytes()+b"modified")
        with self.assertRaisesRegex(ValueError,"bytes changed"):
            self.combine()
        p.write_bytes(p.read_bytes()[:-8])
        derivative=self.derivatives/"official/material.html.txt"
        derivative.write_bytes(derivative.read_bytes()+b"modified")
        with self.assertRaisesRegex(ValueError,"bytes changed"):
            self.combine()

    def test_raw_passage_must_match_digest_and_current_derived_bytes(self):
        raw=dict(self.raw)
        old=self.raw_node("official/material.html")
        raw["results"]=[dict(old,source_passage="invented text")]
        with self.assertRaisesRegex(ValueError,"chunk digest"):
            self.combine(raw=raw)
        raw["results"]=[dict(old,source_passage="invented text",
                             source_passage_sha256=sha(b"invented text"))]
        with self.assertRaisesRegex(ValueError,"absent from current publisher-source"):
            self.combine(raw=raw)
        raw["results"]=[dict(old,exact_passage_hash_verified=False)]
        with self.assertRaisesRegex(ValueError,"unverified"):
            self.combine(raw=raw)

    def test_provenance_metadata_prologue_is_never_publisher_passage(self):
        path="official/material.html"
        derivative=self.derivatives/(path+".txt")
        original=derivative.read_text()
        unsupported="This phrase exists in generated catalog metadata only."
        derivative.write_text(unsupported+"\n"+original)
        current_sha=sha(derivative.read_bytes())
        self.map.execute("UPDATE originals SET derived_sha256=? WHERE source_path=?",
                         (current_sha,path))
        item=dict(self.raw_node(path),source_passage=unsupported,
                  source_passage_sha256=sha(unsupported.encode()))
        with self.assertRaisesRegex(ValueError,"absent from current publisher-source"):
            self.combine(raw={**self.raw,"results":[item]})

    def test_source_map_and_summary_node_sha_are_mandatory(self):
        bad=dict(self.summary)
        bad["documents"]=[dict(self.summarized("official/material.html"),
                               matched_summary_nodes=[{"original_sha256":"0"*64}])]
        with self.assertRaisesRegex(ValueError,"revision differs"):
            self.combine(summary=bad)
        self.map.execute("UPDATE originals SET status='error' WHERE source_path='official/material.html'")
        with self.assertRaisesRegex(ValueError,"source-map original"):
            self.combine()

    def test_symlinked_original_rejected_and_escaped_path_fails(self):
        p=self.originals/"official/material.html"
        p.unlink()
        p.symlink_to(self.derivatives/"official/material.html.txt")
        with self.assertRaisesRegex(ValueError,"symlinked"):
            self.combine()
        for invalid in ("/var/secret","../other","a/../b","a//b","a\\b"):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                safe_corpus_relative(invalid)

    def test_mismatched_queries_or_untrusted_backend_contract_rejected(self):
        with self.assertRaisesRegex(ValueError,"schemas"):
            self.combine(raw={**self.raw,"query":"different question"})
        with self.assertRaisesRegex(ValueError,"schemas"):
            self.combine(summary={**self.summary,"provenance_policy":"none"})
        with self.assertRaisesRegex(ValueError,"schemas"):
            self.combine(raw={**self.raw,"retrieval_mode":"something unsafe"})

    def test_unsummarized_corpus_original_can_be_raw_only(self):
        summary={**self.summary,"documents":[]}
        raw={**self.raw,"results":[self.raw_node("research/other.html")]}
        result=self.combine(summary=summary,raw=raw)
        self.assertEqual(0,result["summary_original_count"])
        self.assertEqual(1,result["raw_original_count"])
        self.assertEqual(1,result["originals_found_only_via_original_passages"])
        self.assertTrue(result["original_passage_channel"][0]["publisher_revision_verified"])


if __name__=="__main__":
    unittest.main()
