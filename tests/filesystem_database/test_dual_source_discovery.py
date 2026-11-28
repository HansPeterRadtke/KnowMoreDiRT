"""Dual read-only KMD channels: exact revision, no unverified rank fusion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from file_system_catalog.dual_source_discovery import (
    safe_corpus_relative, existing_file, verify_current_source,
    combined_dual_discovery,publisher_text_preview,_hash_file_with_passage,
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
            omit_verified_nonpublisher_passages=kwargs.get("omit_unverified",False),
        )

    def test_pdf_publisher_preview_never_quotes_extractor_marker(self):
        source='[[SOURCE UNIT: PDF page 1]]\n' + 'Paper title and abstract about the actual method ' * 9
        preview,label=publisher_text_preview(source,'PDF page 1')
        self.assertEqual('PDF page 1',label)
        self.assertTrue(preview.startswith('Paper title and abstract'))
        self.assertNotIn('[[SOURCE UNIT:',preview)
        self.assertIn(preview,source)
        self.assertLessEqual(len(preview),350)
        self.assertEqual('PDF page 1',label)

    def test_crossing_pdf_page_delimiter_uses_real_selected_page(self):
        source='tail end of older page 13.\n' + '[[SOURCE UNIT: PDF page 14]]\n' + 'New page authoritative text about a relevant issue. ' * 8
        preview,label=publisher_text_preview(source,'PDF page 13')
        self.assertEqual('PDF page 14',label)
        self.assertTrue(preview.startswith('New page authoritative text'))
        self.assertNotIn('[[SOURCE UNIT:',preview)
        self.assertIn(preview,source)
        source='This is a sufficiently detailed statement from page 13 with more than eighty characters.' + '[[SOURCE UNIT: PDF page 14]]\n' + 'Other page separate statement.'
        preview,label=publisher_text_preview(source,'PDF page 13')
        self.assertEqual('PDF page 13',label)
        self.assertTrue(preview.startswith('This is a sufficiently detailed'))
        self.assertNotIn('[[SOURCE UNIT:',preview)

    def test_no_raw_publisher_text_preview_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'only extractor metadata'):
            publisher_text_preview('[[SOURCE UNIT: PDF page 2]]','PDF page 2')
        for bad in (None,'',False):
            with self.assertRaises(ValueError):
                publisher_text_preview(bad,'PDF page 2')
        with self.assertRaises(ValueError):
            publisher_text_preview('This is publisher text','PDF page 2',max_chars=10000)

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

    def test_generated_first_and_subsequent_source_unit_labels_are_not_publisher_text(self):
        path="official/material.html"
        derived=self.derivatives/(path+".txt")
        real_body="An authentic author statement on reflected light and material roughness."
        generated=(
            "KMD generated summary about PDF page 13 is not primary evidence.\n"
            "[[SOURCE UNIT: PDF page 13]]\n"
            +real_body+"\n"
            "[[SOURCE UNIT: PDF page 14]]\n"
            "The second real page discusses rays reflected by metallic surfaces.\n"
        )
        derived.write_text(generated,encoding="utf-8")
        revision=sha(derived.read_bytes())
        self.map.execute("UPDATE originals SET derived_sha256=? WHERE source_path=?",
                         (revision,path))
        for generated_only in ("PDF page 13","PDF page 14",
                               "SOURCE UNIT:","KMD generated summary"):
            with self.subTest(needle=generated_only):
                item=dict(self.raw_node(path),
                          source_passage=generated_only,
                          source_passage_sha256=sha(generated_only.encode()))
                with self.assertRaisesRegex(ValueError,"absent from current publisher-source"):
                    self.combine(raw={**self.raw,"results":[item]})
        for real in (real_body,"rays reflected by metallic surfaces"):
            with self.subTest(real=real):
                item=dict(self.raw_node(path),
                          source_passage=real,
                          source_passage_sha256=sha(real.encode()))
                checked=self.combine(raw={**self.raw,"results":[item]})
                self.assertEqual(1,checked["raw_original_count"])
                self.assertEqual(real[:260],checked["original_passage_channel"][0]["passage_preview"])

    def test_exact_source_passage_does_not_cross_publisher_unit_boundaries(self):
        with TemporaryDirectory() as directory:
            p=Path(directory)/"original-source-derived.txt"
            p.write_bytes(
                b"KMD generated title and url metadata\n"
                b"[[SOURCE UNIT: PDF page 7]]\n"
                b"An actual first page end phrase with microfacet distributions.\n"
                b"[[SOURCE UNIT: PDF page 8]]\n"
                b"Another physical original page begins with dielectric reflectance.\n")
            for needle,expected in (
                (b"PDF page 7",False),(b"PDF page 8",False),
                (b"[[SOURCE UNIT: PDF page 8]]",False),
                (b"Another physical original page begins",True),
                (b"microfacet distributions",True),
                (b"distributions.\n[[SOURCE UNIT: PDF page 8]]\nAnother",False),
                (b"generated title",False),
            ):
                with self.subTest(needle=needle):
                    digest,present=_hash_file_with_passage(p,needle)
                    self.assertEqual(sha(p.read_bytes()),digest)
                    self.assertIs(present,expected)

    def test_source_label_and_publisher_passage_split_at_every_small_read_boundary(self):
        with TemporaryDirectory() as directory:
            p=Path(directory)/"original-derived.txt"
            passage=b"Photorealistic source text uses full spectral radiance and verified page geometry."
            p.write_bytes(
                b"Generated source catalog\n"
                b"[[SOURCE UNIT: PDF page 11]]\n"
                + passage +b"\n"
                b"[[SOURCE UNIT: PDF page 12]]\n"
                b"Second publisher page has authentic original detail."
            )
            original=sha(p.read_bytes())
            for block_bytes in (1,2,5,9,14,17,31,63,256,4096):
                with self.subTest(block_size=block_bytes),mock.patch(
                        "file_system_catalog.dual_source_discovery.READ_BLOCK",block_bytes):
                    for needle,present in (
                        (passage,True),
                        (b"spectral radiance",True),
                        (b"Second publisher page",True),
                        (b"PDF page 11",False),
                        (b"PDF page 12",False),
                        (b"source catalog",False),
                        (b"original detail.",True),
                    ):
                        digest,found=_hash_file_with_passage(p,needle)
                        self.assertEqual(original,digest)
                        self.assertIs(found,present,(block_bytes,needle))
            with self.assertRaisesRegex(ValueError,"nonempty"):
                _hash_file_with_passage(p,b"")

    def test_unclosed_oversized_extractor_label_abstains_from_source_evidence(self):
        with TemporaryDirectory() as directory:
            p=Path(directory)/"unfinished-source.txt"
            p.write_bytes(b"KMD header\n[[SOURCE UNIT: "+b"page-label-only"*200+
                          b"Original author body might appear but has no closed unit.")
            with self.assertRaisesRegex(ValueError,"unclosed source-unit label"):
                _hash_file_with_passage(p,b"Original author body")
            self.assertEqual(
                (sha(p.read_bytes()),True),
                _hash_file_with_passage(p,None),
            )

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

    def test_explicitly_audited_metadata_only_passages_are_omitted_without_fabrication(self):
        path="official/material.html"
        derived=self.derivatives/(path+".txt")
        metadata_only="A model's generated catalog summary is not publisher text."
        original=derived.read_text()
        derived.write_text(metadata_only+"\n"+original)
        updated_sha=sha(derived.read_bytes())
        self.map.execute("UPDATE originals SET derived_sha256=? WHERE source_path=?",(updated_sha,path))
        meta=dict(self.raw_node(path),source_passage=metadata_only,
                  source_passage_sha256=sha(metadata_only.encode()))
        raw={**self.raw,"results":[meta,self.raw_node('research/other.html')]}
        # The default pure function remains strictly fail-closed for callers
        # that require all submitted native candidate proofs to be valid.
        with self.assertRaisesRegex(ValueError,'absent from current publisher-source'):
            self.combine(raw=raw)
        result=self.combine(raw=raw,omit_unverified=True)
        self.assertEqual(1,result['uncitable_native_passage_candidates_omitted'])
        self.assertEqual(path,result['excluded_uncitable_native_passage_sources'][0]['original_path'])
        self.assertEqual(1,result['excluded_uncitable_native_passage_sources'][0]['native_candidate_rank'])
        self.assertEqual(['research/other.html'],[x['original_path'] for x in result['original_passage_channel']])
        self.assertEqual(2,result['original_passage_channel'][0]['native_raw_candidate_rank_before_verification'])
        self.assertEqual(1,result['original_passage_channel'][0]['original_source_rank'])
        self.assertTrue(result['no_database_or_original_mutation'])
        self.assertEqual(1,result['summary_original_count'])

    def test_reported_metadata_abstention_never_hides_changed_original_hash(self):
        path="official/material.html"
        original=self.originals/path
        original.write_bytes(original.read_bytes()+b"tampered")
        with self.assertRaisesRegex(ValueError,'bytes changed'):
            self.combine(omit_unverified=True)
        original.write_bytes(original.read_bytes()[:-8])
        raw={**self.raw,'results':[dict(self.raw_node(path),exact_passage_hash_verified=False)]}
        with self.assertRaisesRegex(ValueError,'unverified'):
            self.combine(raw=raw,omit_unverified=True)
        with self.assertRaisesRegex(ValueError,'untrusted source verification mode'):
            self.combine(omit_unverified='yes')

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
