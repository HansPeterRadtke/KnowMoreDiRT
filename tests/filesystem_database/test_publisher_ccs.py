"""Publisher classifications may repair a missing native keyphrase only with exact provenance."""
from contextlib import closing
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from file_system_catalog.hierarchical_summary import HierarchicalSummarizer
from file_system_catalog.hierarchical_summary_store import append_summary_rows, summary_rows
from file_system_catalog.publisher_ccs import CCS_EXTRACTOR, first_page_ccs
from file_system_catalog.summary_search_audit import make_search_audit
from test_hierarchical_summary import FakeSummaryModel, long_document
from test_hierarchical_summary_store import FakeEmbeddings, fixture
from test_summary_search_audit import load_cli


PAGE = "[[SOURCE UNIT: PDF page 1]]\n"
CCS = " CCS Concepts\n • Computing methodologies → Reflectance modeling;\n"
SOURCE = "KMD source provenance fixture\n\n" + PAGE + "Résumé 北🌱.\n" + CCS + "Reflectance modeling appears elsewhere.\n"


class PublisherClassificationTests(unittest.TestCase):
    def prepare(self, source=SOURCE, kind="topic"):
        tree = HierarchicalSummarizer(FakeSummaryModel(900), output_tokens=64, safety_tokens=16).summarize(
            title="Guide", text=source)
        tree = replace(tree, nodes=tuple(replace(n, native_analysis={
            "document_summary": n.summary, "facets": [{"label": "Methods", "strength": "strong",
                "representations": [{"kind": kind, "strength": "strong", "text": "Scattering theory"}]}]
        }) for n in tree.nodes))
        return dict(original_path="books/guide.pdf", original_sha256="a"*64, original_bytes=12345,
                    derived_sha256=sha256(source.encode()).hexdigest(), derived_text=source,
                    summary_start_char=0, tree=tree, embedding_client=FakeEmbeddings(),
                    analysis_model="fixture-model", collection_id="fixture", require_native_terms=True)

    def test_exact_page_substrings_unicode_crlf_and_duplicate_entries(self):
        extra = " • Computing methodologies → Wave optics;\n"
        text = SOURCE.replace(CCS, CCS + extra + extra).replace("\n", "\r\n")
        subjects = first_page_ccs(text)
        self.assertEqual(list(subjects), ["Reflectance modeling", "Wave optics"])
        for term, anchor in subjects.items():
            self.assertEqual(text[anchor["start_char"]:anchor["end_char"]], term)
            self.assertEqual(anchor["source_section"], "PDF page 1")

    def test_rejects_metadata_later_pages_ambiguous_headings_and_column_fragments(self):
        for text in (CCS + PAGE + "Body", PAGE + "Body\n[[SOURCE UNIT: PDF page 2]]\n" + CCS,
                     "[[SOURCE UNIT: Original plain text]]\n" + CCS, PAGE + CCS + CCS,
                     PAGE + CCS.replace("→", ":"), PAGE + CCS.replace(";", ""),
                     PAGE + CCS.replace("modeling", "modeling     unrelated column")):
            with self.subTest(text=text):
                self.assertEqual({}, first_page_ccs(text))

    def test_old_model_representations_are_preserved_and_insert_is_idempotent(self):
        with closing(fixture()) as db:
            args = self.prepare(SOURCE + long_document(25, 130))

            class BatchSensitiveEmbeddings(FakeEmbeddings):
                def embed(self, texts):
                    return super().embed([f"{len(texts)} {text}" for text in texts])

            args["embedding_client"] = BatchSensitiveEmbeddings()
            chunks, rows = summary_rows(db, **args)
            _, unchanged = summary_rows(db, **dict(args, require_native_terms=False))
            by_id = {r["representation_id"]: r for r in rows}
            for row in unchanged:
                saved = by_id[row["representation_id"]]
                self.assertEqual({k:v for k,v in row.items() if not k.endswith("_at_ns")},
                                 {k:v for k,v in saved.items() if not k.endswith("_at_ns")})
            new = [r for r in rows if r["analysis_model"] == CCS_EXTRACTOR]
            self.assertEqual(len(new), 1)
            g = json.loads(new[0]["generation_json"])
            self.assertEqual(g["node_id"], args["tree"].root_id)
            self.assertEqual(g["publisher_classification"], first_page_ccs(args["derived_text"])["Reflectance modeling"])
            self.assertEqual(new[0]["representation_kind"], "keyphrase")
            self.assertGreater(append_summary_rows(db, chunk_rows=chunks, representation_rows=rows)["new_rows"], 0)
            self.assertEqual(0, append_summary_rows(db, chunk_rows=chunks, representation_rows=rows)["new_rows"])

    def test_existing_generated_keyword_bypasses_fallback(self):
        with closing(fixture()) as db:
            for kind in ("keyword", "keyphrase"):
                _, rows = summary_rows(db, **self.prepare(kind=kind))
                self.assertFalse(any(r["analysis_model"] == CCS_EXTRACTOR for r in rows))

    def test_unsupported_publisher_text_fails_before_any_embedding(self):
        with closing(fixture()) as db:
            args = self.prepare(PAGE + "Ordinary body without subject labels.")
            with patch.object(args["embedding_client"], "embed", side_effect=AssertionError("unused")):
                with self.assertRaisesRegex(RuntimeError, "no indexable terms"):
                    summary_rows(db, **args)

    def test_search_and_audit_recheck_the_real_classification_position(self):
        cli = load_cli()
        args = self.prepare()
        embeddings = args["embedding_client"]
        embeddings.embed_queries = embeddings.embed
        fake = SimpleNamespace(LocalE5=lambda **kwargs: embeddings)
        with tempfile.TemporaryDirectory() as tmp, closing(fixture()) as db:
            path = Path(tmp)
            publisher = path / "paper.pdf"
            publisher.write_bytes(b"p" * 12345)
            args["original_sha256"] = sha256(publisher.read_bytes()).hexdigest()
            derivative = path / "derivative.txt"
            derivative.write_bytes(SOURCE.encode())
            src = args["original_path"]
            versions = ({src: {"title": "Guide"}}, {src: {"bytes_received": 12345, "sha256": args["original_sha256"]}},
                        {src: (src, args["original_sha256"], args["derived_sha256"], "derived.txt", "complete")})
            chunks, rows = summary_rows(db, **args)
            append_summary_rows(db, chunk_rows=chunks, representation_rows=rows)
            disk = path / "native.sqlite3"
            with closing(sqlite3.connect(disk)) as target:
                db.backup(target)
            query = SimpleNamespace(query="Reflectance modeling", top=1, with_facets=True, nodes_per_original=8)
            with patch.dict(sys.modules, knowledge_library_semantic_model=fake), patch.object(cli, "DB", disk), \
                 patch.object(cli, "_current_versions", return_value=versions), \
                 patch.object(cli, "publisher_path", return_value=publisher), \
                 patch.object(cli, "checked_derivative_path", return_value=derivative):
                found = cli._search_impl(query)
                node = found["documents"][0]["matched_summary_nodes"][0]
                anchor = node["publisher_classification"]
                self.assertEqual(anchor, first_page_ccs(SOURCE)[query.query])
                receipt = make_search_audit(query=query.query, result=found, elapsed_seconds=0, model="fixture", database="fixture")
                self.assertEqual(receipt["ranked_results"][0]["matched_nodes"][0]["publisher_classification"], anchor)
                item = next(r for r in rows if r["analysis_model"] == CCS_EXTRACTOR)
                g = json.loads(item["generation_json"])
                # The same literal occurs again in ordinary prose. Only the assigned subject is admissible.
                wrong = SOURCE.rindex(query.query)
                g["publisher_classification"].update(start_char=wrong, end_char=wrong + len(query.query))
                with sqlite3.connect(disk) as writer:
                    writer.execute("UPDATE content_representations SET generation_json=? WHERE representation_id=?",
                                   (json.dumps(g), item["representation_id"]))
                later = cli._search_impl(query)["documents"][0]["matched_summary_nodes"]
                self.assertTrue(later)
                self.assertNotIn(item["representation_id"], [n["native_representation_id"] for n in later])
