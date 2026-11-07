"""Native KMD summary+facet pipeline: original schema, storage and retrieval."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import sqlite3
import unittest

from file_system_catalog.hierarchical_summary import HierarchicalSummarizer
from file_system_catalog.hierarchical_summary_store import summary_rows,append_summary_rows
from file_system_catalog.summary_search_batched import search_summary_entries_batched
from file_system_catalog.content_pipeline import flatten_representations,normalize_analysis
from file_system_catalog.content_schema import REPRESENTATION_KIND_VALUES
from test_hierarchical_summary import FakeSummaryModel,long_document
from test_hierarchical_summary_store import fixture, FakeEmbeddings


class FakeNativeFacetModel(FakeSummaryModel):
    def generate_retrieval_analysis(self,*,system,user,output_tokens):
        self.calls.append((system,user,output_tokens))
        return normalize_analysis({
            "document_summary":"Source-grounded GUI design overview and choices.",
            "facets":[{
                "label":"interface design",
                "strength":"essential",
                "representations":[
                    {"kind":"keyword","strength":"strong","text":"GUI"},
                    {"kind":"keyphrase","strength":"strong","text":"GUI interface design"},
                    {"kind":"description","strength":"moderate",
                     "text":"UI labels and graphical controls organize interaction."},
                ],
            },{
                "label":"interaction tasks",
                "strength":"strong",
                "representations":[
                    {"kind":"keyword","strength":"moderate","text":"navigation"},
                    {"kind":"keyphrase","strength":"strong","text":"user decision time"},
                ],
            }],
        })


class NativeFacetBridgeTests(unittest.TestCase):
    def build(self):
        db=fixture()
        model=FakeNativeFacetModel(3000)
        source=long_document(40,180)
        tree=HierarchicalSummarizer(
            model,output_tokens=700,safety_tokens=100,
            native_facets=True,max_levels=12,
        ).summarize(title="Complete GUI book",text=source,max_calls=300)
        return db,model,source,tree

    def test_same_native_facets_schema_populates_every_recursive_level(self):
        db,model,source,tree=self.build()
        self.assertGreater(tree.root.level,0)
        self.assertEqual(len(model.calls),len(tree.nodes))
        for node in tree.nodes:
            self.assertEqual(node.native_analysis["document_summary"],node.summary)
            flattened=flatten_representations(node.native_analysis)
            kinds=Counter(x["kind"] for x in flattened)
            self.assertGreaterEqual(kinds["keyword"],2)
            self.assertGreaterEqual(kinds["keyphrase"],2)
            self.assertGreaterEqual(kinds["topic"],2)
            self.assertTrue(set(kinds).issubset(set(REPRESENTATION_KIND_VALUES)))
        self.assertEqual(tree.root.end_char,len(source))
        db.close()

    def test_source_linked_keywords_and_vectors_in_original_filedb(self):
        db,model,source,tree=self.build()
        chunks,reps=summary_rows(
            db,original_path="books/guide.pdf",
            original_sha256="a"*64,original_bytes=12345,
            derived_sha256=hashlib.sha256(source.encode()).hexdigest(),
            derived_text=source,summary_start_char=0,tree=tree,
            embedding_client=FakeEmbeddings(),analysis_model="test-local-llm",
            collection_id="kmd-local-summary-v1",
        )
        summaryreps=[x for x in reps if x["representation_kind"] in ("summary","summary_long")]
        self.assertEqual(len(summaryreps),len(tree.nodes))
        self.assertEqual(len(chunks),len(tree.nodes))
        self.assertGreater(len(reps),len(chunks)*3)
        kinds=Counter(r["representation_kind"] for r in reps)
        self.assertEqual(kinds["keyphrase"],2*len(tree.nodes))
        self.assertEqual(kinds["keyword"],2*len(tree.nodes))
        self.assertEqual(kinds["topic"],2*len(tree.nodes))
        self.assertTrue(all(r["embedding_dimension"]==384
                           and len(bytes(r["embedding_blob"]))==768 for r in reps))
        for r in reps:
            g=json.loads(r["generation_json"])
            self.assertEqual(g["source_tree_sha256"],hashlib.sha256(source.encode()).hexdigest())
            self.assertIn(g["node_id"],{n.node_id for n in tree.nodes})
        result=append_summary_rows(db,chunk_rows=chunks,representation_rows=reps)
        self.assertEqual(len(chunks)+len(reps),result["new_rows"])
        self.assertEqual(0,append_summary_rows(db,chunk_rows=chunks,representation_rows=reps)["new_rows"])
        self.assertEqual("ok",db.execute("pragma integrity_check").fetchone()[0])
        self.assertEqual([],db.execute("pragma foreign_key_check").fetchall())
        query=FakeEmbeddings().embed(["GUI interface design"])[0]
        facet=search_summary_entries_batched(db,query,limit=5,include_facets=True,
                                              expected_model="test-embedding-384")
        self.assertEqual("books/guide.pdf",facet[0]["original_path"])
        self.assertGreater(facet[0]["similarity"],0.99)
        self.assertEqual("keyphrase",facet[0]["best_kind"])
        self.assertEqual(len(reps),facet[0]["indexed_summary_nodes_for_document"])
        old=search_summary_entries_batched(db,query,limit=5,include_facets=False,
                                            expected_model="test-embedding-384")
        self.assertEqual("books/guide.pdf",old[0]["original_path"])
        self.assertEqual(len(chunks),old[0]["indexed_summary_nodes_for_document"])
        self.assertTrue(all(v["summary_kind"] in ("summary","summary_long")
                            for v in old[0]["matched_summary_nodes"]))
        db.close()

    def test_no_facet_output_requires_no_new_metadata_tables(self):
        db=fixture()
        source=long_document(5,50)
        tree=HierarchicalSummarizer(
            FakeSummaryModel(1000),output_tokens=64,safety_tokens=16,
        ).summarize(title="Legacy",text=source)
        chunks,reps=summary_rows(
            db,original_path="books/guide.pdf",original_sha256="a"*64,
            original_bytes=12345,
            derived_sha256=hashlib.sha256(source.encode()).hexdigest(),
            derived_text=source,summary_start_char=0,tree=tree,
            embedding_client=FakeEmbeddings(),analysis_model="test-local-llm",
            collection_id="kmd-local-summary-v1",
        )
        self.assertEqual(len(chunks),len(reps))
        db.close()


if __name__=="__main__":
    unittest.main()
