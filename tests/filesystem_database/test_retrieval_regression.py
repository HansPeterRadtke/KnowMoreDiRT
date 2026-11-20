"""Fixed-source ranking regression cannot become guessed answer correctness."""
import json
import sqlite3
import unittest

from file_system_catalog.retrieval_regression import (
    validated_gold,evaluate_case,score,source_record,assert_equivalent_gold,
)


class FixedOriginalRetrievalTests(unittest.TestCase):
    def test_gold_validates_unique_paths_and_explicit_unknowns(self):
        input=[
            {"id":"first","query":"How to measure material roughness?",
             "category":"science","expected_path":"books/reference.pdf"},
            {"id":"unlabeled","query":"An unrelated request with no exact original gold.",
             "expected_path":None},
        ]
        result=validated_gold(input)
        self.assertEqual(2,len(result))
        self.assertIsNone(result[1]["expected_path"])
        with self.assertRaisesRegex(ValueError,"duplicate"):
            validated_gold([input[0],input[0]])
        for illegal in ("/root/foo.pdf","../foo.pdf","a//foo.pdf","a\\b.pdf"):
            with self.subTest(illegal=illegal),self.assertRaises(ValueError):
                validated_gold([dict(input[0],expected_path=illegal)])

    def test_identifiable_rank_preserves_original_identity(self):
        case={"id":"reference","category":"source","query":"question","expected_path":"book/a.pdf"}
        row=evaluate_case(case,lambda q:["book/b.pdf","book/a.pdf"],
                          lambda p:{"gold_status":"native_summaries_match_source_map_revision"})
        self.assertEqual(2,row["gold_rank"])
        self.assertEqual("book/a.pdf",row["gold_path"])
        self.assertFalse(row["historical_negative_means_absent_gold_not_no_answer"])
        self.assertEqual(64,len(row["query_sha256"]))

    def test_duplicate_search_identity_rejected(self):
        case=validated_gold([{"id":"x","query":"A known question","expected_path":"book/a.pdf"}])[0]
        with self.assertRaisesRegex(ValueError,"duplicate"):
            evaluate_case(case,lambda q:["book/a.pdf","book/a.pdf"],
                          lambda p:{"gold_status":"valid"})

    def test_historical_negative_is_not_a_ground_truth_unanswerable(self):
        case=validated_gold([{"id":"negative","query":"Find a specific reference",
                              "expected_path":None}])[0]
        row=evaluate_case(case,lambda q:["still/valid.pdf"],
                          lambda p:{"gold_status":"no_gold_predeclared_not_unanswerable"})
        self.assertIsNone(row["gold_rank"])
        self.assertTrue(row["historical_negative_means_absent_gold_not_no_answer"])
        self.assertEqual(0,score([row])["positive_gold_cases"])

    def test_multiple_relevant_originals_score_first_and_fractional_coverage(self):
        raw={"id":"multi","query":"Compare two independent provenance checks",
             "relevant_paths":["papers/independent-a.pdf","papers/independent-b.pdf"]}
        case=validated_gold([raw])[0]
        self.assertIsNone(case["expected_path"])
        self.assertEqual(2,len(case["relevant_paths"]))
        proofs={p:{"gold_status":"native_summaries_match_source_map_revision"}
                for p in case["relevant_paths"]}
        row=evaluate_case(case,lambda q:["irrelevant.pdf","papers/independent-b.pdf"],
                          lambda p:proofs.get(p,{"gold_status":"not_expected"}))
        self.assertEqual(2,row["gold_rank"])
        self.assertEqual({"papers/independent-a.pdf":None,
                          "papers/independent-b.pdf":2},row["known_relevant_ranks"])
        metrics=score([row])
        self.assertEqual(1,metrics["correct_top3"])
        self.assertEqual(.5,metrics["mean_known_relevant_source_recall_at10"])
        self.assertEqual(0,metrics["complete_known_relevant_source_recall_at10"])
        self.assertEqual(2,metrics["known_relevant_source_identifiers"])
        self.assertEqual(0,metrics["gold_with_missing_or_stale_native_source"])

    def test_multi_gold_preserves_primary_and_rejects_contradictory_labels(self):
        good=validated_gold([{"id":"two","query":"Find manuscript provenance",
           "expected_path":"original/a.pdf", "relevant_paths":["original/a.pdf","original/b.pdf"]}])[0]
        row=evaluate_case(good,lambda _: ["original/b.pdf"],
                          lambda path:{"gold_status":"native_summaries_match_source_map_revision"})
        self.assertEqual("original/a.pdf",row["gold_path"])
        self.assertEqual(1,row["gold_rank"])
        for label in ([], ["original/a.pdf","original/a.pdf"],
                      ["original/a.pdf","../bad.pdf"], ["original/b.pdf"]):
            with self.subTest(label=label),self.assertRaisesRegex(ValueError,"multi-source gold"):
                validated_gold([{"id":"unsafe","query":"Valid query text",
                                 "expected_path":"original/a.pdf","relevant_paths":label}])

    def test_known_stale_second_gold_prevents_false_source_verification(self):
        case=validated_gold([{"id":"two","query":"Independent two-source experiment",
                               "relevant_paths":["a.pdf","b.pdf"]}])[0]
        row=evaluate_case(case,lambda _: ["a.pdf","b.pdf"],
                          lambda p:{"gold_status":("native_summaries_match_source_map_revision"
                                          if p=="a.pdf" else "source_map_missing_or_not_complete")})
        metrics=score([row])
        self.assertEqual(1,metrics["correct_top1"])
        self.assertEqual(1.,metrics["mean_known_relevant_source_recall_at10"])
        self.assertEqual(1,metrics["gold_with_missing_or_stale_native_source"])

    def test_historical_baseline_must_have_exact_same_queries_and_labels(self):
        case=validated_gold([{"id":"old","query":"Old source-specific question",
                              "expected_path":"a.pdf"}])
        prior={"cases":[{"id":"old","query":"Old source-specific question",
                         "expected_path":"a.pdf"}],"summary":{"top1_count":1}}
        assert_equivalent_gold(case,prior)
        altered=dict(prior)
        altered["cases"]=[dict(prior["cases"][0],query="Modified new query")]
        with self.assertRaisesRegex(ValueError,"differ"):
            assert_equivalent_gold(case,altered)
        multi=validated_gold([{"id":"old","query":"Old source-specific question",
                               "expected_path":"a.pdf","relevant_paths":["a.pdf","b.pdf"]}])
        with self.assertRaisesRegex(ValueError,"multi-source"):
            assert_equivalent_gold(multi,prior)

    def make_databases(self):
        db=sqlite3.connect(":memory:")
        source=sqlite3.connect(":memory:")
        db.execute("CREATE TABLE filesystem_entries (id INTEGER,relative_path_display TEXT,entry_type TEXT)")
        db.execute("CREATE TABLE content_chunks (chunk_id TEXT,filesystem_entry_id INTEGER,collection_id TEXT,content_sha256 TEXT)")
        db.execute("CREATE TABLE content_representations (chunk_id TEXT,generation_json TEXT,prompt_version TEXT,analysis_status TEXT,representation_kind TEXT)")
        source.execute("CREATE TABLE originals (source_path TEXT,status TEXT,source_sha256 TEXT,derived_sha256 TEXT)")
        return db,source

    def test_current_source_revision_correspondence_and_unindexed_case(self):
        db,m=self.make_databases()
        a="a"*64;b="b"*64
        db.execute("INSERT INTO filesystem_entries VALUES(7,'source.pdf','file')")
        db.execute("INSERT INTO content_chunks VALUES('chunk7',7,'raspi-original-books-semantic-content-v1:summary',?)",(a,))
        db.execute("INSERT INTO content_representations VALUES('chunk7',?,'kmd-hierarchical-source-summary-v1','complete','summary')",(json.dumps({'original_sha256':a,'derived_sha256':b}),))
        m.execute("INSERT INTO originals VALUES('source.pdf','complete',?,?)",(a,b))
        self.assertEqual("native_summaries_match_source_map_revision",
                         source_record(db,m,"source.pdf")["gold_status"])
        self.assertTrue(source_record(db,m,"source.pdf")["current_source_revision_verified"])
        m.execute("UPDATE originals SET derived_sha256=?",("d"*64,))
        self.assertEqual("native_summary_missing_current_source_revision",
                         source_record(db,m,"source.pdf")["gold_status"])
        m.execute("UPDATE originals SET derived_sha256=?, source_sha256=?",(b,"c"*64))
        self.assertEqual("native_summary_missing_current_source_revision",
                         source_record(db,m,"source.pdf")["gold_status"])
        db.execute("DELETE FROM content_chunks")
        self.assertEqual(0,source_record(db,m,"source.pdf")["native_summary_rows"])
        self.assertEqual("canonical_native_original_absent_or_ambiguous",
                         source_record(db,m,"missing.pdf")["gold_status"])
        db.close();m.close()

    def test_scores_are_exact_counts_not_unverified_probabilities(self):
        valid={"gold_status":"native_summaries_match_source_map_revision"}
        rows=[
            {"gold_path":"a.pdf","gold_rank":1,"source_identity_evidence":valid},
            {"gold_path":"b.pdf","gold_rank":3,"source_identity_evidence":valid},
            {"gold_path":"c.pdf","gold_rank":None,"source_identity_evidence":{
                "gold_status":"native_summary_missing_current_source_revision"}},
            {"gold_path":None,"gold_rank":None,"source_identity_evidence":{
                "gold_status":"no_gold_predeclared_not_unanswerable"}},
        ]
        stats=score(rows,{"top1_count":3,"top10_count":3})
        self.assertEqual(1,stats["correct_top1"])
        self.assertEqual(2,stats["correct_top10"])
        self.assertEqual(-1,stats["top10_difference_vs_historical"])
        self.assertEqual(1,stats["gold_with_missing_or_stale_native_source"])
        self.assertEqual(1,stats["historical_negative_unlabeled_cases"])


if __name__=="__main__":
    unittest.main()
