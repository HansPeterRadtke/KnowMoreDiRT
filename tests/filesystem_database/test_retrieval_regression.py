"""Fixed-source ranking regression cannot become guessed answer correctness."""
import hashlib
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import threading
import time
from tempfile import TemporaryDirectory
import unittest

from file_system_catalog.retrieval_regression import (
    validated_gold,evaluate_case,score,source_record,assert_equivalent_gold,
    attest_html_original_gold,open_live_readonly_db,
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

    def test_historical_hash_only_multi_positive_baseline_requires_frozen_file_sha_and_full_labels(self):
        cases=validated_gold([
            {"id":"a","query":"Material roughness and microfacets",
             "category":"two_references",
             "relevant_paths":["papers/first.html","papers/second.html"]},
            {"id":"b","query":"Original normal maps in real-time renderers",
             "category":"docs",
             "relevant_paths":["docs/normal-a.html","docs/normal-b.html"]},
        ])
        # Mirrors the actual previously frozen 2026-11-20 baseline, whose
        # question text was intentionally omitted from its private report.
        digest="b"*64
        old={
            "fixed_gold_sha256":digest,
            "complete_fixed_population":True,
            "selected_case_ids":["a","b"],
            "cases":[{"id":row["id"],
                      "query_sha256":hashlib.sha256(row["query"].encode()).hexdigest(),
                      "known_relevant_original_paths":list(row["relevant_paths"]),
                      "gold_path":row["relevant_paths"][0],
                      "category":row["category"]}
                     for row in cases],
        }
        assert_equivalent_gold(cases,old,fixed_gold_sha256=digest)
        for edited_cases in [
            [dict(cases[0],query="Changed prompt"),cases[1]],
            [dict(cases[0],relevant_paths=["papers/second.html","papers/first.html"]),cases[1]],
            [dict(cases[0],category="unverified different category"),cases[1]],
        ]:
            with self.subTest(edited=edited_cases[0]),self.assertRaises(ValueError):
                assert_equivalent_gold(edited_cases,old,fixed_gold_sha256=digest)
        for change in [
            ("fixed_gold_sha256","c"*64),
            ("complete_fixed_population",False),
            ("selected_case_ids",["b","a"]),
        ]:
            with self.subTest(key=change[0]),self.assertRaises(ValueError):
                assert_equivalent_gold(cases,{**old,change[0]:change[1]},
                                       fixed_gold_sha256=digest)
        with self.assertRaises(ValueError):
            assert_equivalent_gold(cases,old)
        with self.assertRaises(ValueError):
            assert_equivalent_gold(cases,old,fixed_gold_sha256="0"*64)
        modified={"cases":[dict(old["cases"][0],gold_path="wrong/source.html"),
                           old["cases"][1]], **{k:v for k,v in old.items() if k!="cases"}}
        with self.assertRaises(ValueError):
            assert_equivalent_gold(cases,modified,fixed_gold_sha256=digest)
        short=dict(old)
        short["cases"]=[old["cases"][0]]
        with self.assertRaises(ValueError):
            assert_equivalent_gold(cases,short,fixed_gold_sha256=digest)

    def test_actual_frozen_five_multi_positive_source_baseline_bytes_still_match(self):
        from tempfile import TemporaryDirectory
        # Use only existing local, committed-in-the-owner-library artifacts;
        # skip on developer machines without the private original corpus.
        base=Path('/data/var/knowmoredirt/knowledge-library/search-audit/multirelevant-originals-20261121')
        frozen=base/'original_multirelevant_gold_frozen.json'
        historical=base/'native_top10_gold_source_rechecked.json'
        if not frozen.is_file() or not historical.is_file():
            self.skipTest('owner-private frozen source evidence unavailable here')
        data=frozen.read_bytes()
        cases=validated_gold(json.loads(data)["cases"])
        self.assertEqual(5,len(cases))
        self.assertEqual(11,sum(len(x["relevant_paths"]) for x in cases))
        assert_equivalent_gold(cases,json.loads(historical.read_bytes()),
                               fixed_gold_sha256=hashlib.sha256(data).hexdigest())

    def test_readonly_live_file_db_waits_for_brief_exclusive_writer(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'live.sqlite3'
            writer=sqlite3.connect(path)
            writer.execute('CREATE TABLE source (id INT)')
            writer.commit()
            writer.execute('BEGIN EXCLUSIVE')
            writer.execute('INSERT INTO source VALUES (17)')
            done=[]
            def read_after_writer():
                with closing(open_live_readonly_db(path,timeout_seconds=2.0)) as reader:
                    done.append(reader.execute('SELECT id FROM source').fetchone()[0])
                    with self.assertRaises(sqlite3.OperationalError):
                        reader.execute('INSERT INTO source VALUES (99)')
            thread=threading.Thread(target=read_after_writer)
            thread.start()
            time.sleep(0.15)
            writer.commit()
            thread.join(timeout=3.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual([17],done)
            writer.close()

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

    def test_modern_frozen_baseline_reports_real_rank_delta_without_nulls(self):
        case=validated_gold([{
            "id":"pbr","query":"Explain material roughness",
            "relevant_paths":["originals/pbrt.html","originals/epic.html"]}])[0]
        source_ok={"gold_status":"native_summaries_match_source_map_revision"}
        now=evaluate_case(case,lambda q:["not-related.html","more.html"],
                          lambda p:source_ok)
        baseline={"cases":1,"correct_top1":1,"correct_top10":1,
                  "known_relevant_source_identifiers":2,
                  "mean_known_relevant_source_recall_at10":0.5}
        result=score([now],baseline)
        self.assertEqual(1,result["historical_top1"])
        self.assertEqual(1,result["historical_top10"])
        self.assertEqual(-1,result["top1_difference_vs_historical"])
        self.assertEqual(-1,result["top10_difference_vs_historical"])
        self.assertEqual(0.5,result["historical_mean_known_relevant_source_recall_at10"])
        self.assertEqual(-0.5,result["known_relevant_source_recall_change_at10"])
        with self.assertRaisesRegex(ValueError,"contradict"):
            score([now],{**baseline,"top10_count":3})
        with self.assertRaisesRegex(ValueError,"population differs"):
            score([now],{**baseline,"known_relevant_source_identifiers":3})
        with self.assertRaisesRegex(ValueError,"population differs"):
            score([now],{**baseline,"cases":2})

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


class PublisherGoldAttestationTests(unittest.TestCase):
    def setUp(self):
        self.dir=TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        root=Path(self.dir.name)
        self.original=root/"original"
        self.derived=root/"derived"
        self.original.mkdir()
        self.derived.mkdir()
        self.rel="manuals/example.html"
        path=self.original/self.rel
        path.parent.mkdir(parents=True)
        path.write_text("<html><body><h1>Layer Weight</h1><p>Facing varies with surface angle and blends materials.</p></body></html>")
        target=self.derived/(self.rel+".txt")
        target.parent.mkdir(parents=True)
        target.write_text("[[SOURCE UNIT: HTML original publisher document]]\\n"
                          "Layer Weight Facing varies with surface angle and blends materials.")
        self.path=path
        self.target=target
        self.hash=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        self.db=sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.execute("CREATE TABLE originals (source_path TEXT,status TEXT,"
                        "source_sha256 TEXT,derived_sha256 TEXT,derived_path TEXT)")
        self.db.execute("INSERT INTO originals VALUES (?,?,?,?,?)",(
            self.rel,"complete",self.hash(path),self.hash(target),self.rel+".txt",
        ))
        self.quote="Facing varies with surface angle and blends materials."
        self.case={"id":"one","query":"How to use Blender view-angle shader weights?",
                   "relevant_paths":[self.rel],
                   "provenance":[{"path":self.rel,
                       "original_sha256":self.hash(path),
                       "derived_sha256":self.hash(target),
                       "publisher_anchor_normalized":self.quote.casefold(),
                       "publisher_anchor_sha256":hashlib.sha256(self.quote.encode()).hexdigest()}]}

    def attest(self):
        return attest_html_original_gold([self.case],{"one"},
            original_root=self.original,derivative_root=self.derived,
            source_map=self.db)

    def test_exact_original_and_derivative_bytes_are_rechecked(self):
        result=self.attest()["one"][self.rel]
        self.assertTrue(result["publisher_source_bytes_rechecked"])
        self.assertTrue(result["derived_source_bytes_rechecked"])
        self.assertTrue(result["literal_publisher_phrase_found_after_html_text_normalization"])
        self.assertEqual(self.hash(self.path),result["original_sha256"])

    def test_mutated_original_or_derived_fails_without_claiming_a_source(self):
        self.path.write_text(self.path.read_text()+"changed")
        with self.assertRaisesRegex(ValueError,"bytes differ"):
            self.attest()
        self.path.write_text(self.path.read_text()[:-7])
        assert self.hash(self.path)==self.case["provenance"][0]["original_sha256"]
        self.target.write_text(self.target.read_text()+"changed")
        with self.assertRaisesRegex(ValueError,"bytes differ"):
            self.attest()

    def test_relevant_phrase_missing_in_verified_original_fails(self):
        new="No supporting original text. "*2
        self.path.write_text(new)
        digest=self.hash(self.path)
        self.db.execute("UPDATE originals SET source_sha256=?",(digest,))
        self.case["provenance"][0]["original_sha256"]=digest
        with self.assertRaisesRegex(ValueError,"anchor not present"):
            self.attest()

    def test_symlink_and_extra_gold_label_are_rejected(self):
        self.path.unlink()
        self.path.symlink_to(self.target)
        with self.assertRaisesRegex(ValueError,"symlinked"):
            self.attest()
        self.path.unlink()
        self.path.write_text("<p>"+self.quote+"</p>")
        digest=self.hash(self.path)
        self.db.execute("UPDATE originals SET source_sha256=?",(digest,))
        self.case["provenance"][0]["original_sha256"]=digest
        self.case["relevant_paths"].append("manuals/other.html")
        with self.assertRaisesRegex(ValueError,"all independently attested"):
            self.attest()

    def test_nontext_pdf_source_must_use_separate_attestation(self):
        old=self.case["relevant_paths"][0]
        self.case["relevant_paths"]=["manuals/example.pdf"]
        self.case["provenance"][0]["path"]="manuals/example.pdf"
        with self.assertRaisesRegex(ValueError,"separate source-aware"):
            self.attest()


if __name__=="__main__":
    unittest.main()
