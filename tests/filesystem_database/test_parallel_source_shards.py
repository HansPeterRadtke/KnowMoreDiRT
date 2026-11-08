"""Strict source partition tests for two simultaneous native KMD workers.

These tests use the original fake source map without modifying production
FileDB or spawning either backend. The partition key is the original path,
independent of the source content revision or running worker model.
"""
import hashlib, importlib.util, json, tempfile, unittest
from pathlib import Path
from unittest import mock


def load_worker():
    path=Path(__file__).resolve().parents[2]/"scripts"/"raspi_hierarchical_worker.py"
    spec=importlib.util.spec_from_file_location("kmd_parallel_shard_test",path)
    x=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(x)
    return x


class ParallelShardTests(unittest.TestCase):
    def test_full_corpus_split_has_no_duplicate_or_missing_original(self):
        mod=load_worker()
        items=[f"library/section_{i:05d}.pdf" for i in range(17714)]
        shard0={p for p in items if mod.shard_for_original(p,2)==0}
        shard1={p for p in items if mod.shard_for_original(p,2)==1}
        self.assertFalse(shard0&shard1)
        self.assertEqual(set(items),shard0|shard1)
        self.assertGreater(len(shard0),7500)
        self.assertGreater(len(shard1),7500)
        for p in items[:200]:
            self.assertEqual(mod.shard_for_original(p,2),
                             mod.shard_for_original(p,2))
        for n in (1,2,5):
            self.assertEqual(n, len({mod.shard_for_original(p,n)
                                     for p in items[:10000]}))
        with self.assertRaises(ValueError):
            mod.shard_for_original("document.pdf",0)
        with self.assertRaises(ValueError):
            mod.shard_for_original("../untrusted.pdf",33)

    def test_weighted_quarter_three_quarters_partition_is_disjoint(self):
        mod=load_worker()
        left=mod.parse_assigned_shards(4,0,None)
        right=mod.parse_assigned_shards(4,1,"1,2,3")
        self.assertEqual(frozenset({0}),left)
        self.assertEqual(frozenset({1,2,3}),right)
        self.assertEqual(set(),left & right)
        self.assertEqual({0,1,2,3},left | right)
        docs=[f"technical/volume_{i:05d}.pdf" for i in range(17714)]
        a={p for p in docs if mod.shard_for_original(p,4) in left}
        b={p for p in docs if mod.shard_for_original(p,4) in right}
        self.assertFalse(a & b)
        self.assertEqual(set(docs),a | b)
        self.assertTrue(3500<len(a)<5500)
        for invalid in ("0,0","0,4","1,-1","not-a-shard","","0;1"):
            with self.assertRaises(ValueError, msg=invalid):
                mod.parse_assigned_shards(4,0,invalid)

    def test_two_fake_workers_have_disjoint_locks_status_and_source_selection(self):
        mod=load_worker()
        docs=[f"docs/{i:03d}.pdf" for i in range(50)]
        available=[(p,"a"*64,"b"*64,1000+i)
                   for i,p in enumerate(docs)]
        seen={}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            db=root/"filedb.sqlite3";src=root/"source.sqlite3";cli=root/"library_cli.py"
            for p in (db,src,cli):p.write_text("fixture")
            for shard in (0,1):
                entries=[]
                # Keep CLI, source map and source database the same,
                # but configure a different persistent lock/status per worker.
                with mock.patch.object(mod,"RUNROOT",root), \
                     mock.patch.object(mod,"STATE",root/"hierarchical-worker-status.json"), \
                     mock.patch.object(mod,"LOCK",root/".hierarchical-worker.lock"), \
                     mock.patch.object(mod,"SOURCE_MAP",src), \
                     mock.patch.object(mod,"FILEDB",db), \
                     mock.patch.object(mod,"CLI",cli), \
                     mock.patch.object(mod,"scope",return_value=(available,set())), \
                     mock.patch.object(mod,"run_command",
                         side_effect=lambda parts,label,path:entries.append((parts,label,path))):
                    mod.run(
                       sleep_seconds=0,max_calls=40,once=True,dry_run=True,
                       model_url="http://10.8.0.7:14829" if shard==0
                                 else "http://10.8.0.7:14833",
                       model_name="standard" if shard==0 else "coder",
                       output_tokens=1600 if shard==0 else 2400,
                       context_cap_tokens=16384,
                       shared_slot_idle_seconds=0 if shard==0 else 8,
                       shard_count=4,shard_index=shard,
                       assigned_shards=None if shard==0 else "1,2,3",
                       worker_label="standard" if shard==0 else "",
                    )
                self.assertEqual(1,len(entries))
                args,label,path=entries[0]
                owner=mod.shard_for_original(path,4)
                self.assertTrue((owner==0) if shard==0 else owner in (1,2,3))
                self.assertEqual("PLAN",label)
                key="hierarchical-worker-standard-status.json" if shard==0 else "hierarchical-worker-status.json"
                status=json.loads((root/key).read_text())
                self.assertEqual(shard,status["shard_index"])
                self.assertEqual(4,status["shard_count"])
                self.assertEqual([0] if shard==0 else [1,2,3],status["assigned_shards"])
                self.assertEqual("summary_and_keywords_single_constrained_model_call",status["native_facet_policy"])
                self.assertGreater(status["assigned_eligible_originals"],0)
                seen[shard]=path
            self.assertNotEqual(seen[0],seen[1])
            self.assertTrue((root/".hierarchical-worker.lock").is_file())
            self.assertTrue((root/".hierarchical-worker-standard.lock").is_file())

    def test_two_concurrent_native_filedb_transactions_keep_all_sources(self):
        import base64, concurrent.futures, sqlite3,threading
        from test_hierarchical_summary_store import fixture,FakeEmbeddings
        from test_hierarchical_summary import FakeSummaryModel,long_document
        from file_system_catalog.hierarchical_summary import HierarchicalSummarizer
        from file_system_catalog.hierarchical_summary_store import summary_rows,append_summary_rows
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"shared-native-original.sqlite3"
            setup=fixture()
            original_b="books/other-guide.pdf"
            setup.execute(
                "INSERT INTO filesystem_entries VALUES (2,?,?,?,?)",
                (original_b,base64.b64encode(original_b.encode()).decode(),
                 "file",12345),
            )
            setup.commit()
            dest=__import__("sqlite3").connect(path)
            setup.backup(dest)
            dest.close();setup.close()
            start=threading.Barrier(2)
            def writer(original,seed):
                db=sqlite3.connect(path,timeout=10)
                db.execute("PRAGMA foreign_keys=ON")
                text=long_document(8,60)
                tree=HierarchicalSummarizer(
                    FakeSummaryModel(1200),output_tokens=64,
                    safety_tokens=16,
                ).summarize(title=original,text=text)
                chunks,reps=summary_rows(
                    db,original_path=original,original_sha256=str(seed)*64,
                    original_bytes=12345,
                    derived_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    derived_text=text,summary_start_char=0,tree=tree,
                    embedding_client=FakeEmbeddings(),
                    analysis_model="fixture-model",collection_id="two-worker-test",
                )
                start.wait(timeout=5)
                inserted=append_summary_rows(
                    db,chunk_rows=chunks,representation_rows=reps,
                    max_database_bytes=50*1024**2,
                )["new_rows"]
                db.close()
                return inserted,len(chunks)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                a=pool.submit(writer,"books/guide.pdf","a")
                b=pool.submit(writer,original_b,"b")
                got_a=a.result(timeout=15)
                got_b=b.result(timeout=15)
            self.assertGreater(got_a[0],0)
            self.assertGreater(got_b[0],0)
            db=sqlite3.connect(path)
            self.assertEqual("ok",db.execute("pragma quick_check").fetchone()[0])
            self.assertEqual([],db.execute("pragma foreign_key_check").fetchall())
            counted=db.execute(
                "SELECT f.relative_path_display,COUNT(*) FROM content_representations r "
                "JOIN content_chunks c on r.chunk_id=c.chunk_id "
                "JOIN filesystem_entries f on c.filesystem_entry_id=f.id "
                "GROUP BY f.relative_path_display ORDER BY 1"
            ).fetchall()
            self.assertEqual(
                {"books/guide.pdf","books/other-guide.pdf"},
                {row[0] for row in counted},
            )
            self.assertTrue(all(count>0 for _,count in counted))
            db.close()

    def test_sharding_is_independent_of_original_revision_and_filename_category(self):
        mod=load_worker()
        for path in (
          "money-and-business/report.xlsx",
          "visual-perception/guide.pdf",
          "unity-engine/unity-manual.html",
          "graphics/long-book.epub",
        ):
            self.assertEqual(mod.shard_for_original(path,2),mod.shard_for_original(path,2))
            self.assertEqual(0,mod.shard_for_original(path,1))
        self.assertNotEqual(mod.shard_for_original("same/guide.txt",2),
                            mod.shard_for_original("same/guide.txt",2)^1)


if __name__=="__main__":
    unittest.main()
