"""Source-linked recursive summary progress checkpoints without model/network calls."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from test_hierarchical_summary import FakeSummaryModel, long_document


def load_cli():
    path=Path(__file__).resolve().parents[2]/"scripts"/"raspi_hierarchical_library.py"
    spec=importlib.util.spec_from_file_location("progress_cli_test",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SourceNodeProgressTests(unittest.TestCase):
    def setUp(self):
        self.cli=load_cli()
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/"summaries"
        text=long_document(35,100)
        self.source={
            "original_path":"docs/long-example.pdf",
            "title":"Example full source",
            "original_sha256":"a"*64,
            "original_bytes":len(text.encode()),
            "derived_sha256":hashlib.sha256(text.encode()).hexdigest(),
            "derived_text":text,
        }
        self.args=SimpleNamespace(
            original_path=self.source["original_path"],
            model_url="http://127.0.0.1:14829",
            model="fixture-model",model_timeout=30,
            output_tokens=64,safety_tokens=16,max_levels=12,
            max_nodes=1000,max_calls=100,plan_only=False,
        )

    def progress_path(self):
        return self.cli.source_progress_path(
            self.source["original_path"],
            self.source["original_sha256"],
            self.source["derived_sha256"],
        )

    def test_printed_edition_is_disambiguated_and_kept_with_a_long_title(self):
        source = {**self.source, "title": "T" * 300, "edition_version": "Stand: Mai 2022",
                  "derived_text": 'Catalog: 2026\n[[SOURCE UNIT: PDF page 1]]\n  Stand: Mai 2022\n'}
        label = self.cli.summary_prompt_title(source)
        self.assertLessEqual(len(label), 300)
        self.assertTrue(label.startswith("T"))
        self.assertTrue(label.endswith("Stand: Mai 2022]"))
        self.assertEqual("T" * 300, source["title"])

    def test_catalog_only_partial_or_multiline_editions_do_not_override_source(self):
        source = {**self.source, "edition_version": "Stand: Mai 2022"}
        for body in ("Stand: Mai 2022\n",
                     "Stand: Mai 2022\n[[SOURCE UNIT: PDF page 1]]\nOther text",
                     "[[SOURCE UNIT: PDF page 1]]\nStand: Mai 2022 or later"):
            with self.subTest(body=body):
                self.assertEqual(source["title"], self.cli.summary_prompt_title(
                    {**source, "derived_text": "metadata\n" + body}))
        for edition in (None, 2022, "", "2022\nWrong", "A" * 101):
            self.assertEqual(source["title"], self.cli.summary_prompt_title(
                {**source, "edition_version": edition}))

    def test_complete_multilevel_saves_intermediate_node_progress_and_root(self):
        events=[]
        original=self.cli.save_source_progress
        def recorder(path,data):
            if data.get("status")=="summarizing":
                events.append((data.get("completed_model_calls"),
                               data.get("completed_first_level_nodes"),
                               data.get("first_level_percent")))
            return original(path,data)
        with mock.patch.object(self.cli,"OUT",self.root), \
             mock.patch.object(self.cli,"checked_original",return_value=self.source), \
             mock.patch.object(self.cli,"LocalSummaryModel",
                               return_value=FakeSummaryModel(750)), \
             mock.patch.object(self.cli,"save_source_progress",side_effect=recorder):
            self.cli.build(self.args)
            checkpoint=self.progress_path()
            receipt=json.loads(checkpoint.read_text())
            payload=json.loads(self.cli.payload_path(
                self.source["original_path"],self.source["original_sha256"],
                self.source["derived_sha256"],
            ).read_text())
            tree=self.cli.load_tree(payload)
        self.assertGreater(tree.root.level,0)
        self.assertEqual("complete_tree_saved",receipt["status"])
        self.assertEqual(len(tree.nodes),receipt["completed_model_calls"])
        self.assertEqual(sum(n.level==0 for n in tree.nodes),
                         receipt["completed_first_level_nodes"])
        self.assertEqual(100.0,receipt["first_level_percent"])
        self.assertEqual(len(self.source["derived_text"]),
                         receipt["completed_source_characters_in_leaf_pass"])
        self.assertEqual(tree.root.node_id,receipt["root_node_id"])
        self.assertEqual(0o600,stat.S_IMODE(checkpoint.stat().st_mode))
        self.assertEqual(0o700,stat.S_IMODE(checkpoint.parent.stat().st_mode))
        self.assertGreater(len(events),3)
        self.assertEqual(0,events[0][0])
        self.assertEqual(len(tree.nodes),events[-1][0])
        self.assertTrue(all("summary_text" not in key for key in receipt))

    def test_model_failure_cannot_claim_complete_root(self):
        model=FakeSummaryModel(750)
        with mock.patch.object(self.cli,"OUT",self.root), \
             mock.patch.object(self.cli,"checked_original",return_value=self.source), \
             mock.patch.object(self.cli,"LocalSummaryModel",return_value=model), \
             mock.patch.object(model,"generate_summary",
                               side_effect=RuntimeError("fixture model offline")):
            with self.assertRaisesRegex(RuntimeError,"offline"):
                self.cli.build(self.args)
            progress=json.loads(self.progress_path().read_text())
            self.assertEqual("incomplete_generation",progress["status"])
            self.assertIn("offline",progress["last_error"])
            self.assertFalse(self.cli.payload_path(
                self.source["original_path"],self.source["original_sha256"],
                self.source["derived_sha256"],
            ).exists())

    def test_atomic_progress_write_is_owner_only(self):
        with mock.patch.object(self.cli,"OUT",self.root):
            path=self.progress_path()
            self.cli.save_source_progress(path,{"status":"summarizing","completed_model_calls":2})
            self.cli.save_source_progress(path,{"status":"complete_tree_saved","completed_model_calls":3})
            self.assertEqual(3,json.loads(path.read_text())["completed_model_calls"])
            self.assertEqual([],list(path.parent.glob("*.tmp")))
            self.assertEqual(0o600,stat.S_IMODE(path.stat().st_mode))


if __name__=="__main__":
    unittest.main()
