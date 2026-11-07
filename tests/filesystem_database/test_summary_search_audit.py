"""Persistent original-KMD query/relevance audit safety and CLI behavior."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from file_system_catalog.summary_search_audit import (
    SCHEMA, append_query_audit, audit_path, make_search_audit, recent_search_audits,
)
import importlib.util


def load_cli():
    path = Path(__file__).resolve().parents[2] / "scripts" / "raspi_hierarchical_library.py"
    spec = importlib.util.spec_from_file_location("kmd_summary_cli_audit_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class SearchAuditTest(unittest.TestCase):
    def test_append_private_jsonl_readback_and_revisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp, "private-search")
            timestamp = datetime(2026, 11, 7, 12, 12, 12, tzinfo=timezone.utc)
            query = "What is the purpose of a normal map node in Blender?"
            result = {
                "documents": [
                    {
                        "original_path": "graphics/blender/normal-map-node.html",
                        "title": "Blender normal map node",
                        "similarity": .911,
                        "best_level": 0,
                        "best_summary": "Normal map changes how a 3D shader interprets surface normals.",
                        "indexed_summary_nodes_for_document": 1,
                        "matched_summary_nodes": [
                            {"original_path": "graphics/blender/normal-map-node.html",
                             "original_sha256": "a"*64, "summary_sha256": "b"*64,
                             "source_start_char": 0, "source_end_char": 120,
                             "summary_level": 0, "summary_kind": "summary_long",
                             "summary_node_id": "summary:1",
                             "similarity": .911,
                             "summary": "Normal map node to perturb normals."}
                        ],
                    }
                ],
            }
            receipt = make_search_audit(
                query=query, result=result, elapsed_seconds=.342,
                model="local-e5", database="/data/private/native-filedb.sqlite3",
            )
            filepath = append_query_audit(receipt, directory=root, now=timestamp)
            self.assertTrue(filepath.name.endswith("2026-11-07.jsonl"))
            self.assertEqual(0o600, stat.S_IMODE(filepath.stat().st_mode))
            self.assertEqual(0o700, stat.S_IMODE(root.stat().st_mode))
            with filepath.open() as handle:
                stored = json.loads(handle.readline())
            self.assertEqual(SCHEMA, stored["schema"])
            self.assertEqual(query, stored["query"])
            self.assertEqual("graphics/blender/normal-map-node.html", stored["ranked_results"][0]["original_path"])
            self.assertEqual(1, stored["ranked_results"][0]["rank"])
            self.assertEqual("b"*64, stored["ranked_results"][0]["matched_nodes"][0]["summary_sha256"])
            append_query_audit(receipt, directory=root, now=timestamp)
            logs = recent_search_audits(directory=root, limit=10)
            self.assertEqual(2, len(logs))
            self.assertEqual(query, logs[0]["query"])

    def test_symlink_and_bounded_inputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp, "safe")
            root.mkdir()
            f = root / "summary-search-2026-11-07.jsonl"
            outside = Path(tmp, "outside")
            outside.write_text("unchanged")
            f.symlink_to(outside)
            stamp = datetime(2026, 11, 7, tzinfo=timezone.utc)
            with self.assertRaises(OSError):
                append_query_audit({"query": "safe"}, directory=root, now=stamp)
            self.assertEqual("unchanged", outside.read_text())
            with self.assertRaises(ValueError):
                append_query_audit({"query": "x"*2001}, directory=root, now=stamp)

    def test_audit_marks_failed_search_explicitly(self):
        item = make_search_audit(
            query="unknown subject", result=None, elapsed_seconds=1.3,
            error="RuntimeError: invalid model response", model="E5", database="canonical",
        )
        self.assertEqual(0, item["count"])
        self.assertFalse(item["success"])
        self.assertEqual([], item["ranked_results"])

    def test_public_cli_search_automatically_logs_result(self):
        mod = load_cli()
        response = {
            "query": "Blender normal map",
            "documents": [{"title": "Normal", "original_path": "graphics/normal.pdf",
                           "best_summary": "Normal maps", "similarity": 0.99,
                           "best_level": 0, "indexed_summary_nodes_for_document": 1,
                           "matched_summary_nodes": []}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(mod, "_search_impl", return_value=response), \
                 mock.patch.object(mod, "append_query_audit",
                                   side_effect=lambda x: append_query_audit(x, directory=Path(tmp))) as saved, \
                 mock.patch("builtins.print") as printed:
                mod.search(SimpleNamespace(query="Blender normal map"))
            saved.assert_called_once()
            raw = recent_search_audits(directory=Path(tmp), limit=1)
            self.assertEqual(1, len(raw))
            self.assertEqual("Blender normal map", raw[0]["query"])
            self.assertEqual("graphics/normal.pdf", raw[0]["ranked_results"][0]["original_path"])
            self.assertTrue(printed.called)

    def test_public_cli_search_logs_failure(self):
        mod = load_cli()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(mod, "_search_impl", side_effect=RuntimeError("bad local E5")), \
                 mock.patch.object(mod, "append_query_audit",
                                   side_effect=lambda x: append_query_audit(x, directory=Path(tmp))):
                with self.assertRaisesRegex(RuntimeError, "bad local E5"):
                    mod.search(SimpleNamespace(query="machine-learning methods"))
            raw = recent_search_audits(directory=Path(tmp), limit=1)
            self.assertEqual(1, len(raw))
            self.assertFalse(raw[0]["success"])
            self.assertEqual(0, raw[0]["count"])


if __name__ == "__main__":
    unittest.main()
