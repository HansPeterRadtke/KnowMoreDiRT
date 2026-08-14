from __future__ import annotations

import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from context_capacity import context_token_capacity
from file_system_catalog.content_pipeline import GeneratedAnalysis, ModelContext
from file_system_catalog.content_schema import CHUNK_TABLE_NAME, REPRESENTATION_TABLE_NAME
from file_system_catalog.folder_assistant import (
    FolderQuestionAssistant,
    initialize_text_folder,
    normalize_plan,
    query_plan_schema,
)


class AssistantAnalysisClient:
    model = "assistant-fake-model"
    seed = 42
    temperature = 0.0

    def model_context(self) -> ModelContext:
        return ModelContext(configured_tokens=65536, trained_tokens=65536)

    def output_token_budget(
        self,
        *,
        ratio_names: tuple[str, ...] = (),
        ratio_default: float = 1.0 / 32.0,
    ) -> int:
        return context_token_capacity(
            self.model_context().configured_tokens,
            ratio_names=ratio_names,
            ratio_default=ratio_default,
        )

    def request_fits(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        worst_retry: bool = True,
    ) -> bool:
        return self.token_count(system) + self.token_count(user) + (max_tokens or 0) < self.model_context().configured_tokens

    def available_content_tokens(
        self,
        *,
        system: str,
        user_without_content: str,
        max_tokens: int,
    ) -> int:
        return self.model_context().configured_tokens - self.token_count(system) - self.token_count(user_without_content) - (max_tokens or 0)

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}

    def token_count(self, text: str) -> int:
        return max(1, math.ceil(len(text) / 4)) if text else 0

    @staticmethod
    def _action(action_type: str, query: str = "", **overrides: Any) -> dict[str, Any]:
        value = {
            "action_type": action_type,
            "purpose": "Find evidence for the question.",
            "query": query,
            "case_sensitive": False,
            "whole_word": False,
            "top_k": 10,
            "path_contains": "",
            "name_contains": "",
            "extension": "",
            "mime_prefix": "",
            "min_size_bytes": 0,
            "max_size_bytes": 0,
            "modified_after": "",
            "modified_before": "",
            "sort_by": "path",
            "limit": 30,
        }
        value.update(overrides)
        return value

    def complete(
        self,
        *,
        schema_name: str,
        schema: dict[str, Any],
        system: str,
        user: str,
        max_tokens: int,
    ) -> GeneratedAnalysis:
        if schema_name == "folder_query_plan":
            question = user.casefold()
            if "largest" in question:
                value = {
                    "answer_mode": "metadata",
                    "combine_mode": "union",
                    "actions": [self._action("metadata", sort_by="size_desc", limit=10)],
                    "rationale": "File size is metadata.",
                }
            elif "literally" in question or "exact phrase" in question:
                value = {
                    "answer_mode": "files",
                    "combine_mode": "union",
                    "actions": [
                        self._action(
                            "literal",
                            "Blue Lantern",
                            case_sensitive=False,
                            whole_word=False,
                        )
                    ],
                    "rationale": "The question requests exact wording.",
                }
            else:
                value = {
                    "answer_mode": "files",
                    "combine_mode": "union",
                    "actions": [self._action("semantic", "bicycles and off-road cycling")],
                    "rationale": "The question is conceptual.",
                }
        elif schema_name == "grounded_folder_answer":
            evidence = json.loads(user.split("Evidence:\n", 1)[1])
            selected = next((item for item in evidence if item.get("path")), None)
            if selected is None:
                value = {
                    "status": "not_found",
                    "answer": "No supported file was found.",
                    "files": [],
                    "citations": [],
                }
            else:
                evidence_id = selected["evidence_id"]
                path = selected["path"]
                value = {
                    "status": "answered",
                    "answer": f"The best supported file is {path} [{evidence_id}].",
                    "files": [
                        {
                            "path": path,
                            "reason": "It is the strongest returned evidence.",
                            "evidence_ids": [evidence_id],
                        }
                    ],
                    "citations": [
                        {
                            "evidence_id": evidence_id,
                            "claim": f"{path} supports the answer.",
                        }
                    ],
                }
        else:
            raise AssertionError(schema_name)
        return GeneratedAnalysis(
            value=value,
            response_metadata={
                "model": self.model,
                "finish_reason": "stop",
                "parsed": value,
            },
        )


class AssistantEmbeddingClient:
    model = "assistant-fake-embedding"
    revision = "assistant-fake-revision"

    def model_context(self) -> ModelContext:
        return ModelContext(configured_tokens=32768, trained_tokens=32768)

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        for text in texts:
            lowered = text.casefold()
            vector = np.zeros(64, dtype="<f4")
            if any(term in lowered for term in ("bicycle", "off-road", "trail cycle", "mountain bike")):
                vector[0] = 1.0
            elif any(term in lowered for term in ("tax", "receipt", "deduction")):
                vector[1] = 1.0
            elif "blue lantern" in lowered:
                vector[2] = 1.0
            else:
                vector[3] = 1.0
            result.append(vector)
        return result


class FolderAssistantTest(unittest.TestCase):
    def _folder(self, temporary: str) -> Path:
        root = Path(temporary, "folder")
        root.mkdir()
        Path(root, "trail_notes.txt").write_text(
            "A rugged off-road pedaling machine uses wide tires and low gears on rocky trails.\n",
            encoding="utf-8",
        )
        Path(root, "tax_records.txt").write_text(
            "Tax records preserve receipts, deductible expenses, and filing evidence.\n" * 2,
            encoding="utf-8",
        )
        Path(root, "harbor_protocol.txt").write_text(
            "Blue Lantern means grid power and the external network have both failed.\n",
            encoding="utf-8",
        )
        return root

    def test_plan_schema_is_closed_and_plan_is_bounded(self) -> None:
        schema = query_plan_schema()

        def inspect(value: Any) -> None:
            if not isinstance(value, dict):
                return
            if value.get("type") == "object":
                self.assertEqual(value.get("required"), list(value["properties"]))
                self.assertIs(value.get("additionalProperties"), False)
            for child in value.values():
                if isinstance(child, dict):
                    inspect(child)
                elif isinstance(child, list):
                    for item in child:
                        inspect(item)

        inspect(schema)
        raw = {
            "answer_mode": "files",
            "combine_mode": "union",
            "rationale": "test",
            "actions": [
                AssistantAnalysisClient._action("semantic", " bicycles ", top_k=500, limit=1000),
                AssistantAnalysisClient._action("semantic", " bicycles ", top_k=500, limit=1000),
            ],
        }
        context_size = 65536
        result_capacity = context_token_capacity(
            context_size,
            ratio_names=("KMD_FOLDER_RESULT_COUNT_RATIO",),
            ratio_default=1.0 / 1024.0,
        )
        plan = normalize_plan(raw, "question", context_size=context_size)
        self.assertEqual(len(plan["actions"]), 1)
        self.assertEqual(plan["actions"][0]["top_k"], result_capacity)
        self.assertEqual(plan["actions"][0]["limit"], result_capacity)
        inconsistent = normalize_plan(
            {
                "answer_mode": "metadata",
                "combine_mode": "independent",
                "rationale": "wrong mode",
                "actions": [AssistantAnalysisClient._action("semantic", "bicycles")],
            },
            "Which files are about bicycles?",
            context_size=context_size,
        )
        self.assertEqual(inconsistent["answer_mode"], "files")
        self.assertEqual(inconsistent["combine_mode"], "union")

    def test_atomic_initialization_and_all_three_query_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._folder(temporary)
            database = Path(temporary, "folder.sqlite3")
            analysis = AssistantAnalysisClient()
            embedding = AssistantEmbeddingClient()
            initialized = initialize_text_folder(
                root=root,
                database=database,
                analysis_client=analysis,
                embedding_client=embedding,
                collection_id="assistant-test",
                chunks_only=True,
            )
            self.assertEqual(initialized["status"], "ok")
            self.assertTrue(database.exists())
            self.assertFalse(any(Path(temporary).glob(".*.initialize.*")))
            connection = sqlite3.connect(database)
            try:
                self.assertEqual(
                    connection.execute(f"SELECT count(*) FROM {CHUNK_TABLE_NAME}").fetchone()[0],
                    3,
                )
                self.assertEqual(
                    connection.execute(f"SELECT count(*) FROM {REPRESENTATION_TABLE_NAME}").fetchone()[0],
                    0,
                )
                self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            finally:
                connection.close()

            assistant = FolderQuestionAssistant(
                root=root,
                database=database,
                analysis_client=analysis,
                embedding_client=embedding,
            )
            semantic = assistant.ask("Show me files about bicycles even when they use other words.")
            self.assertEqual(semantic["plan"]["actions"][0]["action_type"], "semantic")
            self.assertEqual(semantic["evidence"][0]["path"], "trail_notes.txt")
            self.assertEqual(semantic["result"]["status"], "answered")
            self.assertEqual(semantic["result"]["files"][0]["path"], "trail_notes.txt")
            self.assertIn("[E1]", semantic["result"]["answer"])

            literal = assistant.ask("Which file literally contains the exact phrase Blue Lantern?")
            self.assertEqual(literal["plan"]["actions"][0]["action_type"], "literal")
            literal_evidence = next(item for item in literal["evidence"] if item["path"])
            self.assertEqual(literal_evidence["path"], "harbor_protocol.txt")
            self.assertEqual(
                literal_evidence["start_char"],
                Path(root, "harbor_protocol.txt").read_text().index("Blue Lantern"),
            )

            metadata = assistant.ask("Which is the largest file?")
            self.assertEqual(metadata["plan"]["actions"][0]["action_type"], "metadata")
            metadata_file = next(item for item in metadata["evidence"] if item["path"])
            self.assertEqual(metadata_file["path"], "tax_records.txt")

    def test_database_inside_indexed_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._folder(temporary)
            with self.assertRaisesRegex(ValueError, "outside the indexed text root"):
                initialize_text_folder(
                    root=root,
                    database=root / "catalog.sqlite3",
                    analysis_client=AssistantAnalysisClient(),
                    embedding_client=AssistantEmbeddingClient(),
                    chunks_only=True,
                )
            self.assertFalse((root / "catalog.sqlite3").exists())

    def test_initialization_is_not_replaced_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._folder(temporary)
            database = Path(temporary, "folder.sqlite3")
            database.write_bytes(b"existing database sentinel")
            with self.assertRaises(FileExistsError):
                initialize_text_folder(
                    root=root,
                    database=database,
                    analysis_client=AssistantAnalysisClient(),
                    embedding_client=AssistantEmbeddingClient(),
                    chunks_only=True,
                    replace=False,
                )
            self.assertEqual(database.read_bytes(), b"existing database sentinel")


if __name__ == "__main__":
    unittest.main()
