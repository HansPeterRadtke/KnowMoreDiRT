from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch

import numpy as np

from context_capacity import context_token_capacity

from file_system_catalog.content_pipeline import (
    AnalysisClient,
    Chunk,
    ContentSemanticPipeline,
    GeneratedAnalysis,
    ModelContext,
    chunk_analysis_schema_for_keys,
    chunk_text,
    migrate_legacy_content_schema,
    normalize_analysis,
    search_literal_chunks,
    search_semantic_entries,
    stable_chunk_id,
    stable_file_id,
    stable_representation_id,
    vector_from_blob,
)
from file_system_catalog.content_schema import (
    CHUNK_TABLE_NAME,
    REPRESENTATION_TABLE_NAME,
)
from file_system_catalog.scanner import FilesystemScanner


class FakeAnalysisClient:
    model = "fake-analysis-model"
    seed = 42

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
        return self.token_count(system) + self.token_count(user) + max_tokens < self.model_context().configured_tokens

    def available_content_tokens(
        self,
        *,
        system: str,
        user_without_content: str,
        max_tokens: int,
    ) -> int:
        return self.model_context().configured_tokens - self.token_count(system) - self.token_count(user_without_content) - max_tokens

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}

    def token_count(self, text: str) -> int:
        return max(1, math.ceil(len(text) / 4)) if text else 0

    def complete(
        self,
        *,
        schema_name: str,
        schema: dict[str, Any],
        system: str,
        user: str,
        max_tokens: int,
    ) -> GeneratedAnalysis:
        if schema_name in {"chunk_facet_analyses", "single_chunk_facet_analysis"}:
            analyses = []
            for match in re.finditer(r'<chunk key="([^"]+)"[^>]*>\n(.*?)\n</chunk>', user, re.DOTALL):
                key, text = match.group(1), match.group(2)
                facets = [
                    {
                        "label": "mountain biking",
                        "strength": "essential",
                        "representations": [
                            {
                                "kind": "description",
                                "strength": "essential",
                                "text": "Mountain bikes use wide tires and low gearing on rough trails.",
                            },
                            {
                                "kind": "keyphrase",
                                "strength": "strong",
                                "text": "off-road cycling",
                            },
                        ],
                    }
                ]
                if "tax" in text.casefold():
                    facets.append(
                        {
                            "label": "tax compliance",
                            "strength": "strong",
                            "representations": [
                                {
                                    "kind": "sentence",
                                    "strength": "strong",
                                    "text": "Tax records preserve receipts and filing evidence.",
                                }
                            ],
                        }
                    )
                if "many representations" in text.casefold():
                    facets.append(
                        {
                            "label": "dense retrieval vocabulary",
                            "strength": "moderate",
                            "representations": [
                                {
                                    "kind": "keyphrase",
                                    "strength": "weak",
                                    "text": f"distinct retrieval phrase {index:03d}",
                                }
                                for index in range(70)
                            ],
                        }
                    )
                analyses.append(
                    {
                        "chunk_key": key,
                        "document_summary": f"Chunk {key} concerns trail cycling.",
                        "facets": facets,
                    }
                )
            value = {"analyses": analyses}
        elif schema_name == "file_facet_analysis":
            value = {
                "document_summary": "The file combines trail cycling and record-keeping subjects.",
                "facets": [
                    {
                        "label": "trail cycling",
                        "strength": "essential",
                        "representations": [
                            {
                                "kind": "description",
                                "strength": "essential",
                                "text": "The file contains extensive mountain-biking guidance.",
                            }
                        ],
                    },
                    {
                        "label": "record keeping",
                        "strength": "strong",
                        "representations": [
                            {
                                "kind": "keyphrase",
                                "strength": "strong",
                                "text": "tax receipt records",
                            }
                        ],
                    },
                ],
            }
        else:
            raise AssertionError(schema_name)
        return GeneratedAnalysis(
            value=value,
            response_metadata={
                "model": self.model,
                "system_fingerprint": "fake-fingerprint",
                "finish_reason": "stop",
                "parsed": value,
            },
        )


class MissingBatchItemAnalysisClient(FakeAnalysisClient):
    def complete(self, **kwargs: Any) -> GeneratedAnalysis:
        generated = super().complete(**kwargs)
        if kwargs["schema_name"] == "chunk_facet_analyses" and len(generated.value["analyses"]) > 1:
            value = {"analyses": generated.value["analyses"][:1]}
            metadata = dict(generated.response_metadata)
            metadata["parsed"] = value
            return GeneratedAnalysis(value=value, response_metadata=metadata)
        return generated


class FakeEmbeddingClient:
    model = "fake-embedding-model"
    revision = "fake-revision"

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}

    def model_context(self):
        from file_system_catalog.content_pipeline import ModelContext
        return ModelContext(configured_tokens=32768, trained_tokens=32768)

    def token_count(self, text: str, *, add_special: bool = True) -> int:
        return (max(1, math.ceil(len(text) / 4)) if text else 0) + int(add_special)

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        result = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vector = np.frombuffer(digest * 8, dtype=np.uint8).astype(np.float32)[:64] - 127.5
            result.append(np.asarray(vector / np.linalg.norm(vector), dtype="<f4"))
        return result


class KeywordEmbeddingClient(FakeEmbeddingClient):
    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        result = []
        for text in texts:
            vector = np.zeros(64, dtype="<f4")
            vector[0 if "zephyrquokka" in text.casefold() else 1] = 1.0
            result.append(vector)
        return result


class ContentPipelineTest(unittest.TestCase):
    def _build_catalog(self, root: Path, database: Path) -> None:
        FilesystemScanner(root, progress_every=0).scan_to_database(database)

    def test_constraint_schema_is_closed_and_uses_portable_enums(self) -> None:
        schema = chunk_analysis_schema_for_keys(["0", "1"])

        def inspect(value: Any) -> None:
            if not isinstance(value, dict):
                return
            if value.get("type") == "object":
                self.assertIn("properties", value)
                self.assertEqual(value.get("required"), list(value["properties"]))
                self.assertIs(value.get("additionalProperties"), False)
            self.assertNotIn("const", value)
            for forbidden in ("minimum", "maximum"):
                self.assertNotIn(forbidden, value)
            for child in value.values():
                if isinstance(child, dict):
                    inspect(child)
                elif isinstance(child, list):
                    for item in child:
                        inspect(item)

        inspect(schema)
        analyses_schema = schema["properties"]["analyses"]
        self.assertEqual(analyses_schema["minItems"], 2)
        self.assertEqual(analyses_schema["maxItems"], 2)
        key_schema = analyses_schema["items"]["properties"]["chunk_key"]
        self.assertEqual(key_schema["enum"], ["0", "1"])

    def test_normalized_pipeline_stores_chunk_metadata_once_and_unlimited_children(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary, "root")
            root.mkdir()
            Path(root, "dense.txt").write_text(
                "Many representations. Mountain biking guidance with tax records.\n",
                encoding="utf-8",
            )
            database = Path(temporary, "catalog.sqlite3")
            self._build_catalog(root, database)
            pipeline = ContentSemanticPipeline(
                database=database,
                root=root,
                collection_id="normalized-test",
                analysis_client=FakeAnalysisClient(),
                embedding_client=FakeEmbeddingClient(),
            )
            first = pipeline.run()
            self.assertEqual(first["processed_files"], 1)
            connection = sqlite3.connect(database)
            connection.row_factory = sqlite3.Row
            try:
                tables = sorted(
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    )
                )
                self.assertEqual(tables, ["content_chunks", "content_representations", "filesystem_entries"])
                chunk_columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({CHUNK_TABLE_NAME})")}
                representation_columns = {
                    row["name"] for row in connection.execute(f"PRAGMA table_info({REPRESENTATION_TABLE_NAME})")
                }
                self.assertIn("start_char", chunk_columns)
                self.assertIn("word_count", chunk_columns)
                self.assertIn("embedding_blob", chunk_columns)
                self.assertNotIn("start_char", representation_columns)
                self.assertNotIn("file_id", representation_columns)
                chunks = list(connection.execute(f"SELECT * FROM {CHUNK_TABLE_NAME}"))
                self.assertEqual(len(chunks), 1)
                self.assertEqual(chunks[0]["chunk_kind"], "chunk")
                self.assertEqual(chunks[0]["start_char"], 0)
                self.assertEqual(chunks[0]["end_char"], len(Path(root, "dense.txt").read_text()))
                self.assertGreater(chunks[0]["word_count"], 0)
                self.assertEqual(len(chunks[0]["embedding_blob"]), 64 * 4)
                representations = list(connection.execute(f"SELECT * FROM {REPRESENTATION_TABLE_NAME}"))
                self.assertGreater(len(representations), 64)
                topic_rows = [row for row in representations if row["representation_kind"] == "topic"]
                self.assertEqual(
                    {row["representation_text"] for row in topic_rows},
                    {"mountain biking", "tax compliance", "dense retrieval vocabulary"},
                )
                self.assertTrue(all(row["item_rank"] == 0 for row in topic_rows))
                self.assertEqual({row["chunk_id"] for row in representations}, {chunks[0]["chunk_id"]})
                self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
                before_chunk_ids = [row["chunk_id"] for row in chunks]
                before_rep_ids = [row["representation_id"] for row in representations]
                before_created = {
                    row["representation_id"]: row["created_at_ns"] for row in representations
                }
            finally:
                connection.close()
            second = pipeline.run()
            self.assertEqual(second["chunk_rows"], first["chunk_rows"])
            self.assertEqual(second["representation_rows"], first["representation_rows"])
            connection = sqlite3.connect(database)
            connection.row_factory = sqlite3.Row
            try:
                self.assertEqual(
                    [row["chunk_id"] for row in connection.execute(f"SELECT * FROM {CHUNK_TABLE_NAME} ORDER BY chunk_id")],
                    sorted(before_chunk_ids),
                )
                after = list(connection.execute(f"SELECT * FROM {REPRESENTATION_TABLE_NAME} ORDER BY representation_id"))
                self.assertEqual([row["representation_id"] for row in after], sorted(before_rep_ids))
                self.assertEqual(
                    {row["representation_id"]: row["created_at_ns"] for row in after}, before_created
                )
            finally:
                connection.close()

    def test_literal_search_reads_canonical_text_and_semantic_search_uses_chunk_vector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary, "root")
            root.mkdir()
            text = "A technical off-road pedaling machine crosses a rocky trail. ZephyrQuokka appears once.\n"
            Path(root, "source.txt").write_text(text, encoding="utf-8")
            database = Path(temporary, "catalog.sqlite3")
            self._build_catalog(root, database)
            pipeline = ContentSemanticPipeline(
                database=database,
                root=root,
                collection_id="search-test",
                analysis_client=FakeAnalysisClient(),
                embedding_client=KeywordEmbeddingClient(),
            )
            pipeline.run()
            connection = sqlite3.connect(database)
            connection.row_factory = sqlite3.Row
            try:
                literal = search_literal_chunks(connection, root, "zephyrquokka", whole_word=True)
                self.assertEqual(len(literal), 1)
                self.assertEqual(literal[0]["match_start_char"], text.index("ZephyrQuokka"))
                query = KeywordEmbeddingClient().embed(["zephyrquokka"])[0]
                semantic = search_semantic_entries(connection, query)
                self.assertEqual(semantic[0]["relative_path_display"], "source.txt")
                self.assertEqual(semantic[0]["analysis_kind"], "chunk")
                self.assertAlmostEqual(semantic[0]["score"], 1.0)
            finally:
                connection.close()

    def test_missing_batch_analysis_is_recovered_individually(self) -> None:
        client = MissingBatchItemAnalysisClient()
        pipeline = ContentSemanticPipeline(
            database="/unused/catalog.sqlite3",
            root="/unused/root",
            collection_id="recovery",
            analysis_client=client,
            embedding_client=FakeEmbeddingClient(),
        )
        chunks = [
            Chunk(0, 0, 20, "Mountain biking one", 5),
            Chunk(1, 20, 40, "Mountain biking two", 5),
        ]
        analyses, metadata = pipeline._analyze_chunks(chunks, "duplicate.txt")
        self.assertEqual(len(analyses), 2)
        self.assertFalse(metadata[0]["batch_recovery"])
        self.assertTrue(metadata[1]["batch_recovery"])

    def test_strength_words_are_not_accepted_as_facet_names(self) -> None:
        normalized = normalize_analysis(
            {
                "document_summary": "A harbor protocol controls courier departure.",
                "facets": [
                    {
                        "facet_name": "essential",
                        "facet_strength": "essential",
                        "representations": [
                            {
                                "kind": "sentence",
                                "item_strength": "essential",
                                "text": "The written procedure and current measurements must agree before departure.",
                            }
                        ],
                    }
                ],
            }
        )
        self.assertNotEqual(normalized["facets"][0]["label"], "essential")
        self.assertEqual(
            normalized["facets"][0]["label"],
            "The written procedure and current measurements must agree before departure.",
        )
        schema = chunk_analysis_schema_for_keys(["0"])
        facet_properties = (
            schema["properties"]["analyses"]["items"]["properties"]["facets"]["items"]["properties"]
        )
        self.assertIn("facet_name", facet_properties)
        self.assertIn("facet_strength", facet_properties)
        representation_properties = facet_properties["representations"]["items"]["properties"]
        self.assertIn("item_strength", representation_properties)
        self.assertNotIn("strength", representation_properties)

    def test_gibberish_and_one_word_normalization(self) -> None:
        gibberish = normalize_analysis({"document_summary": "No coherent content.", "facets": []})
        self.assertEqual(gibberish["facets"], [])
        word = normalize_analysis(
            {
                "document_summary": "Bicycle.",
                "facets": [
                    {
                        "label": "bicycle",
                        "strength": "essential",
                        "representations": [
                            {"kind": "keyword", "strength": "essential", "text": "bicycle"}
                        ],
                    }
                ],
            }
        )
        self.assertEqual(word["facets"][0]["representations"][0]["text"], "bicycle")

    def test_stable_identifiers(self) -> None:
        file_id = stable_file_id("collection", base64.b64encode(b"a.txt").decode())
        self.assertEqual(file_id, stable_file_id("collection", base64.b64encode(b"a.txt").decode()))
        chunk_id = stable_chunk_id(file_id, "abc", "chunk", 0, 0, 10, "def")
        self.assertEqual(chunk_id, stable_chunk_id(file_id, "abc", "chunk", 0, 0, 10, "def"))
        representation_id = stable_representation_id(chunk_id, "a", "p", "e", 1)
        self.assertEqual(representation_id, stable_representation_id(chunk_id, "a", "p", "e", 1))

    def test_length_retry_increases_output_budget(self) -> None:
        responses = [
            {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"document_summary":"ok","facets":[]}'},
                    }
                ],
                "model": "fake",
            },
        ]
        payloads = []

        def fake_request(url: str, payload: dict[str, Any] | None = None, *, timeout: int = 600) -> dict[str, Any]:
            if url.endswith("/v1/models"):
                return {"data": [{"id": "fake", "meta": {"n_ctx": 4096, "n_ctx_train": 4096}}]}
            if url.endswith("/apply-template"):
                assert payload is not None
                return {"prompt": "rendered prompt"}
            if url.endswith("/tokenize"):
                return {"tokens": list(range(12))}
            assert payload is not None
            payloads.append(payload)
            return responses.pop(0)

        def fake_completion_request(
            url: str,
            payload: dict[str, Any],
            *,
            per_token_timeout_seconds: float,
        ) -> dict[str, Any]:
            self.assertEqual(per_token_timeout_seconds, 180)
            self.assertIs(payload["stream"], True)
            payloads.append(payload)
            response = responses.pop(0)
            choice = response["choices"][0]
            return {
                "content": choice.get("message", {}).get("content", ""),
                "reasoning_content": choice.get("message", {}).get("reasoning_content", ""),
                "finish_reason": choice.get("finish_reason"),
                "model": response.get("model"),
                "stream": True,
            }

        client = AnalysisClient("http://unused", model="fake", retries=2)
        with patch(
            "file_system_catalog.content_pipeline.request_json", side_effect=fake_request
        ), patch(
            "file_system_catalog.content_pipeline.stream_chat_completion_json",
            side_effect=fake_completion_request,
        ):
            result = client.complete(
                schema_name="x",
                schema={
                    "type": "object",
                    "properties": {
                        "document_summary": {"type": "string", "x-kmd-string-profile": "reason"},
                        "facets": {"type": "array", "x-kmd-array-profile": "dense", "items": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
                    },
                    "required": ["document_summary", "facets"],
                    "additionalProperties": False,
                },
                system="x",
                user="x",
                max_tokens=100,
            )
        self.assertEqual(result.value["document_summary"], "ok")
        self.assertEqual([payload["max_tokens"] for payload in payloads], [100, 200])
        self.assertTrue(all(payload["provider"]["require_parameters"] for payload in payloads))
        self.assertTrue(all(payload["reasoning_format"] == "deepseek" for payload in payloads))
        self.assertTrue(all(payload["reasoning_budget"] == 0 for payload in payloads))
        self.assertTrue(all(payload["enable_thinking"] is False for payload in payloads))


    def test_length_growth_is_not_limited_by_transient_retry_count(self) -> None:
        responses = [
            {"finish_reason": "length", "content": "{}"},
            {"finish_reason": "length", "content": "{}"},
            {"finish_reason": "length", "content": "{}"},
            {
                "finish_reason": "stop",
                "content": '{"document_summary":"complete","facets":[]}',
                "model": "fake",
            },
        ]
        payloads: list[dict[str, Any]] = []

        def fake_request(url: str, payload: dict[str, Any] | None = None, *, timeout: float | None = 600) -> dict[str, Any]:
            if url.endswith("/v1/models"):
                return {"data": [{"id": "fake", "meta": {"n_ctx": 4096, "n_ctx_train": 4096}}]}
            if url.endswith("/apply-template"):
                return {"prompt": "rendered prompt"}
            if url.endswith("/tokenize"):
                return {"tokens": list(range(12))}
            raise AssertionError(url)

        def fake_completion(
            url: str,
            payload: dict[str, Any],
            *,
            per_token_timeout_seconds: float,
        ) -> dict[str, Any]:
            payloads.append(payload)
            return responses.pop(0)

        client = AnalysisClient("http://unused", model="fake", retries=1)
        with patch("file_system_catalog.content_pipeline.request_json", side_effect=fake_request), patch(
            "file_system_catalog.content_pipeline.stream_chat_completion_json",
            side_effect=fake_completion,
        ):
            result = client.complete(
                schema_name="x",
                schema={
                    "type": "object",
                    "properties": {
                        "document_summary": {"type": "string", "x-kmd-string-profile": "reason"},
                        "facets": {"type": "array", "x-kmd-array-profile": "dense", "items": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
                    },
                    "required": ["document_summary", "facets"],
                    "additionalProperties": False,
                },
                system="x",
                user="x",
                max_tokens=100,
            )

        assert result.value["document_summary"] == "complete"
        assert [payload["max_tokens"] for payload in payloads] == [100, 200, 400, 800]
        assert result.response_metadata["output_budget_index"] == 4
        assert result.response_metadata["output_budget_count"] > client.retries

    def test_chunking_preserves_ranges(self) -> None:
        text = "\n\n".join("Mountain biking material. " * 150 for _ in range(80))
        chunks = chunk_text(text, FakeAnalysisClient())
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].start_char, 0)
        self.assertEqual(chunks[-1].end_char, len(text))
        self.assertTrue(all(chunk.token_count <= 10500 for chunk in chunks))


    def test_chunking_uses_context_relative_embedding_input_share(self) -> None:
        from file_system_catalog.content_pipeline import ModelContext, chunk_text

        class Analysis:
            def model_context(self) -> ModelContext:
                return ModelContext(configured_tokens=1000, trained_tokens=1000)

            def token_count(self, text: str) -> int:
                return len(text.split())

        class Embedding:
            def model_context(self) -> ModelContext:
                return ModelContext(configured_tokens=100, trained_tokens=100)

            def token_count(self, text: str, *, add_special: bool = True) -> int:
                return len(text.split())

        source = " ".join(f"token{index}" for index in range(80))
        chunks = chunk_text(
            source,
            Analysis(),  # type: ignore[arg-type]
            embedding_client=Embedding(),  # type: ignore[arg-type]
            max_tokens=500,
            max_chars=len(source),
            target_chars=len(source),
            overlap_chars=0,
        )

        assert len(chunks) > 1
        assert all(len(chunk.text.split()) <= 25 for chunk in chunks)
        assert "".join("".join(chunk.text.split()) for chunk in chunks) == "".join(source.split())

    def test_embedding_request_uses_finite_response_deadline(self) -> None:
        client = __import__(
            "file_system_catalog.content_pipeline", fromlist=["EmbeddingClient"]
        ).EmbeddingClient(
            "http://unused", model="fake", revision="r", expected_dimension=2,
            request_timeout_seconds=23, retries=1,
        )
        client._model_context = __import__(
            "file_system_catalog.content_pipeline", fromlist=["ModelContext"]
        ).ModelContext(configured_tokens=100, trained_tokens=100)
        with patch.object(client, "_validate_input", return_value=2), patch(
            "file_system_catalog.content_pipeline.request_json",
            return_value={"data": [{"index": 0, "embedding": [1.0, 0.0]}]},
        ) as request:
            vectors = client.embed(["x"])
        self.assertEqual(len(vectors), 1)
        self.assertEqual(request.call_args.kwargs["timeout"], 23.0)


    def test_embedding_batches_respect_context_relative_total_token_budget(self) -> None:
        from file_system_catalog.content_pipeline import EmbeddingClient, ModelContext

        client = EmbeddingClient(
            "http://unused",
            model="fake",
            revision="r",
            expected_dimension=2,
            request_timeout_seconds=23,
            retries=1,
        )
        client._model_context = ModelContext(configured_tokens=80, trained_tokens=80)
        payloads: list[list[str]] = []

        def fake_request(url: str, payload: dict[str, Any] | None = None, *, timeout: float | None = 600) -> dict[str, Any]:
            assert payload is not None
            batch = list(payload["input"])
            payloads.append(batch)
            return {
                "data": [
                    {"index": index, "embedding": [1.0, 0.0]}
                    for index, _value in enumerate(batch)
                ]
            }

        with patch.object(client, "_validate_input", return_value=6), patch(
            "file_system_catalog.content_pipeline.request_json", side_effect=fake_request
        ):
            vectors = client.embed(["a", "b", "c", "d"])

        assert len(vectors) == 4
        assert payloads == [["a"], ["b"], ["c"], ["d"]]

    def test_embedding_rejects_oversized_input_before_request(self) -> None:
        calls: list[str] = []

        def fake_request(url: str, payload: dict[str, Any] | None = None, *, timeout: int = 600) -> dict[str, Any]:
            calls.append(url)
            if url.endswith("/v1/models"):
                return {"data": [{"id": "fake", "meta": {"n_ctx": 5, "n_ctx_train": 8}}]}
            if url.endswith("/tokenize"):
                return {"tokens": list(range(6))}
            raise AssertionError(f"embedding request must not be transmitted: {url}")

        client = __import__(
            "file_system_catalog.content_pipeline", fromlist=["EmbeddingClient"]
        ).EmbeddingClient("http://unused", model="fake", revision="r", expected_dimension=2)
        with patch("file_system_catalog.content_pipeline.request_json", side_effect=fake_request):
            with self.assertRaisesRegex(RuntimeError, "before transmission"):
                client.embed(["too large"] )
        self.assertNotIn("http://unused/v1/embeddings", calls)


    def test_analysis_completion_streams_with_per_token_timeout(self) -> None:
        client = AnalysisClient(
            "http://unused", model="fake", retries=1, per_token_timeout_seconds=23
        )
        response = {
            "model": "fake",
            "content": "{}",
            "reasoning_content": "",
            "finish_reason": "stop",
            "stream": True,
        }
        from file_system_catalog.content_pipeline import ModelContext

        client._model_context = ModelContext(configured_tokens=4096, trained_tokens=4096)
        with patch.object(
            client,
            "_output_token_budgets",
            return_value=(12, [21]),
        ), patch.object(client, "_ensure_request_fits", return_value=12), patch(
            "file_system_catalog.content_pipeline.stream_chat_completion_json",
            return_value=response,
        ) as request:
            generated = client.complete(
                schema_name="x",
                schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
                system="s",
                user="u",
                max_tokens=21,
            )
        self.assertEqual(generated.value, {})
        self.assertIs(request.call_args.args[1]["stream"], True)
        self.assertEqual(
            request.call_args.kwargs, {"per_token_timeout_seconds": 23.0}
        )
        self.assertIs(generated.response_metadata["stream"], True)
        self.assertEqual(generated.response_metadata["per_token_timeout_seconds"], 23.0)


    def test_analysis_rejects_prompt_plus_output_before_request(self) -> None:
        calls: list[str] = []

        def fake_request(url: str, payload: dict[str, Any] | None = None, *, timeout: int = 600) -> dict[str, Any]:
            calls.append(url)
            if url.endswith("/v1/models"):
                return {"data": [{"id": "fake", "meta": {"n_ctx": 100, "n_ctx_train": 100}}]}
            if url.endswith("/apply-template"):
                return {"prompt": "rendered"}
            if url.endswith("/tokenize"):
                return {"tokens": list(range(80))}
            raise AssertionError(f"completion request must not be transmitted: {url}")

        client = AnalysisClient("http://unused", model="fake", retries=1)
        with patch("file_system_catalog.content_pipeline.request_json", side_effect=fake_request):
            with self.assertRaisesRegex(RuntimeError, "before transmission"):
                client.complete(
                    schema_name="x",
                    schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
                    system="s",
                    user="u",
                    max_tokens=21,
                )
        self.assertNotIn("http://unused/v1/chat/completions", calls)

    def test_chunking_obeys_embedding_tokenizer_context(self) -> None:
        class TinyEmbedding(FakeEmbeddingClient):
            def model_context(self):
                from file_system_catalog.content_pipeline import ModelContext
                return ModelContext(configured_tokens=50, trained_tokens=50)

            def token_count(self, text: str, *, add_special: bool = True) -> int:
                return len(text) + int(add_special)

        text = "alpha beta gamma delta " * 30
        chunks = chunk_text(
            text,
            FakeAnalysisClient(),
            embedding_client=TinyEmbedding(),
            target_chars=300,
            max_chars=400,
            overlap_chars=10,
        )
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].start_char, 0)
        self.assertEqual(chunks[-1].end_char, len(text))
        self.assertTrue(all(len(chunk.text) + 1 <= 50 for chunk in chunks))

    def test_legacy_schema_migrates_without_raw_text_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary, "root")
            root.mkdir()
            text = "Mountain biking source text.\n"
            Path(root, "source.txt").write_text(text)
            database = Path(temporary, "catalog.sqlite3")
            self._build_catalog(root, database)
            connection = sqlite3.connect(database)
            try:
                connection.execute("DROP TABLE content_representations")
                connection.execute("DROP TABLE content_chunks")
                connection.execute(
                    """CREATE TABLE content_semantic_entries (
                    semantic_entry_id TEXT PRIMARY KEY,file_id TEXT,collection_id TEXT,filesystem_entry_id INTEGER,
                    relative_path_display TEXT,relative_path_b64 TEXT,content_object_id TEXT,content_sha256 TEXT,
                    source_unit_id TEXT,source_level TEXT,source_index INTEGER,source_start_char INTEGER,
                    source_end_char INTEGER,source_token_count INTEGER,source_text_sha256 TEXT,analysis_kind TEXT,
                    ordinal INTEGER,analysis_text TEXT,analysis_text_sha256 TEXT,analysis_model TEXT,
                    analysis_model_fingerprint TEXT,prompt_version TEXT,generation_seed INTEGER,pipeline_version TEXT,
                    generation_json TEXT,attributes_json TEXT,embedding_model TEXT,embedding_model_revision TEXT,
                    embedding_dimension INTEGER,embedding_dtype TEXT,embedding_norm REAL,embedding_blob BLOB,
                    embedding_sha256 TEXT,analysis_status TEXT,analysis_error TEXT,created_at_ns INTEGER,updated_at_ns INTEGER)"""
                )
                entry = connection.execute(
                    "SELECT id,relative_path_b64,content_sha256 FROM filesystem_entries WHERE relative_path_display='source.txt'"
                ).fetchone()
                file_id = stable_file_id("legacy", entry[1])
                source_id = stable_chunk_id(file_id, entry[2], "chunk", 0, 0, len(text), hashlib.sha256(text.encode()).hexdigest())
                vector = FakeEmbeddingClient().embed([text])[0]
                blob = vector.tobytes()
                common = [
                    file_id,"legacy",entry[0],"source.txt",entry[1],f"sha256:{entry[2]}",entry[2],source_id,
                    "chunk",0,0,len(text),10,hashlib.sha256(text.encode()).hexdigest(),
                ]
                tail = ["fake-analysis",None,"old-prompt",42,"old-pipeline","{}","{}","fake-embedding","old",64,"float32",1.0,blob,hashlib.sha256(blob).hexdigest(),"complete",None,1,1]
                connection.execute(
                    "INSERT INTO content_semantic_entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    ["raw-id",*common,"raw_text",0,text,hashlib.sha256(text.encode()).hexdigest(),*tail],
                )
                connection.execute(
                    "INSERT INTO content_semantic_entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    ["keyword-id",*common,"keyword",0,"mountain bike",hashlib.sha256(b"mountain bike").hexdigest(),*tail],
                )
                connection.execute("PRAGMA user_version=6")
                connection.commit()
                self.assertTrue(migrate_legacy_content_schema(connection, root))
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 7)
                self.assertEqual(connection.execute("SELECT count(*) FROM content_chunks").fetchone()[0], 1)
                self.assertEqual(connection.execute("SELECT count(*) FROM content_representations").fetchone()[0], 1)
                self.assertIsNone(
                    connection.execute(
                        "SELECT name FROM sqlite_schema WHERE type='table' AND name='content_semantic_entries'"
                    ).fetchone()
                )
                columns = {row[1] for row in connection.execute("PRAGMA table_info(content_chunks)")}
                self.assertNotIn("text", columns)
                self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()


def test_analysis_and_embedding_control_requests_use_their_own_timeout_attributes() -> None:
    from file_system_catalog.content_pipeline import AnalysisClient, EmbeddingClient, ModelContext

    analysis = AnalysisClient(
        "http://unused",
        model="analysis",
        retries=1,
        per_token_timeout_seconds=17,
    )
    embedding = EmbeddingClient(
        "http://unused",
        model="embedding",
        revision="r",
        expected_dimension=2,
        request_timeout_seconds=23,
        retries=1,
    )
    embedding._model_context = ModelContext(configured_tokens=100, trained_tokens=100)
    calls: list[tuple[str, float | None]] = []

    def fake_request(url: str, payload: dict[str, Any] | None = None, *, timeout: float | None = 600) -> dict[str, Any]:
        calls.append((url, timeout))
        if url.endswith("/health"):
            return {"status": "ok"}
        if url.endswith("/tokenize"):
            return {"tokens": [1, 2]}
        raise AssertionError(url)

    with patch("file_system_catalog.content_pipeline.request_json", side_effect=fake_request):
        assert analysis.health() == {"status": "ok"}
        assert embedding.health() == {"status": "ok"}
        assert embedding.token_count("x") == 2

    assert calls == [
        ("http://unused/health", 17.0),
        ("http://unused/health", 23.0),
        ("http://unused/tokenize", 23.0),
    ]
