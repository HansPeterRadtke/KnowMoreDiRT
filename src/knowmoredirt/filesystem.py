"""Fast filesystem semantic database and grounded question answering.

This module is deliberately independent of DRT initialization.  It wraps the
vendored :mod:`file_system_catalog` subsystem with KMD defaults while leaving
that package directly importable and usable through its own command-line tools.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from file_system_catalog.content_pipeline import AnalysisClient, EmbeddingClient
from file_system_catalog.folder_assistant import FolderQuestionAssistant, initialize_text_folder


@dataclass(frozen=True)
class FilesystemModelConfig:
    analysis_url: str = "http://127.0.0.1:14829"
    analysis_model: str = "/data/models/llm/Qwen3.5-27B-Q8_0/Qwen3.5-27B-Q8_0.gguf"
    embedding_url: str = "http://127.0.0.1:18139"
    embedding_model: str = "qwen3-embedding-0.6b-q8"
    embedding_revision: str = "370f27d7550e0def9b39c1f16d3fbaa13aa67728:Q8_0"
    embedding_batch_size: int = 8
    embedding_max_batch_characters: int = 60_000
    seed: int = 42
    temperature: float = 0.0

    @classmethod
    def from_environment(cls) -> "FilesystemModelConfig":
        analysis_url = os.getenv("KMD_LOCAL_MODEL_ENDPOINT", cls.analysis_url).rstrip("/")
        for suffix in ("/v1/chat/completions", "/chat/completions", "/v1"):
            if analysis_url.endswith(suffix):
                analysis_url = analysis_url[: -len(suffix)]
                break
        return cls(
            analysis_url=analysis_url,
            analysis_model=os.getenv("KMD_LOCAL_MODEL_NAME", cls.analysis_model),
            embedding_url=os.getenv("KMD_EMBEDDING_ENDPOINT", cls.embedding_url).rstrip("/"),
            embedding_model=os.getenv("KMD_EMBEDDING_MODEL", cls.embedding_model),
            embedding_revision=os.getenv("KMD_EMBEDDING_REVISION", cls.embedding_revision),
            embedding_batch_size=int(os.getenv("KMD_EMBEDDING_BATCH_SIZE", str(cls.embedding_batch_size))),
            embedding_max_batch_characters=int(
                os.getenv("KMD_EMBEDDING_MAX_BATCH_CHARACTERS", str(cls.embedding_max_batch_characters))
            ),
            seed=int(os.getenv("KMD_MODEL_SEED", str(cls.seed))),
            temperature=float(os.getenv("KMD_MODEL_TEMPERATURE", str(cls.temperature))),
        )

    def clients(self) -> tuple[AnalysisClient, EmbeddingClient]:
        return (
            AnalysisClient(
                self.analysis_url,
                model=self.analysis_model,
                seed=self.seed,
                temperature=self.temperature,
            ),
            EmbeddingClient(
                self.embedding_url,
                model=self.embedding_model,
                revision=self.embedding_revision,
                batch_size=self.embedding_batch_size,
                max_batch_characters=self.embedding_max_batch_characters,
            ),
        )


def initialize_filesystem_database(
    folder: os.PathLike[str] | str,
    database: os.PathLike[str] | str,
    *,
    config: FilesystemModelConfig | None = None,
    replace: bool = False,
    chunks_only: bool = False,
    collection_id: str | None = None,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Create the fast semantic filesystem database without building a DRS."""
    analysis, embedding = (config or FilesystemModelConfig.from_environment()).clients()
    return initialize_text_folder(
        root=Path(folder),
        database=Path(database),
        analysis_client=analysis,
        embedding_client=embedding,
        collection_id=collection_id,
        replace=replace,
        chunks_only=chunks_only,
        progress_every=progress_every,
    )


def question_filesystem_database(
    folder: os.PathLike[str] | str,
    database: os.PathLike[str] | str,
    question: str,
    *,
    config: FilesystemModelConfig | None = None,
    max_evidence: int = 24,
) -> dict[str, Any]:
    """Answer directly from the semantic filesystem database with grounded evidence."""
    analysis, embedding = (config or FilesystemModelConfig.from_environment()).clients()
    assistant = FolderQuestionAssistant(
        root=Path(folder),
        database=Path(database),
        analysis_client=analysis,
        embedding_client=embedding,
        max_evidence=max_evidence,
    )
    return assistant.ask(question)
