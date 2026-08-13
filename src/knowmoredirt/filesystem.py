"""Fast filesystem semantic database and grounded question answering.

This module is deliberately independent of DRT initialization.  It wraps the
vendored :mod:`file_system_catalog` subsystem with KMD defaults while leaving
that package directly importable and usable through its own command-line tools.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kmd_runtime_config import floating as _config_float, integer as _config_int, text as _config_text

from file_system_catalog.content_pipeline import AnalysisClient, EmbeddingClient
from file_system_catalog.folder_assistant import FolderQuestionAssistant, initialize_text_folder


@dataclass(frozen=True)
class FilesystemModelConfig:
    analysis_url: str = field(default_factory=lambda: _config_text("KMD_LOCAL_MODEL_ENDPOINT").rstrip("/"))
    analysis_model: str = field(default_factory=lambda: _config_text("KMD_LOCAL_MODEL_NAME"))
    embedding_url: str = field(default_factory=lambda: _config_text("KMD_EMBEDDING_ENDPOINT").rstrip("/"))
    embedding_model: str = field(default_factory=lambda: _config_text("KMD_EMBEDDING_MODEL"))
    embedding_revision: str = field(default_factory=lambda: _config_text("KMD_EMBEDDING_REVISION"))
    embedding_batch_size: int = field(default_factory=lambda: _config_int("KMD_EMBEDDING_BATCH_SIZE"))
    embedding_max_batch_characters: int = field(default_factory=lambda: _config_int("KMD_EMBEDDING_MAX_BATCH_CHARACTERS"))
    seed: int = field(default_factory=lambda: _config_int("KMD_MODEL_SEED"))
    temperature: float = field(default_factory=lambda: _config_float("KMD_MODEL_TEMPERATURE"))

    @classmethod
    def from_environment(cls) -> "FilesystemModelConfig":
        analysis_url = _config_text("KMD_LOCAL_MODEL_ENDPOINT").rstrip("/")
        for suffix in ("/v1/chat/completions", "/chat/completions", "/v1"):
            if analysis_url.endswith(suffix):
                analysis_url = analysis_url[: -len(suffix)]
                break
        return cls(
            analysis_url=analysis_url,
            analysis_model=_config_text("KMD_LOCAL_MODEL_NAME"),
            embedding_url=_config_text("KMD_EMBEDDING_ENDPOINT").rstrip("/"),
            embedding_model=_config_text("KMD_EMBEDDING_MODEL"),
            embedding_revision=_config_text("KMD_EMBEDDING_REVISION"),
            embedding_batch_size=_config_int("KMD_EMBEDDING_BATCH_SIZE"),
            embedding_max_batch_characters=_config_int("KMD_EMBEDDING_MAX_BATCH_CHARACTERS"),
            seed=_config_int("KMD_MODEL_SEED"),
            temperature=_config_float("KMD_MODEL_TEMPERATURE"),
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
