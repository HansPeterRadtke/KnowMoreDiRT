from __future__ import annotations

import argparse
import json
import sys

from .content_pipeline import AnalysisClient, EmbeddingClient
from .folder_assistant import FolderQuestionAssistant, initialize_text_folder


def _model_arguments(value: argparse.ArgumentParser) -> None:
    value.add_argument("--analysis-url", default="http://127.0.0.1:14829")
    value.add_argument(
        "--analysis-model",
        default="",
    )
    value.add_argument("--embedding-url", default="http://127.0.0.1:18139")
    value.add_argument("--embedding-model", default="qwen3-embedding-0.6b-q8")
    value.add_argument(
        "--embedding-revision",
        default="370f27d7550e0def9b39c1f16d3fbaa13aa67728:Q8_0",
    )
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--temperature", type=float, default=0.0)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Initialize and question a text-only folder with constrained LLM planning."
    )
    subparsers = value.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser(
        "init", help="atomically scan, chunk, embed and analyze a text folder"
    )
    initialize.add_argument("root")
    initialize.add_argument("database")
    initialize.add_argument("--collection-id")
    initialize.add_argument("--replace", action="store_true")
    initialize.add_argument(
        "--chunks-only",
        action="store_true",
        help="create filesystem metadata and chunk vectors without LLM representations",
    )
    initialize.add_argument("--progress-every", type=int, default=0)
    initialize.add_argument("--max-hash-bytes", type=int, default=256 * 1024 * 1024)
    _model_arguments(initialize)

    ask = subparsers.add_parser(
        "ask", help="plan searches and answer one natural-language folder question"
    )
    ask.add_argument("root")
    ask.add_argument("database")
    ask.add_argument("question", nargs="+")
    _model_arguments(ask)
    return value


def _clients(arguments: argparse.Namespace) -> tuple[AnalysisClient, EmbeddingClient]:
    analysis = AnalysisClient(
        arguments.analysis_url,
        model=arguments.analysis_model,
        seed=arguments.seed,
        temperature=arguments.temperature,
    )
    embedding = EmbeddingClient(
        arguments.embedding_url,
        model=arguments.embedding_model,
        revision=arguments.embedding_revision,
    )
    return analysis, embedding


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        analysis, embedding = _clients(arguments)
        if arguments.command == "init":
            result = initialize_text_folder(
                root=arguments.root,
                database=arguments.database,
                analysis_client=analysis,
                embedding_client=embedding,
                collection_id=arguments.collection_id,
                replace=arguments.replace,
                chunks_only=arguments.chunks_only,
                progress_every=arguments.progress_every,
                max_hash_bytes=arguments.max_hash_bytes,
            )
        else:
            assistant = FolderQuestionAssistant(
                root=arguments.root,
                database=arguments.database,
                analysis_client=analysis,
                embedding_client=embedding,
            )
            result = assistant.ask(" ".join(arguments.question))
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    except Exception as error:
        print(f"folder assistant failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
