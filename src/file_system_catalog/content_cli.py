from __future__ import annotations

import argparse
import json
import sys

from .content_pipeline import AnalysisClient, ContentSemanticPipeline, EmbeddingClient


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Generate chunk vectors and constrained semantic representations in normalized content tables.")
    value.add_argument("root", help="content root matching the filesystem catalog")
    value.add_argument("database", help="existing filesystem SQLite catalog")
    value.add_argument("--collection-id", required=True, help="stable logical collection identifier")
    value.add_argument("--analysis-url", default="http://127.0.0.1:14829")
    value.add_argument("--analysis-model", default="/data/models/llm/Qwen3.5-27B-Q8_0/Qwen3.5-27B-Q8_0.gguf")
    value.add_argument("--embedding-url", default="http://127.0.0.1:18139")
    value.add_argument("--embedding-model", default="qwen3-embedding-0.6b-q8")
    value.add_argument("--embedding-revision", default="370f27d7550e0def9b39c1f16d3fbaa13aa67728:Q8_0")
    value.add_argument("--raw-only", action="store_true", help="migrate or refresh chunk metadata and whole-chunk vectors without rerunning the LLM")
    value.add_argument("--path", action="append", dest="paths", help="analyze only this relative path; repeatable")
    value.add_argument("--max-files", type=int)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--temperature", type=float, default=0.0)
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
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
        pipeline = ContentSemanticPipeline(
            database=arguments.database,
            root=arguments.root,
            collection_id=arguments.collection_id,
            analysis_client=analysis,
            embedding_client=embedding,
            seed=arguments.seed,
        )
        if arguments.raw_only:
            result = pipeline.backfill_chunks(
                only_paths=set(arguments.paths) if arguments.paths else None,
                max_files=arguments.max_files,
            )
        else:
            result = pipeline.run(
                only_paths=set(arguments.paths) if arguments.paths else None,
                max_files=arguments.max_files,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        print(f"content analysis failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
