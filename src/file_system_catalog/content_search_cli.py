from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from typing import Any

from context_capacity import context_char_capacity, context_token_capacity

from .content_pipeline import EmbeddingClient, search_literal_chunks, search_semantic_entries


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Search literal text or semantic vectors in a filesystem catalog.")
    value.add_argument("database")
    subparsers = value.add_subparsers(dest="mode", required=True)
    literal = subparsers.add_parser("literal")
    literal.add_argument("root", help="canonical All2Text or text-content root")
    literal.add_argument("query")
    literal.add_argument("--case-sensitive", action="store_true")
    literal.add_argument("--whole-word", action="store_true")
    literal.add_argument("--max-matches", type=int)
    literal.add_argument("--excerpt-characters", type=int)
    semantic = subparsers.add_parser("semantic")
    semantic.add_argument("query")
    semantic.add_argument("--embedding-url", default="http://127.0.0.1:18139")
    semantic.add_argument("--embedding-model", default="qwen3-embedding-0.6b-q8")
    semantic.add_argument("--embedding-revision", default="370f27d7550e0def9b39c1f16d3fbaa13aa67728:Q8_0")
    semantic.add_argument("--top", type=int)
    semantic.add_argument("--text-limit", type=int)
    return value


def _compact(item: dict[str, Any], limit: int | None) -> dict[str, Any]:
    result = dict(item)
    text = str(result.pop("analysis_text", ""))
    if limit is None:
        result["matched_text"] = text
        result["matched_text_truncated"] = False
    else:
        result["matched_text"] = text if len(text) <= limit else text[:limit] + "…"
        result["matched_text_truncated"] = len(text) > limit
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        connection = sqlite3.connect(arguments.database)
        connection.row_factory = sqlite3.Row
        try:
            if arguments.mode == "literal":
                results = search_literal_chunks(
                    connection,
                    arguments.root,
                    arguments.query,
                    case_sensitive=arguments.case_sensitive,
                    whole_word=arguments.whole_word,
                    max_matches=arguments.max_matches,
                    excerpt_characters=arguments.excerpt_characters,
                )
            else:
                embedding = EmbeddingClient(
                    arguments.embedding_url,
                    model=arguments.embedding_model,
                    revision=arguments.embedding_revision,
                )
                context = int(embedding.model_context().configured_tokens)
                top = arguments.top or context_token_capacity(
                    context,
                    ratio_names=("KMD_CONTENT_SEARCH_RESULT_RATIO",),
                    ratio_default=1.0 / 1024.0,
                )
                text_limit = arguments.text_limit
                if text_limit is None:
                    text_limit = context_char_capacity(
                        context,
                        ratio_names=("KMD_CONTENT_SEARCH_TEXT_RATIO",),
                        ratio_default=1.0 / 64.0,
                    )
                if top < 1 or text_limit < 0:
                    raise ValueError("top must be positive and text-limit nonnegative")
                vector = embedding.embed([arguments.query])[0]
                results = [_compact(item, text_limit) for item in search_semantic_entries(connection, vector)[:top]]
        finally:
            connection.close()
        print(json.dumps({"mode": arguments.mode, "query": arguments.query, "results": results}, indent=2))
        return 0
    except Exception as error:
        print(f"content search failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
