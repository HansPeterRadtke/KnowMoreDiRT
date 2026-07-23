from __future__ import annotations

LEGACY_CONTENT_TABLE_NAME = "content_semantic_entries"
CHUNK_TABLE_NAME = "content_chunks"
REPRESENTATION_TABLE_NAME = "content_representations"
CONTENT_SCHEMA_VERSION = 3

STRENGTH_VALUES = (
    "very_weak",
    "weak",
    "moderate",
    "strong",
    "very_strong",
    "essential",
)
REPRESENTATION_KIND_VALUES = (
    "summary",
    "description",
    "sentence",
    "keyphrase",
    "keyword",
    "entity",
    "topic",
    "summary_short",
    "summary_long",
    "search_phrase",
)

CHUNK_COLUMN_DEFINITIONS = [
    ("chunk_id", "TEXT PRIMARY KEY"),
    ("file_id", "TEXT NOT NULL"),
    ("collection_id", "TEXT NOT NULL"),
    ("filesystem_entry_id", "INTEGER NOT NULL"),
    ("content_object_id", "TEXT NOT NULL"),
    ("content_sha256", "TEXT NOT NULL"),
    ("chunk_kind", "TEXT NOT NULL CHECK(chunk_kind IN ('chunk','file'))"),
    ("chunk_index", "INTEGER NOT NULL"),
    ("start_char", "INTEGER NOT NULL CHECK(start_char >= 0)"),
    ("end_char", "INTEGER NOT NULL CHECK(end_char >= start_char)"),
    ("character_count", "INTEGER NOT NULL CHECK(character_count = end_char - start_char)"),
    ("word_count", "INTEGER NOT NULL CHECK(word_count >= 0)"),
    ("token_count", "INTEGER NOT NULL CHECK(token_count >= 0)"),
    ("text_sha256", "TEXT NOT NULL"),
    ("embedding_model", "TEXT"),
    ("embedding_model_revision", "TEXT"),
    ("embedding_dimension", "INTEGER"),
    ("embedding_dtype", "TEXT CHECK(embedding_dtype IN ('float32','float16','int8'))"),
    ("embedding_norm", "REAL"),
    ("embedding_blob", "BLOB"),
    ("embedding_sha256", "TEXT"),
    ("created_at_ns", "INTEGER NOT NULL"),
    ("updated_at_ns", "INTEGER NOT NULL"),
    ("FOREIGN KEY(filesystem_entry_id)", "REFERENCES filesystem_entries(id) ON DELETE CASCADE"),
]

REPRESENTATION_COLUMN_DEFINITIONS = [
    ("representation_id", "TEXT PRIMARY KEY"),
    ("chunk_id", "TEXT NOT NULL"),
    ("representation_kind", f"TEXT NOT NULL CHECK(representation_kind IN {REPRESENTATION_KIND_VALUES!r})"),
    ("facet_label", "TEXT NOT NULL"),
    ("facet_strength", f"TEXT NOT NULL CHECK(facet_strength IN {STRENGTH_VALUES!r})"),
    ("item_strength", f"TEXT NOT NULL CHECK(item_strength IN {STRENGTH_VALUES!r})"),
    ("facet_rank", "INTEGER NOT NULL CHECK(facet_rank >= 0)"),
    ("item_rank", "INTEGER NOT NULL CHECK(item_rank >= 0)"),
    ("global_rank", "INTEGER NOT NULL CHECK(global_rank >= 0)"),
    ("representation_text", "TEXT NOT NULL"),
    ("representation_text_sha256", "TEXT NOT NULL"),
    ("analysis_model", "TEXT NOT NULL"),
    ("analysis_model_fingerprint", "TEXT"),
    ("prompt_version", "TEXT NOT NULL"),
    ("generation_seed", "INTEGER NOT NULL"),
    ("pipeline_version", "TEXT NOT NULL"),
    ("generation_json", "TEXT NOT NULL"),
    ("attributes_json", "TEXT NOT NULL"),
    ("embedding_model", "TEXT NOT NULL"),
    ("embedding_model_revision", "TEXT NOT NULL"),
    ("embedding_dimension", "INTEGER NOT NULL CHECK(embedding_dimension > 0)"),
    ("embedding_dtype", "TEXT NOT NULL CHECK(embedding_dtype IN ('float32','float16','int8'))"),
    ("embedding_norm", "REAL NOT NULL"),
    ("embedding_blob", "BLOB NOT NULL"),
    ("embedding_sha256", "TEXT NOT NULL"),
    ("analysis_status", "TEXT NOT NULL CHECK(analysis_status IN ('complete','error'))"),
    ("analysis_error", "TEXT"),
    ("created_at_ns", "INTEGER NOT NULL"),
    ("updated_at_ns", "INTEGER NOT NULL"),
    ("FOREIGN KEY(chunk_id)", f"REFERENCES {CHUNK_TABLE_NAME}(chunk_id) ON DELETE CASCADE"),
]


def _column_names(definitions: list[tuple[str, str]]) -> list[str]:
    return [name for name, _ in definitions if not name.startswith("FOREIGN KEY")]


def _create_table_sql(table: str, definitions: list[tuple[str, str]]) -> str:
    return (
        f"CREATE TABLE {table} (\n  "
        + ",\n  ".join(
            f'"{name}" {definition}' if not name.startswith("FOREIGN KEY") else f"{name} {definition}"
            for name, definition in definitions
        )
        + "\n)"
    )


CHUNK_COLUMN_NAMES = _column_names(CHUNK_COLUMN_DEFINITIONS)
REPRESENTATION_COLUMN_NAMES = _column_names(REPRESENTATION_COLUMN_DEFINITIONS)
CREATE_CHUNK_TABLE_SQL = _create_table_sql(CHUNK_TABLE_NAME, CHUNK_COLUMN_DEFINITIONS)
CREATE_REPRESENTATION_TABLE_SQL = _create_table_sql(
    REPRESENTATION_TABLE_NAME, REPRESENTATION_COLUMN_DEFINITIONS
)
CONTENT_CREATE_SQL = [CREATE_CHUNK_TABLE_SQL, CREATE_REPRESENTATION_TABLE_SQL]

CONTENT_INDEX_NAMES = [
    "idx_content_chunks_file",
    "idx_content_chunks_filesystem_entry",
    "idx_content_chunks_content_object",
    "idx_content_chunks_text_hash",
    "idx_content_chunks_slot",
    "idx_content_representations_chunk",
    "idx_content_representations_kind",
    "idx_content_representations_strength",
    "idx_content_representations_text_hash",
    "idx_content_representations_slot",
]

CONTENT_INDEX_SQL = [
    f"CREATE INDEX idx_content_chunks_file ON {CHUNK_TABLE_NAME}(file_id)",
    f"CREATE INDEX idx_content_chunks_filesystem_entry ON {CHUNK_TABLE_NAME}(filesystem_entry_id)",
    f"CREATE INDEX idx_content_chunks_content_object ON {CHUNK_TABLE_NAME}(content_object_id)",
    f"CREATE INDEX idx_content_chunks_text_hash ON {CHUNK_TABLE_NAME}(text_sha256)",
    f"CREATE UNIQUE INDEX idx_content_chunks_slot ON {CHUNK_TABLE_NAME}(file_id,chunk_kind,chunk_index,content_sha256,start_char,end_char)",
    f"CREATE INDEX idx_content_representations_chunk ON {REPRESENTATION_TABLE_NAME}(chunk_id)",
    f"CREATE INDEX idx_content_representations_kind ON {REPRESENTATION_TABLE_NAME}(representation_kind)",
    f"CREATE INDEX idx_content_representations_strength ON {REPRESENTATION_TABLE_NAME}(facet_strength)",
    f"CREATE INDEX idx_content_representations_text_hash ON {REPRESENTATION_TABLE_NAME}(representation_text_sha256)",
    f"CREATE UNIQUE INDEX idx_content_representations_slot ON {REPRESENTATION_TABLE_NAME}(chunk_id,analysis_model,prompt_version,embedding_model,global_rank)",
]
