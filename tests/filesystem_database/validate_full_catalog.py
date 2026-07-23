#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import sqlite3
from pathlib import Path

EXPECTED_COUNTS = {
    "file": 10842,
    "directory": 515,
    "symlink": 7,
    "fifo": 1,
    "socket": 1,
}
PENDING = "PENDING_CONTENT_EXTRACTION"
EXPECTED_ERROR_PATHS = {
    "10_archives_and_containers/truncated.zip",
    "11_databases_and_state/corrupt-header.sqlite",
    "11_databases_and_state/not-a-database.sqlite",
    "11_databases_and_state/truncated.sqlite",
    "12_corruption_and_truncation/bitflipped.pdf",
    "12_corruption_and_truncation/bitflipped.zip",
    "12_corruption_and_truncation/fake-jpeg.jpg",
    "12_corruption_and_truncation/fake-pdf.pdf",
    "12_corruption_and_truncation/truncated.docx",
    "12_corruption_and_truncation/truncated.jpg",
    "12_corruption_and_truncation/truncated.pdf",
    "12_corruption_and_truncation/truncated.zip",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database")
    arguments = parser.parse_args()
    database = Path(arguments.database)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    failures: list[str] = []
    try:
        tables = connection.execute("SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
        if sorted(row[0] for row in tables) != ["content_chunks", "content_representations", "filesystem_entries"]:
            failures.append(f"unexpected tables: {[row[0] for row in tables]}")
        if connection.execute("SELECT count(*) FROM content_chunks").fetchone()[0] != 0:
            failures.append("new content_chunks table is not empty")
        if connection.execute("SELECT count(*) FROM content_representations").fetchone()[0] != 0:
            failures.append("new content_representations table is not empty")
        rows = connection.execute("SELECT count(*) FROM filesystem_entries").fetchone()[0]
        if rows != 11366:
            failures.append(f"row count {rows} != 11366")
        counts = dict(connection.execute("SELECT entry_type,count(*) FROM filesystem_entries GROUP BY entry_type"))
        if counts != EXPECTED_COUNTS:
            failures.append(f"type counts differ: {counts}")
        regular_files = connection.execute("SELECT count(*) FROM filesystem_entries WHERE entry_type='file'").fetchone()[0]
        raw_missing = connection.execute(
            "SELECT count(*) FROM filesystem_entries WHERE entry_type='file' AND (raw_metadata_json IS NULL OR raw_metadata_json='{}' OR raw_metadata_source_count<1 OR metadata_parser_attempts_json IS NULL OR metadata_parser_attempts_json='[]')"
        ).fetchone()[0]
        exiftool_missing = connection.execute(
            "SELECT count(*) FROM filesystem_entries WHERE entry_type='file' AND exiftool_metadata_json IS NULL"
        ).fetchone()[0]
        if regular_files != 10842 or raw_missing != 0 or exiftool_missing != 0:
            failures.append(f"raw metadata coverage differs: files={regular_files}, raw_missing={raw_missing}, exiftool_missing={exiftool_missing}")
        extraction_statuses = dict(connection.execute(
            "SELECT metadata_extraction_status,count(*) FROM filesystem_entries WHERE entry_type='file' GROUP BY metadata_extraction_status"
        ))
        if (
            sum(extraction_statuses.values()) != regular_files
            or extraction_statuses.get("error", 0) != 0
            or extraction_statuses.get("unsupported", 0) != 0
            or extraction_statuses.get("intentionally_unsupported", 0) != 5
        ):
            failures.append(f"metadata extraction statuses differ: {extraction_statuses}")
        byte_structure_missing = connection.execute(
            "SELECT count(*) FROM filesystem_entries WHERE entry_type='file' AND (byte_structure_metadata_json IS NULL OR byte_structure_error IS NOT NULL)"
        ).fetchone()[0]
        if byte_structure_missing != 0:
            failures.append(f"byte-structure coverage differs: missing_or_error={byte_structure_missing}")
        for row in connection.execute(
            "SELECT relative_path_display,raw_metadata_json,metadata_parser_attempts_json FROM filesystem_entries WHERE entry_type='file'"
        ):
            try:
                raw_value = json.loads(row[1])
                attempts_value = json.loads(row[2])
                if not isinstance(raw_value, dict) or not raw_value or not isinstance(attempts_value, list) or not attempts_value:
                    failures.append(f"invalid raw metadata structure: {row[0]}")
                    break
            except Exception as error:
                failures.append(f"invalid raw metadata JSON for {row[0]}: {error}")
                break
        specialized_cases = [
            ("00_source_corpora/all2text_all_in_one/archives/bundle.zip", "archive_metadata_json", None),
            ("00_source_corpora/all2text_all_in_one/database/sample.sqlite", "database_metadata_json", None),
            ("00_source_corpora/all2text_all_in_one/email/message.eml", "message_metadata_json", None),
            ("00_source_corpora/all2text_all_in_one/font/font.woff", "font_metadata_json", None),
            ("00_source_corpora/all2text_all_in_one/executables/program.elf", "executable_metadata_json", None),
            ("00_source_corpora/all2text_all_in_one/scientific/data.h5", "scientific_metadata_json", None),
            ("00_source_corpora/all2text_all_in_one/scientific/array.npy", "scientific_metadata_json", None),
            ("00_source_corpora/all2text_all_in_one/scientific/table.parquet", "scientific_metadata_json", None),
            ("00_source_corpora/all2text_all_in_one/scientific/image.fits", "scientific_metadata_json", None),
            ("11_databases_and_state/unclean-wal.sqlite-wal", "database_metadata_json", None),
            ("11_databases_and_state/unclean-wal.sqlite-shm", "database_metadata_json", None),
            ("00_source_corpora/all2text_all_in_one/wrong_extension/zip_named_scan.jpeg", "archive_metadata_json", None),
            ("16_security_traps/TEST-ONLY-certificate.pem", "certificate_metadata_json", None),
            ("10_archives_and_containers/truncated.zip", "archive_metadata_json", "archive_error"),
            ("11_databases_and_state/corrupt-header.sqlite", None, "database_error"),
        ]
        for path, metadata_column, error_column in specialized_cases:
            columns = [value for value in (metadata_column, error_column) if value]
            query = "SELECT " + ",".join(columns) + " FROM filesystem_entries WHERE relative_path_display=?"
            value = connection.execute(query, (path,)).fetchone()
            if value is None:
                failures.append(f"specialized metadata row missing: {path}")
                continue
            offset = 0
            if metadata_column:
                if value[offset] is None:
                    failures.append(f"specialized metadata missing: {path} / {metadata_column}")
                else:
                    try:
                        json.loads(value[offset])
                    except Exception as error:
                        failures.append(f"specialized metadata invalid JSON: {path}: {error}")
                offset += 1
            if error_column and not value[offset]:
                failures.append(f"expected parser error missing: {path} / {error_column}")
        mz_rows = connection.execute(
            "SELECT relative_path_display,executable_metadata_json,executable_error,metadata_extraction_status FROM filesystem_entries WHERE magic_description LIKE 'MS-DOS executable%'"
        ).fetchall()
        if len(mz_rows) < 1:
            failures.append("MS-DOS executable metadata rows missing")
        for row in mz_rows:
            if row[1] is None or row[2] is not None or row[3] != "complete":
                failures.append(f"MS-DOS executable metadata differs: {tuple(row)}")
                continue
            try:
                value = json.loads(row[1])
                attempts = value.get("attempts", [])
                builtin = [item for item in attempts if item.get("tool") == "builtin_mz_pe" and item.get("status") == "complete"]
                if not builtin or builtin[0].get("metadata", {}).get("kind") != "dos_mz":
                    failures.append(f"native MZ parser output missing: {row[0]}")
            except Exception as error:
                failures.append(f"native MZ metadata invalid JSON for {row[0]}: {error}")
        for path in (
            "00_source_corpora/all2text_all_in_one/scientific/data.h5",
            "00_source_corpora/all2text_all_in_one/scientific/array.npy",
            "00_source_corpora/all2text_all_in_one/scientific/table.parquet",
        ):
            value = connection.execute(
                "SELECT metadata_extraction_status,scientific_metadata_json,metadata_error FROM filesystem_entries WHERE relative_path_display=?",
                (path,),
            ).fetchone()
            if value is None or value[0] != "complete" or value[1] is None or value[2] is not None:
                failures.append(f"scientific specialized override differs: {path}: {tuple(value) if value else None}")
        intentional = connection.execute(
            "SELECT count(*),sum(intentional_unsupported_metadata_json IS NOT NULL),sum(metadata_extraction_status='intentionally_unsupported') FROM filesystem_entries WHERE lower(extension) IN ('.ifc','.dxf','.stl')"
        ).fetchone()
        if tuple(intentional) != (5, 5, 5):
            failures.append(f"intentional unsupported controls differ: {tuple(intentional)}")
        for row in connection.execute(
            "SELECT relative_path_display,intentional_unsupported_metadata_json,metadata_parser_attempts_json FROM filesystem_entries WHERE lower(extension) IN ('.ifc','.dxf','.stl')"
        ):
            try:
                marker = json.loads(row[1])
                attempts = json.loads(row[2])
                if marker.get("status") != "intentionally_unsupported" or not any(item.get("status") == "intentionally_unsupported" for item in attempts):
                    failures.append(f"intentional unsupported marker invalid: {row[0]}")
            except Exception as error:
                failures.append(f"intentional unsupported JSON invalid for {row[0]}: {error}")
        git_objects = connection.execute(
            "SELECT count(*),sum(version_control_metadata_json IS NOT NULL),sum(version_control_error IS NOT NULL) FROM filesystem_entries WHERE relative_path_display GLOB '14_repository_shapes/real-project/.git/objects/??/*'"
        ).fetchone()
        if git_objects[0] < 1 or git_objects[0] != git_objects[1] or git_objects[2] != 0:
            failures.append(f"Git object metadata differs: {tuple(git_objects)}")
        git_sample = connection.execute(
            "SELECT version_control_metadata_json FROM filesystem_entries WHERE relative_path_display GLOB '14_repository_shapes/real-project/.git/objects/??/*' LIMIT 1"
        ).fetchone()
        if git_sample is None:
            failures.append("Git object metadata sample missing")
        else:
            try:
                value = json.loads(git_sample[0])
                if value.get("kind") != "git_loose_object" or not value.get("object_id_matches") or not value.get("declared_size_matches"):
                    failures.append(f"Git object metadata invalid: {value}")
            except Exception as error:
                failures.append(f"Git object metadata invalid JSON: {error}")
        git_index = connection.execute(
            "SELECT version_control_metadata_json,version_control_error,metadata_extraction_status FROM filesystem_entries WHERE relative_path_display='14_repository_shapes/real-project/.git/index'"
        ).fetchone()
        if git_index is None or git_index[0] is None or git_index[1] is not None or git_index[2] != "complete":
            failures.append(f"Git index metadata differs: {tuple(git_index) if git_index else None}")
        else:
            try:
                value = json.loads(git_index[0])
                if (
                    value.get("kind") != "git_index"
                    or value.get("version") != 2
                    or value.get("declared_entry_count") != 5
                    or value.get("native_entry_count") != 5
                    or value.get("git_entry_count") != 5
                    or value.get("checksum_matches") is not True
                ):
                    failures.append(f"Git index metadata invalid: {value}")
            except Exception as error:
                failures.append(f"Git index metadata invalid JSON: {error}")
        expected_byte_classes = {
            "05_encodings_and_gibberish/deterministic-high-entropy-16MiB.bin": "high_entropy_byte_stream",
            "05_encodings_and_gibberish/invalid-utf8.txt": "text_byte_stream",
            "05_encodings_and_gibberish/latin1.txt": "text_byte_stream",
            "05_encodings_and_gibberish/nul-in-text.txt": "mixed_text_binary_stream",
            "07_links_identity_and_storage/sparse-5GiB.bin": "sparse_byte_stream",
            "15_special_files/empty-file": "empty_byte_stream",
            "15_special_files/one-byte-file": "single_byte_stream",
        }
        for path, expected_class in expected_byte_classes.items():
            value = connection.execute(
                "SELECT byte_structure_metadata_json,metadata_extraction_status,metadata_error,exiftool_error FROM filesystem_entries WHERE relative_path_display=?",
                (path,),
            ).fetchone()
            if value is None or value[0] is None or value[1] != "complete" or value[2] is not None or value[3] is not None:
                failures.append(f"byte fixture classification differs: {path}: {tuple(value) if value else None}")
                continue
            try:
                metadata = json.loads(value[0])
                if metadata.get("classification") != expected_class:
                    failures.append(f"byte fixture class differs: {path}: {metadata.get('classification')} != {expected_class}")
            except Exception as error:
                failures.append(f"byte fixture metadata invalid JSON: {path}: {error}")
        actual_error_paths = {
            row[0] for row in connection.execute(
                "SELECT relative_path_display FROM filesystem_entries WHERE metadata_error IS NOT NULL"
            )
        }
        if actual_error_paths != EXPECTED_ERROR_PATHS:
            failures.append(
                f"metadata error paths differ: missing={sorted(EXPECTED_ERROR_PATHS - actual_error_paths)}, extra={sorted(actual_error_paths - EXPECTED_ERROR_PATHS)}"
            )
        error_columns = [
            row[1] for row in connection.execute("PRAGMA table_info(filesystem_entries)")
            if row[1].endswith("_error")
        ]
        any_error_expression = " OR ".join(f'"{column}" IS NOT NULL' for column in error_columns)
        any_error_paths = {
            row[0] for row in connection.execute(
                f"SELECT relative_path_display FROM filesystem_entries WHERE {any_error_expression}"
            )
        }
        if any_error_paths != EXPECTED_ERROR_PATHS:
            failures.append(
                f"all error-column paths differ: missing={sorted(EXPECTED_ERROR_PATHS - any_error_paths)}, extra={sorted(any_error_paths - EXPECTED_ERROR_PATHS)}"
            )
        ledger_error_paths = set()
        for row in connection.execute(
            "SELECT relative_path_display,metadata_parser_attempts_json FROM filesystem_entries WHERE entry_type='file'"
        ):
            attempts = json.loads(row[1])
            if any(attempt.get("error") for attempt in attempts):
                ledger_error_paths.add(row[0])
        if not ledger_error_paths.issubset(EXPECTED_ERROR_PATHS):
            failures.append(f"parser-ledger errors outside deliberate fixtures: {sorted(ledger_error_paths - EXPECTED_ERROR_PATHS)}")
        false_attempts = [
            ("17_scale_and_fanout/single-directory-2000/entry-0777", "archive_metadata_json", "archive_error"),
            ("14_repository_shapes/real-project/src/package/core.py", "executable_metadata_json", "executable_error"),
            ("00_source_corpora/all2text_all_in_one/wrong_extension/html_named_database.sqlite", "database_metadata_json", "database_error"),
        ]
        for path, metadata_column, error_column in false_attempts:
            value = connection.execute(
                f"SELECT {metadata_column},{error_column} FROM filesystem_entries WHERE relative_path_display=?", (path,)
            ).fetchone()
            if value is None or value[0] is not None or value[1] is not None:
                failures.append(f"false parser attempt remains: {path}: {tuple(value) if value else None}")
        placeholders = connection.execute(
            "SELECT count(*) FROM filesystem_entries WHERE content_summary_short=? AND content_summary_long=? AND content_description=? AND content_keywords=? AND content_entities=?",
            (PENDING, PENDING, PENDING, PENDING, PENDING),
        ).fetchone()[0]
        if placeholders != rows:
            failures.append(f"placeholder rows {placeholders} != {rows}")
        hardlinks = connection.execute(
            "SELECT count(*),count(DISTINCT inode),min(link_count),max(link_count) FROM filesystem_entries WHERE hardlink_key IS NOT NULL"
        ).fetchone()
        if tuple(hardlinks) != (203, 1, 203, 203):
            failures.append(f"hardlink topology differs: {tuple(hardlinks)}")
        sparse = connection.execute(
            "SELECT size_bytes,allocated_bytes,is_sparse,sparse_extents_json,hash_status FROM filesystem_entries WHERE relative_path_display='07_links_identity_and_storage/sparse-5GiB.bin'"
        ).fetchone()
        if sparse is None or sparse[0] != 5 * 1024**3 or sparse[1] >= 1024 * 1024 or sparse[2] != 1 or not json.loads(sparse[3]) or sparse[4] != "skipped_sparse_large":
            failures.append(f"sparse file differs: {tuple(sparse) if sparse else None}")
        malformed = connection.execute(
            "SELECT name_b64,name_hex FROM filesystem_entries WHERE name_display LIKE '%\\udc%' LIMIT 1"
        ).fetchone()
        if malformed is None:
            malformed = connection.execute(
                "SELECT name_b64,name_hex FROM filesystem_entries WHERE name_hex LIKE '%ff%' AND relative_path_display LIKE '04_names_and_paths/%' LIMIT 1"
            ).fetchone()
        if malformed is None or base64.b64decode(malformed[0]).hex() != malformed[1]:
            failures.append("invalid UTF-8 raw name was not preserved")
        xattr = connection.execute(
            "SELECT xattr_count,xattrs_b64_json,pdf_title,pdf_author FROM filesystem_entries WHERE relative_path_display='06_embedded_metadata/documents/impossible-album.pdf'"
        ).fetchone()
        if xattr is None or xattr[0] < 9 or "user.artist" not in json.loads(xattr[1]) or xattr[2] is None:
            failures.append(f"PDF/xattr metadata missing: {tuple(xattr) if xattr else None}")
        office = connection.execute(
            "SELECT office_title,office_creator,office_created,office_modified,metadata_error FROM filesystem_entries WHERE relative_path_display='06_embedded_metadata/documents/contradictory.docx'"
        ).fetchone()
        if office is None or office[0] != "Album of Tax Returns" or office[1] != "DJ Spreadsheet" or office[2] is None or office[3] is None or office[4] is not None:
            failures.append(f"Office metadata missing: {tuple(office) if office else None}")
        acl = connection.execute(
            "SELECT acl_access_present,acl_text FROM filesystem_entries WHERE relative_path_display='08_permissions_owners_and_acl/acl-deny-hans.txt'"
        ).fetchone()
        if acl is None or acl[0] != 1 or "user:hans:---" not in (acl[1] or ""):
            failures.append(f"ACL metadata missing: {tuple(acl) if acl else None}")
        symlinks = connection.execute(
            "SELECT count(*),sum(symlink_target_exists=0),sum(symlink_resolves_outside_root=1) FROM filesystem_entries WHERE entry_type='symlink'"
        ).fetchone()
        if symlinks[0] != 7 or symlinks[1] < 1 or symlinks[2] < 1:
            failures.append(f"symlink metadata differs: {tuple(symlinks)}")
        birth_count = connection.execute("SELECT count(*) FROM filesystem_entries WHERE birth_time_ns IS NOT NULL").fetchone()[0]
        if birth_count != rows:
            failures.append(f"birth times present for {birth_count} of {rows}")
        future_count = connection.execute("SELECT count(*) FROM filesystem_entries WHERE has_future_time=1").fetchone()[0]
        pre_epoch_count = connection.execute("SELECT count(*) FROM filesystem_entries WHERE has_pre_epoch_time=1").fetchone()[0]
        if future_count < 1 or pre_epoch_count < 1:
            failures.append(f"timestamp anomalies missing: future={future_count}, pre_epoch={pre_epoch_count}")
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            failures.append(f"integrity check: {integrity}")
        result = {
            "status": "ok" if not failures else "failed",
            "database": str(database),
            "database_bytes": database.stat().st_size,
            "rows": rows,
            "counts": counts,
            "columns": len(connection.execute("PRAGMA table_info(filesystem_entries)").fetchall()),
            "hardlinked_rows": hardlinks[0],
            "birth_times": birth_count,
            "future_time_rows": future_count,
            "pre_epoch_time_rows": pre_epoch_count,
            "failures": failures,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1 if failures else 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
