from __future__ import annotations

import base64
import json
import os
import socket
import sqlite3
import tempfile
import unittest
from pathlib import Path

from file_system_catalog.scanner import FilesystemScanner, PENDING_CONTENT
from file_system_catalog.content_schema import CHUNK_TABLE_NAME, REPRESENTATION_TABLE_NAME
from file_system_catalog.schema import COLUMN_NAMES, TABLE_NAME


class CatalogScannerTest(unittest.TestCase):
    def test_adversarial_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = os.path.join(temporary, "tree")
            os.mkdir(root)
            Path(root, "ordinary.txt").write_text("hello\n", encoding="utf-8")
            os.link(Path(root, "ordinary.txt"), Path(root, "hardlink.dat"))
            os.symlink("ordinary.txt", Path(root, "relative-link"))
            os.symlink("missing-target", Path(root, "broken-link"))
            os.mkfifo(Path(root, "pipe"))
            stale_socket = socket.socket(socket.AF_UNIX)
            stale_socket.bind(str(Path(root, "stale.socket")))
            stale_socket.close()
            sparse = Path(root, "sparse.bin")
            with sparse.open("wb") as handle:
                handle.seek(16 * 1024 * 1024)
                handle.write(b"tail")
            malformed_name = b"invalid-\xff-name"
            malformed_path = os.fsencode(root) + b"/" + malformed_name
            descriptor = os.open(malformed_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
            os.write(descriptor, b"raw-name")
            os.close(descriptor)
            os.setxattr(Path(root, "ordinary.txt"), "user.test", b"value")
            Path(root, "deliberate.ifc").write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="ascii")
            mz = bytearray(64)
            mz[0:2] = b"MZ"
            mz[2:4] = (64).to_bytes(2, "little")
            mz[4:6] = (1).to_bytes(2, "little")
            mz[8:10] = (4).to_bytes(2, "little")
            Path(root, "minimal.exe").write_bytes(mz)
            nested = Path(root, "directory")
            nested.mkdir()
            Path(nested, "child.json").write_text('{"ok":true}\n', encoding="utf-8")

            database = Path(temporary, "catalog.sqlite3")
            result = FilesystemScanner(root, max_hash_bytes=1024 * 1024, progress_every=0).scan_to_database(database)
            self.assertEqual(result["rows"], 12)
            self.assertEqual(result["user_tables"], 3)
            self.assertEqual(result["columns"], len(COLUMN_NAMES))

            connection = sqlite3.connect(database)
            connection.row_factory = sqlite3.Row
            try:
                self.assertEqual(connection.execute(f"SELECT count(*) FROM {TABLE_NAME}").fetchone()[0], 12)
                self.assertEqual(
                    connection.execute("SELECT count(*) FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchone()[0],
                    3,
                )
                self.assertEqual(connection.execute(f"SELECT count(*) FROM {CHUNK_TABLE_NAME}").fetchone()[0], 0)
                self.assertEqual(connection.execute(f"SELECT count(*) FROM {REPRESENTATION_TABLE_NAME}").fetchone()[0], 0)
                malformed = connection.execute(
                    f"SELECT * FROM {TABLE_NAME} WHERE name_b64=?",
                    (base64.b64encode(malformed_name).decode("ascii"),),
                ).fetchone()
                self.assertIsNotNone(malformed)
                self.assertEqual(bytes.fromhex(malformed["name_hex"]), malformed_name)
                ordinary = connection.execute(
                    f"SELECT * FROM {TABLE_NAME} WHERE relative_path_display='ordinary.txt'"
                ).fetchone()
                hardlink = connection.execute(
                    f"SELECT * FROM {TABLE_NAME} WHERE relative_path_display='hardlink.dat'"
                ).fetchone()
                self.assertEqual(ordinary["inode"], hardlink["inode"])
                self.assertEqual(ordinary["hardlink_key"], hardlink["hardlink_key"])
                self.assertEqual(ordinary["link_count"], 2)
                self.assertEqual(json.loads(ordinary["xattrs_b64_json"])["user.test"], base64.b64encode(b"value").decode("ascii"))
                self.assertEqual(ordinary["hash_status"], "complete")
                self.assertIn(ordinary["metadata_extraction_status"], ("complete", "partial"))
                self.assertGreaterEqual(ordinary["raw_metadata_source_count"], 1)
                self.assertIsNotNone(ordinary["raw_metadata_json"])
                self.assertIsNotNone(ordinary["metadata_parser_attempts_json"])
                self.assertIsNotNone(ordinary["byte_structure_metadata_json"])
                self.assertIsNone(ordinary["byte_structure_error"])
                self.assertIsNotNone(ordinary["exiftool_metadata_json"])
                self.assertEqual(ordinary["content_summary_short"], PENDING_CONTENT)
                executable = connection.execute(
                    f"SELECT metadata_extraction_status,executable_metadata_json,executable_error FROM {TABLE_NAME} WHERE relative_path_display='minimal.exe'"
                ).fetchone()
                self.assertEqual(executable["metadata_extraction_status"], "complete")
                self.assertIsNone(executable["executable_error"])
                executable_value = json.loads(executable["executable_metadata_json"])
                self.assertTrue(any(item.get("tool") == "builtin_mz_pe" and item.get("status") == "complete" for item in executable_value["attempts"]))
                intentional = connection.execute(
                    f"SELECT metadata_extraction_status,intentional_unsupported_metadata_json,metadata_parser_attempts_json FROM {TABLE_NAME} WHERE relative_path_display='deliberate.ifc'"
                ).fetchone()
                self.assertEqual(intentional["metadata_extraction_status"], "intentionally_unsupported")
                self.assertEqual(json.loads(intentional["intentional_unsupported_metadata_json"])["status"], "intentionally_unsupported")
                self.assertTrue(any(item["status"] == "intentionally_unsupported" for item in json.loads(intentional["metadata_parser_attempts_json"])))
                sparse_row = connection.execute(
                    f"SELECT * FROM {TABLE_NAME} WHERE relative_path_display='sparse.bin'"
                ).fetchone()
                self.assertEqual(sparse_row["is_sparse"], 1)
                self.assertGreater(sparse_row["size_bytes"], sparse_row["allocated_bytes"])
                self.assertTrue(json.loads(sparse_row["sparse_extents_json"]))
                self.assertEqual(
                    connection.execute(f"SELECT entry_type FROM {TABLE_NAME} WHERE relative_path_display='pipe'").fetchone()[0],
                    "fifo",
                )
                self.assertEqual(
                    connection.execute(f"SELECT entry_type FROM {TABLE_NAME} WHERE relative_path_display='stale.socket'").fetchone()[0],
                    "socket",
                )
                broken = connection.execute(
                    f"SELECT * FROM {TABLE_NAME} WHERE relative_path_display='broken-link'"
                ).fetchone()
                self.assertEqual(broken["entry_type"], "symlink")
                self.assertEqual(broken["symlink_target_exists"], 0)
                self.assertIsNotNone(ordinary["birth_time_ns"])
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
