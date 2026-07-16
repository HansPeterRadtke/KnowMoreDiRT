#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="dspg_raw_folder_ingest_cli_") as temp:
        base = Path(temp)
        source = base / "arbitrary folder names"
        files = {
            "no-extension-file": "Mira Sol opened PR-918 for NovaLathe. Customer Bronze Kite reported BUG-440.",
            "odd/subdir/raw-json.payload": '{"note": "Oren Vale reviewed PR-918", "url": "https://docs.example/novalathe/design"}',
            "symbols/%%%/table.weird": "thing | owner | state\nNovaLathe | Mira Sol | closed\n",
            "logs/event.streamx": "[09:14] Bronze Kite asked about session_cache.cpp and debug.tmp\n",
        }
        for rel_path, text in files.items():
            path = source / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

        db_path = base / "raw_folder.sqlite"
        report_path = base / "ingest_report.json"
        subprocess.run(
            [
                "python3",
                str(ROOT / "scripts" / "dspg_ingest_folder.py"),
                "--input-folder",
                str(source),
                "--config",
                str(ROOT / "config" / "dspg_system.yaml"),
                "--db",
                str(db_path),
                "--variant",
                "deterministic_only",
                "--run-id",
                "raw_folder_ingest_cli_test",
                "--report",
                str(report_path),
            ],
            cwd=str(ROOT),
            check=True,
            stdout=subprocess.DEVNULL,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["totals"]["files"] == len(files), report["totals"]
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        rel_paths = {row["rel_path"] for row in con.execute("SELECT rel_path FROM documents")}
        assert rel_paths == set(files), rel_paths
        raw_json_chunk = con.execute("SELECT text FROM chunks WHERE text LIKE ?", ('%"note": "Oren Vale reviewed PR-918"%',)).fetchone()
        assert raw_json_chunk is not None, "raw JSON-as-text was not stored"
        assert raw_json_chunk["text"] == files["odd/subdir/raw-json.payload"], raw_json_chunk["text"]
        mention_surfaces = {row["surface"] for row in con.execute("SELECT surface FROM mentions")}
        assert "PR-918" in mention_surfaces, mention_surfaces
        assert "https://docs.example/novalathe/design" in mention_surfaces, mention_surfaces
        con.close()
    print("raw folder ingest CLI test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
