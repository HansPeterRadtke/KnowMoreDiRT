#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import drt  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="drt_public_interface_") as temp:
        folder = Path(temp) / "random root"
        files = {
            "nonsense/no-extension": (
                "zqx %% raw prose only.\n"
                "Product LarchOwl release note.\n"
                "Mira Sol carried PR-7312 for LarchOwl. Ivo Renn reviewed PR-7312.\n"
                "Customer Bronze Kite reported BUG-445 for LarchOwl.\n"
            ),
            "odd/telemetry.streamx": "[09:12] LarchOwl PR-7312 changed state from open to closed.\n",
            "raw-json-as-text.blob": '{"plain_text_claim": "The LarchOwl runbook is https://docs.example/larchowl/runbook"}\n',
        }
        for rel_path, text in files.items():
            path = folder / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

        with drt.initialize(folder) as system:
            answer = system.question("Which engineer carried PR-7312 in LarchOwl?")
            assert isinstance(answer, str), type(answer)
            assert "Mira Sol" in answer, answer
            assert "unknown" != answer.strip().lower(), answer
    print("public interface test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
