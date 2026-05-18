"""Optional local-model integration hooks.

The initial KMD engine does not require a model. This module isolates future
local llama.cpp-style calls so cloud APIs never become part of the core.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class LocalModelClient:
    endpoint: str = os.environ.get("KMD_LOCAL_MODEL_ENDPOINT", "http://127.0.0.1:14829/v1")
    timeout_seconds: float = float(os.environ.get("KMD_LOCAL_MODEL_TIMEOUT", "30"))

    def models(self) -> dict:
        url = self.endpoint.rstrip("/") + "/models"
        with urllib.request.urlopen(url, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

