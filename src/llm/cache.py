"""Record/replay cache for LLM calls (D15).

Each response is stored under a hash of the full request (model, reasoning
effort, temperature, messages, response format, tools). Unchanged prompts
replay for free; any prompt change is a cache miss.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def request_key(request: dict[str, Any]) -> str:
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class LLMCache:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text())["response"]

    def put(self, key: str, request: dict[str, Any], response: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "key": key,
            "model": request.get("model"),
            "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "request": request,
            "response": response,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True, default=str))
        tmp.replace(path)
