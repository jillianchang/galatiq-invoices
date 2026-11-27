"""Structured logs, trace files, and audit rows (PLAN.md section 12, D20).

- Console: one JSON line per pipeline stage, on stderr.
- Trace: traces/<run_id>.jsonl with every stage event and LLM call.
- Audit: one compact audit_events row per stage, for the dashboard timeline.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from src.config import ROOT

TRACE_DIR = ROOT / "traces"


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class RunLogger:
    def __init__(
        self,
        run_id: str,
        conn: sqlite3.Connection,
        *,
        trace_dir: Path = TRACE_DIR,
        stream: TextIO | None = sys.stderr,
    ):
        self.run_id = run_id
        self.conn = conn
        self.stream = stream
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = trace_dir / f"{run_id}.jsonl"
        self._trace = self.trace_path.open("a")
        self.file: str | None = None
        self.submission_id: int | None = None
        self.llm_stats: dict[str, int] = {}

    def begin(self, file: str) -> None:
        self.file = file
        self.submission_id = None
        self.llm_stats = {
            "calls": 0,
            "cache_hits": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "retries": 0,
            "cache_misses": 0,
        }

    def stage(self, stage: str, event: str, **payload: Any) -> None:
        record = {
            "ts": _now(),
            "run_id": self.run_id,
            "file": self.file,
            "stage": stage,
            "event": event,
            **payload,
        }
        line = json.dumps(record, default=str)
        self._trace.write(line + "\n")
        self._trace.flush()
        if self.stream is not None:
            print(line, file=self.stream, flush=True)
        self.conn.execute(
            "INSERT INTO audit_events (run_id, submission_id, node, event, payload_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (self.run_id, self.submission_id, stage, event, json.dumps(payload, default=str)),
        )
        self.conn.commit()

    def llm_event(self, event: str, payload: dict[str, Any]) -> None:
        """LLM client callback: full detail to the trace only."""
        self._trace.write(
            json.dumps(
                {
                    "ts": _now(),
                    "run_id": self.run_id,
                    "file": self.file,
                    "stage": "llm",
                    "event": event,
                    **payload,
                },
                default=str,
            )
            + "\n"
        )
        self._trace.flush()
        if event == "llm_call":
            self.llm_stats["calls"] += 1
            self.llm_stats["cache_hits"] += int(bool(payload.get("cache_hit")))
            self.llm_stats["prompt_tokens"] += payload.get("prompt_tokens", 0)
            self.llm_stats["completion_tokens"] += payload.get("completion_tokens", 0)
        elif event in ("llm_retry", "llm_rate_limited"):
            self.llm_stats["retries"] += 1
        elif event == "llm_cache_miss":
            self.llm_stats["cache_misses"] += 1

    def close(self) -> None:
        self._trace.close()
