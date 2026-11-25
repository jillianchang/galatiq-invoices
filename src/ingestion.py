"""Ingestion stage: any invoice file -> normalized Invoice (PLAN.md section 4).

Routing (D1): JSON/CSV/XML go to deterministic parsers; TXT, email, and PDF
go to the extraction agent. A structured file that fails to parse falls back
to the agent. Anything that can't be extracted confidently becomes an
`extraction_failed` HOLD (D16), never a guess.

CLI (for trying extraction on a few invoices):
    uv run python -m src.ingestion data/invoices/invoice_1001.txt [more files...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.agents.extraction import ExtractionAgent, to_invoice
from src.normalize import normalize_invoice
from src.parsers import STRUCTURED_PARSERS, EmptyDocument, ParseError, load_document
from src.schemas import Action, Finding, Invoice, SourceFormat


@dataclass
class IngestionResult:
    invoice: Invoice
    raw_text: str
    raw_hash: str
    method: Literal["parser", "llm", "failed"]
    attempts: int = 0  # LLM extraction attempts (0 for parsed files)
    findings: list[Finding] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)


def ingest(
    path: Path | str, *, catalog: Iterable[str], agent: ExtractionAgent | None
) -> IngestionResult:
    path = Path(path)
    raw = path.read_bytes()
    raw_hash = hashlib.sha256(raw).hexdigest()
    catalog = list(catalog)
    events: list[dict] = []
    suffix = path.suffix.lower()

    def failed(text: str, fmt: SourceFormat | None, reason: str, **evidence) -> IngestionResult:
        events.append({"event": "extraction_failed", "reason": reason})
        invoice = Invoice(source_file=path.name, source_format=fmt)
        finding = Finding(
            code="extraction_failed",
            action=Action.HOLD,
            message=f"Could not extract this invoice confidently: {reason}",
            evidence=evidence,
        )
        return IngestionResult(
            invoice, text, raw_hash, "failed", evidence.get("attempts", 0), [finding], events
        )

    if suffix in STRUCTURED_PARSERS:
        text = raw.decode(errors="replace")
        fmt: SourceFormat = suffix[1:]  # type: ignore[assignment]
        try:
            invoice = STRUCTURED_PARSERS[suffix](text, path.name)
            events.append({"event": "parsed", "parser": fmt})
            return IngestionResult(
                normalize_invoice(invoice, catalog), text, raw_hash, "parser", events=events
            )
        except ParseError as e:
            events.append({"event": "parse_failed", "parser": fmt, "error": str(e)})
    else:
        try:
            text, fmt = load_document(path)
        except EmptyDocument as e:
            return failed("", "pdf" if suffix == ".pdf" else "txt", str(e))
        except Exception as e:  # corrupt PDF, unreadable file
            return failed("", None, f"could not read file: {e}")

    if agent is None:
        return failed(text, fmt, "no LLM configured for unstructured documents")
    outcome = agent.extract(text)
    events.append(
        {"event": "llm_extraction", "attempts": outcome.attempts, "history": outcome.history}
    )
    if not outcome.ok or outcome.extracted is None:
        return failed(
            text,
            fmt,
            "grounding checks still failing after retries",
            attempts=outcome.attempts,
            errors=outcome.errors,
        )
    invoice = normalize_invoice(to_invoice(outcome.extracted, path.name, fmt), catalog)
    return IngestionResult(invoice, text, raw_hash, "llm", outcome.attempts, events=events)


def main(argv: list[str] | None = None) -> int:
    from src.config import Settings, load_policy
    from src.db import DEFAULT_DB_PATH, connect, get_inventory, init_db
    from src.llm.client import LLMClient

    parser = argparse.ArgumentParser(description="Run ingestion on invoice files")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)

    policy = load_policy()
    settings = Settings.from_env()
    with connect(DEFAULT_DB_PATH) as conn:
        init_db(conn)
        catalog = list(get_inventory(conn))
    llm = LLMClient(
        settings,
        policy.llm,
        on_event=lambda e, p: print(json.dumps({"event": e, **p}), file=sys.stderr),
    )
    agent = ExtractionAgent(llm, policy.llm.max_extraction_retries)
    for path in args.files:
        result = ingest(path, catalog=catalog, agent=agent)
        print(
            json.dumps(
                {
                    "file": path.name,
                    "method": result.method,
                    "attempts": result.attempts,
                    "findings": [f.model_dump(mode="json") for f in result.findings],
                    "invoice": result.invoice.model_dump(mode="json", exclude_defaults=True),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
