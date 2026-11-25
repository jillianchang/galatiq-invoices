"""Validation stage: history checks, deterministic rules, then the investigator.

Validation only produces findings; it never decides (D6). The policy gate and
approval agent decide in the approval stage.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from src.agents.investigator import InvestigationOutcome, InvestigatorAgent
from src.config import Policy
from src.history import check_history
from src.ingestion import IngestionResult
from src.rules import RuleContext, check_text_signals, run_rules
from src.rules.context import Inventory
from src.schemas import Action, Finding


@dataclass
class ValidationResult:
    findings: list[Finding]
    investigation: InvestigationOutcome | None = None
    events: list[dict] = field(default_factory=list)


def validate(
    ingestion: IngestionResult,
    *,
    conn: sqlite3.Connection,
    policy: Policy,
    inventory: Inventory,
    investigator: InvestigatorAgent | None = None,
) -> ValidationResult:
    ctx = RuleContext(ingestion.invoice, ingestion.raw_text, inventory, policy)
    findings = list(ingestion.findings)
    events: list[dict] = []

    if ingestion.method == "failed":
        # Nothing reliable to validate: an empty invoice would trip "missing vendor" and
        # wrongly REJECT. Keep the HOLD and add only the raw-text safety signals.
        findings += check_text_signals(ctx)
        events.append({"event": "rules_skipped", "reason": "extraction failed"})
        return ValidationResult(findings, None, events)

    history = check_history(conn, ingestion.invoice, ingestion.raw_hash)
    findings += history
    findings += run_rules(ctx)
    events.append({"event": "rules", "codes": [f.code for f in findings]})

    investigation = None
    if investigator is not None and not any(f.action == Action.DUPLICATE for f in findings):
        investigation = investigator.investigate(ctx, findings)
        findings += investigation.findings
        events.append(
            {
                "event": "investigation",
                "rounds": investigation.rounds,
                "tool_calls": [c["name"] for c in investigation.tool_calls],
                "added": [f.code for f in investigation.findings],
                "dropped": investigation.dropped,
                "error": investigation.error,
            }
        )
    return ValidationResult(findings, investigation, events)


def main(argv: list[str] | None = None) -> int:
    """Try ingestion + validation on a few invoices:
    uv run python -m src.validation data/invoices/invoice_1005.json [...]"""
    import argparse
    import json
    import sys
    from pathlib import Path

    from src.agents.extraction import ExtractionAgent
    from src.config import Settings, load_policy
    from src.db import DEFAULT_DB_PATH, connect, get_inventory, init_db
    from src.gate import compute_gate
    from src.ingestion import ingest
    from src.llm.client import LLMClient
    from src.tools.db_tools import InvestigatorTools

    parser = argparse.ArgumentParser(description="Run ingestion and validation on invoice files")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--no-llm", action="store_true", help="deterministic checks only")
    args = parser.parse_args(argv)

    policy = load_policy()
    conn = connect(DEFAULT_DB_PATH)
    init_db(conn)
    inventory = get_inventory(conn)
    llm = (
        None
        if args.no_llm
        else LLMClient(
            Settings.from_env(),
            policy.llm,
            on_event=lambda e, p: print(json.dumps({"event": e, **p}), file=sys.stderr),
        )
    )
    extractor = ExtractionAgent(llm, policy.llm.max_extraction_retries) if llm else None
    investigator = (
        InvestigatorAgent(llm, InvestigatorTools(conn, inventory, policy)) if llm else None
    )
    for path in args.files:
        ingestion = ingest(path, catalog=inventory, agent=extractor)
        result = validate(
            ingestion, conn=conn, policy=policy, inventory=inventory, investigator=investigator
        )
        inv = result.investigation
        print(
            json.dumps(
                {
                    "file": path.name,
                    "gate": compute_gate(result.findings),
                    "findings": [
                        f"{f.action:<9} {f.code} [{f.source}]: {f.message}" for f in result.findings
                    ],
                    "investigator": None
                    if inv is None
                    else {
                        "summary": inv.summary,
                        "rounds": inv.rounds,
                        "tool_calls": [c["name"] for c in inv.tool_calls],
                        "dropped": inv.dropped,
                        "error": inv.error,
                    },
                },
                indent=2,
                default=str,
            )
        )
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
