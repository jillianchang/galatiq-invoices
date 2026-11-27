"""Expected results (expected.yaml) and ground-truth extractions (extraction_truth.yaml)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import yaml

from src.agents.extraction import ExtractedInvoice
from src.config import ROOT

EVALS = ROOT / "evals"
INVOICE_DIR = ROOT / "data" / "invoices"


def load_expected() -> dict[str, dict[str, Any]]:
    return yaml.safe_load((EVALS / "expected.yaml").read_text())["invoices"]


def load_truth() -> dict[str, ExtractedInvoice]:
    raw = yaml.safe_load((EVALS / "extraction_truth.yaml").read_text())["invoices"]
    return {name: ExtractedInvoice.model_validate(data) for name, data in raw.items()}


def expectation(name: str, scenario: str, expected: dict | None = None) -> dict[str, Any]:
    """The expectation for one file, with batch overrides applied in the batch scenario."""
    exp = dict((expected or load_expected())[name])
    overrides = exp.pop("batch", None) or {}
    if scenario == "batch":
        exp.update(overrides)
        if "outcome" in overrides and overrides["outcome"] != "PAID":
            exp.pop("amount", None)
            exp.pop("outcome_llm", None)
    exp.setdefault("required_findings", [])
    exp.setdefault("llm_findings", [])
    return exp


def expected_totals(scenario: str, files: list[str], expected: dict | None = None) -> dict:
    """Paid count and amount, duplicates blocked, revisions held, per the expectations."""
    exps = [expectation(f, scenario, expected) for f in files]
    paid = [e for e in exps if e["outcome"] == "PAID"]
    return {
        "paid": len(paid),
        "paid_amount": str(sum((Decimal(e["amount"]) for e in paid), Decimal("0.00"))),
        "duplicates_blocked": sum(e["outcome"] == "DUPLICATE_BLOCKED" for e in exps),
        "revisions_held": sum("revision_of_paid_invoice" in e["required_findings"] for e in exps),
    }
