"""Deterministic rule catalog (PLAN.md section 6). Rules only produce findings."""

from __future__ import annotations

from src.rules.catalog import check_catalog
from src.rules.codes import CODES, finding
from src.rules.commercial import check_commercial
from src.rules.context import Inventory, RuleContext
from src.rules.integrity import check_integrity
from src.rules.text_signals import check_text_signals
from src.schemas import Finding


def check_fraud_pattern(findings: list[Finding]) -> list[Finding]:
    """Zero-stock item billed + payment pressure + over threshold -> REJECT (D17).
    Makes 1003's rejection independent of the LLM."""
    codes = {f.code for f in findings}
    zero_stock = [
        item["item"]
        for f in findings
        if f.code == "stock_exceeded"
        for item in f.evidence.get("items", [])
        if item.get("stock") == 0
    ]
    if zero_stock and {"payment_pressure", "over_threshold"} <= codes:
        return [
            finding(
                "fraud_pattern",
                f"Fraud pattern: bills a zero-stock item ({', '.join(zero_stock)}), pressures "
                "for immediate payment, and is over the approval threshold.",
                {"zero_stock_items": zero_stock},
            )
        ]
    return []


def run_rules(ctx: RuleContext) -> list[Finding]:
    findings = [
        *check_integrity(ctx),
        *check_catalog(ctx),
        *check_commercial(ctx),
        *check_text_signals(ctx),
    ]
    return findings + check_fraud_pattern(findings)


__all__ = ["CODES", "Inventory", "RuleContext", "check_text_signals", "finding", "run_rules"]
