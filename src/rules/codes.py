"""Registry of every finding code and its default action (PLAN.md section 6).

Actions are fixed by code, not chosen by whoever raises the finding: an LLM
agent can name a code but can never pick a softer action for it (D3).
"""

from __future__ import annotations

from typing import Any

from src.schemas import Action, Finding, FindingSource

A = Action
CODES: dict[str, Action] = {
    # data integrity
    "missing_vendor": A.REJECT,
    "nonpositive_quantity": A.REJECT,
    "missing_total": A.RETURN,
    "line_math_mismatch": A.RETURN,
    "subtotal_mismatch": A.RETURN,
    "tax_mismatch": A.RETURN,
    "total_mismatch": A.RETURN,
    "missing_due_date": A.HOLD,
    "unparseable_due_date": A.HOLD,
    "missing_invoice_number": A.HOLD,
    "incomplete_line_item": A.HOLD,
    "extraction_failed": A.HOLD,
    # catalog and stock
    "no_catalog_items": A.REJECT,
    "unknown_item": A.HOLD,
    "stock_exceeded": A.HOLD,
    "price_over_catalog": A.HOLD,  # WARN when modest and a reason is stated
    "billed_sample_or_replacement": A.WARN,
    # commercial and policy
    "over_threshold": A.SCRUTINY,
    "near_threshold": A.WARN,
    "non_usd_currency": A.HOLD,
    "terms_due_mismatch": A.WARN,
    # history
    "reprocessed_submission": A.INFO,
    "duplicate": A.DUPLICATE,
    "revision_of_paid_invoice": A.HOLD,
    "stale_revision": A.HOLD,
    "supersedes_unpaid_invoice": A.INFO,
    "corrected_resubmission": A.INFO,
    # text signals (deterministic backstop) and composite
    "vendor_identity_change": A.HOLD,
    "payment_pressure": A.HOLD,
    "prompt_injection": A.HOLD,
    "fraud_pattern": A.REJECT,
    # investigator agent only
    "suspected_fraud": A.REJECT,
    "suspicious_vendor_address": A.HOLD,
    "bank_detail_change": A.HOLD,
    "other_concern": A.WARN,
    "investigation_incomplete": A.WARN,
}


def finding(
    code: str,
    message: str,
    evidence: dict[str, Any] | None = None,
    *,
    action: Action | None = None,
    source: FindingSource = FindingSource.RULE,
) -> Finding:
    if code not in CODES:
        raise KeyError(f"unknown finding code: {code}")
    return Finding(
        code=code,
        action=action or CODES[code],
        message=message,
        evidence=evidence or {},
        source=source,
    )
