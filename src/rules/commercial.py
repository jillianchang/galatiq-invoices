"""Commercial and policy: approval threshold, currency, payment terms."""

from __future__ import annotations

import re
from datetime import timedelta

from src.rules.codes import finding
from src.rules.context import RuleContext
from src.schemas import Finding


def net_days(terms: str | None) -> int | None:
    """'Net 30' -> 30; 'Immediate' / 'Due on receipt' -> 0; unknown -> None."""
    if not terms:
        return None
    text = terms.lower()
    m = re.search(r"net\s*(\d+)", text)
    if m:
        return int(m.group(1))
    if re.search(r"immediate|on receipt|upon receipt", text):
        return 0
    return None


def check_commercial(ctx: RuleContext) -> list[Finding]:
    inv = ctx.invoice
    approval = ctx.policy.approval
    out: list[Finding] = []

    total_usd = ctx.total_usd
    if total_usd is not None:
        threshold = approval.threshold_usd
        if total_usd > threshold:
            out.append(
                finding(
                    "over_threshold",
                    f"Total ${total_usd:,.2f} is over the ${threshold:,.0f} approval threshold; "
                    "additional scrutiny required.",
                    {"total_usd": str(total_usd), "threshold": str(threshold)},
                )
            )
        elif total_usd >= threshold * (1 - approval.near_threshold_pct):
            out.append(
                finding(
                    "near_threshold",
                    f"Total ${total_usd:,.2f} is just under the ${threshold:,.0f} approval "
                    "threshold.",
                    {"total_usd": str(total_usd), "threshold": str(threshold)},
                )
            )

    if ctx.currency != "USD":
        rate = ctx.fx_rate()
        converted = f" (about ${total_usd:,.2f} at {rate} USD/{ctx.currency})" if total_usd else ""
        out.append(
            finding(
                "non_usd_currency",
                f"Invoice is in {ctx.currency}{converted}; the exchange rate needs confirming.",
                {
                    "currency": ctx.currency,
                    "rate": str(rate) if rate else None,
                    "total_usd": str(total_usd) if total_usd else None,
                },
            )
        )

    days = net_days(inv.payment_terms)
    if days is not None and inv.invoice_date and inv.due_date:
        expected = inv.invoice_date + timedelta(days=days)
        off = abs((inv.due_date - expected).days)
        if off > ctx.policy.terms_tolerance_days:
            out.append(
                finding(
                    "terms_due_mismatch",
                    f"Terms {inv.payment_terms!r} from {inv.invoice_date} imply a due date of "
                    f"{expected}, but the invoice says {inv.due_date} ({off} days off).",
                    {"expected": str(expected), "stated": str(inv.due_date), "days_off": off},
                )
            )
    return out
