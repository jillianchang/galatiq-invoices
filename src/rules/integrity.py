"""Data integrity: missing fields, invalid quantities, and arithmetic."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from src.rules.codes import finding
from src.rules.context import RuleContext
from src.schemas import Finding

CENTS = Decimal("0.01")


def check_integrity(ctx: RuleContext) -> list[Finding]:
    inv = ctx.invoice
    tol = ctx.policy.money_tolerance_usd
    out: list[Finding] = []

    if not inv.vendor_name:
        out.append(finding("missing_vendor", "The vendor name is missing."))
    if not inv.invoice_number:
        out.append(
            finding(
                "missing_invoice_number",
                "The invoice number is missing, so duplicates can't be detected.",
            )
        )

    bad_qty = [li for li in inv.line_items if li.quantity is not None and li.quantity <= 0]
    if bad_qty:
        names = ", ".join(f"{li.raw_name} ({li.quantity})" for li in bad_qty)
        out.append(
            finding(
                "nonpositive_quantity",
                f"Quantity must be positive: {names}.",
                {"lines": [{"item": li.raw_name, "quantity": str(li.quantity)} for li in bad_qty]},
            )
        )

    incomplete = [
        li
        for li in inv.line_items
        if li.quantity is None or (li.unit_price is None and li.amount is None)
    ]
    if incomplete:
        out.append(
            finding(
                "incomplete_line_item",
                "Line items are missing a quantity or price: "
                + ", ".join(str(li.raw_name) for li in incomplete)
                + ".",
                {"items": [li.raw_name for li in incomplete]},
            )
        )

    if inv.total is None:
        out.append(finding("missing_total", "No invoice total was found."))

    wrong_lines = []
    for li in inv.line_items:
        if li.amount is not None and li.quantity is not None and li.unit_price is not None:
            computed = li.quantity * li.unit_price
            if abs(computed - li.amount) > tol:
                wrong_lines.append((li, computed))
    if wrong_lines:
        out.append(
            finding(
                "line_math_mismatch",
                "Line amounts don't match quantity x unit price: "
                + "; ".join(
                    f"{li.raw_name} {li.quantity} x {ctx.money(li.unit_price)} = "
                    f"{ctx.money(c)}, invoice says {ctx.money(li.amount)}"
                    for li, c in wrong_lines
                )
                + ".",
                {
                    "lines": [
                        {"item": li.raw_name, "stated": str(li.amount), "computed": str(c)}
                        for li, c in wrong_lines
                    ]
                },
            )
        )

    line_sum = ctx.line_sum
    if inv.subtotal is not None and line_sum is not None and abs(line_sum - inv.subtotal) > tol:
        out.append(
            finding(
                "subtotal_mismatch",
                f"Line items sum to {ctx.money(line_sum)}, but the stated subtotal is "
                f"{ctx.money(inv.subtotal)}.",
                {"stated": str(inv.subtotal), "computed": str(line_sum)},
            )
        )

    base = inv.subtotal if inv.subtotal is not None else line_sum
    if inv.tax_rate and inv.tax_amount is not None and base is not None:
        expected_tax = (base * inv.tax_rate).quantize(CENTS, rounding=ROUND_HALF_UP)
        if abs(expected_tax - inv.tax_amount) > tol:
            out.append(
                finding(
                    "tax_mismatch",
                    f"Tax at {inv.tax_rate:.2%} of {ctx.money(base)} is {ctx.money(expected_tax)}, "
                    f"but the stated tax is {ctx.money(inv.tax_amount)}.",
                    {"stated": str(inv.tax_amount), "computed": str(expected_tax)},
                )
            )

    if inv.total is not None and base is not None:
        expected = base + (inv.tax_amount or 0) + (inv.shipping or 0)
        diff = inv.total - expected
        if abs(diff) > tol:
            out.append(
                finding(
                    "total_mismatch",
                    f"Subtotal + tax + shipping = {ctx.money(expected)}, but the stated total is "
                    f"{ctx.money(inv.total)} ({ctx.money(abs(diff))} "
                    f"{'over' if diff > 0 else 'under'}).",
                    {"stated": str(inv.total), "computed": str(expected), "difference": str(diff)},
                )
            )

    if inv.due_date_raw is None:
        out.append(finding("missing_due_date", "The due date is missing."))
    elif inv.due_date is None:
        out.append(
            finding(
                "unparseable_due_date",
                f"The due date {inv.due_date_raw!r} isn't a calendar date.",
                {"raw": inv.due_date_raw},
            )
        )
    return out
