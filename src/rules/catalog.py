"""Catalog and stock: unknown items, stock (aggregated per SKU), prices."""

from __future__ import annotations

import re
from collections import defaultdict
from decimal import Decimal

from src.rules.codes import finding
from src.rules.context import RuleContext
from src.schemas import Action, Finding

_SAMPLE = re.compile(r"\b(sample|replacement)s?\b", re.IGNORECASE)


def check_catalog(ctx: RuleContext) -> list[Finding]:
    lines = ctx.invoice.line_items
    if not lines:
        return []
    out: list[Finding] = []

    unknown = [li.raw_name for li in lines if li.sku is None]
    if len(unknown) == len(lines):
        out.append(
            finding(
                "no_catalog_items",
                f"None of the items are in the inventory catalog: {', '.join(map(str, unknown))}.",
                {"items": unknown},
            )
        )
    elif unknown:
        out.append(
            finding(
                "unknown_item",
                f"Items not in the inventory catalog: {', '.join(map(str, unknown))}.",
                {"items": unknown},
            )
        )

    # Aggregate per SKU: 1013 bills WidgetA on three lines that each fit, but not together.
    requested: dict[str, Decimal] = defaultdict(Decimal)
    for li in lines:
        if li.sku and li.quantity is not None and li.quantity > 0:
            requested[li.sku] += li.quantity
    over = [
        {"item": sku, "requested": str(qty), "stock": ctx.inventory[sku][0]}
        for sku, qty in requested.items()
        if sku in ctx.inventory and qty > ctx.inventory[sku][0]
    ]
    if over:
        out.append(
            finding(
                "stock_exceeded",
                "Quantity exceeds available stock: "
                + "; ".join(
                    f"{o['item']} {o['requested']} requested, {o['stock']} in stock" for o in over
                )
                + ".",
                {"items": over},
            )
        )

    out.extend(_price_findings(ctx))

    flagged = [li for li in lines if li.note and _SAMPLE.search(li.note)]
    if flagged:
        out.append(
            finding(
                "billed_sample_or_replacement",
                "Billed lines are marked as samples or replacements, which are usually free: "
                + "; ".join(f"{li.raw_name} x{li.quantity} ({li.note})" for li in flagged)
                + ".",
                {"lines": [{"item": li.raw_name, "note": li.note} for li in flagged]},
            )
        )
    return out


def _price_findings(ctx: RuleContext) -> list[Finding]:
    rate = ctx.fx_rate()
    if rate is None:
        return []  # unknown currency: non_usd_currency already holds it
    tol = ctx.policy.price_tolerance
    out = []
    for li in ctx.invoice.line_items:
        if not li.sku or li.unit_price is None or li.sku not in ctx.inventory:
            continue
        catalog_price = ctx.inventory[li.sku][1]
        if not catalog_price:  # NULL (e.g. FakeItem): nothing to compare against
            continue
        price_usd = li.unit_price * rate
        pct = (price_usd - catalog_price) / catalog_price
        if pct <= tol.silent_pct:
            continue
        reason = li.note
        action = Action.WARN if reason and pct <= tol.warn_with_reason_pct else Action.HOLD
        out.append(
            finding(
                "price_over_catalog",
                f"{li.raw_name} billed at ${price_usd:,.2f} vs catalog ${catalog_price:,.2f} "
                f"(+{pct:.0%})"
                + (f", reason given: {reason!r}." if reason else ", no reason given."),
                {
                    "item": li.sku,
                    "unit_price_usd": str(price_usd),
                    "catalog_price": str(catalog_price),
                    "pct_over": f"{pct:.4f}",
                    "reason": reason,
                },
                action=action,
            )
        )
    return out
