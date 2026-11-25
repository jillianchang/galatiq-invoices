"""Grounding checks for LLM extraction (D2).

`check_grounding` verifies every extracted value actually appears in the
source document; failures are fed back to the model as specific errors.

`check_math` reports internal inconsistencies. These are NOT extraction
failures on their own: real invoices have math errors (1007, 1013) that
validation must flag. They trigger one re-read, then the values are accepted
as written.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from src.normalize import fix_ocr_digits

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
TOLERANCE = Decimal("0.01")

MONEY_FIELDS = ("subtotal", "tax_amount", "shipping", "total")


def source_numbers(text: str) -> set[Decimal]:
    """Every number in the document (OCR-repaired), with both signs; percentages
    also as fractions ('5%' -> 5 and 0.05)."""
    text = fix_ocr_digits(text)
    numbers: set[Decimal] = set()
    for token in _NUMBER.findall(text):
        try:
            value = Decimal(token.replace(",", ""))
        except InvalidOperation:
            continue
        numbers.update({value, -value})
    for pct in _PERCENT.findall(text):
        numbers.add(Decimal(pct) / 100)
    return numbers


def _squash(text: str) -> str:
    """Lowercase alphanumerics only, OCR-repaired: for substring matching."""
    return re.sub(r"[^a-z0-9]", "", fix_ocr_digits(text).lower())


def _as_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def check_grounding(extracted: Any, source: str) -> list[str]:
    """Errors for values that don't appear in `source`. `extracted` is an
    ExtractedInvoice (duck-typed to keep this module schema-agnostic)."""
    numbers = source_numbers(source)
    squashed = _squash(source)
    errors: list[str] = []

    def number_ok(label: str, value: Any) -> None:
        d = _as_decimal(value)
        if d is not None and d not in numbers:
            errors.append(
                f"{label}={value} does not appear in the document. Copy numbers exactly "
                "as written; use null if a value is not stated (do not compute it)."
            )

    def text_ok(label: str, value: str | None) -> None:
        if value and _squash(value) and _squash(value) not in squashed:
            errors.append(f"{label}={value!r} does not appear in the document; copy it verbatim.")

    for name in MONEY_FIELDS:
        number_ok(name, getattr(extracted, name, None))
    number_ok("tax_rate_percent", getattr(extracted, "tax_rate_percent", None))
    text_ok("vendor_name", extracted.vendor_name)
    text_ok("invoice_date_raw", extracted.invoice_date_raw)
    text_ok("due_date_raw", extracted.due_date_raw)
    if extracted.invoice_number:
        digits = re.sub(r"\D", "", fix_ocr_digits(extracted.invoice_number))
        if digits and digits not in re.sub(r"\D", "", fix_ocr_digits(source)):
            errors.append(
                f"invoice_number={extracted.invoice_number!r} does not appear in the document."
            )

    if not extracted.line_items:
        errors.append("No line items were extracted; every invoice lists at least one item.")
    for i, line in enumerate(extracted.line_items, start=1):
        text_ok(f"line {i} name", line.name)
        number_ok(f"line {i} ({line.name}) quantity", line.quantity)
        number_ok(f"line {i} ({line.name}) unit_price", line.unit_price)
        number_ok(f"line {i} ({line.name}) amount", line.amount)
    return errors


def _money(d: Decimal) -> str:
    return f"{d:,.2f}"


def _qty(d: Decimal) -> str:
    return f"{d.normalize():f}"


def check_math(extracted: Any) -> list[str]:
    """Internal inconsistencies between line items, subtotal, tax, and total."""
    issues: list[str] = []
    line_sum = Decimal(0)
    complete = True
    for i, line in enumerate(extracted.line_items, start=1):
        qty, price, amount = (_as_decimal(v) for v in (line.quantity, line.unit_price, line.amount))
        if qty is not None and price is not None:
            computed = qty * price
            if amount is not None and abs(computed - amount) > TOLERANCE:
                issues.append(
                    f"line {i} ({line.name}): {_qty(qty)} x {_money(price)} = "
                    f"{_money(computed)}, not {_money(amount)}"
                )
            line_sum += amount if amount is not None else computed
        elif amount is not None:
            line_sum += amount
        else:
            complete = False

    subtotal = _as_decimal(extracted.subtotal)
    if complete and subtotal is not None and abs(line_sum - subtotal) > TOLERANCE:
        issues.append(f"line items sum to {_money(line_sum)}, but subtotal is {_money(subtotal)}")

    total = _as_decimal(extracted.total)
    base = subtotal if subtotal is not None else (line_sum if complete else None)
    if base is not None and total is not None:
        expected = (
            base + (_as_decimal(extracted.tax_amount) or 0) + (_as_decimal(extracted.shipping) or 0)
        )
        if abs(expected - total) > TOLERANCE:
            issues.append(
                f"subtotal + tax + shipping = {_money(expected)}, but total is {_money(total)}"
            )
    return issues
