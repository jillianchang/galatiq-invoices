"""Deterministic normalization: OCR repair, money, dates, invoice numbers,
vendor names, and item names to catalog SKUs.

Everything here is pure and rule-based. Anything it can't resolve is left
as-is (e.g. an unknown item keeps sku=None) for validation to flag.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from src.schemas import Invoice, LineItem

# ---------- OCR repair ----------

# A run of digits/O/o, commas and dots that isn't part of a word, e.g. "2O26" or "3,500.O0".
_NUMERIC_RUN = re.compile(r"(?<![A-Za-z])[\dOo][\dOo,.]*[\dOo](?![A-Za-z])")


def fix_ocr_digits(text: str) -> str:
    """Replace the letter O with zero inside numeric tokens: '2O26' -> '2026',
    '$3,500.O0' -> '$3,500.00'. Words ('Good', 'PO-2026') are untouched."""

    def repl(m: re.Match[str]) -> str:
        token = m.group(0)
        if not any(c.isdigit() for c in token):
            return token
        return token.replace("O", "0").replace("o", "0")

    return _NUMERIC_RUN.sub(repl, text)


# ---------- money ----------


def parse_money(value: object) -> Decimal | None:
    """'$3,500.O0' -> Decimal('3500.00'); '(250.00)' -> Decimal('-250.00'); junk -> None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int | float):
        return Decimal(str(value))
    text = fix_ocr_digits(str(value)).strip()
    negative = text.startswith("(") and text.endswith(")")
    text = re.sub(r"(?i)usd|eur|gbp|[$€£,\s()]", "", text)
    if not text:
        return None
    try:
        amount = Decimal(text)
    except InvalidOperation:
        return None
    return -amount if negative else amount


# ---------- dates ----------

DATE_FORMATS = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%b %d %Y",
    "%B %d %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d-%b-%Y",
    "%d-%B-%Y",
    "%d %b %Y",
    "%d %B %Y",
)


def parse_date(raw: str | None) -> date | None:
    """Parse the date formats seen on invoices; relative dates ('yesterday') -> None."""
    if not raw:
        return None
    text = re.sub(r"\s+", " ", fix_ocr_digits(raw.strip()))
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


# ---------- identifiers ----------

_INVOICE_NUMBER = re.compile(r"#?\s*(?:INV(?:OICE)?)?\s*(?:NO\.?|NUMBER|#)?[\s#:.\-]*(\d+)")


def normalize_invoice_number(raw: str | None) -> str | None:
    """'1002', 'INV 1012', '#INV-1010' -> 'INV-1002', 'INV-1012', 'INV-1010'."""
    if raw is None or not str(raw).strip():
        return None
    text = fix_ocr_digits(str(raw)).strip().upper()
    m = _INVOICE_NUMBER.fullmatch(text)
    if m:
        return f"INV-{m.group(1)}"
    return re.sub(r"\s+", "-", text)


_VENDOR_SUFFIXES = {"inc", "llc", "ltd", "co", "corp", "corporation", "company", "limited"}


def normalize_vendor(name: str | None) -> str | None:
    """Comparison key for vendor names: 'Widgets Inc.' -> 'widgets'."""
    if not name or not name.strip():
        return None
    words = re.sub(r"[^a-z0-9 ]", " ", name.lower()).split()
    while len(words) > 1 and words[-1] in _VENDOR_SUFFIXES:
        words.pop()
    return " ".join(words)


def _item_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def match_item(raw_name: str | None, catalog: Iterable[str]) -> tuple[str | None, str | None]:
    """Map an item name to a catalog SKU by case/spacing/punctuation only.

    Returns (sku or None, parenthetical note or None):
    'Widget A' -> ('WidgetA', None); 'WidgetA (rush order)' -> ('WidgetA', 'rush order');
    'WidgetC' -> (None, None). No fuzzy matching: 'WidgetC' must never become 'WidgetA'.
    """
    if not raw_name:
        return None, None
    note = None
    m = re.search(r"\(([^)]*)\)", raw_name)
    if m:
        note = m.group(1).strip() or None
        raw_name = raw_name[: m.start()] + raw_name[m.end() :]
    lookup = {_item_key(sku): sku for sku in catalog}
    return lookup.get(_item_key(raw_name)), note


# ---------- whole invoice ----------


def _clean(text: str | None) -> str | None:
    if text is None:
        return None
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text or None


def normalize_invoice(invoice: Invoice, catalog: Iterable[str]) -> Invoice:
    """Return a normalized copy: identifiers, dates, SKUs, blank strings -> None."""
    catalog = list(catalog)
    lines = []
    for line in invoice.line_items:
        sku, paren_note = match_item(line.raw_name, catalog)
        notes = [n for n in (line.note, paren_note) if n]
        lines.append(
            LineItem(
                raw_name=_clean(line.raw_name),
                sku=sku,
                quantity=line.quantity,
                unit_price=line.unit_price,
                amount=line.amount,
                note="; ".join(dict.fromkeys(notes)) or None,
            )
        )
    return invoice.model_copy(
        update={
            "invoice_number": normalize_invoice_number(invoice.invoice_number),
            "vendor_name": _clean(invoice.vendor_name),
            "vendor_address": _clean(invoice.vendor_address),
            "invoice_date_raw": _clean(invoice.invoice_date_raw),
            "due_date_raw": _clean(invoice.due_date_raw),
            "invoice_date": invoice.invoice_date or parse_date(invoice.invoice_date_raw),
            "due_date": invoice.due_date or parse_date(invoice.due_date_raw),
            "currency": (_clean(invoice.currency) or "").upper() or None,
            "payment_terms": _clean(invoice.payment_terms),
            "notes": _clean(invoice.notes),
            "line_items": lines,
        }
    )
