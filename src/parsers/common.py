from __future__ import annotations

import re
from decimal import Decimal

from src.normalize import parse_money


class ParseError(ValueError):
    """The document isn't in a shape this parser understands."""


def money(value: object) -> Decimal | None:
    return parse_money(value)


def text_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def percent_from_label(label: str) -> Decimal | None:
    """'Tax (6%):' -> Decimal('0.06')."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", label)
    return Decimal(m.group(1)) / 100 if m else None
