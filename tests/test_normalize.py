from datetime import date
from decimal import Decimal

import pytest

from src.normalize import (
    fix_ocr_digits,
    match_item,
    normalize_invoice_number,
    normalize_vendor,
    parse_date,
    parse_money,
)

CATALOG = ["WidgetA", "WidgetB", "GadgetX", "FakeItem"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("26-Jan-2O26", "26-Jan-2026"),
        ("$3,500.O0", "$3,500.00"),
        ("Good PO-20260115", "Good PO-20260115"),  # words untouched
        ("Net 30", "Net 30"),
    ],
)
def test_fix_ocr_digits(raw, expected):
    assert fix_ocr_digits(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("$5,000.00", Decimal("5000.00")),
        ("$3,500.O0", Decimal("3500.00")),
        ("(250.00)", Decimal("-250.00")),
        ("-5", Decimal("-5")),
        (750, Decimal("750")),
        ("EUR 225.00", Decimal("225.00")),
        ("n/a", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_money(raw, expected):
    assert parse_money(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2026-01-15", date(2026, 1, 15)),
        ("Jan 30 2026", date(2026, 1, 30)),
        ("01/28/2026", date(2026, 1, 28)),
        ("January 27, 2026", date(2026, 1, 27)),
        ("26-Jan-2O26", date(2026, 1, 26)),
        ("25-Feb-2026", date(2026, 2, 25)),
        ("yesterday", None),
        (None, None),
    ],
)
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("INV-1001", "INV-1001"),
        ("1002", "INV-1002"),
        ("INV 1012", "INV-1012"),
        ("#INV-1010", "INV-1010"),
        ("inv-1O13", "INV-1013"),
        ("", None),
        (None, None),
    ],
)
def test_normalize_invoice_number(raw, expected):
    assert normalize_invoice_number(raw) == expected


def test_normalize_vendor():
    assert normalize_vendor("Widgets Inc.") == normalize_vendor("WIDGETS INC") == "widgets"
    assert normalize_vendor("Precision Parts Ltd.") == "precision parts"
    assert normalize_vendor("Co.") == "co"  # never empty a one-word name
    assert normalize_vendor("  ") is None


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("WidgetA", ("WidgetA", None)),
        ("Widget A", ("WidgetA", None)),
        ("Gadget X", ("GadgetX", None)),
        ("widgetb", ("WidgetB", None)),
        ("WidgetA (rush order)", ("WidgetA", "rush order")),
        ("WidgetC", (None, None)),  # unknown stays unknown: no fuzzy matching
        ("SuperGizmo", (None, None)),
        (None, (None, None)),
    ],
)
def test_match_item(raw, expected):
    assert match_item(raw, CATALOG) == expected
