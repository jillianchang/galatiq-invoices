"""Every structured invoice in data/invoices parses to the right values."""

from datetime import date
from decimal import Decimal

import pytest

from src.config import ROOT
from src.normalize import normalize_invoice
from src.parsers import (
    STRUCTURED_PARSERS,
    ParseError,
    load_document,
    parse_csv,
    parse_json,
    parse_xml,
)

DATA = ROOT / "data" / "invoices"
CATALOG = ["WidgetA", "WidgetB", "GadgetX", "FakeItem"]


def parse(name):
    path = DATA / name
    invoice = STRUCTURED_PARSERS[path.suffix](path.read_text(), name)
    return normalize_invoice(invoice, CATALOG)


def items(invoice):
    return [(li.sku or li.raw_name, li.quantity, li.unit_price) for li in invoice.line_items]


def test_json_basic_1004():
    inv = parse("invoice_1004.json")
    assert inv.invoice_number == "INV-1004"
    assert inv.vendor_name == "Precision Parts Ltd."
    assert inv.invoice_date == date(2026, 1, 22) and inv.due_date == date(2026, 2, 22)
    assert items(inv) == [("WidgetA", 3, Decimal("250.00")), ("WidgetB", 2, Decimal("500.00"))]
    assert (inv.subtotal, inv.tax_rate, inv.tax_amount, inv.total) == (
        Decimal("1750.00"),
        Decimal("0.08"),
        Decimal("140.00"),
        Decimal("1890.00"),
    )
    assert inv.currency == "USD" and inv.source_format == "json"


def test_json_revision_and_notes_1004_revised():
    inv = parse("invoice_1004_revised.json")
    assert inv.revision == "R1"
    assert "PO amendment" in inv.notes
    assert len(inv.line_items) == 3 and inv.total == Decimal("5940.00")


def test_json_invalid_data_kept_as_written_1009():
    inv = parse("invoice_1009.json")
    assert inv.vendor_name is None  # "" -> missing, for validation to flag
    assert inv.due_date is None and inv.due_date_raw is None
    assert inv.line_items[0].quantity == Decimal("-5")
    assert inv.total == Decimal("-250.00") and inv.subtotal == Decimal("1000.00")


def test_json_line_notes_1013():
    inv = parse("invoice_1013.json")
    assert len(inv.line_items) == 8
    assert [li.note for li in inv.line_items][3:] == [
        "Volume discount",
        "Volume discount",
        "Expedited",
        "Replacement",
        "Sample",
    ]
    assert inv.total == Decimal("22562.80")


def test_json_unknown_item_1016():
    inv = parse("invoice_1016.json")
    assert [li.sku for li in inv.line_items] == ["WidgetA", "WidgetB", None]
    assert inv.line_items[2].raw_name == "WidgetC"


def test_csv_key_value_with_repeated_keys_1006():
    inv = parse("invoice_1006.csv")
    assert inv.invoice_number == "INV-1006" and inv.vendor_name == "Acme Industrial Supplies"
    assert items(inv) == [("WidgetA", 5, Decimal("250.00")), ("WidgetB", 3, Decimal("500.00"))]
    assert inv.tax_amount == Decimal("0.00") and inv.total == Decimal("2750.00")
    assert inv.due_date == date(2026, 2, 10)


def test_csv_row_per_item_with_footer_1007():
    inv = parse("invoice_1007.csv")
    assert inv.vendor_name == "MegaWidgets Corp"
    assert inv.invoice_date == date(2026, 1, 28)  # 01/28/2026
    assert items(inv) == [
        ("WidgetA", 20, Decimal("250.00")),
        ("WidgetB", 15, Decimal("500.00")),
        ("GadgetX", 3, Decimal("750.00")),
    ]
    assert inv.line_items[0].amount == Decimal("5000.00")
    assert (inv.subtotal, inv.tax_rate, inv.tax_amount, inv.total) == (
        Decimal("14750.00"),
        Decimal("0.06"),
        Decimal("885.00"),
        Decimal("15525.00"),
    )


def test_csv_row_per_item_1015():
    inv = parse("invoice_1015.csv")
    assert len(inv.line_items) == 3 and inv.total == Decimal("6500.00")


def test_xml_nested_items_and_currency_1014():
    inv = parse("invoice_1014.xml")
    assert inv.invoice_number == "INV-1014" and inv.vendor_name == "TechParts International"
    assert inv.currency == "EUR"
    assert items(inv) == [("WidgetA", 4, Decimal("225.00")), ("WidgetB", 6, Decimal("475.00"))]
    assert (inv.subtotal, inv.tax_rate, inv.tax_amount, inv.total) == (
        Decimal("3750.00"),
        Decimal("0.10"),
        Decimal("375.00"),
        Decimal("4125.00"),
    )


@pytest.mark.parametrize(
    "parser, text",
    [
        (parse_json, "{not json"),
        (parse_json, '{"invoice_number": "X"}'),
        (parse_csv, "a,b\n1,2\n"),
        (parse_csv, "field,value\nquantity,5\n"),
        (parse_xml, "<invoice><broken></invoice>"),
        (parse_xml, "<invoice><header/></invoice>"),
    ],
)
def test_malformed_structured_input_raises_parse_error(parser, text):
    with pytest.raises(ParseError):
        parser(text)


def test_xml_entity_expansion_is_refused():
    bomb = '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><invoice>&a;</invoice>'
    with pytest.raises(ParseError):
        parse_xml(bomb)


def test_load_documents():
    text, fmt = load_document(DATA / "invoice_1008.txt")
    assert fmt == "email" and "SuperGizmo" in text
    text, fmt = load_document(DATA / "invoice_1001.txt")
    assert fmt == "txt"
    text, fmt = load_document(DATA / "invoice_1011.pdf")
    assert fmt == "pdf" and "INV-1011" in text and "Summit Manufacturing" in text
