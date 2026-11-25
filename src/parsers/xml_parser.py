from __future__ import annotations

from xml.etree.ElementTree import Element

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

from src.parsers.common import ParseError, money, text_or_none
from src.schemas import Invoice, LineItem


def _text(node: Element, path: str) -> str | None:
    return text_or_none(node.findtext(path))


def parse_xml(text: str, source_file: str | None = None) -> Invoice:
    try:
        root = ElementTree.fromstring(text)  # defusedxml: no entity expansion attacks
    except (ElementTree.ParseError, DefusedXmlException) as e:
        raise ParseError(f"invalid XML: {e}") from e

    items = root.findall(".//line_items/item")
    if not items:
        raise ParseError("no <line_items><item> elements")
    lines = [
        LineItem(
            raw_name=_text(item, "name") or _text(item, "description"),
            quantity=money(_text(item, "quantity")),
            unit_price=money(_text(item, "unit_price")),
            amount=money(_text(item, "amount")),
            note=_text(item, "note"),
        )
        for item in items
    ]
    vendor = root.find(".//vendor")
    vendor_name = None
    if vendor is not None:
        vendor_name = _text(vendor, "name") if len(vendor) else text_or_none(vendor.text)

    return Invoice(
        invoice_number=_text(root, ".//invoice_number"),
        vendor_name=vendor_name,
        vendor_address=_text(root, ".//vendor/address"),
        invoice_date_raw=_text(root, ".//header/date") or _text(root, ".//date"),
        due_date_raw=_text(root, ".//due_date"),
        currency=_text(root, ".//currency"),
        line_items=lines,
        subtotal=money(_text(root, ".//subtotal")),
        tax_rate=money(_text(root, ".//tax_rate")),
        tax_amount=money(_text(root, ".//tax_amount")),
        shipping=money(_text(root, ".//shipping")),
        total=money(_text(root, ".//totals/total")),
        payment_terms=_text(root, ".//payment_terms"),
        notes=_text(root, ".//notes"),
        source_file=source_file,
        source_format="xml",
    )
