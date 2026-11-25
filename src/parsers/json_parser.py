from __future__ import annotations

import json
from decimal import Decimal

from src.parsers.common import ParseError, money, text_or_none
from src.schemas import Invoice, LineItem


def parse_json(text: str, source_file: str | None = None) -> Invoice:
    try:
        data = json.loads(text, parse_float=Decimal, parse_int=Decimal)
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("line_items"), list):
        raise ParseError("expected an object with a line_items list")

    vendor = data.get("vendor")
    if isinstance(vendor, dict):
        vendor_name, vendor_address = vendor.get("name"), vendor.get("address")
    else:
        vendor_name, vendor_address = vendor, None

    lines = []
    for raw in data["line_items"]:
        if not isinstance(raw, dict):
            raise ParseError("line item is not an object")
        lines.append(
            LineItem(
                raw_name=text_or_none(raw.get("item") or raw.get("name") or raw.get("description")),
                quantity=money(raw.get("quantity")),
                unit_price=money(raw.get("unit_price")),
                amount=money(raw.get("amount")),
                note=text_or_none(raw.get("note")),
            )
        )

    return Invoice(
        invoice_number=text_or_none(data.get("invoice_number")),
        vendor_name=text_or_none(vendor_name),
        vendor_address=text_or_none(vendor_address),
        invoice_date_raw=text_or_none(data.get("date")),
        due_date_raw=text_or_none(data.get("due_date")),
        currency=text_or_none(data.get("currency")),
        line_items=lines,
        subtotal=money(data.get("subtotal")),
        tax_rate=money(data.get("tax_rate")),
        tax_amount=money(data.get("tax_amount")),
        shipping=money(data.get("shipping")),
        total=money(data.get("total")),
        payment_terms=text_or_none(data.get("payment_terms")),
        notes=text_or_none(data.get("notes")),
        revision=text_or_none(data.get("revision")),
        source_file=source_file,
        source_format="json",
    )
