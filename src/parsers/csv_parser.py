"""Two CSV layouts appear in the data:

- key/value (1006): a `field,value` header, invoice fields as rows, and
  line items as repeated `item` / `quantity` / `unit_price` rows.
- row per line item (1007, 1015): one row per item with the invoice fields
  repeated, and footer rows like `,,,,,,Subtotal:,14750.00`.
"""

from __future__ import annotations

import csv
import io

from src.parsers.common import ParseError, money, percent_from_label, text_or_none
from src.schemas import Invoice, LineItem

_HEADER_FIELDS = {
    "invoice_number": "invoice_number",
    "invoice number": "invoice_number",
    "vendor": "vendor_name",
    "date": "invoice_date_raw",
    "due_date": "due_date_raw",
    "due date": "due_date_raw",
    "currency": "currency",
    "payment_terms": "payment_terms",
    "payment terms": "payment_terms",
    "notes": "notes",
}
_MONEY_FIELDS = {
    "subtotal": "subtotal",
    "tax": "tax_amount",
    "tax_amount": "tax_amount",
    "tax_rate": "tax_rate",
    "shipping": "shipping",
    "total": "total",
}
_ITEM_NAME = {"item", "description", "name"}
_ITEM_QTY = {"quantity", "qty"}
_ITEM_PRICE = {"unit_price", "unit price", "price", "rate"}
_ITEM_AMOUNT = {"amount", "line_total", "line total"}


def parse_csv(text: str, source_file: str | None = None) -> Invoice:
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise ParseError("CSV has no data rows")
    header = [h.strip().lower() for h in rows[0]]
    if header == ["field", "value"]:
        fields = _parse_key_value(rows[1:])
    elif _ITEM_NAME & set(header):
        fields = _parse_row_per_item(header, rows[1:])
    else:
        raise ParseError(f"unrecognized CSV header: {rows[0]}")
    if not fields["line_items"]:
        raise ParseError("no line items found")
    return Invoice(**fields, source_file=source_file, source_format="csv")


def _parse_key_value(rows: list[list[str]]) -> dict:
    fields: dict = {"line_items": []}
    current: LineItem | None = None
    for row in rows:
        if len(row) < 2:
            raise ParseError(f"expected field,value row: {row}")
        key, value = row[0].strip().lower(), row[1]
        if key in _ITEM_NAME:
            current = LineItem(raw_name=text_or_none(value))
            fields["line_items"].append(current)
        elif key in _ITEM_QTY | _ITEM_PRICE | _ITEM_AMOUNT:
            if current is None:
                raise ParseError(f"'{key}' appears before any item")
            attr = (
                "quantity" if key in _ITEM_QTY else "unit_price" if key in _ITEM_PRICE else "amount"
            )
            setattr(current, attr, money(value))
        elif key in _HEADER_FIELDS:
            fields[_HEADER_FIELDS[key]] = text_or_none(value)
        elif key in _MONEY_FIELDS:
            fields[_MONEY_FIELDS[key]] = money(value)
    return fields


def _parse_row_per_item(header: list[str], rows: list[list[str]]) -> dict:
    col = {name: i for i, name in enumerate(header)}

    def cell(row: list[str], names: set[str]) -> str | None:
        for name in names:
            if name in col and col[name] < len(row):
                return text_or_none(row[col[name]])
        return None

    fields: dict = {"line_items": []}
    numbers: set[str] = set()
    for row in rows:
        name = cell(row, _ITEM_NAME)
        if name:
            for csv_name, attr in _HEADER_FIELDS.items():
                value = cell(row, {csv_name})
                if value and attr not in fields:
                    fields[attr] = value
            numbers.add(cell(row, {"invoice number", "invoice_number"}) or "")
            fields["line_items"].append(
                LineItem(
                    raw_name=name,
                    quantity=money(cell(row, _ITEM_QTY)),
                    unit_price=money(cell(row, _ITEM_PRICE)),
                    amount=money(cell(row, _ITEM_AMOUNT)),
                )
            )
            continue
        # Footer row: a "Label:" cell followed by its value.
        cells = [c.strip() for c in row]
        for i, label in enumerate(cells):
            if label.endswith(":") and i + 1 < len(cells):
                key = label[:-1].lower()
                value = money(cells[i + 1])
                if key.startswith("subtotal"):
                    fields["subtotal"] = value
                elif key.startswith("tax"):
                    fields["tax_amount"] = value
                    fields["tax_rate"] = percent_from_label(label)
                elif key.startswith("shipping"):
                    fields["shipping"] = value
                elif key.startswith("total"):
                    fields["total"] = value
                break
    if len(numbers - {""}) > 1:
        raise ParseError(f"multiple invoice numbers in one file: {sorted(numbers)}")
    return fields
