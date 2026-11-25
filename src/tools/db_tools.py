"""Read-only tools the investigator agent can call. None of them change state."""

from __future__ import annotations

import json
import sqlite3
from decimal import Decimal
from typing import Any

from src.config import Policy
from src.normalize import match_item, normalize_vendor
from src.rules.context import Inventory

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_item",
            "description": "Look up an item name in the inventory catalog. Returns the "
            "matching SKU (exact match after case/spacing normalization), stock, and price.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Check whether a quantity of a catalog SKU is available.",
            "parameters": {
                "type": "object",
                "properties": {"sku": {"type": "string"}, "quantity": {"type": "number"}},
                "required": ["sku", "quantity"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_price",
            "description": "Compare a unit price with the catalog price for a SKU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string"},
                    "unit_price": {"type": "number"},
                    "currency": {"type": "string"},
                },
                "required": ["sku", "unit_price", "currency"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "invoice_history",
            "description": "List invoices previously processed from a vendor.",
            "parameters": {
                "type": "object",
                "properties": {"vendor_name": {"type": "string"}},
                "required": ["vendor_name"],
                "additionalProperties": False,
            },
        },
    },
]


class InvestigatorTools:
    def __init__(self, conn: sqlite3.Connection, inventory: Inventory, policy: Policy):
        self.conn = conn
        self.inventory = inventory
        self.policy = policy

    def call(self, name: str, arguments: str) -> str:
        """Run a tool by name with JSON arguments; always returns a JSON string."""
        try:
            args = json.loads(arguments or "{}")
            handler = getattr(self, f"_{name}", None)
            if handler is None or name.startswith("_"):
                return json.dumps({"error": f"unknown tool {name!r}"})
            return json.dumps(handler(**args), default=str)
        except Exception as e:  # bad arguments from the model: report, don't crash
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    def _lookup_item(self, name: str) -> dict[str, Any]:
        sku, note = match_item(name, self.inventory)
        if sku is None:
            return {"name": name, "in_catalog": False, "catalog": sorted(self.inventory)}
        stock, price = self.inventory[sku]
        return {
            "name": name,
            "in_catalog": True,
            "sku": sku,
            "stock": stock,
            "unit_price": price,
            "note": note,
        }

    def _check_stock(self, sku: str, quantity: float) -> dict[str, Any]:
        if sku not in self.inventory:
            return {"sku": sku, "in_catalog": False}
        stock = self.inventory[sku][0]
        return {"sku": sku, "requested": quantity, "stock": stock, "sufficient": quantity <= stock}

    def _compare_price(self, sku: str, unit_price: float, currency: str = "USD") -> dict[str, Any]:
        if sku not in self.inventory:
            return {"sku": sku, "in_catalog": False}
        catalog = self.inventory[sku][1]
        rate = (
            Decimal(1)
            if currency.upper() == "USD"
            else self.policy.fx_rates_to_usd.get(currency.upper())
        )
        if catalog is None or rate is None:
            return {"sku": sku, "catalog_price": catalog, "comparable": False}
        usd = Decimal(str(unit_price)) * rate
        return {
            "sku": sku,
            "unit_price_usd": f"{usd:.2f}",
            "catalog_price": str(catalog),
            "pct_vs_catalog": f"{(usd - catalog) / catalog:+.1%}",
        }

    def _invoice_history(self, vendor_name: str) -> dict[str, Any]:
        rows = self.conn.execute(
            "SELECT invoice_number_norm AS invoice_number, total, currency, status, created_at "
            "FROM invoices WHERE vendor_norm = ? ORDER BY id DESC LIMIT 10",
            (normalize_vendor(vendor_name),),
        ).fetchall()
        return {"vendor": vendor_name, "prior_invoices": [dict(r) for r in rows]}
