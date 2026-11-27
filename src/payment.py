"""Idempotent payment around the spec's mock payment API.

A payment intent (PENDING, unique idempotency key) is written before money
moves and marked COMPLETED after. A rerun finds the completed payment and
skips it; an intent that never completed blocks retries until a human
reconciles it, because the money may or may not have moved.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal


def mock_payment(vendor, amount):
    """The spec's mock payment API, unchanged."""
    print(f"Paid {amount} to {vendor}")
    return {"status": "success"}


@dataclass
class PaymentResult:
    status: Literal["paid", "already_paid", "blocked", "failed"]
    message: str
    idempotency_key: str | None = None
    amount: Decimal | None = None

    @property
    def settled(self) -> bool:
        """True when the invoice is paid (now or earlier)."""
        return self.status in ("paid", "already_paid")


def pay_invoice(
    conn: sqlite3.Connection,
    *,
    invoice_id: int,
    fingerprint: str | None,
    content_hash: str,
    vendor: str | None,
    amount: Decimal | None,
    pay_fn: Callable = mock_payment,
) -> PaymentResult:
    if not fingerprint or not vendor or amount is None or amount <= 0:
        return PaymentResult("blocked", "Missing vendor, invoice number, or a positive amount.")
    amount = amount.quantize(Decimal("0.01"))
    key = f"{fingerprint}|{content_hash}"

    existing = conn.execute(
        "SELECT status, amount, completed_at, created_at FROM payments WHERE idempotency_key = ?",
        (key,),
    ).fetchone()
    if existing:
        if existing["status"] == "COMPLETED":
            return PaymentResult(
                "already_paid",
                f"Already paid ${Decimal(existing['amount']):,.2f} to {vendor} at "
                f"{existing['completed_at']}; no double payment.",
                key,
                amount,
            )
        return PaymentResult(
            "blocked",
            f"An earlier payment attempt ({existing['status']}, {existing['created_at']}) never "
            "completed; reconcile with the bank before retrying.",
            key,
            amount,
        )

    other = conn.execute(
        "SELECT p.amount, p.completed_at FROM payments p JOIN invoices i ON i.id = p.invoice_id "
        "WHERE i.fingerprint = ? AND p.status = 'COMPLETED'",
        (fingerprint,),
    ).fetchone()
    if other:
        return PaymentResult(
            "blocked",
            "A different version of this invoice was already paid "
            f"(${Decimal(other['amount']):,.2f} at {other['completed_at']}).",
            key,
            amount,
        )

    conn.execute(
        "INSERT INTO payments (idempotency_key, invoice_id, vendor, amount, status) "
        "VALUES (?, ?, ?, ?, 'PENDING')",
        (key, invoice_id, vendor, str(amount)),
    )
    conn.commit()  # the intent is durable before money moves
    try:
        response = pay_fn(vendor, amount)
        ok = isinstance(response, dict) and response.get("status") == "success"
        error = None if ok else f"payment API returned {response!r}"
    except Exception as e:  # the banking API failed
        ok, error = False, f"{type(e).__name__}: {e}"
    status = "COMPLETED" if ok else "FAILED"
    conn.execute(
        "UPDATE payments SET status = ?, completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
        "WHERE idempotency_key = ?",
        (status, key),
    )
    conn.commit()
    if ok:
        return PaymentResult("paid", f"Paid ${amount:,.2f} to {vendor.rstrip('.')}.", key, amount)
    return PaymentResult("failed", f"Payment failed: {error}", key, amount)
