"""History checks against earlier runs (D9, PLAN.md section 11).

- Same raw file content as before: a rerun. Processed normally; the payment
  step is idempotent.
- Same invoice (vendor + number) with the same normalized content from a
  different file: a duplicate (e.g. 1011.pdf then 1011.txt).
- Same invoice with different content: a revision, handled by what happened
  to the original (paid -> hold with delta; returned -> expected fix;
  otherwise the revision supersedes it).

The fingerprint excludes the total, so a duplicate with an altered amount is
still caught. Content is compared on normalized fields, not raw bytes, so PDF
and TXT versions of one invoice match.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from decimal import Decimal

from src.normalize import normalize_vendor
from src.rules.codes import finding
from src.schemas import Finding, Invoice, Status

CENTS = Decimal("0.01")


def fingerprint(invoice: Invoice) -> str | None:
    vendor = normalize_vendor(invoice.vendor_name)
    if not vendor or not invoice.invoice_number:
        return None
    return f"{vendor}|{invoice.invoice_number}"


def _num(value: Decimal | None, money: bool = False) -> str | None:
    if value is None:
        return None
    return str(value.quantize(CENTS)) if money else format(value.normalize(), "f")


def content_hash(invoice: Invoice) -> str:
    lines = sorted(
        [
            li.sku or re.sub(r"[^a-z0-9]", "", (li.raw_name or "").lower()),
            _num(li.quantity),
            _num(li.unit_price, money=True),
        ]
        for li in invoice.line_items
    )
    canonical = {
        "vendor": normalize_vendor(invoice.vendor_name),
        "number": invoice.invoice_number,
        "total": _num(invoice.total, money=True),
        "lines": lines,
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def revision_number(revision: str | None) -> int:
    """None -> 0, 'R1' -> 1, 'Rev 2' -> 2, 'B' -> 2."""
    if not revision:
        return 0
    digits = re.findall(r"\d+", revision)
    if digits:
        return int(digits[-1])
    letters = re.findall(r"[A-Za-z]", revision.replace("Rev", "").replace("rev", ""))
    return ord(letters[-1].upper()) - ord("A") + 1 if letters else 0


def check_history(conn: sqlite3.Connection, invoice: Invoice, raw_hash: str) -> list[Finding]:
    rerun = conn.execute(
        "SELECT source_path, created_at, outcome FROM submissions "
        "WHERE raw_hash = ? AND outcome IS NOT NULL ORDER BY id DESC LIMIT 1",
        (raw_hash,),
    ).fetchone()
    if rerun:
        return [
            finding(
                "reprocessed_submission",
                f"This exact file was processed before ({rerun['source_path']} at "
                f"{rerun['created_at']}, outcome {rerun['outcome']}). It is processed again; "
                "the payment step will not pay twice.",
                dict(rerun),
            )
        ]

    fp = fingerprint(invoice)
    if fp is None:
        return []
    priors = conn.execute(
        """SELECT i.id, i.content_hash, i.status, i.total, i.revision, i.created_at,
                  (SELECT s.source_path FROM submissions s WHERE s.invoice_id = i.id
                   ORDER BY s.id LIMIT 1) AS source_path
           FROM invoices i
           WHERE i.fingerprint = ? AND i.status <> ?
           ORDER BY i.id DESC""",
        (fp, Status.SUPERSEDED.value),
    ).fetchall()
    if not priors:
        return []

    new_hash = content_hash(invoice)
    same = next((p for p in priors if p["content_hash"] == new_hash), None)
    if same:
        return [
            finding(
                "duplicate",
                f"Same invoice as {same['source_path']} (processed {same['created_at']}, "
                f"status {same['status']}). Payment is blocked to avoid paying twice.",
                {
                    "of_file": same["source_path"],
                    "of_invoice_id": same["id"],
                    "prior_status": same["status"],
                },
            )
        ]

    latest = priors[0]
    evidence = {
        "prior_file": latest["source_path"],
        "prior_invoice_id": latest["id"],
        "prior_status": latest["status"],
        "prior_total": latest["total"],
        "new_total": str(invoice.total) if invoice.total is not None else None,
    }
    if revision_number(invoice.revision) < revision_number(latest["revision"]):
        return [
            finding(
                "stale_revision",
                f"An older version of this invoice arrived after a newer one "
                f"({latest['source_path']}, revision {latest['revision']}).",
                evidence,
            )
        ]
    if latest["status"] == Status.PAID.value:
        delta = (invoice.total or Decimal(0)) - Decimal(latest["total"] or 0)
        return [
            finding(
                "revision_of_paid_invoice",
                f"Revision of an invoice already paid ({latest['source_path']}, "
                f"${Decimal(latest['total'] or 0):,.2f}). The new total is "
                f"${invoice.total or 0:,.2f}, a change of ${delta:,.2f}; only the difference "
                "could be owed.",
                {**evidence, "delta": str(delta)},
            )
        ]
    if latest["status"] == Status.RETURNED_TO_VENDOR.value:
        return [
            finding(
                "corrected_resubmission",
                f"Corrected version of {latest['source_path']}, which was returned to the vendor; "
                "processed as the replacement.",
                evidence,
            )
        ]
    return [
        finding(
            "supersedes_unpaid_invoice",
            f"Replaces the unpaid earlier version {latest['source_path']} "
            f"(status {latest['status']}), which will be marked superseded.",
            evidence,
        )
    ]
