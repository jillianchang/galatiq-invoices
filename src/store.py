"""Persistence of processed invoices in inventory.db (D8, D9).

`register` links a submission to an invoice record right after validation, so
payment has an invoice id. `finalize` writes findings, the decision, and the
outcome. A rerun or duplicate links to the existing invoice record instead of
creating a new one; a revision of an unpaid invoice marks the original
SUPERSEDED.
"""

from __future__ import annotations

import json
import sqlite3

from src.history import content_hash, fingerprint
from src.normalize import normalize_vendor
from src.schemas import Finding, Invoice, Status


def start_submission(conn: sqlite3.Connection, run_id: str, source_path: str, raw_hash: str) -> int:
    cur = conn.execute(
        "INSERT INTO submissions (run_id, source_path, raw_hash) VALUES (?, ?, ?)",
        (run_id, source_path, raw_hash),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def invoice_keys(invoice: Invoice, raw_hash: str) -> tuple[str, str]:
    """(fingerprint, content hash). Unidentifiable invoices get a per-file fingerprint."""
    return fingerprint(invoice) or f"unidentified:{raw_hash[:16]}", content_hash(invoice)


def register(
    conn: sqlite3.Connection,
    submission_id: int,
    invoice: Invoice,
    raw_hash: str,
    findings: list[Finding],
) -> int:
    """Create or link the invoice record for this submission; returns its id."""
    fp, ch = invoice_keys(invoice, raw_hash)
    by_code = {f.code: f for f in findings}

    invoice_id: int | None = None
    if "reprocessed_submission" in by_code:
        row = conn.execute(
            "SELECT invoice_id FROM submissions WHERE raw_hash = ? AND outcome IS NOT NULL "
            "AND invoice_id IS NOT NULL ORDER BY id DESC LIMIT 1",
            (raw_hash,),
        ).fetchone()
        invoice_id = row["invoice_id"] if row else None
    elif "duplicate" in by_code:
        invoice_id = by_code["duplicate"].evidence.get("of_invoice_id")

    if invoice_id is None:
        cur = conn.execute(
            "INSERT INTO invoices (fingerprint, vendor_norm, invoice_number_norm, revision, "
            "content_hash, total, currency, status, data_json) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                fp,
                normalize_vendor(invoice.vendor_name),
                invoice.invoice_number,
                invoice.revision,
                ch,
                str(invoice.total) if invoice.total is not None else None,
                invoice.currency or "USD",
                Status.VALIDATED.value,
                invoice.model_dump_json(),
            ),
        )
        invoice_id = cur.lastrowid
        if "supersedes_unpaid_invoice" in by_code:
            old = by_code["supersedes_unpaid_invoice"].evidence.get("prior_invoice_id")
            conn.execute(
                "UPDATE invoices SET status = ?, superseded_by = ?, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                (Status.SUPERSEDED.value, invoice_id, old),
            )

    conn.execute(
        "UPDATE submissions SET invoice_id = ?, fingerprint = ?, content_hash = ? WHERE id = ?",
        (invoice_id, fp, ch, submission_id),
    )
    conn.commit()
    return invoice_id  # type: ignore[return-value]


def finalize(
    conn: sqlite3.Connection,
    *,
    submission_id: int,
    invoice_id: int,
    gate: str,
    outcome: Status,
    findings: list[Finding],
    decision: str,
    reasoning: str,
    approver: str,
    critic: list[dict] | None,
    update_invoice_status: bool,
) -> None:
    conn.executemany(
        "INSERT INTO findings (submission_id, code, action, message, evidence_json, source) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                submission_id,
                f.code,
                f.action.value,
                f.message,
                json.dumps(f.evidence, default=str),
                f.source.value,
            )
            for f in findings
        ],
    )
    conn.execute(
        "INSERT INTO decisions (submission_id, gate, decision, reasoning, critic_json, approver) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (submission_id, gate, decision, reasoning, json.dumps(critic or []), approver),
    )
    conn.execute(
        "UPDATE submissions SET gate = ?, outcome = ? WHERE id = ?",
        (gate, outcome.value, submission_id),
    )
    if update_invoice_status:
        # A paid invoice stays PAID: money moved, whatever a later rerun decides.
        conn.execute(
            "UPDATE invoices SET status = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            "WHERE id = ? AND status <> ?",
            (outcome.value, invoice_id, Status.PAID.value),
        )
    conn.commit()
