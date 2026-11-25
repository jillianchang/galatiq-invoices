"""SQLite persistence in inventory.db (D8).

Reference data (`inventory`) is ground truth and read-only to the pipeline.
Operational tables are written as invoices are processed. Money is stored as
TEXT and read back as Decimal so no float rounding ever touches it.

Usage: python -m src.db [--reset] [--db PATH]
"""

from __future__ import annotations

import argparse
import sqlite3
from decimal import Decimal
from pathlib import Path

from src.config import ROOT

DEFAULT_DB_PATH = ROOT / "inventory.db"

# The spec's seed, plus catalog prices inferred from the clean invoices (D12).
# FakeItem has no basis for a price, so it is NULL and the price rule skips it.
INVENTORY_SEED: list[tuple[str, int, str | None]] = [
    ("WidgetA", 15, "250.00"),
    ("WidgetB", 10, "500.00"),
    ("GadgetX", 5, "750.00"),
    ("FakeItem", 0, None),
]

REFERENCE_DDL = """
CREATE TABLE IF NOT EXISTS inventory (
    item       TEXT PRIMARY KEY,
    stock      INTEGER NOT NULL,
    unit_price TEXT
);
"""

OPERATIONAL_DDL = """
CREATE TABLE IF NOT EXISTS submissions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL,
    source_path    TEXT NOT NULL,
    raw_hash       TEXT NOT NULL,
    fingerprint    TEXT,
    content_hash   TEXT,
    invoice_id     INTEGER REFERENCES invoices(id),
    gate           TEXT,
    outcome        TEXT,
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_submissions_raw_hash ON submissions(raw_hash);
CREATE INDEX IF NOT EXISTS idx_submissions_fingerprint ON submissions(fingerprint);

CREATE TABLE IF NOT EXISTS invoices (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint         TEXT NOT NULL,
    vendor_norm         TEXT,
    invoice_number_norm TEXT,
    revision            TEXT,
    content_hash        TEXT,
    total               TEXT,
    currency            TEXT,
    status              TEXT NOT NULL,
    superseded_by       INTEGER REFERENCES invoices(id),
    data_json           TEXT NOT NULL,
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_invoices_fingerprint ON invoices(fingerprint);

CREATE TABLE IF NOT EXISTS findings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id INTEGER NOT NULL REFERENCES submissions(id),
    code          TEXT NOT NULL,
    action        TEXT NOT NULL,
    message       TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    source        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id INTEGER NOT NULL REFERENCES submissions(id),
    gate          TEXT NOT NULL,
    decision      TEXT NOT NULL,
    reasoning     TEXT,
    critic_json   TEXT,
    approver      TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS payments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key TEXT NOT NULL UNIQUE,
    invoice_id      INTEGER NOT NULL REFERENCES invoices(id),
    vendor          TEXT NOT NULL,
    amount          TEXT NOT NULL,
    status          TEXT NOT NULL,  -- PENDING, COMPLETED, FAILED
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    completed_at    TEXT
);

-- Queryable audit trail: one compact row per pipeline step (node, event, key outputs).
-- Full LLM detail (prompts, tokens, retries) stays in traces/<run_id>.jsonl.
CREATE TABLE IF NOT EXISTS audit_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    submission_id INTEGER REFERENCES submissions(id),
    node          TEXT NOT NULL,
    event         TEXT NOT NULL,
    payload_json  TEXT NOT NULL DEFAULT '{}',
    ts            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_audit_events_submission ON audit_events(submission_id);

-- The spec's rejection log: every non-paid outcome with its reasoning and reason codes.
-- A view, not a table, so it can never disagree with the decisions it summarizes.
CREATE VIEW IF NOT EXISTS rejection_log AS
SELECT
    s.id          AS submission_id,
    s.run_id,
    s.source_path,
    s.gate,
    s.outcome,
    d.decision,
    d.reasoning,
    d.approver,
    (SELECT group_concat(f.code, ',') FROM findings f
      WHERE f.submission_id = s.id AND f.action <> 'INFO') AS reason_codes,
    s.created_at
FROM submissions s
LEFT JOIN decisions d ON d.submission_id = s.id
WHERE s.outcome IS NOT NULL AND s.outcome <> 'PAID';
"""

OPERATIONAL_TABLES = (
    "audit_events",
    "payments",
    "decisions",
    "findings",
    "submissions",
    "invoices",
)


def connect(path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables if missing and (re)seed inventory. Safe to run repeatedly."""
    conn.executescript(REFERENCE_DDL)
    conn.executescript(OPERATIONAL_DDL)
    conn.executemany(
        "INSERT OR REPLACE INTO inventory (item, stock, unit_price) VALUES (?, ?, ?)",
        INVENTORY_SEED,
    )
    conn.commit()


def reset_db(conn: sqlite3.Connection) -> None:
    """Clear all processing history and re-seed reference data."""
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("DROP VIEW IF EXISTS rejection_log")
    for table in OPERATIONAL_TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute("DROP TABLE IF EXISTS inventory")
    conn.execute("PRAGMA foreign_keys = ON")
    init_db(conn)


def get_inventory(conn: sqlite3.Connection) -> dict[str, tuple[int, Decimal | None]]:
    """item -> (stock, unit_price or None)."""
    rows = conn.execute("SELECT item, stock, unit_price FROM inventory").fetchall()
    return {
        r["item"]: (r["stock"], Decimal(r["unit_price"]) if r["unit_price"] is not None else None)
        for r in rows
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create and seed inventory.db")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--reset", action="store_true", help="clear history and re-seed")
    args = parser.parse_args(argv)
    with connect(args.db) as conn:
        if args.reset:
            reset_db(conn)
        else:
            init_db(conn)
        inventory = get_inventory(conn)
    action = "Reset" if args.reset else "Initialized"
    print(f"{action} {args.db}: {len(inventory)} inventory items")
    for item, (stock, price) in sorted(inventory.items()):
        print(f"  {item:<10} stock={stock:<4} unit_price={price if price is not None else 'NULL'}")


if __name__ == "__main__":
    main()
