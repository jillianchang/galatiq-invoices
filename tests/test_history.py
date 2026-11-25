import json

import pytest

from src.config import ROOT
from src.db import connect, init_db
from src.history import check_history, content_hash, fingerprint, revision_number
from src.ingestion import ingest
from src.schemas import Action, Status

DATA = ROOT / "data" / "invoices"
CATALOG = ["WidgetA", "WidgetB", "GadgetX", "FakeItem"]


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    init_db(c)
    yield c
    c.close()


def record(conn, result, status, outcome=None):
    """Stand-in for the Phase 3 persistence: store a processed submission."""
    inv = result.invoice
    cur = conn.execute(
        "INSERT INTO invoices (fingerprint, vendor_norm, invoice_number_norm, revision, "
        "content_hash, total, currency, status, data_json) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            fingerprint(inv),
            None,
            inv.invoice_number,
            inv.revision,
            content_hash(inv),
            str(inv.total),
            inv.currency,
            status,
            inv.model_dump_json(),
        ),
    )
    conn.execute(
        "INSERT INTO submissions (run_id, source_path, raw_hash, fingerprint, content_hash, "
        "invoice_id, outcome) VALUES ('r', ?, ?, ?, ?, ?, ?)",
        (
            inv.source_file,
            result.raw_hash,
            fingerprint(inv),
            content_hash(inv),
            cur.lastrowid,
            outcome or status,
        ),
    )
    conn.commit()


def load(name):
    return ingest(DATA / name, catalog=CATALOG, agent=None)


def test_first_sighting_has_no_history(conn):
    assert check_history(conn, load("invoice_1004.json").invoice, "h") == []


def test_rerun_of_same_file(conn):
    r = load("invoice_1004.json")
    record(conn, r, Status.PAID.value)
    [f] = check_history(conn, r.invoice, r.raw_hash)
    assert f.code == "reprocessed_submission" and f.action == Action.INFO


def test_same_invoice_from_another_file_is_a_duplicate(conn, tmp_path):
    r = load("invoice_1004.json")
    record(conn, r, Status.ON_HOLD.value)
    # Same invoice, different bytes (reformatted): still a duplicate.
    copy = tmp_path / "invoice_1004_copy.json"
    copy.write_text(json.dumps(json.loads((DATA / "invoice_1004.json").read_text()), indent=4))
    r2 = ingest(copy, catalog=CATALOG, agent=None)
    assert r2.raw_hash != r.raw_hash
    [f] = check_history(conn, r2.invoice, r2.raw_hash)
    assert f.code == "duplicate" and f.action == Action.DUPLICATE
    assert f.evidence["of_file"] == "invoice_1004.json"


def test_altered_amount_is_not_a_silent_duplicate(conn, tmp_path):
    r = load("invoice_1004.json")
    record(conn, r, Status.PAID.value)
    data = json.loads((DATA / "invoice_1004.json").read_text())
    data["total"] = 2890.00
    altered = tmp_path / "invoice_1004_altered.json"
    altered.write_text(json.dumps(data))
    r2 = ingest(altered, catalog=CATALOG, agent=None)
    [f] = check_history(conn, r2.invoice, r2.raw_hash)
    assert f.code == "revision_of_paid_invoice"  # same fingerprint, held with the delta


def test_revision_of_paid_invoice_holds_with_delta(conn):
    record(conn, load("invoice_1004.json"), Status.PAID.value)
    r = load("invoice_1004_revised.json")
    [f] = check_history(conn, r.invoice, r.raw_hash)
    assert f.code == "revision_of_paid_invoice" and f.action == Action.HOLD
    assert f.evidence["delta"] == "4050.00"


def test_revision_of_unpaid_invoice_supersedes(conn):
    record(conn, load("invoice_1004.json"), Status.ON_HOLD.value)
    r = load("invoice_1004_revised.json")
    [f] = check_history(conn, r.invoice, r.raw_hash)
    assert f.code == "supersedes_unpaid_invoice" and f.action == Action.INFO


def test_revision_of_returned_invoice_is_the_expected_fix(conn):
    record(conn, load("invoice_1004.json"), Status.RETURNED_TO_VENDOR.value)
    r = load("invoice_1004_revised.json")
    [f] = check_history(conn, r.invoice, r.raw_hash)
    assert f.code == "corrected_resubmission"


def test_stale_revision(conn):
    record(conn, load("invoice_1004_revised.json"), Status.ON_HOLD.value)
    r = load("invoice_1004.json")
    [f] = check_history(conn, r.invoice, r.raw_hash)
    assert f.code == "stale_revision" and f.action == Action.HOLD


def test_superseded_records_are_ignored(conn):
    record(conn, load("invoice_1004.json"), Status.SUPERSEDED.value)
    assert check_history(conn, load("invoice_1004_revised.json").invoice, "new") == []


def test_content_hash_ignores_formatting_but_not_content():
    a, b = load("invoice_1004.json").invoice, load("invoice_1004_revised.json").invoice
    assert content_hash(a) != content_hash(b)
    assert content_hash(a) == content_hash(a.model_copy(update={"subtotal": None, "notes": "x"}))


def test_fingerprint_needs_vendor_and_number():
    inv = load("invoice_1009.json").invoice  # vendor is blank
    assert fingerprint(inv) is None


@pytest.mark.parametrize("raw, n", [(None, 0), ("R1", 1), ("Rev 2", 2), ("B", 2)])
def test_revision_number(raw, n):
    assert revision_number(raw) == n
