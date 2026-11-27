from decimal import Decimal

import pytest

from src.db import connect, init_db
from src.payment import pay_invoice


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    init_db(c)
    for invoice_id in (1, 2):
        c.execute(
            "INSERT INTO invoices (id, fingerprint, status, data_json) "
            "VALUES (?, 'v|INV-1', 'VALIDATED', '{}')",
            (invoice_id,),
        )
    c.commit()
    yield c
    c.close()


class Bank:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, vendor, amount):
        self.calls.append((vendor, amount))
        if self.fail:
            raise ConnectionError("bank down")
        return {"status": "success"}


def pay(conn, bank, invoice_id=1, content="c1", amount="5000"):
    return pay_invoice(
        conn,
        invoice_id=invoice_id,
        fingerprint="v|INV-1",
        content_hash=content,
        vendor="Widgets Inc.",
        amount=Decimal(amount),
        pay_fn=bank,
    )


def test_pays_once_and_records_completion(conn):
    bank = Bank()
    result = pay(conn, bank)
    assert result.status == "paid" and result.settled
    assert bank.calls == [("Widgets Inc.", Decimal("5000.00"))]
    row = conn.execute("SELECT status, amount FROM payments").fetchone()
    assert (row["status"], row["amount"]) == ("COMPLETED", "5000.00")


def test_rerun_does_not_pay_twice(conn):
    bank = Bank()
    pay(conn, bank)
    again = pay(conn, bank)
    assert again.status == "already_paid" and again.settled
    assert "no double payment" in again.message
    assert len(bank.calls) == 1


def test_other_version_already_paid_is_blocked(conn):
    bank = Bank()
    pay(conn, bank)
    other = pay(conn, bank, invoice_id=2, content="c2", amount="5940")
    assert other.status == "blocked" and not other.settled
    assert len(bank.calls) == 1


def test_bank_failure_is_recorded_and_never_retried_blindly(conn):
    failed = pay(conn, Bank(fail=True))
    assert failed.status == "failed"
    assert conn.execute("SELECT status FROM payments").fetchone()["status"] == "FAILED"
    bank = Bank()
    retry = pay(conn, bank)
    assert retry.status == "blocked" and bank.calls == []  # reconcile first


def test_pending_intent_blocks_payment(conn):
    conn.execute(
        "INSERT INTO payments (idempotency_key, invoice_id, vendor, amount, status) "
        "VALUES ('v|INV-1|c1', 1, 'Widgets Inc.', '5000.00', 'PENDING')"
    )
    bank = Bank()
    assert pay(conn, bank).status == "blocked" and bank.calls == []


@pytest.mark.parametrize("amount", ["0", "-250"])
def test_nonpositive_amount_is_blocked(conn, amount):
    bank = Bank()
    assert pay(conn, bank, amount=amount).status == "blocked" and bank.calls == []
