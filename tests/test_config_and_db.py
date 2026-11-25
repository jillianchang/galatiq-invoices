from decimal import Decimal

import pytest

from src.config import Settings, load_policy
from src.db import connect, get_inventory, init_db, reset_db


def test_policy_loads_with_exact_decimals():
    policy = load_policy()
    assert policy.approval.threshold_usd == Decimal("10000")
    assert policy.approval.near_threshold_pct == Decimal("0.03")
    assert policy.fx_rates_to_usd["EUR"] == Decimal("1.08")
    assert policy.money_tolerance_usd == Decimal("0.01")
    assert set(policy.llm.roles) == {"extraction", "investigator", "approval", "critic"}
    assert policy.llm.roles["extraction"].model == "fast"
    assert policy.llm.roles["critic"].model == "strong"


def test_settings_default_mode_depends_on_key(monkeypatch):
    for var in ("LLM_API_KEY", "LLM_MODE", "LLM_MODEL", "LLM_MODEL_FAST", "LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    assert Settings.from_env(env_file=None).mode == "replay"
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "big")
    settings = Settings.from_env(env_file=None)
    assert settings.mode == "record"
    assert settings.model_fast == "big"  # fast tier defaults to the strong model


def test_settings_rejects_bad_mode(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "yolo")
    with pytest.raises(ValueError):
        Settings.from_env(env_file=None)


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "test.db")
    init_db(c)
    yield c
    c.close()


def test_inventory_seed(conn):
    inv = get_inventory(conn)
    assert inv["WidgetA"] == (15, Decimal("250.00"))
    assert inv["WidgetB"] == (10, Decimal("500.00"))
    assert inv["GadgetX"] == (5, Decimal("750.00"))
    assert inv["FakeItem"] == (0, None)  # no basis for a price (D12)
    assert "WidgetC" not in inv  # keep the spec's unknown-item scenarios intact


def test_init_is_idempotent_and_reset_clears_history(conn):
    init_db(conn)
    assert len(get_inventory(conn)) == 4
    conn.execute(
        "INSERT INTO submissions (run_id, source_path, raw_hash) VALUES ('r1', 'x.txt', 'h')"
    )
    conn.execute(
        "INSERT INTO audit_events (run_id, submission_id, node, event) "
        "VALUES ('r1', 1, 'ingestion', 'parsed')"
    )
    conn.commit()
    init_db(conn)  # re-seeding keeps history
    assert conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0] == 1
    reset_db(conn)
    assert conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0] == 0
    assert len(get_inventory(conn)) == 4


def test_payment_idempotency_key_is_unique(conn):
    conn.execute(
        "INSERT INTO invoices (fingerprint, status, data_json) VALUES ('f', 'VALIDATED', '{}')"
    )
    sql = (
        "INSERT INTO payments (idempotency_key, invoice_id, vendor, amount, status) "
        "VALUES ('k1', 1, 'V', '10.00', 'PENDING')"
    )
    conn.execute(sql)
    with pytest.raises(Exception, match="UNIQUE"):
        conn.execute(sql)


def test_rejection_log_view_lists_non_paid_outcomes_with_reasoning(conn):
    conn.executescript("""
        INSERT INTO submissions (id, run_id, source_path, raw_hash, gate, outcome)
        VALUES (1, 'r1', 'invoice_1003.txt', 'h1', 'REJECT', 'REJECTED'),
               (2, 'r1', 'invoice_1001.txt', 'h2', 'ELIGIBLE', 'PAID'),
               (3, 'r1', 'invoice_1002.txt', 'h3', NULL, NULL);
        INSERT INTO decisions (submission_id, gate, decision, reasoning, approver)
        VALUES (1, 'REJECT', 'REJECT', 'Fake item, payment pressure.', 'agent');
        INSERT INTO findings (submission_id, code, action, message, source)
        VALUES (1, 'fraud_pattern', 'REJECT', 'm', 'rule'),
               (1, 'payment_pressure', 'HOLD', 'm', 'rule'),
               (1, 'reprocessed_submission', 'INFO', 'm', 'rule');
    """)
    rows = conn.execute("SELECT * FROM rejection_log").fetchall()
    assert len(rows) == 1  # PAID and unfinished submissions are excluded
    row = rows[0]
    assert row["source_path"] == "invoice_1003.txt"
    assert row["outcome"] == "REJECTED"
    assert row["reasoning"] == "Fake item, payment pressure."
    assert set(row["reason_codes"].split(",")) == {"fraud_pattern", "payment_pressure"}
