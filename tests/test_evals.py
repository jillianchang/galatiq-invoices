"""The eval harness, run offline: a perfect oracle LLM, an empty replay cache, and scoring."""

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from evals import run as evals
from evals.truth import INVOICE_DIR, expectation, expected_totals, load_expected
from src.config import load_policy
from src.pipeline import list_invoices
from src.schemas import Gate, Status
from tests.fakes import FakeOpenAI, make_client
from tests.oracle import Oracle

POLICY = load_policy()
FILES = list_invoices(INVOICE_DIR)


def run(tmp_path, scenario, files=FILES, *, responder=None, mode="live"):
    router = evals.EventRouter()
    client, _, _ = make_client(tmp_path, FakeOpenAI(responder=responder), mode=mode)
    client._on_event = router
    return evals.run_scenario(
        scenario,
        files,
        client=client,
        router=router,
        policy=POLICY,
        out_dir=tmp_path / "out",
        eval_id="test",
        pay_fn=lambda vendor, amount: {"status": "success"},
    )


def test_batch_totals_follow_from_the_expectations():
    names = [p.name for p in FILES]
    assert expected_totals("batch", names) == {
        "paid": 7,  # PLAN.md's six, plus invoice_2005_clean_large.json
        "paid_amount": "38825.00",
        "duplicates_blocked": 3,
        "revisions_held": 1,
    }
    standalone = expected_totals("standalone", names)
    assert standalone["duplicates_blocked"] == 0 and standalone["revisions_held"] == 0


def test_batch_overrides_drop_the_paid_amount():
    exp = expectation("invoice_1011.txt", "batch")
    assert exp["outcome"] == "DUPLICATE_BLOCKED" and "amount" not in exp
    assert expectation("invoice_1011.txt", "standalone")["amount"] == "3000.00"


@pytest.mark.parametrize("scenario", ["standalone", "batch"])
def test_oracle_passes_every_deterministic_expectation(tmp_path, scenario):
    oracle = Oracle()
    report = run(tmp_path, scenario, responder=oracle)
    failures = {
        i.file: [(c.name, c.expected, c.actual) for c in i.checks if c.hard and not c.ok]
        for i in report.invoices
        if not i.passed
    }
    assert failures == {}
    assert report.totals_ok, (report.totals_expected, report.totals_actual)
    m = report.metrics
    assert m["gate_accuracy"] == m["required_finding_recall"] == m["outcome_accuracy"] == 1.0
    assert m["extraction_field_accuracy"] == 1.0
    assert m["safety_violations"] == 0
    assert m["payment_calls"] == report.totals_actual["paid"]
    # The oracle's investigator reports nothing, so agent-judgment findings are soft misses.
    assert m["llm_finding_recall"] == 0.0
    assert {"extraction", "investigator", "approval", "critic"} <= set(oracle.roles)


def test_empty_replay_cache_fails_closed_and_is_reported(tmp_path):
    files = [INVOICE_DIR / n for n in ("invoice_1001.txt", "invoice_1004.json")]
    report = run(tmp_path, "standalone", files, mode="replay")
    by_file = {i.file: i for i in report.invoices}
    assert all(i.cache_miss for i in report.invoices)
    assert by_file["invoice_1001.txt"].outcome == "ON_HOLD"  # extraction failed closed
    assert by_file["invoice_1001.txt"].extraction["correct"] == 0
    assert by_file["invoice_1004.json"].gate == "ELIGIBLE"  # parser needs no LLM
    assert by_file["invoice_1004.json"].outcome == "ON_HOLD"  # approval failed closed
    assert report.payment_calls == 0 and not report.passed
    assert report.metrics["safety_violations"] == 0


def _result(gate, outcome, codes, paid=None):
    payment = SimpleNamespace(amount=Decimal(paid), settled=True, status="paid") if paid else None
    return SimpleNamespace(
        gate=Gate[gate],
        outcome=Status[outcome],
        findings=[SimpleNamespace(code=c) for c in codes],
        payment=payment,
    )


def test_score_hard_and_soft_checks():
    exp = expectation("invoice_1005.json", "standalone")
    checks = {
        c.name: c
        for c in evals.score(exp, _result("HOLD", "ON_HOLD", ["stock_exceeded", "over_threshold"]))
    }
    assert checks["gate"].ok and checks["required_findings"].ok and checks["must_not_pay"].ok
    assert not checks["llm_findings"].ok and not checks["llm_findings"].hard

    paid_anyway = evals.score(exp, _result("ELIGIBLE", "PAID", [], paid="12862.50"))
    failed = {c.name for c in paid_anyway if c.hard and not c.ok}
    assert failed == {"gate", "required_findings", "outcome", "must_not_pay"}


def test_score_wrong_amount_and_judgment_outcome():
    exp = expectation("invoice_1010.txt", "standalone")  # outcome_llm: true
    held = {
        c.name: c for c in evals.score(exp, _result("ELIGIBLE", "ON_HOLD", ["price_over_catalog"]))
    }
    assert not held["outcome"].ok and not held["outcome"].hard  # agent judgment: soft
    wrong = {
        c.name: c
        for c in evals.score(exp, _result("ELIGIBLE", "PAID", ["price_over_catalog"], "7000"))
    }
    assert not wrong["amount"].ok and wrong["amount"].hard


def test_cli_writes_report_and_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path / "empty-cache"))
    code = evals.main(
        [
            "--scenario",
            "standalone",
            "--files",
            "invoice_1009.json",
            "--llm",
            "replay",
            "--out",
            str(tmp_path / "res"),
            "--json",
        ]
    )
    report = json.loads((tmp_path / "res" / "report.json").read_text())
    assert report == json.loads(capsys.readouterr().out)
    inv = report["scenarios"]["standalone"]["invoices"][0]
    # 1009 is rejected by deterministic rules; the fallback approver still rejects it.
    assert inv["outcome"] == "REJECTED" and inv["passed"] and code == 0
    assert evals.main(["--files", "invoice_9999.txt"]) == 2


def test_every_expected_file_is_scored():
    assert {p.name for p in FILES} == set(load_expected())


def test_soft_outcome_misses_dont_fail_the_totals():
    report = evals.ScenarioReport(
        "batch",
        [],
        {"paid": 7, "paid_amount": "38825.00", "duplicates_blocked": 3, "revisions_held": 1},
        {"paid": 6, "paid_amount": "26325.00", "duplicates_blocked": 3, "revisions_held": 1},
        payment_calls=6,
    )
    assert report.totals_ok
    report.payment_calls = 7  # a payment call that didn't produce a recorded payment
    assert not report.totals_ok
    report.payment_calls = 6
    report.totals_actual["duplicates_blocked"] = 2
    assert not report.totals_ok
