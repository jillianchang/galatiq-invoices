"""End to end through the LangGraph pipeline, with a cooperative fake LLM."""

import json
import re
import shutil
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.agents.approval import ApprovalWorkflow
from src.agents.investigator import InvestigatorAgent
from src.config import ROOT, load_policy
from src.db import connect, get_inventory, init_db
from src.observability import RunLogger
from src.pipeline import Services, build_graph, process_file
from src.schemas import Status
from src.tools.db_tools import InvestigatorTools
from tests.fakes import FakeOpenAI, completion, make_client

DATA = ROOT / "data" / "invoices"
POLICY = load_policy()


def cooperative(request):
    """Investigator: nothing new. Approver: first allowed decision. Critic: agree."""
    system = request["messages"][0]["content"]
    if request.get("tools"):
        args = json.dumps({"findings": [], "summary": "Nothing new."})
        call = SimpleNamespace(
            id="r", function=SimpleNamespace(name="report_findings", arguments=args)
        )
        return completion(None, tool_calls=[call])
    if system.startswith("You are an internal auditor"):
        return completion(json.dumps({"verdict": "agree", "objections": [], "summary": "ok"}))
    allowed = re.search(r"Allowed decisions: ([A-Z_, ]+)\.", request["messages"][1]["content"])
    decision = allowed.group(1).split(", ")[0]
    return completion(
        json.dumps(
            {
                "decision": decision,
                "reasoning": f"{decision} per findings.",
                "recommended_action": "next step",
                "vendor_message": "Please correct the total."
                if decision == "RETURN_TO_VENDOR"
                else None,
            }
        )
    )


class Bank:
    def __init__(self):
        self.calls = []

    def __call__(self, vendor, amount):
        self.calls.append((vendor, amount))
        return {"status": "success"}


@pytest.fixture
def pipeline(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_db(conn)
    inventory = get_inventory(conn)
    logger = RunLogger("test-run", conn, trace_dir=tmp_path / "traces", stream=None)
    client, _, _ = make_client(tmp_path, FakeOpenAI(responder=cooperative), mode="live")
    client._on_event = logger.llm_event  # as main.py wires it
    bank = Bank()
    svc = Services(
        conn=conn,
        policy=POLICY,
        inventory=inventory,
        logger=logger,
        approval=ApprovalWorkflow(client, POLICY),
        investigator=InvestigatorAgent(client, InvestigatorTools(conn, inventory, POLICY)),
        pay_fn=bank,
    )
    graph = build_graph(svc)

    def run(path):
        return process_file(svc, graph, path)

    yield SimpleNamespace(run=run, conn=conn, bank=bank, tmp=tmp_path, logger=logger)
    logger.close()
    conn.close()


def test_clean_invoice_is_paid_once_even_on_rerun(pipeline):
    first = pipeline.run(DATA / "invoice_1004.json")
    assert first.outcome == Status.PAID and first.payment.status == "paid"
    assert pipeline.bank.calls == [("Precision Parts Ltd.", Decimal("1890.00"))]

    rerun = pipeline.run(DATA / "invoice_1004.json")
    assert rerun.outcome == Status.PAID and rerun.payment.status == "already_paid"
    assert "reprocessed_submission" in {f.code for f in rerun.findings}
    assert len(pipeline.bank.calls) == 1
    assert rerun.invoice_id == first.invoice_id  # linked, not a new record


def test_revision_of_paid_invoice_is_held_with_delta(pipeline):
    pipeline.run(DATA / "invoice_1004.json")
    revised = pipeline.run(DATA / "invoice_1004_revised.json")
    assert revised.outcome == Status.ON_HOLD
    f = next(f for f in revised.findings if f.code == "revision_of_paid_invoice")
    assert f.evidence["delta"] == "4050.00"
    assert len(pipeline.bank.calls) == 1


def test_revision_of_unpaid_invoice_supersedes_it(pipeline):
    held = pipeline.run(DATA / "invoice_1005.json")  # stock exceeded: ON_HOLD
    data = json.loads((DATA / "invoice_1005.json").read_text())
    data["revision"] = "R1"
    data["line_items"][1]["quantity"] = 5  # GadgetX now within stock
    data["subtotal"], data["tax_amount"], data["total"] = 12250.00, 612.50, 12862.50
    revised_path = pipeline.tmp / "invoice_1005_revised.json"
    revised_path.write_text(json.dumps(data))
    revised = pipeline.run(revised_path)
    assert "supersedes_unpaid_invoice" in {f.code for f in revised.findings}
    status = pipeline.conn.execute(
        "SELECT status, superseded_by FROM invoices WHERE id = ?", (held.invoice_id,)
    ).fetchone()
    assert (status["status"], status["superseded_by"]) == ("SUPERSEDED", revised.invoice_id)


def test_duplicate_from_another_file_is_blocked_without_approval(pipeline):
    pipeline.run(DATA / "invoice_1004.json")
    copy = pipeline.tmp / "invoice_1004_copy.json"
    copy.write_text(json.dumps(json.loads((DATA / "invoice_1004.json").read_text()), indent=1))
    dup = pipeline.run(copy)
    assert dup.outcome == Status.DUPLICATE_BLOCKED and dup.approval is None
    assert dup.underlying_gate.value == "ELIGIBLE"
    assert "would have been ELIGIBLE" in dup.reasoning
    assert len(pipeline.bank.calls) == 1


@pytest.mark.parametrize(
    "name, outcome",
    [
        ("invoice_1007.csv", Status.RETURNED_TO_VENDOR),
        ("invoice_1009.json", Status.REJECTED),
        ("invoice_1014.xml", Status.ON_HOLD),
        ("invoice_1016.json", Status.ON_HOLD),
        ("invoice_2005_clean_large.json", Status.PAID),
    ],
)
def test_outcomes(pipeline, name, outcome):
    result = pipeline.run(DATA / name)
    assert result.outcome == outcome
    if outcome != Status.PAID:
        assert pipeline.bank.calls == []


def test_over_threshold_clean_invoice_goes_through_the_critic(pipeline):
    result = pipeline.run(DATA / "invoice_2005_clean_large.json")
    assert result.outcome == Status.PAID
    assert [c["verdict"] for c in result.approval.critic_rounds] == ["agree"]


def test_rejection_log_and_audit_trail(pipeline):
    pipeline.run(DATA / "invoice_1009.json")
    row = pipeline.conn.execute("SELECT * FROM rejection_log").fetchone()
    assert row["outcome"] == "REJECTED" and "nonpositive_quantity" in row["reason_codes"]
    assert row["reasoning"]
    nodes = [r["node"] for r in pipeline.conn.execute("SELECT node FROM audit_events ORDER BY id")]
    assert nodes == ["ingestion", "validation", "gate", "approval", "rejection_log", "finalize"]
    trace = pipeline.logger.trace_path.read_text().splitlines()
    assert any('"stage": "llm"' in line for line in trace)


def test_cli_runs_end_to_end_offline(tmp_path, capsys, monkeypatch):
    import main

    monkeypatch.chdir(tmp_path)
    src = DATA / "invoice_1006.csv"
    shutil.copy(src, tmp_path / src.name)
    code = main.main(
        [
            f"--invoice_path={tmp_path / src.name}",
            "--db",
            str(tmp_path / "cli.db"),
            "--llm",
            "replay",
            "--json",
            "--reset",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    # No LLM in replay with an empty cache: fails closed to a human, never pays.
    assert out["outcome"] == "ON_HOLD" and out["gate"] == "ELIGIBLE"
    assert out["decision"]["approver"] == "policy"
