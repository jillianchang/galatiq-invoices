import json
from types import SimpleNamespace

import pytest

from src.agents.investigator import InvestigatorAgent
from src.config import ROOT, load_policy
from src.db import connect, get_inventory, init_db
from src.ingestion import ingest
from src.rules import RuleContext
from src.rules.codes import finding
from src.schemas import Action, FindingSource
from src.tools.db_tools import InvestigatorTools
from tests.fakes import FakeOpenAI, completion, make_client

DATA = ROOT / "data" / "invoices"
POLICY = load_policy()


def call(name, args, id_="c1"):
    return SimpleNamespace(id=id_, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def report(findings, summary="ok"):
    return completion(
        None, tool_calls=[call("report_findings", {"findings": findings, "summary": summary})]
    )


@pytest.fixture
def setup(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_db(conn)
    inventory = get_inventory(conn)
    ingestion = ingest(DATA / "invoice_1005.json", catalog=inventory, agent=None)
    ctx = RuleContext(ingestion.invoice, ingestion.raw_text, inventory, POLICY)

    def agent(*responses, max_rounds=3):
        fake = FakeOpenAI(*responses)
        client, _, _ = make_client(tmp_path, fake)
        return InvestigatorAgent(
            client, InvestigatorTools(conn, inventory, POLICY), max_rounds
        ), fake

    yield ctx, agent
    conn.close()


ADDRESS = {
    "code": "suspicious_vendor_address",
    "message": "Vendor address is the White House.",
    "evidence_quote": "1600 Pennsylvania Ave, Washington, DC 20500",
}


def test_grounded_finding_is_added_with_registry_action(setup):
    ctx, agent = setup
    inv, fake = agent(report([ADDRESS]))
    out = inv.investigate(ctx, [])
    [f] = out.findings
    assert f.code == "suspicious_vendor_address" and f.source == FindingSource.LLM
    # Alone, an implausible address is flagged for the reviewer but doesn't block payment.
    assert f.action == Action.WARN and f.evidence["corroborated"] is False
    assert fake.requests[0]["model"] == "fast-m"
    assert "<document>" in fake.requests[0]["messages"][1]["content"]


def test_tool_call_round_trip(setup):
    ctx, agent = setup
    inv, fake = agent(
        completion(None, tool_calls=[call("check_stock", {"sku": "GadgetX", "quantity": 8})]),
        report([]),
    )
    out = inv.investigate(ctx, [])
    assert out.findings == [] and out.rounds == 2
    assert json.loads(out.tool_calls[0]["result"]) == {
        "sku": "GadgetX",
        "requested": 8,
        "stock": 5,
        "sufficient": False,
    }
    second = fake.requests[1]["messages"]
    assert second[-2]["tool_calls"][0]["function"]["name"] == "check_stock"
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "c1"


def test_ungrounded_and_repeated_findings_are_dropped(setup):
    ctx, agent = setup
    made_up = {"code": "suspected_fraud", "message": "x", "evidence_quote": "wire to Cayman"}
    repeat = {"code": "payment_pressure", "message": "x", "evidence_quote": "Net 60"}
    from src.rules.codes import finding

    existing = [finding("payment_pressure", "already raised")]
    inv, _ = agent(report([made_up, repeat]))
    out = inv.investigate(ctx, existing)
    assert out.findings == []
    assert [d["reason"] for d in out.dropped] == ["evidence not in document", "already raised"]


def test_last_round_forces_a_report(setup):
    ctx, agent = setup
    lookup = completion(None, tool_calls=[call("lookup_item", {"name": "WidgetA"})])
    inv, fake = agent(lookup, report([ADDRESS]), max_rounds=2)
    out = inv.investigate(ctx, [])
    assert out.findings[0].code == "suspicious_vendor_address"
    final = fake.requests[1]
    assert final["tool_choice"] == "required"
    assert [t["function"]["name"] for t in final["tools"]] == ["report_findings"]


def test_failure_adds_investigation_incomplete(setup):
    ctx, agent = setup
    import openai

    from tests.fakes import http_error

    inv, _ = agent(http_error(openai.BadRequestError, 400, body="bad messages"))
    out = inv.investigate(ctx, [])
    [f] = out.findings
    assert f.code == "investigation_incomplete" and f.action == Action.WARN


def test_tools_report_bad_arguments_instead_of_crashing(setup):
    ctx, agent = setup
    inv, _ = agent()
    assert "error" in json.loads(inv.tools.call("check_stock", '{"wrong": 1}'))
    assert "error" in json.loads(inv.tools.call("_private", "{}"))
    price = json.loads(
        inv.tools.call("compare_price", '{"sku": "WidgetB", "unit_price": 475, "currency": "EUR"}')
    )
    assert price["pct_vs_catalog"] == "+2.6%"


def test_rejected_tool_call_is_retried_with_the_error(setup):
    ctx, agent = setup
    import openai

    from tests.fakes import http_error

    rejected = http_error(
        openai.BadRequestError,
        400,
        body="Tool call validation failed: parameters for tool report_findings did not match",
    )
    inv, fake = agent(rejected, report([ADDRESS]))
    out = inv.investigate(ctx, [])
    assert [f.code for f in out.findings] == ["suspicious_vendor_address"]
    assert "rejected" in fake.requests[1]["messages"][-1]["content"]


def test_namespaced_tool_names_are_normalized(setup):
    ctx, agent = setup
    namespaced = completion(
        None,
        tool_calls=[call("functions.report_findings", {"findings": [ADDRESS], "summary": "s"})],
    )
    inv, _ = agent(namespaced)
    out = inv.investigate(ctx, [])
    assert [f.code for f in out.findings] == ["suspicious_vendor_address"] and out.rounds == 1


def test_report_tool_schema_is_flat():
    from src.agents.investigator import REPORT_TOOL

    assert "$defs" not in json.dumps(REPORT_TOOL)


def test_uncorroborated_fraud_suspicion_is_held_not_rejected(setup):
    ctx, agent = setup
    fraud = {**ADDRESS, "code": "suspected_fraud"}
    inv, _ = agent(report([fraud]))
    [f] = inv.investigate(ctx, []).findings
    assert f.code == "suspected_fraud" and f.action == Action.HOLD
    assert f.evidence["corroborated"] is False


def test_corroborated_fraud_suspicion_rejects(setup):
    ctx, agent = setup
    fraud = {**ADDRESS, "code": "suspected_fraud"}
    inv, _ = agent(report([fraud, ADDRESS]))  # address flag corroborates
    codes = {f.code: f.action for f in inv.investigate(ctx, []).findings}
    assert codes == {"suspected_fraud": Action.REJECT, "suspicious_vendor_address": Action.HOLD}


def test_corroborated_address_blocks(setup):
    ctx, agent = setup
    inv, _ = agent(report([ADDRESS]))
    pressure = finding("payment_pressure", "URGENT")
    [f] = inv.investigate(ctx, [pressure]).findings
    assert f.action == Action.HOLD
