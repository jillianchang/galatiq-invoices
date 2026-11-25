"""Extraction agent self-correction loop, with a scripted fake LLM."""

import json

from src.agents.extraction import ExtractionAgent
from src.config import ROOT
from tests.fakes import FakeOpenAI, completion, make_client

DATA = ROOT / "data" / "invoices"
SOURCE_1001 = (DATA / "invoice_1001.txt").read_text()

GOOD_1001 = {
    "invoice_number": "INV-1001",
    "vendor_name": "Widgets Inc.",
    "vendor_address": None,
    "invoice_date_raw": "2026-01-15",
    "due_date_raw": "2026-02-01",
    "currency": "USD",
    "line_items": [
        {"name": "WidgetA", "quantity": 10, "unit_price": 250.0, "amount": None},
        {"name": "WidgetB", "quantity": 5, "unit_price": 500.0, "amount": None},
    ],
    "subtotal": 5000.0,
    "tax_rate_percent": 0,
    "tax_amount": 0.0,
    "shipping": None,
    "total": 5000.0,
    "payment_terms": "Net 15",
    "notes": None,
}


def with_changes(**changes):
    return json.dumps({**GOOD_1001, **changes})


def agent_with(tmp_path, *responses, max_attempts=3):
    fake = FakeOpenAI(*[completion(r) for r in responses])
    client, _, _ = make_client(tmp_path, fake)
    return ExtractionAgent(client, max_attempts), fake


def test_first_try_success(tmp_path):
    agent, fake = agent_with(tmp_path, json.dumps(GOOD_1001))
    outcome = agent.extract(SOURCE_1001)
    assert outcome.ok and outcome.attempts == 1
    assert outcome.extracted.vendor_name == "Widgets Inc."
    doc = fake.requests[0]["messages"][1]["content"]
    assert doc.startswith("<document>") and "Widgets Inc." in doc  # wrapped as untrusted data
    assert fake.requests[0]["model"] == "fast-m"  # extraction runs on the fast tier


def test_grounding_error_is_fed_back_and_corrected(tmp_path):
    agent, fake = agent_with(tmp_path, with_changes(total=5500.0), json.dumps(GOOD_1001))
    outcome = agent.extract(SOURCE_1001)
    assert outcome.ok and outcome.attempts == 2
    retry = fake.requests[1]["messages"]
    assert retry[-2]["role"] == "assistant"
    assert "total=5500.0 does not appear" in retry[-1]["content"]
    assert outcome.history[0]["errors"] and not outcome.history[1]["errors"]


def test_fails_closed_when_grounding_never_passes(tmp_path):
    agent, fake = agent_with(tmp_path, *[with_changes(vendor_name="Made Up LLC")] * 3)
    outcome = agent.extract(SOURCE_1001)
    assert not outcome.ok and outcome.extracted is None
    assert outcome.attempts == 3 and len(fake.requests) == 3
    assert "vendor_name" in outcome.errors[0]


def test_math_inconsistency_gets_one_recheck_then_is_accepted(tmp_path):
    # Every value is grounded, but a line is missing, so the lines don't sum to the subtotal.
    inconsistent = with_changes(
        line_items=[{"name": "WidgetA", "quantity": 10, "unit_price": 250.0, "amount": None}]
    )
    agent, fake = agent_with(tmp_path, inconsistent, inconsistent)
    outcome = agent.extract(SOURCE_1001)
    assert outcome.ok and outcome.attempts == 2  # one re-read, then accepted as written
    assert "do not fix the vendor's math" in fake.requests[1]["messages"][-1]["content"]
    assert outcome.math_issues == ["line items sum to 2,500.00, but subtotal is 5,000.00"]


def test_invalid_json_is_retried(tmp_path):
    agent, fake = agent_with(tmp_path, '{"invoice_number": ', json.dumps(GOOD_1001))
    outcome = agent.extract(SOURCE_1001)
    assert outcome.ok and outcome.attempts == 2
    assert "not valid JSON" in fake.requests[1]["messages"][-1]["content"]
