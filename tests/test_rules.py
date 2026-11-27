"""Deterministic validation against evals/expected.yaml, with no LLM.

Structured files go through the real parsers. Text and PDF files use the
ground-truth extractions in evals/extraction_truth.yaml, since extraction is
tested separately.
"""

import pytest
import yaml

from evals.truth import load_truth
from src.agents.extraction import ExtractedLine, to_invoice
from src.config import ROOT, load_policy
from src.db import connect, get_inventory, init_db
from src.gate import compute_gate
from src.grounding import check_grounding
from src.ingestion import IngestionResult, ingest
from src.normalize import normalize_invoice
from src.parsers.documents import load_document
from src.schemas import Action, Gate, Invoice
from src.validation import validate

DATA = ROOT / "data" / "invoices"
EXPECTED = yaml.safe_load((ROOT / "evals" / "expected.yaml").read_text())["invoices"]
POLICY = load_policy()
TRUTH = load_truth()
CATALOG = ["WidgetA", "WidgetB", "GadgetX", "FakeItem"]
STRUCTURED = sorted(n for n in EXPECTED if n.endswith((".json", ".csv", ".xml")))


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    init_db(c)
    yield c
    c.close()


def run(conn, ingestion):
    inventory = get_inventory(conn)
    return validate(ingestion, conn=conn, policy=POLICY, inventory=inventory).findings


def llm_invoice(name, lines=None, **changes):
    """A correct extraction (evals/extraction_truth.yaml), optionally altered, then
    converted and normalized exactly like the real pipeline."""
    raw, fmt = load_document(DATA / name)
    if lines is not None:
        changes["line_items"] = [
            ExtractedLine(name=n, quantity=q, unit_price=p, amount=None) for n, q, p in lines
        ]
    extracted = TRUTH[name].model_copy(update=changes)
    inv = normalize_invoice(to_invoice(extracted, name, fmt), CATALOG)
    return IngestionResult(inv, raw, f"hash-{name}", "llm", attempts=1)


def assert_expected(name, findings):
    exp = EXPECTED[name]
    codes = {f.code for f in findings}
    missing = set(exp["required_findings"]) - codes
    assert not missing, f"{name}: missing {missing}; got {sorted(codes)}"
    assert compute_gate(findings) == Gate[exp["gate"]], f"{name}: {sorted(codes)}"


@pytest.mark.parametrize("name", STRUCTURED)
def test_structured_invoices_match_expectations(conn, name):
    ingestion = ingest(DATA / name, catalog=get_inventory(conn), agent=None)
    assert_expected(name, run(conn, ingestion))


@pytest.mark.parametrize("name", sorted(TRUTH))
def test_llm_extracted_invoices_match_expectations(conn, name):
    assert_expected(name, run(conn, llm_invoice(name)))


@pytest.mark.parametrize("name", sorted(TRUTH))
def test_ground_truth_is_grounded_in_the_document(name):
    """The truth file must itself pass the extraction agent's grounding check."""
    raw, _ = load_document(DATA / name)
    assert check_grounding(TRUTH[name], raw) == []


def test_every_llm_extracted_invoice_has_ground_truth():
    assert set(TRUTH) == {n for n in EXPECTED if n.endswith((".txt", ".pdf"))}


def test_fraud_pattern_needs_all_three_signals(conn):
    # 9 x $1,000: under threshold, so held but not auto-rejected
    findings = run(conn, llm_invoice("invoice_1003.txt", [("FakeItem", 9, 1000)], total=9000))
    assert "fraud_pattern" not in {f.code for f in findings}
    assert compute_gate(findings) == Gate.HOLD


def test_price_rules(conn):
    # 1010: +20% with a reason -> WARN only
    findings = run(conn, llm_invoice("invoice_1010.txt"))
    price = [f for f in findings if f.code == "price_over_catalog"]
    assert len(price) == 1 and price[0].action == Action.WARN
    assert price[0].evidence["reason"] == "rush order"
    # no reason given -> HOLD
    lines = [("WidgetA", 10, 310), ("WidgetB", 5, 500)]
    findings = run(conn, llm_invoice("invoice_1001.txt", lines, subtotal=5600, total=5600))
    price = [f for f in findings if f.code == "price_over_catalog"]
    assert len(price) == 1 and price[0].action == Action.HOLD


def test_extraction_failed_skips_rules(conn):
    raw = (DATA / "invoice_2001_injection.txt").read_text()
    from src.rules.codes import finding

    failed = IngestionResult(
        Invoice(source_file="x.txt"),
        raw,
        "h",
        "failed",
        findings=[finding("extraction_failed", "could not extract")],
    )
    codes = [f.code for f in run(conn, failed)]
    assert codes == ["extraction_failed", "prompt_injection"]  # no spurious missing_vendor REJECT
