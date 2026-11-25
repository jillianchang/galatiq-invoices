"""Deterministic validation against evals/expected.yaml, with no LLM.

Structured files go through the real parsers. Text files use hand-built
extractions (what a correct extraction returns), since extraction is tested
separately.
"""

from decimal import Decimal

import pytest
import yaml

from src.config import ROOT, load_policy
from src.db import connect, get_inventory, init_db
from src.gate import compute_gate
from src.ingestion import IngestionResult, ingest
from src.normalize import normalize_invoice
from src.schemas import Action, Gate, Invoice, LineItem
from src.validation import validate

DATA = ROOT / "data" / "invoices"
EXPECTED = yaml.safe_load((ROOT / "evals" / "expected.yaml").read_text())["invoices"]
POLICY = load_policy()
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


def text_invoice(name, lines, **fields):
    """A correct extraction of a text invoice, normalized like the real pipeline."""
    raw = (DATA / name).read_text()
    items = [LineItem(raw_name=n, quantity=Decimal(q), unit_price=Decimal(p)) for n, q, p in lines]
    money = {
        k: Decimal(v)
        for k, v in fields.items()
        if k in ("subtotal", "tax_amount", "shipping", "total", "tax_rate")
    }
    other = {k: v for k, v in fields.items() if k not in money}
    inv = Invoice(line_items=items, source_file=name, source_format="txt", **money, **other)
    inv = normalize_invoice(inv, ["WidgetA", "WidgetB", "GadgetX", "FakeItem"])
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


TEXT_CASES = {
    "invoice_1001.txt": dict(
        lines=[("WidgetA", 10, 250), ("WidgetB", 5, 500)],
        invoice_number="INV-1001",
        vendor_name="Widgets Inc.",
        invoice_date_raw="2026-01-15",
        due_date_raw="2026-02-01",
        subtotal=5000,
        tax_rate=0,
        tax_amount=0,
        total=5000,
        payment_terms="Net 15",
    ),
    "invoice_1002.txt": dict(
        lines=[("GadgetX", 20, 750)],
        invoice_number="1002",
        vendor_name="Gadgets Co.",
        invoice_date_raw="Jan 30 2026",
        due_date_raw="2026-01-30",
        total=15000,
        payment_terms="Net 30",
    ),
    "invoice_1003.txt": dict(
        lines=[("FakeItem", 100, 1000)],
        invoice_number="INV-1003",
        vendor_name="Fraudster LLC",
        invoice_date_raw="2026-01-20",
        due_date_raw="yesterday",
        total=100000,
        payment_terms="Immediate",
    ),
    "invoice_1008.txt": dict(
        lines=[("SuperGizmo", 12, 400), ("MegaSprocket", 6, 850)],
        invoice_number="INV-1008",
        vendor_name="NoProd Industries",
        invoice_date_raw="2026-01-10",
        due_date_raw="2026-01-20",
        total=9900,
    ),
    "invoice_1010.txt": dict(
        lines=[
            ("WidgetA", 8, 250),
            ("WidgetB", 4, 500),
            ("GadgetX", 2, 750),
            ("WidgetA (rush order)", 4, 300),
        ],
        invoice_number="INV-1010",
        vendor_name="Consolidated Materials Group",
        invoice_date_raw="January 27, 2026",
        due_date_raw="February 26, 2026",
        subtotal=6700,
        tax_amount=335,
        shipping=150,
        total=7185,
        payment_terms="Net 30",
    ),
    "invoice_1011.txt": dict(
        lines=[("WidgetA", 6, 250), ("WidgetB", 3, 500)],
        invoice_number="INV-1011",
        vendor_name="Summit Manufacturing Co.",
        invoice_date_raw="2026-01-20",
        due_date_raw="2026-02-20",
        subtotal=3000,
        tax_rate=0,
        tax_amount=0,
        total=3000,
        payment_terms="Net 30",
    ),
    "invoice_1012.txt": dict(
        lines=[("Widget A", 12, 250), ("WidgetB", 7, 500), ("Gadget X", 4, 750)],
        invoice_number="INV 1012",
        vendor_name="QuickShip Distributers",
        invoice_date_raw="26-Jan-2O26",
        due_date_raw="25-Feb-2026",
        subtotal=9500,
        tax_rate="0.05",
        tax_amount=475,
        total=9975,
        payment_terms="Net 30",
    ),
    "invoice_2001_injection.txt": dict(
        lines=[("WidgetA", 4, 250), ("WidgetB", 2, 500)],
        invoice_number="INV-2001",
        vendor_name="Northwind Fasteners",
        invoice_date_raw="2026-02-03",
        due_date_raw="2026-03-05",
        subtotal=2000,
        tax_rate=0,
        tax_amount=0,
        total=2000,
        payment_terms="Net 30",
    ),
}


@pytest.mark.parametrize("name", sorted(TEXT_CASES))
def test_text_invoices_match_expectations(conn, name):
    case = dict(TEXT_CASES[name])
    assert_expected(name, run(conn, text_invoice(name, case.pop("lines"), **case)))


def test_every_expected_text_invoice_has_a_case():
    text_files = {n for n in EXPECTED if n.endswith(".txt")}
    assert text_files == set(TEXT_CASES)


def test_fraud_pattern_needs_all_three_signals(conn):
    case = dict(TEXT_CASES["invoice_1003.txt"])
    case.pop("lines")
    case["total"] = 9000  # 9 x $1,000: under threshold, so held but not auto-rejected
    findings = run(conn, text_invoice("invoice_1003.txt", [("FakeItem", 9, 1000)], **case))
    assert "fraud_pattern" not in {f.code for f in findings}
    assert compute_gate(findings) == Gate.HOLD


def test_price_rules(conn):
    # 1010: +20% with a reason -> WARN only
    case = dict(TEXT_CASES["invoice_1010.txt"])
    findings = run(conn, text_invoice("invoice_1010.txt", case.pop("lines"), **case))
    price = [f for f in findings if f.code == "price_over_catalog"]
    assert len(price) == 1 and price[0].action == Action.WARN
    assert price[0].evidence["reason"] == "rush order"
    # no reason given -> HOLD
    case = dict(TEXT_CASES["invoice_1001.txt"])
    case["lines"] = [("WidgetA", 10, 310), ("WidgetB", 5, 500)]
    case.update(subtotal=5600, total=5600)
    findings = run(conn, text_invoice("invoice_1001.txt", case.pop("lines"), **case))
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
