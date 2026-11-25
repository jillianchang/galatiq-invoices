from src.agents.extraction import ExtractedInvoice, ExtractedLine
from src.config import ROOT
from src.grounding import check_grounding, check_math

DATA = ROOT / "data" / "invoices"


def extracted(**overrides):
    base = dict(
        invoice_number="INV-1001",
        vendor_name="Widgets Inc.",
        vendor_address=None,
        invoice_date_raw="2026-01-15",
        due_date_raw="2026-02-01",
        currency="USD",
        line_items=[
            ExtractedLine(name="WidgetA", quantity=10, unit_price=250.0, amount=None),
            ExtractedLine(name="WidgetB", quantity=5, unit_price=500.0, amount=None),
        ],
        subtotal=5000.0,
        tax_rate_percent=0,
        tax_amount=0.0,
        shipping=None,
        total=5000.0,
        payment_terms="Net 15",
        notes=None,
    )
    base.update(overrides)
    return ExtractedInvoice(**base)


SOURCE_1001 = (DATA / "invoice_1001.txt").read_text()


def test_faithful_extraction_is_grounded():
    assert check_grounding(extracted(), SOURCE_1001) == []
    assert check_math(extracted()) == []


def test_computed_line_amount_is_caught():
    lines = [
        ExtractedLine(name="WidgetA", quantity=10, unit_price=250.0, amount=2500.0),  # not printed
        ExtractedLine(name="WidgetB", quantity=5, unit_price=500.0, amount=None),
    ]
    errors = check_grounding(extracted(line_items=lines), SOURCE_1001)
    assert len(errors) == 1 and "line 1 (WidgetA) amount=2500.0" in errors[0]


def test_hallucinated_values_are_caught():
    bad = extracted(
        vendor_name="Widgets International",
        total=5500.0,
        due_date_raw="2026-02-15",
        invoice_number="INV-9999",
        line_items=[ExtractedLine(name="GadgetX", quantity=3, unit_price=750.0, amount=None)],
    )
    errors = "\n".join(check_grounding(bad, SOURCE_1001))
    for fragment in ("vendor_name", "total=5500", "due_date_raw", "invoice_number", "GadgetX"):
        assert fragment in errors


def test_no_line_items_is_an_error():
    assert any("No line items" in e for e in check_grounding(extracted(line_items=[]), SOURCE_1001))


def test_ocr_noise_is_grounded_1012():
    source = (DATA / "invoice_1012.txt").read_text()
    ocr = extracted(
        invoice_number="INV 1012",
        vendor_name="QuickShip Distributers",
        invoice_date_raw="26-Jan-2026",
        due_date_raw="25-Feb-2026",  # model fixed 2O26
        line_items=[
            ExtractedLine(name="Widget A", quantity=12, unit_price=250, amount=3000.0),
            ExtractedLine(name="WidgetB", quantity=7, unit_price=500, amount=3500.0),  # "3,500.O0"
            ExtractedLine(name="Gadget X", quantity=4, unit_price=750, amount=3000.0),
        ],
        subtotal=9500.0,
        tax_rate_percent=5,
        tax_amount=475.0,
        total=9975.0,
        payment_terms="Net 30",
    )
    assert check_grounding(ocr, source) == []
    assert check_math(ocr) == []


def test_math_issues_detected_without_being_grounding_errors():
    source_1013 = (DATA / "invoice_1013.json").read_text()  # stated total is $50 too high
    lines = [
        ExtractedLine(name="WidgetA", quantity=15, unit_price=250, amount=3750),
        ExtractedLine(name="WidgetB", quantity=10, unit_price=500, amount=5000),
        ExtractedLine(name="GadgetX", quantity=5, unit_price=750, amount=3750),
        ExtractedLine(name="WidgetA", quantity=5, unit_price=240, amount=1200),
        ExtractedLine(name="WidgetB", quantity=8, unit_price=480, amount=3840),
        ExtractedLine(name="GadgetX", quantity=3, unit_price=750, amount=2250),
        ExtractedLine(name="WidgetA", quantity=2, unit_price=250, amount=500),
        ExtractedLine(name="GadgetX", quantity=1, unit_price=750, amount=750),
    ]
    e = extracted(
        invoice_number="INV-1013",
        vendor_name="Atlas Industrial Supply",
        invoice_date_raw="2026-01-24",
        due_date_raw="2026-03-24",
        line_items=lines,
        subtotal=21040.0,
        tax_rate_percent=None,
        tax_amount=1472.8,
        total=22562.8,
        payment_terms="Net 60",
    )
    assert check_grounding(e, source_1013) == []
    assert check_math(e) == ["subtotal + tax + shipping = 22,512.80, but total is 22,562.80"]


def test_line_math_and_missing_line_detected():
    lines = [ExtractedLine(name="WidgetA", quantity=10, unit_price=250.0, amount=2400.0)]
    issues = check_math(extracted(line_items=lines))
    assert "line 1 (WidgetA): 10 x 250.00 = 2,500.00, not 2,400.00" in issues
    assert any("sum to 2,400.00" in i for i in issues)
