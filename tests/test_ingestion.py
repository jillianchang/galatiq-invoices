import json

from src.agents.extraction import ExtractionAgent
from src.config import ROOT
from src.ingestion import ingest
from src.schemas import Action
from tests.fakes import FakeOpenAI, completion, make_client
from tests.test_extraction import GOOD_1001

DATA = ROOT / "data" / "invoices"
CATALOG = ["WidgetA", "WidgetB", "GadgetX", "FakeItem"]
STRUCTURED = [p for p in sorted(DATA.iterdir()) if p.suffix in (".json", ".csv", ".xml")]


def test_structured_files_never_call_the_llm():
    for path in STRUCTURED:
        result = ingest(path, catalog=CATALOG, agent=None)
        assert result.method == "parser", path.name
        assert result.findings == [] and result.invoice.line_items, path.name
        assert len(result.raw_hash) == 64


def test_unstructured_without_llm_fails_closed():
    result = ingest(DATA / "invoice_1001.txt", catalog=CATALOG, agent=None)
    assert result.method == "failed"
    assert [(f.code, f.action) for f in result.findings] == [("extraction_failed", Action.HOLD)]
    assert result.invoice.source_file == "invoice_1001.txt"


def test_text_invoice_extracted_and_normalized(tmp_path):
    fake = FakeOpenAI(completion(json.dumps(GOOD_1001)))
    client, _, _ = make_client(tmp_path, fake)
    result = ingest(DATA / "invoice_1001.txt", catalog=CATALOG, agent=ExtractionAgent(client))
    assert result.method == "llm" and result.attempts == 1
    inv = result.invoice
    assert inv.invoice_number == "INV-1001" and inv.source_format == "txt"
    assert [li.sku for li in inv.line_items] == ["WidgetA", "WidgetB"]
    assert str(inv.due_date) == "2026-02-01"
    assert str(inv.total) == "5000.00"
    assert str(inv.line_items[0].quantity) == "10"


def test_broken_structured_file_falls_back_to_llm(tmp_path):
    broken = tmp_path / "invoice_9999.json"
    broken.write_text("Vendor: Widgets Inc. -- not actually JSON " + json.dumps(GOOD_1001))
    fake = FakeOpenAI(completion(json.dumps(GOOD_1001)))
    client, _, _ = make_client(tmp_path, fake)
    result = ingest(broken, catalog=CATALOG, agent=ExtractionAgent(client))
    assert result.method == "llm"
    assert result.events[0]["event"] == "parse_failed"


def test_empty_pdf_fails_closed(tmp_path):
    from pdfplumber.page import Page  # noqa: F401  (pdfplumber importable)

    blank = tmp_path / "scan.pdf"
    blank.write_bytes(
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
        b"trailer<</Root 1 0 R>>\n%%EOF"
    )
    result = ingest(blank, catalog=CATALOG, agent=None)
    assert result.method == "failed"
    assert result.findings[0].code == "extraction_failed"
