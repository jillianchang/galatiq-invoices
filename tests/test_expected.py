"""The eval expectations must cover every invoice file and use valid values."""

from pathlib import Path

import yaml

from src.config import ROOT
from src.schemas import Gate, Status

INVOICE_DIR = ROOT / "data" / "invoices"
EXPECTED = yaml.safe_load((ROOT / "evals" / "expected.yaml").read_text())["invoices"]


def _invoice_files() -> set[str]:
    return {p.name for p in Path(INVOICE_DIR).iterdir() if p.name.startswith("invoice_")}


def test_every_invoice_file_has_an_expectation_and_vice_versa():
    assert set(EXPECTED) == _invoice_files()


def test_expectations_are_well_formed():
    for name, exp in EXPECTED.items():
        for scenario in (exp, exp.get("batch") or {}):
            if "gate" in scenario:
                assert scenario["gate"] in Gate.__members__, name
            if "outcome" in scenario:
                assert scenario["outcome"] in Status.__members__, name
        assert exp["source"] in ("spec", "assumption"), name
        if exp["outcome"] == "PAID":
            assert "amount" in exp and not exp.get("must_not_pay"), name
        else:
            assert exp.get("must_not_pay") is True, name


def test_spec_defined_invoices_are_tagged_spec():
    spec = {n for n, e in EXPECTED.items() if e["source"] == "spec"}
    assert spec == {
        f"invoice_{n}.{ext}"
        for n, ext in [
            ("1001", "txt"),
            ("1002", "txt"),
            ("1003", "txt"),
            ("1004", "json"),
            ("1006", "csv"),
            ("1008", "txt"),
            ("1009", "json"),
            ("1016", "json"),
        ]
    }
