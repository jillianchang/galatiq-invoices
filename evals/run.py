"""Eval runner (PLAN.md section 13): every invoice against evals/expected.yaml.

    python -m evals.run                                   # both scenarios, every invoice
    python -m evals.run --scenario standalone --files invoice_1001.txt invoice_1007.csv
    python -m evals.run --llm replay                      # offline, from evals/llm_cache

Two scenarios, each on fresh databases:
  standalone  each file alone, as a grader running the spec's command would
  batch       every file in sorted order on one database (duplicates, revisions)

Hard checks (fail the run): gate, required findings, outcome (unless the
expectation marks it `outcome_llm`), amount paid, `must_not_pay`, and the batch
totals. Soft checks (reported, never fail): findings expected from agent
judgment, outcomes that depend on it, and extraction field accuracy against
evals/extraction_truth.yaml.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import statistics
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from evals.truth import (
    EVALS,
    INVOICE_DIR,
    expectation,
    expected_totals,
    load_expected,
    load_truth,
)
from src.agents.extraction import ExtractedInvoice, to_invoice
from src.config import Policy, Settings, load_policy
from src.db import connect, get_inventory, reset_db
from src.llm.client import LLMClient
from src.normalize import normalize_invoice
from src.observability import RunLogger, new_run_id
from src.payment import mock_payment
from src.pipeline import ProcessResult, build_graph, list_invoices, make_services, process_file
from src.schemas import Invoice

SCENARIOS = ("standalone", "batch")
RESULTS_DIR = EVALS / "results"


# ---------- scoring ----------


@dataclass
class Check:
    name: str
    ok: bool
    hard: bool
    expected: Any = None
    actual: Any = None


@dataclass
class InvoiceEval:
    file: str
    scenario: str
    checks: list[Check]
    gate: str
    outcome: str
    findings: list[str]
    method: str
    llm: dict[str, int]
    duration_ms: int
    paid_amount: str | None = None
    extraction: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks if c.hard)

    @property
    def soft_misses(self) -> list[Check]:
        return [c for c in self.checks if not c.hard and not c.ok]

    @property
    def cache_miss(self) -> bool:
        return self.llm.get("cache_misses", 0) > 0


def score(exp: dict[str, Any], result: ProcessResult) -> list[Check]:
    codes = {f.code for f in result.findings}
    paid = result.payment.amount if result.payment and result.payment.settled else None
    checks = [
        Check("gate", result.gate.value == exp["gate"], True, exp["gate"], result.gate.value),
        Check(
            "required_findings",
            set(exp["required_findings"]) <= codes,
            True,
            exp["required_findings"],
            sorted(set(exp["required_findings"]) - codes),  # missing
        ),
        Check(
            "outcome",
            result.outcome.value == exp["outcome"],
            not exp.get("outcome_llm", False),
            exp["outcome"],
            result.outcome.value,
        ),
    ]
    if exp["llm_findings"]:
        missing = sorted(set(exp["llm_findings"]) - codes)
        checks.append(Check("llm_findings", not missing, False, exp["llm_findings"], missing))
    if exp["outcome"] == "PAID" and result.outcome.value == "PAID":
        expected_amount = Decimal(exp["amount"])
        checks.append(Check("amount", paid == expected_amount, True, exp["amount"], str(paid)))
    if exp.get("must_not_pay"):
        checks.append(
            Check("must_not_pay", paid is None and result.outcome.value != "PAID", True, None, paid)
        )
    return checks


def _line_key(li) -> tuple:
    name = li.sku or "".join(ch for ch in (li.raw_name or "").lower() if ch.isalnum())
    return (name, (li.note or "").lower(), li.quantity, li.unit_price)


EXTRACTION_FIELDS = (
    "invoice_number",
    "vendor_name",
    "invoice_date",
    "due_date",
    "due_date_raw",
    "currency",
    "subtotal",
    "tax_amount",
    "shipping",
    "total",
    "payment_terms",
)


def score_extraction(
    truth: ExtractedInvoice, actual: Invoice, method: str, catalog: list[str]
) -> dict[str, Any]:
    """Field-by-field accuracy of the extracted invoice, after normalization."""
    expected = normalize_invoice(to_invoice(truth, actual.source_file, "txt"), catalog)
    fields = list(EXTRACTION_FIELDS)
    wrong = []
    if method == "failed":
        wrong = fields + ["line_items"]
    else:
        for name in fields:
            if getattr(expected, name) != getattr(actual, name):
                wrong.append(name)
        exp_lines = [_line_key(li) for li in expected.line_items]
        got_lines = [_line_key(li) for li in actual.line_items]
        if exp_lines != got_lines:
            wrong.append("line_items")
    total = len(fields) + 1
    return {"correct": total - len(wrong), "total": total, "wrong": wrong}


# ---------- running ----------


class EventRouter:
    """One LLM client across many databases: route its events to the current logger."""

    def __init__(self):
        self.logger: RunLogger | None = None

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        if self.logger is not None:
            self.logger.llm_event(event, payload)


class CountingBank:
    """The spec's mock_payment, counting calls so the report can prove no double payment."""

    def __init__(self, pay_fn=mock_payment):
        self.pay_fn = pay_fn
        self.calls: list[tuple[str, Decimal]] = []

    def __call__(self, vendor, amount):
        self.calls.append((vendor, amount))
        return self.pay_fn(vendor, amount)


@dataclass
class ScenarioReport:
    scenario: str
    invoices: list[InvoiceEval]
    totals_expected: dict[str, Any]
    totals_actual: dict[str, Any]
    payment_calls: int
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def totals_ok(self) -> bool:
        return self.totals_expected == self.totals_actual

    @property
    def passed(self) -> bool:
        return all(i.passed for i in self.invoices) and self.totals_ok


def run_scenario(
    scenario: str,
    files: list[Path],
    *,
    client: LLMClient,
    router: EventRouter,
    policy: Policy,
    out_dir: Path,
    eval_id: str,
    pay_fn=mock_payment,
) -> ScenarioReport:
    expected = load_expected()
    truth = load_truth()
    bank = CountingBank(pay_fn)
    groups = [[f] for f in files] if scenario == "standalone" else [files]
    (out_dir / "dbs").mkdir(parents=True, exist_ok=True)
    results: list[InvoiceEval] = []

    for group in groups:
        db_name = f"{scenario}-{group[0].stem}.db" if scenario == "standalone" else "batch.db"
        conn = connect(out_dir / "dbs" / db_name)
        reset_db(conn)
        logger = RunLogger(f"{eval_id}-{scenario}", conn, trace_dir=out_dir / "traces", stream=None)
        router.logger = logger
        svc = make_services(conn, policy, client, logger, pay_fn=bank)
        graph = build_graph(svc)
        catalog = list(get_inventory(conn))
        try:
            for path in group:
                with contextlib.redirect_stdout(io.StringIO()):  # mock_payment prints
                    result = process_file(svc, graph, path)
                exp = expectation(path.name, scenario, expected)
                paid = result.payment.amount if result.payment else None
                results.append(
                    InvoiceEval(
                        file=path.name,
                        scenario=scenario,
                        checks=score(exp, result),
                        gate=result.gate.value,
                        outcome=result.outcome.value,
                        findings=sorted(f.code for f in result.findings),
                        method=result.method,
                        llm=result.llm,
                        duration_ms=result.duration_ms,
                        paid_amount=str(paid)
                        if result.payment and result.payment.status == "paid"
                        else None,
                        extraction=score_extraction(
                            truth[path.name], result.invoice, result.method, catalog
                        )
                        if path.name in truth
                        else None,
                    )
                )
        finally:
            router.logger = None
            logger.close()
            conn.close()

    newly_paid = [r for r in results if r.paid_amount is not None]
    actual = {
        "paid": len(newly_paid),
        "paid_amount": str(sum((Decimal(r.paid_amount) for r in newly_paid), Decimal("0.00"))),
        "duplicates_blocked": sum(r.outcome == "DUPLICATE_BLOCKED" for r in results),
        "revisions_held": sum(
            r.outcome == "ON_HOLD" and "revision_of_paid_invoice" in r.findings for r in results
        ),
    }
    report = ScenarioReport(
        scenario,
        results,
        expected_totals(scenario, [p.name for p in files], expected),
        actual,
        len(bank.calls),
    )
    report.metrics = metrics(report)
    return report


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def metrics(report: ScenarioReport) -> dict[str, Any]:
    inv = report.invoices
    by_name = [{c.name: c for c in i.checks} for i in inv]
    required = sum(len(c["required_findings"].expected) for c in by_name)
    required_missing = sum(len(c["required_findings"].actual) for c in by_name)
    llm_expected = sum(len(c["llm_findings"].expected) for c in by_name if "llm_findings" in c)
    llm_missing = sum(len(c["llm_findings"].actual) for c in by_name if "llm_findings" in c)
    extraction = [i.extraction for i in inv if i.extraction]
    durations = [i.duration_ms for i in inv]
    calls = [i.llm.get("calls", 0) for i in inv]
    return {
        "invoices": len(inv),
        "passed": sum(i.passed for i in inv),
        "gate_accuracy": _ratio(sum(c["gate"].ok for c in by_name), len(inv)),
        "required_finding_recall": _ratio(required - required_missing, required),
        "llm_finding_recall": _ratio(llm_expected - llm_missing, llm_expected),
        "outcome_accuracy": _ratio(sum(c["outcome"].ok for c in by_name), len(inv)),
        "safety_violations": sum(not c["must_not_pay"].ok for c in by_name if "must_not_pay" in c),
        "extraction_field_accuracy": _ratio(
            sum(e["correct"] for e in extraction), sum(e["total"] for e in extraction)
        ),
        "llm_calls_per_invoice": round(statistics.mean(calls), 2) if calls else 0,
        "llm_calls": sum(calls),
        "cache_hits": sum(i.llm.get("cache_hits", 0) for i in inv),
        "cache_misses": sum(i.llm.get("cache_misses", 0) for i in inv),
        "tokens": sum(
            i.llm.get("prompt_tokens", 0) + i.llm.get("completion_tokens", 0) for i in inv
        ),
        "latency_ms_median": int(statistics.median(durations)) if durations else 0,
        "latency_ms_max": max(durations, default=0),
        "payment_calls": report.payment_calls,
        "totals_ok": report.totals_ok,
    }


def report_json(reports: list[ScenarioReport], meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "passed": all(r.passed for r in reports),
        "scenarios": {
            r.scenario: {
                "passed": r.passed,
                "metrics": r.metrics,
                "totals": {"expected": r.totals_expected, "actual": r.totals_actual},
                "invoices": [
                    {
                        **dataclasses.asdict(i),
                        "passed": i.passed,
                        "cache_miss": i.cache_miss,
                    }
                    for i in r.invoices
                ],
            }
            for r in reports
        },
    }


# ---------- output ----------


def _cell(check: Check | None, actual: str) -> str:
    if check is None or check.ok:
        return actual
    style = "red" if check.hard else "yellow"
    return f"[{style}]{actual}[/] [dim](exp {check.expected})[/]"


def render(report: ScenarioReport, console: Console) -> None:
    table = Table(title=f"{report.scenario} scenario", title_justify="left", expand=True)
    for col in ("File", "Gate", "Outcome", "Missing findings", "Extract", "LLM", "ms", ""):
        table.add_column(col, overflow="fold")
    for i in report.invoices:
        c = {ch.name: ch for ch in i.checks}
        missing = c["required_findings"].actual
        soft = c["llm_findings"].actual if "llm_findings" in c else []
        missing_cell = ", ".join(
            [f"[red]{m}[/]" for m in missing] + [f"[yellow]{m}[/]" for m in soft]
        )
        extract = f"{i.extraction['correct']}/{i.extraction['total']}" if i.extraction else i.method
        status = "[green]PASS[/]" if i.passed else "[red]FAIL[/]"
        if i.passed and i.soft_misses:
            status = "[yellow]PASS*[/]"
        if i.cache_miss:
            status += " [dim]cache miss[/]"
        outcome_check = c["outcome"] if not c["outcome"].ok else c.get("amount")
        table.add_row(
            i.file,
            _cell(c["gate"], i.gate),
            _cell(outcome_check, i.outcome + (f" ${i.paid_amount}" if i.paid_amount else "")),
            missing_cell,
            extract,
            str(i.llm.get("calls", 0)),
            str(i.duration_ms),
            status,
        )
    console.print(table)
    m = report.metrics
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"  # noqa: E731
    console.print(
        f"  passed {m['passed']}/{m['invoices']}   gate {pct(m['gate_accuracy'])}   "
        f"findings {pct(m['required_finding_recall'])}   "
        f"agent findings {pct(m['llm_finding_recall'])}   outcome {pct(m['outcome_accuracy'])}   "
        f"extraction {pct(m['extraction_field_accuracy'])}   "
        f"safety violations {m['safety_violations']}"
    )
    console.print(
        f"  LLM calls {m['llm_calls']} ({m['llm_calls_per_invoice']}/invoice, "
        f"{m['cache_hits']} cached, {m['cache_misses']} cache misses)   tokens {m['tokens']}   "
        f"latency median {m['latency_ms_median']} ms, max {m['latency_ms_max']} ms"
    )
    exp, act = report.totals_expected, report.totals_actual
    style = "green" if report.totals_ok else "red"
    console.print(
        f"  [{style}]totals[/]: paid {act['paid']} (${Decimal(act['paid_amount']):,.2f}), "
        f"duplicates blocked {act['duplicates_blocked']}, revisions held "
        f"{act['revisions_held']}" + ("" if report.totals_ok else f"   [red]expected {exp}[/]")
    )
    if m["cache_misses"]:
        console.print(
            "  [yellow]Some LLM calls weren't in the replay cache, so those invoices show "
            "fail-closed results. Set LLM_API_KEY and use --llm record to fill it.[/]"
        )
    console.print()


# ---------- CLI ----------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the invoice evals.")
    p.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    p.add_argument("--files", nargs="+", help="invoice file names (default: every invoice)")
    p.add_argument("--llm", choices=["record", "replay", "live"], help="LLM mode")
    p.add_argument("--out", type=Path, help="results folder (default: evals/results/<id>)")
    p.add_argument("--json", action="store_true", help="print the report JSON on stdout")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    all_files = list_invoices(INVOICE_DIR)
    if args.files:
        known = {p.name: p for p in all_files}
        unknown = [f for f in args.files if f not in known]
        if unknown:
            print(f"error: unknown invoice files: {', '.join(unknown)}", file=sys.stderr)
            return 2
        files = sorted(known[f] for f in args.files)
    else:
        files = all_files
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]

    policy = load_policy()
    settings = Settings.from_env()
    if args.llm:
        settings = dataclasses.replace(settings, mode=args.llm)
    eval_id = new_run_id()
    out_dir = args.out or RESULTS_DIR / eval_id
    out_dir.mkdir(parents=True, exist_ok=True)
    router = EventRouter()
    client = LLMClient(settings, policy.llm, on_event=router)
    console = Console(stderr=True)
    if args.files and "batch" in scenarios:
        console.print(
            "[yellow]Batch expectations assume the full invoice set; "
            "a subset may change duplicates and revisions.[/]"
        )

    reports = []
    for scenario in scenarios:
        report = run_scenario(
            scenario,
            files,
            client=client,
            router=router,
            policy=policy,
            out_dir=out_dir,
            eval_id=eval_id,
        )
        reports.append(report)
        render(report, console)

    meta = {
        "eval_id": eval_id,
        "llm_mode": settings.mode,
        "model": settings.model,
        "model_fast": settings.model_fast,
        "base_url": settings.base_url,
    }
    data = report_json(reports, meta)
    (out_dir / "report.json").write_text(json.dumps(data, indent=2, default=str))
    console.print(f"[dim]Report: {out_dir / 'report.json'}   Mode: {settings.mode}[/]")
    if args.json:
        print(json.dumps(data, indent=2, default=str))
    return 0 if data["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
