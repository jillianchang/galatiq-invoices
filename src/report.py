"""Human-readable console output: a results panel per invoice and a batch summary."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.schemas import Action, Status

if TYPE_CHECKING:
    from src.pipeline import ProcessResult

OUTCOME_STYLE = {
    Status.PAID: "bold green",
    Status.ON_HOLD: "bold yellow",
    Status.RETURNED_TO_VENDOR: "bold magenta",
    Status.REJECTED: "bold red",
    Status.DUPLICATE_BLOCKED: "bold cyan",
}
ACTION_STYLE = {
    Action.REJECT: "red",
    Action.DUPLICATE: "cyan",
    Action.RETURN: "magenta",
    Action.HOLD: "yellow",
    Action.SCRUTINY: "blue",
    Action.WARN: "dim yellow",
    Action.INFO: "dim",
}


def _money(amount: Decimal | None, currency: str | None) -> str:
    if amount is None:
        return "—"
    return f"${amount:,.2f}" if (currency or "USD") == "USD" else f"{amount:,.2f} {currency}"


def render_result(r: ProcessResult, console: Console) -> None:
    inv = r.invoice
    header = Text()
    header.append(
        f" {r.outcome.value.replace('_', ' ')} ", style=f"reverse {OUTCOME_STYLE[r.outcome]}"
    )
    header.append(
        f"  {inv.vendor_name or '(no vendor)'}  ·  {inv.invoice_number or '(no number)'}"
        f"  ·  {_money(inv.total, inv.currency)}",
        style="bold",
    )

    parts: list = [
        header,
        Text(
            f"Gate: {r.gate}"
            + (
                f" (would have been {r.underlying_gate} if not a duplicate)"
                if r.underlying_gate
                else ""
            ),
            style="dim",
        ),
    ]

    if r.findings:
        table = Table(show_header=False, box=None, padding=(0, 1))
        for f in r.findings:
            table.add_row(
                Text(f.action.value, style=ACTION_STYLE[f.action]),
                Text(f.code, style="bold"),
                Text(f.message),
                Text(f.source.value, style="dim"),
            )
        parts += [Text("\nFindings", style="underline"), table]
    else:
        parts.append(Text("\nNo findings.", style="green"))

    if r.approval:
        a = r.approval
        parts += [
            Text("\nDecision", style="underline"),
            Text(f"{a.decision} by {a.approver}: {a.reasoning}"),
        ]
        if a.critic_rounds:
            verdicts = ", ".join(f"round {c['round']}: {c['verdict']}" for c in a.critic_rounds)
            parts.append(Text(f"Critic: {verdicts}", style="dim"))
            for o in a.unresolved_objections:
                parts.append(Text(f"  unresolved: {o}", style="yellow"))
        for note in a.notes:
            parts.append(Text(f"Note: {note}", style="dim"))
        parts.append(Text(f"Next: {a.recommended_action}", style="bold"))
        if a.vendor_message:
            parts.append(Text(f"Message to vendor: {a.vendor_message}", style="magenta"))
    elif r.reasoning:
        parts += [Text("\nDecision", style="underline"), Text(r.reasoning)]

    if r.payment:
        parts.append(
            Text(f"Payment: {r.payment.message}", style="green" if r.payment.settled else "yellow")
        )

    stats = r.llm
    footer = (
        f"{r.file}  ·  {r.duration_ms / 1000:.1f}s  ·  LLM calls {stats.get('calls', 0)}"
        f" ({stats.get('cache_hits', 0)} cached)"
    )
    console.print(
        Panel(
            Group(*parts),
            title=r.file,
            subtitle=footer,
            title_align="left",
            border_style=OUTCOME_STYLE[r.outcome].split()[-1],
        )
    )


def render_summary(results: list[ProcessResult], console: Console) -> None:
    table = Table(title="Batch summary")
    for col in ("File", "Vendor", "Invoice", "Total", "Gate", "Outcome"):
        table.add_column(col)
    for r in results:
        table.add_row(
            r.file,
            r.invoice.vendor_name or "—",
            r.invoice.invoice_number or "—",
            _money(r.invoice.total, r.invoice.currency),
            r.gate.value,
            Text(r.outcome.value, style=OUTCOME_STYLE[r.outcome]),
        )
    console.print(table)
    counts = Counter(r.outcome for r in results)
    paid = sum(
        (
            r.payment.amount
            for r in results
            if r.payment and r.payment.status == "paid" and r.payment.amount
        ),
        Decimal(0),
    )
    console.print(
        "  ".join(f"[{OUTCOME_STYLE[s]}]{s.value}: {counts.get(s, 0)}[/]" for s in OUTCOME_STYLE)
        + f"   Paid this run: ${paid:,.2f}"
    )
