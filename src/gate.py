"""Deterministic policy gate (D5): findings -> one hard constraint."""

from __future__ import annotations

from collections.abc import Iterable

from src.schemas import GATE_PRECEDENCE, Action, Finding, Gate

ACTION_TO_GATE = {
    Action.DUPLICATE: Gate.DUPLICATE,
    Action.REJECT: Gate.REJECT,
    Action.RETURN: Gate.RETURN_TO_VENDOR,
    Action.HOLD: Gate.HOLD,
}


def compute_gate(findings: Iterable[Finding], *, ignore_duplicate: bool = False) -> Gate:
    """Highest-precedence gate implied by the findings. With ignore_duplicate,
    returns what the gate would have been without the duplicate (D9)."""
    gates = {
        ACTION_TO_GATE[f.action]
        for f in findings
        if f.action in ACTION_TO_GATE and not (ignore_duplicate and f.action == Action.DUPLICATE)
    }
    return next((g for g in GATE_PRECEDENCE if g in gates), Gate.ELIGIBLE)
