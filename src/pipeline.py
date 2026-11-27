"""The end-to-end pipeline as a LangGraph state graph (D4).

    ingest -> validate -> gate -+-> approve -+-> pay ----------+-> finalize
                                |            +-> rejection_log -+
                                +-> rejection_log (duplicate) --+

Control flow is fixed and depends only on structured state; agency lives
inside the nodes (extraction loop, investigator, approval + critic).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.approval import ApprovalResult, ApprovalWorkflow, Decision
from src.agents.extraction import ExtractionAgent
from src.agents.investigator import InvestigatorAgent
from src.config import Policy
from src.gate import compute_gate
from src.ingestion import IngestionResult, ingest
from src.observability import RunLogger
from src.payment import PaymentResult, mock_payment, pay_invoice
from src.rules.context import Inventory
from src.schemas import Action, Finding, Gate, Invoice, Status
from src.store import finalize, invoice_keys, register, start_submission
from src.validation import validate

DECISION_TO_STATUS = {
    Decision.HOLD: Status.ON_HOLD,
    Decision.REJECT: Status.REJECTED,
    Decision.RETURN_TO_VENDOR: Status.RETURNED_TO_VENDOR,
}


@dataclass
class Services:
    conn: Any  # sqlite3.Connection
    policy: Policy
    inventory: Inventory
    logger: RunLogger
    approval: ApprovalWorkflow
    extractor: ExtractionAgent | None = None
    investigator: InvestigatorAgent | None = None
    pay_fn: Any = mock_payment


@dataclass
class ProcessResult:
    file: str
    run_id: str
    submission_id: int
    invoice_id: int
    invoice: Invoice
    method: str
    gate: Gate
    findings: list[Finding]
    outcome: Status
    underlying_gate: Gate | None = None  # for duplicates: the gate without the duplicate
    approval: ApprovalResult | None = None
    reasoning: str | None = None  # for outcomes decided without the approval agent
    payment: PaymentResult | None = None
    investigator_summary: str | None = None
    llm: dict[str, int] = field(default_factory=dict)
    duration_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        inv = self.invoice
        a = self.approval
        return {
            "file": self.file,
            "run_id": self.run_id,
            "submission_id": self.submission_id,
            "outcome": self.outcome.value,
            "gate": self.gate.value,
            "underlying_gate": self.underlying_gate.value if self.underlying_gate else None,
            "invoice": {
                "invoice_number": inv.invoice_number,
                "vendor": inv.vendor_name,
                "total": str(inv.total) if inv.total is not None else None,
                "currency": inv.currency or "USD",
                "due_date": str(inv.due_date) if inv.due_date else inv.due_date_raw,
                "extraction": self.method,
            },
            "findings": [f.model_dump(mode="json") for f in self.findings],
            "decision": None
            if a is None
            else {
                "decision": a.decision.value,
                "approver": a.approver,
                "reasoning": a.reasoning,
                "recommended_action": a.recommended_action,
                "vendor_message": a.vendor_message,
                "critic_rounds": a.critic_rounds,
                "unresolved_objections": a.unresolved_objections,
                "notes": a.notes,
            },
            "reasoning": self.reasoning if a is None else a.reasoning,
            "payment": None
            if self.payment is None
            else {
                "status": self.payment.status,
                "message": self.payment.message,
                "amount": str(self.payment.amount) if self.payment.amount is not None else None,
            },
            "investigator_summary": self.investigator_summary,
            "llm": self.llm,
            "duration_ms": self.duration_ms,
        }


class State(TypedDict, total=False):
    path: Path
    submission_id: int
    invoice_id: int
    ingestion: IngestionResult
    findings: list[Finding]
    investigator_summary: str | None
    gate: Gate
    underlying_gate: Gate | None
    approval: ApprovalResult | None
    reasoning: str | None
    payment: PaymentResult | None
    outcome: Status


def build_graph(svc: Services):
    log = svc.logger

    def ingest_node(state: State) -> State:
        path = state["path"]
        result = ingest(path, catalog=svc.inventory, agent=svc.extractor)
        submission_id = start_submission(svc.conn, log.run_id, path.name, result.raw_hash)
        log.submission_id = submission_id
        inv = result.invoice
        log.stage(
            "ingestion",
            result.method,
            format=inv.source_format,
            attempts=result.attempts,
            invoice_number=inv.invoice_number,
            vendor=inv.vendor_name,
            total=inv.total,
            lines=len(inv.line_items),
        )
        return {"ingestion": result, "submission_id": submission_id}

    def validate_node(state: State) -> State:
        ingestion = state["ingestion"]
        result = validate(
            ingestion,
            conn=svc.conn,
            policy=svc.policy,
            inventory=svc.inventory,
            investigator=svc.investigator,
        )
        invoice_id = register(
            svc.conn, state["submission_id"], ingestion.invoice, ingestion.raw_hash, result.findings
        )
        inv = result.investigation
        log.stage(
            "validation",
            "findings",
            findings=[f"{f.code}:{f.action.value}" for f in result.findings],
            investigator=None
            if inv is None
            else {
                "rounds": inv.rounds,
                "tools": [c["name"] for c in inv.tool_calls],
                "added": [f.code for f in inv.findings],
                "dropped": len(inv.dropped),
                "error": inv.error,
            },
        )
        return {
            "findings": result.findings,
            "invoice_id": invoice_id,
            "investigator_summary": inv.summary if inv else None,
        }

    def gate_node(state: State) -> State:
        gate = compute_gate(state["findings"])
        underlying = (
            compute_gate(state["findings"], ignore_duplicate=True)
            if gate == Gate.DUPLICATE
            else None
        )
        log.stage("gate", gate.value, underlying=underlying)
        return {"gate": gate, "underlying_gate": underlying}

    def approve_node(state: State) -> State:
        result = svc.approval.decide(
            state["ingestion"].invoice,
            state["findings"],
            state["gate"],
            state.get("investigator_summary"),
        )
        log.stage(
            "approval",
            result.decision.value,
            approver=result.approver,
            critic=[c["verdict"] for c in result.critic_rounds],
            unresolved=len(result.unresolved_objections),
            notes=result.notes,
        )
        return {"approval": result}

    def pay_node(state: State) -> State:
        inv = state["ingestion"].invoice
        fp, ch = invoice_keys(inv, state["ingestion"].raw_hash)
        payment = pay_invoice(
            svc.conn,
            invoice_id=state["invoice_id"],
            fingerprint=fp,
            content_hash=ch,
            vendor=inv.vendor_name,
            amount=inv.total,
            pay_fn=svc.pay_fn,
        )
        outcome = Status.PAID if payment.settled else Status.ON_HOLD
        log.stage("payment", payment.status, amount=payment.amount, message=payment.message)
        return {"payment": payment, "outcome": outcome}

    def rejection_log_node(state: State) -> State:
        if state["gate"] == Gate.DUPLICATE:
            dup = next(f for f in state["findings"] if f.action == Action.DUPLICATE)
            reasoning = (
                f"{dup.message} Without the duplicate, the gate would have been "
                f"{state['underlying_gate']}."
            )
            outcome = Status.DUPLICATE_BLOCKED
        else:
            approval = state["approval"]
            reasoning = approval.reasoning
            outcome = DECISION_TO_STATUS[approval.decision]
        log.stage("rejection_log", outcome.value, reasoning=reasoning)
        return {"outcome": outcome, "reasoning": reasoning}

    def finalize_node(state: State) -> State:
        approval = state.get("approval")
        payment = state.get("payment")
        if approval is None:
            decision, reasoning, approver, critic = (
                "BLOCKED",
                state.get("reasoning"),
                "policy",
                None,
            )
        else:
            decision, reasoning, approver = (
                approval.decision.value,
                approval.reasoning,
                approval.approver,
            )
            critic = approval.critic_rounds
            if payment and not payment.settled:
                reasoning = f"{reasoning} Payment not made: {payment.message}"
        finalize(
            svc.conn,
            submission_id=state["submission_id"],
            invoice_id=state["invoice_id"],
            gate=state["gate"].value,
            outcome=state["outcome"],
            findings=state["findings"],
            decision=decision,
            reasoning=reasoning or "",
            approver=approver,
            critic=critic,
            update_invoice_status=state["gate"] != Gate.DUPLICATE,
        )
        log.stage("finalize", state["outcome"].value)
        return {}

    graph = StateGraph(State)
    for name, fn in [
        ("ingest", ingest_node),
        ("validate", validate_node),
        ("gate", gate_node),
        ("approve", approve_node),
        ("pay", pay_node),
        ("rejection_log", rejection_log_node),
        ("finalize", finalize_node),
    ]:
        graph.add_node(name, fn)
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "validate")
    graph.add_edge("validate", "gate")
    graph.add_conditional_edges(
        "gate",
        lambda s: "rejection_log" if s["gate"] == Gate.DUPLICATE else "approve",
        ["approve", "rejection_log"],
    )
    graph.add_conditional_edges(
        "approve",
        lambda s: "pay" if s["approval"].decision == Decision.APPROVE else "rejection_log",
        ["pay", "rejection_log"],
    )
    graph.add_edge("pay", "finalize")
    graph.add_edge("rejection_log", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def process_file(svc: Services, graph, path: Path) -> ProcessResult:
    start = time.monotonic()
    svc.logger.begin(path.name)
    state: State = graph.invoke({"path": path})
    return ProcessResult(
        file=path.name,
        run_id=svc.logger.run_id,
        submission_id=state["submission_id"],
        invoice_id=state["invoice_id"],
        invoice=state["ingestion"].invoice,
        method=state["ingestion"].method,
        gate=state["gate"],
        findings=state["findings"],
        outcome=state["outcome"],
        underlying_gate=state.get("underlying_gate"),
        approval=state.get("approval"),
        reasoning=state.get("reasoning"),
        payment=state.get("payment"),
        investigator_summary=state.get("investigator_summary"),
        llm=dict(svc.logger.llm_stats),
        duration_ms=int((time.monotonic() - start) * 1000),
    )
