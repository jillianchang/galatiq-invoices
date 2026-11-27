"""Approval stage: the approval agent (simulated VP) and the critic loop (D6, D13).

The policy gate sets the hard constraint; the agent decides within it. What it
may decide depends on the gate (ALLOWED). A decision outside that set is
replaced by the gate's fail-closed default, so the agent can never pay an
invoice the gate blocks (D3).

The critic reviews the decision when `critic_required_when` in policy.yaml says
so (over the threshold, or any finding), for up to `max_critic_rounds`, on
every gate except DUPLICATE. Its power depends on the gate, through ALLOWED:
on ELIGIBLE and HOLD it can push the decision toward caution; on RETURN it can
push to REJECT and checks the vendor message; on REJECT the decision is fixed
and it checks that the reasoning matches the findings. It can never move a
decision toward payment (D3). An APPROVE that still has unresolved objections
becomes a HOLD.

If the LLM is unavailable, the decision falls back to the gate's default with
deterministic reasoning: ELIGIBLE becomes HOLD, never APPROVE (D16).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

from src.config import ROOT, Policy
from src.llm.client import LLMClient, LLMError
from src.schemas import Action, Finding, Gate, Invoice

APPROVAL_PROMPT = (ROOT / "prompts" / "approval.md").read_text
CRITIC_PROMPT = (ROOT / "prompts" / "critic.md").read_text


class Decision(StrEnum):
    APPROVE = "APPROVE"
    HOLD = "HOLD"
    RETURN_TO_VENDOR = "RETURN_TO_VENDOR"
    REJECT = "REJECT"


ALLOWED: dict[Gate, tuple[Decision, ...]] = {
    Gate.ELIGIBLE: (Decision.APPROVE, Decision.HOLD),
    Gate.HOLD: (Decision.HOLD, Decision.REJECT),
    Gate.RETURN_TO_VENDOR: (Decision.RETURN_TO_VENDOR, Decision.REJECT),
    Gate.REJECT: (Decision.REJECT,),
}
# What a gate decides when the agent can't (or answers outside ALLOWED).
FAIL_CLOSED: dict[Gate, Decision] = {
    Gate.ELIGIBLE: Decision.HOLD,
    Gate.HOLD: Decision.HOLD,
    Gate.RETURN_TO_VENDOR: Decision.RETURN_TO_VENDOR,
    Gate.REJECT: Decision.REJECT,
}


class ApprovalOutput(BaseModel):
    decision: Literal["APPROVE", "HOLD", "RETURN_TO_VENDOR", "REJECT"]
    reasoning: str
    recommended_action: str
    vendor_message: str | None


class CriticOutput(BaseModel):
    verdict: Literal["agree", "object"]
    objections: list[str]
    summary: str


@dataclass
class ApprovalResult:
    decision: Decision
    reasoning: str
    recommended_action: str
    vendor_message: str | None = None
    approver: str = "agent"
    critic_rounds: list[dict] = field(default_factory=list)
    unresolved_objections: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # coercions, fallbacks
    llm_error: str | None = None


CRITIC_SCOPE: dict[Gate, str] = {
    Gate.ELIGIBLE: "The decision is open: you may push APPROVE to HOLD.",
    Gate.HOLD: "You may push HOLD to REJECT, and check the recommended resolution.",
    Gate.RETURN_TO_VENDOR: (
        "The invoice will not be paid. You may push RETURN_TO_VENDOR to REJECT, and check "
        "that the vendor message states the discrepancies accurately."
    ),
    Gate.REJECT: (
        "The rejection is fixed by policy and cannot change. Check only that the reasoning "
        "is accurate and cites the findings that drive it."
    ),
}

CRITIC_TRIGGERS = {
    "over_threshold": lambda findings: any(f.code == "over_threshold" for f in findings),
    "any_finding": lambda findings: any(f.action != Action.INFO for f in findings),
}


def critic_required(findings: list[Finding], gate: Gate, when: list[str]) -> bool:
    """Whether the critic reviews this decision, per `critic_required_when` in policy.yaml."""
    if gate == Gate.DUPLICATE:
        return False
    unknown = set(when) - CRITIC_TRIGGERS.keys()
    if unknown:
        raise ValueError(f"unknown critic_required_when entries: {sorted(unknown)}")
    return any(CRITIC_TRIGGERS[name](findings) for name in when)


def _money(amount: Decimal | None, currency: str | None) -> str:
    if amount is None:
        return "unknown amount"
    return f"${amount:,.2f}" if (currency or "USD") == "USD" else f"{amount:,.2f} {currency}"


def deterministic_reasoning(gate: Gate, findings: list[Finding], invoice: Invoice) -> str:
    blocking = [f for f in findings if f.action not in (Action.INFO, Action.WARN, Action.SCRUTINY)]
    shown = blocking or [f for f in findings if f.action != Action.INFO]
    detail = " ".join(f"[{f.code}] {f.message}" for f in shown[:6]) or "No findings."
    return f"Policy gate: {gate}. {detail}"


class ApprovalWorkflow:
    def __init__(self, llm: LLMClient | None, policy: Policy):
        self.llm = llm
        self.policy = policy

    def decide(
        self,
        invoice: Invoice,
        findings: list[Finding],
        gate: Gate,
        investigator_summary: str | None = None,
    ) -> ApprovalResult:
        if gate == Gate.DUPLICATE:
            raise ValueError("duplicates are blocked before approval")
        context = self._context(invoice, findings, gate, investigator_summary)
        messages = [
            {"role": "system", "content": APPROVAL_PROMPT()},
            {"role": "user", "content": context},
        ]
        try:
            output = self._ask_approver(messages)
        except LLMError as e:
            return self._fallback(gate, findings, invoice, str(e))
        result = self._enforce(output, gate)

        if critic_required(findings, gate, self.policy.approval.critic_required_when):
            self._critic_loop(result, output, messages, context, gate)
        return result

    # ---------- LLM calls ----------

    def _ask_approver(self, messages: list[dict]) -> ApprovalOutput:
        if self.llm is None:
            raise LLMError("no LLM configured")
        output, _ = self.llm.structured("approval", messages, ApprovalOutput)
        return output

    def _critic_loop(
        self,
        result: ApprovalResult,
        output: ApprovalOutput,
        messages: list[dict],
        context: str,
        gate: Gate,
    ) -> None:
        max_rounds = self.policy.approval.max_critic_rounds
        objections: list[str] = []
        for round_no in range(1, max_rounds + 1):
            critic_messages = [
                {"role": "system", "content": CRITIC_PROMPT()},
                {
                    "role": "user",
                    "content": f"{context}\n\nYour scope: {CRITIC_SCOPE[gate]}\n\n"
                    f"VP decision: {result.decision}\n"
                    f"Reasoning: {result.reasoning}\n"
                    f"Recommended action: {result.recommended_action}"
                    + (
                        f"\nVendor message: {result.vendor_message}"
                        if result.vendor_message
                        else ""
                    ),
                },
            ]
            try:
                review, _ = self.llm.structured("critic", critic_messages, CriticOutput)
            except LLMError as e:
                result.notes.append(f"critic unavailable: {e}")
                objections = ["The critic could not review this decision."]
                break
            result.critic_rounds.append(
                {
                    "round": round_no,
                    "decision": output.decision,
                    "verdict": review.verdict,
                    "objections": review.objections,
                    "summary": review.summary,
                }
            )
            if review.verdict == "agree" or not review.objections:
                objections = []
                break
            objections = review.objections
            if round_no == max_rounds:
                break
            # The VP revises, addressing each objection.
            messages += [
                {"role": "assistant", "content": output.model_dump_json()},
                {
                    "role": "user",
                    "content": "The internal auditor objects:\n"
                    + "\n".join(f"- {o}" for o in objections)
                    + "\nAddress each objection and give your final decision.",
                },
            ]
            try:
                output = self._ask_approver(messages)
            except LLMError as e:
                result.notes.append(f"approver unavailable during revision: {e}")
                break
            revised = self._enforce(output, gate)
            result.decision = revised.decision
            result.reasoning = revised.reasoning
            result.recommended_action = revised.recommended_action
            result.vendor_message = revised.vendor_message
            result.notes += revised.notes

        result.unresolved_objections = objections
        if not objections:
            return
        blocks = self.policy.approval.approve_requires_no_unresolved_objections
        if blocks and result.decision == Decision.APPROVE:
            result.decision = Decision.HOLD
            result.notes.append("approval blocked: critic objections unresolved")
            result.recommended_action = (
                "Resolve the auditor's objections before paying: " + "; ".join(objections)
            )
        elif result.decision == Decision.RETURN_TO_VENDOR:
            result.notes.append("critic objections unresolved: review before sending to vendor")
            result.recommended_action += (
                " Before sending, check the auditor's objections: " + "; ".join(objections)
            )
        else:
            result.notes.append("critic objections unresolved; decision stands (fixed by gate)")

    # ---------- rules around the agent ----------

    @staticmethod
    def _enforce(output: ApprovalOutput, gate: Gate) -> ApprovalResult:
        decision = Decision(output.decision)
        notes = []
        if decision not in ALLOWED[gate]:
            notes.append(
                f"agent chose {decision}, which the {gate} gate doesn't allow; "
                f"using {FAIL_CLOSED[gate]}"
            )
            decision = FAIL_CLOSED[gate]
        vendor_message = output.vendor_message if decision == Decision.RETURN_TO_VENDOR else None
        return ApprovalResult(
            decision=decision,
            reasoning=output.reasoning,
            recommended_action=output.recommended_action,
            vendor_message=vendor_message,
            notes=notes,
        )

    @staticmethod
    def _fallback(
        gate: Gate, findings: list[Finding], invoice: Invoice, error: str
    ) -> ApprovalResult:
        decision = FAIL_CLOSED[gate]
        action = {
            Decision.HOLD: "Route to a human approver; the approval agent was unavailable.",
            Decision.RETURN_TO_VENDOR: "Return to the vendor with the discrepancies listed.",
            Decision.REJECT: "Do not pay.",
        }[decision]
        return ApprovalResult(
            decision=decision,
            reasoning=deterministic_reasoning(gate, findings, invoice),
            recommended_action=action,
            approver="policy",
            notes=[f"approval agent unavailable: {error}"],
            llm_error=error,
        )

    def _context(
        self,
        invoice: Invoice,
        findings: list[Finding],
        gate: Gate,
        investigator_summary: str | None,
    ) -> str:
        lines = [
            f"{li.raw_name} x{li.quantity} @ {_money(li.unit_price, invoice.currency)}"
            + (f" ({li.note})" if li.note else "")
            for li in invoice.line_items
        ]
        listed = (
            "\n".join(f"- {f.code} [{f.action}] ({f.source}): {f.message}" for f in findings)
            or "- none"
        )
        threshold = self.policy.approval.threshold_usd
        return (
            f"Invoice {invoice.invoice_number or '(no number)'} from "
            f"{invoice.vendor_name or '(no vendor)'}\n"
            f"Total: {_money(invoice.total, invoice.currency)}; "
            f"due {invoice.due_date or invoice.due_date_raw}"
            f"; terms {invoice.payment_terms}\n"
            f"Lines: {'; '.join(lines) or 'none'}\n"
            f"Approval threshold: ${threshold:,.0f}\n\n"
            f"Findings:\n{listed}\n\n"
            f"Investigator summary: {investigator_summary or 'n/a'}\n\n"
            f"Policy gate: {gate}. Allowed decisions: {', '.join(ALLOWED[gate])}."
        )
