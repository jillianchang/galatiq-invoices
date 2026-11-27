import json

import pytest

from src.agents.approval import ApprovalWorkflow, Decision, allowed_decisions, critic_required
from src.config import load_policy
from src.rules.codes import finding
from src.schemas import Action, FindingSource, Gate, Invoice
from tests.fakes import FakeOpenAI, completion, make_client

POLICY = load_policy()
INVOICE = Invoice(invoice_number="INV-1", vendor_name="Widgets Inc.", total=5000)


def approval(decision, reasoning="r", action="a", vendor_message=None):
    return completion(
        json.dumps(
            {
                "decision": decision,
                "reasoning": reasoning,
                "recommended_action": action,
                "vendor_message": vendor_message,
            }
        )
    )


def critic(verdict, objections=()):
    return completion(
        json.dumps({"verdict": verdict, "objections": list(objections), "summary": "s"})
    )


def workflow(tmp_path, *responses):
    fake = FakeOpenAI(*responses)
    client, _, _ = make_client(tmp_path, fake, mode="live")
    return ApprovalWorkflow(client, POLICY), fake


def test_clean_invoice_approved_without_critic(tmp_path):
    wf, fake = workflow(tmp_path, approval("APPROVE"))
    result = wf.decide(INVOICE, [], Gate.ELIGIBLE)
    assert result.decision == Decision.APPROVE and result.critic_rounds == []
    assert len(fake.requests) == 1 and fake.requests[0]["model"] == "strong-m"
    assert "Allowed decisions: APPROVE, HOLD" in fake.requests[0]["messages"][1]["content"]


def test_agent_cannot_pay_what_the_gate_blocks(tmp_path):
    wf, _ = workflow(tmp_path, approval("APPROVE"), critic("agree"))
    result = wf.decide(INVOICE, [finding("unknown_item", "WidgetC")], Gate.HOLD)
    assert result.decision == Decision.HOLD
    assert "isn't allowed" in result.notes[0]


def test_eligible_garbage_decision_fails_closed(tmp_path):
    wf, _ = workflow(tmp_path, approval("RETURN_TO_VENDOR"))
    assert wf.decide(INVOICE, [], Gate.ELIGIBLE).decision == Decision.HOLD


def test_critic_objection_revision_then_agreement(tmp_path):
    wf, fake = workflow(
        tmp_path,
        approval("APPROVE", reasoning="looks fine"),
        critic("object", ["Total is over the threshold and the reasoning ignores it."]),
        approval("APPROVE", reasoning="Over threshold, but every check passed."),
        critic("agree"),
    )
    result = wf.decide(INVOICE, [finding("over_threshold", "over")], Gate.ELIGIBLE)
    assert result.decision == Decision.APPROVE and result.unresolved_objections == []
    assert [c["verdict"] for c in result.critic_rounds] == ["object", "agree"]
    revision = fake.requests[2]["messages"][-1]["content"]
    assert "auditor objects" in revision and "threshold" in revision


def test_unresolved_objection_blocks_approval(tmp_path):
    wf, _ = workflow(
        tmp_path,
        approval("APPROVE"),
        critic("object", ["Vendor never seen before."]),
        approval("APPROVE"),
        critic("object", ["Still unverified."]),
    )
    result = wf.decide(INVOICE, [finding("over_threshold", "over")], Gate.ELIGIBLE)
    assert result.decision == Decision.HOLD
    assert result.unresolved_objections == ["Still unverified."]
    assert "critic objections unresolved" in result.notes[-1]


def test_return_critic_reviews_vendor_message(tmp_path):
    wf, fake = workflow(
        tmp_path,
        approval("RETURN_TO_VENDOR", vendor_message="Fix the total."),
        critic("agree"),
    )
    result = wf.decide(INVOICE, [finding("total_mismatch", "off by $110")], Gate.RETURN_TO_VENDOR)
    assert result.decision == Decision.RETURN_TO_VENDOR
    assert result.vendor_message == "Fix the total."
    review = fake.requests[1]["messages"][1]["content"]
    assert "Vendor message: Fix the total." in review
    assert "push RETURN_TO_VENDOR to REJECT" in review


def test_return_critic_objection_revises_vendor_message(tmp_path):
    wf, _ = workflow(
        tmp_path,
        approval("RETURN_TO_VENDOR", vendor_message="Total is off by $100."),
        critic("object", ["The discrepancy is $110, not $100."]),
        approval("RETURN_TO_VENDOR", vendor_message="Total is off by $110."),
        critic("agree"),
    )
    result = wf.decide(INVOICE, [finding("total_mismatch", "off by $110")], Gate.RETURN_TO_VENDOR)
    assert result.vendor_message == "Total is off by $110."
    assert [c["verdict"] for c in result.critic_rounds] == ["object", "agree"]


def test_return_unresolved_objection_flags_vendor_message(tmp_path):
    wf, _ = workflow(
        tmp_path,
        approval("RETURN_TO_VENDOR", vendor_message="m1"),
        critic("object", ["Wrong amount."]),
        approval("RETURN_TO_VENDOR", vendor_message="m2"),
        critic("object", ["Still wrong."]),
    )
    result = wf.decide(INVOICE, [finding("total_mismatch", "x")], Gate.RETURN_TO_VENDOR)
    assert result.decision == Decision.RETURN_TO_VENDOR
    assert "review before sending" in result.notes[-1]
    assert "Still wrong." in result.recommended_action


def test_return_critic_can_escalate_to_reject(tmp_path):
    wf, _ = workflow(
        tmp_path,
        approval("RETURN_TO_VENDOR", vendor_message="m"),
        critic("object", ["The bank details changed; this looks like fraud."]),
        approval("REJECT"),
        critic("agree"),
    )
    result = wf.decide(INVOICE, [finding("total_mismatch", "x")], Gate.RETURN_TO_VENDOR)
    assert result.decision == Decision.REJECT and result.vendor_message is None


def test_reject_critic_cannot_overturn(tmp_path):
    wf, fake = workflow(
        tmp_path,
        approval("REJECT", reasoning="Cites the wrong finding."),
        critic("object", ["The reasoning cites stock, but the rejection is for fraud_pattern."]),
        approval("APPROVE", reasoning="Persuaded."),
        critic("object", ["Still wrong."]),
    )
    result = wf.decide(INVOICE, [finding("fraud_pattern", "x")], Gate.REJECT)
    assert result.decision == Decision.REJECT
    assert any("isn't allowed" in n for n in result.notes)
    assert "decision stands" in result.notes[-1]
    assert "rejection is fixed by policy" in fake.requests[1]["messages"][1]["content"]


def test_llm_unavailable_falls_back_to_gate_never_approves(tmp_path):
    wf = ApprovalWorkflow(None, POLICY)
    result = wf.decide(INVOICE, [], Gate.ELIGIBLE)
    assert result.decision == Decision.HOLD and result.approver == "policy"
    rejected = wf.decide(INVOICE, [finding("missing_vendor", "no vendor")], Gate.REJECT)
    assert rejected.decision == Decision.REJECT and "missing_vendor" in rejected.reasoning


def test_critic_required_rules():
    when = POLICY.approval.critic_required_when
    assert not critic_required([], Gate.ELIGIBLE, when)
    assert not critic_required([finding("reprocessed_submission", "x")], Gate.ELIGIBLE, when)
    assert critic_required([finding("near_threshold", "x")], Gate.ELIGIBLE, when)
    assert critic_required([finding("unknown_item", "x")], Gate.HOLD, when)
    assert critic_required([finding("total_mismatch", "x")], Gate.RETURN_TO_VENDOR, when)
    assert critic_required([finding("fraud_pattern", "x")], Gate.REJECT, when)
    assert not critic_required([finding("duplicate", "x")], Gate.DUPLICATE, when)


def test_critic_required_reads_policy():
    near = [finding("near_threshold", "x")]
    assert not critic_required(near, Gate.ELIGIBLE, ["over_threshold"])
    assert critic_required([finding("over_threshold", "x")], Gate.ELIGIBLE, ["over_threshold"])
    assert not critic_required(near, Gate.ELIGIBLE, [])
    with pytest.raises(ValueError):
        critic_required(near, Gate.ELIGIBLE, ["typo"])


def _llm(code, msg="x"):
    return finding(code, msg, source=FindingSource.LLM)


def test_hold_blocked_only_by_agent_findings_cannot_be_rejected(tmp_path):
    """The approval agent and critic can't get around the corroboration rule (D3)."""
    wf, fake = workflow(
        tmp_path,
        approval("HOLD"),
        critic("object", ["The address is fictional; reject it."]),
        approval("REJECT"),
        critic("agree"),
    )
    uncorroborated = finding("suspected_fraud", "x", action=Action.HOLD, source=FindingSource.LLM)
    findings = [uncorroborated, finding("over_threshold", "x")]
    result = wf.decide(INVOICE, findings, Gate.HOLD)
    assert result.decision == Decision.HOLD
    assert "Allowed decisions: HOLD." in fake.requests[0]["messages"][1]["content"]
    assert "can't be rejected without corroboration" in fake.requests[1]["messages"][1]["content"]


def test_allowed_decisions():
    rule_hold = finding("unknown_item", "WidgetC")
    agent_hold = _llm("suspicious_vendor_address")
    assert allowed_decisions(Gate.HOLD, [agent_hold]) == (Decision.HOLD,)
    assert allowed_decisions(Gate.HOLD, [agent_hold, rule_hold]) == (Decision.HOLD, Decision.REJECT)
    assert allowed_decisions(Gate.ELIGIBLE, [agent_hold]) == (Decision.APPROVE, Decision.HOLD)
