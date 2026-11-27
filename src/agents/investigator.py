"""Investigator agent: LLM judgment with read-only tools, bounded rounds (PLAN.md section 4).

It can only ADD findings (D3). It picks a code from a fixed list; the action
comes from the code registry, never from the model. Each finding must quote
evidence that actually appears in the document or invoice data, or it is
dropped: the same grounding idea as extraction (D2).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from pydantic import BaseModel, ValidationError

from src.config import ROOT
from src.llm.client import LLMClient, LLMError, ToolCallRejected
from src.rules.codes import finding
from src.rules.context import RuleContext
from src.schemas import Action, Finding, FindingSource
from src.tools.db_tools import TOOL_SPECS, InvestigatorTools

PROMPT_PATH = ROOT / "prompts" / "investigator.md"

# Independent signals that can corroborate an LLM fraud suspicion.
RED_FLAG_CODES = frozenset(
    {
        "payment_pressure",
        "prompt_injection",
        "vendor_identity_change",
        "bank_detail_change",
        "suspicious_vendor_address",
        "no_catalog_items",
        "fraud_pattern",
        "unparseable_due_date",
    }
)

InvestigatorCode = Literal[
    "suspected_fraud",
    "suspicious_vendor_address",
    "bank_detail_change",
    "vendor_identity_change",
    "payment_pressure",
    "prompt_injection",
    "other_concern",
]


class InvestigatorFinding(BaseModel):
    code: InvestigatorCode
    message: str
    evidence_quote: str


class InvestigatorReport(BaseModel):
    findings: list[InvestigatorFinding]
    summary: str


# Written out flat (no $defs/$ref): smaller models fill nested references unreliably.
REPORT_TOOL = {
    "type": "function",
    "function": {
        "name": "report_findings",
        "description": "Report your findings and finish the investigation.",
        "parameters": {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "description": "New concerns only; empty if none.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "enum": list(get_args(InvestigatorCode))},
                            "message": {"type": "string"},
                            "evidence_quote": {
                                "type": "string",
                                "description": "Exact quote from the document.",
                            },
                        },
                        "required": ["code", "message", "evidence_quote"],
                        "additionalProperties": False,
                    },
                },
                "summary": {"type": "string"},
            },
            "required": ["findings", "summary"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class InvestigationOutcome:
    findings: list[Finding]
    summary: str | None = None
    rounds: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)  # ungrounded or duplicate
    error: str | None = None


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


class InvestigatorAgent:
    def __init__(self, llm: LLMClient, tools: InvestigatorTools, max_rounds: int = 3):
        self.llm = llm
        self.tools = tools
        self.max_rounds = max_rounds
        self.system_prompt = PROMPT_PATH.read_text()

    def investigate(self, ctx: RuleContext, existing: list[Finding]) -> InvestigationOutcome:
        invoice_json = ctx.invoice.model_dump_json(
            exclude_defaults=True, exclude={"field_confidence"}
        )
        already = "\n".join(f"- {f.code} ({f.action}): {f.message}" for f in existing) or "- none"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"<document>\n{ctx.raw_text}\n</document>\n\n"
                f"Extracted invoice:\n{invoice_json}\n\nFindings already raised:\n{already}",
            },
        ]
        outcome = InvestigationOutcome(findings=[])
        for round_no in range(1, self.max_rounds + 1):
            outcome.rounds = round_no
            last = round_no == self.max_rounds
            try:
                response = self.llm.complete(
                    "investigator",
                    messages,
                    tools=[REPORT_TOOL] if last else [*TOOL_SPECS, REPORT_TOOL],
                    tool_choice="required" if last else "auto",
                )
            except ToolCallRejected as e:
                # The model's tool call didn't match the schema: show it the error, try again.
                outcome.tool_calls.append({"name": "rejected_tool_call", "error": str(e)[:500]})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Your tool call was rejected: {str(e)[:500]}\n"
                        "Call report_findings again with valid arguments: each finding needs "
                        "code, message, and evidence_quote.",
                    }
                )
                continue
            except LLMError as e:
                outcome.error = str(e)
                break

            # gpt-oss sometimes prefixes tool names with its native "functions." namespace.
            calls = [
                {**c, "name": c["name"].removeprefix("functions.")} for c in response.tool_calls
            ]
            report_call = next((c for c in calls if c["name"] == "report_findings"), None)
            if report_call or not calls:
                payload = report_call["arguments"] if report_call else (response.content or "")
                try:
                    report = InvestigatorReport.model_validate_json(payload)
                except ValidationError as e:
                    if last:
                        outcome.error = f"invalid report: {e}"
                        break
                    messages += [
                        {"role": "assistant", "content": response.content or ""},
                        {"role": "user", "content": "Finish by calling report_findings."},
                    ]
                    continue
                self._accept(report, ctx, existing, outcome)
                return outcome

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]},
                        }
                        for c in calls
                    ],
                }
            )
            for c in calls:
                result = self.tools.call(c["name"], c["arguments"])
                outcome.tool_calls.append(
                    {"name": c["name"], "arguments": c["arguments"], "result": result}
                )
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})

        if outcome.error is None:
            outcome.error = f"no report after {self.max_rounds} rounds"
        outcome.findings.append(
            finding(
                "investigation_incomplete",
                "The investigator agent could not finish; only deterministic checks apply.",
                {"error": outcome.error},
                source=FindingSource.LLM,
            )
        )
        return outcome

    @staticmethod
    def _accept(
        report: InvestigatorReport,
        ctx: RuleContext,
        existing: list[Finding],
        outcome: InvestigationOutcome,
    ) -> None:
        outcome.summary = report.summary
        haystack = _squash(ctx.raw_text + json.dumps(ctx.invoice.model_dump(mode="json")))
        seen = {f.code for f in existing}
        for f in report.findings:
            quote = _squash(f.evidence_quote)
            if not quote or quote not in haystack:
                outcome.dropped.append({**f.model_dump(), "reason": "evidence not in document"})
                continue
            if f.code in seen:
                outcome.dropped.append({**f.model_dump(), "reason": "already raised"})
                continue
            seen.add(f.code)
            outcome.findings.append(
                finding(f.code, f.message, {"quote": f.evidence_quote}, source=FindingSource.LLM)
            )

        # A fraud REJECT from judgment alone needs a second, independent red flag.
        # Uncorroborated, it is held for a human instead of rejected outright.
        # Likewise an implausible address alone is flagged for the reviewer but
        # doesn't block payment (1004's "742 Evergreen Terrace" is a real order).
        codes = {g.code for g in [*existing, *outcome.findings]}
        flags = codes & RED_FLAG_CODES
        address_corroborated = bool(flags - {"suspicious_vendor_address"}) or (
            "suspected_fraud" in codes
        )
        for i, f in enumerate(outcome.findings):
            if f.code == "suspicious_vendor_address" and not address_corroborated:
                outcome.findings[i] = f.model_copy(
                    update={
                        "action": Action.WARN,
                        "message": f"{f.message} (No other red flag, so flagged for the "
                        "reviewer rather than blocking payment.)",
                        "evidence": {**f.evidence, "corroborated": False},
                    }
                )
        for i, f in enumerate(outcome.findings):
            if f.code == "suspected_fraud" and not flags:
                outcome.findings[i] = f.model_copy(
                    update={
                        "action": Action.HOLD,
                        "message": f"{f.message} (Uncorroborated by any other red flag, so held "
                        "for review rather than rejected.)",
                        "evidence": {**f.evidence, "corroborated": False},
                    }
                )
