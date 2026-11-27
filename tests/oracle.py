"""A perfect fake LLM for testing the eval harness offline.

- Extraction: returns the ground-truth extraction for the document it is given.
- Investigator: reports nothing new.
- Approval agent: the first decision the gate allows (APPROVE on ELIGIBLE).
- Critic: agrees.

With this oracle, every deterministic expectation in evals/expected.yaml should
pass, so a failure in an eval test is a harness or pipeline bug, not model noise.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

from evals.truth import INVOICE_DIR, load_truth
from src.parsers.documents import load_document
from tests.fakes import completion

_DOCUMENT = re.compile(r"<document>\n(.*)\n</document>", re.S)


class Oracle:
    def __init__(self):
        truth = load_truth()
        self.by_text = {load_document(INVOICE_DIR / name)[0]: t for name, t in truth.items()}
        self.roles: list[str] = []

    def __call__(self, request):
        system = request["messages"][0]["content"]
        if request.get("tools"):
            self.roles.append("investigator")
            args = json.dumps({"findings": [], "summary": "Nothing new."})
            call = SimpleNamespace(
                id="r", function=SimpleNamespace(name="report_findings", arguments=args)
            )
            return completion(None, tool_calls=[call])
        if system.startswith("You extract structured data"):
            self.roles.append("extraction")
            text = _DOCUMENT.search(request["messages"][1]["content"]).group(1)
            return completion(self.by_text[text].model_dump_json())
        if system.startswith("You are an internal auditor"):
            self.roles.append("critic")
            return completion(json.dumps({"verdict": "agree", "objections": [], "summary": "ok"}))
        self.roles.append("approval")
        allowed = re.search(r"Allowed decisions: ([A-Z_, ]+)\.", request["messages"][1]["content"])
        decision = allowed.group(1).split(", ")[0]
        return completion(
            json.dumps(
                {
                    "decision": decision,
                    "reasoning": f"{decision} per findings.",
                    "recommended_action": "next step",
                    "vendor_message": "Please correct the invoice."
                    if decision == "RETURN_TO_VENDOR"
                    else None,
                }
            )
        )
