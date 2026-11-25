"""Canonical data model shared by every stage of the pipeline.

Every invoice field is optional: a missing value becomes an explicit finding
downstream, never a guess. Money and quantities are Decimal end to end.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Action(StrEnum):
    """What a single finding asks the policy gate to do."""

    DUPLICATE = "DUPLICATE"
    REJECT = "REJECT"
    RETURN = "RETURN"
    HOLD = "HOLD"
    SCRUTINY = "SCRUTINY"  # critic required, no gate change
    WARN = "WARN"  # logged, critic required, no gate change
    INFO = "INFO"  # logged only


class Gate(StrEnum):
    """Hard constraint computed by the deterministic policy gate."""

    DUPLICATE = "DUPLICATE"
    REJECT = "REJECT"
    RETURN_TO_VENDOR = "RETURN_TO_VENDOR"
    HOLD = "HOLD"
    ELIGIBLE = "ELIGIBLE"


# Highest precedence first (D5).
GATE_PRECEDENCE: tuple[Gate, ...] = (
    Gate.DUPLICATE,
    Gate.REJECT,
    Gate.RETURN_TO_VENDOR,
    Gate.HOLD,
    Gate.ELIGIBLE,
)


class Status(StrEnum):
    """Lifecycle status of a processed invoice."""

    RECEIVED = "RECEIVED"
    EXTRACTED = "EXTRACTED"
    VALIDATED = "VALIDATED"
    PAID = "PAID"
    ON_HOLD = "ON_HOLD"
    RETURNED_TO_VENDOR = "RETURNED_TO_VENDOR"
    REJECTED = "REJECTED"
    DUPLICATE_BLOCKED = "DUPLICATE_BLOCKED"
    SUPERSEDED = "SUPERSEDED"


FINAL_STATUSES: frozenset[Status] = frozenset(
    {
        Status.PAID,
        Status.ON_HOLD,
        Status.RETURNED_TO_VENDOR,
        Status.REJECTED,
        Status.DUPLICATE_BLOCKED,
    }
)


class FindingSource(StrEnum):
    RULE = "rule"
    LLM = "llm"


class Finding(BaseModel):
    code: str
    action: Action
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    source: FindingSource = FindingSource.RULE


SourceFormat = Literal["txt", "pdf", "json", "csv", "xml", "email"]


class LineItem(BaseModel):
    raw_name: str | None = None  # as written on the invoice
    sku: str | None = None  # normalized catalog item, if matched
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    amount: Decimal | None = None
    note: str | None = None


class Invoice(BaseModel):
    invoice_number: str | None = None
    vendor_name: str | None = None
    vendor_address: str | None = None
    invoice_date: date | None = None
    invoice_date_raw: str | None = None  # kept so unparseable dates can be reported
    due_date: date | None = None
    due_date_raw: str | None = None
    currency: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Decimal | None = None
    tax_rate: Decimal | None = None
    tax_amount: Decimal | None = None
    shipping: Decimal | None = None
    total: Decimal | None = None
    payment_terms: str | None = None
    notes: str | None = None
    revision: str | None = None
    source_file: str | None = None
    source_format: SourceFormat | None = None
    field_confidence: dict[str, float] = Field(default_factory=dict)
