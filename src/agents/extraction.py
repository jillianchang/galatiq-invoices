"""Extraction agent: unstructured invoice text -> Invoice, with a grounding-based
self-correction loop (D2).

Each attempt's output is checked against the source document. Values that
don't appear in the document go back to the model as specific errors. Math
inconsistencies get one re-read and are then accepted as written, because a
real invoice can be wrong (validation flags it). If grounding still fails
after the last attempt, extraction fails closed (D16).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from pydantic import BaseModel

from src.config import ROOT
from src.grounding import check_grounding, check_math
from src.llm.client import LLMClient, LLMError, StructuredOutputError
from src.schemas import Invoice, LineItem, SourceFormat

PROMPT_PATH = ROOT / "prompts" / "extraction.md"


class ExtractedLine(BaseModel):
    name: str
    quantity: float | None
    unit_price: float | None
    amount: float | None


class ExtractedInvoice(BaseModel):
    """What the model returns. Dates stay raw text: parsing is deterministic."""

    invoice_number: str | None
    vendor_name: str | None
    vendor_address: str | None
    invoice_date_raw: str | None
    due_date_raw: str | None
    currency: str | None
    line_items: list[ExtractedLine]
    subtotal: float | None
    tax_rate_percent: float | None
    tax_amount: float | None
    shipping: float | None
    total: float | None
    payment_terms: str | None
    notes: str | None


@dataclass
class ExtractionOutcome:
    ok: bool
    extracted: ExtractedInvoice | None
    attempts: int
    errors: list[str] = field(default_factory=list)  # errors from the final attempt
    math_issues: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)  # per-attempt summary for traces


_CENTS = Decimal("0.01")


def _dec(value: float | None) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _money(value: float | None) -> Decimal | None:
    d = _dec(value)
    return None if d is None else d.quantize(_CENTS)


def _qty(value: float | None) -> Decimal | None:
    """10.0 -> Decimal('10'); fractional quantities are kept as written."""
    d = _dec(value)
    return None if d is None else (d.to_integral_value() if d == d.to_integral_value() else d)


def to_invoice(
    extracted: ExtractedInvoice, source_file: str | None, source_format: SourceFormat
) -> Invoice:
    rate = _dec(extracted.tax_rate_percent)
    return Invoice(
        invoice_number=extracted.invoice_number,
        vendor_name=extracted.vendor_name,
        vendor_address=extracted.vendor_address,
        invoice_date_raw=extracted.invoice_date_raw,
        due_date_raw=extracted.due_date_raw,
        currency=extracted.currency,
        line_items=[
            LineItem(
                raw_name=line.name,
                quantity=_qty(line.quantity),
                unit_price=_money(line.unit_price),
                amount=_money(line.amount),
            )
            for line in extracted.line_items
        ],
        subtotal=_money(extracted.subtotal),
        tax_rate=None if rate is None else rate / 100,
        tax_amount=_money(extracted.tax_amount),
        shipping=_money(extracted.shipping),
        total=_money(extracted.total),
        payment_terms=extracted.payment_terms,
        notes=extracted.notes,
        source_file=source_file,
        source_format=source_format,
    )


class ExtractionAgent:
    def __init__(self, llm: LLMClient, max_attempts: int = 3):
        self.llm = llm
        self.max_attempts = max_attempts
        self.system_prompt = PROMPT_PATH.read_text()

    def extract(self, text: str) -> ExtractionOutcome:
        # Stable prefix (system prompt, then document) keeps provider prompt caching effective.
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"<document>\n{text}\n</document>"},
        ]
        last_grounded: ExtractedInvoice | None = None
        math_rechecked = False
        errors: list[str] = []
        math: list[str] = []
        history: list[dict] = []

        for attempt in range(1, self.max_attempts + 1):
            try:
                extracted, response = self.llm.structured("extraction", messages, ExtractedInvoice)
            except StructuredOutputError as e:
                errors = [f"Output was not valid JSON for the schema: {e}"]
                history.append({"attempt": attempt, "errors": errors})
                messages += [
                    {"role": "assistant", "content": e.raw or ""},
                    {"role": "user", "content": _feedback(errors)},
                ]
                continue
            except LLMError as e:
                errors = [f"LLM unavailable: {e}"]
                history.append({"attempt": attempt, "errors": errors})
                break

            errors = check_grounding(extracted, text)
            math = check_math(extracted)
            history.append(
                {
                    "attempt": attempt,
                    "errors": errors,
                    "math_issues": math,
                    "cache_hit": response.cache_hit,
                    "tokens": response.usage.get("total_tokens"),
                }
            )
            if errors:
                messages += [
                    {"role": "assistant", "content": response.content or ""},
                    {"role": "user", "content": _feedback(errors)},
                ]
                continue

            last_grounded = extracted
            if math and not math_rechecked and attempt < self.max_attempts:
                math_rechecked = True
                messages += [
                    {"role": "assistant", "content": response.content or ""},
                    {"role": "user", "content": _math_feedback(math)},
                ]
                continue
            return ExtractionOutcome(True, extracted, attempt, [], math, history)

        if last_grounded is not None:  # grounded earlier; a later re-read failed
            return ExtractionOutcome(
                True, last_grounded, len(history), [], check_math(last_grounded), history
            )
        return ExtractionOutcome(False, None, len(history), errors, math, history)


def _feedback(errors: list[str]) -> str:
    bullets = "\n".join(f"- {e}" for e in errors)
    return (
        "Your extraction has problems:\n"
        f"{bullets}\n"
        "Re-read the document and return the corrected JSON object."
    )


def _math_feedback(issues: list[str]) -> str:
    bullets = "\n".join(f"- {i}" for i in issues)
    return (
        "The extracted numbers are internally inconsistent:\n"
        f"{bullets}\n"
        "Re-read the document carefully, including any line items you may have missed. "
        "If the document itself is inconsistent, keep every value exactly as written; "
        "do not fix the vendor's math. Return the JSON object again."
    )
