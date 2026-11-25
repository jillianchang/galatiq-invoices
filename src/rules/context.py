from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.config import Policy
from src.schemas import Invoice, LineItem

Inventory = dict[str, tuple[int, Decimal | None]]  # sku -> (stock, unit_price or None)


@dataclass(frozen=True)
class RuleContext:
    invoice: Invoice
    raw_text: str
    inventory: Inventory
    policy: Policy

    @property
    def currency(self) -> str:
        return self.invoice.currency or "USD"

    def fx_rate(self) -> Decimal | None:
        """Multiplier to USD, or None if the currency has no configured rate."""
        if self.currency == "USD":
            return Decimal(1)
        return self.policy.fx_rates_to_usd.get(self.currency)

    @staticmethod
    def line_amount(line: LineItem) -> Decimal | None:
        if line.amount is not None:
            return line.amount
        if line.quantity is not None and line.unit_price is not None:
            return line.quantity * line.unit_price
        return None

    @property
    def line_sum(self) -> Decimal | None:
        amounts = [self.line_amount(li) for li in self.invoice.line_items]
        if not amounts or any(a is None for a in amounts):
            return None
        return sum(amounts, Decimal(0))  # type: ignore[arg-type]

    @property
    def total_usd(self) -> Decimal | None:
        rate = self.fx_rate()
        if self.invoice.total is None or rate is None:
            return None
        return self.invoice.total * rate

    def money(self, amount: Decimal | None) -> str:
        if amount is None:
            return "n/a"
        if self.currency == "USD":
            return f"${amount:,.2f}"
        return f"{amount:,.2f} {self.currency}"
