"""Pure interfaces and decisions for simulated pre-trade risk controls."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from trading_research.models import PreTradeDecisionReason


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Immutable whole-share approval, reduction, or rejection result."""

    approved: bool
    approved_quantity: int
    reason: str | None = None
    reason_code: PreTradeDecisionReason | None = None

    def __post_init__(self) -> None:
        """Validate decision consistency and safe explanatory text."""

        if not isinstance(self.approved, bool):
            raise TypeError("approved must be a bool")
        if isinstance(self.approved_quantity, bool) or not isinstance(
            self.approved_quantity,
            int,
        ):
            raise TypeError("approved_quantity must be an integer")
        if self.approved_quantity < 0:
            raise ValueError("approved_quantity must be non-negative")
        if self.approved != (self.approved_quantity > 0):
            raise ValueError("approved must match whether approved_quantity is positive")
        if self.reason is not None:
            if not isinstance(self.reason, str):
                raise TypeError("reason must be a string or None")
            if not self.reason.strip() or not self.reason.isprintable():
                raise ValueError("reason must be non-empty printable text")
        if self.reason_code is not None and not isinstance(
            self.reason_code,
            PreTradeDecisionReason,
        ):
            raise TypeError("reason_code must be a PreTradeDecisionReason or None")


class RiskPolicy(Protocol):
    """Evaluate a proposed simulated long entry before execution."""

    def evaluate(
        self,
        *,
        proposed_quantity: int,
        available_cash: Decimal,
        reference_price: Decimal,
        estimated_fill_price: Decimal,
        estimated_commission: Decimal,
        commission_bps: Decimal,
        current_position_quantity: int,
    ) -> RiskDecision:
        """Return a deterministic whole-share decision without side effects."""

        ...
