"""Replaceable whole-share position sizing for simulated entries."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from trading_research.simulation import (
    calculate_buy_fill_price,
    largest_affordable_quantity,
)

_ONE = Decimal("1")
_ZERO = Decimal("0")


class PositionSizer(Protocol):
    """Calculate an intended whole-share quantity before risk evaluation."""

    def calculate_quantity(
        self,
        *,
        available_cash: Decimal,
        reference_price: Decimal,
        commission_bps: Decimal,
        slippage_bps: Decimal,
    ) -> int:
        """Return a non-negative whole-share proposal without side effects."""

        ...


@dataclass(frozen=True, slots=True)
class FixedQuantitySizer:
    """Always propose one validated fixed whole-share quantity."""

    quantity: int

    def __post_init__(self) -> None:
        """Require the existing positive whole-share policy."""

        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise TypeError("quantity must be an integer")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")

    def calculate_quantity(
        self,
        *,
        available_cash: Decimal,
        reference_price: Decimal,
        commission_bps: Decimal,
        slippage_bps: Decimal,
    ) -> int:
        """Return the configured quantity without inspecting future data."""

        _validate_sizing_inputs(
            available_cash=available_cash,
            reference_price=reference_price,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
        )
        return self.quantity


@dataclass(frozen=True, slots=True)
class CashAllocationSizer:
    """Spend at most a selected fraction of current cash on whole shares."""

    allocation_ratio: Decimal

    def __post_init__(self) -> None:
        """Require an exact ratio in the inclusive interval ``(0, 1]``."""

        _require_finite_decimal(self.allocation_ratio, "allocation_ratio")
        if not (_ZERO < self.allocation_ratio <= _ONE):
            raise ValueError("allocation_ratio must be greater than 0 and at most 1")

    def calculate_quantity(
        self,
        *,
        available_cash: Decimal,
        reference_price: Decimal,
        commission_bps: Decimal,
        slippage_bps: Decimal,
    ) -> int:
        """Return the largest whole quantity within the selected cash budget."""

        _validate_sizing_inputs(
            available_cash=available_cash,
            reference_price=reference_price,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
        )
        budget = available_cash * self.allocation_ratio
        estimated_fill_price = calculate_buy_fill_price(
            reference_price,
            slippage_bps,
        )
        return largest_affordable_quantity(
            budget=budget,
            fill_price=estimated_fill_price,
            commission_bps=commission_bps,
        )


def _validate_sizing_inputs(
    *,
    available_cash: Decimal,
    reference_price: Decimal,
    commission_bps: Decimal,
    slippage_bps: Decimal,
) -> None:
    """Validate exact current-state inputs shared by all sizing policies."""

    values = (
        ("available_cash", available_cash),
        ("reference_price", reference_price),
        ("commission_bps", commission_bps),
        ("slippage_bps", slippage_bps),
    )
    for name, value in values:
        _require_finite_decimal(value, name)
    if available_cash < _ZERO:
        raise ValueError("available_cash must be non-negative")
    if reference_price <= _ZERO:
        raise ValueError("reference_price must be positive")
    if commission_bps < _ZERO or slippage_bps < _ZERO:
        raise ValueError("commission_bps and slippage_bps must be non-negative")


def _require_finite_decimal(value: Decimal, name: str) -> None:
    """Require exact finite Decimal policy configuration and inputs."""

    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")
