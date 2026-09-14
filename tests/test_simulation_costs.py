"""Regression tests for shared Decimal-only simulated execution costs."""

from decimal import Decimal
from typing import Never

import pytest

from trading_research.simulation import (
    calculate_buy_cost,
    calculate_buy_fill_price,
    calculate_proportional_commission,
    calculate_sell_fill_price,
    largest_affordable_quantity,
)


def test_extracted_helpers_preserve_existing_fill_and_commission_values() -> None:
    """Known engine prices and commissions remain exactly unchanged."""

    buy_fill = calculate_buy_fill_price(Decimal("13"), Decimal("100"))
    sell_fill = calculate_sell_fill_price(Decimal("9"), Decimal("100"))

    assert buy_fill == Decimal("13.13")
    assert sell_fill == Decimal("8.91")
    assert calculate_proportional_commission(
        buy_fill,
        2,
        Decimal("100"),
    ) == Decimal("0.2626")
    assert calculate_proportional_commission(
        sell_fill,
        2,
        Decimal("100"),
    ) == Decimal("0.1782")


def test_helpers_preserve_decimal_operation_order_without_quantisation() -> None:
    """Fractional inputs retain the exact result of the original expressions."""

    reference_price = Decimal("12.34567")
    slippage_bps = Decimal("7.5")
    commission_bps = Decimal("3.25")
    quantity = 17

    buy_fill = calculate_buy_fill_price(reference_price, slippage_bps)
    sell_fill = calculate_sell_fill_price(reference_price, slippage_bps)
    commission = calculate_proportional_commission(
        buy_fill,
        quantity,
        commission_bps,
    )

    assert buy_fill == reference_price * (
        Decimal("1") + slippage_bps / Decimal("10000")
    )
    assert sell_fill == reference_price * (
        Decimal("1") - slippage_bps / Decimal("10000")
    )
    assert commission == (
        buy_fill * quantity * (commission_bps / Decimal("10000"))
    )
    buy_exponent = buy_fill.as_tuple().exponent
    commission_exponent = commission.as_tuple().exponent
    assert isinstance(buy_exponent, int) and buy_exponent < -2
    assert isinstance(commission_exponent, int) and commission_exponent < -2


def test_affordability_uses_the_same_fill_and_commission_helpers() -> None:
    """The largest approved quantity fits while the next whole share does not."""

    fill_price = calculate_buy_fill_price(Decimal("100"), Decimal("100"))
    quantity = largest_affordable_quantity(
        budget=Decimal("250"),
        fill_price=fill_price,
        commission_bps=Decimal("100"),
    )

    assert quantity == 2
    assert calculate_buy_cost(fill_price, quantity, Decimal("100")) <= Decimal("250")
    assert calculate_buy_cost(fill_price, quantity + 1, Decimal("100")) > Decimal("250")


def test_cost_helpers_never_convert_through_float(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fill, commission, and affordability arithmetic stays Decimal-only."""

    def reject_float(*_: object, **__: object) -> Never:
        raise AssertionError("float conversion is forbidden")

    monkeypatch.setattr("builtins.float", reject_float)

    fill_price = calculate_buy_fill_price(Decimal("99.95"), Decimal("2.5"))
    assert calculate_proportional_commission(
        fill_price,
        7,
        Decimal("1.25"),
    ) == fill_price * 7 * (Decimal("1.25") / Decimal("10000"))
