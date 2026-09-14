"""Neutral Decimal-only cost helpers for simulated order evaluation."""

from decimal import Decimal

_BASIS_POINTS = Decimal("10000")
_ZERO = Decimal("0")


def calculate_buy_fill_price(
    reference_price: Decimal,
    slippage_bps: Decimal,
) -> Decimal:
    """Apply the existing adverse buy-slippage formula without quantisation."""

    return reference_price * (Decimal("1") + (slippage_bps / _BASIS_POINTS))


def calculate_sell_fill_price(
    reference_price: Decimal,
    slippage_bps: Decimal,
) -> Decimal:
    """Apply the existing adverse sell-slippage formula without quantisation."""

    return reference_price * (Decimal("1") - (slippage_bps / _BASIS_POINTS))


def calculate_proportional_commission(
    fill_price: Decimal,
    quantity: int,
    commission_bps: Decimal,
) -> Decimal:
    """Apply the existing proportional commission formula without quantisation."""

    return fill_price * quantity * (commission_bps / _BASIS_POINTS)


def calculate_buy_cost(
    fill_price: Decimal,
    quantity: int,
    commission_bps: Decimal,
) -> Decimal:
    """Return simulated buy notional plus its proportional commission."""

    return (fill_price * quantity) + calculate_proportional_commission(
        fill_price,
        quantity,
        commission_bps,
    )


def largest_affordable_quantity(
    *,
    budget: Decimal,
    fill_price: Decimal,
    commission_bps: Decimal,
) -> int:
    """Return the largest whole-share buy whose estimated cost fits a budget."""

    if budget <= _ZERO:
        return 0
    one_share_cost = calculate_buy_cost(fill_price, 1, commission_bps)
    if one_share_cost <= _ZERO:
        raise ValueError("estimated one-share acquisition cost must be positive")

    quantity = int(budget // one_share_cost)
    while quantity > 0 and calculate_buy_cost(
        fill_price,
        quantity,
        commission_bps,
    ) > budget:
        quantity -= 1
    while calculate_buy_cost(
        fill_price,
        quantity + 1,
        commission_bps,
    ) <= budget:
        quantity += 1
    return quantity
