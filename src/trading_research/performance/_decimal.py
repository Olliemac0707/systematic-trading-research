"""Internal exact aggregation helpers for finite Decimal values."""

from collections.abc import Iterable
from decimal import Decimal, localcontext
from typing import cast

_ZERO = Decimal("0")


def exact_decimal_sum(values: Iterable[Decimal]) -> Decimal:
    """Add finite decimals without context-driven loss of significant digits."""

    snapshot = tuple(values)
    if not snapshot:
        return _ZERO
    if any(not isinstance(value, Decimal) for value in snapshot):
        raise TypeError("exact decimal sums require Decimal values")
    if any(not value.is_finite() for value in snapshot):
        raise ValueError("exact decimal sums require finite values")

    highest_adjusted = max(value.adjusted() for value in snapshot)
    lowest_exponent = min(
        cast(int, value.as_tuple().exponent) for value in snapshot
    )
    carry_digits = len(str(len(snapshot)))
    precision = max(
        28,
        highest_adjusted - lowest_exponent + carry_digits + 3,
    )
    with localcontext() as context:
        context.prec = precision
        return sum(snapshot, _ZERO)
