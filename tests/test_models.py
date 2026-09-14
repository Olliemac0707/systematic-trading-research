"""Tests for shared domain models."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading_research.models import BacktestResult, EquityPoint, MarketBar


def make_bar(**overrides: object) -> MarketBar:
    """Create a valid bar with optional field overrides."""

    values: dict[str, object] = {
        "symbol": "AAPL",
        "timestamp": datetime(2025, 1, 1, tzinfo=UTC),
        "open": Decimal("100"),
        "high": Decimal("102"),
        "low": Decimal("99"),
        "close": Decimal("101"),
        "volume": 1_000,
    }
    values.update(overrides)
    return MarketBar(**values)  # type: ignore[arg-type]


def test_market_bar_normalizes_symbol() -> None:
    """Symbols are normalized at the domain boundary."""

    bar = make_bar(symbol=" aapl ")

    assert bar.symbol == "AAPL"


def test_market_bar_rejects_naive_timestamp() -> None:
    """Ambiguous timestamps are rejected."""

    with pytest.raises(ValueError, match="timezone-aware"):
        make_bar(timestamp=datetime(2025, 1, 1))


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("high", Decimal("100"), "high"),
        ("low", Decimal("102"), "low"),
        ("close", Decimal("NaN"), "finite"),
        ("open", Decimal("Infinity"), "finite"),
        ("volume", -1, "non-negative"),
        ("volume", True, "integer"),
        ("symbol", "AAPL\nINJECT", "supported market-symbol"),
    ],
)
def test_market_bar_rejects_invalid_domain_values(
    field: str, value: object, error: str
) -> None:
    """Invalid provider data fails at the domain boundary with a clear error."""

    with pytest.raises((TypeError, ValueError), match=error):
        make_bar(**{field: value})


def test_market_bar_requires_decimal_prices() -> None:
    """Float inputs cannot silently enter exact financial arithmetic."""

    with pytest.raises(TypeError, match="Decimal"):
        make_bar(open=100.0)


def test_backtest_result_requires_matching_final_equity() -> None:
    """Summary equity cannot disagree with the final curve value."""

    point = EquityPoint(
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        equity=Decimal("100"),
    )

    with pytest.raises(ValueError, match="last equity-curve"):
        BacktestResult(
            initial_cash=Decimal("100"),
            final_cash=Decimal("99"),
            final_equity=Decimal("99"),
            final_position_quantity=0,
            trades=(),
            equity_curve=(point,),
        )
