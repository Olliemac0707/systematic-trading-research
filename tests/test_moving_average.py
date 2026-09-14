"""Tests for the simple moving-average crossover strategy."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_research.errors import StrategyError
from trading_research.models import MarketBar, SignalAction
from trading_research.strategies import SimpleMovingAverageCrossover


def make_bars(prices: list[int]) -> list[MarketBar]:
    """Create simple daily bars with valid OHLC values."""

    start = datetime(2025, 1, 1, tzinfo=UTC)
    return [
        MarketBar(
            symbol="TEST",
            timestamp=start + timedelta(days=index),
            open=Decimal(price),
            high=Decimal(price),
            low=Decimal(price),
            close=Decimal(price),
            volume=100,
        )
        for index, price in enumerate(prices)
    ]


def test_strategy_emits_buy_then_sell_on_crossovers() -> None:
    """The example identifies both upward and downward crosses."""

    strategy = SimpleMovingAverageCrossover(short_window=2, long_window=3)

    signals = strategy.generate_signals(make_bars([10, 10, 10, 12, 13, 11, 10]))

    assert [signal.action for signal in signals] == [SignalAction.BUY, SignalAction.SELL]


def test_strategy_does_not_treat_initial_trend_as_crossover() -> None:
    """A pre-existing trend needs an actual cross before producing a signal."""

    strategy = SimpleMovingAverageCrossover(short_window=2, long_window=3)

    assert strategy.generate_signals(make_bars([10, 11, 12, 13])) == ()


def test_strategy_returns_no_signals_without_enough_bars() -> None:
    """An incomplete long window does not produce a recommendation."""

    strategy = SimpleMovingAverageCrossover(short_window=2, long_window=3)

    assert strategy.generate_signals(make_bars([10, 11])) == ()


def test_strategy_rejects_invalid_windows() -> None:
    """The short window must be strictly smaller than the long window."""

    with pytest.raises(ValueError, match="short_window"):
        SimpleMovingAverageCrossover(short_window=3, long_window=3)


def test_strategy_validates_short_mixed_symbol_input() -> None:
    """Insufficient data does not bypass collection-level validation."""

    strategy = SimpleMovingAverageCrossover(short_window=2, long_window=3)
    bars = make_bars([10, 11])
    bars[1] = MarketBar(
        symbol="OTHER",
        timestamp=bars[1].timestamp,
        open=bars[1].open,
        high=bars[1].high,
        low=bars[1].low,
        close=bars[1].close,
        volume=bars[1].volume,
    )

    with pytest.raises(StrategyError, match="one symbol"):
        strategy.generate_signals(bars)


def test_strategy_rejects_non_chronological_bars() -> None:
    """Moving averages require strictly chronological observations."""

    strategy = SimpleMovingAverageCrossover(short_window=2, long_window=3)

    with pytest.raises(StrategyError, match="chronological"):
        strategy.generate_signals(list(reversed(make_bars([10, 11, 12]))))
