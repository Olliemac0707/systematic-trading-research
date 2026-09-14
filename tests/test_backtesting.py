"""Tests for deterministic simulated backtesting."""

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_research.backtesting import SimpleBacktestEngine
from trading_research.config import BacktestConfig, PositionSizingMode
from trading_research.errors import BacktestError
from trading_research.models import MarketBar, SignalAction, StrategySignal, TradeSide
from trading_research.risk import (
    AllowAllRiskPolicy,
    CashAllocationSizer,
    FixedQuantitySizer,
    MaximumPositionValuePolicy,
    RiskDecision,
)
from trading_research.strategies import SimpleMovingAverageCrossover, Strategy


class FixedSignalStrategy(Strategy):
    """Return explicit signals for engine validation tests."""

    def __init__(self, signals: Sequence[StrategySignal]) -> None:
        self._signals = tuple(signals)

    def generate_signals(self, bars: Sequence[MarketBar]) -> Sequence[StrategySignal]:
        """Return the configured signals without inspecting bars."""

        return self._signals


class ClearingInputStrategy(Strategy):
    """Attempt to mutate the caller-owned bar list during strategy evaluation."""

    def __init__(self, source: list[MarketBar]) -> None:
        self._source = source

    def generate_signals(self, bars: Sequence[MarketBar]) -> Sequence[StrategySignal]:
        """Clear the original list to exercise the engine's defensive snapshot."""

        self._source.clear()
        return ()


class RecordingSizer:
    """Test sizer that records the current bar inputs used before execution."""

    def __init__(self, quantity: int) -> None:
        self.quantity = quantity
        self.calls: list[tuple[Decimal, Decimal, Decimal, Decimal]] = []

    def calculate_quantity(
        self,
        *,
        available_cash: Decimal,
        reference_price: Decimal,
        commission_bps: Decimal,
        slippage_bps: Decimal,
    ) -> int:
        """Record exact inputs and return the configured proposal."""

        self.calls.append(
            (available_cash, reference_price, commission_bps, slippage_bps)
        )
        return self.quantity


class RecordingRiskPolicy:
    """Test risk policy that records the engine's pre-trade estimate."""

    def __init__(self, approved_quantity: int) -> None:
        self.approved_quantity = approved_quantity
        self.calls: list[tuple[int, Decimal, Decimal, Decimal]] = []

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
        """Record estimates and approve a deterministic reduced quantity."""

        self.calls.append(
            (
                proposed_quantity,
                available_cash,
                reference_price,
                estimated_fill_price,
            )
        )
        return RiskDecision(True, self.approved_quantity, "test quantity reduction")


def make_bars(prices: list[int]) -> list[MarketBar]:
    """Create valid daily bars for an engine test."""

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


def test_engine_simulates_long_only_round_trip() -> None:
    """A buy and sell change cash without any external order submission."""

    bars = make_bars([10, 10, 10, 12, 13, 11, 10, 9])
    strategy = SimpleMovingAverageCrossover(short_window=2, long_window=3)
    config = BacktestConfig(initial_cash=Decimal("1000"), trade_quantity=2)

    result = SimpleBacktestEngine().run(bars, strategy, config)

    assert [trade.side for trade in result.trades] == [TradeSide.BUY, TradeSide.SELL]
    assert [trade.price for trade in result.trades] == [Decimal("13"), Decimal("9")]
    assert result.final_cash == Decimal("992")
    assert result.final_equity == Decimal("992")
    assert result.final_position_quantity == 0
    assert result.total_return == Decimal("-0.008")
    assert len(result.equity_curve) == len(bars)


def test_engine_rejects_empty_input() -> None:
    """An empty dataset reports a clear domain error."""

    strategy = SimpleMovingAverageCrossover(short_window=2, long_window=3)

    with pytest.raises(BacktestError, match="at least one"):
        SimpleBacktestEngine().run([], strategy, BacktestConfig())


def test_engine_executes_signal_on_next_bar() -> None:
    """A close-derived signal cannot fill retrospectively at that close."""

    bars = make_bars([10, 10, 10, 12, 13])
    strategy = SimpleMovingAverageCrossover(short_window=2, long_window=3)
    config = BacktestConfig(initial_cash=Decimal("1000"), trade_quantity=2)

    result = SimpleBacktestEngine().run(bars, strategy, config)

    assert len(result.trades) == 1
    assert result.trades[0].timestamp == bars[4].timestamp
    assert result.trades[0].price == bars[4].open
    assert result.final_position_quantity == 2
    assert result.final_cash == Decimal("974")
    assert result.final_equity == Decimal("1000")


def test_engine_applies_commission_and_adverse_slippage() -> None:
    """Configured costs reduce both sides of a simulated round trip."""

    bars = make_bars([10, 10, 10, 12, 13, 11, 10, 9])
    strategy = SimpleMovingAverageCrossover(short_window=2, long_window=3)
    config = BacktestConfig(
        initial_cash=Decimal("1000"),
        trade_quantity=2,
        commission_bps=Decimal("100"),
        slippage_bps=Decimal("100"),
    )

    result = SimpleBacktestEngine().run(bars, strategy, config)

    assert result.final_equity == Decimal("991.1192")


def test_default_policies_preserve_all_existing_execution_results() -> None:
    """Implicit and explicit compatibility policies are exactly identical."""

    bars = make_bars([10, 10, 10, 12, 13, 11, 10, 9])
    strategy = SimpleMovingAverageCrossover(short_window=2, long_window=3)
    config = BacktestConfig(
        initial_cash=Decimal("1000"),
        trade_quantity=2,
        commission_bps=Decimal("100"),
        slippage_bps=Decimal("100"),
    )

    default_result = SimpleBacktestEngine().run(bars, strategy, config)
    explicit_result = SimpleBacktestEngine().run(
        bars,
        strategy,
        config,
        position_sizer=FixedQuantitySizer(2),
        risk_policy=AllowAllRiskPolicy(),
    )

    assert explicit_result == default_result
    assert [trade.price for trade in default_result.trades] == [
        Decimal("13.13"),
        Decimal("8.91"),
    ]
    assert [trade.commission for trade in default_result.trades] == [
        Decimal("0.2626"),
        Decimal("0.1782"),
    ]
    assert default_result.final_cash == Decimal("991.1192")
    assert default_result.final_equity == Decimal("991.1192")
    assert default_result.reconciliation.ledger.total_slippage_cost == Decimal("0.44")
    assert default_result.reconciliation.cash_difference == Decimal("0")
    assert default_result.reconciliation.reconciliation_difference == Decimal("0")


def test_engine_uses_next_open_for_sizing_risk_and_actual_fill() -> None:
    """The same next-bar reference produces estimated and actual buy fill."""

    bars = make_bars([10, 10, 10, 12, 13])
    strategy = SimpleMovingAverageCrossover(short_window=2, long_window=3)
    config = BacktestConfig(
        initial_cash=Decimal("1000"),
        trade_quantity=10,
        commission_bps=Decimal("100"),
        slippage_bps=Decimal("100"),
    )
    sizer = RecordingSizer(5)
    policy = RecordingRiskPolicy(2)

    result = SimpleBacktestEngine().run(
        bars,
        strategy,
        config,
        position_sizer=sizer,
        risk_policy=policy,
    )

    assert sizer.calls == [
        (Decimal("1000"), Decimal("13"), Decimal("100"), Decimal("100"))
    ]
    assert policy.calls == [
        (5, Decimal("1000"), Decimal("13"), Decimal("13.13"))
    ]
    assert len(result.trades) == 1
    assert result.trades[0].timestamp == bars[4].timestamp
    assert result.trades[0].reference_price == Decimal("13")
    assert result.trades[0].price == policy.calls[0][3] == Decimal("13.13")
    assert result.trades[0].quantity == 2
    assert result.trades[0].commission == Decimal("0.2626")
    assert result.final_cash == Decimal("973.4774")
    assert result.final_equity == Decimal("999.4774")
    assert result.reconciliation.is_reconciled


def test_rejected_risk_decision_creates_no_trade_or_ledger_entry() -> None:
    """A position-value cap can reject an entry before account mutation."""

    bars = make_bars([10, 10, 10, 12, 13])
    strategy = SimpleMovingAverageCrossover(short_window=2, long_window=3)

    result = SimpleBacktestEngine().run(
        bars,
        strategy,
        BacktestConfig(initial_cash=Decimal("1000"), trade_quantity=10),
        risk_policy=MaximumPositionValuePolicy(Decimal("10")),
    )

    assert result.trades == ()
    assert result.reconciliation.ledger.entries == ()
    assert result.final_cash == result.final_equity == Decimal("1000")
    assert result.reconciliation.is_reconciled


def test_zero_sized_entry_stops_before_risk_evaluation() -> None:
    """An allocation unable to buy one share creates no risk call or trade."""

    class FailingRiskPolicy:
        def evaluate(self, **_: object) -> RiskDecision:
            raise AssertionError("risk policy must not run for a zero proposal")

    bars = make_bars([10, 10, 10, 12, 13])
    strategy = SimpleMovingAverageCrossover(short_window=2, long_window=3)

    result = SimpleBacktestEngine().run(
        bars,
        strategy,
        BacktestConfig(initial_cash=Decimal("100"), trade_quantity=1),
        position_sizer=CashAllocationSizer(Decimal("0.01")),
        risk_policy=FailingRiskPolicy(),
    )

    assert result.trades == ()
    assert result.final_cash == result.final_equity == Decimal("100")


def test_engine_builds_configured_sizing_and_risk_policies_by_default() -> None:
    """Serializable settings are honoured even without explicit policy objects."""

    bars = make_bars([10, 10, 10, 12, 13])
    strategy = SimpleMovingAverageCrossover(short_window=2, long_window=3)
    config = BacktestConfig(
        initial_cash=Decimal("1000"),
        position_sizing_mode=PositionSizingMode.CASH_ALLOCATION,
        cash_allocation_ratio=Decimal("0.10"),
        maximum_position_value=Decimal("65"),
    )

    result = SimpleBacktestEngine().run(bars, strategy, config)

    assert len(result.trades) == 1
    assert result.trades[0].quantity == 5
    assert result.final_cash == Decimal("935")
    assert result.final_equity == Decimal("1000")
    assert result.reconciliation.is_reconciled


def test_engine_rejects_signal_outside_backtest_bars() -> None:
    """Strategy/provider mismatches cannot disappear silently."""

    bars = make_bars([10, 11, 12])
    signal = StrategySignal(
        symbol="TEST",
        timestamp=bars[-1].timestamp + timedelta(days=1),
        action=SignalAction.BUY,
    )

    with pytest.raises(BacktestError, match="outside"):
        SimpleBacktestEngine().run(bars, FixedSignalStrategy([signal]), BacktestConfig())


def test_engine_rejects_mixed_symbols() -> None:
    """A single-position backtest cannot combine unrelated instruments."""

    bars = make_bars([10, 11])
    bars[1] = replace(bars[1], symbol="OTHER")

    with pytest.raises(BacktestError, match="one symbol"):
        SimpleBacktestEngine().run(bars, FixedSignalStrategy([]), BacktestConfig())


def test_engine_rejects_non_chronological_bars() -> None:
    """Execution order must be deterministic and chronological."""

    bars = list(reversed(make_bars([10, 11])))

    with pytest.raises(BacktestError, match="chronological"):
        SimpleBacktestEngine().run(bars, FixedSignalStrategy([]), BacktestConfig())


def test_engine_skips_unaffordable_simulated_buy() -> None:
    """A long-only simulation never allows cash to become negative."""

    bars = make_bars([10, 10, 10, 12, 13])
    strategy = SimpleMovingAverageCrossover(short_window=2, long_window=3)
    config = BacktestConfig(initial_cash=Decimal("10"), trade_quantity=1)

    result = SimpleBacktestEngine().run(bars, strategy, config)

    assert result.trades == ()
    assert result.final_cash == Decimal("10")
    assert result.final_position_quantity == 0


def test_engine_snapshots_bars_before_running_strategy() -> None:
    """A strategy cannot invalidate iteration by mutating the caller's list."""

    bars = make_bars([10, 11, 12])
    original_length = len(bars)

    result = SimpleBacktestEngine().run(bars, ClearingInputStrategy(bars), BacktestConfig())

    assert bars == []
    assert len(result.equity_curve) == original_length
