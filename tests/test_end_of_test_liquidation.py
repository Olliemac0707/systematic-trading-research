"""Tests for configurable final-position treatment in historical backtests."""

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_research.backtesting import SimpleBacktestEngine
from trading_research.config import BacktestConfig, EndOfTestPolicy
from trading_research.models import (
    BacktestResult,
    ExecutionReason,
    MarketBar,
    SimulatedTrade,
    StrategySignal,
    TradeSide,
)
from trading_research.performance import (
    calculate_performance,
    calculate_trade_statistics,
    reconstruct_closed_trades,
)
from trading_research.reporting import format_backtest_report
from trading_research.risk import MaximumPositionValuePolicy
from trading_research.simulation import (
    calculate_proportional_commission,
    calculate_sell_fill_price,
)
from trading_research.strategies import SimpleMovingAverageCrossover, Strategy


class NoSignalStrategy(Strategy):
    """Produce no entries or exits."""

    def generate_signals(
        self,
        bars: Sequence[MarketBar],
    ) -> Sequence[StrategySignal]:
        """Return no signals for any validated market bars."""

        return ()


def make_bars(prices: list[int]) -> list[MarketBar]:
    """Create exact daily OHLC bars with matching prices."""

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


def open_position_config(
    policy: EndOfTestPolicy = EndOfTestPolicy.HOLD,
) -> BacktestConfig:
    """Return the exact cost configuration used by final-position tests."""

    return BacktestConfig(
        initial_cash=Decimal("1000"),
        trade_quantity=2,
        commission_bps=Decimal("100"),
        slippage_bps=Decimal("100"),
        end_of_test_policy=policy,
    )


def run_open_position(policy: EndOfTestPolicy) -> BacktestResult:
    """Run an SMA case whose entry executes on the final market bar."""

    bars = make_bars([10, 10, 10, 12, 13])
    return SimpleBacktestEngine().run(
        bars,
        SimpleMovingAverageCrossover(short_window=2, long_window=3),
        open_position_config(policy),
    )


def test_default_and_explicit_hold_are_identical_regressions() -> None:
    """The omitted policy preserves every existing open-position value."""

    bars = make_bars([10, 10, 10, 12, 13])
    strategy = SimpleMovingAverageCrossover(short_window=2, long_window=3)
    default_result = SimpleBacktestEngine().run(
        bars,
        strategy,
        BacktestConfig(
            initial_cash=Decimal("1000"),
            trade_quantity=2,
            commission_bps=Decimal("100"),
            slippage_bps=Decimal("100"),
        ),
    )
    explicit_result = SimpleBacktestEngine().run(
        bars,
        strategy,
        open_position_config(EndOfTestPolicy.HOLD),
    )

    assert BacktestConfig().end_of_test_policy is EndOfTestPolicy.HOLD
    assert explicit_result == default_result
    assert len(default_result.trades) == 1
    assert default_result.trades[0].price == Decimal("13.13")
    assert default_result.trades[0].commission == Decimal("0.2626")
    assert default_result.trades[0].execution_reason is ExecutionReason.STRATEGY_SIGNAL
    assert default_result.final_cash == Decimal("973.4774")
    assert default_result.final_equity == Decimal("999.4774")
    assert default_result.final_position_quantity == 2
    assert default_result.reconciliation.remaining_position_value == Decimal("26")
    assert default_result.reconciliation.cash_difference == Decimal("0")
    assert default_result.reconciliation.reconciliation_difference == Decimal("0")

    performance = calculate_performance(default_result)
    statistics = calculate_trade_statistics(default_result)
    assert performance.unrealised_pnl == Decimal("-0.26")
    assert performance.gross_realised_pnl == Decimal("0")
    assert statistics.completed_trade_count == 0
    assert reconstruct_closed_trades(default_result) == ()


def test_liquidation_uses_final_close_and_existing_exact_cost_helpers() -> None:
    """A final long is fully sold at a slipped final-close synthetic fill."""

    bars = make_bars([10, 10, 10, 12, 13])
    result = SimpleBacktestEngine().run(
        bars,
        SimpleMovingAverageCrossover(short_window=2, long_window=3),
        open_position_config(EndOfTestPolicy.LIQUIDATE),
    )

    expected_fill = calculate_sell_fill_price(Decimal("13"), Decimal("100"))
    expected_commission = calculate_proportional_commission(
        expected_fill,
        2,
        Decimal("100"),
    )
    assert len(result.trades) == 2
    liquidation = result.trades[-1]
    assert liquidation.side is TradeSide.SELL
    assert liquidation.quantity == 2
    assert liquidation.timestamp == bars[-1].timestamp
    assert liquidation.reference_price == bars[-1].close == Decimal("13")
    assert liquidation.price == expected_fill == Decimal("12.87")
    assert liquidation.commission == expected_commission == Decimal("0.2574")
    assert (
        liquidation.execution_reason
        is ExecutionReason.END_OF_TEST_LIQUIDATION
    )
    assert result.was_end_of_test_liquidated
    assert result.final_cash == result.final_equity == Decimal("998.9600")
    assert result.final_position_quantity == 0
    assert result.reconciliation.remaining_position_value == Decimal("0")
    assert result.reconciliation.ledger.total_commissions == Decimal("0.5200")
    assert result.reconciliation.ledger.total_slippage_cost == Decimal("0.52")
    assert result.reconciliation.cash_difference == Decimal("0")
    assert result.reconciliation.reconciliation_difference == Decimal("0")

    performance = calculate_performance(result)
    closed_trades = reconstruct_closed_trades(result)
    statistics = calculate_trade_statistics(result)
    assert performance.open_position_market_value == Decimal("0")
    assert performance.unrealised_pnl == Decimal("0")
    assert performance.gross_realised_pnl == Decimal("-0.52")
    assert performance.net_profit == Decimal("-1.0400")
    assert len(closed_trades) == 1
    assert closed_trades[0].holding_period == timedelta(0)
    assert (
        closed_trades[0].exit_execution_reason
        is ExecutionReason.END_OF_TEST_LIQUIDATION
    )
    assert statistics.completed_trade_count == 1
    assert statistics.losing_trade_count == 1
    assert statistics.net_closed_trade_pnl == Decimal("-1.0400")


def test_reports_state_hold_and_actual_synthetic_liquidation() -> None:
    """Reports explain both the configured convention and an actual final exit."""

    hold_result = run_open_position(EndOfTestPolicy.HOLD)
    liquidation_result = run_open_position(EndOfTestPolicy.LIQUIDATE)

    hold_report = format_backtest_report(hold_result)
    liquidation_report = format_backtest_report(
        liquidation_result,
        include_closed_trades=True,
    )

    assert "End-of-test policy:                Hold open position" in hold_report
    assert "Open position remains:             Yes" in hold_report
    assert "Final open position was synthetically liquidated" not in hold_report
    assert "End-of-test policy:                Liquidate at final close" in liquidation_report
    assert "Open position remains:             No" in liquidation_report
    assert "No open position" in liquidation_report
    assert (
        "Final open position was synthetically liquidated at the final bar close."
        in liquidation_report
    )
    assert "End-of-test liquidation" in liquidation_report


def test_liquidate_policy_does_nothing_when_position_is_already_flat() -> None:
    """A normal final-bar exit is never duplicated by the end policy."""

    bars = make_bars([10, 10, 10, 12, 13, 11, 10, 9])
    result = SimpleBacktestEngine().run(
        bars,
        SimpleMovingAverageCrossover(short_window=2, long_window=3),
        BacktestConfig(
            initial_cash=Decimal("1000"),
            trade_quantity=2,
            end_of_test_policy=EndOfTestPolicy.LIQUIDATE,
        ),
    )

    assert [trade.side for trade in result.trades] == [TradeSide.BUY, TradeSide.SELL]
    assert all(
        trade.execution_reason is ExecutionReason.STRATEGY_SIGNAL
        for trade in result.trades
    )
    assert result.trades[-1].timestamp == bars[-1].timestamp
    assert not result.was_end_of_test_liquidated
    assert result.final_cash == result.final_equity == Decimal("992")


def test_no_trade_and_zero_approved_entry_create_no_liquidation() -> None:
    """Only an actually open long can create a synthetic final sell."""

    bars = make_bars([10, 10, 10, 12, 13])
    no_trade = SimpleBacktestEngine().run(
        bars,
        NoSignalStrategy(),
        BacktestConfig(end_of_test_policy=EndOfTestPolicy.LIQUIDATE),
    )
    rejected_entry = SimpleBacktestEngine().run(
        bars,
        SimpleMovingAverageCrossover(short_window=2, long_window=3),
        BacktestConfig(
            initial_cash=Decimal("1000"),
            trade_quantity=10,
            end_of_test_policy=EndOfTestPolicy.LIQUIDATE,
        ),
        risk_policy=MaximumPositionValuePolicy(Decimal("10")),
    )

    assert no_trade.trades == rejected_entry.trades == ()
    assert no_trade.final_position_quantity == rejected_entry.final_position_quantity == 0
    assert not no_trade.was_end_of_test_liquidated
    assert not rejected_entry.was_end_of_test_liquidated


def test_liquidation_runs_are_deterministic_and_final_prices_stay_validated() -> None:
    """Equivalent runs match exactly and invalid final closes cannot enter them."""

    first = run_open_position(EndOfTestPolicy.LIQUIDATE)
    second = run_open_position(EndOfTestPolicy.LIQUIDATE)

    assert first == second
    with pytest.raises(ValueError, match="positive"):
        replace(make_bars([10])[-1], close=Decimal("0"))


def test_synthetic_reason_is_validated_and_only_allowed_for_sells() -> None:
    """Synthetic identity is explicit, immutable, and cannot label an entry."""

    with pytest.raises(ValueError, match="must be a sell"):
        SimulatedTrade(
            symbol="TEST",
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            side=TradeSide.BUY,
            quantity=1,
            price=Decimal("10"),
            execution_reason=ExecutionReason.END_OF_TEST_LIQUIDATION,
        )


def test_completed_result_enforces_policy_and_execution_consistency() -> None:
    """A completed result cannot mislabel an open or synthetic final state."""

    held = run_open_position(EndOfTestPolicy.HOLD)
    liquidated = run_open_position(EndOfTestPolicy.LIQUIDATE)

    with pytest.raises(ValueError, match="cannot leave an open position"):
        replace(held, end_of_test_policy=EndOfTestPolicy.LIQUIDATE)
    with pytest.raises(ValueError, match="requires the liquidate policy"):
        replace(liquidated, end_of_test_policy=EndOfTestPolicy.HOLD)
