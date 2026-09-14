"""Deterministic tests for the closing-price Donchian strategy."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from trading_research.backtesting import SimpleBacktestEngine
from trading_research.benchmarks import (
    calculate_benchmark_comparison,
    run_buy_and_hold_benchmark,
)
from trading_research.config import BacktestConfig, EndOfTestPolicy, PositionSizingMode
from trading_research.data import CsvMarketDataProvider
from trading_research.errors import StrategyError
from trading_research.models import (
    ExecutionReason,
    MarketBar,
    PreTradeDecisionOutcome,
    SignalAction,
    TradeSide,
)
from trading_research.performance import (
    calculate_exposure_statistics,
    calculate_performance,
    calculate_trade_statistics,
    reconstruct_closed_trades,
)
from trading_research.risk import build_position_sizer, build_risk_policy
from trading_research.strategies import DonchianBreakoutStrategy

_FIXTURE = Path(__file__).parent / "fixtures" / "donchian_prices.csv"


def make_bars(prices: list[int]) -> tuple[MarketBar, ...]:
    """Create simple UTC daily bars whose OHLC values equal each close."""

    start = datetime(2025, 1, 1, tzinfo=UTC)
    return tuple(
        MarketBar(
            symbol="CHAN",
            timestamp=start + timedelta(days=index),
            open=Decimal(price),
            high=Decimal(price),
            low=Decimal(price),
            close=Decimal(price),
            volume=100,
        )
        for index, price in enumerate(prices)
    )


def fixture_bars() -> tuple[MarketBar, ...]:
    """Load the committed synthetic breakout and normal-exit scenario."""

    return tuple(CsvMarketDataProvider(_FIXTURE).get_historical_bars("CHAN", None, None))


@pytest.mark.parametrize(
    ("entry_window", "exit_window", "exception"),
    [
        (0, 2, ValueError),
        (2, -1, ValueError),
        (True, 2, TypeError),
        (2, False, TypeError),
        (Decimal("2"), 2, TypeError),
    ],
)
def test_donchian_windows_require_positive_non_boolean_integers(
    entry_window: object,
    exit_window: object,
    exception: type[Exception],
) -> None:
    """Both independent channel lengths have an explicit whole-number policy."""

    with pytest.raises(exception):
        DonchianBreakoutStrategy(
            entry_window=entry_window,  # type: ignore[arg-type]
            exit_window=exit_window,  # type: ignore[arg-type]
        )


def test_entry_uses_only_prior_closes_and_requires_strict_breakout() -> None:
    """Warmup and equality produce no entry; the first greater close does."""

    bars = make_bars([10, 11, 11, 12])
    signals = DonchianBreakoutStrategy(2, 2).generate_signals(bars)

    assert len(signals) == 1
    assert signals[0].action is SignalAction.BUY
    assert signals[0].timestamp == bars[3].timestamp


def test_one_bar_entry_and_exit_windows_are_valid() -> None:
    """The minimum independent warmups evaluate one prior completed close."""

    bars = make_bars([10, 11, 10, 9])
    signals = DonchianBreakoutStrategy(1, 1).generate_signals(bars)

    assert [signal.action for signal in signals] == [
        SignalAction.BUY,
        SignalAction.SELL,
    ]
    assert [signal.timestamp for signal in signals] == [
        bars[1].timestamp,
        bars[2].timestamp,
    ]


def test_exit_is_strict_and_strategy_never_pyramids_or_goes_short() -> None:
    """Logical long state suppresses repeated entries and equal-low exits."""

    equality = DonchianBreakoutStrategy(2, 2).generate_signals(
        make_bars([10, 11, 12, 11, 11])
    )
    trend = DonchianBreakoutStrategy(2, 2).generate_signals(
        make_bars([10, 11, 12, 13, 14])
    )
    completed = DonchianBreakoutStrategy(2, 2).generate_signals(
        make_bars([10, 11, 12, 13, 11, 10])
    )

    assert [signal.action for signal in equality] == [SignalAction.BUY]
    assert [signal.action for signal in trend] == [SignalAction.BUY]
    assert [signal.action for signal in completed] == [
        SignalAction.BUY,
        SignalAction.SELL,
    ]
    assert DonchianBreakoutStrategy(2, 2).generate_signals(
        make_bars([12, 11, 10, 9])
    ) == ()


def test_signals_do_not_change_when_future_bars_are_appended() -> None:
    """A prefix calculation is invariant to observations that arrive later."""

    bars = make_bars([10, 11, 12, 13, 11, 10, 9])
    strategy = DonchianBreakoutStrategy(2, 2)
    prefix_signals = strategy.generate_signals(bars[:5])
    full_signals = tuple(
        signal
        for signal in strategy.generate_signals(bars)
        if signal.timestamp <= bars[4].timestamp
    )

    assert prefix_signals == full_signals


def test_donchian_validates_symbols_and_chronology() -> None:
    """Collection validation is not bypassed by a short strategy warmup."""

    bars = list(make_bars([10, 11]))
    bars[1] = MarketBar(
        symbol="OTHER",
        timestamp=bars[1].timestamp,
        open=bars[1].open,
        high=bars[1].high,
        low=bars[1].low,
        close=bars[1].close,
        volume=bars[1].volume,
    )
    strategy = DonchianBreakoutStrategy(2, 2)

    with pytest.raises(StrategyError, match="one symbol"):
        strategy.generate_signals(bars)
    with pytest.raises(StrategyError, match="chronological"):
        strategy.generate_signals(tuple(reversed(make_bars([10, 11]))))


def test_normal_exit_preserves_exact_fill_accounting_and_statistics() -> None:
    """Signals fill next-open and all cost and performance values remain exact."""

    result = SimpleBacktestEngine().run(
        fixture_bars(),
        DonchianBreakoutStrategy(2, 2),
        BacktestConfig(
            initial_cash=Decimal("1000"),
            trade_quantity=10,
            commission_bps=Decimal("100"),
            slippage_bps=Decimal("100"),
        ),
    )
    performance = calculate_performance(result)
    statistics = calculate_trade_statistics(result)
    exposure = calculate_exposure_statistics(result)

    assert [trade.side for trade in result.trades] == [TradeSide.BUY, TradeSide.SELL]
    assert [trade.reference_price for trade in result.trades] == [
        Decimal("13"),
        Decimal("10"),
    ]
    assert [trade.price for trade in result.trades] == [
        Decimal("13.13"),
        Decimal("9.90"),
    ]
    assert [trade.commission for trade in result.trades] == [
        Decimal("1.3130"),
        Decimal("0.9900"),
    ]
    assert result.final_cash == Decimal("965.3970")
    assert result.final_equity == Decimal("965.3970")
    assert result.final_position_quantity == 0
    assert result.reconciliation.is_reconciled
    assert performance.net_profit == Decimal("-34.6030")
    assert performance.total_adverse_slippage_cost == Decimal("2.30")
    assert statistics.completed_trade_count == 1
    assert statistics.losing_trade_count == 1
    assert statistics.net_closed_trade_pnl == Decimal("-34.6030")
    assert exposure.time_in_market_ratio == Decimal("0.25")
    assert exposure.full_approval_count == 1
    closed_trade = reconstruct_closed_trades(result)[0]
    assert closed_trade.entry_fill_price == Decimal("13.13")
    assert closed_trade.exit_fill_price == Decimal("9.90")
    assert closed_trade.net_pnl == Decimal("-34.6030")


def test_profitable_completed_trade_uses_next_bar_opens() -> None:
    """A deterministic gap scenario covers the profitable closed-trade branch."""

    bars = list(make_bars([10, 11, 12, 15, 11, 12]))
    bars[3] = replace(bars[3], open=Decimal("10"), low=Decimal("10"))
    bars[5] = replace(bars[5], open=Decimal("14"), high=Decimal("14"))
    result = SimpleBacktestEngine().run(
        bars,
        DonchianBreakoutStrategy(2, 2),
        BacktestConfig(initial_cash=Decimal("1000"), trade_quantity=10),
    )

    assert [trade.reference_price for trade in result.trades] == [
        Decimal("10"),
        Decimal("14"),
    ]
    assert result.final_cash == result.final_equity == Decimal("1040")
    assert calculate_trade_statistics(result).winning_trade_count == 1
    assert reconstruct_closed_trades(result)[0].net_pnl == Decimal("40")
    assert result.reconciliation.is_reconciled


def test_no_breakout_and_hold_open_position_complete_and_reconcile() -> None:
    """No-trade and marked-open outcomes both use authoritative account values."""

    config = BacktestConfig(
        initial_cash=Decimal("1000"),
        trade_quantity=10,
        commission_bps=Decimal("100"),
        slippage_bps=Decimal("100"),
    )
    no_breakout = SimpleBacktestEngine().run(
        make_bars([10, 10, 10, 10]),
        DonchianBreakoutStrategy(2, 2),
        config,
    )
    holding = SimpleBacktestEngine().run(
        fixture_bars()[:4],
        DonchianBreakoutStrategy(2, 2),
        config,
    )

    assert no_breakout.trades == ()
    assert no_breakout.final_cash == no_breakout.final_equity == Decimal("1000")
    assert no_breakout.reconciliation.is_reconciled
    assert holding.final_cash == Decimal("867.3870")
    assert holding.final_equity == Decimal("997.3870")
    assert holding.final_position_quantity == 10
    assert holding.final_position_mark_price == Decimal("13")
    assert holding.reconciliation.is_reconciled


def test_end_of_test_liquidation_uses_existing_final_close_path() -> None:
    """Liquidation closes the Donchian position without a strategy exit signal."""

    result = SimpleBacktestEngine().run(
        fixture_bars()[:4],
        DonchianBreakoutStrategy(2, 2),
        BacktestConfig(
            initial_cash=Decimal("1000"),
            trade_quantity=10,
            commission_bps=Decimal("100"),
            slippage_bps=Decimal("100"),
            end_of_test_policy=EndOfTestPolicy.LIQUIDATE,
        ),
    )

    assert result.final_cash == result.final_equity == Decimal("994.8000")
    assert result.final_position_quantity == 0
    assert result.trades[-1].price == Decimal("12.87")
    assert result.trades[-1].execution_reason is ExecutionReason.END_OF_TEST_LIQUIDATION
    assert calculate_performance(result).total_adverse_slippage_cost == Decimal("2.60")
    assert result.reconciliation.is_reconciled


@pytest.mark.parametrize(
    ("config", "expected_quantity", "expected_outcome"),
    [
        (
            BacktestConfig(
                initial_cash=Decimal("1000"),
                trade_quantity=10,
                position_sizing_mode=PositionSizingMode.CASH_ALLOCATION,
                cash_allocation_ratio=Decimal("0.5"),
                commission_bps=Decimal("100"),
                slippage_bps=Decimal("100"),
            ),
            37,
            PreTradeDecisionOutcome.APPROVED,
        ),
        (
            BacktestConfig(
                initial_cash=Decimal("1000"),
                trade_quantity=10,
                maximum_position_value=Decimal("70"),
                commission_bps=Decimal("100"),
                slippage_bps=Decimal("100"),
            ),
            5,
            PreTradeDecisionOutcome.REDUCED,
        ),
        (
            BacktestConfig(
                initial_cash=Decimal("1000"),
                trade_quantity=10,
                minimum_cash_reserve=Decimal("1000"),
                commission_bps=Decimal("100"),
                slippage_bps=Decimal("100"),
            ),
            0,
            PreTradeDecisionOutcome.REJECTED,
        ),
        (
            BacktestConfig(
                initial_cash=Decimal("1000"),
                trade_quantity=10,
                position_sizing_mode=PositionSizingMode.CASH_ALLOCATION,
                cash_allocation_ratio=Decimal("0.001"),
                commission_bps=Decimal("100"),
                slippage_bps=Decimal("100"),
            ),
            0,
            PreTradeDecisionOutcome.REJECTED,
        ),
    ],
)
def test_existing_sizing_and_risk_controls_apply_to_donchian_entries(
    config: BacktestConfig,
    expected_quantity: int,
    expected_outcome: PreTradeDecisionOutcome,
) -> None:
    """The strategy feeds the unchanged sizing and risk pipeline."""

    result = SimpleBacktestEngine().run(
        fixture_bars()[:4],
        DonchianBreakoutStrategy(2, 2),
        config,
    )

    assert result.pre_trade_decisions[0].approved_quantity == expected_quantity
    assert result.pre_trade_decisions[0].outcome is expected_outcome
    assert result.final_position_quantity == expected_quantity
    if expected_quantity == 0:
        assert result.trades == ()
        assert result.reconciliation.ledger.entries == ()
    assert result.reconciliation.is_reconciled


@pytest.mark.parametrize(
    "end_policy",
    [EndOfTestPolicy.HOLD, EndOfTestPolicy.LIQUIDATE],
)
def test_donchian_uses_unchanged_capital_matched_benchmark(
    end_policy: EndOfTestPolicy,
) -> None:
    """Benchmark analysis shares all assumptions and cannot mutate the strategy run."""

    bars = fixture_bars()
    config = BacktestConfig(
        initial_cash=Decimal("1000"),
        trade_quantity=10,
        commission_bps=Decimal("100"),
        slippage_bps=Decimal("100"),
        end_of_test_policy=end_policy,
    )
    strategy_result = SimpleBacktestEngine().run(
        bars,
        DonchianBreakoutStrategy(2, 2),
        config,
    )
    original_trades = strategy_result.trades
    benchmark = run_buy_and_hold_benchmark(
        bars=bars,
        backtest_configuration=config,
        position_sizer=build_position_sizer(config),
        risk_policy=build_risk_policy(config),
        risk_metric_settings=None,
    )
    comparison = calculate_benchmark_comparison(
        strategy_result,
        benchmark,
        bars=bars,
        backtest_configuration=config,
        risk_metric_settings=None,
    )

    expected_benchmark_equity = (
        Decimal("987.7890")
        if end_policy is EndOfTestPolicy.HOLD
        else Decimal("985.7990")
    )
    assert strategy_result.trades == original_trades
    assert strategy_result.final_equity == Decimal("965.3970")
    assert benchmark.result.final_equity == expected_benchmark_equity
    assert comparison.strategy_ending_equity == strategy_result.final_equity
    assert comparison.benchmark_ending_equity == expected_benchmark_equity
    assert strategy_result.reconciliation.is_reconciled
    assert benchmark.result.reconciliation.is_reconciled
