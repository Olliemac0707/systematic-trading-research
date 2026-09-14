"""Tests for pure Decimal-based backtest performance metrics."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_research.models import (
    BacktestResult,
    EquityPoint,
    SimulatedTrade,
    TradeSide,
)
from trading_research.performance import PerformanceSummary, calculate_performance

_START = datetime(2025, 1, 1, tzinfo=UTC)
_STARTING_CASH = Decimal("100")


def make_equity_curve(values: tuple[Decimal, ...]) -> tuple[EquityPoint, ...]:
    """Create a strictly chronological exact equity curve."""

    return tuple(
        EquityPoint(timestamp=_START + timedelta(days=index), equity=value)
        for index, value in enumerate(values)
    )


def make_no_trade_result(
    equity_values: tuple[Decimal, ...] = (Decimal("100"),),
) -> BacktestResult:
    """Create a reconciled result with no account activity."""

    if equity_values[-1] != _STARTING_CASH:
        raise ValueError("a no-trade test result must finish at starting cash")
    return BacktestResult(
        initial_cash=_STARTING_CASH,
        final_cash=_STARTING_CASH,
        final_equity=_STARTING_CASH,
        final_position_quantity=0,
        trades=(),
        equity_curve=make_equity_curve(equity_values),
    )


def make_closed_result(
    buy_price: Decimal,
    sell_price: Decimal,
    *,
    buy_commission: Decimal = Decimal("0"),
    sell_commission: Decimal = Decimal("0"),
    buy_reference_price: Decimal | None = None,
    sell_reference_price: Decimal | None = None,
) -> BacktestResult:
    """Create a reconciled two-unit closed position."""

    quantity = 2
    trades = (
        SimulatedTrade(
            symbol="TEST",
            timestamp=_START,
            side=TradeSide.BUY,
            quantity=quantity,
            price=buy_price,
            commission=buy_commission,
            reference_price=buy_reference_price,
        ),
        SimulatedTrade(
            symbol="TEST",
            timestamp=_START + timedelta(days=1),
            side=TradeSide.SELL,
            quantity=quantity,
            price=sell_price,
            commission=sell_commission,
            reference_price=sell_reference_price,
        ),
    )
    ending_cash = (
        _STARTING_CASH
        - (buy_price * quantity)
        - buy_commission
        + (sell_price * quantity)
        - sell_commission
    )
    return BacktestResult(
        initial_cash=_STARTING_CASH,
        final_cash=ending_cash,
        final_equity=ending_cash,
        final_position_quantity=0,
        trades=trades,
        equity_curve=make_equity_curve((_STARTING_CASH, ending_cash)),
    )


def make_open_result() -> BacktestResult:
    """Create a reconciled two-unit open position marked above fill cost."""

    trade = SimulatedTrade(
        symbol="TEST",
        timestamp=_START,
        side=TradeSide.BUY,
        quantity=2,
        price=Decimal("10"),
    )
    return BacktestResult(
        initial_cash=_STARTING_CASH,
        final_cash=Decimal("80"),
        final_equity=Decimal("104"),
        final_position_quantity=2,
        trades=(trade,),
        equity_curve=make_equity_curve((Decimal("100"), Decimal("104"))),
        final_position_mark_price=Decimal("12"),
    )


def make_drawdown_result(equity_values: tuple[Decimal, ...]) -> BacktestResult:
    """Create a reconciled flat result with a specified intermediate curve."""

    ending_equity = equity_values[-1]
    if ending_equity == _STARTING_CASH:
        trades: tuple[SimulatedTrade, ...] = ()
    else:
        trades = (
            SimulatedTrade(
                symbol="TEST",
                timestamp=_START - timedelta(days=2),
                side=TradeSide.BUY,
                quantity=1,
                price=_STARTING_CASH,
            ),
            SimulatedTrade(
                symbol="TEST",
                timestamp=_START - timedelta(days=1),
                side=TradeSide.SELL,
                quantity=1,
                price=ending_equity,
            ),
        )
    return BacktestResult(
        initial_cash=_STARTING_CASH,
        final_cash=ending_equity,
        final_equity=ending_equity,
        final_position_quantity=0,
        trades=trades,
        equity_curve=make_equity_curve(equity_values),
    )


def test_no_trade_result_has_zero_profit_return_and_costs() -> None:
    """No account activity produces exact zero performance and attribution."""

    summary = calculate_performance(make_no_trade_result())

    assert summary.starting_cash == Decimal("100")
    assert summary.ending_cash == Decimal("100")
    assert summary.ending_equity == Decimal("100")
    assert summary.net_profit == Decimal("0")
    assert summary.total_return == Decimal("0")
    assert summary.gross_realised_pnl == Decimal("0")
    assert summary.unrealised_pnl == Decimal("0")
    assert summary.total_commission == Decimal("0")
    assert summary.total_adverse_slippage_cost == Decimal("0")
    assert summary.commission_ratio == Decimal("0")
    assert summary.slippage_cost_ratio == Decimal("0")
    assert summary.open_quantity == 0
    assert summary.open_position_market_value == Decimal("0")


@pytest.mark.parametrize(
    ("sell_price", "expected_profit", "expected_return"),
    [
        (Decimal("15"), Decimal("10"), Decimal("0.10")),
        (Decimal("8"), Decimal("-4"), Decimal("-0.04")),
    ],
    ids=["profitable", "losing"],
)
def test_closed_result_profit_and_return(
    sell_price: Decimal,
    expected_profit: Decimal,
    expected_return: Decimal,
) -> None:
    """Closed profitable and losing positions use exact ending equity."""

    summary = calculate_performance(make_closed_result(Decimal("10"), sell_price))

    assert summary.net_profit == expected_profit
    assert summary.total_return == expected_return
    assert summary.gross_realised_pnl == expected_profit
    assert summary.unrealised_pnl == Decimal("0")


def test_open_position_uses_ending_equity_and_reports_unrealised_pnl() -> None:
    """Open-position profit comes from its final mark rather than ending cash."""

    summary = calculate_performance(make_open_result())

    assert summary.ending_cash == Decimal("80")
    assert summary.ending_equity == Decimal("104")
    assert summary.net_profit == Decimal("4")
    assert summary.total_return == Decimal("0.04")
    assert summary.gross_realised_pnl == Decimal("0")
    assert summary.unrealised_pnl == Decimal("4")
    assert summary.open_quantity == 2
    assert summary.open_position_market_value == Decimal("24")


def test_commission_is_reflected_once_and_reported_separately() -> None:
    """Gross P&L minus one commission total equals net profit."""

    summary = calculate_performance(
        make_closed_result(
            Decimal("10"),
            Decimal("15"),
            buy_commission=Decimal("0.20"),
            sell_commission=Decimal("0.30"),
        )
    )

    assert summary.gross_realised_pnl == Decimal("10")
    assert summary.total_commission == Decimal("0.50")
    assert summary.commission_ratio == Decimal("0.005")
    assert summary.net_profit == Decimal("9.50")


def test_slippage_is_attribution_and_is_not_deducted_twice() -> None:
    """Adverse buy and sell fills reduce P&L before attribution is reported."""

    summary = calculate_performance(
        make_closed_result(
            Decimal("10.10"),
            Decimal("14.85"),
            buy_reference_price=Decimal("10"),
            sell_reference_price=Decimal("15"),
        )
    )

    assert summary.total_adverse_slippage_cost == Decimal("0.50")
    assert summary.slippage_cost_ratio == Decimal("0.005")
    assert summary.gross_realised_pnl == Decimal("9.50")
    assert summary.net_profit == Decimal("9.50")
    assert summary.net_profit + summary.total_adverse_slippage_cost == Decimal("10.00")


@pytest.mark.parametrize(
    "equity_values",
    [
        (Decimal("100"),),
        (Decimal("100"), Decimal("100"), Decimal("100")),
        (Decimal("100"), Decimal("110"), Decimal("120")),
    ],
    ids=["minimal", "flat", "increasing"],
)
def test_zero_drawdown_policy_has_no_timestamps(
    equity_values: tuple[Decimal, ...],
) -> None:
    """Minimal, flat, and rising curves report zero with no drawdown dates."""

    summary = calculate_performance(make_drawdown_result(equity_values))

    assert summary.maximum_absolute_drawdown == Decimal("0")
    assert summary.maximum_percentage_drawdown == Decimal("0")
    assert summary.drawdown_peak_timestamp is None
    assert summary.drawdown_trough_timestamp is None
    assert summary.drawdown_recovery_timestamp is None


def test_monotonically_decreasing_curve_uses_first_observed_peak() -> None:
    """A continuous decline ends at its deepest unrecovered trough."""

    result = make_drawdown_result(
        (Decimal("100"), Decimal("90"), Decimal("80"))
    )
    summary = calculate_performance(result)

    assert summary.maximum_absolute_drawdown == Decimal("20")
    assert summary.maximum_percentage_drawdown == Decimal("-0.20")
    assert summary.drawdown_peak_timestamp == result.equity_curve[0].timestamp
    assert summary.drawdown_trough_timestamp == result.equity_curve[2].timestamp
    assert summary.drawdown_recovery_timestamp is None


def test_recovered_drawdown_records_exact_peak_trough_and_recovery() -> None:
    """The example drawdown recovers only when equity exceeds its prior peak."""

    result = make_drawdown_result(
        (Decimal("110"), Decimal("99"), Decimal("105"), Decimal("111"))
    )
    summary = calculate_performance(result)

    assert summary.maximum_absolute_drawdown == Decimal("11")
    assert summary.maximum_percentage_drawdown == Decimal("-0.10")
    assert summary.drawdown_peak_timestamp == result.equity_curve[0].timestamp
    assert summary.drawdown_trough_timestamp == result.equity_curve[1].timestamp
    assert summary.drawdown_recovery_timestamp == result.equity_curve[3].timestamp


def test_unrecovered_drawdown_has_no_recovery_timestamp() -> None:
    """A rebound below the prior peak does not count as recovery."""

    result = make_drawdown_result(
        (Decimal("125"), Decimal("100"), Decimal("110"))
    )
    summary = calculate_performance(result)

    assert summary.maximum_absolute_drawdown == Decimal("25")
    assert summary.maximum_percentage_drawdown == Decimal("-0.20")
    assert summary.drawdown_peak_timestamp == result.equity_curve[0].timestamp
    assert summary.drawdown_trough_timestamp == result.equity_curve[1].timestamp
    assert summary.drawdown_recovery_timestamp is None


def test_later_larger_drawdown_replaces_earlier_drawdown() -> None:
    """Drawdown timestamps follow the later episode when its depth is larger."""

    result = make_drawdown_result(
        (
            Decimal("110"),
            Decimal("105"),
            Decimal("111"),
            Decimal("125"),
            Decimal("100"),
            Decimal("126"),
        )
    )
    summary = calculate_performance(result)

    assert summary.maximum_absolute_drawdown == Decimal("25")
    assert summary.maximum_percentage_drawdown == Decimal("-0.20")
    assert summary.drawdown_peak_timestamp == result.equity_curve[3].timestamp
    assert summary.drawdown_trough_timestamp == result.equity_curve[4].timestamp
    assert summary.drawdown_recovery_timestamp == result.equity_curve[5].timestamp


def test_equal_depth_drawdowns_keep_earliest_occurrence() -> None:
    """Equal absolute depths deterministically retain the first drawdown."""

    result = make_drawdown_result(
        (
            Decimal("110"),
            Decimal("100"),
            Decimal("110"),
            Decimal("120"),
            Decimal("110"),
            Decimal("120"),
        )
    )
    summary = calculate_performance(result)

    assert summary.maximum_absolute_drawdown == Decimal("10")
    assert summary.maximum_percentage_drawdown == (
        Decimal("100") - Decimal("110")
    ) / Decimal("110")
    assert summary.drawdown_peak_timestamp == result.equity_curve[0].timestamp
    assert summary.drawdown_trough_timestamp == result.equity_curve[1].timestamp
    assert summary.drawdown_recovery_timestamp == result.equity_curve[2].timestamp


def test_starting_cash_seeds_peak_when_curve_starts_lower() -> None:
    """A pre-curve starting peak is represented without a peak timestamp."""

    result = make_drawdown_result((Decimal("90"), Decimal("100")))
    summary = calculate_performance(result)

    assert summary.maximum_absolute_drawdown == Decimal("10")
    assert summary.maximum_percentage_drawdown == Decimal("-0.10")
    assert summary.drawdown_peak_timestamp is None
    assert summary.drawdown_trough_timestamp == result.equity_curve[0].timestamp
    assert summary.drawdown_recovery_timestamp == result.equity_curve[1].timestamp


def test_recovery_uses_first_point_at_or_above_peak() -> None:
    """Recovery is the first later observation equal to the prior peak."""

    result = make_drawdown_result(
        (Decimal("110"), Decimal("99"), Decimal("110"), Decimal("112"))
    )
    summary = calculate_performance(result)

    assert summary.drawdown_recovery_timestamp == result.equity_curve[2].timestamp


def test_performance_summary_is_immutable() -> None:
    """Calculated summaries cannot be mutated after construction."""

    summary = calculate_performance(make_no_trade_result())
    attribute = "net_profit"

    with pytest.raises(FrozenInstanceError):
        setattr(summary, attribute, Decimal("1"))
    assert isinstance(summary, PerformanceSummary)
