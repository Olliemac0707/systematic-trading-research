"""Tests for closed-trade reconstruction and deterministic statistics."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_research.models import (
    AccountLedger,
    BacktestResult,
    EquityPoint,
    SimulatedTrade,
    TradeSide,
)
from trading_research.performance import (
    ClosedTrade,
    TradeStatistics,
    calculate_performance,
    calculate_trade_statistics,
    reconstruct_closed_trades,
)

_START = datetime(2025, 1, 1, tzinfo=UTC)
_STARTING_CASH = Decimal("1000")


def execution(
    side: TradeSide,
    day: int,
    price: Decimal,
    *,
    quantity: int = 2,
    symbol: str = "TEST",
    commission: Decimal = Decimal("0"),
    reference_price: Decimal | None = None,
) -> SimulatedTrade:
    """Create one validated simulated execution."""

    return SimulatedTrade(
        symbol=symbol,
        timestamp=_START + timedelta(days=day),
        side=side,
        quantity=quantity,
        price=price,
        commission=commission,
        reference_price=reference_price,
    )


def make_result(
    trades: tuple[SimulatedTrade, ...],
    *,
    final_mark_price: Decimal = Decimal("10"),
) -> BacktestResult:
    """Create a reconciled result from a valid long-only execution sequence."""

    ledger = AccountLedger.from_trades(_STARTING_CASH, trades)
    ending_cash = ledger.calculated_ending_cash
    open_quantity = ledger.ending_position_quantity
    ending_equity = ending_cash + (final_mark_price * open_quantity)
    final_timestamp = trades[-1].timestamp + timedelta(days=1) if trades else _START
    return BacktestResult(
        initial_cash=_STARTING_CASH,
        final_cash=ending_cash,
        final_equity=ending_equity,
        final_position_quantity=open_quantity,
        trades=trades,
        equity_curve=(EquityPoint(timestamp=final_timestamp, equity=ending_equity),),
        final_position_mark_price=final_mark_price if open_quantity else None,
    )


def make_unsafe_result(trades: tuple[SimulatedTrade, ...]) -> BacktestResult:
    """Bypass upstream ledger validation to exercise defensive pairing checks."""

    result = object.__new__(BacktestResult)
    object.__setattr__(result, "trades", trades)
    return result


def test_no_executions_produce_no_closed_trades() -> None:
    """An inactive result has no reconstructed round trips."""

    assert reconstruct_closed_trades(make_result(())) == ()


def test_buy_and_sell_reconstruct_exact_closed_trade() -> None:
    """A full long round trip retains fills, references, costs, and timing."""

    buy = execution(
        TradeSide.BUY,
        0,
        Decimal("10.10"),
        commission=Decimal("0.20"),
        reference_price=Decimal("10"),
    )
    sell = execution(
        TradeSide.SELL,
        3,
        Decimal("14.85"),
        commission=Decimal("0.30"),
        reference_price=Decimal("15"),
    )

    closed_trade = reconstruct_closed_trades(make_result((buy, sell)))[0]

    assert closed_trade.symbol == "TEST"
    assert closed_trade.entry_timestamp == buy.timestamp
    assert closed_trade.exit_timestamp == sell.timestamp
    assert closed_trade.quantity == 2
    assert closed_trade.holding_period == timedelta(days=3)
    assert closed_trade.entry_reference_price == Decimal("10")
    assert closed_trade.entry_fill_price == Decimal("10.10")
    assert closed_trade.exit_reference_price == Decimal("15")
    assert closed_trade.exit_fill_price == Decimal("14.85")
    assert closed_trade.gross_entry_value == Decimal("20.20")
    assert closed_trade.gross_exit_value == Decimal("29.70")
    assert closed_trade.entry_commission == Decimal("0.20")
    assert closed_trade.exit_commission == Decimal("0.30")
    assert closed_trade.total_commission == Decimal("0.50")
    assert closed_trade.entry_slippage_cost == Decimal("0.20")
    assert closed_trade.exit_slippage_cost == Decimal("0.30")
    assert closed_trade.total_slippage_cost == Decimal("0.50")
    assert closed_trade.gross_pnl == Decimal("9.50")
    assert closed_trade.net_pnl == Decimal("9.00")
    assert closed_trade.return_ratio == Decimal("9.00") / Decimal("20.20")
    assert closed_trade.net_pnl == closed_trade.gross_pnl - closed_trade.total_commission


def test_multiple_sequential_round_trips_are_reconstructed_in_order() -> None:
    """Each completed engine position produces one chronological closed trade."""

    trades = (
        execution(TradeSide.BUY, 0, Decimal("10")),
        execution(TradeSide.SELL, 2, Decimal("15")),
        execution(TradeSide.BUY, 3, Decimal("20")),
        execution(TradeSide.SELL, 7, Decimal("18")),
    )

    closed_trades = reconstruct_closed_trades(make_result(trades))

    assert len(closed_trades) == 2
    assert [trade.net_pnl for trade in closed_trades] == [
        Decimal("10"),
        Decimal("-4"),
    ]
    assert [trade.holding_period for trade in closed_trades] == [
        timedelta(days=2),
        timedelta(days=4),
    ]


def test_unmatched_final_buy_remains_open_and_is_excluded() -> None:
    """A final open position is left to account and performance valuation."""

    trades = (
        execution(TradeSide.BUY, 0, Decimal("10")),
        execution(TradeSide.SELL, 2, Decimal("15")),
        execution(
            TradeSide.BUY,
            3,
            Decimal("12"),
            commission=Decimal("0.20"),
        ),
    )
    result = make_result(trades, final_mark_price=Decimal("13"))

    closed_trades = reconstruct_closed_trades(result)
    statistics = calculate_trade_statistics(result)

    assert len(closed_trades) == 1
    assert closed_trades[0].net_pnl == Decimal("10")
    assert statistics.completed_trade_count == 1
    assert calculate_performance(result).unrealised_pnl == Decimal("2")


@pytest.mark.parametrize(
    ("sell_price", "commission", "expected_net_pnl"),
    [
        (Decimal("15"), Decimal("0"), Decimal("10")),
        (Decimal("8"), Decimal("0"), Decimal("-4")),
        (Decimal("11"), Decimal("2"), Decimal("0")),
    ],
    ids=["profitable", "losing", "breakeven-after-commission"],
)
def test_exact_closed_trade_classification_inputs(
    sell_price: Decimal,
    commission: Decimal,
    expected_net_pnl: Decimal,
) -> None:
    """Profit, loss, and exact commission breakeven remain deterministic."""

    buy_commission = commission / Decimal("2")
    sell_commission = commission - buy_commission
    result = make_result(
        (
            execution(
                TradeSide.BUY,
                0,
                Decimal("10"),
                commission=buy_commission,
            ),
            execution(
                TradeSide.SELL,
                1,
                sell_price,
                commission=sell_commission,
            ),
        )
    )

    assert reconstruct_closed_trades(result)[0].net_pnl == expected_net_pnl


def test_sell_before_buy_rejects_unsupported_short_exposure() -> None:
    """A leading sell cannot be interpreted as a long-only round trip."""

    result = make_unsafe_result((execution(TradeSide.SELL, 0, Decimal("10")),))

    with pytest.raises(ValueError, match="sell before buy.*short"):
        reconstruct_closed_trades(result)


def test_oversell_is_rejected() -> None:
    """An exit cannot exceed the quantity of the open entry."""

    result = make_unsafe_result(
        (
            execution(TradeSide.BUY, 0, Decimal("10"), quantity=2),
            execution(TradeSide.SELL, 1, Decimal("11"), quantity=3),
        )
    )

    with pytest.raises(ValueError, match="exceed"):
        reconstruct_closed_trades(result)


def test_partial_exit_is_rejected() -> None:
    """The analyser does not invent partial-exit support absent from the engine."""

    result = make_unsafe_result(
        (
            execution(TradeSide.BUY, 0, Decimal("10"), quantity=3),
            execution(TradeSide.SELL, 1, Decimal("11"), quantity=2),
        )
    )

    with pytest.raises(ValueError, match="partial exits"):
        reconstruct_closed_trades(result)


def test_repeated_entry_is_rejected() -> None:
    """The analyser preserves the engine's one-position-at-a-time invariant."""

    result = make_unsafe_result(
        (
            execution(TradeSide.BUY, 0, Decimal("10")),
            execution(TradeSide.BUY, 1, Decimal("11")),
        )
    )

    with pytest.raises(ValueError, match="repeated entries"):
        reconstruct_closed_trades(result)


def test_symbol_mismatch_is_rejected() -> None:
    """Executions for different symbols cannot form one closed trade."""

    result = make_unsafe_result(
        (
            execution(TradeSide.BUY, 0, Decimal("10"), symbol="AAA"),
            execution(TradeSide.SELL, 1, Decimal("11"), symbol="BBB"),
        )
    )

    with pytest.raises(ValueError, match="same symbol"):
        reconstruct_closed_trades(result)


def test_non_chronological_executions_are_rejected() -> None:
    """An exit occurring before its entry cannot form a valid holding period."""

    result = make_unsafe_result(
        (
            execution(TradeSide.BUY, 2, Decimal("10")),
            execution(TradeSide.SELL, 1, Decimal("11")),
        )
    )

    with pytest.raises(ValueError, match="strictly chronological"):
        reconstruct_closed_trades(result)


def test_no_completed_trades_have_zero_counts_and_undefined_rates() -> None:
    """Empty statistics use zeros for totals and None for undefined means."""

    statistics = calculate_trade_statistics(make_result(()))

    assert statistics.completed_trade_count == 0
    assert statistics.winning_trade_count == 0
    assert statistics.losing_trade_count == 0
    assert statistics.breakeven_trade_count == 0
    assert statistics.win_rate is None
    assert statistics.loss_rate is None
    assert statistics.gross_profit == Decimal("0")
    assert statistics.gross_loss == Decimal("0")
    assert statistics.net_closed_trade_pnl == Decimal("0")
    assert statistics.average_net_pnl is None
    assert statistics.average_winner is None
    assert statistics.average_loser is None
    assert statistics.largest_winner is None
    assert statistics.largest_loser is None
    assert statistics.profit_factor is None
    assert statistics.payoff_ratio is None
    assert statistics.expectancy is None
    assert statistics.average_return_ratio is None
    assert statistics.average_holding_period is None
    assert statistics.longest_holding_period is None
    assert statistics.shortest_holding_period is None
    assert statistics.total_trade_commission == Decimal("0")
    assert statistics.total_trade_slippage_cost == Decimal("0")


def test_single_winner_has_no_profit_factor_without_a_loss() -> None:
    """A zero gross loss produces no infinite profit factor."""

    result = make_result(
        (
            execution(TradeSide.BUY, 0, Decimal("10")),
            execution(TradeSide.SELL, 2, Decimal("15")),
        )
    )

    statistics = calculate_trade_statistics(result)

    assert statistics.completed_trade_count == 1
    assert statistics.winning_trade_count == 1
    assert statistics.win_rate == Decimal("1")
    assert statistics.loss_rate == Decimal("0")
    assert statistics.gross_profit == Decimal("10")
    assert statistics.gross_loss == Decimal("0")
    assert statistics.average_winner == Decimal("10")
    assert statistics.largest_winner == Decimal("10")
    assert statistics.profit_factor is None
    assert statistics.payoff_ratio is None
    assert statistics.expectancy == Decimal("10")


def test_single_loser_uses_positive_loss_magnitudes() -> None:
    """Loss aggregates are non-negative while net P&L remains negative."""

    result = make_result(
        (
            execution(TradeSide.BUY, 0, Decimal("10")),
            execution(TradeSide.SELL, 2, Decimal("8")),
        )
    )

    statistics = calculate_trade_statistics(result)

    assert statistics.losing_trade_count == 1
    assert statistics.gross_profit == Decimal("0")
    assert statistics.gross_loss == Decimal("4")
    assert statistics.net_closed_trade_pnl == Decimal("-4")
    assert statistics.average_loser == Decimal("4")
    assert statistics.largest_loser == Decimal("4")
    assert statistics.profit_factor == Decimal("0")
    assert statistics.payoff_ratio is None


def test_single_breakeven_remains_in_rate_denominator() -> None:
    """Exact breakeven is neither a win nor loss but remains completed."""

    result = make_result(
        (
            execution(
                TradeSide.BUY,
                0,
                Decimal("10"),
                commission=Decimal("0.50"),
            ),
            execution(
                TradeSide.SELL,
                2,
                Decimal("11"),
                commission=Decimal("1.50"),
            ),
        )
    )

    statistics = calculate_trade_statistics(result)

    assert statistics.breakeven_trade_count == 1
    assert statistics.winning_trade_count == 0
    assert statistics.losing_trade_count == 0
    assert statistics.win_rate == Decimal("0")
    assert statistics.loss_rate == Decimal("0")
    assert statistics.expectancy == Decimal("0")
    assert statistics.total_trade_commission == Decimal("2.00")


def test_mixed_trade_statistics_are_exact() -> None:
    """Mixed classifications produce exact rates, means, ratios, and durations."""

    trades = (
        execution(TradeSide.BUY, 0, Decimal("10")),
        execution(TradeSide.SELL, 2, Decimal("15")),
        execution(TradeSide.BUY, 3, Decimal("10")),
        execution(TradeSide.SELL, 7, Decimal("8")),
        execution(
            TradeSide.BUY,
            8,
            Decimal("10"),
            commission=Decimal("0.50"),
        ),
        execution(
            TradeSide.SELL,
            14,
            Decimal("11"),
            commission=Decimal("1.50"),
        ),
    )
    result = make_result(trades)

    statistics = calculate_trade_statistics(result)
    closed_trades = reconstruct_closed_trades(result)

    assert statistics.completed_trade_count == 3
    assert statistics.winning_trade_count == 1
    assert statistics.losing_trade_count == 1
    assert statistics.breakeven_trade_count == 1
    assert statistics.win_rate == Decimal("1") / Decimal("3")
    assert statistics.loss_rate == Decimal("1") / Decimal("3")
    assert statistics.gross_profit == Decimal("10")
    assert statistics.gross_loss == Decimal("4")
    assert statistics.net_closed_trade_pnl == Decimal("6")
    assert statistics.average_net_pnl == Decimal("2")
    assert statistics.average_winner == Decimal("10")
    assert statistics.average_loser == Decimal("4")
    assert statistics.largest_winner == Decimal("10")
    assert statistics.largest_loser == Decimal("4")
    assert statistics.profit_factor == Decimal("2.5")
    assert statistics.payoff_ratio == Decimal("2.5")
    assert statistics.expectancy == Decimal("2")
    assert statistics.average_return_ratio == Decimal("0.1")
    assert statistics.average_holding_period == timedelta(days=4)
    assert statistics.shortest_holding_period == timedelta(days=2)
    assert statistics.longest_holding_period == timedelta(days=6)
    assert statistics.total_trade_commission == Decimal("2.00")
    assert statistics.total_trade_slippage_cost == Decimal("0")
    assert sum((trade.net_pnl for trade in closed_trades), Decimal("0")) == (
        calculate_performance(result).gross_realised_pnl
        - statistics.total_trade_commission
    )


def test_high_precision_round_trips_reconcile_account_and_trade_totals() -> None:
    """Finite-context aggregation cannot create a P&L integrity residual."""

    trades: list[SimulatedTrade] = []
    for index in range(20):
        entry_price = Decimal("1000.123456789012345678901234")
        magnitude = Decimal(
            f"{1 + index % 17}.{index * 104729:024d}"
        )
        exit_price = (
            entry_price + magnitude
            if index % 2 == 0
            else entry_price - magnitude
        )
        trades.extend(
            (
                execution(
                    TradeSide.BUY,
                    index * 2,
                    entry_price,
                    quantity=1,
                ),
                execution(
                    TradeSide.SELL,
                    index * 2 + 1,
                    exit_price,
                    quantity=1,
                ),
            )
        )
    ledger = AccountLedger.from_trades(Decimal("1000000"), tuple(trades))
    ending_cash = ledger.calculated_ending_cash
    result = BacktestResult(
        initial_cash=Decimal("1000000"),
        final_cash=ending_cash,
        final_equity=ending_cash,
        final_position_quantity=0,
        trades=tuple(trades),
        equity_curve=(
            EquityPoint(
                timestamp=trades[-1].timestamp + timedelta(days=1),
                equity=ending_cash,
            ),
        ),
    )

    performance = calculate_performance(result)
    statistics = calculate_trade_statistics(result)

    assert performance.net_profit == Decimal(
        "6.999999999999999998955"
    )
    assert (
        performance.gross_realised_pnl - performance.total_commission
        == performance.net_profit
    )
    assert statistics.net_closed_trade_pnl == performance.net_profit
    assert (
        statistics.gross_profit - statistics.gross_loss
        == statistics.net_closed_trade_pnl
    )


def test_trade_statistics_preserve_long_decimal_coefficients_when_subtracting() -> None:
    """Sign changes must not round an exact aggregate before validation."""

    aggregate = Decimal("122565.06116329251081923379075")
    statistics = TradeStatistics(
        completed_trade_count=2,
        winning_trade_count=1,
        losing_trade_count=1,
        breakeven_trade_count=0,
        win_rate=Decimal("0.5"),
        loss_rate=Decimal("0.5"),
        gross_profit=aggregate,
        gross_loss=aggregate,
        net_closed_trade_pnl=Decimal("0"),
        average_net_pnl=Decimal("0"),
        average_winner=aggregate,
        average_loser=aggregate,
        largest_winner=aggregate,
        largest_loser=aggregate,
        profit_factor=Decimal("1"),
        payoff_ratio=Decimal("1"),
        expectancy=Decimal("0"),
        average_return_ratio=Decimal("0"),
        average_holding_period=timedelta(days=1),
        longest_holding_period=timedelta(days=1),
        shortest_holding_period=timedelta(days=1),
        total_trade_commission=Decimal("0"),
        total_trade_slippage_cost=Decimal("0"),
    )

    assert statistics.gross_profit - statistics.gross_loss == Decimal("0")


def test_trade_statistics_sum_execution_slippage_without_deducting_it() -> None:
    """Trade slippage is reported from fills and never subtracted again."""

    result = make_result(
        (
            execution(
                TradeSide.BUY,
                0,
                Decimal("10.10"),
                reference_price=Decimal("10"),
            ),
            execution(
                TradeSide.SELL,
                2,
                Decimal("14.85"),
                reference_price=Decimal("15"),
            ),
        )
    )

    statistics = calculate_trade_statistics(result)

    assert statistics.total_trade_slippage_cost == Decimal("0.50")
    assert statistics.net_closed_trade_pnl == Decimal("9.50")


def test_closed_trade_and_statistics_are_immutable() -> None:
    """Reconstructed trades and aggregate statistics cannot be mutated."""

    result = make_result(
        (
            execution(TradeSide.BUY, 0, Decimal("10")),
            execution(TradeSide.SELL, 1, Decimal("15")),
        )
    )
    closed_trade = reconstruct_closed_trades(result)[0]
    statistics = calculate_trade_statistics(result)
    trade_attribute = "net_pnl"
    statistics_attribute = "completed_trade_count"

    with pytest.raises(FrozenInstanceError):
        setattr(closed_trade, trade_attribute, Decimal("0"))
    with pytest.raises(FrozenInstanceError):
        setattr(statistics, statistics_attribute, 0)
    assert isinstance(closed_trade, ClosedTrade)
    assert isinstance(statistics, TradeStatistics)
