"""Deterministic tests for account ledger and reconciliation accounting."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_research.backtesting import SimpleBacktestEngine
from trading_research.config import BacktestConfig
from trading_research.models import (
    AccountLedger,
    AccountReconciliation,
    BacktestResult,
    EquityPoint,
    MarketBar,
    SignalAction,
    SimulatedTrade,
    StrategySignal,
    TradeSide,
)
from trading_research.simulation import (
    calculate_buy_fill_price,
    calculate_proportional_commission,
    calculate_sell_fill_price,
)
from trading_research.strategies import Strategy


class FixedSignalStrategy(Strategy):
    """Return explicit signals for deterministic accounting scenarios."""

    def __init__(self, signals: Sequence[StrategySignal]) -> None:
        self._signals = tuple(signals)

    def generate_signals(self, bars: Sequence[MarketBar]) -> Sequence[StrategySignal]:
        """Return configured signals without deriving new decisions."""

        return self._signals


def make_bars(prices: Sequence[Decimal]) -> tuple[MarketBar, ...]:
    """Create daily bars whose open and close share an exact price."""

    start = datetime(2025, 1, 1, tzinfo=UTC)
    return tuple(
        MarketBar(
            symbol="TEST",
            timestamp=start + timedelta(days=index),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=100,
        )
        for index, price in enumerate(prices)
    )


def make_signal(bar: MarketBar, action: SignalAction) -> StrategySignal:
    """Create one explicit signal for a test bar."""

    return StrategySignal(
        symbol=bar.symbol,
        timestamp=bar.timestamp,
        action=action,
    )


def run_round_trip(
    buy_reference_price: Decimal,
    sell_reference_price: Decimal,
    *,
    commission_bps: Decimal = Decimal("0"),
    slippage_bps: Decimal = Decimal("0"),
) -> BacktestResult:
    """Buy two units and later sell them using next-bar execution."""

    bars = make_bars(
        (
            Decimal("1"),
            buy_reference_price,
            Decimal("1"),
            sell_reference_price,
        )
    )
    strategy = FixedSignalStrategy(
        (
            make_signal(bars[0], SignalAction.BUY),
            make_signal(bars[2], SignalAction.SELL),
        )
    )
    config = BacktestConfig(
        initial_cash=Decimal("100"),
        trade_quantity=2,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )
    return SimpleBacktestEngine().run(bars, strategy, config)


def assert_exactly_reconciled(result: BacktestResult) -> None:
    """Assert the completion invariant for both cash and equity."""

    reconciliation = result.reconciliation
    assert reconciliation.cash_difference == Decimal("0")
    assert reconciliation.reconciliation_difference == Decimal("0")
    assert reconciliation.is_reconciled


def test_no_trades_reconcile_to_starting_cash() -> None:
    """An inactive account ends with its exact starting cash and equity."""

    bars = make_bars((Decimal("10"),))

    result = SimpleBacktestEngine().run(
        bars,
        FixedSignalStrategy(()),
        BacktestConfig(initial_cash=Decimal("100")),
    )

    ledger = result.reconciliation.ledger
    assert ledger.entries == ()
    assert ledger.total_buy_notional == Decimal("0")
    assert ledger.total_sell_notional == Decimal("0")
    assert ledger.total_commissions == Decimal("0")
    assert ledger.calculated_ending_cash == Decimal("100")
    assert result.reconciliation.remaining_position_value == Decimal("0")
    assert result.final_equity == Decimal("100")
    assert_exactly_reconciled(result)


def test_open_position_reconciles_with_remaining_marked_value() -> None:
    """Ending equity adds the independent final mark for unsold units."""

    bars = make_bars((Decimal("1"), Decimal("10"), Decimal("12")))
    strategy = FixedSignalStrategy((make_signal(bars[0], SignalAction.BUY),))

    result = SimpleBacktestEngine().run(
        bars,
        strategy,
        BacktestConfig(initial_cash=Decimal("100"), trade_quantity=2),
    )

    reconciliation = result.reconciliation
    assert result.final_cash == Decimal("80")
    assert reconciliation.remaining_position_quantity == 2
    assert reconciliation.ending_position_mark_price == Decimal("12")
    assert reconciliation.remaining_position_value == Decimal("24")
    assert reconciliation.calculated_ending_equity == Decimal("104")
    assert result.final_equity == Decimal("104")
    assert_exactly_reconciled(result)


def test_profitable_closed_trade_reconciles_sale_proceeds() -> None:
    """A profitable round trip explains its exact increase in ending cash."""

    result = run_round_trip(Decimal("10"), Decimal("15"))

    ledger = result.reconciliation.ledger
    assert [entry.cash_change for entry in ledger.entries] == [
        Decimal("-20"),
        Decimal("30"),
    ]
    assert ledger.total_buy_notional == Decimal("20")
    assert ledger.total_sell_notional == Decimal("30")
    assert ledger.net_trade_cash_flow == Decimal("10")
    assert result.final_cash == Decimal("110")
    assert result.final_equity == Decimal("110")
    assert_exactly_reconciled(result)


def test_losing_closed_trade_reconciles_sale_proceeds() -> None:
    """A losing round trip explains its exact decrease in ending cash."""

    result = run_round_trip(Decimal("10"), Decimal("8"))

    ledger = result.reconciliation.ledger
    assert ledger.total_buy_notional == Decimal("20")
    assert ledger.total_sell_notional == Decimal("16")
    assert ledger.net_trade_cash_flow == Decimal("-4")
    assert result.final_cash == Decimal("96")
    assert result.final_equity == Decimal("96")
    assert_exactly_reconciled(result)


def test_commissions_are_deducted_once_from_account_cash() -> None:
    """Buy and sell commissions are visible and each affects cash once."""

    result = run_round_trip(
        Decimal("10"),
        Decimal("15"),
        commission_bps=Decimal("100"),
    )

    ledger = result.reconciliation.ledger
    assert [trade.commission for trade in result.trades] == [
        Decimal("0.20"),
        Decimal("0.30"),
    ]
    assert ledger.total_commissions == Decimal("0.50")
    assert ledger.net_trade_cash_flow == Decimal("9.50")
    assert ledger.calculated_ending_cash == Decimal("109.50")
    assert result.final_cash == Decimal("109.50")
    assert_exactly_reconciled(result)


def test_slippage_is_attributed_without_a_second_cash_deduction() -> None:
    """Fill notional moves cash while slippage remains execution attribution."""

    result = run_round_trip(
        Decimal("10"),
        Decimal("15"),
        slippage_bps=Decimal("100"),
    )

    ledger = result.reconciliation.ledger
    assert [trade.reference_price for trade in result.trades] == [
        Decimal("10"),
        Decimal("15"),
    ]
    assert [trade.price for trade in result.trades] == [
        Decimal("10.10"),
        Decimal("14.85"),
    ]
    assert [entry.slippage_cost for entry in ledger.entries] == [
        Decimal("0.20"),
        Decimal("0.30"),
    ]
    assert ledger.total_slippage_cost == Decimal("0.50")
    assert ledger.net_trade_cash_flow == Decimal("9.50")
    assert ledger.calculated_ending_cash == Decimal("109.50")
    assert result.final_cash + ledger.total_slippage_cost == Decimal("110.00")
    assert_exactly_reconciled(result)


def test_reconciliation_exposes_exact_cash_and_equity_differences() -> None:
    """An audit model reports each unexplained account difference exactly."""

    ledger = AccountLedger(starting_cash=Decimal("100"), entries=())
    reconciliation = AccountReconciliation(
        ledger=ledger,
        ending_cash=Decimal("99"),
        ending_position_mark_price=Decimal("10"),
        ending_equity=Decimal("99"),
    )
    assert reconciliation.cash_difference == Decimal("-1")
    assert reconciliation.reconciliation_difference == Decimal("-1")
    assert not reconciliation.is_reconciled


def test_ledger_replays_high_precision_cash_movements_in_execution_order() -> None:
    """Regrouping precise fills cannot introduce a reconciliation residual."""

    prices = tuple(
        Decimal(
            f"{20 + index}."
            f"{(index * 7919) % 1_000_000_000_000_000:015d}"
        )
        for index in range(1, 41)
    )
    quantities = tuple(1000 + (index * 37) % 500 for index in range(20))
    trades: list[SimulatedTrade] = []
    for index, reference_price in enumerate(prices):
        side = TradeSide.BUY if index % 2 == 0 else TradeSide.SELL
        quantity = quantities[index // 2]
        fill_price = (
            calculate_buy_fill_price(reference_price, Decimal("5"))
            if side is TradeSide.BUY
            else calculate_sell_fill_price(reference_price, Decimal("5"))
        )
        trades.append(
            SimulatedTrade(
                symbol="TEST",
                timestamp=datetime(2025, 1, 1, tzinfo=UTC)
                + timedelta(days=index),
                side=side,
                quantity=quantity,
                price=fill_price,
                reference_price=reference_price,
                commission=calculate_proportional_commission(
                    fill_price,
                    quantity,
                    Decimal("1"),
                ),
            )
        )

    ledger = AccountLedger.from_trades(
        Decimal("100000"),
        tuple(trades),
    )

    assert ledger.calculated_ending_cash == Decimal(
        "122853.4912016855438425046790"
    )
    assert ledger.net_trade_cash_flow == Decimal(
        "22853.4912016855438425046790"
    )


def test_completed_backtest_rejects_cash_reconciliation_mismatch() -> None:
    """Correct equity cannot conceal an unexplained ending-cash difference."""

    timestamp = datetime(2025, 1, 1, tzinfo=UTC)
    trade = SimulatedTrade(
        symbol="TEST",
        timestamp=timestamp,
        side=TradeSide.BUY,
        quantity=2,
        price=Decimal("10"),
    )

    with pytest.raises(ValueError, match="zero reconciliation difference"):
        BacktestResult(
            initial_cash=Decimal("100"),
            final_cash=Decimal("79"),
            final_equity=Decimal("104"),
            final_position_quantity=2,
            trades=(trade,),
            equity_curve=(EquityPoint(timestamp=timestamp, equity=Decimal("104")),),
            final_position_mark_price=Decimal("12"),
        )


def test_completed_backtest_rejects_equity_reconciliation_mismatch() -> None:
    """Correct cash cannot conceal an incorrectly valued ending position."""

    timestamp = datetime(2025, 1, 1, tzinfo=UTC)
    trade = SimulatedTrade(
        symbol="TEST",
        timestamp=timestamp,
        side=TradeSide.BUY,
        quantity=2,
        price=Decimal("10"),
    )

    with pytest.raises(ValueError, match="zero reconciliation difference"):
        BacktestResult(
            initial_cash=Decimal("100"),
            final_cash=Decimal("80"),
            final_equity=Decimal("103"),
            final_position_quantity=2,
            trades=(trade,),
            equity_curve=(EquityPoint(timestamp=timestamp, equity=Decimal("103")),),
            final_position_mark_price=Decimal("12"),
        )
