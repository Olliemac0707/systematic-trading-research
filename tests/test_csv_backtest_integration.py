"""End-to-end test from local CSV observations to a backtest result."""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from trading_research.backtesting import SimpleBacktestEngine
from trading_research.config import BacktestConfig
from trading_research.data import CsvMarketDataProvider
from trading_research.models import SignalAction, TradeSide
from trading_research.strategies import SimpleMovingAverageCrossover


def test_csv_to_sma_backtest_pipeline(tmp_path: Path) -> None:
    """CSV bars produce close-based signals and next-open simulated fills."""

    path = tmp_path / "integration.csv"
    prices = [10, 10, 10, 12, 13, 11, 10, 9]
    rows = ["timestamp,symbol,open,high,low,close,volume"]
    rows.extend(
        f"2025-01-{index + 1:02d}T00:00:00+00:00,TEST,{price},{price},{price},{price},100"
        for index, price in enumerate(prices)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    bars = CsvMarketDataProvider(path).get_historical_bars("TEST", None, None)
    strategy = SimpleMovingAverageCrossover(short_window=2, long_window=3)
    signals = strategy.generate_signals(bars)
    config = BacktestConfig(
        initial_cash=Decimal("1000"),
        trade_quantity=2,
        commission_bps=Decimal("100"),
        slippage_bps=Decimal("100"),
    )

    result = SimpleBacktestEngine().run(bars, strategy, config)

    assert [signal.action for signal in signals] == [SignalAction.BUY, SignalAction.SELL]
    assert signals[0].timestamp == datetime(2025, 1, 4, tzinfo=UTC)
    assert [trade.timestamp for trade in result.trades] == [
        datetime(2025, 1, 5, tzinfo=UTC),
        datetime(2025, 1, 8, tzinfo=UTC),
    ]
    assert [trade.side for trade in result.trades] == [TradeSide.BUY, TradeSide.SELL]
    assert [trade.quantity for trade in result.trades] == [2, 2]
    assert [trade.price for trade in result.trades] == [Decimal("13.13"), Decimal("8.91")]
    assert [trade.commission for trade in result.trades] == [
        Decimal("0.2626"),
        Decimal("0.1782"),
    ]
    assert result.final_cash == Decimal("991.1192")
    assert result.final_equity == Decimal("991.1192")
    assert result.final_position_quantity == 0
    assert result.equity_curve[-1].equity == result.final_equity
    assert result.reconciliation.ledger.total_slippage_cost == Decimal("0.44")
    assert result.reconciliation.cash_difference == Decimal("0")
    assert result.reconciliation.reconciliation_difference == Decimal("0")
