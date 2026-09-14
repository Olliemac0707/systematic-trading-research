"""Tests for deterministic, side-effect-free plain-text backtest reports."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Never

import pytest

from trading_research.models import (
    AccountLedger,
    AccountReconciliation,
    BacktestResult,
    EquityPoint,
    SimulatedTrade,
    TradeSide,
)
from trading_research.reporting import (
    format_backtest_report,
    format_decimal_currency,
)

_START = datetime(2025, 1, 1, 14, 30, tzinfo=UTC)
_STARTING_CASH = Decimal("1000")


def execution(
    side: TradeSide,
    offset: timedelta,
    price: Decimal,
    *,
    quantity: int = 2,
    commission: Decimal = Decimal("0"),
    reference_price: Decimal | None = None,
) -> SimulatedTrade:
    """Create one exact simulated execution for report tests."""

    return SimulatedTrade(
        symbol="TEST",
        timestamp=_START + offset,
        side=side,
        quantity=quantity,
        price=price,
        commission=commission,
        reference_price=reference_price,
    )


def make_result(
    trades: tuple[SimulatedTrade, ...],
    *,
    final_mark_price: Decimal = Decimal("0"),
    equity_values: tuple[Decimal, ...] | None = None,
) -> BacktestResult:
    """Build a reconciled result directly from its immutable account ledger."""

    ledger = AccountLedger.from_trades(_STARTING_CASH, trades)
    ending_cash = ledger.calculated_ending_cash
    open_quantity = ledger.ending_position_quantity
    ending_equity = ending_cash + final_mark_price * open_quantity
    equity_curve: tuple[EquityPoint, ...]
    if equity_values is None:
        final_timestamp = trades[-1].timestamp + timedelta(days=1) if trades else _START
        equity_curve = (EquityPoint(final_timestamp, ending_equity),)
    else:
        timestamps = [
            _START + timedelta(days=index) for index in range(len(equity_values))
        ]
        if trades and trades[-1].timestamp > timestamps[-1]:
            timestamps[-1] = trades[-1].timestamp
        equity_curve = tuple(
            EquityPoint(timestamp, equity)
            for timestamp, equity in zip(timestamps, equity_values, strict=True)
        )
    return BacktestResult(
        initial_cash=_STARTING_CASH,
        final_cash=ending_cash,
        final_equity=ending_equity,
        final_position_quantity=open_quantity,
        trades=trades,
        equity_curve=equity_curve,
        final_position_mark_price=final_mark_price if open_quantity else None,
    )


def make_canonical_result() -> BacktestResult:
    """Create one closed winner and one open marked position."""

    trades = (
        execution(
            TradeSide.BUY,
            timedelta(0),
            Decimal("100.50"),
            commission=Decimal("1"),
            reference_price=Decimal("100"),
        ),
        execution(
            TradeSide.SELL,
            timedelta(days=2, hours=12),
            Decimal("109.45"),
            commission=Decimal("1"),
            reference_price=Decimal("110"),
        ),
        execution(
            TradeSide.BUY,
            timedelta(days=3),
            Decimal("50.25"),
            commission=Decimal("0.50"),
            reference_price=Decimal("50"),
        ),
    )
    return make_result(
        trades,
        final_mark_price=Decimal("55"),
        equity_values=(
            Decimal("1000"),
            Decimal("1050"),
            Decimal("1000"),
            Decimal("1060"),
            Decimal("1024.90"),
        ),
    )


def report_value(report: str, label: str) -> str:
    """Return the trimmed value from one labelled report line."""

    prefix = f"{label}:"
    return next(
        line[len(prefix) :].strip()
        for line in report.splitlines()
        if line.startswith(prefix)
    )


def test_core_report_displays_exact_account_pnl_risk_and_trade_values() -> None:
    """The canonical report distinguishes every required financial category."""

    report = format_backtest_report(make_canonical_result(), currency_symbol="$")

    assert isinstance(report, str)
    assert "BACKTEST PERFORMANCE REPORT" in report
    assert report_value(report, "Symbol") == "TEST"
    assert report_value(report, "Market bars") == "5"
    assert report_value(report, "Strategy") == "Not available"
    assert report_value(report, "Open position remains") == "Yes"
    assert report_value(report, "Starting cash") == "$1,000.00"
    assert report_value(report, "Ending cash") == "$914.90"
    assert report_value(report, "Open-position market value") == "$110.00"
    assert report_value(report, "Ending equity") == "$1,024.90"
    assert report_value(report, "Net profit") == "$24.90"
    assert report_value(report, "Total return") == "2.49%"
    assert report_value(report, "Equity composition") == (
        "Ending cash + open-position market value"
    )
    assert report_value(report, "Gross realised P&L") == "$17.90"
    assert report_value(report, "Unrealised P&L") == "$9.50"
    assert report_value(report, "Total commissions") == "$2.50"
    assert report_value(report, "Adverse slippage attribution") == "$2.60"
    assert report_value(report, "Maximum absolute drawdown") == "$50.00"
    assert report_value(report, "Maximum percentage drawdown") == "-4.76%"
    assert report_value(report, "Drawdown peak timestamp") == (
        _START + timedelta(days=1)
    ).isoformat()
    assert report_value(report, "Drawdown trough timestamp") == (
        _START + timedelta(days=2)
    ).isoformat()
    assert report_value(report, "Drawdown recovery timestamp") == (
        _START + timedelta(days=3)
    ).isoformat()
    assert report_value(report, "Completed trades") == "1"
    assert report_value(report, "Winning trades") == "1"
    assert report_value(report, "Net closed-trade P&L") == "$15.90"
    assert report_value(report, "Profit factor") == "Not available"
    assert report_value(report, "Trade commissions") == "$2.00"
    assert report_value(report, "Trade slippage attribution") == "$2.10"
    assert report_value(report, "Open quantity") == "2"
    assert report_value(report, "Cash reconciliation difference") == "$0.00"
    assert report_value(report, "Equity reconciliation difference") == "$0.00"
    assert report_value(report, "Reconciliation status") == "PASS"
    assert "Exposure and Capital Use" in report
    assert "Order Decisions" in report
    assert report_value(report, "Equity observations") == "5"
    assert report_value(report, "Time in market (observations)") == "100.00%"
    assert report_value(report, "Average position value") == "$182.22"
    assert report_value(report, "Maximum position value") == "$252.00"
    assert report_value(report, "Average cash") == "$844.76"
    assert report_value(report, "Minimum cash") == "$798.00"
    assert report_value(report, "Actionable proposals") == "0"
    assert report_value(report, "Execution approval rate") == "Not available"
    assert "Slippage is already reflected in execution fill prices" in report


@pytest.mark.parametrize(
    ("value", "symbol", "expected"),
    [
        (Decimal("12.345"), None, "12.34"),
        (Decimal("-325.4"), None, "-325.40"),
        (Decimal("-0"), None, "0.00"),
        (Decimal("10099.255"), None, "10,099.26"),
        (Decimal("-12.345"), "$", "-$12.34"),
    ],
    ids=["positive-half-even", "negative", "negative-zero", "thousands", "symbol"],
)
def test_currency_formatting_is_exact_and_deterministic(
    value: Decimal,
    symbol: str | None,
    expected: str,
) -> None:
    """Currency display uses Decimal, separators, and half-even rounding."""

    assert format_decimal_currency(value, symbol) == expected


@pytest.mark.parametrize("symbol", ["", " ", "\n", "\x00"])
def test_invalid_currency_symbols_are_rejected(symbol: str) -> None:
    """Empty, whitespace, and control-character symbols cannot alter layout."""

    with pytest.raises(ValueError, match="currency_symbol"):
        format_backtest_report(make_canonical_result(), currency_symbol=symbol)


def test_invalid_formatter_inputs_are_rejected() -> None:
    """The public formatting boundary rejects wrong and non-finite values."""

    with pytest.raises(TypeError, match="Decimal"):
        format_decimal_currency(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        format_decimal_currency(Decimal("NaN"))
    with pytest.raises(TypeError, match="currency_symbol"):
        format_decimal_currency(Decimal("1"), 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="BacktestResult"):
        format_backtest_report(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="include_closed_trades"):
        format_backtest_report(make_canonical_result(), include_closed_trades=1)  # type: ignore[arg-type]


def test_no_trade_report_uses_missing_value_and_zero_policies() -> None:
    """An inactive result reports zero account metrics without fake statistics."""

    report = format_backtest_report(make_result(()))

    assert report_value(report, "Symbol") == "Not available"
    assert report_value(report, "Start") == _START.isoformat()
    assert report_value(report, "End") == _START.isoformat()
    assert report_value(report, "Total return") == "0.00%"
    assert report_value(report, "Maximum absolute drawdown") == "0.00"
    assert report_value(report, "Maximum percentage drawdown") == "0.00%"
    assert report_value(report, "Drawdown peak timestamp") == "Not available"
    assert report_value(report, "Drawdown trough timestamp") == "Not available"
    assert report_value(report, "Drawdown recovery timestamp") == "Not available"
    assert report_value(report, "Win rate") == "Not available"
    assert report_value(report, "Time in market (observations)") == "0.00%"
    assert report_value(report, "Average position quantity") == "0.00"
    assert report_value(report, "Actionable proposals") == "0"
    assert report_value(report, "Execution approval rate") == "Not available"
    assert report_value(report, "Average holding period") == "Not available"
    assert "No open position" in report
    assert "None" not in report


def test_open_position_without_closed_trades_is_explicit() -> None:
    """An unmatched buy remains open and does not become a completed trade."""

    result = make_result(
        (
            execution(
                TradeSide.BUY,
                timedelta(0),
                Decimal("100"),
                commission=Decimal("1"),
            ),
        ),
        final_mark_price=Decimal("110"),
        equity_values=(Decimal("1000"), Decimal("1019")),
    )

    report = format_backtest_report(result, include_closed_trades=True)

    assert report_value(report, "Ending cash") == "799.00"
    assert report_value(report, "Ending equity") == "1,019.00"
    assert report_value(report, "Net profit") == "19.00"
    assert report_value(report, "Gross realised P&L") == "0.00"
    assert report_value(report, "Unrealised P&L") == "20.00"
    assert report_value(report, "Completed trades") == "0"
    assert report_value(report, "Open quantity") == "2"
    assert "No completed trades" in report


def test_negative_return_and_unrecovered_drawdown_are_clear() -> None:
    """Losses retain their sign and unrecovered risk is stated in words."""

    result = make_result(
        (
            execution(TradeSide.BUY, timedelta(0), Decimal("100")),
            execution(TradeSide.SELL, timedelta(days=2), Decimal("90")),
        ),
        equity_values=(Decimal("1000"), Decimal("1050"), Decimal("980")),
    )

    report = format_backtest_report(result)

    assert report_value(report, "Total return") == "-2.00%"
    assert report_value(report, "Net profit") == "-20.00"
    assert report_value(report, "Gross loss") == "20.00"
    assert report_value(report, "Largest loser") == "20.00"
    assert report_value(report, "Maximum absolute drawdown") == "70.00"
    assert report_value(report, "Drawdown recovery timestamp") == (
        "Not recovered by end of backtest"
    )
    assert "No open position" in report


def test_breakeven_only_report_keeps_undefined_ratios_unavailable() -> None:
    """Exact breakeven is counted while profit and payoff ratios stay undefined."""

    result = make_result(
        (
            execution(
                TradeSide.BUY,
                timedelta(0),
                Decimal("100"),
                commission=Decimal("1"),
            ),
            execution(
                TradeSide.SELL,
                timedelta(days=1),
                Decimal("101"),
                commission=Decimal("1"),
            ),
        ),
        equity_values=(Decimal("1000"), Decimal("1000")),
    )

    report = format_backtest_report(result)

    assert report_value(report, "Completed trades") == "1"
    assert report_value(report, "Breakeven trades") == "1"
    assert report_value(report, "Win rate") == "0.00%"
    assert report_value(report, "Net closed-trade P&L") == "0.00"
    assert report_value(report, "Profit factor") == "Not available"
    assert report_value(report, "Payoff ratio") == "Not available"
    assert report_value(report, "Expectancy") == "0.00"


def test_percentage_timestamp_and_subday_duration_formatting() -> None:
    """Ratios are display percentages and durations retain sub-day components."""

    result = make_result(
        (
            execution(TradeSide.BUY, timedelta(0), Decimal("100")),
            execution(
                TradeSide.SELL,
                timedelta(days=2, hours=4, minutes=30),
                Decimal("112.50"),
            ),
        ),
        equity_values=(Decimal("1000"), Decimal("1025")),
    )

    report = format_backtest_report(result)

    assert report_value(report, "Total return") == "2.50%"
    assert report_value(report, "Start") == _START.isoformat()
    assert report_value(report, "Average holding period") == "2 days, 04:30:00"
    assert report_value(report, "Shortest holding period") == "2 days, 04:30:00"
    assert report_value(report, "Longest holding period") == "2 days, 04:30:00"


def test_closed_trade_details_are_opt_in_ordered_and_exclude_open_entry() -> None:
    """Optional rows use reconstructed trades and omit the unmatched final buy."""

    trades = (
        execution(TradeSide.BUY, timedelta(0), Decimal("10")),
        execution(TradeSide.SELL, timedelta(days=2), Decimal("15")),
        execution(TradeSide.BUY, timedelta(days=3), Decimal("20")),
        execution(TradeSide.SELL, timedelta(days=7), Decimal("18")),
        execution(TradeSide.BUY, timedelta(days=8), Decimal("12")),
    )
    result = make_result(trades, final_mark_price=Decimal("13"))

    summary_only = format_backtest_report(result)
    with_details = format_backtest_report(result, include_closed_trades=True)

    assert "Closed Trade Details" not in summary_only
    assert "Closed Trade Details" in with_details
    first_entry = trades[0].timestamp.isoformat()
    second_entry = trades[2].timestamp.isoformat()
    open_entry = trades[4].timestamp.isoformat()
    assert with_details.index(first_entry) < with_details.index(second_entry)
    assert open_entry not in with_details
    assert "2 | 10.00 | 15.00 | 10.00 | 50.00% | 2 days, 00:00:00" in with_details
    assert "2 | 20.00 | 18.00 | -4.00 | -10.00% | 4 days, 00:00:00" in with_details


def test_report_is_deterministic_and_has_no_output_or_file_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Formatting neither mutates the result nor performs observable I/O."""

    result = make_canonical_result()
    before = repr(result)
    monkeypatch.chdir(tmp_path)

    first = format_backtest_report(result, include_closed_trades=True)
    second = format_backtest_report(result, include_closed_trades=True)

    captured = capsys.readouterr()
    assert first == second
    assert repr(result) == before
    assert captured.out == ""
    assert captured.err == ""
    assert tuple(tmp_path.iterdir()) == ()


def test_report_formatting_does_not_convert_through_float(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Display arithmetic remains Decimal-only even at rounding boundaries."""

    def reject_float(*_: object, **__: object) -> Never:
        raise AssertionError("float conversion is forbidden")

    monkeypatch.setattr("builtins.float", reject_float)

    assert format_decimal_currency(Decimal("10099.255")) == "10,099.26"
    assert "2.49%" in format_backtest_report(make_canonical_result())


def test_report_rejects_a_stored_nonzero_reconciliation() -> None:
    """Defensive reporting validation cannot label a mismatch as PASS."""

    result = make_canonical_result()
    reconciliation = result.reconciliation
    bad_reconciliation = AccountReconciliation(
        ledger=reconciliation.ledger,
        ending_cash=reconciliation.ending_cash + Decimal("1"),
        ending_position_mark_price=reconciliation.ending_position_mark_price,
        ending_equity=reconciliation.ending_equity + Decimal("1"),
    )
    object.__setattr__(result, "reconciliation", bad_reconciliation)

    with pytest.raises(ValueError, match="exactly reconciled"):
        format_backtest_report(result)
