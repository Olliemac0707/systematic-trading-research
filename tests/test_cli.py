"""Tests for the local, simulated backtest command-line interface."""

import csv
import json
import socket
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import Never

import pytest

import trading_research.cli as cli
from trading_research.backtesting import SimpleBacktestEngine
from trading_research.data import CsvMarketDataProvider
from trading_research.errors import BacktestError
from trading_research.models import BacktestResult, MarketBar
from trading_research.reporting import StrategyRunConfiguration

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_DATA = _ROOT / "data" / "sample_prices.csv"
_DONCHIAN_DATA = _ROOT / "tests" / "fixtures" / "donchian_prices.csv"


def backtest_arguments(
    *,
    data: Path = _SAMPLE_DATA,
    symbol: str = "DEMO",
    strategy: str | None = None,
    fast_window: str | None = "2",
    slow_window: str | None = "3",
    entry_window: str | None = None,
    exit_window: str | None = None,
    starting_cash: str = "10000",
    position_size_mode: str = "fixed",
    quantity: str | None = "10",
    cash_allocation_ratio: str | None = None,
    commission_bps: str = "1",
    slippage_bps: str = "5",
    maximum_position_value: str | None = None,
    minimum_cash_reserve: str | None = None,
    end_of_test: str | None = None,
    periods_per_year: str | None = None,
    risk_free_rate_per_period: str | None = None,
    target_return_per_period: str | None = None,
    start: str | None = None,
    end: str | None = None,
    include_closed_trades: bool = False,
    currency_symbol: str | None = None,
    json_report: Path | None = None,
    summary_csv: Path | None = None,
    equity_csv: Path | None = None,
    closed_trades_csv: Path | None = None,
    decisions_csv: Path | None = None,
    benchmark: str | None = None,
    benchmark_equity_csv: Path | None = None,
    overwrite_reports: bool = False,
) -> list[str]:
    """Build a complete valid argument list with optional overrides."""

    arguments = [
        "backtest",
        "--data",
        str(data),
        "--symbol",
        symbol,
        "--starting-cash",
        starting_cash,
        "--position-size-mode",
        position_size_mode,
        "--commission-bps",
        commission_bps,
        "--slippage-bps",
        slippage_bps,
    ]
    if strategy is not None:
        arguments.extend(("--strategy", strategy))
    if fast_window is not None:
        arguments.extend(("--fast-window", fast_window))
    if slow_window is not None:
        arguments.extend(("--slow-window", slow_window))
    if entry_window is not None:
        arguments.extend(("--entry-window", entry_window))
    if exit_window is not None:
        arguments.extend(("--exit-window", exit_window))
    if quantity is not None:
        arguments.extend(("--quantity", quantity))
    if cash_allocation_ratio is not None:
        arguments.extend(("--cash-allocation-ratio", cash_allocation_ratio))
    if maximum_position_value is not None:
        arguments.extend(("--maximum-position-value", maximum_position_value))
    if minimum_cash_reserve is not None:
        arguments.extend(("--minimum-cash-reserve", minimum_cash_reserve))
    if end_of_test is not None:
        arguments.extend(("--end-of-test", end_of_test))
    if periods_per_year is not None:
        arguments.extend(("--periods-per-year", periods_per_year))
    if risk_free_rate_per_period is not None:
        arguments.extend(("--risk-free-rate-per-period", risk_free_rate_per_period))
    if target_return_per_period is not None:
        arguments.extend(("--target-return-per-period", target_return_per_period))
    if start is not None:
        arguments.extend(("--start", start))
    if end is not None:
        arguments.extend(("--end", end))
    if include_closed_trades:
        arguments.append("--include-closed-trades")
    if currency_symbol is not None:
        arguments.extend(("--currency-symbol", currency_symbol))
    if json_report is not None:
        arguments.extend(("--json-report", str(json_report)))
    if summary_csv is not None:
        arguments.extend(("--summary-csv", str(summary_csv)))
    if equity_csv is not None:
        arguments.extend(("--equity-csv", str(equity_csv)))
    if closed_trades_csv is not None:
        arguments.extend(("--closed-trades-csv", str(closed_trades_csv)))
    if decisions_csv is not None:
        arguments.extend(("--decisions-csv", str(decisions_csv)))
    if benchmark is not None:
        arguments.extend(("--benchmark", benchmark))
    if benchmark_equity_csv is not None:
        arguments.extend(("--benchmark-equity-csv", str(benchmark_equity_csv)))
    if overwrite_reports:
        arguments.append("--overwrite-reports")
    return arguments


def report_value(report: str, label: str) -> str:
    """Return one trimmed value from the generated plain-text report."""

    prefix = f"{label}:"
    return next(
        line[len(prefix) :].strip()
        for line in report.splitlines()
        if line.startswith(prefix)
    )


@pytest.mark.parametrize(
    ("arguments", "expected_text"),
    [
        (["--help"], "{backtest,experiment,evaluate,holdout,dataset}"),
        (["backtest", "--help"], "--starting-cash"),
    ],
    ids=["top-level", "backtest"],
)
def test_module_entrypoint_help(
    arguments: list[str],
    expected_text: str,
) -> None:
    """Both required module help forms exit cleanly without stderr output."""

    completed = subprocess.run(
        [sys.executable, "-m", "trading_research", *arguments],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == cli.EXIT_SUCCESS
    assert expected_text in completed.stdout
    assert completed.stderr == ""


def test_backtest_help_exposes_unambiguous_sizing_and_risk_controls() -> None:
    """Help names basis points explicitly and lists each new simulated control."""

    completed = subprocess.run(
        [sys.executable, "-m", "trading_research", "backtest", "--help"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == cli.EXIT_SUCCESS
    assert "--commission-bps" in completed.stdout
    assert "--strategy {sma-crossover,donchian-breakout}" in completed.stdout
    assert "--entry-window" in completed.stdout
    assert "--exit-window" in completed.stdout
    assert "--position-size-mode" in completed.stdout
    assert "--cash-allocation-ratio" in completed.stdout
    assert "--maximum-position-value" in completed.stdout
    assert "--minimum-cash-reserve" in completed.stdout
    assert "--end-of-test {hold,liquidate}" in completed.stdout
    assert "final close" in completed.stdout
    assert "--periods-per-year" in completed.stdout
    assert "--risk-free-rate-per-period" in completed.stdout
    assert "--target-return-per-period" in completed.stdout
    assert "--json-report" in completed.stdout
    assert "--summary-csv" in completed.stdout
    assert "--equity-csv" in completed.stdout
    assert "--closed-trades-csv" in completed.stdout
    assert "--decisions-csv" in completed.stdout
    assert "--benchmark {none,buy-and-hold}" in completed.stdout
    assert "--benchmark-equity-csv" in completed.stdout
    assert "--overwrite-reports" in completed.stdout
    assert "per-period decimal return ratio" in completed.stdout
    assert "--commission " not in completed.stdout
    assert completed.stderr == ""


def test_module_entrypoint_runs_sample_backtest() -> None:
    """The true module entry point executes the committed synthetic dataset."""

    completed = subprocess.run(
        [sys.executable, "-m", "trading_research", *backtest_arguments()],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == cli.EXIT_SUCCESS
    assert "BACKTEST PERFORMANCE REPORT" in completed.stdout
    assert report_value(completed.stdout, "Symbol") == "DEMO"
    assert report_value(completed.stdout, "Market bars") == "8"
    assert report_value(completed.stdout, "Strategy") == "SMA crossover"
    assert report_value(completed.stdout, "Total commissions") == "0.02"
    assert str(_SAMPLE_DATA) not in completed.stdout
    assert completed.stderr == ""


def test_donchian_cli_runs_hold_and_liquidation_scenarios(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The selected strategy reports only its channel parameters under both policies."""

    shared = {
        "data": _DONCHIAN_DATA,
        "symbol": "CHAN",
        "strategy": "donchian-breakout",
        "fast_window": None,
        "slow_window": None,
        "entry_window": "2",
        "exit_window": "2",
        "end": "2025-02-04T00:00:00+00:00",
    }
    hold_exit = cli.main(
        backtest_arguments(**shared, end_of_test="hold")  # type: ignore[arg-type]
    )
    hold = capsys.readouterr()
    liquidate_exit = cli.main(
        backtest_arguments(
            **shared,  # type: ignore[arg-type]
            end_of_test="liquidate",
        )
    )
    liquidate = capsys.readouterr()

    assert hold_exit == liquidate_exit == cli.EXIT_SUCCESS
    assert report_value(hold.out, "Strategy") == "Donchian breakout"
    assert report_value(hold.out, "Entry window") == "2"
    assert report_value(hold.out, "Exit window") == "2"
    assert "Fast window:" not in hold.out
    assert "Slow window:" not in hold.out
    assert report_value(hold.out, "Open position remains") == "Yes"
    assert report_value(liquidate.out, "Open position remains") == "No"
    assert report_value(liquidate.out, "Ending cash") == report_value(
        liquidate.out,
        "Ending equity",
    )
    assert hold.err == liquidate.err == ""


def test_donchian_cli_accepts_exact_entry_warmup_boundary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Entry-window prior closes plus the current close are sufficient data."""

    exit_code = cli.main(
        backtest_arguments(
            data=_DONCHIAN_DATA,
            symbol="CHAN",
            strategy="donchian-breakout",
            fast_window=None,
            slow_window=None,
            entry_window="2",
            exit_window="2",
            end="2025-02-03T00:00:00+00:00",
        )
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_SUCCESS
    assert report_value(captured.out, "Market bars") == "3"
    assert report_value(captured.out, "Actionable proposals") == "0"
    assert captured.err == ""


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "strategy": "donchian-breakout",
                "entry_window": "2",
                "exit_window": "2",
            },
            "apply only to sma-crossover",
        ),
        (
            {
                "strategy": "donchian-breakout",
                "fast_window": None,
                "slow_window": None,
                "entry_window": None,
                "exit_window": "2",
            },
            "are required for donchian-breakout",
        ),
        ({"entry_window": "2"}, "apply only to donchian-breakout"),
        ({"fast_window": None}, "are required for sma-crossover"),
        ({"strategy": "module:strategy"}, "invalid choice"),
    ],
)
def test_cli_rejects_unknown_mixed_and_missing_strategy_parameters(
    overrides: dict[str, object],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid strategy flag combinations are deterministic usage errors."""

    exit_code = cli.main(backtest_arguments(**overrides))  # type: ignore[arg-type]

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_USAGE_ERROR
    assert captured.out == ""
    assert message in captured.err
    assert "Traceback" not in captured.err


def test_donchian_cli_exports_generic_metadata_and_runs_benchmark(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """JSON and CSV contain only relevant parameters in a benchmarked run."""

    json_path = tmp_path / "donchian.json"
    summary_path = tmp_path / "donchian.csv"
    benchmark_path = tmp_path / "benchmark.csv"
    exit_code = cli.main(
        backtest_arguments(
            data=_DONCHIAN_DATA,
            symbol="CHAN",
            strategy="donchian-breakout",
            fast_window=None,
            slow_window=None,
            entry_window="2",
            exit_window="2",
            benchmark="buy-and-hold",
            periods_per_year="252",
            json_report=json_path,
            summary_csv=summary_path,
            equity_csv=tmp_path / "equity.csv",
            closed_trades_csv=tmp_path / "trades.csv",
            decisions_csv=tmp_path / "decisions.csv",
            benchmark_equity_csv=benchmark_path,
        )
    )

    captured = capsys.readouterr()
    json_text = json_path.read_text(encoding="utf-8")
    payload = json.loads(json_text)
    row = next(csv.DictReader(StringIO(summary_path.read_text(encoding="utf-8"))))

    assert exit_code == cli.EXIT_SUCCESS
    assert payload["schema_version"] == "1.3"
    assert payload["metadata"]["strategy"] == {
        "name": "donchian-breakout",
        "parameters": {"entry_window": 2, "exit_window": 2},
    }
    assert payload["benchmark"]["name"] == "buy-and-hold"
    assert isinstance(payload["backtest_result"]["final_cash"], str)
    assert str(_ROOT) not in json_text
    assert row["strategy_name"] == "donchian-breakout"
    assert row["fast_window"] == row["slow_window"] == ""
    assert row["entry_window"] == row["exit_window"] == "2"
    assert all(
        path.exists()
        for path in (
            json_path,
            summary_path,
            tmp_path / "equity.csv",
            tmp_path / "trades.csv",
            tmp_path / "decisions.csv",
            benchmark_path,
        )
    )
    assert report_value(captured.out, "Strategy") == "Donchian breakout"
    assert captured.err == ""


def test_repeated_equivalent_donchian_cli_runs_are_deterministic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Equivalent inputs produce byte-identical human-readable output."""

    arguments = backtest_arguments(
        data=_DONCHIAN_DATA,
        symbol="CHAN",
        strategy="donchian-breakout",
        fast_window=None,
        slow_window=None,
        entry_window="2",
        exit_window="2",
    )
    first_exit = cli.main(arguments)
    first = capsys.readouterr()
    second_exit = cli.main(arguments)
    second = capsys.readouterr()

    assert first_exit == second_exit == cli.EXIT_SUCCESS
    assert first.out == second.out
    assert first.err == second.err == ""


def test_success_prints_one_summary_report_without_closed_trade_rows(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A normal direct call prints only one completed summary report."""

    exit_code = cli.main(backtest_arguments())

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_SUCCESS
    assert captured.out.count("BACKTEST PERFORMANCE REPORT") == 1
    assert "Reconciliation status:" in captured.out
    assert "PASS" in captured.out
    assert "Closed Trade Details" not in captured.out
    assert report_value(captured.out, "End-of-test policy") == "Hold open position"
    assert captured.err == ""


def test_cli_hold_and_liquidate_final_open_position(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both explicit policies succeed and liquidation removes the final long."""

    end = "2025-01-05T00:00:00+00:00"
    hold_exit = cli.main(backtest_arguments(end=end, end_of_test="hold"))
    hold = capsys.readouterr()
    liquidation_exit = cli.main(
        backtest_arguments(
            end=end,
            end_of_test="liquidate",
            include_closed_trades=True,
        )
    )
    liquidation = capsys.readouterr()

    assert hold_exit == liquidation_exit == cli.EXIT_SUCCESS
    assert report_value(hold.out, "End-of-test policy") == "Hold open position"
    assert report_value(hold.out, "Open position remains") == "Yes"
    assert report_value(hold.out, "Ending cash") != report_value(
        hold.out,
        "Ending equity",
    )
    assert (
        report_value(liquidation.out, "End-of-test policy")
        == "Liquidate at final close"
    )
    assert report_value(liquidation.out, "Open position remains") == "No"
    assert report_value(liquidation.out, "Ending cash") == report_value(
        liquidation.out,
        "Ending equity",
    )
    assert "Final open position was synthetically liquidated" in liquidation.out
    assert "End-of-test liquidation" in liquidation.out
    assert hold.err == liquidation.err == ""


def test_invalid_end_of_test_policy_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Argparse restricts the public end policy to hold or liquidate."""

    exit_code = cli.main(backtest_arguments(end_of_test="sell"))

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_USAGE_ERROR
    assert captured.out == ""
    assert "invalid choice" in captured.err
    assert "hold" in captured.err
    assert "liquidate" in captured.err
    assert "Traceback" not in captured.err


def test_cli_periodic_and_annualised_return_metrics_are_explicit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Per-period settings opt in while annual values require their factor."""

    periodic_exit = cli.main(
        backtest_arguments(risk_free_rate_per_period="0.0001")
    )
    periodic = capsys.readouterr()
    annualised_exit = cli.main(
        backtest_arguments(
            periods_per_year="252",
            risk_free_rate_per_period="0",
            target_return_per_period="0",
        )
    )
    annualised = capsys.readouterr()

    assert periodic_exit == annualised_exit == cli.EXIT_SUCCESS
    assert "Return and Risk Statistics" in periodic.out
    assert report_value(periodic.out, "Periods per year") == "Not available"
    assert report_value(periodic.out, "Annualised volatility") == "Not available"
    assert report_value(periodic.out, "Risk-free rate per period") == "0.01%"
    assert "Return and Risk Statistics" in annualised.out
    assert report_value(annualised.out, "Periods per year") == "252"
    assert report_value(annualised.out, "Annualised volatility") != "Not available"
    assert periodic.err == annualised.err == ""


@pytest.mark.parametrize(
    ("argument_name", "argument_value"),
    [
        ("periods_per_year", "0"),
        ("periods_per_year", "-1"),
        ("periods_per_year", "NaN"),
        ("periods_per_year", "Infinity"),
        ("risk_free_rate_per_period", "NaN"),
        ("risk_free_rate_per_period", "Infinity"),
        ("target_return_per_period", "NaN"),
        ("target_return_per_period", "-Infinity"),
    ],
)
def test_invalid_cli_risk_metric_settings_are_usage_errors(
    argument_name: str,
    argument_value: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Annualisation and per-period rates reject invalid Decimals cleanly."""

    exit_code = cli.main(
        backtest_arguments(**{argument_name: argument_value})  # type: ignore[arg-type]
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_USAGE_ERROR
    assert captured.out == ""
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


def test_cash_allocation_mode_runs_existing_report_with_sized_quantity(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Allocation mode sizes whole shares before the existing simulated engine."""

    exit_code = cli.main(
        backtest_arguments(
            position_size_mode="cash-allocation",
            quantity=None,
            cash_allocation_ratio="0.25",
            include_closed_trades=True,
        )
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_SUCCESS
    assert "BACKTEST PERFORMANCE REPORT" in captured.out
    assert report_value(captured.out, "Completed trades") == "1"
    assert " | 192 | " in captured.out
    assert report_value(captured.out, "Reconciliation status") == "PASS"
    assert captured.err == ""


def test_cli_risk_controls_reduce_fixed_quantity_before_execution(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Position cap then reserve policies reduce the intended simulated order."""

    exit_code = cli.main(
        backtest_arguments(
            maximum_position_value="100",
            minimum_cash_reserve="9000",
            include_closed_trades=True,
        )
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_SUCCESS
    assert " | 7 | " in captured.out
    assert report_value(captured.out, "Reconciliation status") == "PASS"
    assert captured.err == ""


def test_removed_ambiguous_commission_flag_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The pre-release CLI removes rather than aliases the ambiguous old name."""

    arguments = backtest_arguments()
    arguments[arguments.index("--commission-bps")] = "--commission"

    exit_code = cli.main(arguments)

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_USAGE_ERROR
    assert captured.out == ""
    assert "required: --commission-bps" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("position_size_mode", "quantity", "cash_allocation_ratio", "message"),
    [
        ("fixed", None, None, "--quantity is required"),
        ("fixed", "10", "0.25", "only valid in cash-allocation"),
        ("cash-allocation", "10", "0.25", "only valid in fixed"),
        ("cash-allocation", None, None, "--cash-allocation-ratio is required"),
    ],
    ids=[
        "fixed-missing-quantity",
        "fixed-with-ratio",
        "allocation-with-quantity",
        "allocation-missing-ratio",
    ],
)
def test_incompatible_position_sizing_arguments_are_usage_errors(
    position_size_mode: str,
    quantity: str | None,
    cash_allocation_ratio: str | None,
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Only parameters meaningful to the selected sizing mode are accepted."""

    exit_code = cli.main(
        backtest_arguments(
            position_size_mode=position_size_mode,
            quantity=quantity,
            cash_allocation_ratio=cash_allocation_ratio,
        )
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_USAGE_ERROR
    assert captured.out == ""
    assert message in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    "ratio",
    ["0", "-0.1", "1.01", "NaN", "Infinity"],
)
def test_invalid_cli_cash_allocation_ratios_are_usage_errors(
    ratio: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Allocation mode accepts only exact finite ratios in ``(0, 1]``."""

    exit_code = cli.main(
        backtest_arguments(
            position_size_mode="cash-allocation",
            quantity=None,
            cash_allocation_ratio=ratio,
        )
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_USAGE_ERROR
    assert captured.out == ""
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


def test_invalid_cli_risk_amounts_are_usage_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Position caps are positive and cash reserves are non-negative Decimals."""

    maximum_exit = cli.main(backtest_arguments(maximum_position_value="0"))
    maximum = capsys.readouterr()
    reserve_exit = cli.main(backtest_arguments(minimum_cash_reserve="-1"))
    reserve = capsys.readouterr()

    assert maximum_exit == reserve_exit == cli.EXIT_USAGE_ERROR
    assert maximum.out == reserve.out == ""
    assert "must be positive" in maximum.err
    assert "must be non-negative" in reserve.err
    assert "Traceback" not in maximum.err + reserve.err


def test_closed_trade_details_are_included_only_when_requested(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The boolean detail flag is passed to the existing report formatter."""

    exit_code = cli.main(backtest_arguments(include_closed_trades=True))

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_SUCCESS
    assert "Closed Trade Details" in captured.out
    assert "Entry | Exit | Quantity" in captured.out
    assert captured.err == ""


def test_date_filters_are_inclusive_and_passed_to_the_provider(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Aware start and end values constrain the existing CSV provider."""

    exit_code = cli.main(
        backtest_arguments(
            start="2025-01-02T00:00:00+00:00",
            end="2025-01-06T00:00:00+00:00",
        )
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_SUCCESS
    assert report_value(captured.out, "Start") == "2025-01-02T00:00:00+00:00"
    assert report_value(captured.out, "End") == "2025-01-06T00:00:00+00:00"
    assert report_value(captured.out, "Market bars") == "5"
    assert captured.err == ""


def test_currency_symbol_changes_display_without_changing_accounting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A currency label changes formatting only, not the resulting equity."""

    neutral_exit = cli.main(backtest_arguments())
    neutral = capsys.readouterr()
    labelled_exit = cli.main(backtest_arguments(currency_symbol="$"))
    labelled = capsys.readouterr()

    assert neutral_exit == labelled_exit == cli.EXIT_SUCCESS
    assert report_value(neutral.out, "Ending equity") == "9,959.87"
    assert report_value(labelled.out, "Ending equity") == "$9,959.87"
    assert report_value(neutral.out, "Total return") == report_value(
        labelled.out,
        "Total return",
    )
    assert neutral.err == labelled.err == ""


@pytest.mark.parametrize(
    ("starting_cash", "expected"),
    [
        ("10000", "10,000.00"),
        ("10000.25", "10,000.25"),
    ],
    ids=["integer-like", "fractional"],
)
def test_valid_starting_cash_is_parsed_as_decimal(
    starting_cash: str,
    expected: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Integer-like and fractional text retain exact Decimal semantics."""

    exit_code = cli.main(backtest_arguments(starting_cash=starting_cash))

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_SUCCESS
    assert report_value(captured.out, "Starting cash") == expected
    assert captured.err == ""


@pytest.mark.parametrize(
    ("argument_name", "argument_value"),
    [
        ("starting_cash", "-1"),
        ("starting_cash", "0"),
        ("starting_cash", "NaN"),
        ("starting_cash", "Infinity"),
        ("starting_cash", "-Infinity"),
        ("commission_bps", "-1"),
        ("commission_bps", "NaN"),
        ("commission_bps", "Infinity"),
        ("commission_bps", "-Infinity"),
        ("commission_bps", "10000.01"),
        ("slippage_bps", "-1"),
        ("slippage_bps", "NaN"),
        ("slippage_bps", "Infinity"),
        ("slippage_bps", "-Infinity"),
        ("slippage_bps", "10000"),
    ],
)
def test_invalid_decimal_arguments_are_usage_errors_without_tracebacks(
    argument_name: str,
    argument_value: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Non-finite and out-of-range financial inputs fail during parsing."""

    overrides = {argument_name: argument_value}
    arguments = backtest_arguments(**overrides)  # type: ignore[arg-type]

    exit_code = cli.main(arguments)

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_USAGE_ERROR
    assert captured.out == ""
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


def test_decimal_parser_never_converts_through_float(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI financial text is converted directly to exact Decimal values."""

    def reject_float(*_: object, **__: object) -> Never:
        raise AssertionError("float conversion is forbidden")

    monkeypatch.setattr("builtins.float", reject_float)

    assert cli._parse_finite_decimal("10000.125") == Decimal("10000.125")


@pytest.mark.parametrize("currency_symbol", ["", "\n", "\x00"])
def test_invalid_currency_symbols_are_usage_errors(
    currency_symbol: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI reuses the report formatter's safe label validation."""

    exit_code = cli.main(backtest_arguments(currency_symbol=currency_symbol))

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_USAGE_ERROR
    assert captured.out == ""
    assert "currency_symbol" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("fast_window", "slow_window"),
    [
        ("0", "3"),
        ("-1", "3"),
        ("3", "3"),
        ("4", "3"),
    ],
    ids=["zero", "negative", "equal", "reversed"],
)
def test_invalid_strategy_windows_are_rejected(
    fast_window: str,
    slow_window: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Window lengths must be positive and strictly ordered."""

    exit_code = cli.main(
        backtest_arguments(fast_window=fast_window, slow_window=slow_window)
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_USAGE_ERROR
    assert captured.out == ""
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("quantity", ["0", "-1", "1.5"])
def test_invalid_whole_share_quantity_is_rejected(
    quantity: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI preserves the engine's positive whole-share quantity policy."""

    exit_code = cli.main(backtest_arguments(quantity=quantity))

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_USAGE_ERROR
    assert captured.out == ""
    assert "error:" in captured.err


def test_insufficient_filtered_data_is_reported_clearly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI requires two fully formed moving-average comparisons."""

    exit_code = cli.main(
        backtest_arguments(end="2025-01-03T00:00:00+00:00")
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_DATA_ERROR
    assert captured.out == ""
    assert "insufficient market data" in captured.err
    assert "at least 4" in captured.err
    assert "Traceback" not in captured.err


def test_missing_file_and_directory_paths_are_usage_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Only an existing local regular file is accepted as market data."""

    missing_exit = cli.main(backtest_arguments(data=tmp_path / "missing.csv"))
    missing = capsys.readouterr()
    directory_exit = cli.main(backtest_arguments(data=tmp_path))
    directory = capsys.readouterr()

    assert missing_exit == directory_exit == cli.EXIT_USAGE_ERROR
    assert missing.out == directory.out == ""
    assert "does not exist" in missing.err
    assert "not a file" in directory.err
    assert "Traceback" not in missing.err + directory.err


def test_invalid_csv_and_symbol_mismatch_are_data_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Provider schema and symbol errors remain concise data failures."""

    invalid_csv = tmp_path / "invalid.csv"
    invalid_csv.write_text("bad,header\nvalue,value\n", encoding="utf-8")

    invalid_exit = cli.main(backtest_arguments(data=invalid_csv))
    invalid = capsys.readouterr()
    symbol_exit = cli.main(backtest_arguments(symbol="OTHER"))
    symbol = capsys.readouterr()

    assert invalid_exit == symbol_exit == cli.EXIT_DATA_ERROR
    assert invalid.out == symbol.out == ""
    assert "missing required columns" in invalid.err
    assert "not requested symbol" in symbol.err
    assert "Traceback" not in invalid.err + symbol.err


def test_missing_and_blank_symbols_are_usage_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A required nonblank symbol is validated using domain conventions."""

    arguments = backtest_arguments()
    symbol_index = arguments.index("--symbol")
    del arguments[symbol_index : symbol_index + 2]
    missing_exit = cli.main(arguments)
    missing = capsys.readouterr()
    blank_exit = cli.main(backtest_arguments(symbol=""))
    blank = capsys.readouterr()

    assert missing_exit == blank_exit == cli.EXIT_USAGE_ERROR
    assert missing.out == blank.out == ""
    assert "required" in missing.err
    assert "invalid symbol" in blank.err
    assert "Traceback" not in missing.err + blank.err


@pytest.mark.parametrize(
    ("start", "end", "expected_fragment"),
    [
        ("2025-01-01T00:00:00", None, "timezone offset"),
        (None, "2025-01-08T00:00:00", "timezone offset"),
        (
            "2025-01-08T00:00:00+00:00",
            "2025-01-01T00:00:00+00:00",
            "must not be after",
        ),
    ],
    ids=["naive-start", "naive-end", "reversed"],
)
def test_invalid_date_filters_are_usage_errors(
    start: str | None,
    end: str | None,
    expected_fragment: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Date filters require offsets and an ordered inclusive range."""

    exit_code = cli.main(backtest_arguments(start=start, end=end))

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_USAGE_ERROR
    assert captured.out == ""
    assert expected_fragment in captured.err
    assert "Traceback" not in captured.err


def test_empty_filtered_result_is_a_data_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A valid range with no matching observations cannot start a backtest."""

    exit_code = cli.main(
        backtest_arguments(start="2026-01-01T00:00:00+00:00")
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_DATA_ERROR
    assert captured.out == ""
    assert "no market bars match" in captured.err


def test_normal_run_does_not_modify_files_or_access_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The command reads one local file and has no persistence or network path."""

    local_data = tmp_path / "prices.csv"
    local_data.write_bytes(_SAMPLE_DATA.read_bytes())
    before_contents = local_data.read_bytes()
    before_timestamp = local_data.stat().st_mtime_ns
    before_entries = tuple(tmp_path.iterdir())

    def reject_network(*_: object, **__: object) -> Never:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(socket, "getaddrinfo", reject_network)

    first_exit = cli.main(backtest_arguments(data=local_data))
    first = capsys.readouterr()
    second_exit = cli.main(backtest_arguments(data=local_data))
    second = capsys.readouterr()

    assert first_exit == second_exit == cli.EXIT_SUCCESS
    assert first.out == second.out
    assert first.err == second.err == ""
    assert local_data.read_bytes() == before_contents
    assert local_data.stat().st_mtime_ns == before_timestamp
    assert tuple(tmp_path.iterdir()) == before_entries


def test_cli_calls_existing_report_formatter_once(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The orchestration layer delegates all report calculations and formatting."""

    calls: list[tuple[BacktestResult, str | None, bool]] = []

    def fake_formatter(
        result: BacktestResult,
        *,
        currency_symbol: str | None = None,
        include_closed_trades: bool = False,
        strategy_configuration: StrategyRunConfiguration | None = None,
    ) -> str:
        assert strategy_configuration is not None
        calls.append((result, currency_symbol, include_closed_trades))
        return "SENTINEL REPORT\n"

    monkeypatch.setattr(cli, "format_backtest_report", fake_formatter)

    exit_code = cli.main(
        backtest_arguments(currency_symbol="$", include_closed_trades=True)
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_SUCCESS
    assert captured.out == "SENTINEL REPORT\n"
    assert captured.err == ""
    assert len(calls) == 1
    assert calls[0][1:] == ("$", True)


def test_backtest_errors_use_exit_code_four_without_tracebacks(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expected engine failures have their documented deterministic exit code."""

    def reject_backtest(*_: object, **__: object) -> Never:
        raise BacktestError("simulated engine failure")

    monkeypatch.setattr(SimpleBacktestEngine, "run", reject_backtest)

    exit_code = cli.main(backtest_arguments())

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_BACKTEST_ERROR
    assert captured.out == ""
    assert captured.err == "error: simulated engine failure\n"
    assert "Traceback" not in captured.err


def test_library_warning_logs_do_not_pollute_success_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unaffordable simulated fill remains a quiet successful backtest."""

    exit_code = cli.main(
        backtest_arguments(starting_cash="1", quantity="100000")
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_SUCCESS
    assert "BACKTEST PERFORMANCE REPORT" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize(
    ("argument_name", "file_name"),
    [
        ("json_report", "run.json"),
        ("summary_csv", "summary.csv"),
        ("equity_csv", "equity.csv"),
        ("closed_trades_csv", "trades.csv"),
        ("decisions_csv", "decisions.csv"),
    ],
)
def test_each_structured_export_flag_independently_writes_one_requested_file(
    argument_name: str,
    file_name: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each opt-in path works alone and creates no unrequested sibling output."""

    destination = tmp_path / file_name

    arguments = backtest_arguments()
    arguments.extend((f"--{argument_name.replace('_', '-')}", str(destination)))
    exit_code = cli.main(arguments)

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_SUCCESS
    assert destination.is_file()
    assert tuple(tmp_path.iterdir()) == (destination,)
    assert "BACKTEST PERFORMANCE REPORT" in captured.out
    assert "Reports written:" in captured.out
    assert str(tmp_path) not in captured.out
    assert "Traceback" not in captured.err


def test_all_exports_support_paths_with_spaces_and_valid_structured_content(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One run may atomically publish every format under an explicit new directory."""

    output = tmp_path / "reports with spaces"
    json_path = output / "full report.json"
    summary_path = output / "summary report.csv"
    equity_path = output / "equity curve.csv"
    trades_path = output / "closed trades.csv"
    decisions_path = output / "entry decisions.csv"

    exit_code = cli.main(
        backtest_arguments(
            periods_per_year="252",
            json_report=json_path,
            summary_csv=summary_path,
            equity_csv=equity_path,
            closed_trades_csv=trades_path,
            decisions_csv=decisions_path,
        )
    )

    captured = capsys.readouterr()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    summary_rows = list(
        csv.DictReader(StringIO(summary_path.read_text(encoding="utf-8")))
    )
    equity_rows = list(
        csv.DictReader(StringIO(equity_path.read_text(encoding="utf-8")))
    )
    trade_rows = list(
        csv.DictReader(StringIO(trades_path.read_text(encoding="utf-8")))
    )
    decision_rows = list(
        csv.DictReader(StringIO(decisions_path.read_text(encoding="utf-8")))
    )

    assert exit_code == cli.EXIT_SUCCESS
    assert payload["schema_version"] == "1.3"
    assert payload["metadata"]["dataset"]["identifier"] == "data/sample_prices.csv"
    assert len(payload["metadata"]["dataset"]["sha256"]) == 64
    assert str(_ROOT) not in json_path.read_text(encoding="utf-8")
    assert summary_rows[0]["ending_equity"] == payload["performance"]["ending_equity"]
    assert equity_rows[-1]["equity"] == payload["performance"]["ending_equity"]
    assert trade_rows[0]["net_pnl"] == payload["closed_trades"][0]["net_pnl"]
    assert decision_rows[0]["outcome"] == payload["pre_trade_decisions"][0][
        "outcome"
    ]
    assert captured.out.count("- ") == 5
    assert captured.err == ""


def test_existing_report_is_rejected_then_replaced_only_with_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default protects existing content and the explicit flag permits replacement."""

    destination = tmp_path / "report.json"
    destination.write_text("protected", encoding="utf-8")

    rejected_exit = cli.main(backtest_arguments(json_report=destination))
    rejected = capsys.readouterr()
    unchanged = destination.read_text(encoding="utf-8")
    replaced_exit = cli.main(
        backtest_arguments(json_report=destination, overwrite_reports=True)
    )
    replaced = capsys.readouterr()

    assert rejected_exit == cli.EXIT_REPORT_ERROR
    assert rejected.out == ""
    assert "already exists" in rejected.err
    assert "Traceback" not in rejected.err
    assert unchanged == "protected"
    assert replaced_exit == cli.EXIT_SUCCESS
    assert json.loads(destination.read_text(encoding="utf-8"))["schema_version"] == "1.3"
    assert "Reports written:" in replaced.out
    assert replaced.err == ""


def test_existing_decision_csv_is_rejected_then_replaced_with_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The new export uses the same preflight and explicit overwrite contract."""

    destination = tmp_path / "decisions.csv"
    destination.write_text("protected", encoding="utf-8")

    rejected_exit = cli.main(backtest_arguments(decisions_csv=destination))
    rejected = capsys.readouterr()
    assert rejected_exit == cli.EXIT_REPORT_ERROR
    assert rejected.out == ""
    assert "already exists" in rejected.err
    assert destination.read_text(encoding="utf-8") == "protected"

    replaced_exit = cli.main(
        backtest_arguments(
            decisions_csv=destination,
            overwrite_reports=True,
        )
    )
    replaced = capsys.readouterr()
    rows = list(csv.DictReader(StringIO(destination.read_text(encoding="utf-8"))))
    assert replaced_exit == cli.EXIT_SUCCESS
    assert rows[0]["outcome"] == "approved"
    assert "Decisions CSV" in replaced.out
    assert replaced.err == ""


def test_invalid_report_destination_is_exit_five_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A directory where a file is expected is a concise report-writing failure."""

    exit_code = cli.main(backtest_arguments(summary_csv=tmp_path))

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_REPORT_ERROR
    assert captured.out == ""
    assert "destination is a directory" in captured.err
    assert "Traceback" not in captured.err


def test_exports_do_not_change_financial_text_or_use_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Output paths affect persistence only, never simulated results or networking."""

    baseline_exit = cli.main(backtest_arguments())
    baseline = capsys.readouterr()

    def reject_network(*_: object, **__: object) -> Never:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(socket, "getaddrinfo", reject_network)
    exported_exit = cli.main(
        backtest_arguments(
            json_report=tmp_path / "run.json",
            equity_csv=tmp_path / "equity.csv",
            decisions_csv=tmp_path / "decisions.csv",
        )
    )
    exported = capsys.readouterr()
    exported_text = exported.out.split("\nReports written:", maxsplit=1)[0]

    assert baseline_exit == exported_exit == cli.EXIT_SUCCESS
    assert exported_text == baseline.out
    assert report_value(exported.out, "Ending cash") == report_value(
        baseline.out,
        "Ending cash",
    )
    assert report_value(exported.out, "Ending equity") == report_value(
        baseline.out,
        "Ending equity",
    )
    assert report_value(exported.out, "Total return") == report_value(
        baseline.out,
        "Total return",
    )
    assert exported.err == baseline.err == ""


def test_duplicate_report_paths_are_rejected_before_backtest_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Two formats cannot overwrite each other at one ambiguous destination."""

    destination = tmp_path / "same-output"
    exit_code = cli.main(
        backtest_arguments(json_report=destination, summary_csv=destination)
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_USAGE_ERROR
    assert captured.out == ""
    assert "different path" in captured.err
    assert not destination.exists()


def test_dataset_hash_read_failure_is_a_concise_data_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A source disappearing before export identity is captured has no traceback."""

    def reject_hash(_: Path) -> Never:
        raise OSError("simulated dataset hash failure")

    monkeypatch.setattr(
        "trading_research.reporting.metadata.sha256_file",
        reject_hash,
    )

    exit_code = cli.main(backtest_arguments(json_report=tmp_path / "run.json"))

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_DATA_ERROR
    assert captured.out == ""
    assert captured.err == "error: simulated dataset hash failure\n"
    assert "Traceback" not in captured.err


def test_cli_benchmark_is_opt_in_and_preserves_sma_financial_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default stays unchanged and a second run cannot alter SMA accounting."""

    default_exit = cli.main(backtest_arguments())
    default = capsys.readouterr()
    benchmark_exit = cli.main(backtest_arguments(benchmark="buy-and-hold"))
    benchmark = capsys.readouterr()

    assert default_exit == benchmark_exit == cli.EXIT_SUCCESS
    assert "Buy-and-Hold Benchmark" not in default.out
    assert "Buy-and-Hold Benchmark" in benchmark.out
    assert "Next-bar executable" in benchmark.out
    assert report_value(default.out, "Ending cash") == report_value(
        benchmark.out,
        "Ending cash",
    )
    assert report_value(default.out, "Ending equity") == report_value(
        benchmark.out,
        "Ending equity",
    )
    assert report_value(default.out, "Total return") == report_value(
        benchmark.out,
        "Total return",
    )
    assert default.err == benchmark.err == ""


def test_cli_benchmark_all_exports_and_risk_metrics_succeed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One benchmark run publishes every existing and new requested format."""

    paths = {
        "json_report": tmp_path / "reports with spaces" / "run.json",
        "summary_csv": tmp_path / "reports with spaces" / "summary.csv",
        "equity_csv": tmp_path / "reports with spaces" / "strategy equity.csv",
        "closed_trades_csv": tmp_path / "reports with spaces" / "trades.csv",
        "decisions_csv": tmp_path / "reports with spaces" / "decisions.csv",
        "benchmark_equity_csv": tmp_path
        / "reports with spaces"
        / "benchmark equity.csv",
    }

    exit_code = cli.main(
        backtest_arguments(
            benchmark="buy-and-hold",
            end_of_test="liquidate",
            periods_per_year="252",
            json_report=paths["json_report"],
            summary_csv=paths["summary_csv"],
            equity_csv=paths["equity_csv"],
            closed_trades_csv=paths["closed_trades_csv"],
            decisions_csv=paths["decisions_csv"],
            benchmark_equity_csv=paths["benchmark_equity_csv"],
        )
    )

    captured = capsys.readouterr()
    payload = json.loads(paths["json_report"].read_text(encoding="utf-8"))
    comparison_rows = list(
        csv.DictReader(
            StringIO(paths["benchmark_equity_csv"].read_text(encoding="utf-8"))
        )
    )
    assert exit_code == cli.EXIT_SUCCESS
    assert payload["schema_version"] == "1.3"
    assert payload["benchmark"]["risk_adjusted_metrics"] is not None
    assert payload["benchmark"]["result"]["reconciliation"]["ending_equity"] == (
        payload["benchmark"]["result"]["final_equity"]
    )
    assert payload["benchmark_comparison"]["benchmark_name"] == "buy-and-hold"
    assert comparison_rows[-1]["benchmark_cash"] == comparison_rows[-1][
        "benchmark_equity"
    ]
    assert all(path.is_file() for path in paths.values())
    assert captured.out.count("- ") == 6
    assert captured.err == ""


def test_cli_invalid_benchmark_and_orphan_equity_export_are_usage_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Argparse restricts names and comparison exports require an enabled benchmark."""

    invalid_exit = cli.main(backtest_arguments(benchmark="index"))
    invalid = capsys.readouterr()
    orphan_exit = cli.main(
        backtest_arguments(benchmark_equity_csv=tmp_path / "comparison.csv")
    )
    orphan = capsys.readouterr()

    assert invalid_exit == orphan_exit == cli.EXIT_USAGE_ERROR
    assert "invalid choice" in invalid.err
    assert "requires --benchmark buy-and-hold" in orphan.err
    assert "Traceback" not in invalid.err + orphan.err
    assert not (tmp_path / "comparison.csv").exists()


def test_cli_benchmark_with_fewer_than_two_bars_fails_concisely(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unexecutable benchmark dataset produces the existing concise data error."""

    data = tmp_path / "one-bar.csv"
    data.write_text(
        "timestamp,symbol,open,high,low,close,volume\n"
        "2026-01-01T00:00:00+00:00,DEMO,10,10,10,10,100\n",
        encoding="utf-8",
    )

    exit_code = cli.main(
        backtest_arguments(
            data=data,
            fast_window="1",
            slow_window="2",
            benchmark="buy-and-hold",
        )
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_DATA_ERROR
    assert captured.out == ""
    assert "loaded 1 bars" in captured.err
    assert "Traceback" not in captured.err


def test_cli_benchmark_uses_one_csv_load(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The validated source is loaded once and the resulting tuple is shared."""

    original = CsvMarketDataProvider.get_historical_bars
    calls = 0

    def counted_load(
        self: CsvMarketDataProvider,
        symbol: str,
        start: datetime | None,
        end: datetime | None,
    ) -> Sequence[MarketBar]:
        nonlocal calls
        calls += 1
        return original(self, symbol, start, end)

    monkeypatch.setattr(
        CsvMarketDataProvider,
        "get_historical_bars",
        counted_load,
    )

    exit_code = cli.main(backtest_arguments(benchmark="buy-and-hold"))

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_SUCCESS
    assert calls == 1
    assert captured.err == ""


def test_cli_reduced_and_rejected_benchmark_entries_are_reported(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Benchmark risk controls may reduce or reject rather than force investment."""

    reduced_exit = cli.main(
        backtest_arguments(
            benchmark="buy-and-hold",
            quantity="100",
            maximum_position_value="100",
        )
    )
    reduced = capsys.readouterr()
    rejected_exit = cli.main(
        backtest_arguments(
            benchmark="buy-and-hold",
            minimum_cash_reserve="10000",
        )
    )
    rejected = capsys.readouterr()

    assert reduced_exit == rejected_exit == cli.EXIT_SUCCESS
    assert report_value(reduced.out, "Benchmark time in market") != "0.00%"
    assert report_value(rejected.out, "Benchmark time in market") == "0.00%"
    assert reduced.err == rejected.err == ""
