"""Pure structured serializers and safe atomic local report writers."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import fields, is_dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from io import StringIO
from pathlib import Path

from trading_research.benchmarks import BenchmarkAnalysis, BenchmarkComparison
from trading_research.errors import ReportWriteError
from trading_research.reporting.structured import (
    StructuredBacktestReport,
    prepare_benchmark_equity_rows,
    prepare_equity_curve_rows,
)

SUMMARY_CSV_COLUMNS = (
    "schema_version",
    "run_id",
    "generated_at",
    "application_version",
    "git_commit",
    "data_identifier",
    "data_sha256",
    "symbol",
    "actual_start",
    "actual_end",
    "bar_count",
    "strategy_name",
    "fast_window",
    "slow_window",
    "entry_window",
    "exit_window",
    "starting_cash",
    "commission_bps",
    "slippage_bps",
    "position_sizing_mode",
    "quantity",
    "cash_allocation_ratio",
    "risk_policy_name",
    "maximum_position_value",
    "minimum_cash_reserve",
    "end_of_test_policy",
    "periods_per_year",
    "risk_free_rate_per_period",
    "target_return_per_period",
    "ending_cash",
    "ending_equity",
    "net_profit",
    "total_return",
    "maximum_absolute_drawdown",
    "maximum_percentage_drawdown",
    "completed_trade_count",
    "winning_trade_count",
    "losing_trade_count",
    "breakeven_trade_count",
    "win_rate",
    "profit_factor",
    "periodic_volatility",
    "annualised_volatility",
    "periodic_sharpe_ratio",
    "annualised_sharpe_ratio",
    "periodic_sortino_ratio",
    "annualised_sortino_ratio",
    "observation_count",
    "invested_observation_count",
    "cash_only_observation_count",
    "time_in_market_ratio",
    "average_position_quantity",
    "maximum_position_quantity",
    "average_position_market_value",
    "maximum_position_market_value",
    "average_cash",
    "minimum_cash",
    "average_cash_ratio",
    "minimum_cash_ratio",
    "average_capital_utilisation_ratio",
    "maximum_capital_utilisation_ratio",
    "actionable_signal_count",
    "full_approval_count",
    "reduced_decision_count",
    "rejected_decision_count",
    "execution_approval_rate",
    "full_approval_rate",
    "reduction_rate",
    "rejection_rate",
    "proposed_quantity_total",
    "approved_quantity_total",
    "quantity_reduction_total",
    "benchmark_name",
    "benchmark_ending_cash",
    "benchmark_ending_equity",
    "benchmark_net_profit",
    "benchmark_total_return",
    "benchmark_maximum_absolute_drawdown",
    "benchmark_maximum_percentage_drawdown",
    "benchmark_completed_trade_count",
    "benchmark_time_in_market_ratio",
    "benchmark_periodic_volatility",
    "benchmark_annualised_volatility",
    "benchmark_periodic_sharpe_ratio",
    "benchmark_annualised_sharpe_ratio",
    "benchmark_periodic_sortino_ratio",
    "benchmark_annualised_sortino_ratio",
    "ending_equity_difference",
    "excess_return",
    "maximum_drawdown_improvement",
    "time_in_market_difference",
    "reconciliation_status",
)

EQUITY_CSV_COLUMNS = (
    "timestamp",
    "cash",
    "position_quantity",
    "position_market_value",
    "equity",
    "running_peak_equity",
    "absolute_drawdown",
    "percentage_drawdown",
    "period_return",
    "capital_utilisation_ratio",
    "cash_ratio",
    "invested",
)

CLOSED_TRADES_CSV_COLUMNS = (
    "symbol",
    "entry_timestamp",
    "exit_timestamp",
    "quantity",
    "entry_reference_price",
    "entry_fill_price",
    "exit_reference_price",
    "exit_fill_price",
    "gross_entry_value",
    "gross_exit_value",
    "entry_commission",
    "exit_commission",
    "total_commission",
    "entry_slippage_cost",
    "exit_slippage_cost",
    "total_slippage_cost",
    "gross_pnl",
    "net_pnl",
    "return_ratio",
    "holding_period_seconds",
    "exit_reason",
)

DECISIONS_CSV_COLUMNS = (
    "timestamp",
    "symbol",
    "side",
    "reference_price",
    "proposed_quantity",
    "approved_quantity",
    "quantity_reduction",
    "estimated_fill_price",
    "estimated_commission",
    "estimated_total_cost",
    "available_cash_before",
    "position_quantity_before",
    "outcome",
    "reason",
)

BENCHMARK_EQUITY_CSV_COLUMNS = (
    "timestamp",
    "strategy_cash",
    "strategy_position_market_value",
    "strategy_equity",
    "strategy_drawdown",
    "benchmark_cash",
    "benchmark_position_market_value",
    "benchmark_equity",
    "benchmark_drawdown",
    "equity_difference",
)


def structured_report_to_json(
    report: StructuredBacktestReport,
    *,
    indent: int = 2,
) -> str:
    """Return deterministic JSON with Decimals and durations encoded as strings."""

    if not isinstance(report, StructuredBacktestReport):
        raise TypeError("report must be a StructuredBacktestReport")
    if isinstance(indent, bool) or not isinstance(indent, int):
        raise TypeError("indent must be an integer")
    if indent < 0:
        raise ValueError("indent must be non-negative")
    return json.dumps(
        _json_value(report),
        ensure_ascii=False,
        indent=indent,
        allow_nan=False,
    )


def summary_report_to_csv(report: StructuredBacktestReport) -> str:
    """Return a stable one-header, one-row raw summary CSV."""

    _require_report(report)
    metadata = report.metadata
    simulation = metadata.simulation
    sizing = metadata.position_sizing
    risk = metadata.risk_policy
    performance = report.performance
    trades = report.trade_statistics
    risk_settings = metadata.risk_metric_settings
    risk_metrics = report.risk_adjusted_metrics
    exposure = report.exposure_statistics
    benchmark = report.benchmark
    comparison = report.benchmark_comparison
    benchmark_risk = None if benchmark is None else benchmark.risk_adjusted_metrics
    values: dict[str, str | int] = {
        "schema_version": report.schema_version,
        "run_id": metadata.run_id,
        "generated_at": metadata.generated_at.isoformat(),
        "application_version": metadata.application_version,
        "git_commit": metadata.git_commit or "",
        "data_identifier": metadata.dataset.identifier,
        "data_sha256": metadata.dataset.sha256 or "",
        "symbol": metadata.dataset.symbol,
        "actual_start": metadata.dataset.actual_start.isoformat(),
        "actual_end": metadata.dataset.actual_end.isoformat(),
        "bar_count": metadata.dataset.bar_count,
        "strategy_name": metadata.strategy.name.value,
        "fast_window": _optional_integer(metadata.strategy.fast_window),
        "slow_window": _optional_integer(metadata.strategy.slow_window),
        "entry_window": _optional_integer(metadata.strategy.entry_window),
        "exit_window": _optional_integer(metadata.strategy.exit_window),
        "starting_cash": str(simulation.starting_cash),
        "commission_bps": str(simulation.commission_bps),
        "slippage_bps": str(simulation.slippage_bps),
        "position_sizing_mode": sizing.mode.value,
        "quantity": "" if sizing.quantity is None else sizing.quantity,
        "cash_allocation_ratio": _optional_decimal(sizing.cash_allocation_ratio),
        "risk_policy_name": risk.name,
        "maximum_position_value": _optional_decimal(risk.maximum_position_value),
        "minimum_cash_reserve": _optional_decimal(risk.minimum_cash_reserve),
        "end_of_test_policy": simulation.end_of_test_policy.value,
        "periods_per_year": _optional_decimal(
            None if risk_settings is None else risk_settings.periods_per_year
        ),
        "risk_free_rate_per_period": _optional_decimal(
            None if risk_settings is None else risk_settings.risk_free_rate_per_period
        ),
        "target_return_per_period": _optional_decimal(
            None if risk_settings is None else risk_settings.target_return_per_period
        ),
        "ending_cash": str(performance.ending_cash),
        "ending_equity": str(performance.ending_equity),
        "net_profit": str(performance.net_profit),
        "total_return": str(performance.total_return),
        "maximum_absolute_drawdown": str(performance.maximum_absolute_drawdown),
        "maximum_percentage_drawdown": str(performance.maximum_percentage_drawdown),
        "completed_trade_count": trades.completed_trade_count,
        "winning_trade_count": trades.winning_trade_count,
        "losing_trade_count": trades.losing_trade_count,
        "breakeven_trade_count": trades.breakeven_trade_count,
        "win_rate": _optional_decimal(trades.win_rate),
        "profit_factor": _optional_decimal(trades.profit_factor),
        "periodic_volatility": _metric_decimal(
            risk_metrics,
            "periodic_volatility",
        ),
        "annualised_volatility": _metric_decimal(
            risk_metrics,
            "annualised_volatility",
        ),
        "periodic_sharpe_ratio": _metric_decimal(
            risk_metrics,
            "periodic_sharpe_ratio",
        ),
        "annualised_sharpe_ratio": _metric_decimal(
            risk_metrics,
            "annualised_sharpe_ratio",
        ),
        "periodic_sortino_ratio": _metric_decimal(
            risk_metrics,
            "periodic_sortino_ratio",
        ),
        "annualised_sortino_ratio": _metric_decimal(
            risk_metrics,
            "annualised_sortino_ratio",
        ),
        "observation_count": exposure.observation_count,
        "invested_observation_count": exposure.invested_observation_count,
        "cash_only_observation_count": exposure.cash_only_observation_count,
        "time_in_market_ratio": _optional_decimal(exposure.time_in_market_ratio),
        "average_position_quantity": _optional_decimal(
            exposure.average_position_quantity
        ),
        "maximum_position_quantity": str(exposure.maximum_position_quantity),
        "average_position_market_value": _optional_decimal(
            exposure.average_position_market_value
        ),
        "maximum_position_market_value": str(
            exposure.maximum_position_market_value
        ),
        "average_cash": _optional_decimal(exposure.average_cash),
        "minimum_cash": _optional_decimal(exposure.minimum_cash),
        "average_cash_ratio": _optional_decimal(exposure.average_cash_ratio),
        "minimum_cash_ratio": _optional_decimal(exposure.minimum_cash_ratio),
        "average_capital_utilisation_ratio": _optional_decimal(
            exposure.average_capital_utilisation_ratio
        ),
        "maximum_capital_utilisation_ratio": _optional_decimal(
            exposure.maximum_capital_utilisation_ratio
        ),
        "actionable_signal_count": exposure.actionable_signal_count,
        "full_approval_count": exposure.full_approval_count,
        "reduced_decision_count": exposure.reduced_decision_count,
        "rejected_decision_count": exposure.rejected_decision_count,
        "execution_approval_rate": _optional_decimal(
            exposure.execution_approval_rate
        ),
        "full_approval_rate": _optional_decimal(exposure.full_approval_rate),
        "reduction_rate": _optional_decimal(exposure.reduction_rate),
        "rejection_rate": _optional_decimal(exposure.rejection_rate),
        "proposed_quantity_total": str(exposure.proposed_quantity_total),
        "approved_quantity_total": str(exposure.approved_quantity_total),
        "quantity_reduction_total": str(exposure.quantity_reduction_total),
        "benchmark_name": "" if benchmark is None else benchmark.name,
        "benchmark_ending_cash": _benchmark_decimal(benchmark, "ending_cash"),
        "benchmark_ending_equity": _benchmark_decimal(benchmark, "ending_equity"),
        "benchmark_net_profit": _benchmark_decimal(benchmark, "net_profit"),
        "benchmark_total_return": _benchmark_decimal(benchmark, "total_return"),
        "benchmark_maximum_absolute_drawdown": _benchmark_decimal(
            benchmark,
            "maximum_absolute_drawdown",
        ),
        "benchmark_maximum_percentage_drawdown": _benchmark_decimal(
            benchmark,
            "maximum_percentage_drawdown",
        ),
        "benchmark_completed_trade_count": (
            "" if benchmark is None else benchmark.trade_statistics.completed_trade_count
        ),
        "benchmark_time_in_market_ratio": (
            ""
            if benchmark is None
            else _optional_decimal(benchmark.exposure_statistics.time_in_market_ratio)
        ),
        "benchmark_periodic_volatility": _metric_decimal(
            benchmark_risk,
            "periodic_volatility",
        ),
        "benchmark_annualised_volatility": _metric_decimal(
            benchmark_risk,
            "annualised_volatility",
        ),
        "benchmark_periodic_sharpe_ratio": _metric_decimal(
            benchmark_risk,
            "periodic_sharpe_ratio",
        ),
        "benchmark_annualised_sharpe_ratio": _metric_decimal(
            benchmark_risk,
            "annualised_sharpe_ratio",
        ),
        "benchmark_periodic_sortino_ratio": _metric_decimal(
            benchmark_risk,
            "periodic_sortino_ratio",
        ),
        "benchmark_annualised_sortino_ratio": _metric_decimal(
            benchmark_risk,
            "annualised_sortino_ratio",
        ),
        "ending_equity_difference": _comparison_decimal(
            comparison,
            "ending_equity_difference",
        ),
        "excess_return": _comparison_decimal(comparison, "excess_return"),
        "maximum_drawdown_improvement": _comparison_decimal(
            comparison,
            "maximum_drawdown_improvement",
        ),
        "time_in_market_difference": _comparison_decimal(
            comparison,
            "time_in_market_difference",
        ),
        "reconciliation_status": (
            "PASS" if report.backtest_result.reconciliation.is_reconciled else "FAIL"
        ),
    }
    return _csv_text(
        SUMMARY_CSV_COLUMNS,
        (tuple(values[column] for column in SUMMARY_CSV_COLUMNS),),
    )


def equity_curve_to_csv(report: StructuredBacktestReport) -> str:
    """Return one chronological row per authoritative equity observation."""

    rows = prepare_equity_curve_rows(report)
    return _csv_text(
        EQUITY_CSV_COLUMNS,
        tuple(
            (
                row.timestamp.isoformat(),
                str(row.cash),
                row.position_quantity,
                str(row.position_market_value),
                str(row.equity),
                str(row.running_peak_equity),
                str(row.absolute_drawdown),
                str(row.percentage_drawdown),
                str(row.period_return),
                str(row.capital_utilisation_ratio),
                str(row.cash_ratio),
                str(row.invested).lower(),
            )
            for row in rows
        ),
    )


def closed_trades_to_csv(report: StructuredBacktestReport) -> str:
    """Return chronological reconstructed closed trades, or a header-only CSV."""

    _require_report(report)
    return _csv_text(
        CLOSED_TRADES_CSV_COLUMNS,
        tuple(
            (
                trade.symbol,
                trade.entry_timestamp.isoformat(),
                trade.exit_timestamp.isoformat(),
                trade.quantity,
                str(trade.entry_reference_price),
                str(trade.entry_fill_price),
                str(trade.exit_reference_price),
                str(trade.exit_fill_price),
                str(trade.gross_entry_value),
                str(trade.gross_exit_value),
                str(trade.entry_commission),
                str(trade.exit_commission),
                str(trade.total_commission),
                str(trade.entry_slippage_cost),
                str(trade.exit_slippage_cost),
                str(trade.total_slippage_cost),
                str(trade.gross_pnl),
                str(trade.net_pnl),
                str(trade.return_ratio),
                _timedelta_seconds(trade.holding_period),
                trade.exit_execution_reason.value,
            )
            for trade in report.closed_trades
        ),
    )


def decisions_to_csv(report: StructuredBacktestReport) -> str:
    """Return ordered audited entry decisions, or a header-only CSV."""

    _require_report(report)
    return _csv_text(
        DECISIONS_CSV_COLUMNS,
        tuple(
            (
                decision.timestamp.isoformat(),
                decision.symbol,
                decision.side.value,
                str(decision.reference_price),
                decision.proposed_quantity,
                decision.approved_quantity,
                decision.quantity_reduction,
                str(decision.estimated_fill_price),
                str(decision.estimated_commission),
                str(decision.estimated_total_cost),
                str(decision.available_cash_before),
                decision.position_quantity_before,
                decision.outcome.value,
                "" if decision.reason is None else decision.reason.value,
            )
            for decision in report.pre_trade_decisions
        ),
    )


def benchmark_equity_to_csv(report: StructuredBacktestReport) -> str:
    """Return strictly aligned strategy and benchmark equity observations."""

    rows = prepare_benchmark_equity_rows(report)
    return _csv_text(
        BENCHMARK_EQUITY_CSV_COLUMNS,
        tuple(
            (
                row.timestamp.isoformat(),
                str(row.strategy_cash),
                str(row.strategy_position_market_value),
                str(row.strategy_equity),
                str(row.strategy_drawdown),
                str(row.benchmark_cash),
                str(row.benchmark_position_market_value),
                str(row.benchmark_equity),
                str(row.benchmark_drawdown),
                str(row.equity_difference),
            )
            for row in rows
        ),
    )


def write_json_report(
    report: StructuredBacktestReport,
    path: Path,
    *,
    overwrite: bool = False,
    create_parents: bool = False,
) -> None:
    """Atomically write UTF-8 JSON, protecting existing files by default."""

    _atomic_write_text(
        Path(path),
        structured_report_to_json(report) + "\n",
        overwrite=overwrite,
        create_parents=create_parents,
    )


def write_summary_csv(
    report: StructuredBacktestReport,
    path: Path,
    *,
    overwrite: bool = False,
    create_parents: bool = False,
) -> None:
    """Atomically write the flat summary CSV."""

    _atomic_write_text(
        Path(path),
        summary_report_to_csv(report),
        overwrite=overwrite,
        create_parents=create_parents,
    )


def write_equity_curve_csv(
    report: StructuredBacktestReport,
    path: Path,
    *,
    overwrite: bool = False,
    create_parents: bool = False,
) -> None:
    """Atomically write the chronological equity-curve CSV."""

    _atomic_write_text(
        Path(path),
        equity_curve_to_csv(report),
        overwrite=overwrite,
        create_parents=create_parents,
    )


def write_closed_trades_csv(
    report: StructuredBacktestReport,
    path: Path,
    *,
    overwrite: bool = False,
    create_parents: bool = False,
) -> None:
    """Atomically write closed trades, including a header when there are none."""

    _atomic_write_text(
        Path(path),
        closed_trades_to_csv(report),
        overwrite=overwrite,
        create_parents=create_parents,
    )


def write_decisions_csv(
    report: StructuredBacktestReport,
    path: Path,
    *,
    overwrite: bool = False,
    create_parents: bool = False,
) -> None:
    """Atomically write audited pre-trade decisions, including an empty header."""

    _atomic_write_text(
        Path(path),
        decisions_to_csv(report),
        overwrite=overwrite,
        create_parents=create_parents,
    )


def write_benchmark_equity_csv(
    report: StructuredBacktestReport,
    path: Path,
    *,
    overwrite: bool = False,
    create_parents: bool = False,
) -> None:
    """Atomically write the aligned benchmark equity comparison."""

    _atomic_write_text(
        Path(path),
        benchmark_equity_to_csv(report),
        overwrite=overwrite,
        create_parents=create_parents,
    )


def write_text_export(
    content: str,
    path: Path,
    *,
    overwrite: bool = False,
    create_parents: bool = False,
) -> None:
    """Atomically write already-serialised UTF-8 report content."""

    if not isinstance(content, str):
        raise TypeError("content must be a string")
    _atomic_write_text(
        Path(path),
        content,
        overwrite=overwrite,
        create_parents=create_parents,
    )


def _json_value(value: object) -> object:
    """Convert supported domain values without introducing binary floats."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("JSON timestamps must be timezone-aware")
        return value.isoformat()
    if isinstance(value, timedelta):
        return _timedelta_seconds(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON mappings must use string keys")
        return {key: _json_value(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, float):
        raise TypeError("binary float values are not supported in structured reports")
    raise TypeError(f"unsupported structured-report value: {type(value).__name__}")


def _csv_text(
    columns: tuple[str, ...],
    rows: tuple[tuple[object, ...], ...],
) -> str:
    """Build deterministic RFC-style CSV text using LF line endings."""

    buffer = StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return buffer.getvalue()


def _timedelta_seconds(value: timedelta) -> str:
    """Encode a duration as exact decimal seconds without float conversion."""

    total_microseconds = (
        (value.days * 86_400 + value.seconds) * 1_000_000 + value.microseconds
    )
    seconds = Decimal(total_microseconds) / Decimal(1_000_000)
    return format(seconds, "f")


def _optional_decimal(value: Decimal | None) -> str:
    """Return a raw Decimal string or the CSV missing-value convention."""

    return "" if value is None else str(value)


def _optional_integer(value: int | None) -> str | int:
    """Return an integer or the CSV missing-value convention."""

    return "" if value is None else value


def _metric_decimal(
    metrics: object | None,
    attribute: str,
) -> str:
    """Return one optional risk metric without formatting its raw value."""

    if metrics is None:
        return ""
    value = getattr(metrics, attribute)
    if value is not None and not isinstance(value, Decimal):
        raise TypeError(f"{attribute} must be a Decimal or None")
    return _optional_decimal(value)


def _benchmark_decimal(
    benchmark: BenchmarkAnalysis | None,
    attribute: str,
) -> str:
    """Return one raw benchmark performance value or an empty field."""

    if benchmark is None:
        return ""
    value = getattr(benchmark.performance, attribute)
    if not isinstance(value, Decimal):
        raise TypeError(f"benchmark {attribute} must be a Decimal")
    return str(value)


def _comparison_decimal(
    comparison: BenchmarkComparison | None,
    attribute: str,
) -> str:
    """Return one optional comparison value using the CSV missing convention."""

    if comparison is None:
        return ""
    value = getattr(comparison, attribute)
    if value is not None and not isinstance(value, Decimal):
        raise TypeError(f"comparison {attribute} must be a Decimal or None")
    return _optional_decimal(value)


def _require_report(report: StructuredBacktestReport) -> None:
    """Validate public serializer input."""

    if not isinstance(report, StructuredBacktestReport):
        raise TypeError("report must be a StructuredBacktestReport")


def _atomic_write_text(
    path: Path,
    content: str,
    *,
    overwrite: bool,
    create_parents: bool,
) -> None:
    """Write a complete UTF-8 file atomically in its caller-selected directory."""

    if not isinstance(overwrite, bool) or not isinstance(create_parents, bool):
        raise TypeError("overwrite and create_parents must be bool values")
    target = Path(path)
    if target.exists():
        if target.is_dir():
            raise ReportWriteError(f"report destination is a directory: {target}")
        if not overwrite:
            raise ReportWriteError(f"report file already exists: {target}")
    parent = target.parent
    try:
        if not parent.exists():
            if not create_parents:
                raise ReportWriteError(f"report parent directory does not exist: {parent}")
            parent.mkdir(parents=True, exist_ok=True)
        if not parent.is_dir():
            raise ReportWriteError(f"report parent path is not a directory: {parent}")
    except ReportWriteError:
        raise
    except OSError as exc:
        raise ReportWriteError(f"could not prepare report directory {parent}: {exc}") from exc

    descriptor: int | None = None
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            text=True,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            descriptor = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, target)
        else:
            try:
                os.link(temporary_path, target)
            except FileExistsError as exc:
                raise ReportWriteError(f"report file already exists: {target}") from exc
            temporary_path.unlink()
        temporary_path = None
    except ReportWriteError:
        raise
    except OSError as exc:
        raise ReportWriteError(f"could not write report file {target}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
