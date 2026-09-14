"""Distinct schema-1.0 exports for split-aware evaluations."""

import csv
import json
from dataclasses import fields, is_dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from io import StringIO
from pathlib import Path

from pydantic import BaseModel

from trading_research.errors import ReportWriteError
from trading_research.evaluation.models import (
    SPLIT_EVALUATION_SCHEMA_NAME,
    SPLIT_EVALUATION_SCHEMA_VERSION,
    SplitEvaluationResult,
)
from trading_research.evaluation.text import format_split_evaluation_report
from trading_research.reporting import (
    StructuredBacktestReport,
    write_json_report,
    write_text_export,
)
from trading_research.strategies import (
    DonchianBreakoutParameters,
    SmaCrossoverParameters,
)

SPLIT_COMPARISON_CSV_COLUMNS = (
    "evaluation_id",
    "configuration_sha256",
    "strategy_name",
    "strategy_parameters",
    "development_start",
    "development_end",
    "holdout_start",
    "holdout_end",
    "boundary_gap_seconds",
    "warmup_policy",
    "warmup_bar_count",
    "development_total_return",
    "holdout_total_return",
    "total_return_change",
    "development_excess_return",
    "holdout_excess_return",
    "excess_return_change",
    "development_maximum_drawdown",
    "holdout_maximum_drawdown",
    "maximum_drawdown_change",
    "development_sharpe",
    "holdout_sharpe",
    "sharpe_change",
    "development_trade_count",
    "holdout_trade_count",
    "development_time_in_market",
    "holdout_time_in_market",
    "dataset_id",
    "data_sha256",
    "price_adjustment",
)


def split_evaluation_to_json(
    result: SplitEvaluationResult,
    *,
    indent: int = 2,
) -> str:
    """Serialize the combined evaluation with exact Decimal strings."""

    if not isinstance(result, SplitEvaluationResult):
        raise TypeError("result must be SplitEvaluationResult")
    payload = {
        "schema": SPLIT_EVALUATION_SCHEMA_NAME,
        "schema_version": SPLIT_EVALUATION_SCHEMA_VERSION,
        "evaluation_id": result.evaluation_id,
        "generated_at": result.generated_at,
        "git_commit": result.git_commit,
        "data_sha256": result.data_sha256,
        "configuration_sha256": result.configuration_sha256,
        "development_configuration_sha256": (
            result.development_configuration_sha256
        ),
        "holdout_configuration_sha256": result.holdout_configuration_sha256,
        "provenance": result.provenance,
        "quality_summary": result.quality_summary,
        "split": result.split,
        "warmup_bar_count": result.warmup_bar_count,
        "warmup_first_timestamp": result.warmup_first_timestamp,
        "warmup_last_timestamp": result.warmup_last_timestamp,
        "development_report": "development.json",
        "holdout_report": "holdout.json",
        "development": _report_summary(result.development),
        "holdout": _report_summary(result.holdout),
        "stability_comparison": result.stability_comparison,
        "quality_warnings": _quality_warnings(result),
    }
    return json.dumps(
        _json_value(payload),
        ensure_ascii=False,
        indent=indent,
        allow_nan=False,
    )


def split_comparison_to_csv(result: SplitEvaluationResult) -> str:
    """Return one raw-value comparison row with no presentation rounding."""

    if not isinstance(result, SplitEvaluationResult):
        raise TypeError("result must be SplitEvaluationResult")
    comparison = result.stability_comparison
    split = result.split
    row = {
        "evaluation_id": result.evaluation_id,
        "configuration_sha256": result.configuration_sha256,
        "strategy_name": result.development.metadata.strategy.name.value,
        "strategy_parameters": _strategy_parameters_json(result),
        "development_start": split.development_start.isoformat(),
        "development_end": split.development_end.isoformat(),
        "holdout_start": split.holdout_start.isoformat(),
        "holdout_end": split.holdout_end.isoformat(),
        "boundary_gap_seconds": _timedelta_decimal_seconds(split.boundary_gap),
        "warmup_policy": split.warmup_policy.value,
        "warmup_bar_count": result.warmup_bar_count,
        "development_total_return": comparison.development_total_return,
        "holdout_total_return": comparison.holdout_total_return,
        "total_return_change": comparison.total_return_change,
        "development_excess_return": comparison.development_excess_return,
        "holdout_excess_return": comparison.holdout_excess_return,
        "excess_return_change": comparison.excess_return_change,
        "development_maximum_drawdown": comparison.development_maximum_drawdown,
        "holdout_maximum_drawdown": comparison.holdout_maximum_drawdown,
        "maximum_drawdown_change": comparison.maximum_drawdown_change,
        "development_sharpe": comparison.development_sharpe,
        "holdout_sharpe": comparison.holdout_sharpe,
        "sharpe_change": comparison.sharpe_change,
        "development_trade_count": comparison.development_trade_count,
        "holdout_trade_count": comparison.holdout_trade_count,
        "development_time_in_market": comparison.development_time_in_market,
        "holdout_time_in_market": comparison.holdout_time_in_market,
        "dataset_id": result.provenance.dataset_id,
        "data_sha256": result.data_sha256,
        "price_adjustment": result.provenance.price_adjustment.value,
    }
    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=SPLIT_COMPARISON_CSV_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerow(
        {
            key: "" if value is None else str(value)
            for key, value in row.items()
        }
    )
    return buffer.getvalue()


def write_split_evaluation_exports(
    result: SplitEvaluationResult,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write all known split files after complete existence preflight."""

    output = Path(output_directory)
    targets = tuple(
        output / name
        for name in (
            "evaluation.json",
            "development.json",
            "holdout.json",
            "comparison.csv",
            "report.txt",
        )
    )
    if not overwrite:
        existing = tuple(path for path in targets if path.exists())
        if existing:
            raise ReportWriteError(f"report file already exists: {existing[0]}")
    write_text_export(
        split_evaluation_to_json(result) + "\n",
        targets[0],
        overwrite=overwrite,
        create_parents=True,
    )
    write_json_report(
        result.development,
        targets[1],
        overwrite=overwrite,
        create_parents=True,
    )
    write_json_report(
        result.holdout,
        targets[2],
        overwrite=overwrite,
        create_parents=True,
    )
    write_text_export(
        split_comparison_to_csv(result),
        targets[3],
        overwrite=overwrite,
        create_parents=True,
    )
    write_text_export(
        format_split_evaluation_report(result),
        targets[4],
        overwrite=overwrite,
        create_parents=True,
    )


def _report_summary(report: StructuredBacktestReport) -> dict[str, object]:
    """Return core raw metrics while individual schema 1.3 remains authoritative."""

    if not isinstance(report, StructuredBacktestReport):
        raise TypeError("report must be StructuredBacktestReport")
    return {
        "run_id": report.metadata.run_id,
        "actual_start": report.metadata.dataset.actual_start,
        "actual_end": report.metadata.dataset.actual_end,
        "bar_count": report.metadata.dataset.bar_count,
        "ending_cash": report.performance.ending_cash,
        "ending_equity": report.performance.ending_equity,
        "total_return": report.performance.total_return,
        "maximum_percentage_drawdown": (
            report.performance.maximum_percentage_drawdown
        ),
        "completed_trade_count": report.trade_statistics.completed_trade_count,
        "periodic_sharpe_ratio": (
            None
            if report.risk_adjusted_metrics is None
            else report.risk_adjusted_metrics.periodic_sharpe_ratio
        ),
        "time_in_market_ratio": report.exposure_statistics.time_in_market_ratio,
        "benchmark_comparison": report.benchmark_comparison,
        "reconciliation_status": (
            "PASS" if report.backtest_result.reconciliation.is_reconciled else "FAIL"
        ),
    }


def _strategy_parameters_json(result: SplitEvaluationResult) -> str:
    """Return stable compact strategy parameters for one CSV cell."""

    parameters = result.development.metadata.strategy.parameters
    if isinstance(parameters, SmaCrossoverParameters):
        payload = {
            "fast_window": parameters.fast_window,
            "slow_window": parameters.slow_window,
        }
    elif isinstance(parameters, DonchianBreakoutParameters):
        payload = {
            "entry_window": parameters.entry_window,
            "exit_window": parameters.exit_window,
        }
    else:
        raise TypeError("unsupported strategy parameter model")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _quality_warnings(result: SplitEvaluationResult) -> list[str]:
    """Return diagnostics without asserting that observations are invalid."""

    warnings: list[str] = []
    quality = result.quality_summary
    if quality.suspicious_return_count:
        warnings.append(
            "Suspicious returns may reflect a split, corporate action, bad data, "
            "or genuine extreme movement; observations were not changed."
        )
    if quality.maximum_gap is not None:
        warnings.append(
            "Timestamp gaps are reported diagnostically and do not imply missing bars."
        )
    return warnings


def _json_value(value: object) -> object:
    """Recursively serialize supported exact models without introducing floats."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(_timedelta_decimal_seconds(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="python"))
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(f"unsupported split export value: {type(value).__name__}")


def _timedelta_decimal_seconds(value: timedelta) -> Decimal:
    """Return exact elapsed seconds without using ``total_seconds`` float."""

    return (
        Decimal(value.days) * Decimal("86400")
        + Decimal(value.seconds)
        + Decimal(value.microseconds) / Decimal("1000000")
    )
