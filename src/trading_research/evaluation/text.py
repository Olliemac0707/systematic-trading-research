"""Human-readable split evaluation reporting without automatic verdicts."""

from decimal import Decimal

from trading_research.evaluation.models import SplitEvaluationResult
from trading_research.reporting import StructuredBacktestReport
from trading_research.strategies import (
    DonchianBreakoutParameters,
    SmaCrossoverParameters,
)


def format_split_evaluation_report(result: SplitEvaluationResult) -> str:
    """Format provenance, both periods, and raw stability differences."""

    if not isinstance(result, SplitEvaluationResult):
        raise TypeError("result must be SplitEvaluationResult")
    development = result.development
    holdout = result.holdout
    stability = result.stability_comparison
    provenance = result.provenance
    lines = [
        "OUT-OF-SAMPLE EVALUATION",
        "========================",
        "",
        "Dataset",
        "-------",
        f"Dataset ID: {provenance.dataset_id}",
        f"Source: {provenance.source_name}",
        f"Price adjustment: {provenance.price_adjustment.value}",
        f"Dividend treatment: {provenance.dividend_treatment.value}",
        f"Split treatment: {provenance.split_treatment.value}",
        f"Data SHA-256: {result.data_sha256}",
        "",
        "Evaluation",
        "----------",
        f"Evaluation ID: {result.evaluation_id}",
        f"Strategy: {development.metadata.strategy.name.value}",
        f"Parameters: {_strategy_parameters(result)}",
        f"Configuration fingerprint: {result.configuration_sha256}",
        f"Warm-up policy: {result.split.warmup_policy.value}",
        f"Warm-up bars: {result.warmup_bar_count}",
        "",
        "Development Period",
        "------------------",
        _period_line(development),
        f"Total return: {_optional_decimal(stability.development_total_return)}",
        f"Excess return: {_optional_decimal(stability.development_excess_return)}",
        f"Maximum drawdown: {_optional_decimal(stability.development_maximum_drawdown)}",
        f"Periodic Sharpe ratio: {_optional_decimal(stability.development_sharpe)}",
        f"Trades: {stability.development_trade_count}",
        "",
        "Holdout Period",
        "--------------",
        _period_line(holdout),
        f"Total return: {_optional_decimal(stability.holdout_total_return)}",
        f"Excess return: {_optional_decimal(stability.holdout_excess_return)}",
        f"Maximum drawdown: {_optional_decimal(stability.holdout_maximum_drawdown)}",
        f"Periodic Sharpe ratio: {_optional_decimal(stability.holdout_sharpe)}",
        f"Trades: {stability.holdout_trade_count}",
        "",
        "Change from Development to Holdout",
        "----------------------------------",
        f"Total-return change: {_optional_decimal(stability.total_return_change)}",
        f"Excess-return change: {_optional_decimal(stability.excess_return_change)}",
        f"Drawdown change: {_optional_decimal(stability.maximum_drawdown_change)}",
        f"Sharpe change: {_optional_decimal(stability.sharpe_change)}",
        "",
        (
            "Note: The software reports differences but does not determine whether "
            "the strategy is robust, profitable, or likely to perform in the future."
        ),
    ]
    if result.quality_summary.suspicious_return_count:
        lines.extend(
            [
                "",
                (
                    "Data warning: suspicious close returns may represent a stock split, "
                    "corporate action, bad data, or genuine extreme movement; no values "
                    "were repaired or removed."
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def _period_line(report: StructuredBacktestReport) -> str:
    """Return the actual inclusive observation range for one structured report."""

    if not isinstance(report, StructuredBacktestReport):
        raise TypeError("report must be StructuredBacktestReport")
    dataset = report.metadata.dataset
    return f"Period: {dataset.actual_start.isoformat()} to {dataset.actual_end.isoformat()}"


def _optional_decimal(value: Decimal | None) -> str:
    """Render an exact Decimal or an explicit unavailable marker."""

    return "Not available" if value is None else str(value)


def _strategy_parameters(result: SplitEvaluationResult) -> str:
    """Return stable human-readable parameters for supported strategies."""

    parameters = result.development.metadata.strategy.parameters
    if isinstance(parameters, SmaCrossoverParameters):
        return f"fast={parameters.fast_window}, slow={parameters.slow_window}"
    if isinstance(parameters, DonchianBreakoutParameters):
        return f"entry={parameters.entry_window}, exit={parameters.exit_window}"
    raise TypeError("unsupported strategy parameter model")
