"""Split-aware execution composed from the existing single-run pipeline."""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from trading_research.backtesting.pipeline import (
    BacktestRunRequest,
    build_execution_report,
    run_backtest_pipeline_with_bars,
)
from trading_research.evaluation.fingerprint import configuration_sha256
from trading_research.evaluation.models import (
    DEFAULT_SUSPICIOUS_RETURN_THRESHOLD,
    DatasetProvenance,
    EvaluationSplit,
    FrozenEvaluationConfiguration,
    SplitEvaluationResult,
    SplitStabilityComparison,
    WarmupPolicy,
)
from trading_research.evaluation.quality import calculate_dataset_quality
from trading_research.experiments.models import BenchmarkSelection
from trading_research.models import MarketBar, normalize_symbol
from trading_research.reporting import StructuredBacktestReport
from trading_research.strategies import create_strategy
from trading_research.strategies.base import Strategy


def run_split_evaluation(
    *,
    evaluation_id: str,
    data_path: Path,
    symbol: str,
    bars: Sequence[MarketBar],
    split: EvaluationSplit,
    configuration: FrozenEvaluationConfiguration,
    provenance: DatasetProvenance,
    git_commit: str,
    suspicious_return_threshold: Decimal = DEFAULT_SUSPICIOUS_RETURN_THRESHOLD,
    generated_at: datetime | None = None,
    base_directory: Path | None = None,
) -> SplitEvaluationResult:
    """Evaluate one frozen configuration on development and fresh holdout accounts."""

    if not isinstance(split, EvaluationSplit):
        raise TypeError("split must be EvaluationSplit")
    if not isinstance(configuration, FrozenEvaluationConfiguration):
        raise TypeError("configuration must be FrozenEvaluationConfiguration")
    if not isinstance(provenance, DatasetProvenance):
        raise TypeError("provenance must be DatasetProvenance")
    selected_symbol = normalize_symbol(symbol)
    snapshot = tuple(bars)
    if not snapshot:
        raise ValueError("split evaluation requires at least one market bar")
    if any(not isinstance(bar, MarketBar) for bar in snapshot):
        raise TypeError("split evaluation bars must be MarketBar values")
    if any(bar.symbol != selected_symbol for bar in snapshot):
        raise ValueError("split evaluation bars must match the requested symbol")
    if any(
        current.timestamp <= previous.timestamp
        for previous, current in zip(snapshot, snapshot[1:], strict=False)
    ):
        raise ValueError("split evaluation bars must be strictly chronological")
    strategy = create_strategy(
        configuration.strategy.name,
        configuration.strategy.parameters,
    )
    validate_strategy_data_policy(
        strategy,
        provenance,
        allow_unapproved_volume=configuration.allow_unapproved_volume,
    )

    development_bars = tuple(
        bar
        for bar in snapshot
        if split.development_start <= bar.timestamp <= split.development_end
    )
    holdout_bars = tuple(
        bar for bar in snapshot if split.holdout_start <= bar.timestamp <= split.holdout_end
    )
    timestamp = datetime.now(UTC) if generated_at is None else generated_at
    fingerprint = configuration_sha256(configuration)
    include_benchmark = configuration.benchmark is BenchmarkSelection.BUY_AND_HOLD

    development_request = BacktestRunRequest(
        data_path=Path(data_path),
        symbol=selected_symbol,
        strategy_configuration=configuration.strategy,
        backtest_configuration=configuration.backtest,
        start=split.development_start,
        end=split.development_end,
        risk_metric_settings=configuration.risk_metrics,
        include_buy_and_hold_benchmark=include_benchmark,
    )
    development_execution = run_backtest_pipeline_with_bars(
        development_request,
        development_bars,
    )
    development_report = build_execution_report(
        development_execution,
        run_id=f"{evaluation_id}:development",
        generated_at=timestamp,
        git_commit=git_commit,
        base_directory=base_directory,
    )

    holdout_request = BacktestRunRequest(
        data_path=Path(data_path),
        symbol=selected_symbol,
        strategy_configuration=configuration.strategy,
        backtest_configuration=configuration.backtest,
        start=split.holdout_start,
        end=split.holdout_end,
        risk_metric_settings=configuration.risk_metrics,
        include_buy_and_hold_benchmark=include_benchmark,
    )
    warmup_bars: tuple[MarketBar, ...] = ()
    if split.warmup_policy is WarmupPolicy.CARRY_HISTORY:
        required = strategy.required_warmup_bars()
        preceding = tuple(bar for bar in snapshot if bar.timestamp < split.holdout_start)
        if len(preceding) < required:
            raise ValueError(
                "insufficient pre-holdout data for carry-history warm-up: "
                f"requires {required} bars; found {len(preceding)}"
            )
        warmup_bars = preceding[-required:] if required else ()
    holdout_execution = run_backtest_pipeline_with_bars(
        holdout_request,
        holdout_bars,
        warmup_bars=warmup_bars,
    )
    holdout_report = build_execution_report(
        holdout_execution,
        run_id=f"{evaluation_id}:holdout",
        generated_at=timestamp,
        git_commit=git_commit,
        base_directory=base_directory,
    )

    development_fingerprint = configuration_sha256(configuration)
    holdout_fingerprint = configuration_sha256(configuration)
    if development_fingerprint != holdout_fingerprint:
        raise ValueError("development and holdout configuration fingerprints differ")
    data_sha256 = development_report.metadata.dataset.sha256
    if data_sha256 is None:
        raise ValueError("split evaluation requires a dataset SHA-256")
    if data_sha256 != holdout_report.metadata.dataset.sha256:
        raise ValueError("development and holdout reports must use the same dataset bytes")

    return SplitEvaluationResult(
        evaluation_id=evaluation_id,
        generated_at=timestamp,
        git_commit=git_commit,
        data_sha256=data_sha256,
        configuration_sha256=fingerprint,
        development_configuration_sha256=development_fingerprint,
        holdout_configuration_sha256=holdout_fingerprint,
        provenance=provenance,
        quality_summary=calculate_dataset_quality(
            snapshot,
            suspicious_return_threshold=suspicious_return_threshold,
        ),
        split=split,
        warmup_bar_count=len(warmup_bars),
        warmup_first_timestamp=(warmup_bars[0].timestamp if warmup_bars else None),
        warmup_last_timestamp=(warmup_bars[-1].timestamp if warmup_bars else None),
        development=development_report,
        holdout=holdout_report,
        development_benchmark_comparison=development_report.benchmark_comparison,
        holdout_benchmark_comparison=holdout_report.benchmark_comparison,
        stability_comparison=calculate_split_stability(
            development_report,
            holdout_report,
        ),
    )


def validate_strategy_data_policy(
    strategy: Strategy,
    provenance: DatasetProvenance,
    *,
    allow_unapproved_volume: bool = False,
) -> None:
    """Block volume-dependent research unless provenance or an override permits it."""

    if not isinstance(strategy, Strategy):
        raise TypeError("strategy must be a Strategy")
    if not isinstance(provenance, DatasetProvenance):
        raise TypeError("provenance must be DatasetProvenance")
    if not isinstance(allow_unapproved_volume, bool):
        raise TypeError("allow_unapproved_volume must be a bool")
    fields = strategy.required_market_data_fields()
    if not isinstance(fields, frozenset) or any(
        not isinstance(field, str) or not field.strip() for field in fields
    ):
        raise TypeError("strategy market-data fields must be a frozenset of names")
    normalized = frozenset(field.strip().lower() for field in fields)
    if (
        "volume" in normalized
        and provenance.volume_approved_for_strategy_signals is not True
        and not allow_unapproved_volume
    ):
        raise ValueError(
            "strategy requires volume, but dataset provenance does not approve "
            "volume-based signals; set an explicit override only after review"
        )


def calculate_split_stability(
    development: StructuredBacktestReport,
    holdout: StructuredBacktestReport,
) -> SplitStabilityComparison:
    """Calculate raw holdout-minus-development metric differences."""

    if not isinstance(development, StructuredBacktestReport):
        raise TypeError("development must be StructuredBacktestReport")
    if not isinstance(holdout, StructuredBacktestReport):
        raise TypeError("holdout must be StructuredBacktestReport")
    development_excess = (
        None
        if development.benchmark_comparison is None
        else development.benchmark_comparison.excess_return
    )
    holdout_excess = (
        None
        if holdout.benchmark_comparison is None
        else holdout.benchmark_comparison.excess_return
    )
    development_sharpe = (
        None
        if development.risk_adjusted_metrics is None
        else development.risk_adjusted_metrics.periodic_sharpe_ratio
    )
    holdout_sharpe = (
        None
        if holdout.risk_adjusted_metrics is None
        else holdout.risk_adjusted_metrics.periodic_sharpe_ratio
    )
    return SplitStabilityComparison(
        development_total_return=development.performance.total_return,
        holdout_total_return=holdout.performance.total_return,
        total_return_change=(
            holdout.performance.total_return - development.performance.total_return
        ),
        development_excess_return=development_excess,
        holdout_excess_return=holdout_excess,
        excess_return_change=_optional_change(development_excess, holdout_excess),
        development_maximum_drawdown=(
            development.performance.maximum_percentage_drawdown
        ),
        holdout_maximum_drawdown=holdout.performance.maximum_percentage_drawdown,
        maximum_drawdown_change=(
            holdout.performance.maximum_percentage_drawdown
            - development.performance.maximum_percentage_drawdown
        ),
        development_sharpe=development_sharpe,
        holdout_sharpe=holdout_sharpe,
        sharpe_change=_optional_change(development_sharpe, holdout_sharpe),
        development_trade_count=development.trade_statistics.completed_trade_count,
        holdout_trade_count=holdout.trade_statistics.completed_trade_count,
        development_time_in_market=(
            development.exposure_statistics.time_in_market_ratio
        ),
        holdout_time_in_market=holdout.exposure_statistics.time_in_market_ratio,
    )


def _optional_change(
    development: Decimal | None,
    holdout: Decimal | None,
) -> Decimal | None:
    """Return holdout minus development only when both values are defined."""

    if development is None or holdout is None:
        return None
    return holdout - development
