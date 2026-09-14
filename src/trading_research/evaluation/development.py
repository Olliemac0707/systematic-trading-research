"""Sealed development-period evaluation, reporting, and candidate selection."""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum, StrEnum
from io import StringIO
from pathlib import Path

from pydantic import BaseModel

from trading_research.backtesting.pipeline import (
    BacktestRunRequest,
    build_execution_report,
    run_backtest_pipeline_with_bars,
)
from trading_research.data import CsvMarketDataProvider
from trading_research.errors import (
    BacktestError,
    MarketDataError,
    ReportWriteError,
    StrategyError,
)
from trading_research.evaluation.fingerprint import configuration_sha256
from trading_research.evaluation.manifest import (
    SplitEvaluationManifest,
    SplitEvaluationManifestEntry,
)
from trading_research.evaluation.models import (
    DEFAULT_SUSPICIOUS_RETURN_THRESHOLD,
    DatasetProvenance,
    DatasetQualitySummary,
    EvaluationSplit,
    FrozenEvaluationConfiguration,
    WarmupPolicy,
)
from trading_research.evaluation.provenance import load_dataset_provenance
from trading_research.evaluation.quality import calculate_dataset_quality
from trading_research.evaluation.runner import validate_strategy_data_policy
from trading_research.experiments.models import BenchmarkSelection
from trading_research.models import MarketBar, normalize_symbol
from trading_research.reporting import (
    StructuredBacktestReport,
    current_git_commit,
    installed_application_version,
    sha256_file,
    structured_report_to_json,
    write_text_export,
)
from trading_research.strategies import (
    DonchianBreakoutParameters,
    SmaCrossoverParameters,
    create_strategy,
)

DEVELOPMENT_RESULT_SCHEMA = "trading-research.development-evaluation"
DEVELOPMENT_BATCH_SCHEMA = "trading-research.development-evaluation-batch"
DEVELOPMENT_SCHEMA_VERSION = "1.0"
_EXPECTED_ASSET_COUNT = 8
_MINIMUM_POSITIVE_ASSETS = 6
_MINIMUM_DRAWDOWN_IMPROVEMENT_ASSETS = 5
_MINIMUM_COMPLETED_TRADES = 3

DEVELOPMENT_SUMMARY_COLUMNS = (
    "evaluation_id",
    "candidate_id",
    "status",
    "symbol",
    "strategy_name",
    "strategy_parameters",
    "development_start",
    "development_end",
    "warmup_policy",
    "warmup_bar_count",
    "starting_cash",
    "ending_cash",
    "ending_equity",
    "net_profit",
    "net_return",
    "maximum_absolute_drawdown",
    "maximum_percentage_drawdown",
    "drawdown_peak_timestamp",
    "drawdown_trough_timestamp",
    "drawdown_recovery_timestamp",
    "periodic_sharpe_ratio",
    "annualised_sharpe_ratio",
    "periodic_sortino_ratio",
    "annualised_sortino_ratio",
    "closed_trade_count",
    "winning_trade_count",
    "losing_trade_count",
    "breakeven_trade_count",
    "win_rate",
    "profit_factor",
    "gross_realised_pnl",
    "unrealised_pnl",
    "total_commission",
    "total_slippage",
    "time_in_market_ratio",
    "average_capital_utilisation_ratio",
    "maximum_capital_utilisation_ratio",
    "benchmark_return",
    "benchmark_maximum_percentage_drawdown",
    "reconciliation_status",
    "cash_reconciliation_difference",
    "equity_reconciliation_difference",
    "net_return_rank",
    "sharpe_rank",
    "drawdown_magnitude_rank",
    "dataset_sha256",
    "git_commit",
    "manifest_sha256",
    "application_version",
    "run_id",
    "warning_count",
    "warnings",
    "failure_kind",
    "failure_message",
)

CANDIDATE_SUMMARY_COLUMNS = (
    "candidate_id",
    "evaluation_count",
    "success_count",
    "reconciled_count",
    "positive_return_count",
    "positive_sharpe_count",
    "drawdown_improvement_count",
    "minimum_closed_trade_count",
    "assets_below_three_trades",
    "integrity_warning_count",
    "eligible",
    "disqualification_reasons",
    "overall_rank_score",
    "median_sharpe_ratio",
    "median_drawdown_magnitude",
    "median_net_return",
    "total_commission_and_slippage",
    "total_closed_trades",
)

BENCHMARK_COMPARISON_COLUMNS = (
    "evaluation_id",
    "candidate_id",
    "symbol",
    "strategy_net_return",
    "benchmark_return",
    "excess_return",
    "strategy_maximum_percentage_drawdown",
    "benchmark_maximum_percentage_drawdown",
    "maximum_drawdown_improvement",
)


class DevelopmentEvaluationStatus(StrEnum):
    """Stable development-only completion states."""

    SUCCESS = "success"
    FAILURE = "failure"


class DevelopmentFailureKind(StrEnum):
    """Expected retained failure categories."""

    CONFIGURATION = "configuration_error"
    MARKET_DATA = "market_data_error"
    EVALUATION = "evaluation_error"


@dataclass(frozen=True, slots=True)
class DevelopmentEvaluationResult:
    """One sealed development result containing no future-period result."""

    evaluation_id: str
    generated_at: datetime
    git_commit: str
    manifest_sha256: str
    configuration_sha256: str
    data_sha256: str
    provenance: DatasetProvenance
    quality_summary: DatasetQualitySummary
    split: EvaluationSplit
    warmup_bar_count: int
    warmup_first_timestamp: datetime | None
    warmup_last_timestamp: datetime | None
    report: StructuredBacktestReport
    integrity_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require exact identity, reconciliation, and a sealed date boundary."""

        if not self.evaluation_id:
            raise ValueError("evaluation_id must not be empty")
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        for name, digest in (
            ("manifest_sha256", self.manifest_sha256),
            ("configuration_sha256", self.configuration_sha256),
            ("data_sha256", self.data_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if self.report.metadata.run_id != f"{self.evaluation_id}:development":
            raise ValueError("development run ID must match the evaluation ID")
        if self.report.metadata.git_commit != self.git_commit:
            raise ValueError("development Git revision must match the batch revision")
        if self.report.metadata.dataset.sha256 != self.data_sha256:
            raise ValueError("development dataset hash must match the report")
        if not self.report.backtest_result.reconciliation.is_reconciled:
            raise ValueError("development result must reconcile exactly")
        actual_start = self.report.metadata.dataset.actual_start
        actual_end = self.report.metadata.dataset.actual_end
        if actual_start < self.split.development_start:
            raise ValueError("development report starts before its boundary")
        if actual_end > self.split.development_end:
            raise ValueError("development report ends after its boundary")
        if self.warmup_bar_count < 0:
            raise ValueError("warmup_bar_count must be non-negative")
        if self.warmup_bar_count == 0:
            if (
                self.warmup_first_timestamp is not None
                or self.warmup_last_timestamp is not None
            ):
                raise ValueError("empty development warm-up cannot have timestamps")
        elif (
            self.warmup_first_timestamp is None
            or self.warmup_last_timestamp is None
            or self.warmup_last_timestamp >= self.split.development_start
        ):
            raise ValueError("development warm-up must strictly precede its boundary")
        _validate_report_boundary(self.report, self.split.development_end)


@dataclass(frozen=True, slots=True)
class DevelopmentEvaluationRecord:
    """One ordered development success or explicit expected failure."""

    evaluation_id: str
    status: DevelopmentEvaluationStatus
    failure_kind: DevelopmentFailureKind | None
    failure_message: str | None
    output_directory: str | None
    result: DevelopmentEvaluationResult | None

    def __post_init__(self) -> None:
        """Require mutually exclusive success and failure payloads."""

        if not self.evaluation_id:
            raise ValueError("evaluation_id must not be empty")
        if self.status is DevelopmentEvaluationStatus.SUCCESS:
            if (
                self.failure_kind is not None
                or self.failure_message is not None
                or self.output_directory is None
                or self.result is None
            ):
                raise ValueError("successful development records require a result")
        elif (
            self.failure_kind is None
            or not self.failure_message
            or self.output_directory is not None
            or self.result is not None
        ):
            raise ValueError("failed development records require failure information")


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    """Pre-registered cross-asset eligibility and ranking facts."""

    candidate_id: str
    evaluation_count: int
    success_count: int
    reconciled_count: int
    positive_return_count: int
    positive_sharpe_count: int
    drawdown_improvement_count: int
    minimum_closed_trade_count: int | None
    assets_below_three_trades: tuple[str, ...]
    integrity_warning_count: int
    eligible: bool
    disqualification_reasons: tuple[str, ...]
    overall_rank_score: Decimal | None
    median_sharpe_ratio: Decimal | None
    median_drawdown_magnitude: Decimal | None
    median_net_return: Decimal | None
    total_commission_and_slippage: Decimal | None
    total_closed_trades: int | None


@dataclass(frozen=True, slots=True)
class DevelopmentEvaluationBatchResult:
    """Immutable ordered development batch plus frozen selection outcome."""

    source_manifest: str
    manifest_sha256: str
    git_commit: str
    generated_at: datetime
    run_id: str
    records: tuple[DevelopmentEvaluationRecord, ...]
    candidates: tuple[CandidateAssessment, ...]
    selected_candidate: str | None
    selection_status: str

    @property
    def success_count(self) -> int:
        """Return the number of completed development evaluations."""

        return sum(
            record.status is DevelopmentEvaluationStatus.SUCCESS
            for record in self.records
        )

    @property
    def failure_count(self) -> int:
        """Return the number of explicit failures."""

        return len(self.records) - self.success_count

    @property
    def succeeded(self) -> bool:
        """Return whether every declared development evaluation completed."""

        return self.failure_count == 0


def run_development_evaluation(
    *,
    evaluation_id: str,
    manifest_sha256: str,
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
) -> DevelopmentEvaluationResult:
    """Run only the inclusive development period and reject any later bar."""

    if not isinstance(split, EvaluationSplit):
        raise TypeError("split must be an EvaluationSplit")
    if not isinstance(configuration, FrozenEvaluationConfiguration):
        raise TypeError("configuration must be FrozenEvaluationConfiguration")
    if not isinstance(provenance, DatasetProvenance):
        raise TypeError("provenance must be DatasetProvenance")
    selected_symbol = normalize_symbol(symbol)
    snapshot = tuple(bars)
    if not snapshot:
        raise ValueError("development evaluation requires at least one market bar")
    if any(not isinstance(bar, MarketBar) for bar in snapshot):
        raise TypeError("development evaluation bars must be MarketBar values")
    if any(bar.symbol != selected_symbol for bar in snapshot):
        raise ValueError("development evaluation bars must match the requested symbol")
    if any(
        current.timestamp <= previous.timestamp
        for previous, current in zip(snapshot, snapshot[1:], strict=False)
    ):
        raise ValueError("development evaluation bars must be strictly chronological")
    if any(bar.timestamp > split.development_end for bar in snapshot):
        raise ValueError("development evaluation received a bar after its sealed boundary")

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
    preceding = tuple(
        bar for bar in snapshot if bar.timestamp < split.development_start
    )
    warmup_bars: tuple[MarketBar, ...] = ()
    if split.warmup_policy is WarmupPolicy.CARRY_HISTORY:
        required = strategy.required_warmup_bars()
        warmup_bars = preceding[-required:] if required else ()

    timestamp = datetime.now(UTC) if generated_at is None else generated_at
    include_benchmark = configuration.benchmark is BenchmarkSelection.BUY_AND_HOLD
    request = BacktestRunRequest(
        data_path=Path(data_path),
        symbol=selected_symbol,
        strategy_configuration=configuration.strategy,
        backtest_configuration=configuration.backtest,
        start=split.development_start,
        end=split.development_end,
        risk_metric_settings=configuration.risk_metrics,
        include_buy_and_hold_benchmark=include_benchmark,
    )
    execution = run_backtest_pipeline_with_bars(
        request,
        development_bars,
        warmup_bars=warmup_bars,
    )
    report = build_execution_report(
        execution,
        run_id=f"{evaluation_id}:development",
        generated_at=timestamp,
        git_commit=git_commit,
        base_directory=base_directory,
    )
    data_sha256 = report.metadata.dataset.sha256
    if data_sha256 is None:
        raise ValueError("development evaluation requires a dataset SHA-256")
    warnings = _integrity_warnings(report, development_bars)
    return DevelopmentEvaluationResult(
        evaluation_id=evaluation_id,
        generated_at=timestamp,
        git_commit=git_commit,
        manifest_sha256=manifest_sha256,
        configuration_sha256=configuration_sha256(configuration),
        data_sha256=data_sha256,
        provenance=provenance,
        quality_summary=calculate_dataset_quality(
            development_bars,
            suspicious_return_threshold=suspicious_return_threshold,
        ),
        split=split,
        warmup_bar_count=len(warmup_bars),
        warmup_first_timestamp=warmup_bars[0].timestamp if warmup_bars else None,
        warmup_last_timestamp=warmup_bars[-1].timestamp if warmup_bars else None,
        report=report,
        integrity_warnings=warnings,
    )


def run_development_evaluation_batch(
    manifest: SplitEvaluationManifest,
    output_directory: Path,
    *,
    overwrite: bool = False,
    repository: Path | None = None,
    generated_at: datetime | None = None,
) -> DevelopmentEvaluationBatchResult:
    """Run declared development periods sequentially and write sealed outputs."""

    if not isinstance(manifest, SplitEvaluationManifest):
        raise TypeError("manifest must be SplitEvaluationManifest")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")
    output = Path(output_directory)
    _prepare_output_directory(output, overwrite=overwrite)
    repository_path = Path.cwd() if repository is None else Path(repository)
    git_commit = current_git_commit(repository_path)
    if git_commit is None:
        raise BacktestError("could not determine current Git revision for development")
    manifest_sha256 = sha256_file(manifest.source_path)
    timestamp = datetime.now(UTC) if generated_at is None else generated_at
    run_id = f"development-{manifest_sha256[:12]}-{git_commit[:12]}"

    records: list[DevelopmentEvaluationRecord] = []
    for entry in manifest.entries:
        record = _run_development_entry(
            entry,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            git_commit=git_commit,
            generated_at=timestamp,
            repository=repository_path,
        )
        records.append(record)
        if record.result is not None and record.output_directory is not None:
            write_text_export(
                development_evaluation_to_json(record.result) + "\n",
                output / record.output_directory / "development-result.json",
                overwrite=overwrite,
                create_parents=True,
            )

    rank_values = _asset_level_ranks(records)
    candidates = _assess_candidates(records, rank_values)
    selected_candidate, selection_status = _select_candidate(candidates)
    result = DevelopmentEvaluationBatchResult(
        source_manifest=manifest.source_path.name,
        manifest_sha256=manifest_sha256,
        git_commit=git_commit,
        generated_at=timestamp,
        run_id=run_id,
        records=tuple(records),
        candidates=candidates,
        selected_candidate=selected_candidate,
        selection_status=selection_status,
    )
    _write_development_batch_outputs(
        result,
        output,
        rank_values=rank_values,
        overwrite=overwrite,
    )
    return result


def development_evaluation_to_json(
    result: DevelopmentEvaluationResult,
    *,
    indent: int = 2,
) -> str:
    """Serialize one development result without any future-period fields."""

    if not isinstance(result, DevelopmentEvaluationResult):
        raise TypeError("result must be DevelopmentEvaluationResult")
    report_payload = json.loads(structured_report_to_json(result.report))
    payload = {
        "schema": DEVELOPMENT_RESULT_SCHEMA,
        "schema_version": DEVELOPMENT_SCHEMA_VERSION,
        "evaluation_id": result.evaluation_id,
        "period": "development",
        "generated_at": result.generated_at,
        "git_commit": result.git_commit,
        "manifest_sha256": result.manifest_sha256,
        "configuration_sha256": result.configuration_sha256,
        "data_sha256": result.data_sha256,
        "provenance": result.provenance,
        "quality_summary": result.quality_summary,
        "development_boundary": {
            "start": result.split.development_start,
            "end": result.split.development_end,
            "inclusive": True,
            "warmup_policy": result.split.warmup_policy,
            "warmup_bar_count": result.warmup_bar_count,
            "warmup_first_timestamp": result.warmup_first_timestamp,
            "warmup_last_timestamp": result.warmup_last_timestamp,
        },
        "integrity": {
            "future_observations_contributed": False,
            "final_valuation_timestamp": (
                result.report.backtest_result.equity_curve[-1].timestamp
            ),
            "benchmark_final_timestamp": (
                None
                if result.report.benchmark is None
                else result.report.benchmark.result.equity_curve[-1].timestamp
            ),
            "reconciliation_status": (
                "PASS"
                if result.report.backtest_result.reconciliation.is_reconciled
                else "FAIL"
            ),
            "warnings": result.integrity_warnings,
        },
        "report": report_payload,
    }
    return json.dumps(
        _json_value(payload),
        ensure_ascii=False,
        indent=indent,
        allow_nan=False,
    )


def development_summary_to_csv(
    result: DevelopmentEvaluationBatchResult,
    rank_values: Mapping[tuple[str, str], Decimal | None],
) -> str:
    """Return one exact row per declared development evaluation."""

    rows = tuple(
        _development_summary_row(record, result, rank_values)
        for record in result.records
    )
    return _csv_text(DEVELOPMENT_SUMMARY_COLUMNS, rows)


def candidate_summary_to_csv(result: DevelopmentEvaluationBatchResult) -> str:
    """Return one exact cross-asset row per universal candidate."""

    rows = tuple(_candidate_row(candidate) for candidate in result.candidates)
    return _csv_text(CANDIDATE_SUMMARY_COLUMNS, rows)


def benchmark_comparison_to_csv(result: DevelopmentEvaluationBatchResult) -> str:
    """Return development-only strategy-versus-benchmark rows."""

    rows: list[dict[str, object]] = []
    for record in result.records:
        if record.result is None:
            continue
        report = record.result.report
        comparison = report.benchmark_comparison
        if comparison is None:
            continue
        rows.append(
            {
                "evaluation_id": record.evaluation_id,
                "candidate_id": _candidate_id(report),
                "symbol": report.metadata.dataset.symbol,
                "strategy_net_return": report.performance.total_return,
                "benchmark_return": comparison.benchmark_total_return,
                "excess_return": comparison.excess_return,
                "strategy_maximum_percentage_drawdown": (
                    report.performance.maximum_percentage_drawdown
                ),
                "benchmark_maximum_percentage_drawdown": (
                    comparison.benchmark_maximum_percentage_drawdown
                ),
                "maximum_drawdown_improvement": (
                    comparison.maximum_drawdown_improvement
                ),
            }
        )
    return _csv_text(BENCHMARK_COMPARISON_COLUMNS, tuple(rows))


def _run_development_entry(
    entry: SplitEvaluationManifestEntry,
    *,
    manifest: SplitEvaluationManifest,
    manifest_sha256: str,
    git_commit: str,
    generated_at: datetime,
    repository: Path,
) -> DevelopmentEvaluationRecord:
    """Execute one sealed entry or retain its expected failure."""

    if entry.configuration_error is not None:
        return _failure_record(
            entry.evaluation_id,
            DevelopmentFailureKind.CONFIGURATION,
            entry.configuration_error,
        )
    definition = entry.definition
    if definition is None:
        raise AssertionError("validated development entry is missing its definition")
    dataset = (
        definition.dataset
        if definition.dataset.is_absolute()
        else manifest.source_path.parent / definition.dataset
    )
    if not dataset.is_file():
        return _failure_record(
            entry.evaluation_id,
            DevelopmentFailureKind.MARKET_DATA,
            f"dataset file not found: {dataset.name}",
        )
    try:
        provenance = load_dataset_provenance(dataset, definition.provenance)
        configuration = definition.frozen_configuration()
    except (TypeError, ValueError) as exc:
        return _failure_record(
            entry.evaluation_id,
            DevelopmentFailureKind.CONFIGURATION,
            _safe_failure_message(exc, dataset),
        )
    try:
        bars = tuple(
            CsvMarketDataProvider(dataset).get_historical_bars(
                symbol=definition.symbol,
                start=None,
                end=definition.split.development_end,
            )
        )
        evaluation = run_development_evaluation(
            evaluation_id=entry.evaluation_id,
            manifest_sha256=manifest_sha256,
            data_path=dataset,
            symbol=definition.symbol,
            bars=bars,
            split=definition.split,
            configuration=configuration,
            provenance=provenance,
            git_commit=git_commit,
            suspicious_return_threshold=definition.suspicious_return_threshold,
            generated_at=generated_at,
            base_directory=repository,
        )
    except (MarketDataError, OSError) as exc:
        return _failure_record(
            entry.evaluation_id,
            DevelopmentFailureKind.MARKET_DATA,
            _safe_failure_message(exc, dataset),
        )
    except (BacktestError, StrategyError, TypeError, ValueError) as exc:
        return _failure_record(
            entry.evaluation_id,
            DevelopmentFailureKind.EVALUATION,
            _safe_failure_message(exc, dataset),
        )
    return DevelopmentEvaluationRecord(
        evaluation_id=entry.evaluation_id,
        status=DevelopmentEvaluationStatus.SUCCESS,
        failure_kind=None,
        failure_message=None,
        output_directory=entry.evaluation_id,
        result=evaluation,
    )


def _failure_record(
    evaluation_id: str,
    kind: DevelopmentFailureKind,
    message: str,
) -> DevelopmentEvaluationRecord:
    """Create one explicit development failure record."""

    return DevelopmentEvaluationRecord(
        evaluation_id=evaluation_id,
        status=DevelopmentEvaluationStatus.FAILURE,
        failure_kind=kind,
        failure_message=message,
        output_directory=None,
        result=None,
    )


def _integrity_warnings(
    report: StructuredBacktestReport,
    bars: Sequence[MarketBar],
) -> tuple[str, ...]:
    """Return unresolved warnings that affect approved financial fields."""

    quality = calculate_dataset_quality(bars)
    warnings: list[str] = []
    if quality.suspicious_return_count:
        warnings.append("development data contains suspicious close returns")
    if not report.backtest_result.reconciliation.is_reconciled:
        warnings.append("account reconciliation failed")
    return tuple(warnings)


def _validate_report_boundary(
    report: StructuredBacktestReport,
    development_end: datetime,
) -> None:
    """Reject any financial or benchmark observation after the sealed end."""

    timestamps = [
        *(point.timestamp for point in report.backtest_result.equity_curve),
        *(trade.timestamp for trade in report.backtest_result.trades),
        *(decision.timestamp for decision in report.pre_trade_decisions),
        *(trade.entry_timestamp for trade in report.closed_trades),
        *(trade.exit_timestamp for trade in report.closed_trades),
    ]
    if report.benchmark is not None:
        timestamps.extend(
            point.timestamp for point in report.benchmark.result.equity_curve
        )
        timestamps.extend(trade.timestamp for trade in report.benchmark.result.trades)
    if any(timestamp > development_end for timestamp in timestamps):
        raise ValueError("development report contains an observation after its boundary")


def _asset_level_ranks(
    records: Sequence[DevelopmentEvaluationRecord],
) -> dict[tuple[str, str], Decimal | None]:
    """Rank all candidates within each asset using deterministic average ties."""

    by_symbol: dict[str, list[DevelopmentEvaluationRecord]] = {}
    for record in records:
        if record.result is not None:
            symbol = record.result.report.metadata.dataset.symbol
            by_symbol.setdefault(symbol, []).append(record)
    ranks: dict[tuple[str, str], Decimal | None] = {}
    for symbol_records in by_symbol.values():
        return_values = [
            (record.evaluation_id, record.result.report.performance.total_return)
            for record in symbol_records
            if record.result is not None
        ]
        sharpe_values: list[tuple[str, Decimal]] = []
        for record in symbol_records:
            if (
                record.result is None
                or record.result.report.risk_adjusted_metrics is None
                or record.result.report.risk_adjusted_metrics.periodic_sharpe_ratio
                is None
            ):
                continue
            sharpe_values.append(
                (
                    record.evaluation_id,
                    record.result.report.risk_adjusted_metrics.periodic_sharpe_ratio,
                )
            )
        drawdown_values = [
            (
                record.evaluation_id,
                abs(
                    record.result.report.performance.maximum_percentage_drawdown
                ),
            )
            for record in symbol_records
            if record.result is not None
        ]
        for evaluation_id, rank in _average_ranks(
            return_values,
            higher_is_better=True,
        ).items():
            ranks[(evaluation_id, "net_return")] = rank
        for evaluation_id, rank in _average_ranks(
            sharpe_values,
            higher_is_better=True,
        ).items():
            ranks[(evaluation_id, "sharpe")] = rank
        for evaluation_id, rank in _average_ranks(
            drawdown_values,
            higher_is_better=False,
        ).items():
            ranks[(evaluation_id, "drawdown")] = rank
    return ranks


def _average_ranks(
    values: Sequence[tuple[str, Decimal]],
    *,
    higher_is_better: bool,
) -> dict[str, Decimal]:
    """Return 1-based ranks with exact average ranks for ties."""

    ordered = sorted(values, key=lambda item: item[1], reverse=higher_is_better)
    result: dict[str, Decimal] = {}
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[index][1]:
            end += 1
        rank = (Decimal(index + 1) + Decimal(end + 1)) / Decimal("2")
        for position in range(index, end + 1):
            result[ordered[position][0]] = rank
        index = end + 1
    return result


def _assess_candidates(
    records: Sequence[DevelopmentEvaluationRecord],
    rank_values: Mapping[tuple[str, str], Decimal | None],
) -> tuple[CandidateAssessment, ...]:
    """Apply the frozen universal-candidate rules across all assets."""

    candidate_order: list[str] = []
    grouped: dict[str, list[DevelopmentEvaluationRecord]] = {}
    for record in records:
        candidate_id = (
            _candidate_id(record.result.report)
            if record.result is not None
            else _candidate_id_from_evaluation_id(record.evaluation_id)
        )
        if candidate_id not in grouped:
            candidate_order.append(candidate_id)
            grouped[candidate_id] = []
        grouped[candidate_id].append(record)

    assessments: list[CandidateAssessment] = []
    for candidate_id in candidate_order:
        candidate_records = grouped[candidate_id]
        successful = tuple(
            record.result for record in candidate_records if record.result is not None
        )
        reports = tuple(result.report for result in successful)
        reconciled_count = sum(
            report.backtest_result.reconciliation.is_reconciled for report in reports
        )
        positive_return_count = sum(
            report.performance.total_return > 0 for report in reports
        )
        positive_sharpe_count = sum(
            report.risk_adjusted_metrics is not None
            and report.risk_adjusted_metrics.periodic_sharpe_ratio is not None
            and report.risk_adjusted_metrics.periodic_sharpe_ratio > 0
            for report in reports
        )
        drawdown_improvement_count = sum(
            report.benchmark_comparison is not None
            and abs(report.performance.maximum_percentage_drawdown)
            < abs(
                report.benchmark_comparison.benchmark_maximum_percentage_drawdown
            )
            for report in reports
        )
        trade_counts = tuple(
            report.trade_statistics.completed_trade_count for report in reports
        )
        low_trade_assets = tuple(
            report.metadata.dataset.symbol
            for report in reports
            if report.trade_statistics.completed_trade_count
            < _MINIMUM_COMPLETED_TRADES
        )
        warning_count = sum(
            len(result.integrity_warnings) for result in successful
        )
        reasons: list[str] = []
        if len(candidate_records) != _EXPECTED_ASSET_COUNT:
            reasons.append(
                f"expected {_EXPECTED_ASSET_COUNT} assets; found {len(candidate_records)}"
            )
        if len(successful) != _EXPECTED_ASSET_COUNT:
            reasons.append(
                f"completed {len(successful)} of {_EXPECTED_ASSET_COUNT} evaluations"
            )
        if reconciled_count != _EXPECTED_ASSET_COUNT:
            reasons.append(
                f"reconciled {reconciled_count} of {_EXPECTED_ASSET_COUNT} evaluations"
            )
        if positive_return_count < _MINIMUM_POSITIVE_ASSETS:
            reasons.append(
                f"positive net return in {positive_return_count} of "
                f"{_EXPECTED_ASSET_COUNT} assets"
            )
        if positive_sharpe_count < _MINIMUM_POSITIVE_ASSETS:
            reasons.append(
                f"positive Sharpe ratio in {positive_sharpe_count} of "
                f"{_EXPECTED_ASSET_COUNT} assets"
            )
        if drawdown_improvement_count < _MINIMUM_DRAWDOWN_IMPROVEMENT_ASSETS:
            reasons.append(
                "smaller drawdown magnitude in "
                f"{drawdown_improvement_count} of {_EXPECTED_ASSET_COUNT} assets"
            )
        if low_trade_assets:
            reasons.append(
                "fewer than three closed trades for "
                + ", ".join(low_trade_assets)
            )
        if warning_count:
            reasons.append(f"{warning_count} unresolved integrity warning(s)")

        ranks = tuple(
            rank_values.get((record.evaluation_id, metric))
            for record in candidate_records
            for metric in ("net_return", "sharpe", "drawdown")
        )
        overall_rank = (
            sum((rank for rank in ranks if rank is not None), Decimal("0"))
            / Decimal(len(ranks))
            if len(ranks) == _EXPECTED_ASSET_COUNT * 3
            and all(rank is not None for rank in ranks)
            else None
        )
        sharpes = tuple(
            report.risk_adjusted_metrics.periodic_sharpe_ratio
            for report in reports
            if report.risk_adjusted_metrics is not None
            and report.risk_adjusted_metrics.periodic_sharpe_ratio is not None
        )
        returns = tuple(report.performance.total_return for report in reports)
        drawdowns = tuple(
            abs(report.performance.maximum_percentage_drawdown)
            for report in reports
        )
        total_cost = sum(
            (
                report.performance.total_commission
                + report.performance.total_adverse_slippage_cost
                for report in reports
            ),
            Decimal("0"),
        )
        assessments.append(
            CandidateAssessment(
                candidate_id=candidate_id,
                evaluation_count=len(candidate_records),
                success_count=len(successful),
                reconciled_count=reconciled_count,
                positive_return_count=positive_return_count,
                positive_sharpe_count=positive_sharpe_count,
                drawdown_improvement_count=drawdown_improvement_count,
                minimum_closed_trade_count=min(trade_counts) if trade_counts else None,
                assets_below_three_trades=low_trade_assets,
                integrity_warning_count=warning_count,
                eligible=not reasons,
                disqualification_reasons=tuple(reasons),
                overall_rank_score=overall_rank,
                median_sharpe_ratio=(
                    statistics.median(sharpes)
                    if len(sharpes) == _EXPECTED_ASSET_COUNT
                    else None
                ),
                median_drawdown_magnitude=(
                    statistics.median(drawdowns)
                    if len(drawdowns) == _EXPECTED_ASSET_COUNT
                    else None
                ),
                median_net_return=(
                    statistics.median(returns)
                    if len(returns) == _EXPECTED_ASSET_COUNT
                    else None
                ),
                total_commission_and_slippage=(
                    total_cost
                    if len(reports) == _EXPECTED_ASSET_COUNT
                    else None
                ),
                total_closed_trades=(
                    sum(trade_counts)
                    if len(trade_counts) == _EXPECTED_ASSET_COUNT
                    else None
                ),
            )
        )
    return tuple(assessments)


def _select_candidate(
    candidates: Sequence[CandidateAssessment],
) -> tuple[str | None, str]:
    """Apply the frozen ranking and tie-breakers to eligible candidates."""

    eligible = tuple(
        candidate
        for candidate in candidates
        if candidate.eligible
        and candidate.overall_rank_score is not None
        and candidate.median_sharpe_ratio is not None
        and candidate.median_drawdown_magnitude is not None
        and candidate.median_net_return is not None
        and candidate.total_commission_and_slippage is not None
        and candidate.total_closed_trades is not None
    )
    if not eligible:
        return None, "NO DEVELOPMENT CANDIDATE QUALIFIED"

    def key(candidate: CandidateAssessment) -> tuple[Decimal, ...]:
        assert candidate.overall_rank_score is not None
        assert candidate.median_sharpe_ratio is not None
        assert candidate.median_drawdown_magnitude is not None
        assert candidate.median_net_return is not None
        assert candidate.total_commission_and_slippage is not None
        assert candidate.total_closed_trades is not None
        return (
            candidate.overall_rank_score,
            -candidate.median_sharpe_ratio,
            candidate.median_drawdown_magnitude,
            -candidate.median_net_return,
            candidate.total_commission_and_slippage,
            Decimal(candidate.total_closed_trades),
        )

    ordered = sorted(eligible, key=key)
    if len(ordered) > 1 and key(ordered[0]) == key(ordered[1]):
        return None, "ELIGIBLE CANDIDATES REMAIN TIED AFTER ALL PRE-REGISTERED RULES"
    return ordered[0].candidate_id, "SELECTED"


def _write_development_batch_outputs(
    result: DevelopmentEvaluationBatchResult,
    output: Path,
    *,
    rank_values: Mapping[tuple[str, str], Decimal | None],
    overwrite: bool,
) -> None:
    """Write all aggregate development artifacts atomically."""

    summary_rows = tuple(
        _development_summary_row(record, result, rank_values)
        for record in result.records
    )
    summary_json = {
        "schema": DEVELOPMENT_BATCH_SCHEMA,
        "schema_version": DEVELOPMENT_SCHEMA_VERSION,
        "run_id": result.run_id,
        "source_manifest": result.source_manifest,
        "manifest_sha256": result.manifest_sha256,
        "git_commit": result.git_commit,
        "generated_at": result.generated_at,
        "evaluation_count": len(result.records),
        "success_count": result.success_count,
        "failure_count": result.failure_count,
        "results": summary_rows,
    }
    selection = {
        "schema": "trading-research.development-selection-policy",
        "schema_version": DEVELOPMENT_SCHEMA_VERSION,
        "rules": {
            "required_successes": 8,
            "required_reconciliations": 8,
            "minimum_positive_return_assets": 6,
            "minimum_positive_sharpe_assets": 6,
            "minimum_drawdown_improvement_assets": 5,
            "minimum_closed_trades_per_asset": 3,
            "maximum_unresolved_integrity_warnings": 0,
        },
        "ranking": {
            "asset_measures": (
                "net_return_descending",
                "periodic_sharpe_descending",
                "maximum_drawdown_magnitude_ascending",
            ),
            "tie_rank_policy": "exact average rank",
            "overall_rank_score": "mean of 24 asset-level ranks",
            "tie_breakers": (
                "higher median periodic Sharpe ratio",
                "smaller median maximum-drawdown magnitude",
                "higher median net return",
                "lower total commission plus slippage",
                "fewer total closed trades",
            ),
        },
        "candidates": result.candidates,
        "selected_candidate": result.selected_candidate,
        "selection_status": result.selection_status,
    }
    run_metadata = {
        "schema": "trading-research.development-run-integrity",
        "schema_version": DEVELOPMENT_SCHEMA_VERSION,
        "run_id": result.run_id,
        "generated_at": result.generated_at,
        "application_version": installed_application_version(),
        "git_commit": result.git_commit,
        "source_manifest": result.source_manifest,
        "manifest_sha256": result.manifest_sha256,
        "period": {
            "name": "development",
            "start": _common_boundary(result.records, "start"),
            "end": _common_boundary(result.records, "end"),
            "inclusive": True,
        },
        "evaluation_count": len(result.records),
        "success_count": result.success_count,
        "failure_count": result.failure_count,
        "strategy_network_requests": False,
        "volume_used_by_strategies": False,
        "future_observations_contributed": False,
        "datasets": _dataset_integrity_rows(result.records),
    }
    outputs = {
        "development-summary.csv": development_summary_to_csv(result, rank_values),
        "development-summary.json": json.dumps(
            _json_value(summary_json),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        "development-report.txt": _format_development_report(result),
        "candidate-summary.csv": candidate_summary_to_csv(result),
        "benchmark-comparison.csv": benchmark_comparison_to_csv(result),
        "selection-policy.json": json.dumps(
            _json_value(selection),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        "run-metadata.json": json.dumps(
            _json_value(run_metadata),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    }
    for name, content in outputs.items():
        write_text_export(
            content,
            output / name,
            overwrite=overwrite,
            create_parents=True,
        )


def _development_summary_row(
    record: DevelopmentEvaluationRecord,
    batch: DevelopmentEvaluationBatchResult,
    rank_values: Mapping[tuple[str, str], Decimal | None],
) -> dict[str, object]:
    """Return one exact development summary row."""

    if record.result is None:
        return {
            column: (
                record.evaluation_id
                if column == "evaluation_id"
                else record.status.value
                if column == "status"
                else record.failure_kind.value
                if column == "failure_kind" and record.failure_kind is not None
                else record.failure_message
                if column == "failure_message"
                else ""
            )
            for column in DEVELOPMENT_SUMMARY_COLUMNS
        }
    result = record.result
    report = result.report
    performance = report.performance
    trades = report.trade_statistics
    risk = report.risk_adjusted_metrics
    exposure = report.exposure_statistics
    comparison = report.benchmark_comparison
    reconciliation = report.backtest_result.reconciliation
    return {
        "evaluation_id": record.evaluation_id,
        "candidate_id": _candidate_id(report),
        "status": record.status.value,
        "symbol": report.metadata.dataset.symbol,
        "strategy_name": report.metadata.strategy.name.value,
        "strategy_parameters": json.dumps(
            _strategy_parameters(report),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "development_start": result.split.development_start,
        "development_end": result.split.development_end,
        "warmup_policy": result.split.warmup_policy.value,
        "warmup_bar_count": result.warmup_bar_count,
        "starting_cash": performance.starting_cash,
        "ending_cash": performance.ending_cash,
        "ending_equity": performance.ending_equity,
        "net_profit": performance.net_profit,
        "net_return": performance.total_return,
        "maximum_absolute_drawdown": performance.maximum_absolute_drawdown,
        "maximum_percentage_drawdown": performance.maximum_percentage_drawdown,
        "drawdown_peak_timestamp": performance.drawdown_peak_timestamp,
        "drawdown_trough_timestamp": performance.drawdown_trough_timestamp,
        "drawdown_recovery_timestamp": performance.drawdown_recovery_timestamp,
        "periodic_sharpe_ratio": (
            None if risk is None else risk.periodic_sharpe_ratio
        ),
        "annualised_sharpe_ratio": (
            None if risk is None else risk.annualised_sharpe_ratio
        ),
        "periodic_sortino_ratio": (
            None if risk is None else risk.periodic_sortino_ratio
        ),
        "annualised_sortino_ratio": (
            None if risk is None else risk.annualised_sortino_ratio
        ),
        "closed_trade_count": trades.completed_trade_count,
        "winning_trade_count": trades.winning_trade_count,
        "losing_trade_count": trades.losing_trade_count,
        "breakeven_trade_count": trades.breakeven_trade_count,
        "win_rate": trades.win_rate,
        "profit_factor": trades.profit_factor,
        "gross_realised_pnl": performance.gross_realised_pnl,
        "unrealised_pnl": performance.unrealised_pnl,
        "total_commission": performance.total_commission,
        "total_slippage": performance.total_adverse_slippage_cost,
        "time_in_market_ratio": exposure.time_in_market_ratio,
        "average_capital_utilisation_ratio": (
            exposure.average_capital_utilisation_ratio
        ),
        "maximum_capital_utilisation_ratio": (
            exposure.maximum_capital_utilisation_ratio
        ),
        "benchmark_return": (
            None if comparison is None else comparison.benchmark_total_return
        ),
        "benchmark_maximum_percentage_drawdown": (
            None
            if comparison is None
            else comparison.benchmark_maximum_percentage_drawdown
        ),
        "reconciliation_status": (
            "PASS" if reconciliation.is_reconciled else "FAIL"
        ),
        "cash_reconciliation_difference": reconciliation.cash_difference,
        "equity_reconciliation_difference": (
            reconciliation.reconciliation_difference
        ),
        "net_return_rank": rank_values.get(
            (record.evaluation_id, "net_return")
        ),
        "sharpe_rank": rank_values.get((record.evaluation_id, "sharpe")),
        "drawdown_magnitude_rank": rank_values.get(
            (record.evaluation_id, "drawdown")
        ),
        "dataset_sha256": result.data_sha256,
        "git_commit": result.git_commit,
        "manifest_sha256": batch.manifest_sha256,
        "application_version": report.metadata.application_version,
        "run_id": report.metadata.run_id,
        "warning_count": len(result.integrity_warnings),
        "warnings": "; ".join(result.integrity_warnings),
        "failure_kind": "",
        "failure_message": "",
    }


def _candidate_row(candidate: CandidateAssessment) -> dict[str, object]:
    """Return one candidate CSV row."""

    return {
        "candidate_id": candidate.candidate_id,
        "evaluation_count": candidate.evaluation_count,
        "success_count": candidate.success_count,
        "reconciled_count": candidate.reconciled_count,
        "positive_return_count": candidate.positive_return_count,
        "positive_sharpe_count": candidate.positive_sharpe_count,
        "drawdown_improvement_count": candidate.drawdown_improvement_count,
        "minimum_closed_trade_count": candidate.minimum_closed_trade_count,
        "assets_below_three_trades": ", ".join(
            candidate.assets_below_three_trades
        ),
        "integrity_warning_count": candidate.integrity_warning_count,
        "eligible": candidate.eligible,
        "disqualification_reasons": "; ".join(
            candidate.disqualification_reasons
        ),
        "overall_rank_score": candidate.overall_rank_score,
        "median_sharpe_ratio": candidate.median_sharpe_ratio,
        "median_drawdown_magnitude": candidate.median_drawdown_magnitude,
        "median_net_return": candidate.median_net_return,
        "total_commission_and_slippage": (
            candidate.total_commission_and_slippage
        ),
        "total_closed_trades": candidate.total_closed_trades,
    }


def _format_development_report(result: DevelopmentEvaluationBatchResult) -> str:
    """Return a concise human-readable development-only report."""

    lines = [
        "APPROVED ETF STUDY — DEVELOPMENT RESULTS",
        "========================================",
        f"Run ID: {result.run_id}",
        f"Application version: {installed_application_version()}",
        f"Manifest SHA-256: {result.manifest_sha256}",
        f"Git commit: {result.git_commit}",
        f"Evaluations: {len(result.records)}",
        f"Completed: {result.success_count}",
        f"Failed: {result.failure_count}",
        "Future-period observations contributed: false",
        "",
        "EVALUATION STATUS",
        "-----------------",
    ]
    for record in result.records:
        detail = (
            "COMPLETED"
            if record.result is not None
            else f"FAILURE — {record.failure_message}"
        )
        lines.append(f"{record.evaluation_id}: {detail}")
    lines.extend(("", "CANDIDATE ELIGIBILITY", "---------------------"))
    for candidate in result.candidates:
        status = "ELIGIBLE" if candidate.eligible else "DISQUALIFIED"
        lines.append(f"{candidate.candidate_id}: {status}")
        lines.append(
            "  positive returns / positive Sharpe / smaller drawdown: "
            f"{candidate.positive_return_count} / "
            f"{candidate.positive_sharpe_count} / "
            f"{candidate.drawdown_improvement_count}"
        )
        lines.append(
            "  overall rank score: "
            + _optional_text(candidate.overall_rank_score)
        )
        for reason in candidate.disqualification_reasons:
            lines.append(f"  reason: {reason}")
    lines.extend(
        (
            "",
            "SELECTION",
            "---------",
            result.selection_status,
            (
                "Selected universal candidate: "
                + (
                    "not available"
                    if result.selected_candidate is None
                    else result.selected_candidate
                )
            ),
            "",
        )
    )
    return "\n".join(lines)


def _candidate_id(report: StructuredBacktestReport) -> str:
    """Return the canonical universal configuration identifier."""

    parameters = report.metadata.strategy.parameters
    if isinstance(parameters, SmaCrossoverParameters):
        return f"sma-{parameters.fast_window}-{parameters.slow_window}"
    if isinstance(parameters, DonchianBreakoutParameters):
        return f"donchian-{parameters.entry_window}-{parameters.exit_window}"
    raise TypeError("unsupported development strategy parameters")


def _candidate_id_from_evaluation_id(evaluation_id: str) -> str:
    """Recover the declared candidate suffix for a failed approved entry."""

    parts = evaluation_id.split("-", 1)
    return parts[1] if len(parts) == 2 else evaluation_id


def _strategy_parameters(report: StructuredBacktestReport) -> dict[str, int]:
    """Return exact typed parameters for JSON and CSV."""

    parameters = report.metadata.strategy.parameters
    if isinstance(parameters, SmaCrossoverParameters):
        return {
            "fast_window": parameters.fast_window,
            "slow_window": parameters.slow_window,
        }
    if isinstance(parameters, DonchianBreakoutParameters):
        return {
            "entry_window": parameters.entry_window,
            "exit_window": parameters.exit_window,
        }
    raise TypeError("unsupported development strategy parameters")


def _common_boundary(
    records: Sequence[DevelopmentEvaluationRecord],
    boundary: str,
) -> datetime | None:
    """Return one common successful development boundary."""

    values = {
        (
            record.result.split.development_start
            if boundary == "start"
            else record.result.split.development_end
        )
        for record in records
        if record.result is not None
    }
    return next(iter(values)) if len(values) == 1 else None


def _dataset_integrity_rows(
    records: Sequence[DevelopmentEvaluationRecord],
) -> tuple[dict[str, object], ...]:
    """Return one unique dataset identity row per successful symbol."""

    selected: dict[str, dict[str, object]] = {}
    for record in records:
        if record.result is None:
            continue
        report = record.result.report
        symbol = report.metadata.dataset.symbol
        selected.setdefault(
            symbol,
            {
                "symbol": symbol,
                "identifier": report.metadata.dataset.identifier,
                "sha256": record.result.data_sha256,
                "source_name": record.result.provenance.source_name,
                "price_adjustment": record.result.provenance.price_adjustment,
                "dividend_treatment": record.result.provenance.dividend_treatment,
                "volume_approved_for_strategy_signals": (
                    record.result.provenance.volume_approved_for_strategy_signals
                ),
            },
        )
    return tuple(selected.values())


def _prepare_output_directory(path: Path, *, overwrite: bool) -> None:
    """Reject an existing output root unless replacement was explicit."""

    if path.exists():
        if not path.is_dir():
            raise ReportWriteError(
                f"development output path is not a directory: {path}"
            )
        if not overwrite:
            raise ReportWriteError(
                f"development output directory already exists: {path}"
            )
        return
    try:
        path.mkdir(parents=True)
    except OSError as exc:
        raise ReportWriteError(
            f"could not create development output directory {path}: {exc}"
        ) from exc


def _safe_failure_message(error: BaseException, dataset: Path) -> str:
    """Remove absolute local paths from persisted failures."""

    message = str(error)
    for candidate in {str(dataset), str(dataset.resolve(strict=False))}:
        if candidate:
            message = message.replace(candidate, dataset.name)
    return message


def _csv_text(
    columns: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> str:
    """Serialize stable CSV while distinguishing unavailable values from zero."""

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                column: _csv_value(row.get(column))
                for column in columns
            }
        )
    return buffer.getvalue()


def _csv_value(value: object) -> object:
    """Return a machine-readable scalar for exact CSV output."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(_timedelta_decimal_seconds(value))
    if isinstance(value, Enum):
        return value.value
    return value


def _json_value(value: object) -> object:
    """Recursively serialize exact models without binary floating point."""

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
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(f"unsupported development export value: {type(value).__name__}")


def _timedelta_decimal_seconds(value: timedelta) -> Decimal:
    """Return exact elapsed seconds without an intermediate float."""

    return (
        Decimal(value.days) * Decimal("86400")
        + Decimal(value.seconds)
        + Decimal(value.microseconds) / Decimal("1000000")
    )


def _optional_text(value: object) -> str:
    """Return explicit text for an unavailable metric."""

    return "not available" if value is None else str(value)
