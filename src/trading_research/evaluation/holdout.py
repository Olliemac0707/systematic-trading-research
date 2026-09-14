"""One-shot sealed holdout evaluation for the approved ETF study."""

from __future__ import annotations

import csv
import json
import tomllib
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
from trading_research.benchmarks import BenchmarkComparison
from trading_research.config import PositionSizingMode
from trading_research.errors import (
    BacktestError,
    EvaluationManifestError,
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
    DividendTreatment,
    EvaluationSplit,
    FrozenEvaluationConfiguration,
    PriceAdjustmentPolicy,
    SplitTreatment,
    WarmupPolicy,
)
from trading_research.evaluation.provenance import load_dataset_provenance
from trading_research.evaluation.quality import calculate_dataset_quality
from trading_research.evaluation.runner import validate_strategy_data_policy
from trading_research.experiments.models import BenchmarkSelection
from trading_research.models import (
    EndOfTestPolicy,
    ExecutionReason,
    MarketBar,
    normalize_symbol,
)
from trading_research.performance._decimal import exact_decimal_sum
from trading_research.reporting import (
    StructuredBacktestReport,
    current_git_commit,
    installed_application_version,
    sha256_file,
    structured_report_to_json,
    write_text_export,
)
from trading_research.strategies import (
    SmaCrossoverParameters,
    StrategyName,
    create_strategy,
)

HOLDOUT_RESULT_SCHEMA = "trading-research.sealed-holdout-evaluation"
HOLDOUT_BATCH_SCHEMA = "trading-research.sealed-holdout-batch"
HOLDOUT_SCHEMA_VERSION = "1.0"

FROZEN_GIT_COMMIT = "f8aa2a03b1dc5f4e8c5a09ab28530cee545e3c3b"
FROZEN_APPLICATION_VERSION = "0.4.0"
REPRODUCTION_APPLICATION_VERSION = "0.5.0"
FROZEN_SELECTION_SHA256 = (
    "d663a2ce3e0787fc10d2441fae9de44c3a93c6f20898d851df78178a519a0cff"
)
FROZEN_MANIFEST_SHA256 = (
    "73769c33454cc59b59c22a0091940eaa84e83167d7a1a910ceeabeeb9c1bc236"
)
FROZEN_DEVELOPMENT_SUMMARY_SHA256 = (
    "a60d20551c9aa5f079427ef0c34135df79cae52116b7ae2624307863b66fe9df"
)
FROZEN_SYMBOLS = ("SPY", "QQQ", "IWM", "XLF", "XLE", "XLV", "TLT", "GLD")
FROZEN_DATASET_SHA256 = {
    "SPY": "d1edbc93f2e89cff7827b96c2b7a820281a07cabac1dffa6443037011ae6a2ec",
    "QQQ": "c3a23aa983a3a0cd5080edf11bcf7e1ea1c1a927b72fdddcae5e6fe446371034",
    "IWM": "eef32e78bceaa966a5ed767d359d0db64223fe12a42b251b16e1a1eab985e804",
    "XLF": "61d7384e4374830786a716e414201e801c49605d8a02575dad34d896df258e1f",
    "XLE": "7e98ad15262bfc88c657f6a803d757b487a14aa2c9dd2e6763910bb5877b6b94",
    "XLV": "8d5917f63c07e23b628db87a691e894256a8ea735265d98c38ac5b7a17ced99d",
    "TLT": "66aa6d3ef6cff63fd6a8db3a05879e863d91797bb041fc6124f41db705d3cfd4",
    "GLD": "e3ddffa9c366ac8385883dca6c38d9805e1ab4b2360476a724b88020a70a5656",
}

_FROZEN_HOLDOUT_START = datetime(2019, 1, 2, tzinfo=UTC)
_FROZEN_HOLDOUT_END = datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC)
_FROZEN_DEVELOPMENT_END = datetime(2018, 12, 31, 23, 59, 59, tzinfo=UTC)
_ZERO = Decimal("0")
_MATERIAL_RETURN_DETERIORATION = Decimal("-0.10")

HOLDOUT_SUMMARY_COLUMNS = (
    "evaluation_id",
    "symbol",
    "strategy_name",
    "strategy_parameters",
    "holdout_start",
    "holdout_end",
    "warmup_observation_count",
    "warmup_first_timestamp",
    "warmup_last_timestamp",
    "first_financial_observation",
    "first_signal_date",
    "first_execution_date",
    "starting_equity",
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
    "total_simulated_cost",
    "time_in_market_ratio",
    "average_capital_utilisation_ratio",
    "maximum_capital_utilisation_ratio",
    "benchmark_return",
    "benchmark_maximum_percentage_drawdown",
    "return_difference_vs_benchmark",
    "drawdown_improvement_vs_benchmark",
    "cash_reconciliation_difference",
    "equity_reconciliation_difference",
    "integrity_status",
    "integrity_warnings",
    "dataset_sha256",
    "git_commit",
    "selection_record_sha256",
    "manifest_sha256",
    "configuration_sha256",
    "application_version",
    "run_id",
)

BENCHMARK_COMPARISON_COLUMNS = (
    "evaluation_id",
    "symbol",
    "strategy_return",
    "benchmark_return",
    "return_difference",
    "strategy_maximum_percentage_drawdown",
    "benchmark_maximum_percentage_drawdown",
    "drawdown_improvement",
)

DEVELOPMENT_HOLDOUT_COMPARISON_COLUMNS = (
    "symbol",
    "development_return",
    "holdout_return",
    "return_change",
    "development_sharpe",
    "holdout_sharpe",
    "sharpe_change",
    "development_maximum_drawdown",
    "holdout_maximum_drawdown",
    "development_benchmark_return",
    "holdout_benchmark_return",
    "development_trade_count",
    "holdout_trade_count",
    "development_exposure",
    "holdout_exposure",
    "return_remained_positive",
    "sharpe_remained_positive",
    "drawdown_advantage_persisted",
    "material_return_deterioration",
    "return_improved",
)


class GeneralisationClassification(StrEnum):
    """Pre-registered terminal classifications for the sealed study."""

    INVALID = "INVALID HOLDOUT EXECUTION"
    SUPPORTED = "GENERALISATION SUPPORTED"
    NOT_SUPPORTED = "GENERALISATION NOT SUPPORTED BY PRE-REGISTERED GATE"


@dataclass(frozen=True, slots=True)
class DevelopmentBaseline:
    """Frozen selected-candidate development metrics for one asset."""

    symbol: str
    net_return: Decimal
    periodic_sharpe_ratio: Decimal
    maximum_percentage_drawdown: Decimal
    benchmark_return: Decimal
    benchmark_maximum_percentage_drawdown: Decimal
    closed_trade_count: int
    time_in_market_ratio: Decimal


@dataclass(frozen=True, slots=True)
class PreparedHoldoutEntry:
    """Validated local inputs for one authorised asset."""

    entry: SplitEvaluationManifestEntry
    dataset: Path
    provenance: DatasetProvenance
    configuration: FrozenEvaluationConfiguration


@dataclass(frozen=True, slots=True)
class HoldoutEvaluationResult:
    """One complete sealed holdout result and its integrity evidence."""

    evaluation_id: str
    generated_at: datetime
    git_commit: str
    selection_record_sha256: str
    manifest_sha256: str
    configuration_sha256: str
    data_sha256: str
    provenance: DatasetProvenance
    quality_summary: DatasetQualitySummary
    split: EvaluationSplit
    warmup_bar_count: int
    warmup_first_timestamp: datetime
    warmup_last_timestamp: datetime
    first_signal_timestamp: datetime | None
    first_execution_timestamp: datetime | None
    report: StructuredBacktestReport
    integrity_warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        """Require exact identity, boundaries, and reconciliation."""

        if self.report.metadata.run_id != f"{self.evaluation_id}:sealed-holdout":
            raise ValueError("sealed holdout run ID must match the evaluation ID")
        if self.report.metadata.git_commit != self.git_commit:
            raise ValueError("holdout report Git revision must match the batch")
        if self.report.metadata.dataset.sha256 != self.data_sha256:
            raise ValueError("holdout dataset hash must match the report")
        if not self.report.backtest_result.reconciliation.is_reconciled:
            raise ValueError("holdout result must reconcile exactly")
        if self.warmup_bar_count != 200:
            raise ValueError("sealed SMA 50/200 holdout requires 200 warm-up bars")
        if self.warmup_last_timestamp >= self.split.holdout_start:
            raise ValueError("holdout warm-up must strictly precede the boundary")
        if self.report.metadata.dataset.actual_start < self.split.holdout_start:
            raise ValueError("holdout financial observations start before the boundary")
        if self.report.metadata.dataset.actual_end > self.split.holdout_end:
            raise ValueError("holdout financial observations end after the boundary")
        _validate_report_boundaries(
            self.report,
            self.split.holdout_start,
            self.split.holdout_end,
        )


@dataclass(frozen=True, slots=True)
class HoldoutAggregate:
    """Exact aggregate facts and the pre-registered gate outcome."""

    positive_return_count: int
    positive_sharpe_count: int
    available_sharpe_count: int
    drawdown_improvement_count: int
    median_strategy_return: Decimal
    mean_strategy_return: Decimal
    median_sharpe_ratio: Decimal | None
    mean_sharpe_ratio: Decimal | None
    median_maximum_drawdown: Decimal
    median_benchmark_return: Decimal
    median_benchmark_maximum_drawdown: Decimal
    total_closed_trades: int
    total_commission: Decimal
    total_slippage: Decimal
    mean_exposure: Decimal
    assets_below_three_closed_trades: tuple[str, ...]
    integrity_gate_passed: bool
    classification: GeneralisationClassification
    classification_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DevelopmentHoldoutComparisonSummary:
    """Cross-asset sign persistence and pre-declared return-change observations."""

    positive_return_persisted_assets: tuple[str, ...]
    positive_sharpe_persisted_assets: tuple[str, ...]
    drawdown_advantage_persisted_assets: tuple[str, ...]
    material_return_deterioration_assets: tuple[str, ...]
    return_improvement_assets: tuple[str, ...]
    material_deterioration_rule: str
    improvement_rule: str


@dataclass(frozen=True, slots=True)
class SealedHoldoutBatchResult:
    """All eight results plus aggregate and development comparison evidence."""

    source_manifest: str
    selection_record: str
    development_summary: str
    manifest_sha256: str
    selection_record_sha256: str
    development_summary_sha256: str
    git_commit: str
    application_version: str
    generated_at: datetime
    run_id: str
    results: tuple[HoldoutEvaluationResult, ...]
    aggregate: HoldoutAggregate
    development_comparison_rows: tuple[dict[str, object], ...]
    development_comparison_summary: DevelopmentHoldoutComparisonSummary


def run_holdout_evaluation(
    *,
    evaluation_id: str,
    selection_record_sha256: str,
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
) -> HoldoutEvaluationResult:
    """Run one authorised holdout account with carry-history used only for signals."""

    _validate_selected_configuration(configuration)
    selected_symbol = normalize_symbol(symbol)
    snapshot = tuple(bars)
    if not snapshot:
        raise ValueError("holdout evaluation requires market bars")
    if any(not isinstance(bar, MarketBar) for bar in snapshot):
        raise TypeError("holdout evaluation bars must be MarketBar values")
    if any(bar.symbol != selected_symbol for bar in snapshot):
        raise ValueError("holdout evaluation bars must match the requested symbol")
    if any(
        current.timestamp <= previous.timestamp
        for previous, current in zip(snapshot, snapshot[1:], strict=False)
    ):
        raise ValueError("holdout evaluation bars must be strictly chronological")
    if any(bar.timestamp > split.holdout_end for bar in snapshot):
        raise ValueError("holdout evaluation received a bar after its sealed boundary")

    strategy = create_strategy(
        configuration.strategy.name,
        configuration.strategy.parameters,
    )
    validate_strategy_data_policy(
        strategy,
        provenance,
        allow_unapproved_volume=configuration.allow_unapproved_volume,
    )
    holdout_bars = tuple(
        bar for bar in snapshot if split.holdout_start <= bar.timestamp <= split.holdout_end
    )
    preceding = tuple(bar for bar in snapshot if bar.timestamp < split.holdout_start)
    required = strategy.required_warmup_bars()
    if len(preceding) < required:
        raise ValueError(
            "insufficient pre-holdout carry-history: "
            f"requires {required} bars; found {len(preceding)}"
        )
    warmup_bars = preceding[-required:]
    period_signals = tuple(
        strategy.generate_signals_for_period(
            warmup_bars + holdout_bars,
            split.holdout_start,
        )
    )
    first_signal = (
        None if not period_signals else period_signals[0].timestamp
    )

    timestamp = datetime.now(UTC) if generated_at is None else generated_at
    request = BacktestRunRequest(
        data_path=Path(data_path),
        symbol=selected_symbol,
        strategy_configuration=configuration.strategy,
        backtest_configuration=configuration.backtest,
        start=split.holdout_start,
        end=split.holdout_end,
        risk_metric_settings=configuration.risk_metrics,
        include_buy_and_hold_benchmark=True,
    )
    execution = run_backtest_pipeline_with_bars(
        request,
        holdout_bars,
        warmup_bars=warmup_bars,
    )
    report = build_execution_report(
        execution,
        run_id=f"{evaluation_id}:sealed-holdout",
        generated_at=timestamp,
        git_commit=git_commit,
        base_directory=base_directory,
    )
    data_sha256 = report.metadata.dataset.sha256
    if data_sha256 is None:
        raise ValueError("holdout evaluation requires a dataset SHA-256")
    first_execution = (
        None
        if not report.backtest_result.trades
        else report.backtest_result.trades[0].timestamp
    )
    if first_execution is not None and first_execution <= split.holdout_start:
        raise ValueError("next-bar execution must occur after the holdout begins")
    if (
        first_signal is not None
        and first_execution is not None
        and first_execution <= first_signal
    ):
        raise ValueError("first execution must follow its eligible signal")
    _validate_final_liquidation(report, holdout_bars[-1])
    quality = calculate_dataset_quality(
        holdout_bars,
        suspicious_return_threshold=suspicious_return_threshold,
    )
    warnings = _holdout_integrity_warnings(report, quality)
    return HoldoutEvaluationResult(
        evaluation_id=evaluation_id,
        generated_at=timestamp,
        git_commit=git_commit,
        selection_record_sha256=selection_record_sha256,
        manifest_sha256=manifest_sha256,
        configuration_sha256=configuration_sha256(configuration),
        data_sha256=data_sha256,
        provenance=provenance,
        quality_summary=quality,
        split=split,
        warmup_bar_count=len(warmup_bars),
        warmup_first_timestamp=warmup_bars[0].timestamp,
        warmup_last_timestamp=warmup_bars[-1].timestamp,
        first_signal_timestamp=first_signal,
        first_execution_timestamp=first_execution,
        report=report,
        integrity_warnings=warnings,
    )


def run_sealed_holdout_batch(
    manifest: SplitEvaluationManifest,
    *,
    selection_record: Path,
    development_summary: Path,
    output_directory: Path,
    repository: Path | None = None,
    generated_at: datetime | None = None,
) -> SealedHoldoutBatchResult:
    """Execute the eight frozen holdouts once, fail fast, then export all results."""

    if not isinstance(manifest, SplitEvaluationManifest):
        raise TypeError("manifest must be SplitEvaluationManifest")
    repository_path = Path.cwd() if repository is None else Path(repository)
    prepared, selection_document = _prepare_frozen_plan(
        manifest,
        selection_record=Path(selection_record),
        repository=repository_path,
    )
    baselines = _load_development_baselines(
        Path(development_summary),
        selection_document,
    )
    output = Path(output_directory)
    _prepare_output_directory(output)

    git_commit = current_git_commit(repository_path)
    if git_commit is None:
        raise BacktestError("could not determine current Git revision for holdout")
    timestamp = datetime.now(UTC) if generated_at is None else generated_at
    run_id = f"holdout-{FROZEN_SELECTION_SHA256[:12]}-{git_commit[:12]}"
    results: list[HoldoutEvaluationResult] = []
    active_evaluation = "pre-execution"
    try:
        for item in prepared:
            active_evaluation = item.entry.evaluation_id
            definition = item.entry.definition
            if definition is None:
                raise AssertionError("prepared holdout entry is missing its definition")
            bars = tuple(
                _load_holdout_source_bars(
                    item.dataset,
                    definition.symbol,
                    definition.split,
                )
            )
            evaluation = run_holdout_evaluation(
                evaluation_id=item.entry.evaluation_id,
                selection_record_sha256=FROZEN_SELECTION_SHA256,
                manifest_sha256=FROZEN_MANIFEST_SHA256,
                data_path=item.dataset,
                symbol=definition.symbol,
                bars=bars,
                split=definition.split,
                configuration=item.configuration,
                provenance=item.provenance,
                git_commit=git_commit,
                suspicious_return_threshold=definition.suspicious_return_threshold,
                generated_at=timestamp,
                base_directory=repository_path,
            )
            results.append(evaluation)
    except (BacktestError, MarketDataError, StrategyError, OSError, TypeError, ValueError) as exc:
        _write_failure_diagnostic(
            output,
            evaluation_id=active_evaluation,
            error=exc,
            repository=repository_path,
        )
        raise BacktestError(
            "sealed holdout batch failed; no partial performance report was exported"
        ) from exc

    completed = tuple(results)
    if len(completed) != len(FROZEN_SYMBOLS):
        raise BacktestError("sealed holdout batch did not complete all eight assets")
    aggregate = _calculate_aggregate(completed)
    comparison_rows, comparison_summary = _development_holdout_comparison(
        completed,
        baselines,
    )
    batch_result = SealedHoldoutBatchResult(
        source_manifest=manifest.source_path.name,
        selection_record=Path(selection_record).name,
        development_summary=Path(development_summary).name,
        manifest_sha256=FROZEN_MANIFEST_SHA256,
        selection_record_sha256=FROZEN_SELECTION_SHA256,
        development_summary_sha256=FROZEN_DEVELOPMENT_SUMMARY_SHA256,
        git_commit=git_commit,
        application_version=installed_application_version(),
        generated_at=timestamp,
        run_id=run_id,
        results=completed,
        aggregate=aggregate,
        development_comparison_rows=comparison_rows,
        development_comparison_summary=comparison_summary,
    )
    _write_holdout_outputs(batch_result, output)
    return batch_result


def holdout_evaluation_to_json(
    result: HoldoutEvaluationResult,
    *,
    indent: int = 2,
) -> str:
    """Serialize one holdout result with boundary and integrity evidence."""

    payload = {
        "schema": HOLDOUT_RESULT_SCHEMA,
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "evaluation_id": result.evaluation_id,
        "period": "holdout",
        "generated_at": result.generated_at,
        "git_commit": result.git_commit,
        "selection_record_sha256": result.selection_record_sha256,
        "manifest_sha256": result.manifest_sha256,
        "configuration_sha256": result.configuration_sha256,
        "data_sha256": result.data_sha256,
        "provenance": result.provenance,
        "quality_summary": result.quality_summary,
        "holdout_boundary": {
            "start": result.split.holdout_start,
            "end": result.split.holdout_end,
            "inclusive": True,
            "warmup_policy": result.split.warmup_policy,
            "warmup_bar_count": result.warmup_bar_count,
            "warmup_first_timestamp": result.warmup_first_timestamp,
            "warmup_last_timestamp": result.warmup_last_timestamp,
        },
        "activity": {
            "first_financial_observation": (
                result.report.metadata.dataset.actual_start
            ),
            "first_signal_date": result.first_signal_timestamp,
            "first_execution_date": result.first_execution_timestamp,
        },
        "integrity": {
            "warmup_contributed_financial_activity": False,
            "future_observations_contributed": False,
            "live_network_request": False,
            "volume_signal_used": False,
            "reconciliation_status": (
                "PASS"
                if result.report.backtest_result.reconciliation.is_reconciled
                else "FAIL"
            ),
            "warnings": result.integrity_warnings,
        },
        "report": json.loads(structured_report_to_json(result.report)),
    }
    return json.dumps(
        _json_value(payload),
        ensure_ascii=False,
        indent=indent,
        allow_nan=False,
    )


def holdout_summary_to_csv(result: SealedHoldoutBatchResult) -> str:
    """Return one exact row per completed holdout evaluation."""

    return _csv_text(
        HOLDOUT_SUMMARY_COLUMNS,
        tuple(_holdout_summary_row(item) for item in result.results),
    )


def benchmark_comparison_to_csv(result: SealedHoldoutBatchResult) -> str:
    """Return exact holdout strategy-versus-benchmark rows."""

    return _csv_text(
        BENCHMARK_COMPARISON_COLUMNS,
        tuple(_benchmark_row(item) for item in result.results),
    )


def development_holdout_comparison_to_csv(
    result: SealedHoldoutBatchResult,
) -> str:
    """Return selected-candidate development-versus-holdout rows."""

    return _csv_text(
        DEVELOPMENT_HOLDOUT_COMPARISON_COLUMNS,
        result.development_comparison_rows,
    )


def _prepare_frozen_plan(
    manifest: SplitEvaluationManifest,
    *,
    selection_record: Path,
    repository: Path,
) -> tuple[tuple[PreparedHoldoutEntry, ...], Mapping[str, object]]:
    """Validate every frozen identity and prepare exactly eight local inputs."""

    current_commit = current_git_commit(repository)
    if current_commit != FROZEN_GIT_COMMIT:
        raise EvaluationManifestError("current Git commit is not the frozen v0.4.0 commit")
    if installed_application_version() not in {
        FROZEN_APPLICATION_VERSION,
        REPRODUCTION_APPLICATION_VERSION,
    }:
        raise EvaluationManifestError(
            "installed application version is not an authorised holdout version"
        )
    if sha256_file(selection_record) != FROZEN_SELECTION_SHA256:
        raise EvaluationManifestError("development selection record SHA-256 mismatch")
    if sha256_file(manifest.source_path) != FROZEN_MANIFEST_SHA256:
        raise EvaluationManifestError("approved ETF manifest SHA-256 mismatch")
    try:
        with selection_record.open("rb") as handle:
            selection: Mapping[str, object] = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise EvaluationManifestError(
            "could not load the frozen development selection record"
        ) from exc
    _validate_selection_document(selection)

    entries_by_id = {entry.evaluation_id: entry for entry in manifest.entries}
    prepared: list[PreparedHoldoutEntry] = []
    for symbol in FROZEN_SYMBOLS:
        evaluation_id = f"{symbol.lower()}-sma-50-200"
        entry = entries_by_id.get(evaluation_id)
        if entry is None or entry.configuration_error is not None:
            raise EvaluationManifestError(
                f"frozen manifest entry is unavailable: {evaluation_id}"
            )
        definition = entry.definition
        if definition is None:
            raise EvaluationManifestError(
                f"frozen manifest entry is invalid: {evaluation_id}"
            )
        configuration = definition.frozen_configuration()
        _validate_selected_definition(definition.symbol, definition.split, configuration)
        dataset = (
            definition.dataset
            if definition.dataset.is_absolute()
            else manifest.source_path.parent / definition.dataset
        )
        if not dataset.is_file():
            raise EvaluationManifestError(f"approved dataset is missing: {symbol}")
        if sha256_file(dataset) != FROZEN_DATASET_SHA256[symbol]:
            raise EvaluationManifestError(f"approved dataset SHA-256 mismatch: {symbol}")
        provenance = load_dataset_provenance(dataset, definition.provenance)
        _validate_provenance(provenance, symbol)
        strategy = create_strategy(
            configuration.strategy.name,
            configuration.strategy.parameters,
        )
        if "volume" in strategy.required_market_data_fields():
            raise EvaluationManifestError("sealed SMA strategy must not consume volume")
        prepared.append(
            PreparedHoldoutEntry(
                entry=entry,
                dataset=dataset,
                provenance=provenance,
                configuration=configuration,
            )
        )
    if len(prepared) != 8:
        raise EvaluationManifestError("sealed holdout plan must contain eight assets")
    return tuple(prepared), selection


def _validate_selection_document(selection: Mapping[str, object]) -> None:
    """Require the exact pre-holdout facts committed before unsealing."""

    expected = {
        "study": "approved-etf-study",
        "phase": "development",
        "development_end": "2018-12-31",
        "holdout_start": "2019-01-02",
        "holdout_end": "2025-12-31",
        "holdout_status": "sealed",
        "source_dataset_commit": (
            "718816c1e3471dba95086ea64c73e24a907d2b71"
        ),
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "selected_candidate_id": "sma-50-200",
        "selected_strategy": "sma-crossover",
        "fast_period": 50,
        "slow_period": 200,
        "candidate_specific_asset_selection": False,
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise EvaluationManifestError(f"selection record field mismatch: {key}")
    artifacts = selection.get("artifact_sha256")
    if not isinstance(artifacts, Mapping):
        raise EvaluationManifestError("selection record artifact hashes are missing")
    if artifacts.get("development_summary_csv") != FROZEN_DEVELOPMENT_SUMMARY_SHA256:
        raise EvaluationManifestError("frozen development summary SHA-256 mismatch")


def _validate_selected_definition(
    symbol: str,
    split: EvaluationSplit,
    configuration: FrozenEvaluationConfiguration,
) -> None:
    """Require one exact asset-independent SMA 50/200 configuration."""

    if normalize_symbol(symbol) not in FROZEN_SYMBOLS:
        raise EvaluationManifestError("holdout definition contains an unapproved symbol")
    if (
        split.development_end != _FROZEN_DEVELOPMENT_END
        or split.holdout_start != _FROZEN_HOLDOUT_START
        or split.holdout_end != _FROZEN_HOLDOUT_END
        or split.warmup_policy is not WarmupPolicy.CARRY_HISTORY
    ):
        raise EvaluationManifestError("holdout definition boundaries are not frozen")
    try:
        _validate_selected_configuration(configuration)
    except (TypeError, ValueError) as exc:
        raise EvaluationManifestError(str(exc)) from exc


def _validate_selected_configuration(
    configuration: FrozenEvaluationConfiguration,
) -> None:
    """Reject any strategy, cost, sizing, risk, or benchmark variation."""

    if not isinstance(configuration, FrozenEvaluationConfiguration):
        raise TypeError("configuration must be FrozenEvaluationConfiguration")
    strategy = configuration.strategy
    if strategy.name is not StrategyName.SMA_CROSSOVER or not isinstance(
        strategy.parameters,
        SmaCrossoverParameters,
    ):
        raise ValueError("sealed holdout permits only SMA crossover")
    if (
        strategy.parameters.fast_window != 50
        or strategy.parameters.slow_window != 200
    ):
        raise ValueError("sealed holdout permits only SMA 50/200")
    config = configuration.backtest
    if (
        config.initial_cash != Decimal("100000")
        or config.commission_bps != Decimal("1")
        or config.slippage_bps != Decimal("5")
        or config.position_sizing_mode is not PositionSizingMode.CASH_ALLOCATION
        or config.cash_allocation_ratio != Decimal("1.0")
        or config.maximum_position_value is not None
        or config.minimum_cash_reserve is not None
        or config.end_of_test_policy is not EndOfTestPolicy.LIQUIDATE
    ):
        raise ValueError("sealed holdout financial assumptions differ")
    metrics = configuration.risk_metrics
    if (
        metrics is None
        or metrics.periods_per_year != Decimal("252")
        or metrics.risk_free_rate_per_period != _ZERO
        or metrics.target_return_per_period != _ZERO
    ):
        raise ValueError("sealed holdout risk-metric settings differ")
    if configuration.benchmark is not BenchmarkSelection.BUY_AND_HOLD:
        raise ValueError("sealed holdout requires buy-and-hold benchmark")
    if configuration.allow_unapproved_volume:
        raise ValueError("sealed holdout cannot override the volume policy")


def _validate_provenance(provenance: DatasetProvenance, symbol: str) -> None:
    """Require the approved Yahoo price interpretation."""

    if (
        provenance.source_name != "Yahoo Finance via yfinance"
        or provenance.price_adjustment is not PriceAdjustmentPolicy.SPLIT_ADJUSTED
        or provenance.dividend_treatment is not DividendTreatment.EXCLUDED
        or provenance.split_treatment is not SplitTreatment.ADJUSTED
        or provenance.volume_approved_for_strategy_signals is not False
    ):
        raise EvaluationManifestError(
            f"approved dataset interpretation mismatch: {symbol}"
        )


def _load_holdout_source_bars(
    dataset: Path,
    symbol: str,
    split: EvaluationSplit,
) -> Sequence[MarketBar]:
    """Load local history only through the existing CSV provider."""

    from trading_research.data import CsvMarketDataProvider

    return CsvMarketDataProvider(dataset).get_historical_bars(
        symbol=symbol,
        start=None,
        end=split.holdout_end,
    )


def _load_development_baselines(
    path: Path,
    selection: Mapping[str, object],
) -> Mapping[str, DevelopmentBaseline]:
    """Load the frozen selected development rows after validating their file hash."""

    if sha256_file(path) != FROZEN_DEVELOPMENT_SUMMARY_SHA256:
        raise EvaluationManifestError("development summary SHA-256 mismatch")
    artifacts = selection.get("artifact_sha256")
    if (
        not isinstance(artifacts, Mapping)
        or artifacts.get("development_summary_csv")
        != FROZEN_DEVELOPMENT_SUMMARY_SHA256
    ):
        raise EvaluationManifestError("selection record does not identify the summary")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = tuple(csv.DictReader(handle))
    except OSError as exc:
        raise EvaluationManifestError("could not read frozen development summary") from exc
    selected = tuple(row for row in rows if row.get("candidate_id") == "sma-50-200")
    if len(selected) != 8:
        raise EvaluationManifestError("development summary must contain eight selected rows")
    baselines: dict[str, DevelopmentBaseline] = {}
    try:
        for row in selected:
            symbol = row["symbol"]
            if symbol in baselines:
                raise EvaluationManifestError(
                    f"duplicate development baseline symbol: {symbol}"
                )
            baselines[symbol] = DevelopmentBaseline(
                symbol=symbol,
                net_return=Decimal(row["net_return"]),
                periodic_sharpe_ratio=Decimal(row["periodic_sharpe_ratio"]),
                maximum_percentage_drawdown=Decimal(
                    row["maximum_percentage_drawdown"]
                ),
                benchmark_return=Decimal(row["benchmark_return"]),
                benchmark_maximum_percentage_drawdown=Decimal(
                    row["benchmark_maximum_percentage_drawdown"]
                ),
                closed_trade_count=int(row["closed_trade_count"]),
                time_in_market_ratio=Decimal(row["time_in_market_ratio"]),
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationManifestError(
            "frozen development summary contains an invalid selected row"
        ) from exc
    if tuple(baselines) != FROZEN_SYMBOLS:
        raise EvaluationManifestError("development summary asset order differs")
    return baselines


def _holdout_integrity_warnings(
    report: StructuredBacktestReport,
    quality: DatasetQualitySummary,
) -> tuple[str, ...]:
    """Return unresolved technical warnings without interpreting performance."""

    warnings: list[str] = []
    if quality.duplicate_timestamp_count or quality.non_increasing_timestamp_count:
        warnings.append("holdout observations are not strictly unique and chronological")
    if quality.suspicious_return_count:
        warnings.append("holdout data contains suspicious close returns")
    if quality.zero_volume_count:
        warnings.append("holdout data contains zero-volume observations")
    reconciliation = report.backtest_result.reconciliation
    if (
        not reconciliation.is_reconciled
        or reconciliation.cash_difference != _ZERO
        or reconciliation.reconciliation_difference != _ZERO
    ):
        warnings.append("account reconciliation failed")
    return tuple(warnings)


def _validate_report_boundaries(
    report: StructuredBacktestReport,
    start: datetime,
    end: datetime,
) -> None:
    """Reject warm-up or future timestamps in every financial component."""

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
    if any(timestamp < start or timestamp > end for timestamp in timestamps):
        raise ValueError("holdout report contains an out-of-bound financial timestamp")


def _validate_final_liquidation(
    report: StructuredBacktestReport,
    final_bar: MarketBar,
) -> None:
    """Require terminal valuation and any synthetic exit to use the final close."""

    result = report.backtest_result
    if result.equity_curve[-1].timestamp != final_bar.timestamp:
        raise ValueError("holdout valuation must use the final eligible bar")
    if result.final_position_quantity != 0:
        raise ValueError("liquidate policy must leave no final position")
    liquidations = tuple(
        trade
        for trade in result.trades
        if trade.execution_reason is ExecutionReason.END_OF_TEST_LIQUIDATION
    )
    if liquidations:
        final_trade = liquidations[-1]
        if (
            final_trade.timestamp != final_bar.timestamp
            or final_trade.reference_price != final_bar.close
        ):
            raise ValueError("end-of-test liquidation must use the final close")


def _calculate_aggregate(
    results: Sequence[HoldoutEvaluationResult],
) -> HoldoutAggregate:
    """Calculate the pre-registered integrity and generalisation gates."""

    returns = tuple(item.report.performance.total_return for item in results)
    optional_sharpes = tuple(_periodic_sharpe(item.report) for item in results)
    sharpes = tuple(value for value in optional_sharpes if value is not None)
    drawdowns = tuple(
        item.report.performance.maximum_percentage_drawdown for item in results
    )
    benchmark_returns = tuple(
        _require_benchmark(item.report).benchmark_total_return for item in results
    )
    benchmark_drawdowns = tuple(
        _require_benchmark(item.report).benchmark_maximum_percentage_drawdown
        for item in results
    )
    positive_returns = sum(value > _ZERO for value in returns)
    positive_sharpes = sum(
        value is not None and value > _ZERO for value in optional_sharpes
    )
    drawdown_improvements = sum(
        _require_benchmark(item.report).maximum_drawdown_improvement > _ZERO
        for item in results
    )
    low_trade_assets = tuple(
        item.report.metadata.dataset.symbol
        for item in results
        if item.report.trade_statistics.completed_trade_count < 3
    )
    integrity_passed = (
        len(results) == 8
        and all(not item.integrity_warnings for item in results)
        and all(
            item.report.backtest_result.reconciliation.is_reconciled
            and item.report.backtest_result.reconciliation.cash_difference == _ZERO
            and (
                item.report.backtest_result.reconciliation.reconciliation_difference
                == _ZERO
            )
            for item in results
        )
    )
    classification, reasons = _classify_generalisation(
        integrity_gate_passed=integrity_passed,
        positive_return_count=positive_returns,
        positive_sharpe_count=positive_sharpes,
        drawdown_improvement_count=drawdown_improvements,
    )
    return HoldoutAggregate(
        positive_return_count=positive_returns,
        positive_sharpe_count=positive_sharpes,
        available_sharpe_count=len(sharpes),
        drawdown_improvement_count=drawdown_improvements,
        median_strategy_return=_median(returns),
        mean_strategy_return=_mean(returns),
        median_sharpe_ratio=None if not sharpes else _median(sharpes),
        mean_sharpe_ratio=None if not sharpes else _mean(sharpes),
        median_maximum_drawdown=_median(drawdowns),
        median_benchmark_return=_median(benchmark_returns),
        median_benchmark_maximum_drawdown=_median(benchmark_drawdowns),
        total_closed_trades=sum(
            item.report.trade_statistics.completed_trade_count for item in results
        ),
        total_commission=exact_decimal_sum(
            item.report.performance.total_commission for item in results
        ),
        total_slippage=exact_decimal_sum(
            item.report.performance.total_adverse_slippage_cost for item in results
        ),
        mean_exposure=_mean(
            tuple(
                _require_exposure(item.report)
                for item in results
            )
        ),
        assets_below_three_closed_trades=low_trade_assets,
        integrity_gate_passed=integrity_passed,
        classification=classification,
        classification_reasons=tuple(reasons),
    )


def _classify_generalisation(
    *,
    integrity_gate_passed: bool,
    positive_return_count: int,
    positive_sharpe_count: int,
    drawdown_improvement_count: int,
) -> tuple[GeneralisationClassification, tuple[str, ...]]:
    """Apply only the three pre-registered gates after technical validity."""

    if not integrity_gate_passed:
        return (
            GeneralisationClassification.INVALID,
            ("one or more pre-registered integrity requirements failed",),
        )
    reasons: list[str] = []
    if positive_return_count < 6:
        reasons.append(
            "positive net holdout return occurred in "
            f"{positive_return_count} of 8 assets"
        )
    if positive_sharpe_count < 6:
        reasons.append(
            f"positive holdout Sharpe occurred in {positive_sharpe_count} of 8 assets"
        )
    if drawdown_improvement_count < 5:
        reasons.append(
            "smaller drawdown than buy-and-hold occurred in "
            f"{drawdown_improvement_count} of 8 assets"
        )
    if reasons:
        return GeneralisationClassification.NOT_SUPPORTED, tuple(reasons)
    return (
        GeneralisationClassification.SUPPORTED,
        ("all three pre-registered generalisation conditions passed",),
    )


def _development_holdout_comparison(
    results: Sequence[HoldoutEvaluationResult],
    baselines: Mapping[str, DevelopmentBaseline],
) -> tuple[
    tuple[dict[str, object], ...],
    DevelopmentHoldoutComparisonSummary,
]:
    """Compare the frozen development rows with the one-shot holdout results."""

    rows: list[dict[str, object]] = []
    positive_return_persisted: list[str] = []
    positive_sharpe_persisted: list[str] = []
    drawdown_advantage_persisted: list[str] = []
    material_deterioration: list[str] = []
    improvement: list[str] = []
    for item in results:
        report = item.report
        symbol = report.metadata.dataset.symbol
        development = baselines[symbol]
        holdout_return = report.performance.total_return
        holdout_sharpe = _periodic_sharpe(report)
        holdout_drawdown = report.performance.maximum_percentage_drawdown
        comparison = _require_benchmark(report)
        return_change = holdout_return - development.net_return
        sharpe_change = (
            None
            if holdout_sharpe is None
            else holdout_sharpe - development.periodic_sharpe_ratio
        )
        return_persisted = development.net_return > _ZERO and holdout_return > _ZERO
        sharpe_persisted = (
            development.periodic_sharpe_ratio > _ZERO
            and holdout_sharpe is not None
            and holdout_sharpe > _ZERO
        )
        development_drawdown_improvement = (
            development.maximum_percentage_drawdown
            - development.benchmark_maximum_percentage_drawdown
        )
        drawdown_persisted = (
            development_drawdown_improvement > _ZERO
            and comparison.maximum_drawdown_improvement > _ZERO
        )
        materially_worse = return_change <= _MATERIAL_RETURN_DETERIORATION
        improved = return_change > _ZERO
        if return_persisted:
            positive_return_persisted.append(symbol)
        if sharpe_persisted:
            positive_sharpe_persisted.append(symbol)
        if drawdown_persisted:
            drawdown_advantage_persisted.append(symbol)
        if materially_worse:
            material_deterioration.append(symbol)
        if improved:
            improvement.append(symbol)
        rows.append(
            {
                "symbol": symbol,
                "development_return": development.net_return,
                "holdout_return": holdout_return,
                "return_change": return_change,
                "development_sharpe": development.periodic_sharpe_ratio,
                "holdout_sharpe": holdout_sharpe,
                "sharpe_change": sharpe_change,
                "development_maximum_drawdown": (
                    development.maximum_percentage_drawdown
                ),
                "holdout_maximum_drawdown": holdout_drawdown,
                "development_benchmark_return": development.benchmark_return,
                "holdout_benchmark_return": comparison.benchmark_total_return,
                "development_trade_count": development.closed_trade_count,
                "holdout_trade_count": report.trade_statistics.completed_trade_count,
                "development_exposure": development.time_in_market_ratio,
                "holdout_exposure": (
                    report.exposure_statistics.time_in_market_ratio
                ),
                "return_remained_positive": return_persisted,
                "sharpe_remained_positive": sharpe_persisted,
                "drawdown_advantage_persisted": drawdown_persisted,
                "material_return_deterioration": materially_worse,
                "return_improved": improved,
            }
        )
    summary = DevelopmentHoldoutComparisonSummary(
        positive_return_persisted_assets=tuple(positive_return_persisted),
        positive_sharpe_persisted_assets=tuple(positive_sharpe_persisted),
        drawdown_advantage_persisted_assets=tuple(drawdown_advantage_persisted),
        material_return_deterioration_assets=tuple(material_deterioration),
        return_improvement_assets=tuple(improvement),
        material_deterioration_rule=(
            "holdout return minus development return is at most -0.10"
        ),
        improvement_rule="holdout return is greater than development return",
    )
    return tuple(rows), summary


def _holdout_summary_row(result: HoldoutEvaluationResult) -> dict[str, object]:
    """Return all required exact metrics for one asset."""

    report = result.report
    performance = report.performance
    trades = report.trade_statistics
    risk = report.risk_adjusted_metrics
    exposure = report.exposure_statistics
    comparison = _require_benchmark(report)
    reconciliation = report.backtest_result.reconciliation
    total_cost = exact_decimal_sum(
        (
            performance.total_commission,
            performance.total_adverse_slippage_cost,
        )
    )
    return {
        "evaluation_id": result.evaluation_id,
        "symbol": report.metadata.dataset.symbol,
        "strategy_name": report.metadata.strategy.name.value,
        "strategy_parameters": '{"fast_window":50,"slow_window":200}',
        "holdout_start": result.split.holdout_start,
        "holdout_end": result.split.holdout_end,
        "warmup_observation_count": result.warmup_bar_count,
        "warmup_first_timestamp": result.warmup_first_timestamp,
        "warmup_last_timestamp": result.warmup_last_timestamp,
        "first_financial_observation": report.metadata.dataset.actual_start,
        "first_signal_date": result.first_signal_timestamp,
        "first_execution_date": result.first_execution_timestamp,
        "starting_equity": performance.starting_cash,
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
        "total_simulated_cost": total_cost,
        "time_in_market_ratio": exposure.time_in_market_ratio,
        "average_capital_utilisation_ratio": (
            exposure.average_capital_utilisation_ratio
        ),
        "maximum_capital_utilisation_ratio": (
            exposure.maximum_capital_utilisation_ratio
        ),
        "benchmark_return": comparison.benchmark_total_return,
        "benchmark_maximum_percentage_drawdown": (
            comparison.benchmark_maximum_percentage_drawdown
        ),
        "return_difference_vs_benchmark": comparison.excess_return,
        "drawdown_improvement_vs_benchmark": (
            comparison.maximum_drawdown_improvement
        ),
        "cash_reconciliation_difference": reconciliation.cash_difference,
        "equity_reconciliation_difference": (
            reconciliation.reconciliation_difference
        ),
        "integrity_status": (
            "PASS" if not result.integrity_warnings else "FAIL"
        ),
        "integrity_warnings": "; ".join(result.integrity_warnings),
        "dataset_sha256": result.data_sha256,
        "git_commit": result.git_commit,
        "selection_record_sha256": result.selection_record_sha256,
        "manifest_sha256": result.manifest_sha256,
        "configuration_sha256": result.configuration_sha256,
        "application_version": report.metadata.application_version,
        "run_id": report.metadata.run_id,
    }


def _benchmark_row(result: HoldoutEvaluationResult) -> dict[str, object]:
    """Return one concise benchmark comparison row."""

    report = result.report
    comparison = _require_benchmark(report)
    return {
        "evaluation_id": result.evaluation_id,
        "symbol": report.metadata.dataset.symbol,
        "strategy_return": report.performance.total_return,
        "benchmark_return": comparison.benchmark_total_return,
        "return_difference": comparison.excess_return,
        "strategy_maximum_percentage_drawdown": (
            report.performance.maximum_percentage_drawdown
        ),
        "benchmark_maximum_percentage_drawdown": (
            comparison.benchmark_maximum_percentage_drawdown
        ),
        "drawdown_improvement": comparison.maximum_drawdown_improvement,
    }


def _write_holdout_outputs(
    result: SealedHoldoutBatchResult,
    output: Path,
) -> None:
    """Write individual and aggregate files only after all eight runs finish."""

    individual_outputs = tuple(
        (
            output / item.evaluation_id / "holdout-result.json",
            holdout_evaluation_to_json(item) + "\n",
        )
        for item in result.results
    )
    summary_rows = tuple(_holdout_summary_row(item) for item in result.results)
    summary_payload = {
        "schema": HOLDOUT_BATCH_SCHEMA,
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "run_id": result.run_id,
        "generated_at": result.generated_at,
        "application_version": result.application_version,
        "git_commit": result.git_commit,
        "source_manifest": result.source_manifest,
        "manifest_sha256": result.manifest_sha256,
        "selection_record": result.selection_record,
        "selection_record_sha256": result.selection_record_sha256,
        "development_summary": result.development_summary,
        "development_summary_sha256": result.development_summary_sha256,
        "evaluation_count": len(result.results),
        "results": summary_rows,
        "aggregate": result.aggregate,
        "development_comparison_summary": result.development_comparison_summary,
    }
    assessment_payload = {
        "schema": "trading-research.generalisation-assessment",
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "integrity_gate": {
            "passed": result.aggregate.integrity_gate_passed,
            "requirements": (
                "eight completed evaluations",
                "eight exact reconciliations",
                "no dataset-integrity warnings",
                "no observation after 2025-12-31",
                "no warm-up financial activity",
                "no volume signal",
                "no live data request",
                "no configuration variation",
            ),
        },
        "generalisation_gate": {
            "minimum_positive_return_assets": 6,
            "actual_positive_return_assets": (
                result.aggregate.positive_return_count
            ),
            "minimum_positive_sharpe_assets": 6,
            "actual_positive_sharpe_assets": (
                result.aggregate.positive_sharpe_count
            ),
            "minimum_drawdown_improvement_assets": 5,
            "actual_drawdown_improvement_assets": (
                result.aggregate.drawdown_improvement_count
            ),
        },
        "classification": result.aggregate.classification,
        "classification_reasons": result.aggregate.classification_reasons,
        "assets_below_three_closed_trades": (
            result.aggregate.assets_below_three_closed_trades
        ),
    }
    run_metadata = {
        "schema": "trading-research.sealed-holdout-run-integrity",
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "run_id": result.run_id,
        "generated_at": result.generated_at,
        "application_version": result.application_version,
        "git_commit": result.git_commit,
        "source_manifest": result.source_manifest,
        "manifest_sha256": result.manifest_sha256,
        "selection_record": result.selection_record,
        "selection_record_sha256": result.selection_record_sha256,
        "development_summary": result.development_summary,
        "development_summary_sha256": result.development_summary_sha256,
        "period": {
            "name": "holdout",
            "start": _FROZEN_HOLDOUT_START,
            "end": _FROZEN_HOLDOUT_END,
            "inclusive": True,
        },
        "evaluation_count": len(result.results),
        "strategy": "sma-crossover",
        "parameters": {"fast_window": 50, "slow_window": 200},
        "assets": FROZEN_SYMBOLS,
        "strategy_network_requests": False,
        "volume_used_by_strategy": False,
        "warmup_contributed_financial_activity": False,
        "future_observations_contributed": False,
        "alternative_candidates_evaluated": False,
        "post_holdout_tuning_performed": False,
        "integrity_gate_passed": result.aggregate.integrity_gate_passed,
        "datasets": tuple(
            {
                "symbol": item.report.metadata.dataset.symbol,
                "sha256": item.data_sha256,
                "identifier": item.report.metadata.dataset.identifier,
            }
            for item in result.results
        ),
    }
    outputs = {
        "holdout-summary.csv": holdout_summary_to_csv(result),
        "holdout-summary.json": _json_text(summary_payload),
        "holdout-report.txt": _format_holdout_report(result),
        "benchmark-comparison.csv": benchmark_comparison_to_csv(result),
        "development-holdout-comparison.csv": (
            development_holdout_comparison_to_csv(result)
        ),
        "generalisation-assessment.json": _json_text(assessment_payload),
        "run-metadata.json": _json_text(run_metadata),
    }
    for name, content in outputs.items():
        write_text_export(
            content,
            output / name,
            overwrite=False,
            create_parents=True,
        )
    for path, content in individual_outputs:
        write_text_export(
            content,
            path,
            overwrite=False,
            create_parents=True,
        )


def _format_holdout_report(result: SealedHoldoutBatchResult) -> str:
    """Return the required one-shot research interpretation."""

    aggregate = result.aggregate
    lines = [
        "APPROVED ETF STUDY — FIRST SEALED HOLDOUT EVALUATION",
        "====================================================",
        f"Run ID: {result.run_id}",
        f"Application version: {result.application_version}",
        f"Git commit: {result.git_commit}",
        f"Selection-record SHA-256: {result.selection_record_sha256}",
        f"Manifest SHA-256: {result.manifest_sha256}",
        "Holdout: 2019-01-02 through 2025-12-31 inclusive",
        "Strategy: universal SMA crossover 50/200",
        "Completed evaluations: 8 of 8",
        "",
        f"Integrity gate: {'PASS' if aggregate.integrity_gate_passed else 'FAIL'}",
            f"Classification: {aggregate.classification.value}",
    ]
    lines.extend(f"Reason: {reason}" for reason in aggregate.classification_reasons)
    lines.extend(
        (
            "",
            "PRE-REGISTERED GATE COUNTS",
            "--------------------------",
            f"Positive net returns: {aggregate.positive_return_count} of 8",
            f"Positive periodic Sharpe ratios: {aggregate.positive_sharpe_count} of 8",
            (
                "Smaller drawdown than buy-and-hold: "
                f"{aggregate.drawdown_improvement_count} of 8"
            ),
            f"Median strategy return: {aggregate.median_strategy_return}",
            f"Mean strategy return: {aggregate.mean_strategy_return}",
            (
                "Median periodic Sharpe ratio: "
                f"{_optional_text(aggregate.median_sharpe_ratio)}"
            ),
            (
                "Mean periodic Sharpe ratio: "
                f"{_optional_text(aggregate.mean_sharpe_ratio)}"
            ),
            f"Median strategy maximum drawdown: {aggregate.median_maximum_drawdown}",
            f"Median benchmark return: {aggregate.median_benchmark_return}",
            (
                "Median benchmark maximum drawdown: "
                f"{aggregate.median_benchmark_maximum_drawdown}"
            ),
            f"Total closed trades: {aggregate.total_closed_trades}",
            f"Total commission: {aggregate.total_commission}",
            f"Total slippage attribution: {aggregate.total_slippage}",
            f"Mean exposure: {aggregate.mean_exposure}",
            "",
            "DEVELOPMENT-VERSUS-HOLDOUT",
            "--------------------------",
            (
                "Positive return persisted: "
                + _asset_list(
                    result.development_comparison_summary
                    .positive_return_persisted_assets
                )
            ),
            (
                "Positive Sharpe persisted: "
                + _asset_list(
                    result.development_comparison_summary
                    .positive_sharpe_persisted_assets
                )
            ),
            (
                "Drawdown advantage persisted: "
                + _asset_list(
                    result.development_comparison_summary
                    .drawdown_advantage_persisted_assets
                )
            ),
            (
                "Material return deterioration: "
                + _asset_list(
                    result.development_comparison_summary
                    .material_return_deterioration_assets
                )
            ),
            (
                "Return improvement: "
                + _asset_list(
                    result.development_comparison_summary.return_improvement_assets
                )
            ),
            "",
            "PER-ASSET RESULTS",
            "-----------------",
        )
    )
    for item in result.results:
        row = _holdout_summary_row(item)
        lines.append(
            f"{row['symbol']}: return={row['net_return']}; "
            f"Sharpe={_optional_text(row['periodic_sharpe_ratio'])}; "
            f"drawdown={row['maximum_percentage_drawdown']}; "
            f"benchmark return={row['benchmark_return']}; "
            f"closed trades={row['closed_trade_count']}; "
            f"cost={row['total_simulated_cost']}; "
            f"exposure={row['time_in_market_ratio']}; integrity=PASS"
        )
    low_trade_text = (
        "none"
        if not aggregate.assets_below_three_closed_trades
        else ", ".join(aggregate.assets_below_three_closed_trades)
    )
    lines.extend(
        (
            "",
            "INTERPRETATION LIMITS",
            "---------------------",
            "This was the first sealed holdout evaluation of the selected candidate.",
            "SMA 50/200 was selected before viewing holdout performance.",
            "Only eight predetermined evaluations were run.",
            "No alternative strategy or parameter was tested on the holdout.",
            "No post-holdout parameter tuning was performed.",
            f"Assets with fewer than three closed trades: {low_trade_text}.",
            "This result is a research finding, not proof of future profitability.",
            (
                "Prices exclude dividends, so benchmark and strategy returns are "
                "price-return comparisons rather than total returns."
            ),
            "Simulated costs do not guarantee live execution costs.",
            (
                "Yahoo data is approved only for personal price-based historical "
                "research."
            ),
            "Volume-dependent conclusions are not authorised.",
            (
                "The eight ETF results are not eight fully independent statistical "
                "experiments."
            ),
            "",
        )
    )
    return "\n".join(lines)


def _prepare_output_directory(path: Path) -> None:
    """Refuse every existing output path for the one-shot execution."""

    if path.exists():
        raise ReportWriteError(f"sealed holdout output already exists: {path}")
    try:
        path.mkdir(parents=True)
    except OSError as exc:
        raise ReportWriteError(
            f"could not create sealed holdout output directory: {path.name}"
        ) from exc


def _write_failure_diagnostic(
    output: Path,
    *,
    evaluation_id: str,
    error: BaseException,
    repository: Path,
) -> None:
    """Persist only non-performance failure information for an aborted batch."""

    message = str(error)
    for candidate in (str(repository), str(repository.resolve(strict=False))):
        message = message.replace(candidate, "<repository>")
    payload = {
        "schema": "trading-research.sealed-holdout-failure",
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "status": "aborted",
        "evaluation_id": evaluation_id,
        "error_type": type(error).__name__,
        "message": message,
        "partial_performance_exported": False,
    }
    write_text_export(
        _json_text(payload),
        output / "failure-diagnostic.json",
        overwrite=False,
        create_parents=True,
    )


def _periodic_sharpe(report: StructuredBacktestReport) -> Decimal | None:
    """Return the configured periodic Sharpe ratio when defined."""

    if report.risk_adjusted_metrics is None:
        raise ValueError("sealed holdout requires risk-adjusted metrics")
    return report.risk_adjusted_metrics.periodic_sharpe_ratio


def _require_exposure(report: StructuredBacktestReport) -> Decimal:
    """Return time in market for the non-empty holdout period."""

    exposure = report.exposure_statistics.time_in_market_ratio
    if exposure is None:
        raise ValueError("sealed holdout requires exposure observations")
    return exposure


def _require_benchmark(report: StructuredBacktestReport) -> BenchmarkComparison:
    """Return the required benchmark comparison."""

    comparison = report.benchmark_comparison
    if comparison is None:
        raise ValueError("sealed holdout requires a benchmark comparison")
    return comparison


def _mean(values: Sequence[Decimal]) -> Decimal:
    """Return a Decimal-only arithmetic mean."""

    if not values:
        raise ValueError("mean requires at least one value")
    return exact_decimal_sum(values) / Decimal(len(values))


def _median(values: Sequence[Decimal]) -> Decimal:
    """Return a Decimal-only median without converting through float."""

    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return exact_decimal_sum((ordered[middle - 1], ordered[middle])) / Decimal("2")


def _csv_text(
    columns: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> str:
    """Serialize stable exact CSV and leave unavailable values blank."""

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {column: _csv_value(row.get(column)) for column in columns}
        )
    return buffer.getvalue()


def _csv_value(value: object) -> object:
    """Return a machine-readable exact CSV scalar."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    return value


def _json_text(value: object) -> str:
    """Return deterministic indented JSON with exact Decimal strings."""

    return (
        json.dumps(
            _json_value(value),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _json_value(value: object) -> object:
    """Recursively serialize exact models without binary floating point."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(
            Decimal(value.days) * Decimal("86400")
            + Decimal(value.seconds)
            + Decimal(value.microseconds) / Decimal("1000000")
        )
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
    raise TypeError(f"unsupported holdout export value: {type(value).__name__}")


def _optional_text(value: object) -> str:
    """Distinguish unavailable values from numeric zero."""

    return "not available" if value is None else str(value)


def _asset_list(values: Sequence[str]) -> str:
    """Return a readable stable asset list."""

    return "none" if not values else ", ".join(values)
