"""Sequential predefined experiment execution and aggregate result exports."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from enum import StrEnum
from io import StringIO
from pathlib import Path

from trading_research.backtesting.pipeline import (
    BacktestRunRequest,
    build_execution_report,
    run_backtest_pipeline,
)
from trading_research.errors import (
    BacktestError,
    MarketDataError,
    ReportWriteError,
    StrategyError,
)
from trading_research.experiments.loader import (
    EXPERIMENT_MANIFEST_SCHEMA_VERSION,
    ExperimentManifest,
    ExperimentManifestEntry,
)
from trading_research.experiments.models import BenchmarkSelection
from trading_research.reporting import (
    SUMMARY_CSV_COLUMNS,
    StructuredBacktestReport,
    current_git_commit,
    summary_report_to_csv,
    write_json_report,
    write_text_export,
)

BATCH_RESULT_SCHEMA_VERSION = "1.0"
AGGREGATE_CSV_COLUMNS = (
    "experiment_id",
    "status",
    "failure_kind",
    "failure_message",
    "report_file",
    *SUMMARY_CSV_COLUMNS,
)


class ExperimentStatus(StrEnum):
    """Stable completion states for one declared experiment."""

    SUCCESS = "success"
    FAILURE = "failure"


class ExperimentFailureKind(StrEnum):
    """Stable expected failure categories retained in batch outputs."""

    CONFIGURATION = "configuration_error"
    MARKET_DATA = "market_data_error"
    BACKTEST = "backtest_error"


@dataclass(frozen=True, slots=True)
class ExperimentRunRecord:
    """One ordered success or explicit failure from a batch execution."""

    experiment_id: str
    status: ExperimentStatus
    failure_kind: ExperimentFailureKind | None
    failure_message: str | None
    report_file: str | None
    report: StructuredBacktestReport | None

    def __post_init__(self) -> None:
        """Require complete mutually exclusive success and failure records."""

        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")
        if self.status is ExperimentStatus.SUCCESS:
            if (
                self.failure_kind is not None
                or self.failure_message is not None
                or self.report_file is None
                or self.report is None
            ):
                raise ValueError("successful records require only report information")
        elif (
            self.failure_kind is None
            or not self.failure_message
            or self.report_file is not None
            or self.report is not None
        ):
            raise ValueError("failed records require only failure information")


@dataclass(frozen=True, slots=True)
class ExperimentBatchResult:
    """Immutable ordered outcome of one sequential predefined batch."""

    manifest_file: str
    git_commit: str
    records: tuple[ExperimentRunRecord, ...]

    def __post_init__(self) -> None:
        """Require one non-empty result record per manifest entry."""

        if not self.manifest_file:
            raise ValueError("manifest_file must not be empty")
        if not self.git_commit:
            raise ValueError("git_commit must not be empty")
        if not self.records:
            raise ValueError("records must not be empty")

    @property
    def success_count(self) -> int:
        """Return the number of successful experiments."""

        return sum(record.status is ExperimentStatus.SUCCESS for record in self.records)

    @property
    def failure_count(self) -> int:
        """Return the number of explicitly recorded failures."""

        return len(self.records) - self.success_count

    @property
    def succeeded(self) -> bool:
        """Return whether every declared experiment completed successfully."""

        return self.failure_count == 0


def run_experiment_batch(
    manifest: ExperimentManifest,
    output_directory: Path,
    *,
    overwrite: bool = False,
    repository: Path | None = None,
) -> ExperimentBatchResult:
    """Run every predefined experiment sequentially and write all batch outputs."""

    if not isinstance(manifest, ExperimentManifest):
        raise TypeError("manifest must be ExperimentManifest")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")
    output = Path(output_directory)
    _prepare_output_directory(output, overwrite=overwrite)
    repository_path = Path.cwd() if repository is None else Path(repository)
    git_commit = current_git_commit(repository_path)
    if git_commit is None:
        raise BacktestError("could not determine the current Git revision for the batch")

    records = tuple(
        _run_manifest_entry(
            entry,
            manifest=manifest,
            git_commit=git_commit,
            repository=repository_path,
        )
        for entry in manifest.entries
    )
    result = ExperimentBatchResult(
        manifest_file=manifest.source_path.name,
        git_commit=git_commit,
        records=records,
    )
    _write_batch_outputs(result, output, overwrite=overwrite)
    return result


def aggregate_result_to_csv(result: ExperimentBatchResult) -> str:
    """Return one ordered row per experiment using existing summary serialization."""

    if not isinstance(result, ExperimentBatchResult):
        raise TypeError("result must be ExperimentBatchResult")
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=AGGREGATE_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for record in result.records:
        row = {column: "" for column in AGGREGATE_CSV_COLUMNS}
        row.update(
            {
                "experiment_id": record.experiment_id,
                "status": record.status.value,
                "failure_kind": (
                    "" if record.failure_kind is None else record.failure_kind.value
                ),
                "failure_message": record.failure_message or "",
                "report_file": record.report_file or "",
            }
        )
        if record.report is not None:
            summary_row = next(
                csv.DictReader(StringIO(summary_report_to_csv(record.report)))
            )
            row.update(summary_row)
        writer.writerow(row)
    return buffer.getvalue()


def batch_result_to_json(result: ExperimentBatchResult, *, indent: int = 2) -> str:
    """Return a machine-readable ordered batch manifest without Decimal floats."""

    if not isinstance(result, ExperimentBatchResult):
        raise TypeError("result must be ExperimentBatchResult")
    if isinstance(indent, bool) or not isinstance(indent, int):
        raise TypeError("indent must be an integer")
    if indent < 0:
        raise ValueError("indent must be non-negative")
    payload = {
        "schema_version": BATCH_RESULT_SCHEMA_VERSION,
        "experiment_manifest_schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "source_manifest": result.manifest_file,
        "git_commit": result.git_commit,
        "experiment_count": len(result.records),
        "success_count": result.success_count,
        "failure_count": result.failure_count,
        "aggregate_csv": "aggregate.csv",
        "results": [_batch_record_payload(record) for record in result.records],
    }
    return json.dumps(payload, ensure_ascii=False, indent=indent, allow_nan=False)


def _run_manifest_entry(
    entry: ExperimentManifestEntry,
    *,
    manifest: ExperimentManifest,
    git_commit: str,
    repository: Path,
) -> ExperimentRunRecord:
    """Run one entry or retain its expected failure without affecting later entries."""

    if entry.configuration_error is not None:
        return _failure_record(
            entry.experiment_id,
            ExperimentFailureKind.CONFIGURATION,
            entry.configuration_error,
        )
    definition = entry.definition
    if definition is None:
        raise AssertionError("validated manifest entry is missing its definition")
    try:
        dataset = _resolve_dataset_path(manifest.source_path, definition.dataset)
        request = BacktestRunRequest(
            data_path=dataset,
            symbol=definition.symbol,
            strategy_configuration=definition.strategy_configuration(),
            backtest_configuration=definition.backtest_configuration(),
            start=definition.start,
            end=definition.end,
            risk_metric_settings=definition.risk_metric_settings(),
            include_buy_and_hold_benchmark=(
                definition.benchmark is BenchmarkSelection.BUY_AND_HOLD
            ),
        )
    except (TypeError, ValueError) as exc:
        return _failure_record(
            entry.experiment_id,
            ExperimentFailureKind.CONFIGURATION,
            str(exc),
        )
    try:
        execution = run_backtest_pipeline(request)
        report = build_execution_report(
            execution,
            run_id=entry.experiment_id,
            git_commit=git_commit,
            base_directory=repository,
        )
    except (MarketDataError, OSError) as exc:
        return _failure_record(
            entry.experiment_id,
            ExperimentFailureKind.MARKET_DATA,
            str(exc),
        )
    except (BacktestError, StrategyError) as exc:
        return _failure_record(
            entry.experiment_id,
            ExperimentFailureKind.BACKTEST,
            str(exc),
        )
    return ExperimentRunRecord(
        experiment_id=entry.experiment_id,
        status=ExperimentStatus.SUCCESS,
        failure_kind=None,
        failure_message=None,
        report_file=f"reports/{entry.experiment_id}.json",
        report=report,
    )


def _failure_record(
    experiment_id: str,
    kind: ExperimentFailureKind,
    message: str,
) -> ExperimentRunRecord:
    """Create one explicit expected failure record."""

    return ExperimentRunRecord(
        experiment_id=experiment_id,
        status=ExperimentStatus.FAILURE,
        failure_kind=kind,
        failure_message=message,
        report_file=None,
        report=None,
    )


def _resolve_dataset_path(manifest_path: Path, dataset: Path) -> Path:
    """Resolve portable relative datasets against the manifest directory."""

    return dataset if dataset.is_absolute() else manifest_path.parent / dataset


def _prepare_output_directory(path: Path, *, overwrite: bool) -> None:
    """Protect the complete output directory unless replacement is explicit."""

    if path.exists():
        if not path.is_dir():
            raise ReportWriteError(f"experiment output path is not a directory: {path}")
        if not overwrite:
            raise ReportWriteError(
                f"experiment output directory already exists: {path}"
            )
        return
    try:
        path.mkdir(parents=True)
    except OSError as exc:
        raise ReportWriteError(
            f"could not create experiment output directory {path}: {exc}"
        ) from exc


def _write_batch_outputs(
    result: ExperimentBatchResult,
    output: Path,
    *,
    overwrite: bool,
) -> None:
    """Write successful reports, aggregate CSV, then the completion manifest."""

    for record in result.records:
        if record.report is not None and record.report_file is not None:
            write_json_report(
                record.report,
                output / Path(record.report_file),
                overwrite=overwrite,
                create_parents=True,
            )
    write_text_export(
        aggregate_result_to_csv(result),
        output / "aggregate.csv",
        overwrite=overwrite,
        create_parents=True,
    )
    write_text_export(
        batch_result_to_json(result) + "\n",
        output / "batch-manifest.json",
        overwrite=overwrite,
        create_parents=True,
    )


def _batch_record_payload(record: ExperimentRunRecord) -> dict[str, object]:
    """Return stable status data and successful reproducibility identifiers."""

    report = record.report
    return {
        "experiment_id": record.experiment_id,
        "status": record.status.value,
        "failure_kind": (
            None if record.failure_kind is None else record.failure_kind.value
        ),
        "failure_message": record.failure_message,
        "report_file": record.report_file,
        "dataset_sha256": (
            None if report is None else report.metadata.dataset.sha256
        ),
        "git_commit": None if report is None else report.metadata.git_commit,
    }
