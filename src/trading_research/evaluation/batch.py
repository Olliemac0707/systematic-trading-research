"""Deterministic sequential execution for declared split evaluations."""

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from trading_research.data import CsvMarketDataProvider
from trading_research.errors import (
    BacktestError,
    MarketDataError,
    ReportWriteError,
    StrategyError,
)
from trading_research.evaluation.exports import write_split_evaluation_exports
from trading_research.evaluation.manifest import (
    SplitEvaluationManifest,
    SplitEvaluationManifestEntry,
)
from trading_research.evaluation.models import SplitEvaluationResult
from trading_research.evaluation.provenance import load_dataset_provenance
from trading_research.evaluation.runner import run_split_evaluation
from trading_research.reporting import current_git_commit, write_text_export

SPLIT_EVALUATION_BATCH_SCHEMA_NAME = "trading-research.split-evaluation-batch"
SPLIT_EVALUATION_BATCH_SCHEMA_VERSION = "1.0"


class SplitEvaluationStatus(StrEnum):
    """Stable per-evaluation completion states."""

    SUCCESS = "success"
    FAILURE = "failure"


class SplitEvaluationFailureKind(StrEnum):
    """Expected retained failure categories."""

    CONFIGURATION = "configuration_error"
    MARKET_DATA = "market_data_error"
    EVALUATION = "evaluation_error"


@dataclass(frozen=True, slots=True)
class SplitEvaluationRecord:
    """One ordered successful evaluation or explicit expected failure."""

    evaluation_id: str
    status: SplitEvaluationStatus
    failure_kind: SplitEvaluationFailureKind | None
    failure_message: str | None
    output_directory: str | None
    result: SplitEvaluationResult | None

    def __post_init__(self) -> None:
        """Require mutually exclusive success and failure payloads."""

        if not self.evaluation_id:
            raise ValueError("evaluation_id must not be empty")
        if not isinstance(self.status, SplitEvaluationStatus):
            raise TypeError("status must be SplitEvaluationStatus")
        if self.status is SplitEvaluationStatus.SUCCESS:
            if (
                self.failure_kind is not None
                or self.failure_message is not None
                or self.output_directory is None
                or self.result is None
            ):
                raise ValueError("successful evaluation records require result information")
        elif (
            self.failure_kind is None
            or not self.failure_message
            or self.output_directory is not None
            or self.result is not None
        ):
            raise ValueError("failed evaluation records require failure information")


@dataclass(frozen=True, slots=True)
class SplitEvaluationBatchResult:
    """Immutable ordered result of one declared sequential evaluation batch."""

    source_manifest: str
    git_commit: str
    records: tuple[SplitEvaluationRecord, ...]

    def __post_init__(self) -> None:
        """Require a non-empty typed result for one named source manifest."""

        if not self.source_manifest:
            raise ValueError("source_manifest must not be empty")
        if not self.git_commit:
            raise ValueError("git_commit must not be empty")
        if not self.records:
            raise ValueError("records must not be empty")
        if any(not isinstance(record, SplitEvaluationRecord) for record in self.records):
            raise TypeError("records must contain SplitEvaluationRecord values")

    @property
    def success_count(self) -> int:
        """Return the number of completed evaluations."""

        return sum(record.status is SplitEvaluationStatus.SUCCESS for record in self.records)

    @property
    def failure_count(self) -> int:
        """Return the number of explicitly recorded failures."""

        return len(self.records) - self.success_count

    @property
    def succeeded(self) -> bool:
        """Return whether all declared evaluations completed."""

        return self.failure_count == 0


def run_split_evaluation_batch(
    manifest: SplitEvaluationManifest,
    output_directory: Path,
    *,
    overwrite: bool = False,
    repository: Path | None = None,
) -> SplitEvaluationBatchResult:
    """Run every declared evaluation sequentially and retain every outcome."""

    if not isinstance(manifest, SplitEvaluationManifest):
        raise TypeError("manifest must be SplitEvaluationManifest")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")
    output = Path(output_directory)
    _prepare_output_directory(output, overwrite=overwrite)
    repository_path = Path.cwd() if repository is None else Path(repository)
    git_commit = current_git_commit(repository_path)
    if git_commit is None:
        raise BacktestError("could not determine the current Git revision for evaluation")

    records: list[SplitEvaluationRecord] = []
    for entry in manifest.entries:
        record = _run_entry(
            entry,
            manifest=manifest,
            git_commit=git_commit,
            repository=repository_path,
        )
        records.append(record)
        if record.result is not None and record.output_directory is not None:
            write_split_evaluation_exports(
                record.result,
                output / record.output_directory,
                overwrite=overwrite,
            )
    result = SplitEvaluationBatchResult(
        source_manifest=manifest.source_path.name,
        git_commit=git_commit,
        records=tuple(records),
    )
    write_text_export(
        split_evaluation_batch_to_json(result) + "\n",
        output / "evaluation-manifest.json",
        overwrite=overwrite,
        create_parents=True,
    )
    return result


def split_evaluation_batch_to_json(
    result: SplitEvaluationBatchResult,
    *,
    indent: int = 2,
) -> str:
    """Serialize ordered success/failure records without financial floats."""

    if not isinstance(result, SplitEvaluationBatchResult):
        raise TypeError("result must be SplitEvaluationBatchResult")
    payload = {
        "schema": SPLIT_EVALUATION_BATCH_SCHEMA_NAME,
        "schema_version": SPLIT_EVALUATION_BATCH_SCHEMA_VERSION,
        "source_manifest": result.source_manifest,
        "git_commit": result.git_commit,
        "evaluation_count": len(result.records),
        "success_count": result.success_count,
        "failure_count": result.failure_count,
        "results": [
            {
                "evaluation_id": record.evaluation_id,
                "status": record.status.value,
                "failure_kind": (
                    None if record.failure_kind is None else record.failure_kind.value
                ),
                "failure_message": record.failure_message,
                "output_directory": record.output_directory,
                "configuration_sha256": (
                    None
                    if record.result is None
                    else record.result.configuration_sha256
                ),
                "data_sha256": (
                    None if record.result is None else record.result.data_sha256
                ),
            }
            for record in result.records
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=indent, allow_nan=False)


def _run_entry(
    entry: SplitEvaluationManifestEntry,
    *,
    manifest: SplitEvaluationManifest,
    git_commit: str,
    repository: Path,
) -> SplitEvaluationRecord:
    """Execute one entry or retain its expected failure for the final manifest."""

    if entry.configuration_error is not None:
        return _failure_record(
            entry.evaluation_id,
            SplitEvaluationFailureKind.CONFIGURATION,
            entry.configuration_error,
        )
    definition = entry.definition
    if definition is None:
        raise AssertionError("validated evaluation entry is missing its definition")
    dataset = (
        definition.dataset
        if definition.dataset.is_absolute()
        else manifest.source_path.parent / definition.dataset
    )
    if not dataset.is_file():
        return _failure_record(
            entry.evaluation_id,
            SplitEvaluationFailureKind.MARKET_DATA,
            f"dataset file not found: {dataset.name}",
        )
    try:
        provenance = load_dataset_provenance(dataset, definition.provenance)
        configuration = definition.frozen_configuration()
    except (TypeError, ValueError) as exc:
        return _failure_record(
            entry.evaluation_id,
            SplitEvaluationFailureKind.CONFIGURATION,
            _safe_failure_message(exc, dataset),
        )
    try:
        bars = tuple(
            CsvMarketDataProvider(dataset).get_historical_bars(
                symbol=definition.symbol,
                start=None,
                end=None,
            )
        )
        evaluation = run_split_evaluation(
            evaluation_id=entry.evaluation_id,
            data_path=dataset,
            symbol=definition.symbol,
            bars=bars,
            split=definition.split,
            configuration=configuration,
            provenance=provenance,
            git_commit=git_commit,
            suspicious_return_threshold=definition.suspicious_return_threshold,
            base_directory=repository,
        )
    except (MarketDataError, OSError) as exc:
        return _failure_record(
            entry.evaluation_id,
            SplitEvaluationFailureKind.MARKET_DATA,
            _safe_failure_message(exc, dataset),
        )
    except (BacktestError, StrategyError, TypeError, ValueError) as exc:
        return _failure_record(
            entry.evaluation_id,
            SplitEvaluationFailureKind.EVALUATION,
            _safe_failure_message(exc, dataset),
        )
    return SplitEvaluationRecord(
        evaluation_id=entry.evaluation_id,
        status=SplitEvaluationStatus.SUCCESS,
        failure_kind=None,
        failure_message=None,
        output_directory=entry.evaluation_id,
        result=evaluation,
    )


def _failure_record(
    evaluation_id: str,
    kind: SplitEvaluationFailureKind,
    message: str,
) -> SplitEvaluationRecord:
    """Create one explicit expected failure without an invented result."""

    return SplitEvaluationRecord(
        evaluation_id=evaluation_id,
        status=SplitEvaluationStatus.FAILURE,
        failure_kind=kind,
        failure_message=message,
        output_directory=None,
        result=None,
    )


def _prepare_output_directory(path: Path, *, overwrite: bool) -> None:
    """Reject an existing output root unless replacement was explicit."""

    if path.exists():
        if not path.is_dir():
            raise ReportWriteError(f"evaluation output path is not a directory: {path}")
        if not overwrite:
            raise ReportWriteError(f"evaluation output directory already exists: {path}")
        return
    try:
        path.mkdir(parents=True)
    except OSError as exc:
        raise ReportWriteError(
            f"could not create evaluation output directory {path}: {exc}"
        ) from exc


def _safe_failure_message(error: BaseException, dataset: Path) -> str:
    """Remove local absolute dataset paths from persisted failure descriptions."""

    message = str(error)
    candidates = {str(dataset), str(dataset.resolve(strict=False))}
    for candidate in candidates:
        if candidate:
            message = message.replace(candidate, dataset.name)
    return message
