"""Predefined, sequential batch experiment APIs."""

from trading_research.experiments.loader import (
    EXPERIMENT_MANIFEST_SCHEMA_VERSION,
    ExperimentManifest,
    ExperimentManifestEntry,
    load_experiment_manifest,
)
from trading_research.experiments.models import (
    BenchmarkSelection,
    ExperimentDefinition,
)
from trading_research.experiments.runner import (
    AGGREGATE_CSV_COLUMNS,
    BATCH_RESULT_SCHEMA_VERSION,
    ExperimentBatchResult,
    ExperimentFailureKind,
    ExperimentRunRecord,
    ExperimentStatus,
    aggregate_result_to_csv,
    batch_result_to_json,
    run_experiment_batch,
)

__all__ = [
    "EXPERIMENT_MANIFEST_SCHEMA_VERSION",
    "AGGREGATE_CSV_COLUMNS",
    "BATCH_RESULT_SCHEMA_VERSION",
    "BenchmarkSelection",
    "ExperimentDefinition",
    "ExperimentBatchResult",
    "ExperimentFailureKind",
    "ExperimentManifest",
    "ExperimentManifestEntry",
    "ExperimentRunRecord",
    "ExperimentStatus",
    "aggregate_result_to_csv",
    "batch_result_to_json",
    "load_experiment_manifest",
    "run_experiment_batch",
]
