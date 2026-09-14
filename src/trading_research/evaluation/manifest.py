"""Validated split-evaluation definitions and JSON/TOML manifest loading."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from trading_research.config import BacktestConfig, PositionSizingMode
from trading_research.errors import EvaluationManifestError, ExperimentManifestError
from trading_research.evaluation.models import (
    DEFAULT_SUSPICIOUS_RETURN_THRESHOLD,
    DatasetProvenanceOverride,
    EvaluationSplit,
    FrozenEvaluationConfiguration,
)
from trading_research.experiments.loader import (
    EXPERIMENT_MANIFEST_SCHEMA_VERSION,
    read_manifest_document,
)
from trading_research.experiments.models import (
    BenchmarkSelection,
    CommissionBasisPoints,
    FixedPositionSizingDefinition,
    PositionSizingDefinition,
    PositiveDecimal,
    RiskMetricDefinition,
    RiskPolicyDefinition,
    SlippageBasisPoints,
    StrategyDefinition,
)
from trading_research.models import EndOfTestPolicy, normalize_symbol

_ALLOWED_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "experiments", "split_evaluations"}
)
_EVALUATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class SplitEvaluationDefinition(BaseModel):
    """One explicit frozen strategy configuration and its evaluation boundaries."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    evaluation_id: str = Field(alias="id", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    dataset: Path
    symbol: str
    provenance: DatasetProvenanceOverride | None = None
    split: EvaluationSplit
    strategy: StrategyDefinition
    starting_cash: PositiveDecimal
    commission_bps: CommissionBasisPoints
    slippage_bps: SlippageBasisPoints
    position_sizing: PositionSizingDefinition
    risk_policy: RiskPolicyDefinition = Field(default_factory=RiskPolicyDefinition)
    end_of_test: EndOfTestPolicy = EndOfTestPolicy.HOLD
    risk_metrics: RiskMetricDefinition | None = None
    benchmark: BenchmarkSelection = BenchmarkSelection.NONE
    suspicious_return_threshold: PositiveDecimal = DEFAULT_SUSPICIOUS_RETURN_THRESHOLD
    allow_unapproved_volume: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_flat_strategy_parameters(cls, value: Any) -> Any:
        """Accept the documented flat strategy table as well as ordinary nesting."""

        if not isinstance(value, Mapping):
            return value
        strategy = value.get("strategy")
        if not isinstance(strategy, Mapping) or "parameters" in strategy:
            return value
        name = strategy.get("name")
        parameter_names = (
            ("fast_window", "slow_window")
            if name == "sma-crossover"
            else ("entry_window", "exit_window")
        )
        copied = dict(value)
        copied_strategy = dict(strategy)
        copied_strategy["parameters"] = {
            key: copied_strategy.pop(key)
            for key in parameter_names
            if key in copied_strategy
        }
        copied["strategy"] = copied_strategy
        return copied

    @field_validator("dataset", mode="before")
    @classmethod
    def validate_local_dataset_path(cls, value: object) -> Path:
        """Accept a non-empty local path and reject network URLs."""

        if isinstance(value, Path):
            return value
        if not isinstance(value, str) or not value.strip():
            raise ValueError("dataset must be a non-empty local filesystem path")
        if "://" in value:
            raise ValueError("dataset must be a local filesystem path")
        return Path(value)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        """Apply the existing single-symbol normalization policy."""

        return normalize_symbol(value)

    def frozen_configuration(self) -> FrozenEvaluationConfiguration:
        """Return the single immutable configuration shared by both periods."""

        sizing = self.position_sizing
        if isinstance(sizing, FixedPositionSizingDefinition):
            mode = PositionSizingMode.FIXED
            quantity = sizing.quantity
            allocation = None
        else:
            mode = PositionSizingMode.CASH_ALLOCATION
            quantity = 10
            allocation = sizing.cash_allocation_ratio
        backtest = BacktestConfig(
            initial_cash=self.starting_cash,
            trade_quantity=quantity,
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
            position_sizing_mode=mode,
            cash_allocation_ratio=allocation,
            maximum_position_value=self.risk_policy.maximum_position_value,
            minimum_cash_reserve=self.risk_policy.minimum_cash_reserve,
            end_of_test_policy=self.end_of_test,
        )
        return FrozenEvaluationConfiguration(
            strategy=self.strategy.to_domain(),
            backtest=backtest,
            risk_metrics=(
                None if self.risk_metrics is None else self.risk_metrics.to_domain()
            ),
            benchmark=self.benchmark,
            allow_unapproved_volume=self.allow_unapproved_volume,
        )


@dataclass(frozen=True, slots=True)
class SplitEvaluationManifestEntry:
    """One ordered valid evaluation or retained configuration failure."""

    evaluation_id: str
    definition: SplitEvaluationDefinition | None
    configuration_error: str | None

    def __post_init__(self) -> None:
        """Require exactly one definition or configuration error."""

        if not self.evaluation_id:
            raise ValueError("evaluation_id must not be empty")
        if (self.definition is None) == (self.configuration_error is None):
            raise ValueError("evaluation entry must contain exactly one definition or error")


@dataclass(frozen=True, slots=True)
class SplitEvaluationManifest:
    """Ordered split evaluations loaded from one existing manifest format."""

    source_path: Path
    entries: tuple[SplitEvaluationManifestEntry, ...]
    schema_version: str = EXPERIMENT_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Require a supported schema and at least one declared evaluation."""

        object.__setattr__(self, "source_path", Path(self.source_path))
        if self.schema_version != EXPERIMENT_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported experiment manifest schema version")
        if not self.entries:
            raise ValueError("manifest must contain at least one split evaluation")


def load_split_evaluation_manifest(path: Path) -> SplitEvaluationManifest:
    """Load ordered split definitions while retaining per-entry validation failures."""

    source = Path(path)
    try:
        raw = read_manifest_document(source)
    except ExperimentManifestError as exc:
        raise EvaluationManifestError(str(exc)) from exc
    unexpected = sorted(set(raw).difference(_ALLOWED_TOP_LEVEL_KEYS))
    if unexpected:
        raise EvaluationManifestError(
            f"manifest contains unsupported top-level field(s): {', '.join(unexpected)}"
        )
    if raw.get("schema_version") != EXPERIMENT_MANIFEST_SCHEMA_VERSION:
        raise EvaluationManifestError(
            f"manifest schema_version must be {EXPERIMENT_MANIFEST_SCHEMA_VERSION!r}"
        )
    raw_evaluations = raw.get("split_evaluations")
    if not isinstance(raw_evaluations, list) or not raw_evaluations:
        raise EvaluationManifestError(
            "manifest split_evaluations must be a non-empty array"
        )

    entries: list[SplitEvaluationManifestEntry] = []
    seen_ids: set[str] = set()
    for index, raw_evaluation in enumerate(raw_evaluations):
        if not isinstance(raw_evaluation, Mapping):
            raise EvaluationManifestError(
                f"split_evaluations[{index}] must be an object or table"
            )
        evaluation_id = _validated_evaluation_id(raw_evaluation.get("id"), index)
        if evaluation_id in seen_ids:
            raise EvaluationManifestError(
                f"duplicate split evaluation ID: {evaluation_id!r}"
            )
        seen_ids.add(evaluation_id)
        try:
            definition = SplitEvaluationDefinition.model_validate(raw_evaluation)
        except ValidationError as exc:
            entries.append(
                SplitEvaluationManifestEntry(
                    evaluation_id=evaluation_id,
                    definition=None,
                    configuration_error=_format_validation_error(exc),
                )
            )
        else:
            entries.append(
                SplitEvaluationManifestEntry(
                    evaluation_id=evaluation_id,
                    definition=definition,
                    configuration_error=None,
                )
            )
    return SplitEvaluationManifest(source_path=source, entries=tuple(entries))


def _validated_evaluation_id(value: object, index: int) -> str:
    """Require a filesystem-safe unique ID before retained validation."""

    if not isinstance(value, str) or _EVALUATION_ID_PATTERN.fullmatch(value) is None:
        raise EvaluationManifestError(
            f"split_evaluations[{index}].id must be a safe 1-64 character identifier"
        )
    return value


def _format_validation_error(error: ValidationError) -> str:
    """Return deterministic field errors without echoing user-supplied values."""

    return "; ".join(
        (
            f"{'.'.join(str(part) for part in detail['loc'])}: {detail['msg']}"
            if detail["loc"]
            else detail["msg"]
        )
        for detail in error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
    )
