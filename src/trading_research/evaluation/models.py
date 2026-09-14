"""Immutable domain models for split-aware out-of-sample evaluation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trading_research.benchmarks import BenchmarkComparison
from trading_research.config import BacktestConfig
from trading_research.experiments.models import (
    BenchmarkSelection,
    ExactDecimal,
    PositiveDecimal,
)
from trading_research.performance import RiskMetricSettings
from trading_research.reporting import StrategyRunConfiguration, StructuredBacktestReport

SPLIT_EVALUATION_SCHEMA_NAME = "trading-research.split-evaluation"
SPLIT_EVALUATION_SCHEMA_VERSION = "1.0"
DEFAULT_SUSPICIOUS_RETURN_THRESHOLD = Decimal("0.40")


class PriceAdjustmentPolicy(StrEnum):
    """Declared close-price adjustment convention; never inferred."""

    RAW = "raw"
    SPLIT_ADJUSTED = "split-adjusted"
    TOTAL_RETURN_ADJUSTED = "total-return-adjusted"
    UNKNOWN = "unknown"


class DividendTreatment(StrEnum):
    """Declared handling of dividends in the supplied dataset."""

    EXCLUDED = "excluded"
    INCLUDED_IN_ADJUSTED_PRICE = "included-in-adjusted-price"
    SEPARATE_CASH_FLOW = "separate-cash-flow"
    UNKNOWN = "unknown"


class SplitTreatment(StrEnum):
    """Declared handling of stock splits in the supplied dataset."""

    UNADJUSTED = "unadjusted"
    ADJUSTED = "adjusted"
    UNKNOWN = "unknown"


class VolumeValidationStatus(StrEnum):
    """Declared suitability of volume for research signals."""

    APPROVED = "approved"
    INTERNALLY_VALID_WITH_CROSS_PROVIDER_WARNING = (
        "internally_valid_with_cross_provider_warning"
    )
    UNKNOWN = "unknown"


class WarmupPolicy(StrEnum):
    """Explicit strategy-history policy at the holdout boundary."""

    ISOLATED = "isolated"
    CARRY_HISTORY = "carry-history"


def _strip_required_text(value: str, name: str) -> str:
    """Return non-blank text with stable surrounding whitespace removal."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-blank text")
    return value.strip()


class DatasetProvenance(BaseModel):
    """Explicit immutable statements about a local dataset's origin and treatment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    source_name: str
    source_url: str | None = None
    downloaded_at: datetime | None = None
    price_adjustment: PriceAdjustmentPolicy = PriceAdjustmentPolicy.UNKNOWN
    dividend_treatment: DividendTreatment = DividendTreatment.UNKNOWN
    split_treatment: SplitTreatment = SplitTreatment.UNKNOWN
    bar_frequency: str
    exchange_timezone: str | None = None
    currency: str | None = None
    volume_validation_status: VolumeValidationStatus = VolumeValidationStatus.UNKNOWN
    volume_approved_for_strategy_signals: bool | None = None
    notes: str | None = None

    @field_validator("dataset_id", "source_name", "bar_frequency")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Require documented identifiers, source, and observation frequency."""

        name = getattr(info, "field_name", "value")
        return _strip_required_text(value, name)

    @field_validator("source_url", "exchange_timezone", "currency", "notes")
    @classmethod
    def validate_optional_text(cls, value: str | None, info: object) -> str | None:
        """Reject supplied-but-blank optional provenance statements."""

        if value is None:
            return None
        name = getattr(info, "field_name", "value")
        return _strip_required_text(value, name)

    @field_validator("downloaded_at")
    @classmethod
    def require_aware_download_timestamp(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        """Require an explicit offset whenever acquisition time is documented."""

        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("downloaded_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def reject_contradictory_adjustment_claims(self) -> DatasetProvenance:
        """Reject combinations whose stable meanings cannot simultaneously hold."""

        if self.price_adjustment is PriceAdjustmentPolicy.TOTAL_RETURN_ADJUSTED:
            if self.dividend_treatment is not DividendTreatment.INCLUDED_IN_ADJUSTED_PRICE:
                raise ValueError(
                    "total-return-adjusted prices require dividends included in adjusted price"
                )
            if self.split_treatment is not SplitTreatment.ADJUSTED:
                raise ValueError(
                    "total-return-adjusted prices require splits adjusted"
                )
        if (
            self.price_adjustment is PriceAdjustmentPolicy.SPLIT_ADJUSTED
            and self.split_treatment is not SplitTreatment.ADJUSTED
        ):
            raise ValueError("split-adjusted prices require splits adjusted")
        if (
            self.price_adjustment is PriceAdjustmentPolicy.RAW
            and self.split_treatment is SplitTreatment.ADJUSTED
        ):
            raise ValueError("raw prices cannot declare splits adjusted")
        if (
            self.dividend_treatment is DividendTreatment.INCLUDED_IN_ADJUSTED_PRICE
            and self.price_adjustment is not PriceAdjustmentPolicy.TOTAL_RETURN_ADJUSTED
        ):
            raise ValueError(
                "included-in-adjusted-price dividends require total-return-adjusted prices"
            )
        return self


class DatasetProvenanceOverride(BaseModel):
    """Optional explicit manifest fields merged over a dataset sidecar."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str | None = None
    source_name: str | None = None
    source_url: str | None = None
    downloaded_at: datetime | None = None
    price_adjustment: PriceAdjustmentPolicy | None = None
    dividend_treatment: DividendTreatment | None = None
    split_treatment: SplitTreatment | None = None
    bar_frequency: str | None = None
    exchange_timezone: str | None = None
    currency: str | None = None
    volume_validation_status: VolumeValidationStatus | None = None
    volume_approved_for_strategy_signals: bool | None = None
    notes: str | None = None


class DatasetQualitySummary(BaseModel):
    """Pure diagnostics for supplied observations; no bar is repaired or removed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bar_count: int = Field(ge=1)
    first_timestamp: datetime
    last_timestamp: datetime
    duplicate_timestamp_count: int = Field(ge=0)
    non_increasing_timestamp_count: int = Field(ge=0)
    minimum_gap: timedelta | None
    maximum_gap: timedelta | None
    median_gap_seconds: ExactDecimal | None
    zero_volume_count: int = Field(ge=0)
    suspicious_return_threshold: PositiveDecimal
    suspicious_return_count: int = Field(ge=0)
    largest_positive_close_return: ExactDecimal | None
    largest_negative_close_return: ExactDecimal | None

    @model_validator(mode="after")
    def validate_summary_relationships(self) -> DatasetQualitySummary:
        """Validate chronology, counts, gaps, and exact diagnostic ratios."""

        for timestamp in (self.first_timestamp, self.last_timestamp):
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("quality timestamps must be timezone-aware")
        if self.first_timestamp > self.last_timestamp:
            raise ValueError("first_timestamp must not follow last_timestamp")
        counts = (
            self.duplicate_timestamp_count,
            self.non_increasing_timestamp_count,
            self.zero_volume_count,
            self.suspicious_return_count,
        )
        if any(count > self.bar_count for count in counts):
            raise ValueError("quality diagnostic counts cannot exceed bar_count")
        if self.suspicious_return_threshold <= 0:
            raise ValueError("suspicious_return_threshold must be positive")
        if (self.minimum_gap is None) != (self.maximum_gap is None):
            raise ValueError("minimum and maximum gaps must be present together")
        if (self.minimum_gap is None) != (self.median_gap_seconds is None):
            raise ValueError("median gap must be present exactly when gaps exist")
        if (
            self.minimum_gap is not None
            and self.maximum_gap is not None
            and (
                self.minimum_gap <= timedelta(0)
                or self.minimum_gap > self.maximum_gap
            )
        ):
            raise ValueError("quality gaps must be positive and ordered")
        return self


class EvaluationSplit(BaseModel):
    """Explicit inclusive, non-overlapping development and holdout boundaries."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    development_start: datetime
    development_end: datetime
    holdout_start: datetime
    holdout_end: datetime
    warmup_policy: WarmupPolicy = WarmupPolicy.ISOLATED

    @field_validator(
        "development_start",
        "development_end",
        "holdout_start",
        "holdout_end",
    )
    @classmethod
    def require_aware_boundary(cls, value: datetime) -> datetime:
        """Require every split boundary to carry an explicit UTC offset."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("split timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_ranges(self) -> EvaluationSplit:
        """Require ordered inclusive ranges separated by at least one instant."""

        if self.development_start > self.development_end:
            raise ValueError("development_start must not be after development_end")
        if self.development_end >= self.holdout_start:
            raise ValueError("development and holdout ranges must not overlap")
        if self.holdout_start > self.holdout_end:
            raise ValueError("holdout_start must not be after holdout_end")
        return self

    @property
    def boundary_gap(self) -> timedelta:
        """Return the explicit elapsed gap from development end to holdout start."""

        return self.holdout_start - self.development_end


@dataclass(frozen=True, slots=True)
class FrozenEvaluationConfiguration:
    """One immutable configuration used unchanged for both split runs."""

    strategy: StrategyRunConfiguration
    backtest: BacktestConfig
    risk_metrics: RiskMetricSettings | None
    benchmark: BenchmarkSelection
    allow_unapproved_volume: bool = False

    def __post_init__(self) -> None:
        """Validate the existing typed configuration components."""

        if not isinstance(self.strategy, StrategyRunConfiguration):
            raise TypeError("strategy must be StrategyRunConfiguration")
        if not isinstance(self.backtest, BacktestConfig):
            raise TypeError("backtest must be BacktestConfig")
        if self.risk_metrics is not None and not isinstance(
            self.risk_metrics,
            RiskMetricSettings,
        ):
            raise TypeError("risk_metrics must be RiskMetricSettings or None")
        if not isinstance(self.benchmark, BenchmarkSelection):
            raise TypeError("benchmark must be BenchmarkSelection")
        if not isinstance(self.allow_unapproved_volume, bool):
            raise TypeError("allow_unapproved_volume must be a bool")


@dataclass(frozen=True, slots=True)
class SplitStabilityComparison:
    """Holdout-minus-development differences without an overall score."""

    development_total_return: Decimal
    holdout_total_return: Decimal
    total_return_change: Decimal
    development_excess_return: Decimal | None
    holdout_excess_return: Decimal | None
    excess_return_change: Decimal | None
    development_maximum_drawdown: Decimal
    holdout_maximum_drawdown: Decimal
    maximum_drawdown_change: Decimal
    development_sharpe: Decimal | None
    holdout_sharpe: Decimal | None
    sharpe_change: Decimal | None
    development_trade_count: int
    holdout_trade_count: int
    development_time_in_market: Decimal | None
    holdout_time_in_market: Decimal | None

    def __post_init__(self) -> None:
        """Require exact values and direct holdout-minus-development relationships."""

        for item in fields(self):
            name = item.name
            value = getattr(self, name)
            if name.endswith("trade_count"):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{name} must be a non-negative integer")
            elif value is not None and not isinstance(value, Decimal):
                raise TypeError(f"{name} must be a Decimal or None")
        if self.total_return_change != (
            self.holdout_total_return - self.development_total_return
        ):
            raise ValueError("total_return_change must be holdout minus development")
        if self.maximum_drawdown_change != (
            self.holdout_maximum_drawdown - self.development_maximum_drawdown
        ):
            raise ValueError("maximum_drawdown_change must be holdout minus development")
        _validate_optional_change(
            self.development_excess_return,
            self.holdout_excess_return,
            self.excess_return_change,
            "excess_return_change",
        )
        _validate_optional_change(
            self.development_sharpe,
            self.holdout_sharpe,
            self.sharpe_change,
            "sharpe_change",
        )


@dataclass(frozen=True, slots=True)
class SplitEvaluationResult:
    """Complete reproducible development-versus-holdout evaluation."""

    evaluation_id: str
    generated_at: datetime
    git_commit: str
    data_sha256: str
    configuration_sha256: str
    development_configuration_sha256: str
    holdout_configuration_sha256: str
    provenance: DatasetProvenance
    quality_summary: DatasetQualitySummary
    split: EvaluationSplit
    warmup_bar_count: int
    warmup_first_timestamp: datetime | None
    warmup_last_timestamp: datetime | None
    development: StructuredBacktestReport
    holdout: StructuredBacktestReport
    development_benchmark_comparison: BenchmarkComparison | None
    holdout_benchmark_comparison: BenchmarkComparison | None
    stability_comparison: SplitStabilityComparison

    def __post_init__(self) -> None:
        """Reject mismatched frozen configurations or inconsistent report periods."""

        _strip_required_text(self.evaluation_id, "evaluation_id")
        if not isinstance(self.provenance, DatasetProvenance):
            raise TypeError("provenance must be DatasetProvenance")
        if not isinstance(self.quality_summary, DatasetQualitySummary):
            raise TypeError("quality_summary must be DatasetQualitySummary")
        if not isinstance(self.split, EvaluationSplit):
            raise TypeError("split must be EvaluationSplit")
        if not isinstance(self.stability_comparison, SplitStabilityComparison):
            raise TypeError("stability_comparison must be SplitStabilityComparison")
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        for name, value in (
            ("git_commit", self.git_commit),
            ("data_sha256", self.data_sha256),
            ("configuration_sha256", self.configuration_sha256),
            ("development_configuration_sha256", self.development_configuration_sha256),
            ("holdout_configuration_sha256", self.holdout_configuration_sha256),
        ):
            _strip_required_text(value, name)
        for name, value in (
            ("data_sha256", self.data_sha256),
            ("configuration_sha256", self.configuration_sha256),
            ("development_configuration_sha256", self.development_configuration_sha256),
            ("holdout_configuration_sha256", self.holdout_configuration_sha256),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 hexadecimal digest")
        if not (
            self.configuration_sha256
            == self.development_configuration_sha256
            == self.holdout_configuration_sha256
        ):
            raise ValueError("development and holdout configuration fingerprints differ")
        if isinstance(self.warmup_bar_count, bool) or not isinstance(
            self.warmup_bar_count,
            int,
        ):
            raise TypeError("warmup_bar_count must be an integer")
        if self.warmup_bar_count < 0:
            raise ValueError("warmup_bar_count must be non-negative")
        warmup_timestamps = (
            self.warmup_first_timestamp,
            self.warmup_last_timestamp,
        )
        if self.warmup_bar_count == 0 and any(
            timestamp is not None for timestamp in warmup_timestamps
        ):
            raise ValueError("zero warm-up bars require no warm-up timestamps")
        if self.warmup_bar_count > 0 and any(
            timestamp is None for timestamp in warmup_timestamps
        ):
            raise ValueError("warm-up bars require first and last timestamps")
        if self.warmup_last_timestamp is not None:
            if self.warmup_last_timestamp >= self.split.holdout_start:
                raise ValueError("warm-up history must strictly precede holdout")
            if (
                self.warmup_first_timestamp is not None
                and self.warmup_first_timestamp > self.warmup_last_timestamp
            ):
                raise ValueError("warm-up timestamps must be ordered")
        for report_name, report in (
            ("development", self.development),
            ("holdout", self.holdout),
        ):
            if not isinstance(report, StructuredBacktestReport):
                raise TypeError(f"{report_name} must be StructuredBacktestReport")
            if not report.backtest_result.reconciliation.is_reconciled:
                raise ValueError(f"{report_name} result must reconcile exactly")
        if self.development.metadata.dataset.actual_start < self.split.development_start:
            raise ValueError("development report starts before its inclusive boundary")
        if self.development.metadata.dataset.actual_end > self.split.development_end:
            raise ValueError("development report ends after its inclusive boundary")
        if self.holdout.metadata.dataset.actual_start < self.split.holdout_start:
            raise ValueError("holdout report starts before its inclusive boundary")
        if self.holdout.metadata.dataset.actual_end > self.split.holdout_end:
            raise ValueError("holdout report ends after its inclusive boundary")
        if self.development.metadata.git_commit != self.git_commit:
            raise ValueError("development report Git revision must match evaluation")
        if self.holdout.metadata.git_commit != self.git_commit:
            raise ValueError("holdout report Git revision must match evaluation")
        if self.development.metadata.dataset.sha256 != self.data_sha256:
            raise ValueError("development report data hash must match evaluation")
        if self.holdout.metadata.dataset.sha256 != self.data_sha256:
            raise ValueError("holdout report data hash must match evaluation")
        if (
            self.development_benchmark_comparison
            != self.development.benchmark_comparison
        ):
            raise ValueError("development benchmark comparison must match its report")
        if self.holdout_benchmark_comparison != self.holdout.benchmark_comparison:
            raise ValueError("holdout benchmark comparison must match its report")
        stability = self.stability_comparison
        if stability.development_total_return != self.development.performance.total_return:
            raise ValueError("development stability return must match its report")
        if stability.holdout_total_return != self.holdout.performance.total_return:
            raise ValueError("holdout stability return must match its report")
        if stability.development_maximum_drawdown != (
            self.development.performance.maximum_percentage_drawdown
        ):
            raise ValueError("development stability drawdown must match its report")
        if stability.holdout_maximum_drawdown != (
            self.holdout.performance.maximum_percentage_drawdown
        ):
            raise ValueError("holdout stability drawdown must match its report")
        expected_development_excess = (
            None
            if self.development.benchmark_comparison is None
            else self.development.benchmark_comparison.excess_return
        )
        expected_holdout_excess = (
            None
            if self.holdout.benchmark_comparison is None
            else self.holdout.benchmark_comparison.excess_return
        )
        if stability.development_excess_return != expected_development_excess:
            raise ValueError("development stability excess return must match its report")
        if stability.holdout_excess_return != expected_holdout_excess:
            raise ValueError("holdout stability excess return must match its report")
        expected_development_sharpe = (
            None
            if self.development.risk_adjusted_metrics is None
            else self.development.risk_adjusted_metrics.periodic_sharpe_ratio
        )
        expected_holdout_sharpe = (
            None
            if self.holdout.risk_adjusted_metrics is None
            else self.holdout.risk_adjusted_metrics.periodic_sharpe_ratio
        )
        if stability.development_sharpe != expected_development_sharpe:
            raise ValueError("development stability Sharpe must match its report")
        if stability.holdout_sharpe != expected_holdout_sharpe:
            raise ValueError("holdout stability Sharpe must match its report")
        if stability.development_trade_count != (
            self.development.trade_statistics.completed_trade_count
        ):
            raise ValueError("development stability trades must match its report")
        if stability.holdout_trade_count != (
            self.holdout.trade_statistics.completed_trade_count
        ):
            raise ValueError("holdout stability trades must match its report")
        if stability.development_time_in_market != (
            self.development.exposure_statistics.time_in_market_ratio
        ):
            raise ValueError("development stability exposure must match its report")
        if stability.holdout_time_in_market != (
            self.holdout.exposure_statistics.time_in_market_ratio
        ):
            raise ValueError("holdout stability exposure must match its report")


def _validate_optional_change(
    development: Decimal | None,
    holdout: Decimal | None,
    change: Decimal | None,
    name: str,
) -> None:
    """Require an optional difference only when both source values exist."""

    if development is None or holdout is None:
        if change is not None:
            raise ValueError(f"{name} must be None when either source value is missing")
        return
    if change != holdout - development:
        raise ValueError(f"{name} must be holdout minus development")
