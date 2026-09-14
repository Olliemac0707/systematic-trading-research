"""Strongly typed predefined experiment configuration models."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from trading_research.config import BacktestConfig, PositionSizingMode
from trading_research.models import EndOfTestPolicy, normalize_symbol
from trading_research.performance import RiskMetricSettings
from trading_research.reporting import StrategyRunConfiguration
from trading_research.strategies import (
    DonchianBreakoutParameters,
    SmaCrossoverParameters,
    StrategyName,
)

_ZERO = Decimal("0")


def _parse_exact_decimal(value: object) -> Decimal:
    """Parse exact decimal strings and integers without accepting binary floats."""

    if isinstance(value, bool):
        raise ValueError("must be a decimal string or integer, not a boolean")
    if isinstance(value, Decimal):
        parsed = value
    elif isinstance(value, int):
        parsed = Decimal(value)
    elif isinstance(value, str):
        try:
            parsed = Decimal(value.strip())
        except InvalidOperation as exc:
            raise ValueError("must be an exact decimal string") from exc
    else:
        raise ValueError("must be an exact decimal string or integer")
    if not parsed.is_finite():
        raise ValueError("must be finite")
    return parsed


ExactDecimal = Annotated[Decimal, BeforeValidator(_parse_exact_decimal)]
PositiveDecimal = Annotated[ExactDecimal, Field(gt=0)]
NonNegativeDecimal = Annotated[ExactDecimal, Field(ge=0)]
CommissionBasisPoints = Annotated[ExactDecimal, Field(ge=0, le=10_000)]
SlippageBasisPoints = Annotated[ExactDecimal, Field(ge=0, lt=10_000)]
AllocationRatio = Annotated[ExactDecimal, Field(gt=0, le=1)]
PositiveInteger = Annotated[StrictInt, Field(ge=1)]


class BenchmarkSelection(StrEnum):
    """Supported optional experiment benchmark selections."""

    NONE = "none"
    BUY_AND_HOLD = "buy-and-hold"


class SmaParametersDefinition(BaseModel):
    """Manifest parameters for the existing SMA crossover strategy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fast_window: PositiveInteger
    slow_window: PositiveInteger

    @model_validator(mode="after")
    def validate_window_order(self) -> SmaParametersDefinition:
        """Retain the existing strictly ordered moving-average policy."""

        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be smaller than slow_window")
        return self


class DonchianParametersDefinition(BaseModel):
    """Manifest parameters for independent Donchian closing channels."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry_window: PositiveInteger
    exit_window: PositiveInteger


class SmaStrategyDefinition(BaseModel):
    """Typed SMA strategy selection from a predefined manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["sma-crossover"]
    parameters: SmaParametersDefinition

    def to_domain(self) -> StrategyRunConfiguration:
        """Return the existing immutable strategy configuration."""

        return StrategyRunConfiguration(
            name=StrategyName.SMA_CROSSOVER,
            parameters=SmaCrossoverParameters(
                fast_window=self.parameters.fast_window,
                slow_window=self.parameters.slow_window,
            ),
        )


class DonchianStrategyDefinition(BaseModel):
    """Typed Donchian strategy selection from a predefined manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["donchian-breakout"]
    parameters: DonchianParametersDefinition

    def to_domain(self) -> StrategyRunConfiguration:
        """Return the existing immutable strategy configuration."""

        return StrategyRunConfiguration(
            name=StrategyName.DONCHIAN_BREAKOUT,
            parameters=DonchianBreakoutParameters(
                entry_window=self.parameters.entry_window,
                exit_window=self.parameters.exit_window,
            ),
        )


StrategyDefinition = Annotated[
    SmaStrategyDefinition | DonchianStrategyDefinition,
    Field(discriminator="name"),
]


class FixedPositionSizingDefinition(BaseModel):
    """Predefined fixed whole-share sizing parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["fixed"]
    quantity: PositiveInteger


class CashAllocationSizingDefinition(BaseModel):
    """Predefined exact cash-allocation sizing parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["cash-allocation"]
    cash_allocation_ratio: AllocationRatio


PositionSizingDefinition = Annotated[
    FixedPositionSizingDefinition | CashAllocationSizingDefinition,
    Field(discriminator="mode"),
]


class RiskPolicyDefinition(BaseModel):
    """Optional existing pre-trade risk-policy limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    maximum_position_value: PositiveDecimal | None = None
    minimum_cash_reserve: NonNegativeDecimal | None = None


class RiskMetricDefinition(BaseModel):
    """Optional exact settings for existing return and risk metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    periods_per_year: PositiveDecimal | None = None
    risk_free_rate_per_period: ExactDecimal | None = None
    target_return_per_period: ExactDecimal | None = None

    @model_validator(mode="after")
    def require_an_explicit_setting(self) -> RiskMetricDefinition:
        """Reject an empty opt-in risk-metric section."""

        if all(
            value is None
            for value in (
                self.periods_per_year,
                self.risk_free_rate_per_period,
                self.target_return_per_period,
            )
        ):
            raise ValueError("risk_metrics must contain at least one setting")
        return self

    def to_domain(self) -> RiskMetricSettings:
        """Return existing risk settings using the CLI's explicit zero defaults."""

        return RiskMetricSettings(
            periods_per_year=self.periods_per_year,
            risk_free_rate_per_period=(
                _ZERO
                if self.risk_free_rate_per_period is None
                else self.risk_free_rate_per_period
            ),
            target_return_per_period=(
                _ZERO
                if self.target_return_per_period is None
                else self.target_return_per_period
            ),
        )


class ExperimentDefinition(BaseModel):
    """One explicit, independently validated historical experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    experiment_id: str = Field(alias="id", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    dataset: Path
    symbol: str
    start: datetime | None = None
    end: datetime | None = None
    strategy: StrategyDefinition
    starting_cash: PositiveDecimal
    commission_bps: CommissionBasisPoints
    slippage_bps: SlippageBasisPoints
    position_sizing: PositionSizingDefinition
    risk_policy: RiskPolicyDefinition = Field(default_factory=RiskPolicyDefinition)
    end_of_test: EndOfTestPolicy = EndOfTestPolicy.HOLD
    risk_metrics: RiskMetricDefinition | None = None
    benchmark: BenchmarkSelection = BenchmarkSelection.NONE

    @field_validator("dataset", mode="before")
    @classmethod
    def validate_local_dataset_path(cls, value: object) -> Path:
        """Accept only non-empty local filesystem paths, never URLs."""

        if isinstance(value, Path):
            path = value
        elif isinstance(value, str) and value.strip():
            if "://" in value:
                raise ValueError("dataset must be a local filesystem path")
            path = Path(value)
        else:
            raise ValueError("dataset must be a non-empty local filesystem path")
        return path

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        """Apply the existing domain symbol policy."""

        return normalize_symbol(value)

    @field_validator("start", "end")
    @classmethod
    def require_aware_timestamps(cls, value: datetime | None) -> datetime | None:
        """Require optional experiment filters to carry an explicit UTC offset."""

        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("timestamp must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_period(self) -> ExperimentDefinition:
        """Reject reversed inclusive dataset filters."""

        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("start must not be after end")
        return self

    def strategy_configuration(self) -> StrategyRunConfiguration:
        """Return the existing typed strategy descriptor."""

        return self.strategy.to_domain()

    def backtest_configuration(self) -> BacktestConfig:
        """Return existing simulation configuration without financial calculations."""

        sizing = self.position_sizing
        if isinstance(sizing, FixedPositionSizingDefinition):
            mode = PositionSizingMode.FIXED
            quantity = sizing.quantity
            allocation_ratio = None
        else:
            mode = PositionSizingMode.CASH_ALLOCATION
            quantity = 10
            allocation_ratio = sizing.cash_allocation_ratio
        return BacktestConfig(
            initial_cash=self.starting_cash,
            trade_quantity=quantity,
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
            position_sizing_mode=mode,
            cash_allocation_ratio=allocation_ratio,
            maximum_position_value=self.risk_policy.maximum_position_value,
            minimum_cash_reserve=self.risk_policy.minimum_cash_reserve,
            end_of_test_policy=self.end_of_test,
        )

    def risk_metric_settings(self) -> RiskMetricSettings | None:
        """Return optional existing metric settings without inferring frequency."""

        return None if self.risk_metrics is None else self.risk_metrics.to_domain()
