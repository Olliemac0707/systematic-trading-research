"""Pydantic models for environment-backed application configuration."""

import re
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from trading_research.models import EndOfTestPolicy

_PROVIDER_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


class PositionSizingMode(StrEnum):
    """Supported policy-construction modes for simulated long entries."""

    FIXED = "fixed"
    CASH_ALLOCATION = "cash_allocation"


class MarketDataConfig(BaseModel):
    """Configuration shared by replaceable market-data providers."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    provider: str = Field(default="mock", min_length=1)
    request_timeout_seconds: int = Field(default=10, gt=0, le=120)

    @field_validator("provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        """Normalize and constrain provider identifiers used in configuration."""

        normalized = value.strip().lower()
        if not _PROVIDER_PATTERN.fullmatch(normalized):
            raise ValueError("provider must be a simple lowercase identifier")
        return normalized


class BacktestConfig(BaseModel):
    """Parameters for deterministic, simulated backtests."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    initial_cash: Decimal = Field(default=Decimal("100000"), gt=0)
    trade_quantity: int = Field(default=10, gt=0)
    commission_bps: Decimal = Field(default=Decimal("0"), ge=0, le=10_000)
    slippage_bps: Decimal = Field(default=Decimal("0"), ge=0, lt=10_000)
    position_sizing_mode: PositionSizingMode = PositionSizingMode.FIXED
    cash_allocation_ratio: Decimal | None = Field(default=None, gt=0, le=1)
    maximum_position_value: Decimal | None = Field(default=None, gt=0)
    minimum_cash_reserve: Decimal | None = Field(default=None, ge=0)
    end_of_test_policy: EndOfTestPolicy = EndOfTestPolicy.HOLD

    @model_validator(mode="after")
    def validate_position_sizing_configuration(self) -> "BacktestConfig":
        """Require only the parameter relevant to the selected sizing mode."""

        if (
            self.position_sizing_mode is PositionSizingMode.CASH_ALLOCATION
            and self.cash_allocation_ratio is None
        ):
            raise ValueError("cash_allocation_ratio is required for cash allocation")
        if (
            self.position_sizing_mode is PositionSizingMode.FIXED
            and self.cash_allocation_ratio is not None
        ):
            raise ValueError("cash_allocation_ratio is only valid for cash allocation")
        return self


class PaperTradingConfig(BaseModel):
    """Safe defaults for future simulated paper-trading sessions."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    initial_cash: Decimal = Field(default=Decimal("100000"), gt=0)


class ApplicationSettings(BaseSettings):
    """Top-level settings loaded from environment variables or a local ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="TRADING_RESEARCH_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    paper_trading: PaperTradingConfig = Field(default_factory=PaperTradingConfig)
