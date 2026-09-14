"""Tests for validated configuration models."""

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_research.config import (
    ApplicationSettings,
    BacktestConfig,
    EndOfTestPolicy,
    MarketDataConfig,
    PositionSizingMode,
)


def test_application_settings_have_safe_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default settings use mock data and simulation cash."""

    monkeypatch.chdir(tmp_path)
    settings = ApplicationSettings()

    assert settings.market_data.provider == "mock"
    assert settings.backtest.initial_cash == Decimal("100000")
    assert settings.paper_trading.initial_cash == Decimal("100000")


def test_backtest_config_rejects_non_positive_cash() -> None:
    """A simulation cannot start with zero capital."""

    with pytest.raises(ValidationError):
        BacktestConfig(initial_cash=Decimal("0"))


def test_application_settings_load_nested_environment_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nested environment variables override only their target setting."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRADING_RESEARCH_BACKTEST__INITIAL_CASH", "2500.50")

    settings = ApplicationSettings()

    assert settings.backtest.initial_cash == Decimal("2500.50")
    assert settings.backtest.trade_quantity == 10


def test_market_data_provider_identifier_is_normalized() -> None:
    """Provider identifiers are predictable and safe for adapter lookup."""

    config = MarketDataConfig(provider=" Mock-Data ")

    assert config.provider == "mock-data"


@pytest.mark.parametrize("provider", ["", "   ", "../provider", "provider\nname"])
def test_market_data_provider_rejects_unsafe_identifiers(provider: str) -> None:
    """Provider names cannot contain paths, whitespace, or control characters."""

    with pytest.raises(ValidationError):
        MarketDataConfig(provider=provider)


def test_backtest_config_rejects_non_finite_values() -> None:
    """NaN values cannot enter portfolio arithmetic."""

    with pytest.raises(ValidationError):
        BacktestConfig(initial_cash=Decimal("NaN"))


def test_backtest_config_preserves_fixed_sizing_defaults() -> None:
    """Existing configurations remain fixed-size with quantity ten."""

    config = BacktestConfig()

    assert config.position_sizing_mode is PositionSizingMode.FIXED
    assert config.trade_quantity == 10
    assert config.cash_allocation_ratio is None
    assert config.maximum_position_value is None
    assert config.minimum_cash_reserve is None
    assert config.end_of_test_policy is EndOfTestPolicy.HOLD


def test_backtest_config_accepts_cash_allocation_and_risk_settings() -> None:
    """Serialisable settings describe policy construction without policy objects."""

    config = BacktestConfig(
        position_sizing_mode=PositionSizingMode.CASH_ALLOCATION,
        cash_allocation_ratio=Decimal("0.25"),
        maximum_position_value=Decimal("2500"),
        minimum_cash_reserve=Decimal("1000"),
    )

    assert config.cash_allocation_ratio == Decimal("0.25")
    assert config.maximum_position_value == Decimal("2500")
    assert config.minimum_cash_reserve == Decimal("1000")


def test_backtest_config_accepts_and_serialises_liquidation_policy() -> None:
    """The validated final-position policy remains suitable for configuration."""

    config = BacktestConfig(end_of_test_policy=EndOfTestPolicy.LIQUIDATE)

    assert config.end_of_test_policy is EndOfTestPolicy.LIQUIDATE
    assert config.model_dump(mode="json")["end_of_test_policy"] == "liquidate"


def test_backtest_config_rejects_incompatible_sizing_settings() -> None:
    """Each mode requires only its relevant sizing parameter."""

    with pytest.raises(ValidationError, match="required"):
        BacktestConfig(position_sizing_mode=PositionSizingMode.CASH_ALLOCATION)
    with pytest.raises(ValidationError, match="only valid"):
        BacktestConfig(cash_allocation_ratio=Decimal("0.25"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cash_allocation_ratio", Decimal("0")),
        ("cash_allocation_ratio", Decimal("1.01")),
        ("maximum_position_value", Decimal("0")),
        ("minimum_cash_reserve", Decimal("-1")),
    ],
)
def test_backtest_config_rejects_invalid_sizing_and_risk_values(
    field: str,
    value: Decimal,
) -> None:
    """Sizing ratios and risk amounts retain their declared numeric bounds."""

    values: dict[str, object] = {field: value}
    if field == "cash_allocation_ratio":
        values["position_sizing_mode"] = PositionSizingMode.CASH_ALLOCATION
    with pytest.raises(ValidationError):
        BacktestConfig(**values)  # type: ignore[arg-type]
