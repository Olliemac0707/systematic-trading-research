"""Tests for the closed typed registry of built-in strategies."""

from dataclasses import FrozenInstanceError

import pytest

from trading_research.strategies import (
    DEFAULT_STRATEGY_NAME,
    DonchianBreakoutParameters,
    DonchianBreakoutStrategy,
    SimpleMovingAverageCrossover,
    SmaCrossoverParameters,
    StrategyName,
    create_strategy,
)


def test_registry_has_stable_closed_names_and_sma_default() -> None:
    """The public choices are explicit and retain the existing default strategy."""

    assert DEFAULT_STRATEGY_NAME is StrategyName.SMA_CROSSOVER
    assert tuple(strategy.value for strategy in StrategyName) == (
        "sma-crossover",
        "donchian-breakout",
    )


def test_registry_builds_each_strategy_from_its_typed_parameters() -> None:
    """Construction requires no module paths, arbitrary imports, or plugin loading."""

    sma = create_strategy("sma-crossover", SmaCrossoverParameters(2, 3))
    donchian = create_strategy(
        StrategyName.DONCHIAN_BREAKOUT,
        DonchianBreakoutParameters(20, 10),
    )

    assert isinstance(sma, SimpleMovingAverageCrossover)
    assert sma.short_window == 2
    assert sma.long_window == 3
    assert donchian == DonchianBreakoutStrategy(entry_window=20, exit_window=10)


def test_registry_rejects_unknown_names_and_mismatched_parameter_models() -> None:
    """Unknown or mixed strategy construction requests fail closed."""

    with pytest.raises(ValueError, match="unsupported strategy"):
        create_strategy("package.module:Strategy", SmaCrossoverParameters(2, 3))
    with pytest.raises(TypeError, match="SmaCrossoverParameters"):
        create_strategy("sma-crossover", DonchianBreakoutParameters(2, 2))
    with pytest.raises(TypeError, match="DonchianBreakoutParameters"):
        create_strategy("donchian-breakout", SmaCrossoverParameters(2, 3))


def test_parameter_models_are_immutable_and_validate_cross_field_rules() -> None:
    """Typed strategy descriptors cannot drift after a run begins."""

    parameters = DonchianBreakoutParameters(20, 10)
    with pytest.raises(FrozenInstanceError):
        parameters.entry_window = 30  # type: ignore[misc]
    strategy = DonchianBreakoutStrategy(20, 10)
    with pytest.raises(FrozenInstanceError):
        strategy.exit_window = 5  # type: ignore[misc]
    with pytest.raises(ValueError, match="smaller"):
        SmaCrossoverParameters(3, 3)
