"""Closed, typed construction registry for built-in educational strategies."""

from dataclasses import dataclass
from enum import StrEnum

from trading_research.strategies.base import Strategy
from trading_research.strategies.donchian import DonchianBreakoutStrategy
from trading_research.strategies.moving_average import SimpleMovingAverageCrossover


class StrategyName(StrEnum):
    """Stable identifiers for locally implemented strategy choices."""

    SMA_CROSSOVER = "sma-crossover"
    DONCHIAN_BREAKOUT = "donchian-breakout"


DEFAULT_STRATEGY_NAME = StrategyName.SMA_CROSSOVER


@dataclass(frozen=True, slots=True)
class SmaCrossoverParameters:
    """Validated construction parameters for the SMA crossover strategy."""

    fast_window: int
    slow_window: int

    def __post_init__(self) -> None:
        """Require positive ordered whole-number moving-average windows."""

        _validate_positive_integer(self.fast_window, "fast_window")
        _validate_positive_integer(self.slow_window, "slow_window")
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be smaller than slow_window")


@dataclass(frozen=True, slots=True)
class DonchianBreakoutParameters:
    """Validated construction parameters for independent Donchian channels."""

    entry_window: int
    exit_window: int

    def __post_init__(self) -> None:
        """Require positive whole-number entry and exit windows."""

        _validate_positive_integer(self.entry_window, "entry_window")
        _validate_positive_integer(self.exit_window, "exit_window")


StrategyParameters = SmaCrossoverParameters | DonchianBreakoutParameters


def create_strategy(
    strategy_name: str | StrategyName,
    parameters: StrategyParameters,
) -> Strategy:
    """Construct one built-in strategy without dynamic imports or plugins."""

    if not isinstance(strategy_name, (str, StrategyName)):
        raise TypeError("strategy_name must be a string or StrategyName")
    try:
        selected_name = StrategyName(strategy_name)
    except ValueError as exc:
        raise ValueError(f"unsupported strategy: {strategy_name}") from exc

    if selected_name is StrategyName.SMA_CROSSOVER:
        if not isinstance(parameters, SmaCrossoverParameters):
            raise TypeError("sma-crossover requires SmaCrossoverParameters")
        return SimpleMovingAverageCrossover(
            short_window=parameters.fast_window,
            long_window=parameters.slow_window,
        )
    if not isinstance(parameters, DonchianBreakoutParameters):
        raise TypeError("donchian-breakout requires DonchianBreakoutParameters")
    return DonchianBreakoutStrategy(
        entry_window=parameters.entry_window,
        exit_window=parameters.exit_window,
    )


def _validate_positive_integer(value: int, name: str) -> None:
    """Reject booleans, non-integers, and non-positive strategy parameters."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
