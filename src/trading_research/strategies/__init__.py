"""Strategy contracts and educational examples."""

from trading_research.strategies.base import Strategy
from trading_research.strategies.donchian import DonchianBreakoutStrategy
from trading_research.strategies.moving_average import SimpleMovingAverageCrossover
from trading_research.strategies.registry import (
    DEFAULT_STRATEGY_NAME,
    DonchianBreakoutParameters,
    SmaCrossoverParameters,
    StrategyName,
    StrategyParameters,
    create_strategy,
)

__all__ = [
    "DEFAULT_STRATEGY_NAME",
    "DonchianBreakoutParameters",
    "DonchianBreakoutStrategy",
    "SimpleMovingAverageCrossover",
    "SmaCrossoverParameters",
    "Strategy",
    "StrategyName",
    "StrategyParameters",
    "create_strategy",
]
