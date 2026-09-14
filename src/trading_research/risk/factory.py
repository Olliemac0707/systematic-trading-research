"""Construct pure simulation policies from serialisable backtest settings."""

from trading_research.config import BacktestConfig, PositionSizingMode
from trading_research.risk.base import RiskPolicy
from trading_research.risk.policies import (
    AllowAllRiskPolicy,
    CompositeRiskPolicy,
    MaximumPositionValuePolicy,
    MinimumCashReservePolicy,
)
from trading_research.risk.sizing import (
    CashAllocationSizer,
    FixedQuantitySizer,
    PositionSizer,
)


def build_position_sizer(config: BacktestConfig) -> PositionSizer:
    """Construct the configured whole-share sizing policy."""

    if config.position_sizing_mode is PositionSizingMode.FIXED:
        return FixedQuantitySizer(config.trade_quantity)
    allocation_ratio = config.cash_allocation_ratio
    if allocation_ratio is None:
        raise ValueError("cash allocation mode requires an allocation ratio")
    return CashAllocationSizer(allocation_ratio)


def build_risk_policy(config: BacktestConfig) -> RiskPolicy:
    """Construct configured policies in position-cap then reserve order."""

    policies: list[RiskPolicy] = []
    if config.maximum_position_value is not None:
        policies.append(MaximumPositionValuePolicy(config.maximum_position_value))
    if config.minimum_cash_reserve is not None:
        policies.append(MinimumCashReservePolicy(config.minimum_cash_reserve))
    if not policies:
        return AllowAllRiskPolicy()
    return CompositeRiskPolicy(tuple(policies))
