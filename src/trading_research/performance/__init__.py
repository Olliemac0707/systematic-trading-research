"""Pure performance metrics for reconciled backtest results."""

from trading_research.performance.exposure import (
    EXPOSURE_DECIMAL_PRECISION,
    ExposureObservation,
    ExposureStatistics,
    calculate_exposure_statistics,
    derive_exposure_observations,
)
from trading_research.performance.metrics import (
    PerformanceSummary,
    calculate_performance,
)
from trading_research.performance.returns import (
    STATISTICAL_DECIMAL_PRECISION,
    PeriodReturn,
    ReturnSeries,
    RiskAdjustedMetrics,
    RiskMetricSettings,
    calculate_risk_adjusted_metrics,
    construct_return_series,
)
from trading_research.performance.trades import (
    ClosedTrade,
    TradeStatistics,
    calculate_trade_statistics,
    reconstruct_closed_trades,
)

__all__ = [
    "ClosedTrade",
    "EXPOSURE_DECIMAL_PRECISION",
    "ExposureObservation",
    "ExposureStatistics",
    "PerformanceSummary",
    "PeriodReturn",
    "ReturnSeries",
    "RiskAdjustedMetrics",
    "RiskMetricSettings",
    "STATISTICAL_DECIMAL_PRECISION",
    "TradeStatistics",
    "calculate_exposure_statistics",
    "calculate_performance",
    "calculate_risk_adjusted_metrics",
    "calculate_trade_statistics",
    "construct_return_series",
    "derive_exposure_observations",
    "reconstruct_closed_trades",
]
