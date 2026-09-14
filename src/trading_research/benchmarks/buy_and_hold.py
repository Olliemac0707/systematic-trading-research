"""Next-bar-executable buy-and-hold benchmark using the normal engine."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading_research.backtesting.engine import SimpleBacktestEngine
from trading_research.config import BacktestConfig, PositionSizingMode
from trading_research.errors import BacktestError
from trading_research.models import (
    BacktestResult,
    EndOfTestPolicy,
    MarketBar,
    PreTradeDecision,
    SignalAction,
    StrategySignal,
    normalize_symbol,
)
from trading_research.performance import (
    ExposureStatistics,
    PerformanceSummary,
    ReturnSeries,
    RiskAdjustedMetrics,
    RiskMetricSettings,
    TradeStatistics,
    calculate_exposure_statistics,
    calculate_performance,
    calculate_risk_adjusted_metrics,
    calculate_trade_statistics,
    construct_return_series,
)
from trading_research.risk import PositionSizer, RiskPolicy
from trading_research.strategies import Strategy

BUY_AND_HOLD_BENCHMARK_NAME = "buy-and-hold"
_ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class BenchmarkAssumptions:
    """Serializable snapshot of every material benchmark-run assumption."""

    symbol: str
    actual_start: datetime
    actual_end: datetime
    bar_count: int
    starting_cash: Decimal
    commission_bps: Decimal
    slippage_bps: Decimal
    position_sizing_mode: PositionSizingMode
    trade_quantity: int | None
    cash_allocation_ratio: Decimal | None
    maximum_position_value: Decimal | None
    minimum_cash_reserve: Decimal | None
    end_of_test_policy: EndOfTestPolicy
    risk_metric_settings: RiskMetricSettings | None

    def __post_init__(self) -> None:
        """Validate exact values and the period represented by the snapshot."""

        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        for name, timestamp in (
            ("actual_start", self.actual_start),
            ("actual_end", self.actual_end),
        ):
            if not isinstance(timestamp, datetime):
                raise TypeError(f"{name} must be a datetime")
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
        if self.actual_start >= self.actual_end:
            raise ValueError("benchmark period must contain at least two timestamps")
        if isinstance(self.bar_count, bool) or not isinstance(self.bar_count, int):
            raise TypeError("bar_count must be an integer")
        if self.bar_count < 2:
            raise ValueError("benchmark requires at least two market bars")
        for name, value in (
            ("starting_cash", self.starting_cash),
            ("commission_bps", self.commission_bps),
            ("slippage_bps", self.slippage_bps),
        ):
            _require_finite_decimal(value, name)
        if self.starting_cash <= _ZERO:
            raise ValueError("starting_cash must be positive")
        if not (_ZERO <= self.commission_bps <= Decimal("10000")):
            raise ValueError("commission_bps must be between 0 and 10000")
        if not (_ZERO <= self.slippage_bps < Decimal("10000")):
            raise ValueError("slippage_bps must be between 0 and less than 10000")
        if not isinstance(self.position_sizing_mode, PositionSizingMode):
            raise TypeError("position_sizing_mode must be PositionSizingMode")
        for optional_name, optional_value in (
            ("cash_allocation_ratio", self.cash_allocation_ratio),
            ("maximum_position_value", self.maximum_position_value),
            ("minimum_cash_reserve", self.minimum_cash_reserve),
        ):
            if optional_value is not None:
                _require_finite_decimal(optional_value, optional_name)
        if self.position_sizing_mode is PositionSizingMode.FIXED:
            if isinstance(self.trade_quantity, bool) or not isinstance(
                self.trade_quantity, int
            ):
                raise TypeError("fixed sizing requires an integer trade_quantity")
            if self.trade_quantity <= 0:
                raise ValueError("trade_quantity must be positive")
            if self.cash_allocation_ratio is not None:
                raise ValueError("cash_allocation_ratio is only valid for allocation")
        else:
            if self.trade_quantity is not None:
                raise ValueError("trade_quantity is only valid for fixed sizing")
            if self.cash_allocation_ratio is None:
                raise ValueError("cash_allocation_ratio is required for allocation")
            if not (_ZERO < self.cash_allocation_ratio <= Decimal("1")):
                raise ValueError("cash_allocation_ratio must be greater than 0 and at most 1")
        if (
            self.maximum_position_value is not None
            and self.maximum_position_value <= _ZERO
        ):
            raise ValueError("maximum_position_value must be positive")
        if self.minimum_cash_reserve is not None and self.minimum_cash_reserve < _ZERO:
            raise ValueError("minimum_cash_reserve must be non-negative")
        if not isinstance(self.end_of_test_policy, EndOfTestPolicy):
            raise TypeError("end_of_test_policy must be EndOfTestPolicy")
        if self.risk_metric_settings is not None and not isinstance(
            self.risk_metric_settings, RiskMetricSettings
        ):
            raise TypeError("risk_metric_settings must be RiskMetricSettings or None")


@dataclass(frozen=True, slots=True)
class BenchmarkAnalysis:
    """Immutable benchmark result and all existing derived analyses."""

    name: str
    assumptions: BenchmarkAssumptions
    result: BacktestResult
    performance: PerformanceSummary
    trade_statistics: TradeStatistics
    return_series: ReturnSeries
    risk_adjusted_metrics: RiskAdjustedMetrics | None
    exposure_statistics: ExposureStatistics
    pre_trade_decisions: tuple[PreTradeDecision, ...]

    def __post_init__(self) -> None:
        """Require the aggregate to contain authoritative public calculations."""

        if self.name != BUY_AND_HOLD_BENCHMARK_NAME:
            raise ValueError(f"name must be {BUY_AND_HOLD_BENCHMARK_NAME!r}")
        if not isinstance(self.assumptions, BenchmarkAssumptions):
            raise TypeError("assumptions must be BenchmarkAssumptions")
        if not isinstance(self.result, BacktestResult):
            raise TypeError("result must be BacktestResult")
        if self.result.initial_cash != self.assumptions.starting_cash:
            raise ValueError("benchmark starting cash must match its assumptions")
        if self.result.end_of_test_policy is not self.assumptions.end_of_test_policy:
            raise ValueError("benchmark end policy must match its assumptions")
        if len(self.result.equity_curve) != self.assumptions.bar_count:
            raise ValueError("benchmark equity observations must match its assumptions")
        if self.result.equity_curve[0].timestamp != self.assumptions.actual_start:
            raise ValueError("benchmark start must match its assumptions")
        if self.result.equity_curve[-1].timestamp != self.assumptions.actual_end:
            raise ValueError("benchmark end must match its assumptions")
        if self.result.trades and any(
            trade.symbol != self.assumptions.symbol for trade in self.result.trades
        ):
            raise ValueError("benchmark trade symbol must match its assumptions")
        if self.performance != calculate_performance(self.result):
            raise ValueError("benchmark performance is not authoritative")
        if self.trade_statistics != calculate_trade_statistics(self.result):
            raise ValueError("benchmark trade statistics are not authoritative")
        if self.return_series != construct_return_series(self.result):
            raise ValueError("benchmark return series is not authoritative")
        if self.exposure_statistics != calculate_exposure_statistics(self.result):
            raise ValueError("benchmark exposure statistics are not authoritative")
        settings = self.assumptions.risk_metric_settings
        expected_risk_metrics = (
            None
            if settings is None
            else calculate_risk_adjusted_metrics(self.result, settings)
        )
        if self.risk_adjusted_metrics != expected_risk_metrics:
            raise ValueError("benchmark risk metrics are not authoritative")
        if self.pre_trade_decisions != self.result.pre_trade_decisions:
            raise ValueError("benchmark decisions must match its backtest result")


class BuyAndHoldBenchmarkStrategy(Strategy):
    """Emit one buy after the first close and never emit a normal exit."""

    def generate_signals(self, bars: Sequence[MarketBar]) -> tuple[StrategySignal, ...]:
        """Return the sole signal that the engine can execute at bar two's open."""

        snapshot = tuple(bars)
        if len(snapshot) < 2:
            raise BacktestError("buy-and-hold benchmark requires at least two market bars")
        first_bar = snapshot[0]
        return (
            StrategySignal(
                symbol=first_bar.symbol,
                timestamp=first_bar.timestamp,
                action=SignalAction.BUY,
                reason="buy-and-hold entry after the first available close",
            ),
        )


def run_buy_and_hold_benchmark(
    *,
    bars: Sequence[MarketBar],
    backtest_configuration: BacktestConfig,
    position_sizer: PositionSizer,
    risk_policy: RiskPolicy,
    risk_metric_settings: RiskMetricSettings | None = None,
) -> BenchmarkAnalysis:
    """Run a capital-matched benchmark entirely through the existing engine."""

    snapshot = tuple(bars)
    assumptions = benchmark_assumptions(
        snapshot,
        backtest_configuration,
        risk_metric_settings,
    )
    result = SimpleBacktestEngine().run(
        snapshot,
        BuyAndHoldBenchmarkStrategy(),
        backtest_configuration,
        position_sizer=position_sizer,
        risk_policy=risk_policy,
    )
    return BenchmarkAnalysis(
        name=BUY_AND_HOLD_BENCHMARK_NAME,
        assumptions=assumptions,
        result=result,
        performance=calculate_performance(result),
        trade_statistics=calculate_trade_statistics(result),
        return_series=construct_return_series(result),
        risk_adjusted_metrics=(
            None
            if risk_metric_settings is None
            else calculate_risk_adjusted_metrics(result, risk_metric_settings)
        ),
        exposure_statistics=calculate_exposure_statistics(result),
        pre_trade_decisions=result.pre_trade_decisions,
    )


def benchmark_assumptions(
    bars: Sequence[MarketBar],
    config: BacktestConfig,
    risk_metric_settings: RiskMetricSettings | None,
) -> BenchmarkAssumptions:
    """Snapshot shared bars and configuration for mismatch prevention."""

    snapshot = tuple(bars)
    if len(snapshot) < 2:
        raise BacktestError("buy-and-hold benchmark requires at least two market bars")
    if any(not isinstance(bar, MarketBar) for bar in snapshot):
        raise TypeError("bars must contain MarketBar values")
    if len({bar.symbol for bar in snapshot}) != 1:
        raise BacktestError("benchmark bars must contain exactly one symbol")
    if any(
        current.timestamp <= previous.timestamp
        for previous, current in zip(snapshot, snapshot[1:], strict=False)
    ):
        raise BacktestError("benchmark bars must be strictly chronological")
    if not isinstance(config, BacktestConfig):
        raise TypeError("backtest_configuration must be BacktestConfig")
    if risk_metric_settings is not None and not isinstance(
        risk_metric_settings, RiskMetricSettings
    ):
        raise TypeError("risk_metric_settings must be RiskMetricSettings or None")
    return BenchmarkAssumptions(
        symbol=snapshot[0].symbol,
        actual_start=snapshot[0].timestamp,
        actual_end=snapshot[-1].timestamp,
        bar_count=len(snapshot),
        starting_cash=config.initial_cash,
        commission_bps=config.commission_bps,
        slippage_bps=config.slippage_bps,
        position_sizing_mode=config.position_sizing_mode,
        trade_quantity=(
            config.trade_quantity
            if config.position_sizing_mode is PositionSizingMode.FIXED
            else None
        ),
        cash_allocation_ratio=config.cash_allocation_ratio,
        maximum_position_value=config.maximum_position_value,
        minimum_cash_reserve=config.minimum_cash_reserve,
        end_of_test_policy=config.end_of_test_policy,
        risk_metric_settings=risk_metric_settings,
    )


def _require_finite_decimal(value: Decimal, name: str) -> None:
    """Require exact finite Decimal assumptions."""

    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")
