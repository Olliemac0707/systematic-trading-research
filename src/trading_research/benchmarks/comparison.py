"""Exact comparisons between a strategy and its capital-matched benchmark."""

from collections.abc import Sequence
from dataclasses import dataclass, fields
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext

from trading_research.benchmarks.buy_and_hold import (
    BenchmarkAnalysis,
    benchmark_assumptions,
)
from trading_research.config import BacktestConfig
from trading_research.models import BacktestResult, MarketBar
from trading_research.performance import (
    RiskMetricSettings,
    calculate_exposure_statistics,
    calculate_performance,
    calculate_risk_adjusted_metrics,
)

COMPARISON_DECIMAL_PRECISION = 34


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    """Immutable Decimal-only strategy-minus-benchmark statistics."""

    benchmark_name: str
    strategy_ending_equity: Decimal
    benchmark_ending_equity: Decimal
    ending_equity_difference: Decimal
    strategy_total_return: Decimal
    benchmark_total_return: Decimal
    excess_return: Decimal
    strategy_maximum_absolute_drawdown: Decimal
    benchmark_maximum_absolute_drawdown: Decimal
    absolute_drawdown_difference: Decimal
    strategy_maximum_percentage_drawdown: Decimal
    benchmark_maximum_percentage_drawdown: Decimal
    maximum_drawdown_improvement: Decimal
    strategy_periodic_volatility: Decimal | None
    benchmark_periodic_volatility: Decimal | None
    periodic_volatility_difference: Decimal | None
    strategy_annualised_volatility: Decimal | None
    benchmark_annualised_volatility: Decimal | None
    annualised_volatility_difference: Decimal | None
    strategy_periodic_sharpe_ratio: Decimal | None
    benchmark_periodic_sharpe_ratio: Decimal | None
    periodic_sharpe_difference: Decimal | None
    strategy_annualised_sharpe_ratio: Decimal | None
    benchmark_annualised_sharpe_ratio: Decimal | None
    annualised_sharpe_difference: Decimal | None
    strategy_periodic_sortino_ratio: Decimal | None
    benchmark_periodic_sortino_ratio: Decimal | None
    periodic_sortino_difference: Decimal | None
    strategy_annualised_sortino_ratio: Decimal | None
    benchmark_annualised_sortino_ratio: Decimal | None
    annualised_sortino_difference: Decimal | None
    strategy_time_in_market_ratio: Decimal | None
    benchmark_time_in_market_ratio: Decimal | None
    time_in_market_difference: Decimal | None
    strategy_total_commission: Decimal
    benchmark_total_commission: Decimal
    strategy_total_slippage_cost: Decimal
    benchmark_total_slippage_cost: Decimal

    def __post_init__(self) -> None:
        """Validate Decimal storage and every direct difference relationship."""

        if not isinstance(self.benchmark_name, str) or not self.benchmark_name:
            raise ValueError("benchmark_name must be non-empty text")
        for item in fields(self):
            if item.name == "benchmark_name":
                continue
            value = getattr(self, item.name)
            if value is not None and not isinstance(value, Decimal):
                raise TypeError(f"{item.name} must be a Decimal or None")
            if isinstance(value, Decimal) and not value.is_finite():
                raise ValueError(f"{item.name} must be finite")
        _require_difference(
            self.strategy_ending_equity,
            self.benchmark_ending_equity,
            self.ending_equity_difference,
            "ending_equity_difference",
        )
        _require_difference(
            self.strategy_total_return,
            self.benchmark_total_return,
            self.excess_return,
            "excess_return",
        )
        _require_difference(
            self.strategy_maximum_absolute_drawdown,
            self.benchmark_maximum_absolute_drawdown,
            self.absolute_drawdown_difference,
            "absolute_drawdown_difference",
        )
        _require_difference(
            self.strategy_maximum_percentage_drawdown,
            self.benchmark_maximum_percentage_drawdown,
            self.maximum_drawdown_improvement,
            "maximum_drawdown_improvement",
        )
        for left, right, difference, name in (
            (
                self.strategy_periodic_volatility,
                self.benchmark_periodic_volatility,
                self.periodic_volatility_difference,
                "periodic_volatility_difference",
            ),
            (
                self.strategy_annualised_volatility,
                self.benchmark_annualised_volatility,
                self.annualised_volatility_difference,
                "annualised_volatility_difference",
            ),
            (
                self.strategy_periodic_sharpe_ratio,
                self.benchmark_periodic_sharpe_ratio,
                self.periodic_sharpe_difference,
                "periodic_sharpe_difference",
            ),
            (
                self.strategy_annualised_sharpe_ratio,
                self.benchmark_annualised_sharpe_ratio,
                self.annualised_sharpe_difference,
                "annualised_sharpe_difference",
            ),
            (
                self.strategy_periodic_sortino_ratio,
                self.benchmark_periodic_sortino_ratio,
                self.periodic_sortino_difference,
                "periodic_sortino_difference",
            ),
            (
                self.strategy_annualised_sortino_ratio,
                self.benchmark_annualised_sortino_ratio,
                self.annualised_sortino_difference,
                "annualised_sortino_difference",
            ),
            (
                self.strategy_time_in_market_ratio,
                self.benchmark_time_in_market_ratio,
                self.time_in_market_difference,
                "time_in_market_difference",
            ),
        ):
            _require_optional_difference(left, right, difference, name)


def calculate_benchmark_comparison(
    strategy_result: BacktestResult,
    benchmark: BenchmarkAnalysis,
    *,
    bars: Sequence[MarketBar],
    backtest_configuration: BacktestConfig,
    risk_metric_settings: RiskMetricSettings | None = None,
) -> BenchmarkComparison:
    """Compare runs after validating shared period and material assumptions."""

    if not isinstance(strategy_result, BacktestResult):
        raise TypeError("strategy_result must be BacktestResult")
    if not isinstance(benchmark, BenchmarkAnalysis):
        raise TypeError("benchmark must be BenchmarkAnalysis")
    shared_assumptions = benchmark_assumptions(
        bars,
        backtest_configuration,
        risk_metric_settings,
    )
    if benchmark.assumptions != shared_assumptions:
        raise ValueError("strategy and benchmark assumptions must be identical")
    if strategy_result.initial_cash != shared_assumptions.starting_cash:
        raise ValueError("strategy and benchmark starting cash must match")
    if strategy_result.end_of_test_policy is not shared_assumptions.end_of_test_policy:
        raise ValueError("strategy and benchmark end policy must match")
    if len(strategy_result.equity_curve) != shared_assumptions.bar_count:
        raise ValueError("strategy and benchmark observation periods must match")
    if (
        strategy_result.equity_curve[0].timestamp != shared_assumptions.actual_start
        or strategy_result.equity_curve[-1].timestamp != shared_assumptions.actual_end
    ):
        raise ValueError("strategy and benchmark actual periods must match")
    if strategy_result.trades and any(
        trade.symbol != shared_assumptions.symbol for trade in strategy_result.trades
    ):
        raise ValueError("strategy and benchmark symbols must match")

    strategy_performance = calculate_performance(strategy_result)
    strategy_exposure = calculate_exposure_statistics(strategy_result)
    strategy_risk = (
        None
        if risk_metric_settings is None
        else calculate_risk_adjusted_metrics(strategy_result, risk_metric_settings)
    )
    benchmark_risk = benchmark.risk_adjusted_metrics

    def metric(name: str, *, strategy: bool) -> Decimal | None:
        selected = strategy_risk if strategy else benchmark_risk
        if selected is None:
            return None
        value = getattr(selected, name)
        if value is not None and not isinstance(value, Decimal):
            raise TypeError(f"risk metric {name} must be Decimal or None")
        return value

    strategy_periodic_volatility = metric("periodic_volatility", strategy=True)
    benchmark_periodic_volatility = metric("periodic_volatility", strategy=False)
    strategy_annualised_volatility = metric("annualised_volatility", strategy=True)
    benchmark_annualised_volatility = metric("annualised_volatility", strategy=False)
    strategy_periodic_sharpe = metric("periodic_sharpe_ratio", strategy=True)
    benchmark_periodic_sharpe = metric("periodic_sharpe_ratio", strategy=False)
    strategy_annualised_sharpe = metric("annualised_sharpe_ratio", strategy=True)
    benchmark_annualised_sharpe = metric("annualised_sharpe_ratio", strategy=False)
    strategy_periodic_sortino = metric("periodic_sortino_ratio", strategy=True)
    benchmark_periodic_sortino = metric("periodic_sortino_ratio", strategy=False)
    strategy_annualised_sortino = metric("annualised_sortino_ratio", strategy=True)
    benchmark_annualised_sortino = metric("annualised_sortino_ratio", strategy=False)
    return BenchmarkComparison(
        benchmark_name=benchmark.name,
        strategy_ending_equity=strategy_performance.ending_equity,
        benchmark_ending_equity=benchmark.performance.ending_equity,
        ending_equity_difference=_difference(
            strategy_performance.ending_equity,
            benchmark.performance.ending_equity,
        ),
        strategy_total_return=strategy_performance.total_return,
        benchmark_total_return=benchmark.performance.total_return,
        excess_return=_difference(
            strategy_performance.total_return,
            benchmark.performance.total_return,
        ),
        strategy_maximum_absolute_drawdown=(
            strategy_performance.maximum_absolute_drawdown
        ),
        benchmark_maximum_absolute_drawdown=(
            benchmark.performance.maximum_absolute_drawdown
        ),
        absolute_drawdown_difference=_difference(
            strategy_performance.maximum_absolute_drawdown,
            benchmark.performance.maximum_absolute_drawdown,
        ),
        strategy_maximum_percentage_drawdown=(
            strategy_performance.maximum_percentage_drawdown
        ),
        benchmark_maximum_percentage_drawdown=(
            benchmark.performance.maximum_percentage_drawdown
        ),
        maximum_drawdown_improvement=_difference(
            strategy_performance.maximum_percentage_drawdown,
            benchmark.performance.maximum_percentage_drawdown,
        ),
        strategy_periodic_volatility=strategy_periodic_volatility,
        benchmark_periodic_volatility=benchmark_periodic_volatility,
        periodic_volatility_difference=_optional_difference(
            strategy_periodic_volatility, benchmark_periodic_volatility
        ),
        strategy_annualised_volatility=strategy_annualised_volatility,
        benchmark_annualised_volatility=benchmark_annualised_volatility,
        annualised_volatility_difference=_optional_difference(
            strategy_annualised_volatility, benchmark_annualised_volatility
        ),
        strategy_periodic_sharpe_ratio=strategy_periodic_sharpe,
        benchmark_periodic_sharpe_ratio=benchmark_periodic_sharpe,
        periodic_sharpe_difference=_optional_difference(
            strategy_periodic_sharpe, benchmark_periodic_sharpe
        ),
        strategy_annualised_sharpe_ratio=strategy_annualised_sharpe,
        benchmark_annualised_sharpe_ratio=benchmark_annualised_sharpe,
        annualised_sharpe_difference=_optional_difference(
            strategy_annualised_sharpe, benchmark_annualised_sharpe
        ),
        strategy_periodic_sortino_ratio=strategy_periodic_sortino,
        benchmark_periodic_sortino_ratio=benchmark_periodic_sortino,
        periodic_sortino_difference=_optional_difference(
            strategy_periodic_sortino, benchmark_periodic_sortino
        ),
        strategy_annualised_sortino_ratio=strategy_annualised_sortino,
        benchmark_annualised_sortino_ratio=benchmark_annualised_sortino,
        annualised_sortino_difference=_optional_difference(
            strategy_annualised_sortino, benchmark_annualised_sortino
        ),
        strategy_time_in_market_ratio=strategy_exposure.time_in_market_ratio,
        benchmark_time_in_market_ratio=(
            benchmark.exposure_statistics.time_in_market_ratio
        ),
        time_in_market_difference=_optional_difference(
            strategy_exposure.time_in_market_ratio,
            benchmark.exposure_statistics.time_in_market_ratio,
        ),
        strategy_total_commission=strategy_performance.total_commission,
        benchmark_total_commission=benchmark.performance.total_commission,
        strategy_total_slippage_cost=(
            strategy_performance.total_adverse_slippage_cost
        ),
        benchmark_total_slippage_cost=(
            benchmark.performance.total_adverse_slippage_cost
        ),
    )


def _optional_difference(
    left: Decimal | None,
    right: Decimal | None,
) -> Decimal | None:
    """Subtract only when both optional metrics are available."""

    if left is None or right is None:
        return None
    return _difference(left, right)


def _difference(left: Decimal, right: Decimal) -> Decimal:
    """Subtract in an isolated fixed-precision context for determinism."""

    with localcontext(
        Context(prec=COMPARISON_DECIMAL_PRECISION, rounding=ROUND_HALF_EVEN)
    ):
        return left - right


def _require_difference(
    left: Decimal,
    right: Decimal,
    difference: Decimal,
    name: str,
) -> None:
    """Require one exact non-optional subtraction relationship."""

    if difference != _difference(left, right):
        raise ValueError(f"{name} must equal strategy minus benchmark")


def _require_optional_difference(
    left: Decimal | None,
    right: Decimal | None,
    difference: Decimal | None,
    name: str,
) -> None:
    """Require missingness propagation or one exact subtraction."""

    if left is None or right is None:
        if difference is not None:
            raise ValueError(f"{name} must be None when either metric is unavailable")
    elif difference != _difference(left, right):
        raise ValueError(f"{name} must equal strategy minus benchmark")
