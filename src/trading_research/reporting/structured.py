"""Immutable structured reports assembled from authoritative calculations."""

from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext

from trading_research.benchmarks import (
    BenchmarkAnalysis,
    BenchmarkComparison,
)
from trading_research.benchmarks.comparison import COMPARISON_DECIMAL_PRECISION
from trading_research.models import BacktestResult, EquityPoint, PreTradeDecision
from trading_research.performance import (
    ClosedTrade,
    ExposureStatistics,
    PerformanceSummary,
    ReturnSeries,
    RiskAdjustedMetrics,
    TradeStatistics,
    calculate_exposure_statistics,
    calculate_performance,
    calculate_risk_adjusted_metrics,
    calculate_trade_statistics,
    construct_return_series,
    derive_exposure_observations,
    reconstruct_closed_trades,
)
from trading_research.reporting.metadata import BacktestRunMetadata

STRUCTURED_REPORT_SCHEMA_VERSION = "1.3"
_ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class EquityCurveExportRow:
    """One export-ready observation derived without changing the equity curve."""

    timestamp: datetime
    cash: Decimal
    position_quantity: int
    position_market_value: Decimal
    equity: Decimal
    running_peak_equity: Decimal
    absolute_drawdown: Decimal
    percentage_drawdown: Decimal
    period_return: Decimal
    capital_utilisation_ratio: Decimal
    cash_ratio: Decimal
    invested: bool


@dataclass(frozen=True, slots=True)
class BenchmarkEquityComparisonRow:
    """One exactly aligned strategy-versus-benchmark equity observation."""

    timestamp: datetime
    strategy_cash: Decimal
    strategy_position_market_value: Decimal
    strategy_equity: Decimal
    strategy_drawdown: Decimal
    benchmark_cash: Decimal
    benchmark_position_market_value: Decimal
    benchmark_equity: Decimal
    benchmark_drawdown: Decimal
    equity_difference: Decimal


@dataclass(frozen=True, slots=True)
class StructuredBacktestReport:
    """Complete immutable machine-readable result for one reproducible run."""

    schema_version: str
    metadata: BacktestRunMetadata
    backtest_result: BacktestResult
    performance: PerformanceSummary
    trade_statistics: TradeStatistics
    closed_trades: tuple[ClosedTrade, ...]
    return_series: ReturnSeries
    risk_adjusted_metrics: RiskAdjustedMetrics | None
    exposure_statistics: ExposureStatistics
    pre_trade_decisions: tuple[PreTradeDecision, ...]
    benchmark: BenchmarkAnalysis | None = None
    benchmark_comparison: BenchmarkComparison | None = None

    def __post_init__(self) -> None:
        """Validate component types and their authoritative run relationships."""

        if self.schema_version != STRUCTURED_REPORT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {STRUCTURED_REPORT_SCHEMA_VERSION!r}"
            )
        if not isinstance(self.metadata, BacktestRunMetadata):
            raise TypeError("metadata must be BacktestRunMetadata")
        if not isinstance(self.backtest_result, BacktestResult):
            raise TypeError("backtest_result must be BacktestResult")
        if not isinstance(self.performance, PerformanceSummary):
            raise TypeError("performance must be PerformanceSummary")
        if not isinstance(self.trade_statistics, TradeStatistics):
            raise TypeError("trade_statistics must be TradeStatistics")
        if not isinstance(self.closed_trades, tuple) or any(
            not isinstance(trade, ClosedTrade) for trade in self.closed_trades
        ):
            raise TypeError("closed_trades must be a tuple of ClosedTrade values")
        if not isinstance(self.return_series, ReturnSeries):
            raise TypeError("return_series must be ReturnSeries")
        if self.risk_adjusted_metrics is not None and not isinstance(
            self.risk_adjusted_metrics,
            RiskAdjustedMetrics,
        ):
            raise TypeError("risk_adjusted_metrics must be RiskAdjustedMetrics or None")
        if not isinstance(self.exposure_statistics, ExposureStatistics):
            raise TypeError("exposure_statistics must be ExposureStatistics")
        if not isinstance(self.pre_trade_decisions, tuple) or any(
            not isinstance(decision, PreTradeDecision)
            for decision in self.pre_trade_decisions
        ):
            raise TypeError(
                "pre_trade_decisions must be a tuple of PreTradeDecision values"
            )

        result = self.backtest_result
        metadata = self.metadata
        if metadata.dataset.actual_start != result.equity_curve[0].timestamp:
            raise ValueError("metadata actual_start must match the first equity point")
        if metadata.dataset.actual_end != result.equity_curve[-1].timestamp:
            raise ValueError("metadata actual_end must match the final equity point")
        if metadata.dataset.bar_count != len(result.equity_curve):
            raise ValueError("metadata bar_count must match the equity curve")
        if metadata.simulation.starting_cash != result.initial_cash:
            raise ValueError("metadata starting cash must match the backtest result")
        if metadata.simulation.end_of_test_policy is not result.end_of_test_policy:
            raise ValueError("metadata end policy must match the backtest result")
        if result.trades and any(
            trade.symbol != metadata.dataset.symbol for trade in result.trades
        ):
            raise ValueError("metadata symbol must match every simulated trade")
        if self.performance.ending_cash != result.final_cash:
            raise ValueError("performance ending cash must match the backtest result")
        if self.performance.ending_equity != result.final_equity:
            raise ValueError("performance ending equity must match the backtest result")
        if self.trade_statistics.completed_trade_count != len(self.closed_trades):
            raise ValueError("trade statistics must match reconstructed closed trades")
        if len(self.return_series.returns) != len(result.equity_curve):
            raise ValueError("return series must match the equity observation count")
        if self.pre_trade_decisions != result.pre_trade_decisions:
            raise ValueError("structured decisions must match the backtest result")
        if (self.benchmark is None) != (self.benchmark_comparison is None):
            raise ValueError("benchmark and benchmark_comparison must be present together")
        if self.benchmark is not None and not isinstance(
            self.benchmark,
            BenchmarkAnalysis,
        ):
            raise TypeError("benchmark must be BenchmarkAnalysis or None")
        if self.benchmark_comparison is not None and not isinstance(
            self.benchmark_comparison,
            BenchmarkComparison,
        ):
            raise TypeError("benchmark_comparison must be BenchmarkComparison or None")
        settings_present = metadata.risk_metric_settings is not None
        metrics_present = self.risk_adjusted_metrics is not None
        if settings_present != metrics_present:
            raise ValueError("risk metrics must be present exactly when settings are present")
        if self.performance != calculate_performance(result):
            raise ValueError("performance must come from calculate_performance")
        if self.closed_trades != reconstruct_closed_trades(result):
            raise ValueError("closed_trades must come from reconstruct_closed_trades")
        if self.trade_statistics != calculate_trade_statistics(result):
            raise ValueError("trade_statistics must come from calculate_trade_statistics")
        if self.return_series != construct_return_series(result):
            raise ValueError("return_series must come from construct_return_series")
        if self.exposure_statistics != calculate_exposure_statistics(result):
            raise ValueError(
                "exposure_statistics must come from calculate_exposure_statistics"
            )
        if (
            metadata.risk_metric_settings is not None
            and self.risk_adjusted_metrics
            != calculate_risk_adjusted_metrics(
                result,
                metadata.risk_metric_settings,
            )
        ):
            raise ValueError(
                "risk_adjusted_metrics must come from calculate_risk_adjusted_metrics"
            )
        if self.benchmark is not None and self.benchmark_comparison is not None:
            _validate_benchmark_metadata(self.benchmark, metadata)
            _validate_benchmark_comparison(
                self.benchmark_comparison,
                self.benchmark,
                self.performance,
                self.risk_adjusted_metrics,
                self.exposure_statistics,
            )


def build_structured_backtest_report(
    result: BacktestResult,
    metadata: BacktestRunMetadata,
    *,
    benchmark: BenchmarkAnalysis | None = None,
    benchmark_comparison: BenchmarkComparison | None = None,
) -> StructuredBacktestReport:
    """Assemble a report exclusively through existing public calculations."""

    if not isinstance(result, BacktestResult):
        raise TypeError("result must be a BacktestResult")
    if not isinstance(metadata, BacktestRunMetadata):
        raise TypeError("metadata must be BacktestRunMetadata")
    settings = metadata.risk_metric_settings
    return StructuredBacktestReport(
        schema_version=STRUCTURED_REPORT_SCHEMA_VERSION,
        metadata=metadata,
        backtest_result=result,
        performance=calculate_performance(result),
        trade_statistics=calculate_trade_statistics(result),
        closed_trades=reconstruct_closed_trades(result),
        return_series=construct_return_series(result),
        risk_adjusted_metrics=(
            None if settings is None else calculate_risk_adjusted_metrics(result, settings)
        ),
        exposure_statistics=calculate_exposure_statistics(result),
        pre_trade_decisions=result.pre_trade_decisions,
        benchmark=benchmark,
        benchmark_comparison=benchmark_comparison,
    )


def prepare_equity_curve_rows(
    report: StructuredBacktestReport,
) -> tuple[EquityCurveExportRow, ...]:
    """Derive cash, position, drawdown, and return columns in source order."""

    if not isinstance(report, StructuredBacktestReport):
        raise TypeError("report must be a StructuredBacktestReport")
    return _prepare_result_equity_rows(
        report.backtest_result,
        report.return_series,
    )


def prepare_benchmark_equity_rows(
    report: StructuredBacktestReport,
) -> tuple[BenchmarkEquityComparisonRow, ...]:
    """Align strategy and benchmark equity observations without interpolation."""

    if not isinstance(report, StructuredBacktestReport):
        raise TypeError("report must be a StructuredBacktestReport")
    if report.benchmark is None:
        raise ValueError("benchmark equity export requires a benchmark")
    strategy_rows = prepare_equity_curve_rows(report)
    benchmark_rows = _prepare_result_equity_rows(
        report.benchmark.result,
        report.benchmark.return_series,
    )
    if len(strategy_rows) != len(benchmark_rows):
        raise ValueError("strategy and benchmark equity sequences must have equal length")
    rows: list[BenchmarkEquityComparisonRow] = []
    for strategy_row, benchmark_row in zip(
        strategy_rows,
        benchmark_rows,
        strict=True,
    ):
        if strategy_row.timestamp != benchmark_row.timestamp:
            raise ValueError("strategy and benchmark equity timestamps must match exactly")
        rows.append(
            BenchmarkEquityComparisonRow(
                timestamp=strategy_row.timestamp,
                strategy_cash=strategy_row.cash,
                strategy_position_market_value=strategy_row.position_market_value,
                strategy_equity=strategy_row.equity,
                strategy_drawdown=strategy_row.percentage_drawdown,
                benchmark_cash=benchmark_row.cash,
                benchmark_position_market_value=benchmark_row.position_market_value,
                benchmark_equity=benchmark_row.equity,
                benchmark_drawdown=benchmark_row.percentage_drawdown,
                equity_difference=_decimal_difference(
                    strategy_row.equity,
                    benchmark_row.equity,
                ),
            )
        )
    return tuple(rows)


def _prepare_result_equity_rows(
    result: BacktestResult,
    return_series: ReturnSeries,
) -> tuple[EquityCurveExportRow, ...]:
    """Derive one result's chronological account and drawdown observations."""

    returns = return_series.returns
    exposure_observations = derive_exposure_observations(result)
    if len(returns) != len(result.equity_curve):
        raise ValueError("return series must align with the equity curve")
    if len(exposure_observations) != len(result.equity_curve):
        raise ValueError("exposure observations must align with the equity curve")
    running_peak = result.initial_cash
    rows: list[EquityCurveExportRow] = []

    for point, period_return, exposure in zip(
        result.equity_curve,
        returns,
        exposure_observations,
        strict=True,
    ):
        _validate_equity_return_alignment(point, period_return.timestamp)
        if exposure.timestamp != point.timestamp:
            raise ValueError("exposure timestamp must match its equity observation")
        if point.equity > running_peak:
            running_peak = point.equity
        absolute_drawdown = running_peak - point.equity
        percentage_drawdown = (point.equity - running_peak) / running_peak
        rows.append(
            EquityCurveExportRow(
                timestamp=point.timestamp,
                cash=exposure.cash,
                position_quantity=exposure.position_quantity,
                position_market_value=exposure.position_market_value,
                equity=point.equity,
                running_peak_equity=running_peak,
                absolute_drawdown=absolute_drawdown,
                percentage_drawdown=percentage_drawdown,
                period_return=period_return.return_ratio,
                capital_utilisation_ratio=exposure.capital_utilisation_ratio,
                cash_ratio=exposure.cash_ratio,
                invested=exposure.invested,
            )
        )

    if rows[-1].equity != result.final_equity:
        raise ValueError("final export row must contain authoritative ending equity")
    if rows[-1].cash != result.final_cash:
        raise ValueError("final export row must contain authoritative ending cash")
    if rows[-1].position_quantity != result.final_position_quantity:
        raise ValueError("final export row must contain authoritative ending position")
    return tuple(rows)


def _validate_benchmark_metadata(
    benchmark: BenchmarkAnalysis,
    metadata: BacktestRunMetadata,
) -> None:
    """Require benchmark assumptions to match the strategy run metadata."""

    assumptions = benchmark.assumptions
    if (
        assumptions.symbol != metadata.dataset.symbol
        or assumptions.actual_start != metadata.dataset.actual_start
        or assumptions.actual_end != metadata.dataset.actual_end
        or assumptions.bar_count != metadata.dataset.bar_count
    ):
        raise ValueError("benchmark dataset assumptions must match metadata")
    simulation = metadata.simulation
    if (
        assumptions.starting_cash != simulation.starting_cash
        or assumptions.commission_bps != simulation.commission_bps
        or assumptions.slippage_bps != simulation.slippage_bps
        or assumptions.end_of_test_policy is not simulation.end_of_test_policy
    ):
        raise ValueError("benchmark simulation assumptions must match metadata")
    sizing = metadata.position_sizing
    if (
        assumptions.position_sizing_mode is not sizing.mode
        or assumptions.cash_allocation_ratio != sizing.cash_allocation_ratio
        or assumptions.trade_quantity != sizing.quantity
    ):
        raise ValueError("benchmark sizing assumptions must match metadata")
    risk = metadata.risk_policy
    if (
        assumptions.maximum_position_value != risk.maximum_position_value
        or assumptions.minimum_cash_reserve != risk.minimum_cash_reserve
    ):
        raise ValueError("benchmark risk assumptions must match metadata")
    if assumptions.risk_metric_settings != metadata.risk_metric_settings:
        raise ValueError("benchmark metric settings must match metadata")


def _validate_benchmark_comparison(
    comparison: BenchmarkComparison,
    benchmark: BenchmarkAnalysis,
    strategy_performance: PerformanceSummary,
    strategy_risk: RiskAdjustedMetrics | None,
    strategy_exposure: ExposureStatistics,
) -> None:
    """Require structured comparison inputs to come from the report components."""

    expected_values = {
        "benchmark_name": benchmark.name,
        "strategy_ending_equity": strategy_performance.ending_equity,
        "benchmark_ending_equity": benchmark.performance.ending_equity,
        "strategy_total_return": strategy_performance.total_return,
        "benchmark_total_return": benchmark.performance.total_return,
        "strategy_maximum_absolute_drawdown": (
            strategy_performance.maximum_absolute_drawdown
        ),
        "benchmark_maximum_absolute_drawdown": (
            benchmark.performance.maximum_absolute_drawdown
        ),
        "strategy_maximum_percentage_drawdown": (
            strategy_performance.maximum_percentage_drawdown
        ),
        "benchmark_maximum_percentage_drawdown": (
            benchmark.performance.maximum_percentage_drawdown
        ),
        "strategy_time_in_market_ratio": strategy_exposure.time_in_market_ratio,
        "benchmark_time_in_market_ratio": (
            benchmark.exposure_statistics.time_in_market_ratio
        ),
        "strategy_total_commission": strategy_performance.total_commission,
        "benchmark_total_commission": benchmark.performance.total_commission,
        "strategy_total_slippage_cost": (
            strategy_performance.total_adverse_slippage_cost
        ),
        "benchmark_total_slippage_cost": (
            benchmark.performance.total_adverse_slippage_cost
        ),
    }
    risk_attributes = (
        "periodic_volatility",
        "annualised_volatility",
        "periodic_sharpe_ratio",
        "annualised_sharpe_ratio",
        "periodic_sortino_ratio",
        "annualised_sortino_ratio",
    )
    for attribute in risk_attributes:
        expected_values[f"strategy_{attribute}"] = (
            None if strategy_risk is None else getattr(strategy_risk, attribute)
        )
        expected_values[f"benchmark_{attribute}"] = (
            None
            if benchmark.risk_adjusted_metrics is None
            else getattr(benchmark.risk_adjusted_metrics, attribute)
        )
    for attribute, expected in expected_values.items():
        if getattr(comparison, attribute) != expected:
            raise ValueError(f"comparison {attribute} must match report calculations")


def _decimal_difference(left: Decimal, right: Decimal) -> Decimal:
    """Subtract equity values in the report's fixed Decimal precision policy."""

    with localcontext(
        Context(prec=COMPARISON_DECIMAL_PRECISION, rounding=ROUND_HALF_EVEN)
    ):
        return left - right


def _validate_equity_return_alignment(point: EquityPoint, timestamp: datetime) -> None:
    """Require each existing return to describe the same equity observation."""

    if point.timestamp != timestamp:
        raise ValueError("period return timestamp must match its equity observation")
