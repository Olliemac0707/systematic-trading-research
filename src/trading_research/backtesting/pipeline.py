"""Reusable deterministic orchestration for one configured historical backtest."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from trading_research.backtesting.engine import SimpleBacktestEngine
from trading_research.benchmarks import (
    BenchmarkAnalysis,
    BenchmarkComparison,
    calculate_benchmark_comparison,
    run_buy_and_hold_benchmark,
)
from trading_research.config import BacktestConfig
from trading_research.data import CsvMarketDataProvider
from trading_research.errors import MarketDataError
from trading_research.models import (
    BacktestResult,
    MarketBar,
    StrategySignal,
    normalize_symbol,
)
from trading_research.performance import RiskMetricSettings
from trading_research.reporting import (
    StrategyRunConfiguration,
    StructuredBacktestReport,
    build_structured_backtest_report,
    create_backtest_run_metadata,
)
from trading_research.risk import build_position_sizer, build_risk_policy
from trading_research.strategies import (
    DonchianBreakoutParameters,
    SmaCrossoverParameters,
    Strategy,
    create_strategy,
)


@dataclass(frozen=True, slots=True)
class _PrecomputedSignalStrategy(Strategy):
    """Expose already-derived boundary-safe signals to the unchanged engine."""

    signals: tuple[StrategySignal, ...]

    def generate_signals(self, bars: Sequence[MarketBar]) -> tuple[StrategySignal, ...]:
        """Return signals calculated from evaluation bars plus warm-up history."""

        return self.signals


@dataclass(frozen=True, slots=True)
class BacktestRunRequest:
    """All validated inputs required by the shared single-run pipeline."""

    data_path: Path
    symbol: str
    strategy_configuration: StrategyRunConfiguration
    backtest_configuration: BacktestConfig
    start: datetime | None = None
    end: datetime | None = None
    risk_metric_settings: RiskMetricSettings | None = None
    include_buy_and_hold_benchmark: bool = False

    def __post_init__(self) -> None:
        """Validate orchestration types without duplicating domain calculations."""

        object.__setattr__(self, "data_path", Path(self.data_path))
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        if not isinstance(self.strategy_configuration, StrategyRunConfiguration):
            raise TypeError("strategy_configuration must be StrategyRunConfiguration")
        if not isinstance(self.backtest_configuration, BacktestConfig):
            raise TypeError("backtest_configuration must be BacktestConfig")
        if self.risk_metric_settings is not None and not isinstance(
            self.risk_metric_settings,
            RiskMetricSettings,
        ):
            raise TypeError("risk_metric_settings must be RiskMetricSettings or None")
        if not isinstance(self.include_buy_and_hold_benchmark, bool):
            raise TypeError("include_buy_and_hold_benchmark must be a bool")


@dataclass(frozen=True, slots=True)
class BacktestRunExecution:
    """Immutable calculations produced by one shared pipeline execution."""

    request: BacktestRunRequest
    bars: tuple[MarketBar, ...]
    result: BacktestResult
    benchmark: BenchmarkAnalysis | None
    benchmark_comparison: BenchmarkComparison | None
    warmup_bars: tuple[MarketBar, ...] = ()

    def __post_init__(self) -> None:
        """Require one internally consistent execution outcome."""

        if not isinstance(self.request, BacktestRunRequest):
            raise TypeError("request must be BacktestRunRequest")
        if not self.bars or any(not isinstance(bar, MarketBar) for bar in self.bars):
            raise TypeError("bars must be a non-empty tuple of MarketBar values")
        if not isinstance(self.result, BacktestResult):
            raise TypeError("result must be BacktestResult")
        if (self.benchmark is None) != (self.benchmark_comparison is None):
            raise ValueError("benchmark and comparison must be present together")
        if any(not isinstance(bar, MarketBar) for bar in self.warmup_bars):
            raise TypeError("warmup_bars must contain MarketBar values")
        if self.warmup_bars and self.warmup_bars[-1].timestamp >= self.bars[0].timestamp:
            raise ValueError("warm-up bars must strictly precede evaluation bars")


def run_backtest_pipeline(request: BacktestRunRequest) -> BacktestRunExecution:
    """Execute one strategy sequentially through all existing financial components."""

    if not isinstance(request, BacktestRunRequest):
        raise TypeError("request must be BacktestRunRequest")
    bars = tuple(
        CsvMarketDataProvider(request.data_path).get_historical_bars(
            symbol=request.symbol,
            start=request.start,
            end=request.end,
        )
    )
    return run_backtest_pipeline_with_bars(request, bars)


def run_backtest_pipeline_with_bars(
    request: BacktestRunRequest,
    bars: Sequence[MarketBar],
    *,
    warmup_bars: Sequence[MarketBar] = (),
) -> BacktestRunExecution:
    """Execute existing calculations on explicit evaluation and optional warm-up bars."""

    if not isinstance(request, BacktestRunRequest):
        raise TypeError("request must be BacktestRunRequest")
    evaluation = tuple(bars)
    warmup = tuple(warmup_bars)
    if not evaluation:
        raise MarketDataError("no market bars match the requested symbol and date range")
    combined = warmup + evaluation
    if any(not isinstance(bar, MarketBar) for bar in combined):
        raise TypeError("pipeline bars must contain MarketBar values")
    if any(bar.symbol != request.symbol for bar in combined):
        raise MarketDataError("pipeline bars must match the requested symbol")
    if any(
        current.timestamp <= previous.timestamp
        for previous, current in zip(combined, combined[1:], strict=False)
    ):
        raise MarketDataError("pipeline bars must be strictly chronological")

    strategy = create_strategy(
        request.strategy_configuration.name,
        request.strategy_configuration.parameters,
    )
    if warmup:
        required_warmup = strategy.required_warmup_bars()
        if len(warmup) != required_warmup:
            raise MarketDataError(
                "carry-history requires exactly "
                f"{required_warmup} warm-up bars; received {len(warmup)}"
            )
        period_signals = tuple(
            strategy.generate_signals_for_period(combined, evaluation[0].timestamp)
        )
        engine_strategy: Strategy = _PrecomputedSignalStrategy(period_signals)
    else:
        minimum_bars = minimum_required_bars(request.strategy_configuration)
        if len(evaluation) < minimum_bars:
            raise MarketDataError(
                "insufficient market data for the selected strategy windows: "
                f"loaded {len(evaluation)} bars; at least {minimum_bars} are required"
            )
        engine_strategy = strategy

    config = request.backtest_configuration
    position_sizer = build_position_sizer(config)
    risk_policy = build_risk_policy(config)
    result = SimpleBacktestEngine().run(
        evaluation,
        engine_strategy,
        config,
        position_sizer=position_sizer,
        risk_policy=risk_policy,
    )
    benchmark = None
    comparison = None
    if request.include_buy_and_hold_benchmark:
        benchmark = run_buy_and_hold_benchmark(
            bars=evaluation,
            backtest_configuration=config,
            position_sizer=position_sizer,
            risk_policy=risk_policy,
            risk_metric_settings=request.risk_metric_settings,
        )
        comparison = calculate_benchmark_comparison(
            result,
            benchmark,
            bars=evaluation,
            backtest_configuration=config,
            risk_metric_settings=request.risk_metric_settings,
        )
    return BacktestRunExecution(
        request=request,
        bars=evaluation,
        result=result,
        benchmark=benchmark,
        benchmark_comparison=comparison,
        warmup_bars=warmup,
    )


def build_execution_report(
    execution: BacktestRunExecution,
    *,
    run_id: str | None = None,
    generated_at: datetime | None = None,
    application_version: str | None = None,
    git_commit: str | None = None,
    base_directory: Path | None = None,
) -> StructuredBacktestReport:
    """Attach existing reproducibility metadata and structured calculations."""

    if not isinstance(execution, BacktestRunExecution):
        raise TypeError("execution must be BacktestRunExecution")
    request = execution.request
    metadata = create_backtest_run_metadata(
        data_path=request.data_path,
        bars=execution.bars,
        symbol=request.symbol,
        requested_start=request.start,
        requested_end=request.end,
        strategy_configuration=request.strategy_configuration,
        config=request.backtest_configuration,
        risk_metric_settings=request.risk_metric_settings,
        run_id=run_id,
        generated_at=generated_at,
        application_version=application_version,
        git_commit=git_commit,
        base_directory=base_directory,
    )
    return build_structured_backtest_report(
        execution.result,
        metadata,
        benchmark=execution.benchmark,
        benchmark_comparison=execution.benchmark_comparison,
    )


def minimum_required_bars(configuration: StrategyRunConfiguration) -> int:
    """Return observations needed to form a possible first entry signal."""

    if not isinstance(configuration, StrategyRunConfiguration):
        raise TypeError("configuration must be StrategyRunConfiguration")
    parameters = configuration.parameters
    if isinstance(parameters, SmaCrossoverParameters):
        return parameters.slow_window + 1
    if isinstance(parameters, DonchianBreakoutParameters):
        return parameters.entry_window + 1
    raise TypeError("unsupported strategy parameter model")
