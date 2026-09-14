"""Deterministic capital-matched buy-and-hold benchmark tests."""

import csv
import json
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from io import StringIO
from pathlib import Path

import pytest

from trading_research.backtesting import SimpleBacktestEngine
from trading_research.benchmarks import (
    BenchmarkAnalysis,
    BuyAndHoldBenchmarkStrategy,
    calculate_benchmark_comparison,
    run_buy_and_hold_benchmark,
)
from trading_research.config import (
    BacktestConfig,
    EndOfTestPolicy,
    PositionSizingMode,
)
from trading_research.data import CsvMarketDataProvider
from trading_research.errors import BacktestError, ReportWriteError
from trading_research.models import (
    EquityPoint,
    ExecutionReason,
    MarketBar,
    PreTradeDecisionOutcome,
    SignalAction,
    StrategySignal,
    TradeSide,
)
from trading_research.performance import (
    RiskMetricSettings,
    calculate_exposure_statistics,
    calculate_performance,
    calculate_trade_statistics,
    construct_return_series,
)
from trading_research.reporting import (
    BENCHMARK_EQUITY_CSV_COLUMNS,
    StructuredBacktestReport,
    benchmark_equity_to_csv,
    build_structured_backtest_report,
    create_backtest_run_metadata,
    format_backtest_report,
    structured_report_to_json,
    summary_report_to_csv,
    write_benchmark_equity_csv,
)
from trading_research.risk import (
    AllowAllRiskPolicy,
    FixedQuantitySizer,
    build_position_sizer,
    build_risk_policy,
)
from trading_research.strategies import SimpleMovingAverageCrossover, Strategy

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_DATA = _ROOT / "data" / "sample_prices.csv"
_START = datetime(2026, 1, 5, 16, tzinfo=UTC)


def make_bars(
    closes: tuple[str, ...] = ("100", "120", "130"),
    opens: tuple[str, ...] = ("100", "110", "125"),
) -> tuple[MarketBar, ...]:
    """Create valid daily bars whose opens and closes are independently visible."""

    bars: list[MarketBar] = []
    for index, (open_text, close_text) in enumerate(zip(opens, closes, strict=True)):
        open_price = Decimal(open_text)
        close_price = Decimal(close_text)
        bars.append(
            MarketBar(
                symbol="DEMO",
                timestamp=_START + timedelta(days=index),
                open=open_price,
                high=max(open_price, close_price),
                low=min(open_price, close_price),
                close=close_price,
                volume=1000,
            )
        )
    return tuple(bars)


def run_benchmark(
    *,
    config: BacktestConfig | None = None,
    bars: tuple[MarketBar, ...] | None = None,
    settings: RiskMetricSettings | None = None,
) -> BenchmarkAnalysis:
    """Run the public benchmark with policy objects built from one configuration."""

    selected_config = (
        BacktestConfig(
            initial_cash=Decimal("1000"),
            trade_quantity=2,
            commission_bps=Decimal("100"),
            slippage_bps=Decimal("100"),
        )
        if config is None
        else config
    )
    selected_bars = make_bars() if bars is None else bars
    return run_buy_and_hold_benchmark(
        bars=selected_bars,
        backtest_configuration=selected_config,
        position_sizer=build_position_sizer(selected_config),
        risk_policy=build_risk_policy(selected_config),
        risk_metric_settings=settings,
    )


class NoSignalsStrategy(Strategy):
    """Remain in cash for comparison tests."""

    def generate_signals(self, bars: Sequence[MarketBar]) -> tuple[StrategySignal, ...]:
        """Return no actionable observations."""

        return ()


class ZeroSizer:
    """Return an explicit zero-sized proposal without side effects."""

    def calculate_quantity(
        self,
        *,
        available_cash: Decimal,
        reference_price: Decimal,
        commission_bps: Decimal,
        slippage_bps: Decimal,
    ) -> int:
        """Return zero for every valid input."""

        return 0


def test_benchmark_emits_one_first_close_signal_without_mutating_bars() -> None:
    """The convention is one first-timestamp buy and no repeats or exits."""

    bars = make_bars()
    original = tuple(bars)

    signals = BuyAndHoldBenchmarkStrategy().generate_signals(bars)

    assert bars == original
    assert len(signals) == 1
    assert signals[0].timestamp == bars[0].timestamp
    assert signals[0].symbol == "DEMO"
    assert signals[0].action is SignalAction.BUY
    assert "first available close" in signals[0].reason


def test_benchmark_rejects_fewer_than_two_bars_clearly() -> None:
    """An entry cannot be executed without the next bar's open."""

    with pytest.raises(BacktestError, match="at least two"):
        BuyAndHoldBenchmarkStrategy().generate_signals(make_bars()[:1])
    with pytest.raises(BacktestError, match="at least two"):
        run_buy_and_hold_benchmark(
            bars=make_bars()[:1],
            backtest_configuration=BacktestConfig(),
            position_sizer=FixedQuantitySizer(1),
            risk_policy=AllowAllRiskPolicy(),
        )


def test_hold_benchmark_uses_second_open_shared_costs_and_exact_equity() -> None:
    """Entry, cash, open mark, commission, and slippage use existing engine math."""

    analysis = run_benchmark()
    trade = analysis.result.trades[0]

    assert len(analysis.result.trades) == 1
    assert trade.timestamp == make_bars()[1].timestamp
    assert trade.side is TradeSide.BUY
    assert trade.reference_price == Decimal("110")
    assert trade.price == Decimal("111.10")
    assert trade.commission == Decimal("2.2220")
    assert trade.slippage_cost == Decimal("2.20")
    assert analysis.result.final_cash == Decimal("775.5780")
    assert analysis.result.final_position_quantity == 2
    assert analysis.result.final_equity == Decimal("1035.5780")
    assert analysis.performance.unrealised_pnl == Decimal("37.80")
    assert analysis.trade_statistics.completed_trade_count == 0
    assert analysis.result.reconciliation.is_reconciled


def test_liquidation_benchmark_uses_final_close_and_costs_once() -> None:
    """Synthetic exit uses sell friction once and produces one closed round trip."""

    config = BacktestConfig(
        initial_cash=Decimal("1000"),
        trade_quantity=2,
        commission_bps=Decimal("100"),
        slippage_bps=Decimal("100"),
        end_of_test_policy=EndOfTestPolicy.LIQUIDATE,
    )
    analysis = run_benchmark(config=config)
    exit_trade = analysis.result.trades[-1]

    assert exit_trade.side is TradeSide.SELL
    assert exit_trade.reference_price == Decimal("130")
    assert exit_trade.price == Decimal("128.70")
    assert exit_trade.commission == Decimal("2.5740")
    assert exit_trade.slippage_cost == Decimal("2.60")
    assert exit_trade.execution_reason is ExecutionReason.END_OF_TEST_LIQUIDATION
    assert analysis.performance.total_commission == Decimal("4.7960")
    assert analysis.performance.total_adverse_slippage_cost == Decimal("4.80")
    assert analysis.result.final_cash == Decimal("1030.4040")
    assert analysis.result.final_equity == Decimal("1030.4040")
    assert analysis.trade_statistics.completed_trade_count == 1
    assert analysis.result.reconciliation.is_reconciled


def test_cash_allocation_and_reduced_risk_use_normal_decision_workflow() -> None:
    """Benchmark sizing and risk reduction remain audited engine decisions."""

    allocation = BacktestConfig(
        initial_cash=Decimal("1000"),
        position_sizing_mode=PositionSizingMode.CASH_ALLOCATION,
        cash_allocation_ratio=Decimal("0.50"),
    )
    allocated = run_benchmark(config=allocation)
    reduced_config = BacktestConfig(
        initial_cash=Decimal("1000"),
        trade_quantity=10,
        maximum_position_value=Decimal("350"),
    )
    reduced = run_benchmark(config=reduced_config)

    assert allocated.result.trades[0].quantity == 4
    assert allocated.pre_trade_decisions[0].proposed_quantity == 4
    assert allocated.assumptions.cash_allocation_ratio == Decimal("0.50")
    assert reduced.pre_trade_decisions[0].proposed_quantity == 10
    assert reduced.pre_trade_decisions[0].approved_quantity == 3
    assert reduced.pre_trade_decisions[0].outcome is PreTradeDecisionOutcome.REDUCED
    assert reduced.result.trades[0].quantity == 3


def test_rejected_and_zero_sized_benchmarks_remain_in_cash() -> None:
    """Risk rejection and zero proposals never force benchmark investment."""

    rejected_config = BacktestConfig(
        initial_cash=Decimal("1000"),
        trade_quantity=2,
        minimum_cash_reserve=Decimal("1000"),
    )
    rejected = run_benchmark(config=rejected_config)
    zero = run_buy_and_hold_benchmark(
        bars=make_bars(),
        backtest_configuration=BacktestConfig(initial_cash=Decimal("1000")),
        position_sizer=ZeroSizer(),
        risk_policy=AllowAllRiskPolicy(),
    )

    assert rejected.result.trades == ()
    assert rejected.pre_trade_decisions[0].outcome is PreTradeDecisionOutcome.REJECTED
    assert rejected.result.final_equity == Decimal("1000")
    assert rejected.exposure_statistics.time_in_market_ratio == Decimal("0")
    assert zero.result.trades == ()
    assert zero.pre_trade_decisions[0].proposed_quantity == 0
    assert zero.result.final_cash == zero.result.final_equity == Decimal("1000")


def test_comparison_is_exact_deterministic_and_propagates_missing_metrics() -> None:
    """Equal runs have zero differences and unavailable risk metrics stay absent."""

    bars = make_bars()
    config = BacktestConfig(initial_cash=Decimal("1000"), trade_quantity=2)
    benchmark = run_benchmark(config=config, bars=bars)

    first = calculate_benchmark_comparison(
        benchmark.result,
        benchmark,
        bars=bars,
        backtest_configuration=config,
    )
    second = calculate_benchmark_comparison(
        benchmark.result,
        benchmark,
        bars=bars,
        backtest_configuration=config,
    )

    assert first == second
    assert first.ending_equity_difference == Decimal("0")
    assert first.excess_return == Decimal("0")
    assert first.maximum_drawdown_improvement == Decimal("0")
    assert first.time_in_market_difference == Decimal("0")
    assert first.periodic_volatility_difference is None
    assert first.periodic_sharpe_difference is None
    assert first.periodic_sortino_difference is None


def test_cash_strategy_underperforms_rising_benchmark_and_improves_drawdown() -> None:
    """Strategy-minus-benchmark signs follow the documented conventions."""

    bars = make_bars(closes=("100", "80", "130"))
    config = BacktestConfig(initial_cash=Decimal("1000"), trade_quantity=2)
    benchmark = run_benchmark(config=config, bars=bars)
    cash_result = SimpleBacktestEngine().run(
        bars,
        NoSignalsStrategy(),
        config,
    )
    comparison = calculate_benchmark_comparison(
        cash_result,
        benchmark,
        bars=bars,
        backtest_configuration=config,
    )

    assert comparison.ending_equity_difference == Decimal("-40")
    assert comparison.excess_return == Decimal("-0.04")
    assert comparison.maximum_drawdown_improvement == Decimal("0.06")
    assert comparison.time_in_market_difference == Decimal("-0.6666666666666666666666666666666667")


def test_cash_strategy_outperforms_falling_benchmark() -> None:
    """Avoiding a benchmark loss produces positive excess return."""

    bars = make_bars(closes=("100", "100", "80"), opens=("100", "100", "90"))
    config = BacktestConfig(initial_cash=Decimal("1000"), trade_quantity=2)
    benchmark = run_benchmark(config=config, bars=bars)
    cash_result = SimpleBacktestEngine().run(bars, NoSignalsStrategy(), config)
    comparison = calculate_benchmark_comparison(
        cash_result,
        benchmark,
        bars=bars,
        backtest_configuration=config,
    )

    assert comparison.ending_equity_difference == Decimal("40")
    assert comparison.excess_return == Decimal("0.04")


def test_risk_metric_settings_are_shared_and_optional_differences_are_exact() -> None:
    """Risk-adjusted calculations use one settings object for both results."""

    bars = make_bars()
    config = BacktestConfig(initial_cash=Decimal("1000"), trade_quantity=2)
    settings = RiskMetricSettings(periods_per_year=Decimal("252"))
    benchmark = run_benchmark(config=config, bars=bars, settings=settings)
    cash_result = SimpleBacktestEngine().run(bars, NoSignalsStrategy(), config)
    comparison = calculate_benchmark_comparison(
        cash_result,
        benchmark,
        bars=bars,
        backtest_configuration=config,
        risk_metric_settings=settings,
    )

    assert benchmark.risk_adjusted_metrics is not None
    assert comparison.strategy_periodic_volatility == Decimal("0")
    assert comparison.benchmark_periodic_volatility is not None
    assert comparison.periodic_volatility_difference == Decimal(
        "-0.01143548060132871600914361424448508"
    )
    assert comparison.periodic_sharpe_difference is None


def test_mismatched_cost_or_period_assumptions_are_rejected() -> None:
    """External comparisons cannot silently mix material assumptions."""

    bars = make_bars()
    config = BacktestConfig(initial_cash=Decimal("1000"), trade_quantity=2)
    benchmark = run_benchmark(config=config, bars=bars)
    mismatched_cost = BacktestConfig(
        initial_cash=Decimal("1000"),
        trade_quantity=2,
        commission_bps=Decimal("1"),
    )

    with pytest.raises(ValueError, match="assumptions must be identical"):
        calculate_benchmark_comparison(
            benchmark.result,
            benchmark,
            bars=bars,
            backtest_configuration=mismatched_cost,
        )
    with pytest.raises(ValueError, match="assumptions must be identical"):
        calculate_benchmark_comparison(
            benchmark.result,
            benchmark,
            bars=make_bars()[:2],
            backtest_configuration=config,
        )


def test_benchmark_does_not_change_existing_sma_result_or_decisions() -> None:
    """The optional second run leaves strategy values and audit records untouched."""

    bars = tuple(
        CsvMarketDataProvider(_SAMPLE_DATA).get_historical_bars("DEMO", None, None)
    )
    config = BacktestConfig(
        initial_cash=Decimal("10000"),
        trade_quantity=10,
        commission_bps=Decimal("1"),
        slippage_bps=Decimal("5"),
    )
    strategy = SimpleMovingAverageCrossover(short_window=2, long_window=3)
    before = SimpleBacktestEngine().run(bars, strategy, config)
    decisions_before = before.pre_trade_decisions
    benchmark = run_buy_and_hold_benchmark(
        bars=bars,
        backtest_configuration=config,
        position_sizer=build_position_sizer(config),
        risk_policy=build_risk_policy(config),
    )
    after = SimpleBacktestEngine().run(bars, strategy, config)

    assert before == after
    assert before.final_cash == Decimal("9959.86799800")
    assert before.final_equity == Decimal("9959.86799800")
    assert before.pre_trade_decisions == decisions_before
    assert benchmark.pre_trade_decisions is not before.pre_trade_decisions
    assert before.reconciliation.is_reconciled
    assert benchmark.result.reconciliation.is_reconciled


def make_structured_benchmark_report(
    *,
    end_policy: EndOfTestPolicy = EndOfTestPolicy.HOLD,
    settings: RiskMetricSettings | None = None,
) -> StructuredBacktestReport:
    """Build one reproducible strategy and benchmark report from the same CSV load."""

    bars = tuple(
        CsvMarketDataProvider(_SAMPLE_DATA).get_historical_bars("DEMO", None, None)
    )
    config = BacktestConfig(
        initial_cash=Decimal("10000"),
        trade_quantity=10,
        commission_bps=Decimal("1"),
        slippage_bps=Decimal("5"),
        end_of_test_policy=end_policy,
    )
    strategy_result = SimpleBacktestEngine().run(
        bars,
        SimpleMovingAverageCrossover(short_window=2, long_window=3),
        config,
    )
    benchmark = run_buy_and_hold_benchmark(
        bars=bars,
        backtest_configuration=config,
        position_sizer=build_position_sizer(config),
        risk_policy=build_risk_policy(config),
        risk_metric_settings=settings,
    )
    comparison = calculate_benchmark_comparison(
        strategy_result,
        benchmark,
        bars=bars,
        backtest_configuration=config,
        risk_metric_settings=settings,
    )
    metadata = create_backtest_run_metadata(
        data_path=_SAMPLE_DATA,
        bars=bars,
        symbol="DEMO",
        requested_start=None,
        requested_end=None,
        fast_window=2,
        slow_window=3,
        config=config,
        risk_metric_settings=settings,
        run_id="benchmark-test",
        generated_at=datetime(2026, 7, 18, tzinfo=UTC),
        application_version="test",
        git_commit=None,
        base_directory=_ROOT,
    )
    return build_structured_backtest_report(
        strategy_result,
        metadata,
        benchmark=benchmark,
        benchmark_comparison=comparison,
    )


def test_text_benchmark_section_is_opt_in_and_uses_existing_formatters() -> None:
    """Default text is unchanged while opt-in text identifies fair conventions."""

    report = make_structured_benchmark_report()
    baseline = format_backtest_report(report.backtest_result)
    with_benchmark = format_backtest_report(
        report.backtest_result,
        benchmark=report.benchmark,
        benchmark_comparison=report.benchmark_comparison,
    )

    assert "Buy-and-Hold Benchmark" not in baseline
    assert "Buy-and-Hold Benchmark" in with_benchmark
    assert "Next-bar executable" in with_benchmark
    assert "same costs, sizing, and risk controls" in with_benchmark
    assert "Periodic Sharpe difference:        Not available" in with_benchmark
    assert "Ending-equity difference:" in with_benchmark


def test_structured_json_and_summary_csv_use_schema_1_3_and_optional_fields() -> None:
    """Benchmark payloads are complete and absent values follow stable conventions."""

    benchmark_report = make_structured_benchmark_report(
        end_policy=EndOfTestPolicy.LIQUIDATE,
    )
    without_benchmark = build_structured_backtest_report(
        benchmark_report.backtest_result,
        benchmark_report.metadata,
    )
    payload = json.loads(structured_report_to_json(benchmark_report))
    absent_payload = json.loads(structured_report_to_json(without_benchmark))
    present_summary = next(
        csv.DictReader(StringIO(summary_report_to_csv(benchmark_report)))
    )
    absent_summary = next(
        csv.DictReader(StringIO(summary_report_to_csv(without_benchmark)))
    )

    assert payload["schema_version"] == "1.3"
    assert payload["benchmark"]["name"] == "buy-and-hold"
    assert payload["benchmark"]["result"]["trades"][-1]["execution_reason"] == (
        "end_of_test_liquidation"
    )
    assert payload["benchmark"]["pre_trade_decisions"]
    assert payload["benchmark_comparison"]["excess_return"] == present_summary[
        "excess_return"
    ]
    assert absent_payload["benchmark"] is None
    assert absent_payload["benchmark_comparison"] is None
    assert present_summary["benchmark_name"] == "buy-and-hold"
    assert present_summary["benchmark_ending_equity"] != ""
    assert absent_summary["benchmark_name"] == ""
    assert absent_summary["benchmark_ending_equity"] == ""
    assert absent_summary["maximum_drawdown_improvement"] == ""


@pytest.mark.parametrize("policy", list(EndOfTestPolicy))
def test_benchmark_equity_csv_aligns_hold_and_liquidate_rows(
    policy: EndOfTestPolicy,
) -> None:
    """Both end policies export exact chronological strategy-minus-benchmark rows."""

    report = make_structured_benchmark_report(end_policy=policy)
    rows = list(csv.DictReader(StringIO(benchmark_equity_to_csv(report))))

    assert tuple(rows[0]) == BENCHMARK_EQUITY_CSV_COLUMNS
    assert len(rows) == report.metadata.dataset.bar_count
    assert [row["timestamp"] for row in rows] == sorted(
        row["timestamp"] for row in rows
    )
    assert rows[-1]["strategy_equity"] == str(report.backtest_result.final_equity)
    assert report.benchmark is not None
    assert rows[-1]["benchmark_equity"] == str(report.benchmark.result.final_equity)
    assert Decimal(rows[-1]["equity_difference"]) == (
        report.backtest_result.final_equity
        - report.benchmark.result.final_equity
    )
    if policy is EndOfTestPolicy.HOLD:
        assert rows[-1]["benchmark_cash"] != rows[-1]["benchmark_equity"]
    else:
        assert rows[-1]["benchmark_cash"] == rows[-1]["benchmark_equity"]


def test_benchmark_equity_writer_protects_and_atomically_overwrites_space_path(
    tmp_path: Path,
) -> None:
    """The new writer follows existing safe-file and explicit-overwrite behavior."""

    report = make_structured_benchmark_report()
    destination = tmp_path / "reports with spaces" / "benchmark equity.csv"
    write_benchmark_equity_csv(report, destination, create_parents=True)

    original = destination.read_text(encoding="utf-8")
    with pytest.raises(ReportWriteError, match="already exists"):
        write_benchmark_equity_csv(report, destination)
    assert destination.read_text(encoding="utf-8") == original
    write_benchmark_equity_csv(report, destination, overwrite=True)
    assert destination.read_text(encoding="utf-8") == original


def test_benchmark_equity_csv_rejects_an_inconsistent_timestamp_sequence() -> None:
    """Equal endpoints and lengths cannot hide a misaligned middle observation."""

    bars = make_bars(
        closes=("100", "120", "125", "130"),
        opens=("100", "110", "123", "128"),
    )
    config = BacktestConfig(initial_cash=Decimal("1000"), trade_quantity=2)
    strategy_result = SimpleBacktestEngine().run(bars, NoSignalsStrategy(), config)
    benchmark = run_benchmark(config=config, bars=bars)
    shifted_curve = list(benchmark.result.equity_curve)
    shifted_curve[2] = EquityPoint(
        timestamp=shifted_curve[2].timestamp + timedelta(hours=1),
        equity=shifted_curve[2].equity,
    )
    shifted_result = replace(
        benchmark.result,
        equity_curve=tuple(shifted_curve),
    )
    shifted_benchmark = BenchmarkAnalysis(
        name=benchmark.name,
        assumptions=benchmark.assumptions,
        result=shifted_result,
        performance=calculate_performance(shifted_result),
        trade_statistics=calculate_trade_statistics(shifted_result),
        return_series=construct_return_series(shifted_result),
        risk_adjusted_metrics=None,
        exposure_statistics=calculate_exposure_statistics(shifted_result),
        pre_trade_decisions=shifted_result.pre_trade_decisions,
    )
    comparison = calculate_benchmark_comparison(
        strategy_result,
        shifted_benchmark,
        bars=bars,
        backtest_configuration=config,
    )
    metadata = create_backtest_run_metadata(
        data_path=_SAMPLE_DATA,
        bars=bars,
        symbol="DEMO",
        requested_start=None,
        requested_end=None,
        fast_window=2,
        slow_window=3,
        config=config,
        risk_metric_settings=None,
        run_id="misaligned-benchmark",
        generated_at=datetime(2026, 7, 18, tzinfo=UTC),
        application_version="test",
        git_commit=None,
        base_directory=_ROOT,
    )
    report = build_structured_backtest_report(
        strategy_result,
        metadata,
        benchmark=shifted_benchmark,
        benchmark_comparison=comparison,
    )

    with pytest.raises(ValueError, match="timestamps must match exactly"):
        benchmark_equity_to_csv(report)


def test_benchmark_aggregate_is_immutable() -> None:
    """Benchmark analysis and nested assumptions cannot be reassigned."""

    analysis = run_benchmark()

    with pytest.raises(FrozenInstanceError):
        analysis.name = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        analysis.assumptions.starting_cash = Decimal("1")  # type: ignore[misc]
