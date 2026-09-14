"""Deterministic split runner, warm-up, benchmark, and reconciliation tests."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from evaluation_helpers import (
    SYNTHETIC_START,
    evaluation_split,
    frozen_configuration,
    provenance,
    synthetic_bars,
    write_bars_csv,
)

from trading_research.errors import MarketDataError
from trading_research.evaluation import (
    SplitEvaluationResult,
    WarmupPolicy,
    configuration_sha256,
    run_split_evaluation,
)
from trading_research.experiments import BenchmarkSelection
from trading_research.models import EndOfTestPolicy, MarketBar
from trading_research.strategies import StrategyName

_GENERATED_AT = datetime(2025, 2, 1, tzinfo=UTC)
_GIT_COMMIT = "a" * 40


def run_evaluation(
    tmp_path: Path,
    *,
    strategy_name: StrategyName = StrategyName.SMA_CROSSOVER,
    warmup_policy: WarmupPolicy = WarmupPolicy.ISOLATED,
    benchmark: BenchmarkSelection = BenchmarkSelection.NONE,
    risk_metrics: bool = False,
    end_policy: EndOfTestPolicy = EndOfTestPolicy.HOLD,
    bars: tuple[MarketBar, ...] | None = None,
) -> SplitEvaluationResult:
    """Run one fixed-time evaluation over a matching local dataset file."""

    selected_bars = synthetic_bars() if bars is None else bars
    data_path = write_bars_csv(tmp_path / "prices.csv", selected_bars)
    return run_split_evaluation(
        evaluation_id="synthetic-evaluation",
        data_path=data_path,
        symbol="TEST",
        bars=selected_bars,
        split=evaluation_split(warmup_policy),
        configuration=frozen_configuration(
            strategy_name=strategy_name,
            benchmark=benchmark,
            risk_metrics=risk_metrics,
            end_policy=end_policy,
        ),
        provenance=provenance(),
        git_commit=_GIT_COMMIT,
        generated_at=_GENERATED_AT,
        base_directory=tmp_path,
    )


@pytest.mark.parametrize(
    ("strategy_name", "warmup_policy", "expected_warmup"),
    [
        (StrategyName.SMA_CROSSOVER, WarmupPolicy.ISOLATED, 0),
        (StrategyName.SMA_CROSSOVER, WarmupPolicy.CARRY_HISTORY, 3),
        (StrategyName.DONCHIAN_BREAKOUT, WarmupPolicy.ISOLATED, 0),
        (StrategyName.DONCHIAN_BREAKOUT, WarmupPolicy.CARRY_HISTORY, 2),
    ],
)
def test_strategy_warmup_policies_keep_holdout_account_fresh(
    tmp_path: Path,
    strategy_name: StrategyName,
    warmup_policy: WarmupPolicy,
    expected_warmup: int,
) -> None:
    """History affects signals only; account, ledger, and observations start fresh."""

    result = run_evaluation(
        tmp_path,
        strategy_name=strategy_name,
        warmup_policy=warmup_policy,
    )
    holdout = result.holdout

    assert result.warmup_bar_count == expected_warmup
    assert holdout.backtest_result.initial_cash == frozen_configuration().backtest.initial_cash
    assert holdout.backtest_result.reconciliation.ledger.starting_cash == (
        frozen_configuration().backtest.initial_cash
    )
    assert holdout.backtest_result.reconciliation.is_reconciled
    assert result.development.backtest_result.reconciliation.is_reconciled
    assert all(
        point.timestamp >= result.split.holdout_start
        for point in holdout.backtest_result.equity_curve
    )
    assert all(
        trade.timestamp >= result.split.holdout_start
        for trade in holdout.backtest_result.trades
    )
    if expected_warmup:
        assert result.warmup_first_timestamp is not None
        assert result.warmup_last_timestamp == SYNTHETIC_START + timedelta(days=7)
        assert result.warmup_last_timestamp < result.split.holdout_start
    else:
        assert result.warmup_first_timestamp is None
        assert result.warmup_last_timestamp is None


def test_inclusive_boundaries_match_report_observations(tmp_path: Path) -> None:
    """Bars exactly on all four declared boundaries are included."""

    result = run_evaluation(tmp_path)

    assert result.development.metadata.dataset.actual_start == (
        result.split.development_start
    )
    assert result.development.metadata.dataset.actual_end == result.split.development_end
    assert result.holdout.metadata.dataset.actual_start == result.split.holdout_start
    assert result.holdout.metadata.dataset.actual_end == result.split.holdout_end
    assert result.development.metadata.dataset.bar_count == 8
    assert result.holdout.metadata.dataset.bar_count == 8


@pytest.mark.parametrize("end_policy", list(EndOfTestPolicy))
def test_benchmark_risk_metrics_and_end_policies_apply_to_both_periods(
    tmp_path: Path,
    end_policy: EndOfTestPolicy,
) -> None:
    """Both runs reuse identical benchmark, metric, and liquidation conventions."""

    result = run_evaluation(
        tmp_path,
        warmup_policy=WarmupPolicy.CARRY_HISTORY,
        benchmark=BenchmarkSelection.BUY_AND_HOLD,
        risk_metrics=True,
        end_policy=end_policy,
    )

    for report in (result.development, result.holdout):
        assert report.benchmark is not None
        assert report.benchmark_comparison is not None
        assert report.risk_adjusted_metrics is not None
        assert report.backtest_result.end_of_test_policy is end_policy
        assert report.backtest_result.reconciliation.is_reconciled
    assert result.development_benchmark_comparison is not None
    assert result.holdout_benchmark_comparison is not None
    assert result.stability_comparison.development_excess_return is not None
    assert result.stability_comparison.holdout_excess_return is not None


def test_repeated_execution_is_deterministic_with_fixed_run_metadata(
    tmp_path: Path,
) -> None:
    """The same bytes, split, and frozen settings produce the same complete result."""

    first = run_evaluation(tmp_path, warmup_policy=WarmupPolicy.CARRY_HISTORY)
    second = run_evaluation(tmp_path, warmup_policy=WarmupPolicy.CARRY_HISTORY)

    assert second == first
    assert second.configuration_sha256 == configuration_sha256(
        frozen_configuration()
    )


def test_bars_after_holdout_end_cannot_affect_holdout_financial_results(
    tmp_path: Path,
) -> None:
    """Future observations remain outside signals, execution, and holdout metrics."""

    baseline_bars = synthetic_bars()
    baseline = run_evaluation(tmp_path, bars=baseline_bars)
    future = replace(
        baseline_bars[-1],
        timestamp=baseline_bars[-1].timestamp + timedelta(days=1),
        open=baseline_bars[-1].open * 10,
        high=baseline_bars[-1].high * 10,
        low=baseline_bars[-1].low * 10,
        close=baseline_bars[-1].close * 10,
    )
    with_future = run_evaluation(tmp_path, bars=baseline_bars + (future,))

    assert with_future.holdout.backtest_result == baseline.holdout.backtest_result
    assert with_future.stability_comparison == baseline.stability_comparison
    assert with_future.quality_summary.bar_count == baseline.quality_summary.bar_count + 1


@pytest.mark.parametrize("period", ["development", "holdout"])
def test_insufficient_isolated_period_data_is_rejected(
    tmp_path: Path,
    period: str,
) -> None:
    """Neither period silently runs without its strategy's minimum observations."""

    bars = synthetic_bars()
    data_path = write_bars_csv(tmp_path / "prices.csv", bars)
    split = evaluation_split()
    if period == "development":
        split = split.model_copy(
            update={"development_end": split.development_start + timedelta(days=1)}
        )
    else:
        split = split.model_copy(
            update={"holdout_end": split.holdout_start + timedelta(days=1)}
        )

    with pytest.raises(MarketDataError, match="insufficient market data"):
        run_split_evaluation(
            evaluation_id="insufficient",
            data_path=data_path,
            symbol="TEST",
            bars=bars,
            split=split,
            configuration=frozen_configuration(),
            provenance=provenance(),
            git_commit=_GIT_COMMIT,
            generated_at=_GENERATED_AT,
        )


def test_configuration_mismatch_is_rejected_by_result_guardrail(tmp_path: Path) -> None:
    """A result cannot represent differently fingerprinted development and holdout runs."""

    result = run_evaluation(tmp_path)

    with pytest.raises(ValueError, match="fingerprints differ"):
        replace(result, holdout_configuration_sha256="0" * 64)
