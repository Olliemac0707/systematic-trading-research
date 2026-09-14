"""Deterministic tests for sealed development-only evaluation."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
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
from test_split_evaluation_batch import evaluation_definition, write_manifest

import trading_research.cli as cli
from trading_research.errors import ReportWriteError
from trading_research.evaluation import (
    CandidateAssessment,
    DevelopmentEvaluationResult,
    DevelopmentEvaluationStatus,
    WarmupPolicy,
    development_evaluation_to_json,
    load_split_evaluation_manifest,
    run_development_evaluation,
    run_development_evaluation_batch,
)
from trading_research.evaluation.development import (
    _average_ranks,
    _select_candidate,
)
from trading_research.experiments import BenchmarkSelection
from trading_research.models import EndOfTestPolicy, ExecutionReason, MarketBar
from trading_research.reporting import installed_application_version

_GENERATED_AT = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)
_GIT_COMMIT = "a" * 40
_MANIFEST_SHA256 = "b" * 64
_ROOT = Path(__file__).resolve().parents[1]


def _run_development(
    tmp_path: Path,
    bars: tuple[MarketBar, ...],
    *,
    warmup_bars: tuple[MarketBar, ...] = (),
) -> DevelopmentEvaluationResult:
    """Run one fixed-time, benchmark-enabled development evaluation."""

    data_path = write_bars_csv(tmp_path / "prices.csv", warmup_bars + bars)
    return run_development_evaluation(
        evaluation_id="test-sma-2-3",
        manifest_sha256=_MANIFEST_SHA256,
        data_path=data_path,
        symbol="TEST",
        bars=warmup_bars + bars,
        split=evaluation_split(),
        configuration=frozen_configuration(
            benchmark=BenchmarkSelection.BUY_AND_HOLD,
            risk_metrics=True,
            end_policy=EndOfTestPolicy.LIQUIDATE,
        ),
        provenance=provenance(),
        git_commit=_GIT_COMMIT,
        generated_at=_GENERATED_AT,
        base_directory=tmp_path,
    )


def test_future_observation_is_rejected_before_strategy_execution(
    tmp_path: Path,
) -> None:
    """The runner refuses any observation after the development boundary."""

    bars = synthetic_bars()
    data_path = write_bars_csv(tmp_path / "prices.csv", bars)

    with pytest.raises(ValueError, match="sealed boundary"):
        run_development_evaluation(
            evaluation_id="sealed",
            manifest_sha256=_MANIFEST_SHA256,
            data_path=data_path,
            symbol="TEST",
            bars=bars,
            split=evaluation_split(),
            configuration=frozen_configuration(),
            provenance=provenance(),
            git_commit=_GIT_COMMIT,
            generated_at=_GENERATED_AT,
        )


def test_final_liquidation_and_benchmark_use_last_development_bar(
    tmp_path: Path,
) -> None:
    """Strategy liquidation and benchmark valuation stop at the declared close."""

    development = synthetic_bars(
        closes=("10", "9", "8", "9", "10", "11", "12", "13")
    )
    result = _run_development(tmp_path, development)
    report = result.report
    final_bar = development[-1]

    assert report.backtest_result.equity_curve[-1].timestamp == final_bar.timestamp
    assert report.backtest_result.trades[-1].timestamp == final_bar.timestamp
    assert report.backtest_result.trades[-1].reference_price == final_bar.close
    assert (
        report.backtest_result.trades[-1].execution_reason
        is ExecutionReason.END_OF_TEST_LIQUIDATION
    )
    assert report.benchmark is not None
    assert report.benchmark.result.equity_curve[-1].timestamp == final_bar.timestamp
    assert report.benchmark.result.trades[-1].reference_price == final_bar.close
    assert report.backtest_result.reconciliation.is_reconciled


def test_carry_history_warmup_is_strictly_before_development(
    tmp_path: Path,
) -> None:
    """Carry-history can use earlier bars but cannot admit future observations."""

    development = synthetic_bars()[:8]
    first = development[0]
    warmup = (
        replace(first, timestamp=SYNTHETIC_START - timedelta(days=3)),
        replace(first, timestamp=SYNTHETIC_START - timedelta(days=2)),
        replace(first, timestamp=SYNTHETIC_START - timedelta(days=1)),
    )
    data_path = write_bars_csv(tmp_path / "prices.csv", warmup + development)
    split = evaluation_split(WarmupPolicy.CARRY_HISTORY)

    result = run_development_evaluation(
        evaluation_id="warmup",
        manifest_sha256=_MANIFEST_SHA256,
        data_path=data_path,
        symbol="TEST",
        bars=warmup + development,
        split=split,
        configuration=frozen_configuration(),
        provenance=provenance(),
        git_commit=_GIT_COMMIT,
        generated_at=_GENERATED_AT,
    )

    assert result.warmup_bar_count == 3
    assert result.warmup_last_timestamp == warmup[-1].timestamp
    assert result.warmup_last_timestamp < split.development_start
    assert all(
        point.timestamp <= split.development_end
        for point in result.report.backtest_result.equity_curve
    )


def test_structured_export_contains_no_future_period_result(tmp_path: Path) -> None:
    """The dedicated schema has only development metrics and sealing evidence."""

    result = _run_development(tmp_path, synthetic_bars()[:8])
    exported = development_evaluation_to_json(result)

    assert "holdout" not in exported.lower()
    assert '"future_observations_contributed": false' in exported
    assert "development_boundary" in exported
    assert "benchmark_final_timestamp" in exported


def test_batch_filters_to_development_and_protects_existing_outputs(
    tmp_path: Path,
) -> None:
    """The batch exports no future result and refuses an existing root."""

    write_bars_csv(tmp_path / "prices.csv", synthetic_bars())
    definition = evaluation_definition("test-sma-2-3")
    definition["end_of_test"] = "liquidate"
    manifest = load_split_evaluation_manifest(
        write_manifest(tmp_path / "manifest.json", [definition])
    )
    output = tmp_path / "development"

    batch = run_development_evaluation_batch(
        manifest,
        output,
        repository=_ROOT,
        generated_at=_GENERATED_AT,
    )

    assert batch.records[0].status is DevelopmentEvaluationStatus.SUCCESS
    assert (output / "test-sma-2-3" / "development-result.json").is_file()
    run_metadata = json.loads(
        (output / "run-metadata.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (output / "development-summary.json").read_text(encoding="utf-8")
    )
    human_report = (output / "development-report.txt").read_text(encoding="utf-8")
    expected_version = installed_application_version()
    assert run_metadata["application_version"] == expected_version
    assert summary["results"][0]["application_version"] == expected_version
    assert f"Application version: {expected_version}" in human_report
    assert not tuple(output.rglob("*holdout*"))
    for path in output.rglob("*"):
        if path.is_file():
            assert "holdout" not in path.read_text(encoding="utf-8").lower()
    with pytest.raises(ReportWriteError, match="already exists"):
        run_development_evaluation_batch(
            manifest,
            output,
            repository=_ROOT,
            generated_at=_GENERATED_AT,
        )


def test_average_ranks_and_frozen_selection_tie_breakers_are_exact() -> None:
    """Exact ties receive average ranks and the declared Sharpe tie-breaker wins."""

    ranks = _average_ranks(
        (
            ("a", Decimal("2")),
            ("b", Decimal("2")),
            ("c", Decimal("1")),
            ("d", Decimal("0")),
        ),
        higher_is_better=True,
    )
    assert ranks == {
        "a": Decimal("1.5"),
        "b": Decimal("1.5"),
        "c": Decimal("3"),
        "d": Decimal("4"),
    }
    def candidate(candidate_id: str, median_sharpe: Decimal) -> CandidateAssessment:
        return CandidateAssessment(
            candidate_id=candidate_id,
            evaluation_count=8,
            success_count=8,
            reconciled_count=8,
            positive_return_count=8,
            positive_sharpe_count=8,
            drawdown_improvement_count=8,
            minimum_closed_trade_count=3,
            assets_below_three_trades=(),
            integrity_warning_count=0,
            eligible=True,
            disqualification_reasons=(),
            overall_rank_score=Decimal("2"),
            median_sharpe_ratio=median_sharpe,
            median_drawdown_magnitude=Decimal("0.1"),
            median_net_return=Decimal("0.2"),
            total_commission_and_slippage=Decimal("100"),
            total_closed_trades=24,
        )

    lower_sharpe = candidate("lower", Decimal("0.5"))
    higher_sharpe = candidate("higher", Decimal("0.6"))

    selected, status = _select_candidate((lower_sharpe, higher_sharpe))

    assert selected == "higher"
    assert status == "SELECTED"


def test_cli_development_only_never_writes_future_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The explicit CLI flag routes only to the sealed batch."""

    write_bars_csv(tmp_path / "prices.csv", synthetic_bars())
    definition = evaluation_definition("test-sma-2-3")
    definition["end_of_test"] = "liquidate"
    manifest = write_manifest(tmp_path / "manifest.json", [definition])
    output = tmp_path / "development"

    exit_code = cli.main(
        [
            "evaluate",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--development-only",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == cli.EXIT_SUCCESS
    assert "DEVELOPMENT-ONLY EVALUATION BATCH" in captured.out
    assert "holdout" not in captured.out.lower()
    assert not tuple(output.rglob("*holdout*"))
