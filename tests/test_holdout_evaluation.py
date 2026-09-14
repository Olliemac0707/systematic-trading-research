"""Deterministic tests for the one-shot sealed holdout path."""

import csv
import json
import tomllib
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

import trading_research.cli as cli
import trading_research.evaluation.holdout as holdout
from trading_research.config import BacktestConfig, PositionSizingMode
from trading_research.errors import BacktestError, ReportWriteError
from trading_research.evaluation import (
    DatasetProvenance,
    DividendTreatment,
    EvaluationSplit,
    FrozenEvaluationConfiguration,
    GeneralisationClassification,
    HoldoutEvaluationResult,
    PriceAdjustmentPolicy,
    SealedHoldoutBatchResult,
    SplitTreatment,
    WarmupPolicy,
    holdout_evaluation_to_json,
    load_split_evaluation_manifest,
    run_holdout_evaluation,
)
from trading_research.evaluation.holdout import (
    FROZEN_DATASET_SHA256,
    FROZEN_DEVELOPMENT_SUMMARY_SHA256,
    FROZEN_GIT_COMMIT,
    FROZEN_MANIFEST_SHA256,
    FROZEN_SELECTION_SHA256,
    DevelopmentBaseline,
    PreparedHoldoutEntry,
    _calculate_aggregate,
    _classify_generalisation,
    _csv_value,
    _development_holdout_comparison,
    _json_value,
    _load_development_baselines,
    _mean,
    _median,
    _prepare_frozen_plan,
    _prepare_output_directory,
    _validate_selection_document,
    _write_holdout_outputs,
)
from trading_research.experiments import BenchmarkSelection
from trading_research.models import EndOfTestPolicy, MarketBar
from trading_research.performance import RiskMetricSettings
from trading_research.reporting import StrategyRunConfiguration, sha256_file
from trading_research.strategies import (
    SmaCrossoverParameters,
    StrategyName,
)

_ROOT = Path(__file__).resolve().parents[1]
_HOLDOUT_START = datetime(2019, 1, 2, tzinfo=UTC)
_GENERATED_AT = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)


def _configuration() -> FrozenEvaluationConfiguration:
    """Return the exact frozen SMA 50/200 financial configuration."""

    return FrozenEvaluationConfiguration(
        strategy=StrategyRunConfiguration(
            name=StrategyName.SMA_CROSSOVER,
            parameters=SmaCrossoverParameters(
                fast_window=50,
                slow_window=200,
            ),
        ),
        backtest=BacktestConfig(
            initial_cash=Decimal("100000"),
            trade_quantity=10,
            commission_bps=Decimal("1"),
            slippage_bps=Decimal("5"),
            position_sizing_mode=PositionSizingMode.CASH_ALLOCATION,
            cash_allocation_ratio=Decimal("1.0"),
            end_of_test_policy=EndOfTestPolicy.LIQUIDATE,
        ),
        risk_metrics=RiskMetricSettings(
            periods_per_year=Decimal("252"),
            risk_free_rate_per_period=Decimal("0"),
            target_return_per_period=Decimal("0"),
        ),
        benchmark=BenchmarkSelection.BUY_AND_HOLD,
    )


def _provenance(symbol: str) -> DatasetProvenance:
    """Return approved-semantics synthetic provenance without provider access."""

    return DatasetProvenance(
        dataset_id=f"synthetic-{symbol.lower()}",
        source_name="Yahoo Finance via yfinance",
        price_adjustment=PriceAdjustmentPolicy.SPLIT_ADJUSTED,
        dividend_treatment=DividendTreatment.EXCLUDED,
        split_treatment=SplitTreatment.ADJUSTED,
        bar_frequency="daily",
        volume_approved_for_strategy_signals=False,
    )


def _bars(symbol: str, holdout_count: int = 8) -> tuple[MarketBar, ...]:
    """Return 200 warm-up bars and a deterministic holdout crossover."""

    warmup_start = _HOLDOUT_START - timedelta(days=200)
    closes = (Decimal("100"),) * 200 + (Decimal("110"),) * holdout_count
    return tuple(
        MarketBar(
            symbol=symbol,
            timestamp=(
                warmup_start + timedelta(days=index)
                if index < 200
                else _HOLDOUT_START + timedelta(days=index - 200)
            ),
            open=close,
            high=close + Decimal("1"),
            low=close - Decimal("1"),
            close=close,
            volume=1000,
        )
        for index, close in enumerate(closes)
    )


def _write_bars(path: Path, bars: tuple[MarketBar, ...]) -> Path:
    """Write synthetic bars through the project CSV schema."""

    lines = ["timestamp,symbol,open,high,low,close,volume"]
    lines.extend(
        ",".join(
            (
                bar.timestamp.isoformat(),
                bar.symbol,
                str(bar.open),
                str(bar.high),
                str(bar.low),
                str(bar.close),
                str(bar.volume),
            )
        )
        for bar in bars
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _split(holdout_count: int = 8) -> EvaluationSplit:
    """Return a short deterministic holdout with the frozen carry-history policy."""

    return EvaluationSplit(
        development_start=datetime(2018, 1, 1, tzinfo=UTC),
        development_end=datetime(2018, 12, 31, 23, 59, 59, tzinfo=UTC),
        holdout_start=_HOLDOUT_START,
        holdout_end=_HOLDOUT_START + timedelta(days=holdout_count - 1),
        warmup_policy=WarmupPolicy.CARRY_HISTORY,
    )


def _run_one(tmp_path: Path, symbol: str) -> HoldoutEvaluationResult:
    """Run one synthetic selected-candidate holdout."""

    bars = _bars(symbol)
    path = _write_bars(tmp_path / f"{symbol.lower()}.csv", bars)
    return run_holdout_evaluation(
        evaluation_id=f"{symbol.lower()}-sma-50-200",
        selection_record_sha256=FROZEN_SELECTION_SHA256,
        manifest_sha256=FROZEN_MANIFEST_SHA256,
        data_path=path,
        symbol=symbol,
        bars=bars,
        split=_split(),
        configuration=_configuration(),
        provenance=_provenance(symbol),
        git_commit=FROZEN_GIT_COMMIT,
        generated_at=_GENERATED_AT,
        base_directory=tmp_path,
    )


def _contains_float(value: object) -> bool:
    """Return whether a parsed JSON tree contains binary floating point."""

    if isinstance(value, float):
        return True
    if isinstance(value, dict):
        return any(_contains_float(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_float(item) for item in value)
    return False


def _development_baseline(
    result: HoldoutEvaluationResult,
) -> DevelopmentBaseline:
    """Build comparison data from one synthetic result."""

    report = result.report
    comparison = report.benchmark_comparison
    assert comparison is not None
    risk = report.risk_adjusted_metrics
    assert risk is not None
    exposure = report.exposure_statistics.time_in_market_ratio
    assert exposure is not None
    return DevelopmentBaseline(
        symbol=report.metadata.dataset.symbol,
        net_return=report.performance.total_return,
        periodic_sharpe_ratio=(
            Decimal("0")
            if risk.periodic_sharpe_ratio is None
            else risk.periodic_sharpe_ratio
        ),
        maximum_percentage_drawdown=(
            report.performance.maximum_percentage_drawdown
        ),
        benchmark_return=comparison.benchmark_total_return,
        benchmark_maximum_percentage_drawdown=(
            comparison.benchmark_maximum_percentage_drawdown
        ),
        closed_trade_count=report.trade_statistics.completed_trade_count,
        time_in_market_ratio=exposure,
    )


def test_frozen_selection_and_manifest_identify_only_eight_sma_50_200() -> None:
    """Tracked pre-holdout evidence names one universal candidate."""

    selection_path = (
        _ROOT / "experiments" / "approved-etf-study-development-selection.toml"
    )
    manifest_path = _ROOT / "experiments" / "approved-etf-study.toml"
    with selection_path.open("rb") as handle:
        selection = tomllib.load(handle)

    _validate_selection_document(selection)
    manifest = load_split_evaluation_manifest(manifest_path)
    selected = tuple(
        entry
        for entry in manifest.entries
        if entry.evaluation_id.endswith("-sma-50-200")
    )

    assert sha256_file(selection_path) == FROZEN_SELECTION_SHA256
    assert sha256_file(manifest_path) == FROZEN_MANIFEST_SHA256
    assert len(selected) == 8
    assert all(entry.definition is not None for entry in selected)
    assert [
        entry.definition.symbol
        for entry in selected
        if entry.definition is not None
    ] == ["SPY", "QQQ", "IWM", "XLF", "XLE", "XLV", "TLT", "GLD"]


def test_completed_holdout_record_freezes_failed_generalisation_gate() -> None:
    """The compact tracked record preserves the result without reusable holdout data."""

    record_path = (
        _ROOT / "experiments" / "approved-etf-study-holdout-result.toml"
    )
    with record_path.open("rb") as handle:
        record = tomllib.load(handle)

    assert record["application_version"] == "0.5.0"
    assert record["source_application_version"] == "0.4.0"
    assert record["holdout_attempt"] == 1
    assert record["holdout_reusable_for_selection"] is False
    assert record["authorised_evaluation_count"] == 8
    assert record["completed_evaluation_count"] == 8
    assert record["all_evaluations_used_sma_50_200"] is True
    assert record["all_evaluations_reconciled_exactly"] is True
    assert record["no_other_candidate_or_parameter_evaluated"] is True
    assert record["no_post_holdout_tuning"] is True
    assert record["integrity_status"] == "passed"
    assert record["generalisation_status"] == "not_supported"
    assert record["classification"] == (
        "GENERALISATION NOT SUPPORTED BY PRE-REGISTERED GATE"
    )
    assert record["positive_return_assets"] == 6
    assert record["positive_sharpe_assets"] == 7
    assert record["smaller_drawdown_assets"] == 3
    assert [item["symbol"] for item in record["asset_outcomes"]] == [
        "SPY",
        "QQQ",
        "IWM",
        "XLF",
        "XLE",
        "XLV",
        "TLT",
        "GLD",
    ]
    assert record["aggregate"]["total_simulated_costs"] == (
        "7013.18617629300151386187780"
    )


def test_frozen_plan_validates_all_local_evidence_without_evaluating(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The execution gate remains deterministic without ignored local artifacts."""

    manifest = load_split_evaluation_manifest(
        _ROOT / "experiments" / "approved-etf-study.toml"
    )
    selection_path = (
        _ROOT / "experiments" / "approved-etf-study-development-selection.toml"
    )
    development_summary = tmp_path / "development-summary.csv"
    development_summary.write_text(
        (
            "candidate_id,symbol,net_return,periodic_sharpe_ratio,"
            "maximum_percentage_drawdown,benchmark_return,"
            "benchmark_maximum_percentage_drawdown,closed_trade_count,"
            "time_in_market_ratio\n"
            + "".join(
                f"sma-50-200,{symbol},1,0.1,-0.2,2,-0.3,4,0.5\n"
                for symbol in FROZEN_DATASET_SHA256
            )
        ),
        encoding="utf-8",
    )
    original_is_file = Path.is_file

    def fake_is_file(path: Path) -> bool:
        if "approved-etf-study" in path.parts and path.suffix == ".csv":
            return True
        return original_is_file(path)

    def fake_sha256(path: Path) -> str:
        candidate = Path(path)
        if candidate == selection_path:
            return FROZEN_SELECTION_SHA256
        if candidate == manifest.source_path:
            return FROZEN_MANIFEST_SHA256
        if candidate == development_summary:
            return FROZEN_DEVELOPMENT_SUMMARY_SHA256
        for symbol, expected in FROZEN_DATASET_SHA256.items():
            if symbol in candidate.parts:
                return expected
        raise AssertionError(f"unexpected hash request: {candidate.name}")

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    monkeypatch.setattr(holdout, "sha256_file", fake_sha256)
    monkeypatch.setattr(
        holdout,
        "current_git_commit",
        lambda repository: FROZEN_GIT_COMMIT,
    )
    monkeypatch.setattr(
        holdout,
        "load_dataset_provenance",
        lambda dataset, sidecar: _provenance(Path(dataset).parent.name),
    )

    prepared, selection = _prepare_frozen_plan(
        manifest,
        selection_record=selection_path,
        repository=_ROOT,
    )
    baselines = _load_development_baselines(development_summary, selection)

    assert [item.entry.evaluation_id for item in prepared] == [
        "spy-sma-50-200",
        "qqq-sma-50-200",
        "iwm-sma-50-200",
        "xlf-sma-50-200",
        "xle-sma-50-200",
        "xlv-sma-50-200",
        "tlt-sma-50-200",
        "gld-sma-50-200",
    ]
    assert tuple(baselines) == (
        "SPY",
        "QQQ",
        "IWM",
        "XLF",
        "XLE",
        "XLV",
        "TLT",
        "GLD",
    )


def test_holdout_warmup_has_no_financial_activity_and_execution_is_next_bar(
    tmp_path: Path,
) -> None:
    """The first crossover is observed on day one and can execute only later."""

    result = _run_one(tmp_path, "SPY")

    assert result.warmup_bar_count == 200
    assert result.warmup_last_timestamp < result.split.holdout_start
    assert result.first_signal_timestamp == result.split.holdout_start
    assert result.first_execution_timestamp == (
        result.split.holdout_start + timedelta(days=1)
    )
    assert (
        result.report.metadata.dataset.actual_start
        == result.split.holdout_start
    )
    assert all(
        point.timestamp >= result.split.holdout_start
        for point in result.report.backtest_result.equity_curve
    )
    assert result.report.backtest_result.reconciliation.is_reconciled
    assert result.integrity_warnings == ()


def test_holdout_rejects_an_observation_after_the_sealed_end(
    tmp_path: Path,
) -> None:
    """Future data is rejected before the strategy pipeline can execute."""

    bars = _bars("SPY")
    extra = MarketBar(
        symbol="SPY",
        timestamp=_split().holdout_end + timedelta(days=1),
        open=Decimal("110"),
        high=Decimal("111"),
        low=Decimal("109"),
        close=Decimal("110"),
        volume=1000,
    )
    path = _write_bars(tmp_path / "spy.csv", bars + (extra,))

    with pytest.raises(ValueError, match="after its sealed boundary"):
        run_holdout_evaluation(
            evaluation_id="spy-sma-50-200",
            selection_record_sha256=FROZEN_SELECTION_SHA256,
            manifest_sha256=FROZEN_MANIFEST_SHA256,
            data_path=path,
            symbol="SPY",
            bars=bars + (extra,),
            split=_split(),
            configuration=_configuration(),
            provenance=_provenance("SPY"),
            git_commit=FROZEN_GIT_COMMIT,
        )


def test_holdout_rejects_invalid_or_insufficient_bar_sequences(
    tmp_path: Path,
) -> None:
    """Input validation stops before pipeline execution for malformed history."""

    def evaluate(bars: tuple[MarketBar, ...]) -> None:
        run_holdout_evaluation(
            evaluation_id="spy-sma-50-200",
            selection_record_sha256=FROZEN_SELECTION_SHA256,
            manifest_sha256=FROZEN_MANIFEST_SHA256,
            data_path=tmp_path / "unused.csv",
            symbol="SPY",
            bars=bars,
            split=_split(),
            configuration=_configuration(),
            provenance=_provenance("SPY"),
            git_commit=FROZEN_GIT_COMMIT,
        )

    valid = _bars("SPY")
    with pytest.raises(ValueError, match="requires market bars"):
        evaluate(())
    with pytest.raises(ValueError, match="match the requested symbol"):
        evaluate(_bars("QQQ"))
    with pytest.raises(ValueError, match="strictly chronological"):
        evaluate(valid[:1] + valid[:1])
    with pytest.raises(ValueError, match="insufficient pre-holdout"):
        evaluate(valid[:199])


def test_holdout_json_distinguishes_integrity_and_contains_no_float(
    tmp_path: Path,
) -> None:
    """Structured output keeps exact numbers as strings and records sealing facts."""

    result = _run_one(tmp_path, "SPY")
    exported = holdout_evaluation_to_json(result)
    payload = json.loads(exported)

    assert payload["integrity"] == {
        "warmup_contributed_financial_activity": False,
        "future_observations_contributed": False,
        "live_network_request": False,
        "volume_signal_used": False,
        "reconciliation_status": "PASS",
        "warnings": [],
    }
    assert isinstance(
        payload["report"]["performance"]["ending_equity"],
        str,
    )
    assert not _contains_float(payload)


def test_exact_holdout_serialisation_helpers_cover_unavailable_edges() -> None:
    """Decimal aggregation stays exact and unsupported export values fail loudly."""

    class PlainEnum(Enum):
        VALUE = "value"

    assert _mean((Decimal("1"), Decimal("2"))) == Decimal("1.5")
    assert _median((Decimal("3"), Decimal("1"), Decimal("2"))) == Decimal("2")
    with pytest.raises(ValueError, match="mean requires"):
        _mean(())
    with pytest.raises(ValueError, match="median requires"):
        _median(())
    assert _csv_value(EndOfTestPolicy.LIQUIDATE) == "liquidate"
    assert _json_value(PlainEnum.VALUE) == "value"
    with pytest.raises(TypeError, match="unsupported holdout export value"):
        _json_value(object())


def test_generalisation_classification_uses_only_preregistered_counts() -> None:
    """Integrity precedes the exact six/six/five performance thresholds."""

    supported = _classify_generalisation(
        integrity_gate_passed=True,
        positive_return_count=6,
        positive_sharpe_count=6,
        drawdown_improvement_count=5,
    )
    not_supported = _classify_generalisation(
        integrity_gate_passed=True,
        positive_return_count=5,
        positive_sharpe_count=6,
        drawdown_improvement_count=5,
    )
    invalid = _classify_generalisation(
        integrity_gate_passed=False,
        positive_return_count=8,
        positive_sharpe_count=8,
        drawdown_improvement_count=8,
    )

    assert supported[0] is GeneralisationClassification.SUPPORTED
    assert not_supported[0] is GeneralisationClassification.NOT_SUPPORTED
    assert not_supported[1] == (
        "positive net holdout return occurred in 5 of 8 assets",
    )
    assert invalid[0] is GeneralisationClassification.INVALID


def test_complete_synthetic_batch_writes_all_required_outputs(
    tmp_path: Path,
) -> None:
    """Aggregate exports appear only for a complete ordered eight-result batch."""

    results = tuple(_run_one(tmp_path, symbol) for symbol in (
        "SPY",
        "QQQ",
        "IWM",
        "XLF",
        "XLE",
        "XLV",
        "TLT",
        "GLD",
    ))
    baselines = {
        result.report.metadata.dataset.symbol: _development_baseline(result)
        for result in results
    }
    aggregate = _calculate_aggregate(results)
    rows, comparison_summary = _development_holdout_comparison(
        results,
        baselines,
    )
    batch = SealedHoldoutBatchResult(
        source_manifest="approved-etf-study.toml",
        selection_record="approved-etf-study-development-selection.toml",
        development_summary="development-summary.csv",
        manifest_sha256=FROZEN_MANIFEST_SHA256,
        selection_record_sha256=FROZEN_SELECTION_SHA256,
        development_summary_sha256="a" * 64,
        git_commit=FROZEN_GIT_COMMIT,
        application_version="0.4.0",
        generated_at=_GENERATED_AT,
        run_id="holdout-test",
        results=results,
        aggregate=aggregate,
        development_comparison_rows=rows,
        development_comparison_summary=comparison_summary,
    )
    output = tmp_path / "complete"
    output.mkdir()

    _write_holdout_outputs(batch, output)

    required = {
        "holdout-summary.csv",
        "holdout-summary.json",
        "holdout-report.txt",
        "benchmark-comparison.csv",
        "development-holdout-comparison.csv",
        "generalisation-assessment.json",
        "run-metadata.json",
    }
    assert required.issubset(path.name for path in output.iterdir())
    assert len(tuple(output.glob("*/holdout-result.json"))) == 8
    with (output / "holdout-summary.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        assert len(tuple(csv.DictReader(handle))) == 8
    report = (output / "holdout-report.txt").read_text(encoding="utf-8")
    assert "first sealed holdout evaluation" in report.lower()
    assert "No alternative strategy or parameter was tested" in report


def test_one_shot_output_directory_has_no_overwrite_path(tmp_path: Path) -> None:
    """The sealed holdout cannot reuse any existing output directory."""

    output = tmp_path / "holdout"
    _prepare_output_directory(output)

    with pytest.raises(ReportWriteError, match="already exists"):
        _prepare_output_directory(output)


def test_sealed_batch_orchestrates_only_the_prepared_eight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The one-shot coordinator preserves order and exports only after completion."""

    manifest = load_split_evaluation_manifest(
        _ROOT / "experiments" / "approved-etf-study.toml"
    )
    results = tuple(
        _run_one(tmp_path, symbol)
        for symbol in ("SPY", "QQQ", "IWM", "XLF", "XLE", "XLV", "TLT", "GLD")
    )
    results_by_symbol = {
        result.report.metadata.dataset.symbol: result for result in results
    }
    entries_by_id = {entry.evaluation_id: entry for entry in manifest.entries}
    prepared = tuple(
        PreparedHoldoutEntry(
            entry=entries_by_id[result.evaluation_id],
            dataset=tmp_path / f"{symbol.lower()}.csv",
            provenance=_provenance(symbol),
            configuration=_configuration(),
        )
        for symbol, result in zip(results_by_symbol, results, strict=True)
    )
    baselines = {
        symbol: _development_baseline(result)
        for symbol, result in results_by_symbol.items()
    }
    seen: list[str] = []

    monkeypatch.setattr(
        holdout,
        "_prepare_frozen_plan",
        lambda *args, **kwargs: (prepared, {"artifact_sha256": {}}),
    )
    monkeypatch.setattr(
        holdout,
        "_load_development_baselines",
        lambda *args, **kwargs: baselines,
    )
    monkeypatch.setattr(
        holdout,
        "_load_holdout_source_bars",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(
        holdout,
        "current_git_commit",
        lambda repository: FROZEN_GIT_COMMIT,
    )

    def fake_run(**kwargs: object) -> HoldoutEvaluationResult:
        symbol = str(kwargs["symbol"])
        seen.append(symbol)
        return results_by_symbol[symbol]

    monkeypatch.setattr(holdout, "run_holdout_evaluation", fake_run)
    output = tmp_path / "orchestrated"

    batch = holdout.run_sealed_holdout_batch(
        manifest,
        selection_record=tmp_path / "selection.toml",
        development_summary=tmp_path / "development.csv",
        output_directory=output,
        repository=tmp_path,
        generated_at=_GENERATED_AT,
    )

    assert seen == ["SPY", "QQQ", "IWM", "XLF", "XLE", "XLV", "TLT", "GLD"]
    assert batch.results == results
    assert batch.aggregate.integrity_gate_passed
    assert (output / "holdout-summary.json").is_file()


def test_sealed_batch_failure_exports_diagnostic_without_performance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An operational failure records a redacted diagnostic and no partial results."""

    manifest = load_split_evaluation_manifest(
        _ROOT / "experiments" / "approved-etf-study.toml"
    )
    entry = next(
        item for item in manifest.entries if item.evaluation_id == "spy-sma-50-200"
    )
    prepared = (
        PreparedHoldoutEntry(
            entry=entry,
            dataset=tmp_path / "spy.csv",
            provenance=_provenance("SPY"),
            configuration=_configuration(),
        ),
    )
    monkeypatch.setattr(
        holdout,
        "_prepare_frozen_plan",
        lambda *args, **kwargs: (prepared, {"artifact_sha256": {}}),
    )
    monkeypatch.setattr(
        holdout,
        "_load_development_baselines",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        holdout,
        "_load_holdout_source_bars",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(
        holdout,
        "current_git_commit",
        lambda repository: FROZEN_GIT_COMMIT,
    )
    monkeypatch.setattr(
        holdout,
        "run_holdout_evaluation",
        lambda **kwargs: (_ for _ in ()).throw(
            ValueError(f"failed under {tmp_path}")
        ),
    )
    output = tmp_path / "failed"

    with pytest.raises(
        BacktestError,
        match="no partial performance report was exported",
    ):
        holdout.run_sealed_holdout_batch(
            manifest,
            selection_record=tmp_path / "selection.toml",
            development_summary=tmp_path / "development.csv",
            output_directory=output,
            repository=tmp_path,
            generated_at=_GENERATED_AT,
        )

    assert [path.name for path in output.iterdir()] == [
        "failure-diagnostic.json"
    ]
    diagnostic = json.loads(
        (output / "failure-diagnostic.json").read_text(encoding="utf-8")
    )
    assert diagnostic["evaluation_id"] == "spy-sma-50-200"
    assert diagnostic["partial_performance_exported"] is False
    assert diagnostic["message"] == "failed under <repository>"


def test_cli_holdout_command_routes_only_to_sealed_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The dedicated CLI has no strategy or parameter option."""

    classification = SimpleNamespace(
        value=GeneralisationClassification.SUPPORTED.value
    )
    aggregate = SimpleNamespace(
        integrity_gate_passed=True,
        classification=classification,
    )
    fake_result = SimpleNamespace(results=(), aggregate=aggregate)
    called: list[object] = []

    def fake_load(path: Path) -> object:
        called.append(path)
        return object()

    monkeypatch.setattr(
        cli,
        "load_split_evaluation_manifest",
        fake_load,
    )
    monkeypatch.setattr(
        cli,
        "run_sealed_holdout_batch",
        lambda *args, **kwargs: fake_result,
    )

    exit_code = cli.main(
        [
            "holdout",
            "--manifest",
            str(tmp_path / "manifest.toml"),
            "--selection-record",
            str(tmp_path / "selection.toml"),
            "--development-summary",
            str(tmp_path / "development.csv"),
            "--output-dir",
            str(tmp_path / "holdout"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == cli.EXIT_SUCCESS
    assert len(called) == 1
    assert "SEALED HOLDOUT EVALUATION COMPLETE" in captured.out
    assert "SMA 20" not in captured.out
