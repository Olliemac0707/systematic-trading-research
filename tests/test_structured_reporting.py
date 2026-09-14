"""Tests for reproducible metadata and structured report exports."""

import csv
import json
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import Never

import pytest

from trading_research.backtesting import SimpleBacktestEngine
from trading_research.config import BacktestConfig, EndOfTestPolicy
from trading_research.data import CsvMarketDataProvider
from trading_research.errors import ReportWriteError
from trading_research.models import ExecutionReason
from trading_research.performance import (
    RiskMetricSettings,
    calculate_exposure_statistics,
    calculate_performance,
    calculate_risk_adjusted_metrics,
    calculate_trade_statistics,
    construct_return_series,
    reconstruct_closed_trades,
)
from trading_research.reporting import (
    CLOSED_TRADES_CSV_COLUMNS,
    DECISIONS_CSV_COLUMNS,
    EQUITY_CSV_COLUMNS,
    STRUCTURED_REPORT_SCHEMA_VERSION,
    SUMMARY_CSV_COLUMNS,
    BacktestRunMetadata,
    DatasetIdentity,
    StructuredBacktestReport,
    build_structured_backtest_report,
    closed_trades_to_csv,
    create_backtest_run_metadata,
    decisions_to_csv,
    equity_curve_to_csv,
    prepare_equity_curve_rows,
    safe_data_identifier,
    sha256_file,
    structured_report_to_json,
    summary_report_to_csv,
    write_decisions_csv,
    write_json_report,
)
from trading_research.strategies import SimpleMovingAverageCrossover

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_DATA = _ROOT / "data" / "sample_prices.csv"
_GENERATED_AT = datetime(2026, 7, 18, 12, 30, tzinfo=UTC)


def make_report(
    *,
    end_of_test_policy: EndOfTestPolicy = EndOfTestPolicy.HOLD,
    bar_count: int = 8,
    risk_settings: RiskMetricSettings | None = None,
    data_path: Path = _SAMPLE_DATA,
    generated_at: datetime = _GENERATED_AT,
) -> StructuredBacktestReport:
    """Build a deterministic report through the public integration APIs."""

    bars = tuple(
        CsvMarketDataProvider(data_path).get_historical_bars(
            symbol="DEMO",
            start=None,
            end=None,
        )
    )[:bar_count]
    config = BacktestConfig(
        initial_cash=Decimal("10000"),
        trade_quantity=10,
        commission_bps=Decimal("1"),
        slippage_bps=Decimal("5"),
        maximum_position_value=Decimal("5000"),
        minimum_cash_reserve=Decimal("1000"),
        end_of_test_policy=end_of_test_policy,
    )
    result = SimpleBacktestEngine().run(
        bars,
        SimpleMovingAverageCrossover(short_window=2, long_window=3),
        config,
    )
    metadata = create_backtest_run_metadata(
        data_path=data_path,
        bars=bars,
        symbol="DEMO",
        requested_start=None,
        requested_end=None,
        fast_window=2,
        slow_window=3,
        config=config,
        risk_metric_settings=risk_settings,
        run_id="deterministic-run-id",
        generated_at=generated_at,
        application_version="0.1.0-test",
        git_commit=None,
        base_directory=_ROOT,
    )
    return build_structured_backtest_report(result, metadata)


def csv_rows(text: str) -> list[dict[str, str]]:
    """Parse CSV text while preserving its header names."""

    return list(csv.DictReader(StringIO(text)))


def assert_no_float(value: object) -> None:
    """Recursively reject binary floating-point values in parsed JSON."""

    assert not isinstance(value, float)
    if isinstance(value, dict):
        for child in value.values():
            assert_no_float(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_float(child)


def test_metadata_is_immutable_aware_and_retains_all_run_assumptions() -> None:
    """One record retains deterministic identity, policy, and risk configuration."""

    settings = RiskMetricSettings(
        periods_per_year=Decimal("252"),
        risk_free_rate_per_period=Decimal("0.0001"),
        target_return_per_period=Decimal("0"),
    )
    metadata = make_report(risk_settings=settings).metadata

    assert metadata.run_id == "deterministic-run-id"
    assert metadata.generated_at == _GENERATED_AT
    assert metadata.generated_at.utcoffset() is not None
    assert metadata.application_version == "0.1.0-test"
    assert metadata.git_commit is None
    assert metadata.strategy.name == "sma-crossover"
    assert metadata.strategy.fast_window == 2
    assert metadata.strategy.slow_window == 3
    assert metadata.simulation.starting_cash == Decimal("10000")
    assert metadata.simulation.commission_bps == Decimal("1")
    assert metadata.simulation.slippage_bps == Decimal("5")
    assert metadata.position_sizing.quantity == 10
    assert metadata.risk_policy.name == "composite"
    assert metadata.risk_policy.maximum_position_value == Decimal("5000")
    assert metadata.risk_policy.minimum_cash_reserve == Decimal("1000")
    assert metadata.risk_metric_settings == settings
    with pytest.raises(FrozenInstanceError):
        metadata.run_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("policy", list(EndOfTestPolicy))
def test_metadata_retains_each_end_of_test_policy(policy: EndOfTestPolicy) -> None:
    """Hold and liquidation conventions remain explicit metadata assumptions."""

    report = make_report(end_of_test_policy=policy, bar_count=5)

    assert report.metadata.simulation.end_of_test_policy is policy
    assert report.backtest_result.end_of_test_policy is policy


def test_default_run_identity_and_timestamp_are_created_once_and_are_valid() -> None:
    """Callers may omit run-only identity values without affecting calculations."""

    report = make_report()
    metadata = report.metadata
    bars = tuple(
        CsvMarketDataProvider(_SAMPLE_DATA).get_historical_bars("DEMO", None, None)
    )
    generated = create_backtest_run_metadata(
        data_path=_SAMPLE_DATA,
        bars=bars,
        symbol="DEMO",
        requested_start=None,
        requested_end=None,
        fast_window=2,
        slow_window=3,
        config=BacktestConfig(initial_cash=Decimal("10000"), trade_quantity=10),
        risk_metric_settings=None,
        base_directory=_ROOT,
    )

    assert generated.run_id
    assert generated.generated_at.tzinfo is not None
    assert generated.generated_at.utcoffset() is not None
    assert report.backtest_result.final_equity == Decimal("9959.8679980")
    assert metadata.simulation.starting_cash == generated.simulation.starting_cash


def test_invalid_metadata_timestamp_is_rejected() -> None:
    """An ambiguous generated timestamp cannot enter run metadata."""

    metadata = make_report().metadata

    with pytest.raises(ValueError, match="timezone-aware"):
        replace(metadata, generated_at=datetime(2026, 1, 1))


def test_invalid_metadata_bar_count_is_rejected() -> None:
    """An empty validated dataset cannot enter run metadata."""

    dataset = make_report().metadata.dataset

    with pytest.raises(ValueError, match="bar_count must be positive"):
        replace(dataset, bar_count=0)


def test_sha256_identity_is_deterministic_read_only_and_path_safe(tmp_path: Path) -> None:
    """Dataset identity hashes exact bytes and never exports the absolute path."""

    source = tmp_path / "nested" / "known.csv"
    source.parent.mkdir()
    source.write_bytes(b"known\x00bytes\r\n")
    before = source.read_bytes()
    before_timestamp = source.stat().st_mtime_ns

    first = sha256_file(source)
    second = sha256_file(source)

    assert first == second == "cfb4835e041c1ef96d7f1344260a0cfde6ed54ef6519ad8fa1d102a2be5853eb"
    assert source.read_bytes() == before
    assert source.stat().st_mtime_ns == before_timestamp
    assert safe_data_identifier(source, base_directory=tmp_path) == "nested/known.csv"
    unrelated_base = tmp_path.parent / "unrelated-base"
    assert safe_data_identifier(source, base_directory=unrelated_base) == "known.csv"
    assert str(tmp_path) not in safe_data_identifier(
        source,
        base_directory=unrelated_base,
    )


def test_missing_dataset_hash_source_has_clear_error(tmp_path: Path) -> None:
    """A missing source cannot silently produce an absent or invented digest."""

    with pytest.raises(FileNotFoundError, match="data file does not exist"):
        sha256_file(tmp_path / "missing.csv")


def test_dataset_identity_rejects_absolute_identifiers() -> None:
    """The validated export record prevents accidental user-path disclosure."""

    dataset = make_report().metadata.dataset

    with pytest.raises(ValueError, match="must not be an absolute path"):
        replace(dataset, identifier=r"C:\Users\example\prices.csv")
    with pytest.raises(ValueError, match="parent-directory traversal"):
        replace(dataset, identifier="../prices.csv")


def test_dataset_identity_rejects_naive_actual_timestamp() -> None:
    """Actual dataset bounds remain unambiguous and timezone-aware."""

    dataset = make_report().metadata.dataset

    with pytest.raises(ValueError, match="timezone-aware"):
        replace(dataset, actual_start=datetime(2025, 1, 1))


def test_structured_report_reuses_authoritative_calculation_results() -> None:
    """Aggregate values equal the existing public performance APIs exactly."""

    report = make_report(
        risk_settings=RiskMetricSettings(periods_per_year=Decimal("252"))
    )
    result = report.backtest_result
    settings = report.metadata.risk_metric_settings
    assert settings is not None

    assert report.schema_version == STRUCTURED_REPORT_SCHEMA_VERSION
    assert report.performance == calculate_performance(result)
    assert report.closed_trades == reconstruct_closed_trades(result)
    assert report.trade_statistics == calculate_trade_statistics(result)
    assert report.return_series == construct_return_series(result)
    assert report.exposure_statistics == calculate_exposure_statistics(result)
    assert report.pre_trade_decisions == result.pre_trade_decisions
    assert report.risk_adjusted_metrics == calculate_risk_adjusted_metrics(
        result,
        settings,
    )
    assert report.performance.ending_cash == result.final_cash
    assert report.performance.ending_equity == result.final_equity
    assert result.reconciliation.cash_difference == Decimal("0")
    assert result.reconciliation.reconciliation_difference == Decimal("0")


def test_hold_report_retains_open_position_and_liquidation_retains_exit_reason() -> None:
    """Structured output preserves both terminal-position conventions."""

    hold = make_report(end_of_test_policy=EndOfTestPolicy.HOLD, bar_count=5)
    liquidated = make_report(
        end_of_test_policy=EndOfTestPolicy.LIQUIDATE,
        bar_count=5,
    )

    assert hold.backtest_result.final_position_quantity == 10
    assert hold.performance.ending_cash != hold.performance.ending_equity
    assert hold.closed_trades == ()
    assert liquidated.backtest_result.final_position_quantity == 0
    assert liquidated.performance.ending_cash == liquidated.performance.ending_equity
    assert liquidated.closed_trades[-1].exit_execution_reason is (
        ExecutionReason.END_OF_TEST_LIQUIDATION
    )


def test_json_is_deterministic_valid_and_uses_strings_for_exact_values() -> None:
    """JSON retains nulls, aware ISO times, tuple order, and no binary floats."""

    report = make_report()
    first = structured_report_to_json(report)
    second = structured_report_to_json(report)
    payload = json.loads(first)

    assert first == second
    assert payload["schema_version"] == "1.3"
    assert payload["metadata"]["git_commit"] is None
    assert payload["metadata"]["dataset"]["requested_start"] is None
    assert payload["metadata"]["generated_at"] == _GENERATED_AT.isoformat()
    assert payload["performance"]["ending_equity"] == str(
        report.performance.ending_equity
    )
    assert payload["performance"]["total_return"] == str(
        report.performance.total_return
    )
    assert payload["trade_statistics"]["average_holding_period"] == "259200"
    assert payload["risk_adjusted_metrics"] is None
    assert payload["closed_trades"][0]["symbol"] == "DEMO"
    assert payload["exposure_statistics"]["observation_count"] == len(
        report.backtest_result.equity_curve
    )
    assert payload["exposure_statistics"]["time_in_market_ratio"] == str(
        report.exposure_statistics.time_in_market_ratio
    )
    assert payload["exposure_statistics"]["average_position_value_ratio"] == str(
        report.exposure_statistics.average_position_value_ratio
    )
    assert payload["pre_trade_decisions"][0]["outcome"] == "approved"
    assert payload["pre_trade_decisions"][0]["reference_price"] == str(
        report.pre_trade_decisions[0].reference_price
    )
    assert_no_float(payload)
    assert str(_ROOT) not in first


def test_json_writer_is_utf8_newline_terminated_and_protects_existing_file(
    tmp_path: Path,
) -> None:
    """Explicit overwrite replaces a complete UTF-8 file and default refuses."""

    report = make_report()
    report = replace(
        report,
        metadata=replace(report.metadata, application_version="rélease-α"),
    )
    destination = tmp_path / "nested" / "report.json"

    write_json_report(report, destination, create_parents=True)
    first_bytes = destination.read_bytes()

    assert first_bytes.endswith(b"\n")
    assert "rélease-α" in first_bytes.decode("utf-8")
    with pytest.raises(ReportWriteError, match="already exists"):
        write_json_report(report, destination)
    destination.write_text("old", encoding="utf-8")
    write_json_report(report, destination, overwrite=True)
    assert json.loads(destination.read_text(encoding="utf-8"))["schema_version"] == "1.3"


def test_json_writer_rejects_missing_parent_without_explicit_creation(
    tmp_path: Path,
) -> None:
    """Library writers create directories only when callers opt in."""

    destination = tmp_path / "missing" / "report.json"

    with pytest.raises(ReportWriteError, match="parent directory does not exist"):
        write_json_report(make_report(), destination)
    assert not destination.exists()


def test_atomic_writer_cleans_temporary_file_after_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed atomic publication leaves neither destination nor temporary file."""

    destination = tmp_path / "report.json"

    def reject_link(*_: object, **__: object) -> Never:
        raise PermissionError("simulated publication failure")

    monkeypatch.setattr("trading_research.reporting.exports.os.link", reject_link)

    with pytest.raises(ReportWriteError, match="could not write report file"):
        write_json_report(make_report(), destination)
    assert not destination.exists()
    assert tuple(tmp_path.glob(".*.tmp")) == ()


def test_summary_csv_has_stable_raw_columns_and_one_result_row() -> None:
    """The summary is flat, exact, unformatted, and explicit about reconciliation."""

    report = make_report(
        risk_settings=RiskMetricSettings(periods_per_year=Decimal("252"))
    )
    text = summary_report_to_csv(report)
    parsed = csv.DictReader(StringIO(text))
    rows = list(parsed)
    metrics = report.risk_adjusted_metrics
    assert metrics is not None

    assert tuple(parsed.fieldnames or ()) == SUMMARY_CSV_COLUMNS
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "deterministic-run-id"
    assert row["data_identifier"] == "data/sample_prices.csv"
    assert row["data_sha256"] == sha256_file(_SAMPLE_DATA)
    assert row["strategy_name"] == "sma-crossover"
    assert row["fast_window"] == "2"
    assert row["starting_cash"] == "10000"
    assert row["ending_equity"] == str(report.performance.ending_equity)
    assert row["total_return"] == str(report.performance.total_return)
    assert row["periodic_volatility"] == str(metrics.periodic_volatility)
    assert row["reconciliation_status"] == "PASS"
    assert row["observation_count"] == str(report.exposure_statistics.observation_count)
    assert row["time_in_market_ratio"] == str(
        report.exposure_statistics.time_in_market_ratio
    )
    assert row["actionable_signal_count"] == "1"
    assert row["full_approval_count"] == "1"
    assert row["execution_approval_rate"] == "1"
    assert "%" not in text
    assert "$" not in text


def test_summary_csv_uses_empty_fields_for_unavailable_metrics() -> None:
    """Missing optional risk and trade ratios remain empty rather than invented."""

    report = make_report(end_of_test_policy=EndOfTestPolicy.HOLD, bar_count=5)
    row = csv_rows(summary_report_to_csv(report))[0]

    assert row["periods_per_year"] == ""
    assert row["periodic_volatility"] == ""
    assert row["annualised_sharpe_ratio"] == ""
    assert row["win_rate"] == ""
    assert row["profit_factor"] == ""
    assert row["ending_cash"] != row["ending_equity"]


def test_summary_csv_distinguishes_hold_and_liquidate_results() -> None:
    """The flat summary retains terminal policy and its exact account effect."""

    hold = csv_rows(
        summary_report_to_csv(
            make_report(end_of_test_policy=EndOfTestPolicy.HOLD, bar_count=5)
        )
    )[0]
    liquidated = csv_rows(
        summary_report_to_csv(
            make_report(end_of_test_policy=EndOfTestPolicy.LIQUIDATE, bar_count=5)
        )
    )[0]

    assert hold["end_of_test_policy"] == "hold"
    assert hold["ending_cash"] != hold["ending_equity"]
    assert liquidated["end_of_test_policy"] == "liquidate"
    assert liquidated["ending_cash"] == liquidated["ending_equity"]


def test_equity_csv_preserves_source_rows_and_authoritative_final_values() -> None:
    """Equity export adds no seed row, sorting, duplicate terminal row, or formatting."""

    report = make_report()
    prepared = prepare_equity_curve_rows(report)
    text = equity_curve_to_csv(report)
    parsed = csv.DictReader(StringIO(text))
    rows = list(parsed)

    assert tuple(parsed.fieldnames or ()) == EQUITY_CSV_COLUMNS
    assert len(rows) == len(report.backtest_result.equity_curve)
    assert [row["timestamp"] for row in rows] == [
        point.timestamp.isoformat() for point in report.backtest_result.equity_curve
    ]
    assert rows[-1]["equity"] == str(report.backtest_result.final_equity)
    assert rows[-1]["cash"] == str(report.backtest_result.final_cash)
    assert rows[-1]["position_quantity"] == str(
        report.backtest_result.final_position_quantity
    )
    assert rows[-1]["period_return"] == str(
        report.return_series.returns[-1].return_ratio
    )
    assert rows[-1]["absolute_drawdown"] == str(prepared[-1].absolute_drawdown)
    assert rows[-1]["percentage_drawdown"] == str(
        prepared[-1].percentage_drawdown
    )
    assert rows[-1]["capital_utilisation_ratio"] == str(
        prepared[-1].capital_utilisation_ratio
    )
    assert rows[-1]["cash_ratio"] == str(prepared[-1].cash_ratio)
    assert rows[-1]["invested"] == str(prepared[-1].invested).lower()


def test_equity_csv_distinguishes_hold_and_liquidate_terminal_values() -> None:
    """Final export composition follows the selected authoritative end policy."""

    hold = csv_rows(
        equity_curve_to_csv(
            make_report(end_of_test_policy=EndOfTestPolicy.HOLD, bar_count=5)
        )
    )[-1]
    liquidated = csv_rows(
        equity_curve_to_csv(
            make_report(end_of_test_policy=EndOfTestPolicy.LIQUIDATE, bar_count=5)
        )
    )[-1]

    assert hold["position_quantity"] == "10"
    assert hold["position_market_value"] != "0"
    assert hold["cash"] != hold["equity"]
    assert liquidated["position_quantity"] == "0"
    assert Decimal(liquidated["position_market_value"]) == Decimal("0")
    assert liquidated["cash"] == liquidated["equity"]


def test_closed_trade_csv_is_header_only_for_open_entry() -> None:
    """An explicitly requested empty export keeps its stable header only."""

    report = make_report(end_of_test_policy=EndOfTestPolicy.HOLD, bar_count=5)
    text = closed_trades_to_csv(report)

    assert text == ",".join(CLOSED_TRADES_CSV_COLUMNS) + "\n"


def test_decision_csv_is_header_only_when_no_entry_was_proposed() -> None:
    """An explicitly requested audit export remains useful for an inactive run."""

    report = make_report(bar_count=4)
    text = decisions_to_csv(report)

    assert report.pre_trade_decisions == ()
    assert text == ",".join(DECISIONS_CSV_COLUMNS) + "\n"


def test_decision_csv_preserves_exact_ordered_audit_values() -> None:
    """Decision rows expose raw estimates, stable outcomes, and stable reasons."""

    report = make_report()
    rows = csv_rows(decisions_to_csv(report))
    audit = report.pre_trade_decisions[0]

    assert len(rows) == len(report.pre_trade_decisions) == 1
    assert rows[0] == {
        "timestamp": audit.timestamp.isoformat(),
        "symbol": audit.symbol,
        "side": audit.side.value,
        "reference_price": str(audit.reference_price),
        "proposed_quantity": str(audit.proposed_quantity),
        "approved_quantity": str(audit.approved_quantity),
        "quantity_reduction": str(audit.quantity_reduction),
        "estimated_fill_price": str(audit.estimated_fill_price),
        "estimated_commission": str(audit.estimated_commission),
        "estimated_total_cost": str(audit.estimated_total_cost),
        "available_cash_before": str(audit.available_cash_before),
        "position_quantity_before": str(audit.position_quantity_before),
        "outcome": "approved",
        "reason": "",
    }


def test_decision_writer_uses_existing_atomic_overwrite_policy(tmp_path: Path) -> None:
    """Decision export refuses an existing destination unless explicitly replaced."""

    destination = tmp_path / "decision audit.csv"
    destination.write_text("protected", encoding="utf-8")

    with pytest.raises(ReportWriteError, match="already exists"):
        write_decisions_csv(make_report(), destination)
    assert destination.read_text(encoding="utf-8") == "protected"

    write_decisions_csv(make_report(), destination, overwrite=True)
    assert csv_rows(destination.read_text(encoding="utf-8"))[0]["outcome"] == (
        "approved"
    )


def test_closed_trade_csv_reuses_trade_values_and_synthetic_reason() -> None:
    """Completed rows retain exact P&L, duration, chronology, and exit attribution."""

    report = make_report(
        end_of_test_policy=EndOfTestPolicy.LIQUIDATE,
        bar_count=5,
    )
    row = csv_rows(closed_trades_to_csv(report))[0]
    trade = report.closed_trades[0]

    assert row["quantity"] == str(trade.quantity)
    assert row["entry_fill_price"] == str(trade.entry_fill_price)
    assert row["exit_fill_price"] == str(trade.exit_fill_price)
    assert row["gross_pnl"] == str(trade.gross_pnl)
    assert row["net_pnl"] == str(trade.net_pnl)
    assert row["return_ratio"] == str(trade.return_ratio)
    assert row["holding_period_seconds"] == "0"
    assert row["exit_reason"] == "end_of_test_liquidation"


def test_structured_reporting_does_not_mutate_input_models() -> None:
    """Repeated exports leave the authoritative immutable result unchanged."""

    report = make_report()
    before_trades = report.backtest_result.trades
    before_curve = report.backtest_result.equity_curve
    before_equity = report.backtest_result.final_equity

    structured_report_to_json(report)
    summary_report_to_csv(report)
    equity_curve_to_csv(report)
    closed_trades_to_csv(report)
    decisions_to_csv(report)

    assert report.backtest_result.trades == before_trades
    assert report.backtest_result.equity_curve == before_curve
    assert report.backtest_result.final_equity == before_equity


def test_dataset_identity_model_accepts_missing_digest_for_non_file_sources() -> None:
    """The general metadata model permits an explicitly unavailable digest."""

    source = make_report().metadata.dataset
    without_digest = replace(source, sha256=None, source_type="test_fixture")

    assert isinstance(without_digest, DatasetIdentity)
    assert without_digest.sha256 is None


def test_metadata_model_type_is_public() -> None:
    """The aggregate exposes the stable public metadata model."""

    assert isinstance(make_report().metadata, BacktestRunMetadata)
