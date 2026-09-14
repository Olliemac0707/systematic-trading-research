"""Tests for predefined, sequential batch experiment execution."""

import csv
import json
from decimal import Decimal
from io import StringIO
from pathlib import Path

import pytest

from trading_research.backtesting import SimpleBacktestEngine
from trading_research.data import CsvMarketDataProvider
from trading_research.errors import ExperimentManifestError, ReportWriteError
from trading_research.experiments import (
    AGGREGATE_CSV_COLUMNS,
    BATCH_RESULT_SCHEMA_VERSION,
    BenchmarkSelection,
    ExperimentFailureKind,
    ExperimentStatus,
    aggregate_result_to_csv,
    batch_result_to_json,
    load_experiment_manifest,
    run_experiment_batch,
)
from trading_research.reporting import sha256_file
from trading_research.strategies import create_strategy

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_DATA = _ROOT / "data" / "sample_prices.csv"
_DONCHIAN_DATA = _ROOT / "tests" / "fixtures" / "donchian_prices.csv"
_EXAMPLE_MANIFEST = _ROOT / "experiments" / "example.toml"


def experiment_definition(
    experiment_id: str,
    *,
    dataset: Path = _SAMPLE_DATA,
    symbol: str = "DEMO",
    strategy: str = "sma-crossover",
    parameters: dict[str, object] | None = None,
    benchmark: str = "none",
) -> dict[str, object]:
    """Build one valid exact fixed-sizing experiment dictionary."""

    selected_parameters = (
        {"fast_window": 2, "slow_window": 3}
        if parameters is None
        else parameters
    )
    return {
        "id": experiment_id,
        "dataset": str(dataset),
        "symbol": symbol,
        "strategy": {"name": strategy, "parameters": selected_parameters},
        "starting_cash": "10000",
        "commission_bps": "1",
        "slippage_bps": "5",
        "position_sizing": {"mode": "fixed", "quantity": 10},
        "risk_policy": {},
        "end_of_test": "hold",
        "benchmark": benchmark,
    }


def write_json_manifest(
    path: Path,
    experiments: list[dict[str, object]],
) -> Path:
    """Write one test manifest with a stable schema and declared order."""

    path.write_text(
        json.dumps({"schema_version": "1.0", "experiments": experiments}),
        encoding="utf-8",
    )
    return path


def test_loads_json_and_toml_with_typed_mixed_strategies(tmp_path: Path) -> None:
    """Both formats preserve order and construct closed typed strategy choices."""

    json_path = write_json_manifest(
        tmp_path / "mixed.json",
        [
            experiment_definition("sma"),
            experiment_definition(
                "donchian",
                dataset=_DONCHIAN_DATA,
                symbol="CHAN",
                strategy="donchian-breakout",
                parameters={"entry_window": 2, "exit_window": 2},
            ),
        ],
    )

    json_manifest = load_experiment_manifest(json_path)
    toml_manifest = load_experiment_manifest(_EXAMPLE_MANIFEST)

    assert [entry.experiment_id for entry in json_manifest.entries] == [
        "sma",
        "donchian",
    ]
    assert all(entry.configuration_error is None for entry in json_manifest.entries)
    assert [entry.experiment_id for entry in toml_manifest.entries] == [
        "sma-baseline",
        "donchian-baseline",
    ]
    assert toml_manifest.entries[0].definition is not None
    assert (
        toml_manifest.entries[0].definition.benchmark
        is BenchmarkSelection.BUY_AND_HOLD
    )


def test_duplicate_experiment_ids_are_refused(tmp_path: Path) -> None:
    """Duplicate IDs fail the complete manifest before an output can be ambiguous."""

    path = write_json_manifest(
        tmp_path / "duplicates.json",
        [experiment_definition("same"), experiment_definition("same")],
    )

    with pytest.raises(ExperimentManifestError, match="duplicate experiment ID"):
        load_experiment_manifest(path)


def test_invalid_parameters_are_retained_as_ordered_configuration_failure(
    tmp_path: Path,
) -> None:
    """A bad typed parameter set remains visible rather than aborting manifest load."""

    path = write_json_manifest(
        tmp_path / "invalid.json",
        [
            experiment_definition(
                "bad-sma",
                parameters={"fast_window": 3, "slow_window": 2},
            ),
            experiment_definition("later-valid"),
        ],
    )

    manifest = load_experiment_manifest(path)

    assert manifest.entries[0].definition is None
    assert "fast_window must be smaller" in (
        manifest.entries[0].configuration_error or ""
    )
    assert manifest.entries[1].definition is not None


def test_toml_binary_float_financial_values_are_rejected(tmp_path: Path) -> None:
    """TOML financial inputs cannot enter Decimal accounting through floats."""

    path = tmp_path / "float.toml"
    path.write_text(
        """schema_version = "1.0"

[[experiments]]
id = "float-cash"
dataset = "prices.csv"
symbol = "DEMO"
starting_cash = 10000.0
commission_bps = "1"
slippage_bps = "5"
end_of_test = "hold"

[experiments.strategy]
name = "sma-crossover"

[experiments.strategy.parameters]
fast_window = 2
slow_window = 3

[experiments.position_sizing]
mode = "fixed"
quantity = 10
""",
        encoding="utf-8",
    )
    manifest = load_experiment_manifest(path)

    assert manifest.entries[0].definition is None
    assert "starting_cash" in (manifest.entries[0].configuration_error or "")
    assert "exact decimal string or integer" in (
        manifest.entries[0].configuration_error or ""
    )


def test_optional_timestamp_filters_are_applied_by_shared_pipeline(
    tmp_path: Path,
) -> None:
    """Inclusive aware start and end filters constrain the existing CSV provider."""

    experiment = experiment_definition("filtered")
    experiment["start"] = "2025-01-02T00:00:00+00:00"
    experiment["end"] = "2025-01-06T00:00:00+00:00"
    manifest = load_experiment_manifest(
        write_json_manifest(tmp_path / "filtered.json", [experiment])
    )

    result = run_experiment_batch(manifest, tmp_path / "output", repository=_ROOT)
    report = result.records[0].report

    assert report is not None
    assert report.metadata.dataset.bar_count == 5
    assert report.metadata.dataset.actual_start.isoformat() == "2025-01-02T00:00:00+00:00"
    assert report.metadata.dataset.actual_end.isoformat() == "2025-01-06T00:00:00+00:00"


def test_mixed_batch_reuses_exact_results_benchmark_and_existing_exports(
    tmp_path: Path,
) -> None:
    """Mixed strategies retain exact individual results and reproducibility fields."""

    manifest_path = write_json_manifest(
        tmp_path / "mixed.json",
        [
            experiment_definition("sma", benchmark="buy-and-hold"),
            experiment_definition(
                "donchian",
                dataset=_DONCHIAN_DATA,
                symbol="CHAN",
                strategy="donchian-breakout",
                parameters={"entry_window": 2, "exit_window": 2},
            ),
        ],
    )
    manifest = load_experiment_manifest(manifest_path)

    result = run_experiment_batch(
        manifest,
        tmp_path / "output",
        repository=_ROOT,
    )
    aggregate = list(csv.DictReader(StringIO(aggregate_result_to_csv(result))))
    machine_manifest = json.loads(batch_result_to_json(result))

    assert result.succeeded
    assert [record.experiment_id for record in result.records] == ["sma", "donchian"]
    sma_report = result.records[0].report
    donchian_report = result.records[1].report
    assert sma_report is not None and donchian_report is not None
    assert sma_report.backtest_result.final_cash == sma_report.backtest_result.final_equity
    assert sma_report.backtest_result.final_equity == Decimal("9959.86799800")
    assert sma_report.benchmark is not None
    assert sma_report.backtest_result.reconciliation.is_reconciled
    assert donchian_report.backtest_result.final_equity == Decimal("9969.86199850")
    assert donchian_report.benchmark is None
    assert donchian_report.backtest_result.reconciliation.is_reconciled
    assert tuple(aggregate[0]) == AGGREGATE_CSV_COLUMNS
    assert [row["experiment_id"] for row in aggregate] == ["sma", "donchian"]
    assert aggregate[0]["ending_equity"] == "9959.86799800"
    assert aggregate[1]["ending_equity"] == "9969.86199850"
    assert aggregate[0]["data_sha256"] == sha256_file(_SAMPLE_DATA)
    assert machine_manifest["schema_version"] == BATCH_RESULT_SCHEMA_VERSION
    assert [row["experiment_id"] for row in machine_manifest["results"]] == [
        "sma",
        "donchian",
    ]
    assert all(row["dataset_sha256"] for row in machine_manifest["results"])
    assert all(row["git_commit"] == result.git_commit for row in machine_manifest["results"])
    assert (tmp_path / "output" / "reports" / "sma.json").is_file()
    assert (tmp_path / "output" / "reports" / "donchian.json").is_file()
    assert (tmp_path / "output" / "aggregate.csv").is_file()
    assert (tmp_path / "output" / "batch-manifest.json").is_file()
    individual_payload = json.loads(
        (tmp_path / "output" / "reports" / "sma.json").read_text(encoding="utf-8")
    )
    assert individual_payload["backtest_result"]["final_equity"] == "9959.86799800"
    assert isinstance(individual_payload["backtest_result"]["final_equity"], str)


def test_committed_toml_example_runs_sizing_risk_and_benchmark_configuration(
    tmp_path: Path,
) -> None:
    """The documented mixed example is executable without inferred parameters."""

    result = run_experiment_batch(
        load_experiment_manifest(_EXAMPLE_MANIFEST),
        tmp_path / "example-output",
        repository=_ROOT,
    )

    assert result.succeeded
    first = result.records[0].report
    second = result.records[1].report
    assert first is not None and second is not None
    assert first.benchmark is not None
    assert first.metadata.risk_metric_settings is not None
    assert second.metadata.position_sizing.mode.value == "cash_allocation"
    assert second.metadata.position_sizing.cash_allocation_ratio == Decimal("0.25")
    assert second.metadata.risk_policy.name == "composite"
    assert first.backtest_result.reconciliation.is_reconciled
    assert second.backtest_result.reconciliation.is_reconciled


def test_batch_result_matches_direct_engine_execution(tmp_path: Path) -> None:
    """The extracted pipeline does not change the authoritative individual result."""

    manifest = load_experiment_manifest(
        write_json_manifest(
            tmp_path / "single.json",
            [experiment_definition("sma-direct")],
        )
    )
    definition = manifest.entries[0].definition
    assert definition is not None
    batch = run_experiment_batch(manifest, tmp_path / "output", repository=_ROOT)
    report = batch.records[0].report
    assert report is not None
    bars = tuple(
        CsvMarketDataProvider(_SAMPLE_DATA).get_historical_bars("DEMO", None, None)
    )
    direct = SimpleBacktestEngine().run(
        bars,
        create_strategy(
            definition.strategy_configuration().name,
            definition.strategy_configuration().parameters,
        ),
        definition.backtest_configuration(),
    )

    assert report.backtest_result == direct
    assert direct.final_cash == direct.final_equity == Decimal("9959.86799800")


def test_expected_failures_do_not_stop_later_experiments_or_disappear(
    tmp_path: Path,
) -> None:
    """Invalid configuration and missing data remain rows between successful runs."""

    experiments = [
        experiment_definition("first-success"),
        experiment_definition(
            "invalid-parameters",
            strategy="donchian-breakout",
            parameters={"entry_window": 0, "exit_window": 2},
        ),
        experiment_definition("missing-data", dataset=tmp_path / "missing.csv"),
        experiment_definition("last-success"),
    ]
    manifest = load_experiment_manifest(
        write_json_manifest(tmp_path / "partial.json", experiments)
    )

    result = run_experiment_batch(manifest, tmp_path / "output", repository=_ROOT)
    aggregate = list(csv.DictReader(StringIO(aggregate_result_to_csv(result))))
    machine = json.loads(batch_result_to_json(result))

    assert [record.experiment_id for record in result.records] == [
        "first-success",
        "invalid-parameters",
        "missing-data",
        "last-success",
    ]
    assert [record.status for record in result.records] == [
        ExperimentStatus.SUCCESS,
        ExperimentStatus.FAILURE,
        ExperimentStatus.FAILURE,
        ExperimentStatus.SUCCESS,
    ]
    assert result.records[1].failure_kind is ExperimentFailureKind.CONFIGURATION
    assert result.records[2].failure_kind is ExperimentFailureKind.MARKET_DATA
    assert result.success_count == result.failure_count == 2
    assert len(aggregate) == len(machine["results"]) == 4
    assert aggregate[1]["ending_equity"] == aggregate[2]["ending_equity"] == ""
    assert aggregate[3]["ending_equity"] == "9959.86799800"
    assert not (tmp_path / "output" / "reports" / "invalid-parameters.json").exists()
    assert (tmp_path / "output" / "reports" / "last-success.json").is_file()


def test_existing_output_directory_requires_explicit_authorisation(
    tmp_path: Path,
) -> None:
    """The whole destination is protected by default and unrelated content survives."""

    manifest = load_experiment_manifest(
        write_json_manifest(
            tmp_path / "manifest.json",
            [experiment_definition("protected")],
        )
    )
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "user-note.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ReportWriteError, match="already exists"):
        run_experiment_batch(manifest, output, repository=_ROOT)
    assert sentinel.read_text(encoding="utf-8") == "keep"

    result = run_experiment_batch(
        manifest,
        output,
        overwrite=True,
        repository=_ROOT,
    )
    assert result.succeeded
    assert sentinel.read_text(encoding="utf-8") == "keep"
    repeated = run_experiment_batch(
        manifest,
        output,
        overwrite=True,
        repository=_ROOT,
    )
    first_report = result.records[0].report
    repeated_report = repeated.records[0].report
    assert first_report is not None and repeated_report is not None
    assert repeated_report.backtest_result == first_report.backtest_result


@pytest.mark.parametrize("suffix", [".yaml", ".txt"])
def test_unsupported_manifest_formats_are_refused(tmp_path: Path, suffix: str) -> None:
    """Only standard-library JSON and TOML loaders are available."""

    path = tmp_path / f"manifest{suffix}"
    path.write_text("schema_version: 1.0", encoding="utf-8")

    with pytest.raises(ExperimentManifestError, match=".json or .toml"):
        load_experiment_manifest(path)
