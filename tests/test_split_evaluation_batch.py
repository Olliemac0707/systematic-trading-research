"""Manifest, provenance precedence, sequential failure, and evaluation CLI tests."""

import json
from decimal import Decimal
from pathlib import Path

import pytest
from evaluation_helpers import synthetic_bars, write_bars_csv

import trading_research.cli as cli
from trading_research.errors import EvaluationManifestError, ReportWriteError
from trading_research.evaluation import (
    SplitEvaluationFailureKind,
    SplitEvaluationStatus,
    load_split_evaluation_manifest,
    run_split_evaluation_batch,
)
from trading_research.experiments import load_experiment_manifest

_ROOT = Path(__file__).resolve().parents[1]


def evaluation_definition(
    evaluation_id: str,
    *,
    dataset: str = "prices.csv",
    source_name: str = "manifest source",
) -> dict[str, object]:
    """Return one complete flat-strategy split definition."""

    return {
        "id": evaluation_id,
        "dataset": dataset,
        "symbol": "TEST",
        "provenance": {
            "dataset_id": "synthetic-v1",
            "source_name": source_name,
            "price_adjustment": "unknown",
            "dividend_treatment": "unknown",
            "split_treatment": "unknown",
            "bar_frequency": "daily",
        },
        "split": {
            "development_start": "2025-01-01T00:00:00+00:00",
            "development_end": "2025-01-08T00:00:00+00:00",
            "holdout_start": "2025-01-09T00:00:00+00:00",
            "holdout_end": "2025-01-16T00:00:00+00:00",
            "warmup_policy": "carry-history",
        },
        "strategy": {
            "name": "sma-crossover",
            "fast_window": 2,
            "slow_window": 3,
        },
        "starting_cash": "10000",
        "commission_bps": "1",
        "slippage_bps": "5",
        "position_sizing": {"mode": "fixed", "quantity": 10},
        "risk_policy": {},
        "end_of_test": "hold",
        "risk_metrics": {
            "periods_per_year": "252",
            "risk_free_rate_per_period": "0",
            "target_return_per_period": "0",
        },
        "benchmark": "buy-and-hold",
    }


def write_manifest(path: Path, evaluations: list[dict[str, object]]) -> Path:
    """Write a schema-1.0 JSON manifest preserving list order."""

    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "split_evaluations": evaluations,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_split_loader_accepts_flat_strategy_and_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    """The documented strategy form is normalized while duplicate identity is fatal."""

    valid = load_split_evaluation_manifest(
        write_manifest(tmp_path / "valid.json", [evaluation_definition("one")])
    )
    assert valid.entries[0].definition is not None
    assert valid.entries[0].definition.strategy.to_domain().name.value == "sma-crossover"

    duplicate = write_manifest(
        tmp_path / "duplicate.json",
        [evaluation_definition("same"), evaluation_definition("same")],
    )
    with pytest.raises(EvaluationManifestError, match="duplicate split evaluation ID"):
        load_split_evaluation_manifest(duplicate)


def test_invalid_split_is_retained_as_configuration_failure(tmp_path: Path) -> None:
    """A reversed declared boundary is never silently repaired or omitted."""

    definition = evaluation_definition("invalid-split")
    split = definition["split"]
    assert isinstance(split, dict)
    split["holdout_start"] = "2025-01-07T00:00:00+00:00"
    manifest = load_split_evaluation_manifest(
        write_manifest(tmp_path / "invalid.json", [definition])
    )

    assert manifest.entries[0].definition is None
    assert "must not overlap" in (manifest.entries[0].configuration_error or "")


def test_manifest_explicit_values_override_sidecar_only_when_supplied(
    tmp_path: Path,
) -> None:
    """The sidecar supplies defaults and an explicit manifest field wins."""

    write_bars_csv(tmp_path / "prices.csv", synthetic_bars())
    (tmp_path / "prices.metadata.toml").write_text(
        """[provenance]
dataset_id = "sidecar-v1"
source_name = "sidecar source"
price_adjustment = "unknown"
dividend_treatment = "unknown"
split_treatment = "unknown"
bar_frequency = "daily"
currency = "GBP"
""",
        encoding="utf-8",
    )
    definition = evaluation_definition("sidecar")
    definition["provenance"] = {"source_name": "explicit manifest source"}
    manifest = load_split_evaluation_manifest(
        write_manifest(tmp_path / "sidecar.json", [definition])
    )

    batch = run_split_evaluation_batch(
        manifest,
        tmp_path / "output",
        repository=_ROOT,
    )
    result = batch.records[0].result

    assert result is not None
    assert result.provenance.dataset_id == "sidecar-v1"
    assert result.provenance.source_name == "explicit manifest source"
    assert result.provenance.currency == "GBP"


def test_one_failed_evaluation_between_successes_is_never_omitted(
    tmp_path: Path,
) -> None:
    """Missing local data is recorded and later declared evaluation still runs."""

    write_bars_csv(tmp_path / "prices.csv", synthetic_bars())
    manifest = load_split_evaluation_manifest(
        write_manifest(
            tmp_path / "ordered.json",
            [
                evaluation_definition("first"),
                evaluation_definition("missing", dataset="missing.csv"),
                evaluation_definition("last"),
            ],
        )
    )

    batch = run_split_evaluation_batch(
        manifest,
        tmp_path / "output",
        repository=_ROOT,
    )
    machine = json.loads(
        (tmp_path / "output" / "evaluation-manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert [record.evaluation_id for record in batch.records] == [
        "first",
        "missing",
        "last",
    ]
    assert [record.status for record in batch.records] == [
        SplitEvaluationStatus.SUCCESS,
        SplitEvaluationStatus.FAILURE,
        SplitEvaluationStatus.SUCCESS,
    ]
    assert batch.records[1].failure_kind is SplitEvaluationFailureKind.MARKET_DATA
    assert [row["evaluation_id"] for row in machine["results"]] == [
        "first",
        "missing",
        "last",
    ]
    assert machine["failure_count"] == 1
    assert (tmp_path / "output" / "first" / "evaluation.json").is_file()
    assert not (tmp_path / "output" / "missing").exists()
    assert (tmp_path / "output" / "last" / "holdout.json").is_file()


def test_batch_existing_output_root_requires_explicit_authorisation(
    tmp_path: Path,
) -> None:
    """A pre-existing root is protected before any evaluation executes."""

    write_bars_csv(tmp_path / "prices.csv", synthetic_bars())
    manifest = load_split_evaluation_manifest(
        write_manifest(tmp_path / "manifest.json", [evaluation_definition("protected")])
    )
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "note.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ReportWriteError, match="already exists"):
        run_split_evaluation_batch(manifest, output, repository=_ROOT)
    assert sentinel.read_text(encoding="utf-8") == "keep"

    batch = run_split_evaluation_batch(
        manifest,
        output,
        overwrite=True,
        repository=_ROOT,
    )
    assert batch.succeeded
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_ordinary_experiment_manifest_remains_compatible() -> None:
    """Adding a split top-level type does not alter the existing batch example."""

    manifest = load_experiment_manifest(_ROOT / "experiments" / "example.toml")

    assert [entry.experiment_id for entry in manifest.entries] == [
        "sma-baseline",
        "donchian-baseline",
    ]


def test_committed_toml_example_runs_all_documented_split_modes(tmp_path: Path) -> None:
    """The real example covers SMA isolated/carry, Donchian carry, and benchmark."""

    manifest = load_split_evaluation_manifest(
        _ROOT / "experiments" / "out-of-sample.toml"
    )
    batch = run_split_evaluation_batch(
        manifest,
        tmp_path / "output",
        repository=_ROOT,
    )

    assert [record.evaluation_id for record in batch.records] == [
        "sma-isolated",
        "sma-carry-history",
        "donchian-carry-history",
    ]
    assert batch.succeeded
    isolated = batch.records[0].result
    carry = batch.records[1].result
    donchian = batch.records[2].result
    assert isolated is not None and carry is not None and donchian is not None
    assert isolated.holdout.performance.ending_equity == Decimal("10000")
    assert carry.holdout.performance.ending_equity == Decimal("10000")
    assert carry.holdout.benchmark is not None
    assert donchian.holdout.performance.ending_equity == Decimal("9989.87399950")
    assert isolated.warmup_bar_count == 0
    assert carry.warmup_bar_count == donchian.warmup_bar_count == 2


def test_evaluate_cli_help_and_partial_failure_exit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The separate CLI is concise and returns a distinct completed-with-failure code."""

    help_code = cli.main(["evaluate", "--help"])
    help_output = capsys.readouterr()
    assert help_code == cli.EXIT_SUCCESS
    assert "--manifest" in help_output.out
    assert "optimisation" in help_output.out

    write_bars_csv(tmp_path / "prices.csv", synthetic_bars())
    manifest = write_manifest(
        tmp_path / "partial.json",
        [
            evaluation_definition("success"),
            evaluation_definition("missing", dataset="absent.csv"),
        ],
    )
    output = tmp_path / "output"

    exit_code = cli.main(
        ["evaluate", "--manifest", str(manifest), "--output-dir", str(output)]
    )
    captured = capsys.readouterr()

    assert exit_code == cli.EXIT_EVALUATION_FAILURE
    assert "success: COMPLETED" in captured.out
    assert "missing: FAILURE" in captured.out
    assert "Traceback" not in captured.out + captured.err
    assert str(tmp_path) not in captured.out
    assert captured.err == ""


def test_evaluate_cli_reports_malformed_manifest_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Shared JSON/TOML parse failures are translated to evaluation usage errors."""

    manifest = tmp_path / "broken.json"
    manifest.write_text("{not valid JSON", encoding="utf-8")
    output = tmp_path / "output"

    exit_code = cli.main(
        ["evaluate", "--manifest", str(manifest), "--output-dir", str(output)]
    )
    captured = capsys.readouterr()

    assert exit_code == cli.EXIT_USAGE_ERROR
    assert captured.out == ""
    assert "invalid JSON manifest" in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()
