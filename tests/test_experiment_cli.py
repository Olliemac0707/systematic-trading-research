"""CLI tests for predefined sequential experiment batches."""

import json
from pathlib import Path

import pytest

import trading_research.cli as cli

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_DATA = _ROOT / "data" / "sample_prices.csv"


def write_manifest(path: Path, *, include_failure: bool = False) -> Path:
    """Write a complete CLI manifest with an optional missing-data entry."""

    experiments: list[dict[str, object]] = [
        {
            "id": "sma-cli",
            "dataset": str(_SAMPLE_DATA),
            "symbol": "DEMO",
            "strategy": {
                "name": "sma-crossover",
                "parameters": {"fast_window": 2, "slow_window": 3},
            },
            "starting_cash": "10000",
            "commission_bps": "1",
            "slippage_bps": "5",
            "position_sizing": {"mode": "fixed", "quantity": 10},
            "end_of_test": "hold",
        }
    ]
    if include_failure:
        experiments.insert(
            0,
            {
                **experiments[0],
                "id": "missing-cli",
                "dataset": str(path.parent / "missing.csv"),
            },
        )
    path.write_text(
        json.dumps({"schema_version": "1.0", "experiments": experiments}),
        encoding="utf-8",
    )
    return path


def test_experiment_help_exposes_manifest_and_output_controls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The new command documents only predefined sequential batch controls."""

    exit_code = cli.main(["experiment", "--help"])

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_SUCCESS
    assert "--manifest" in captured.out
    assert "--output-dir" in captured.out
    assert "--overwrite-output-dir" in captured.out
    assert "optim" not in captured.out.lower()
    assert captured.err == ""


def test_experiment_cli_writes_outputs_and_reports_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A successful command writes existing JSON plus both aggregate outputs."""

    manifest = write_manifest(tmp_path / "manifest.json")
    output = tmp_path / "output"

    exit_code = cli.main(
        ["experiment", "--manifest", str(manifest), "--output-dir", str(output)]
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_SUCCESS
    assert "Succeeded:   1" in captured.out
    assert "Failed:      0" in captured.out
    assert "sma-cli: SUCCESS" in captured.out
    assert str(tmp_path) not in captured.out
    assert (output / "reports" / "sma-cli.json").is_file()
    assert (output / "aggregate.csv").is_file()
    assert (output / "batch-manifest.json").is_file()
    assert captured.err == ""


def test_experiment_cli_returns_six_after_recording_partial_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expected failures remain visible and later successful entries still execute."""

    manifest = write_manifest(tmp_path / "partial.json", include_failure=True)
    output = tmp_path / "output"

    exit_code = cli.main(
        ["experiment", "--manifest", str(manifest), "--output-dir", str(output)]
    )

    captured = capsys.readouterr()
    machine = json.loads((output / "batch-manifest.json").read_text(encoding="utf-8"))
    assert exit_code == cli.EXIT_EXPERIMENT_FAILURE
    assert "missing-cli: FAILURE" in captured.out
    assert "sma-cli: SUCCESS" in captured.out
    assert [row["status"] for row in machine["results"]] == ["failure", "success"]
    assert machine["failure_count"] == 1
    assert "Traceback" not in captured.out + captured.err
    assert captured.err == ""


def test_experiment_cli_rejects_duplicate_ids_without_creating_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A structurally ambiguous manifest is a concise usage error."""

    manifest = write_manifest(tmp_path / "duplicates.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["experiments"].append(payload["experiments"][0])
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "output"

    exit_code = cli.main(
        ["experiment", "--manifest", str(manifest), "--output-dir", str(output)]
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_USAGE_ERROR
    assert captured.out == ""
    assert "duplicate experiment ID" in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()
