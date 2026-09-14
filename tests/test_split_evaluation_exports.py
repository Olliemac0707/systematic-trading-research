"""Split schema, CSV, text, and safe atomic export tests."""

import csv
import json
from io import StringIO
from pathlib import Path

import pytest
from test_split_evaluation_runner import run_evaluation

from trading_research.errors import ReportWriteError
from trading_research.evaluation import (
    SPLIT_COMPARISON_CSV_COLUMNS,
    SPLIT_EVALUATION_SCHEMA_NAME,
    SPLIT_EVALUATION_SCHEMA_VERSION,
    format_split_evaluation_report,
    split_comparison_to_csv,
    split_evaluation_to_json,
    write_split_evaluation_exports,
)
from trading_research.experiments import BenchmarkSelection


def test_combined_json_uses_distinct_schema_exact_strings_and_provenance(
    tmp_path: Path,
) -> None:
    """Split JSON cannot be confused with individual schema 1.3 reports."""

    result = run_evaluation(
        tmp_path,
        benchmark=BenchmarkSelection.BUY_AND_HOLD,
        risk_metrics=True,
    )
    payload = json.loads(split_evaluation_to_json(result))

    assert payload["schema"] == SPLIT_EVALUATION_SCHEMA_NAME
    assert payload["schema_version"] == SPLIT_EVALUATION_SCHEMA_VERSION
    assert payload["provenance"]["dataset_id"] == "synthetic-daily-v1"
    assert payload["data_sha256"] == result.data_sha256
    assert payload["configuration_sha256"] == result.configuration_sha256
    assert payload["development"]["total_return"] == str(
        result.development.performance.total_return
    )
    assert isinstance(payload["holdout"]["ending_equity"], str)
    assert payload["split"]["warmup_policy"] == "isolated"
    assert payload["development_report"] == "development.json"
    assert payload["holdout_report"] == "holdout.json"


def test_comparison_csv_contains_raw_stability_values_and_declared_periods(
    tmp_path: Path,
) -> None:
    """The one-row comparison preserves exact metrics and inclusive boundaries."""

    result = run_evaluation(tmp_path, risk_metrics=True)
    rows = list(csv.DictReader(StringIO(split_comparison_to_csv(result))))

    assert len(rows) == 1
    row = rows[0]
    assert tuple(row) == SPLIT_COMPARISON_CSV_COLUMNS
    assert row["evaluation_id"] == result.evaluation_id
    assert row["development_start"] == result.split.development_start.isoformat()
    assert row["holdout_end"] == result.split.holdout_end.isoformat()
    assert row["total_return_change"] == str(
        result.stability_comparison.total_return_change
    )
    assert row["development_sharpe"] == str(
        result.stability_comparison.development_sharpe
    )
    assert row["data_sha256"] == result.data_sha256


def test_text_report_is_descriptive_without_an_automatic_verdict(tmp_path: Path) -> None:
    """Human output explains differences and explicitly declines robustness claims."""

    report = format_split_evaluation_report(run_evaluation(tmp_path))

    assert report.startswith("OUT-OF-SAMPLE EVALUATION")
    assert "Development Period" in report
    assert "Holdout Period" in report
    assert "Change from Development to Holdout" in report
    assert "does not determine whether the strategy is robust" in report
    assert "PASS" not in report


def test_all_split_exports_are_written_atomically_without_absolute_paths(
    tmp_path: Path,
) -> None:
    """All required files use safe metadata and leave no temporary artifacts."""

    result = run_evaluation(tmp_path)
    output = tmp_path / "exports"

    write_split_evaluation_exports(result, output)

    expected = {
        "evaluation.json",
        "development.json",
        "holdout.json",
        "comparison.csv",
        "report.txt",
    }
    assert {path.name for path in output.iterdir()} == expected
    combined = (output / "evaluation.json").read_text(encoding="utf-8")
    development = (output / "development.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in combined
    assert str(tmp_path) not in development
    assert not tuple(output.glob("*.tmp"))


def test_existing_file_protection_preflights_all_exports(tmp_path: Path) -> None:
    """A collision prevents every other split artifact before any write occurs."""

    result = run_evaluation(tmp_path)
    output = tmp_path / "exports"
    output.mkdir()
    existing = output / "evaluation.json"
    existing.write_text("keep", encoding="utf-8")

    with pytest.raises(ReportWriteError, match="already exists"):
        write_split_evaluation_exports(result, output)

    assert existing.read_text(encoding="utf-8") == "keep"
    assert not (output / "development.json").exists()
    assert not (output / "comparison.csv").exists()

    write_split_evaluation_exports(result, output, overwrite=True)
    assert json.loads(existing.read_text(encoding="utf-8"))["schema_version"] == "1.0"
