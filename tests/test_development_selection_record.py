"""Tests sealing the pre-registered development-study selection."""

import re
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SELECTION_PATH = (
    _ROOT / "experiments" / "approved-etf-study-development-selection.toml"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_ARTIFACT_HASHES = {
    "development_summary_csv": (
        "a60d20551c9aa5f079427ef0c34135df79cae52116b7ae2624307863b66fe9df"
    ),
    "development_summary_json": (
        "facadace275c4417eec9def40767478dcaf5f3e58f0d8143ed896163e1e11469"
    ),
    "candidate_summary_csv": (
        "cd126ef4e7451765e9521be21db72f5f51cff197f0d2684ea2b3852e896daae3"
    ),
    "benchmark_comparison_csv": (
        "b963ddf864a1d8def37cb5776caee9677af14707c6ae89d23016827205621b0e"
    ),
    "selection_policy_json": (
        "f5019365887b0d1e728d0a42c7d9677d6bf2d0c16a4c0a9f136d7d421be1dca7"
    ),
    "run_integrity_json": (
        "9091625a932227b28748ba6605652a7f9c40cc497f2a616253890b7b821d5253"
    ),
}


def _selection() -> dict[str, object]:
    """Load the committed selection record as TOML data."""

    with _SELECTION_PATH.open("rb") as selection_file:
        return tomllib.load(selection_file)


def test_selection_record_keeps_holdout_sealed_and_reproduction_development_only() -> None:
    """The frozen reproduction command cannot evaluate the holdout."""

    selection = _selection()

    assert selection["phase"] == "development"
    assert selection["development_end"] == "2018-12-31"
    assert selection["holdout_status"] == "sealed"
    assert (
        selection["source_dataset_commit"]
        == "718816c1e3471dba95086ea64c73e24a907d2b71"
    )
    assert (
        selection["manifest_sha256"]
        == "73769c33454cc59b59c22a0091940eaa84e83167d7a1a910ceeabeeb9c1bc236"
    )
    assert selection["selected_candidate_id"] == "sma-50-200"
    assert selection["candidate_specific_asset_selection"] is False
    assert selection["tie_breaker_used"] is False
    command = selection["reproduction_command"]
    assert isinstance(command, str)
    assert command.endswith("--development-only")
    assert "--overwrite-output-dir" not in command


def test_selection_record_freezes_all_candidates_and_aggregate_hashes() -> None:
    """Candidate evidence and ignored aggregate artifacts remain auditable."""

    selection = _selection()
    candidates = selection["candidates"]
    assert isinstance(candidates, list)
    assert [
        candidate["candidate_id"] for candidate in candidates
    ] == [
        "sma-20-50",
        "sma-50-200",
        "donchian-20-10",
        "donchian-50-20",
    ]
    assert [candidate["overall_rank_score"] for candidate in candidates] == [
        "2.75",
        "1.583333333333333333333333333",
        "2.875",
        "2.791666666666666666666666667",
    ]
    assert all(candidate["eligible"] is True for candidate in candidates)

    artifact_hashes = selection["artifact_sha256"]
    assert isinstance(artifact_hashes, dict)
    assert artifact_hashes == _EXPECTED_ARTIFACT_HASHES
    assert all(
        isinstance(digest, str) and _SHA256.fullmatch(digest)
        for digest in artifact_hashes.values()
    )


def test_selection_record_contains_no_user_specific_absolute_path() -> None:
    """The version-controlled evidence remains portable."""

    text = _SELECTION_PATH.read_text(encoding="utf-8")

    assert not re.search(r"(?i)[a-z]:[\\/]", text)
    assert "C:\\Users\\" not in text
