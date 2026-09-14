"""Tests for the single authoritative application version."""

import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest

import trading_research.cli as cli
from trading_research import __version__
from trading_research.reporting import installed_application_version

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DISTRIBUTION_NAME = "trading-research"
_EXPECTED_VERSION = "0.5.0"


def test_runtime_version_matches_authoritative_project_metadata() -> None:
    """Package and reporting versions cannot diverge from project metadata."""

    with (_PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    assert project["project"]["version"] == _EXPECTED_VERSION
    assert version(_DISTRIBUTION_NAME) == _EXPECTED_VERSION
    assert __version__ == _EXPECTED_VERSION
    assert installed_application_version() == _EXPECTED_VERSION


def test_cli_version_uses_authoritative_application_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The top-level CLI reports the same installed application version."""

    assert cli.main(["--version"]) == cli.EXIT_SUCCESS
    captured = capsys.readouterr()

    assert captured.out == f"trading-research {_EXPECTED_VERSION}\n"
    assert captured.err == ""
