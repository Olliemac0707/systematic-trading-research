"""Tests for logging setup validation."""

import pytest

from trading_research.logging_config import configure_logging


def test_configure_logging_rejects_unknown_level() -> None:
    """A typo in a log level fails clearly instead of changing behavior silently."""

    with pytest.raises(ValueError, match="Unknown logging level"):
        configure_logging("verbose")
