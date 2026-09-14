"""Logging configuration shared by command-line entry points and applications."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure concise application logging with the requested log level.

    Args:
        level: A standard logging level name such as ``INFO`` or ``DEBUG``.

    Raises:
        ValueError: If ``level`` is not a recognized logging level name.
    """

    numeric_level = logging.getLevelNamesMapping().get(level.upper())
    if numeric_level is None:
        raise ValueError(f"Unknown logging level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
