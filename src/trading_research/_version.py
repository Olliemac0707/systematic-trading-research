"""Single runtime access point for the installed application version."""

from importlib.metadata import PackageNotFoundError, version

_DISTRIBUTION_NAME = "trading-research"
_UNAVAILABLE_VERSION = "Not available"


def application_version() -> str:
    """Return the version defined by the installed project metadata."""

    try:
        return version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return _UNAVAILABLE_VERSION


__version__ = application_version()
