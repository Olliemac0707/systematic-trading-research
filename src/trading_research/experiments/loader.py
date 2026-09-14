"""Validated JSON and TOML loading for predefined experiment manifests."""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from trading_research.errors import ExperimentManifestError
from trading_research.experiments.models import ExperimentDefinition

EXPERIMENT_MANIFEST_SCHEMA_VERSION = "1.0"
_ALLOWED_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "experiments", "split_evaluations"}
)
_EXPERIMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True, slots=True)
class ExperimentManifestEntry:
    """One ordered experiment, valid or explicitly retained as a failure."""

    experiment_id: str
    definition: ExperimentDefinition | None
    configuration_error: str | None

    def __post_init__(self) -> None:
        """Require exactly one valid definition or configuration error."""

        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")
        if (self.definition is None) == (self.configuration_error is None):
            raise ValueError(
                "an experiment entry must contain exactly one definition or error"
            )


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    """Structurally validated manifest preserving declared experiment order."""

    source_path: Path
    entries: tuple[ExperimentManifestEntry, ...]
    schema_version: str = EXPERIMENT_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Require a supported schema and a non-empty ordered entry sequence."""

        object.__setattr__(self, "source_path", Path(self.source_path))
        if self.schema_version != EXPERIMENT_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {EXPERIMENT_MANIFEST_SCHEMA_VERSION!r}"
            )
        if not self.entries:
            raise ValueError("experiment manifest must contain at least one experiment")


def load_experiment_manifest(path: Path) -> ExperimentManifest:
    """Load JSON or TOML while retaining per-experiment validation failures."""

    source = Path(path)
    raw = read_manifest_document(source)
    unexpected = sorted(set(raw).difference(_ALLOWED_TOP_LEVEL_KEYS))
    if unexpected:
        raise ExperimentManifestError(
            f"manifest contains unsupported top-level field(s): {', '.join(unexpected)}"
        )
    version = raw.get("schema_version")
    if version != EXPERIMENT_MANIFEST_SCHEMA_VERSION:
        raise ExperimentManifestError(
            "manifest schema_version must be "
            f"{EXPERIMENT_MANIFEST_SCHEMA_VERSION!r}"
        )
    raw_experiments = raw.get("experiments")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise ExperimentManifestError(
            "manifest experiments must be a non-empty array"
        )

    entries: list[ExperimentManifestEntry] = []
    seen_ids: set[str] = set()
    for index, raw_experiment in enumerate(raw_experiments):
        if not isinstance(raw_experiment, Mapping):
            raise ExperimentManifestError(
                f"experiments[{index}] must be an object or table"
            )
        experiment_id = _validated_experiment_id(raw_experiment.get("id"), index)
        if experiment_id in seen_ids:
            raise ExperimentManifestError(
                f"duplicate experiment ID: {experiment_id!r}"
            )
        seen_ids.add(experiment_id)
        try:
            definition = ExperimentDefinition.model_validate(raw_experiment)
        except ValidationError as exc:
            entries.append(
                ExperimentManifestEntry(
                    experiment_id=experiment_id,
                    definition=None,
                    configuration_error=_format_validation_error(exc),
                )
            )
        else:
            entries.append(
                ExperimentManifestEntry(
                    experiment_id=experiment_id,
                    definition=definition,
                    configuration_error=None,
                )
            )
    return ExperimentManifest(source_path=source, entries=tuple(entries))


def read_manifest_document(path: Path) -> Mapping[str, object]:
    """Read one supported UTF-8 manifest with bounded, contextual errors."""

    suffix = path.suffix.lower()
    if suffix not in {".json", ".toml"}:
        raise ExperimentManifestError("manifest must use a .json or .toml extension")
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ExperimentManifestError(f"manifest file not found: {path}") from exc
    except IsADirectoryError as exc:
        raise ExperimentManifestError(f"manifest path is not a file: {path}") from exc
    except (OSError, UnicodeError) as exc:
        raise ExperimentManifestError(f"could not read manifest {path}: {exc}") from exc
    try:
        if suffix == ".json":
            document = json.loads(
                text,
                parse_float=Decimal,
                parse_constant=_reject_json_constant,
            )
        else:
            document = tomllib.loads(text)
    except (json.JSONDecodeError, tomllib.TOMLDecodeError, ValueError) as exc:
        raise ExperimentManifestError(f"invalid {suffix[1:].upper()} manifest: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ExperimentManifestError("manifest root must be an object or table")
    return document


def _validated_experiment_id(value: object, index: int) -> str:
    """Require a safe ID before per-experiment validation and duplicate checks."""

    if not isinstance(value, str):
        raise ExperimentManifestError(
            f"experiments[{index}].id must be a string"
        )
    if _EXPERIMENT_ID_PATTERN.fullmatch(value) is None:
        raise ExperimentManifestError(
            f"experiments[{index}].id must be a safe 1-64 character identifier"
        )
    return value


def _format_validation_error(error: ValidationError) -> str:
    """Return concise deterministic field paths without echoing supplied values."""

    messages: list[str] = []
    for detail in error.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) for part in detail["loc"])
        messages.append(f"{location}: {detail['msg']}" if location else detail["msg"])
    return "; ".join(messages)


def _reject_json_constant(value: str) -> object:
    """Reject non-standard NaN and infinity constants during JSON parsing."""

    raise ValueError(f"unsupported JSON constant: {value}")
