"""Local dataset provenance sidecar loading with explicit manifest precedence."""

import tomllib
from collections.abc import Mapping
from pathlib import Path

from pydantic import ValidationError

from trading_research.evaluation.models import (
    DatasetProvenance,
    DatasetProvenanceOverride,
)


def provenance_sidecar_path(dataset_path: Path) -> Path:
    """Return the documented sibling ``.metadata.toml`` sidecar path."""

    return Path(dataset_path).with_suffix(".metadata.toml")


def load_dataset_provenance(
    dataset_path: Path,
    manifest_override: DatasetProvenanceOverride | None = None,
) -> DatasetProvenance:
    """Load a sidecar, then apply only explicitly supplied manifest fields.

    No source URL is contacted. A manifest value, including explicit ``None``,
    takes precedence over the corresponding sidecar value.
    """

    sidecar = provenance_sidecar_path(dataset_path)
    values: dict[str, object] = {}
    if sidecar.exists():
        if not sidecar.is_file():
            raise ValueError(f"dataset provenance sidecar is not a file: {sidecar}")
        try:
            document = tomllib.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"invalid dataset provenance sidecar {sidecar}: {exc}") from exc
        selected = document.get("provenance", document)
        if not isinstance(selected, Mapping):
            raise ValueError("dataset provenance sidecar must contain a table")
        values.update(selected)
    if manifest_override is not None:
        if not isinstance(manifest_override, DatasetProvenanceOverride):
            raise TypeError("manifest_override must be DatasetProvenanceOverride or None")
        values.update(manifest_override.model_dump(exclude_unset=True, mode="python"))
    if not values:
        raise ValueError(
            "dataset provenance is required in the manifest or a .metadata.toml sidecar"
        )
    try:
        return DatasetProvenance.model_validate(values)
    except ValidationError as exc:
        messages = "; ".join(
            f"{'.'.join(str(part) for part in detail['loc'])}: {detail['msg']}"
            for detail in exc.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            )
        )
        raise ValueError(f"invalid dataset provenance: {messages}") from exc
