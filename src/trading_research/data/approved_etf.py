"""Yahoo-backed acquisition for the fixed personal-research ETF study.

The module is deliberately isolated from strategy execution.  It downloads daily
history sequentially, preserves a lossless deterministic representation of the
original yfinance dataframe, and maps Yahoo's already split-adjusted, dividend-
excluded ``Close`` series into the existing provider-neutral CSV schema.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import os
import statistics
import tempfile
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout, suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime
from datetime import time as wall_time
from decimal import Decimal, InvalidOperation, localcontext
from enum import StrEnum
from importlib import metadata
from numbers import Integral, Real
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from trading_research.data.csv_provider import CsvMarketDataProvider
from trading_research.errors import (
    ApprovedDatasetError,
    ApprovedDatasetProviderError,
)
from trading_research.evaluation.manifest import load_split_evaluation_manifest
from trading_research.evaluation.models import DatasetQualitySummary
from trading_research.evaluation.quality import calculate_dataset_quality

APPROVED_ETF_SYMBOLS = ("SPY", "QQQ", "IWM", "XLF", "XLE", "XLV", "TLT", "GLD")
APPROVED_START = date(2005, 1, 3)
APPROVED_END = date(2025, 12, 31)
YFINANCE_REQUEST_END = date(2026, 1, 1)
YFINANCE_VERSION = "1.5.1"
YAHOO_SOURCE_NAME = "Yahoo Finance via yfinance"
YAHOO_SOURCE_TYPE = "unofficial public market-data interface"
YAHOO_USAGE_SCOPE = "personal research only"
TRANSFORMATION_VERSION = "3.0"

_RAW_SCHEMA = "trading-research.yfinance-dataframe"
_RAW_SCHEMA_VERSION = "1.0"
_MARKET_CLOSE_TIME = wall_time(16, 0)
_PREPARED_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")
_REQUIRED_COLUMNS = frozenset(
    {
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Stock Splits",
    }
)
_CONTROL_START = date(2025, 12, 15)
_CONTROL_END_EXCLUSIVE = date(2025, 12, 24)
_SUSPICIOUS_RETURN_THRESHOLD = Decimal("0.40")
_PRICE_COMPARISON_THRESHOLD = Decimal("0.00001")
_KNOWN_VOLUME_WARNING_DATE = date(2025, 9, 30)
_KNOWN_VOLUME_DIFFERENCE = 7_863_106
_KNOWN_VOLUME_RELATIVE_DIFFERENCE = Decimal(
    "0.083515811274697081094299625115396944991809230578768"
)
_ERROR_MESSAGE_MARKERS = (
    "failed download",
    "rate limit",
    "rate-limit",
    "too many requests",
    "throttl",
    "unauthor",
    "forbidden",
)
_REDACTED = "<redacted>"
_PERSONAL_RESEARCH_LIMITATION = (
    "Approved only for personal historical price-based research over the fixed "
    "2005-01-03 through 2025-12-31 period. Not approved for production or live "
    "trading, commercial redistribution, execution-quality analysis, intraday "
    "strategies, or volume-based signals without renewed validation."
)
_PRICE_NOTES = (
    "Yahoo Open, High, Low, Close, and Volume are already split-adjusted. Close "
    "excludes dividend adjustment and is used for execution, valuation, benchmark, "
    "and return calculations. Adj Close includes dividend adjustment and is excluded "
    "from all financial calculations. Dividends and Stock Splits are diagnostic only. "
    "No second split adjustment is performed."
)

Downloader = Callable[..., object]
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    """Return an aware UTC timestamp for acquisition metadata."""

    return datetime.now(UTC)


def _market_timezone() -> ZoneInfo:
    """Load New York exchange time with a portable Windows failure."""

    try:
        return ZoneInfo("America/New_York")
    except ZoneInfoNotFoundError as exc:
        raise ApprovedDatasetError(
            "America/New_York timezone data is unavailable; install project dependencies"
        ) from exc


class DatasetRunMode(StrEnum):
    """Supported acquisition and preparation modes."""

    FULL = "full"
    PREPARE_ONLY = "prepare-only"
    DOWNLOAD_ONLY = "download-only"


class DatasetRecordStatus(StrEnum):
    """Stable per-symbol outcomes."""

    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True)
class YahooRequestParameters:
    """The exact, explicit yfinance request used for primary study data."""

    start: date = APPROVED_START
    end_exclusive: date = YFINANCE_REQUEST_END
    interval: str = "1d"
    auto_adjust: bool = False
    actions: bool = True
    repair: bool = False
    keepna: bool = True
    threads: bool = False
    progress: bool = False

    def as_serializable(self, symbol: str) -> dict[str, object]:
        """Return stable request metadata with the exclusive end made explicit."""

        return {
            "symbol": symbol,
            "start": self.start.isoformat(),
            "end": self.end_exclusive.isoformat(),
            "end_semantics": "exclusive",
            "interval": self.interval,
            "auto_adjust": self.auto_adjust,
            "actions": self.actions,
            "repair": self.repair,
            "keepna": self.keepna,
            "threads": self.threads,
            "progress": self.progress,
        }


APPROVED_REQUEST = YahooRequestParameters()


@dataclass(frozen=True, slots=True)
class RawVendorObservation:
    """One exact provider-neutral view of a Yahoo daily observation."""

    trading_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    adjusted_close: Decimal
    volume: int
    dividend_amount: Decimal
    split_coefficient: Decimal


@dataclass(frozen=True, slots=True)
class RawYahooDataset:
    """Validated lossless raw record plus its provider-neutral rows."""

    symbol: str
    retrieved_at: datetime
    columns: tuple[str, ...]
    dtypes: tuple[str, ...]
    index_name: str | None
    index_dtype: str
    index_timezone: str | None
    request: YahooRequestParameters
    rows: tuple[RawVendorObservation, ...]
    raw_bytes: bytes
    canonical_data_sha256: str
    provider_messages: tuple[str, ...] = ()
    repeatability: RepeatabilityDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class PreparedObservation:
    """One unchanged Yahoo OHLCV row in the existing CSV-provider schema."""

    trading_date: date
    timestamp: datetime
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    adjusted_close: Decimal


@dataclass(frozen=True, slots=True)
class ReturnDiagnostic:
    """One exact close-to-close return identified by its later date."""

    trading_date: date
    value: Decimal


@dataclass(frozen=True, slots=True)
class CorporateActionDiagnostics:
    """Corporate actions and exact price-return diagnostics."""

    split_events: tuple[tuple[date, Decimal], ...]
    dividend_events: tuple[tuple[date, Decimal], ...]
    largest_positive_return: Decimal | None
    largest_negative_return: Decimal | None
    suspicious_returns: tuple[ReturnDiagnostic, ...]
    largest_adjusted_close_difference: Decimal


@dataclass(frozen=True, slots=True)
class RepeatabilityDiagnostic:
    """Two execution-data-equivalent in-memory control downloads."""

    start: date
    end_exclusive: date
    row_count: int
    canonical_sha256: str


@dataclass(frozen=True, slots=True)
class FmpReferenceComparison:
    """Exact Yahoo-versus-FMP SPY validation-reference facts."""

    common_date_count: int
    yahoo_only_date_count: int
    fmp_only_date_count: int
    maximum_absolute_close_difference: Decimal
    maximum_relative_close_difference: Decimal
    median_relative_close_difference: Decimal
    price_differences_above_threshold: int
    volume_difference_date: date
    yahoo_volume: int
    fmp_volume: int
    volume_absolute_difference: int
    volume_relative_difference: Decimal
    validation_reference_only: bool = True
    excluded_from_primary_research_universe: bool = True


@dataclass(frozen=True, slots=True)
class QualityGateSettings:
    """Explicit fixed-universe completeness requirements."""

    minimum_observations: int = 5_000
    expected_first_date: date = APPROVED_START
    expected_last_date: date = APPROVED_END

    def __post_init__(self) -> None:
        """Reject unusable test or production gate settings."""

        if (
            isinstance(self.minimum_observations, bool)
            or not isinstance(self.minimum_observations, int)
            or self.minimum_observations < 1
        ):
            raise ValueError("minimum_observations must be a positive integer")
        if self.expected_first_date > self.expected_last_date:
            raise ValueError("expected first date must not follow expected last date")


_DEFAULT_QUALITY_GATE = QualityGateSettings()


@dataclass(frozen=True, slots=True)
class ApprovedEtfStudyConfig:
    """Filesystem and run-mode configuration for the fixed Yahoo study."""

    output_root: Path
    quality_report_directory: Path
    experiment_manifest: Path
    symbols: tuple[str, ...] = APPROVED_ETF_SYMBOLS
    mode: DatasetRunMode = DatasetRunMode.FULL
    overwrite: bool = False
    timeout_seconds: float = 30.0
    quality_gate: QualityGateSettings = QualityGateSettings()

    def __post_init__(self) -> None:
        """Normalize paths and enforce the fixed ordered symbol subset."""

        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(
            self,
            "quality_report_directory",
            Path(self.quality_report_directory),
        )
        object.__setattr__(self, "experiment_manifest", Path(self.experiment_manifest))
        if not isinstance(self.mode, DatasetRunMode):
            raise TypeError("mode must be a DatasetRunMode")
        if not isinstance(self.overwrite, bool):
            raise TypeError("overwrite must be a bool")
        normalized = tuple(symbol.strip().upper() for symbol in self.symbols)
        if not normalized:
            raise ValueError("symbols must contain at least one approved ETF")
        if len(normalized) != len(set(normalized)):
            raise ValueError("symbols must not contain duplicates")
        unsupported = tuple(symbol for symbol in normalized if symbol not in APPROVED_ETF_SYMBOLS)
        if unsupported:
            raise ValueError(f"unsupported approved-study symbol: {unsupported[0]}")
        object.__setattr__(
            self,
            "symbols",
            tuple(symbol for symbol in APPROVED_ETF_SYMBOLS if symbol in normalized),
        )
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class SymbolDatasetRecord:
    """One ordered symbol outcome with complete audit hashes."""

    symbol: str
    status: DatasetRecordStatus
    raw_path: Path
    prepared_path: Path | None = None
    provenance_path: Path | None = None
    quality_report_path: Path | None = None
    raw_sha256: str | None = None
    canonical_data_sha256: str | None = None
    prepared_sha256: str | None = None
    row_count: int | None = None
    first_date: date | None = None
    last_date: date | None = None
    diagnostics: CorporateActionDiagnostics | None = None
    repeatability: RepeatabilityDiagnostic | None = None
    fmp_comparison: FmpReferenceComparison | None = None
    quality_summary: DatasetQualitySummary | None = None
    quality_gate_passed: bool = False
    review_notes: tuple[str, ...] = ()
    failure_message: str | None = None

    def __post_init__(self) -> None:
        """Require success audit hashes or an explicit failure."""

        if self.status is DatasetRecordStatus.SUCCESS:
            if self.failure_message is not None:
                raise ValueError("successful records cannot contain a failure")
            if self.raw_sha256 is None or self.canonical_data_sha256 is None:
                raise ValueError("successful records require raw data hashes")
        elif not self.failure_message:
            raise ValueError("failed records require a failure message")


@dataclass(frozen=True, slots=True)
class YahooSpyDiagnostic:
    """Safe summary of the authoritative SPY primary-data acquisition."""

    valid_data: bool
    earliest_date: date
    latest_date: date
    required_fields_present: bool
    row_count: int
    split_event_count: int
    dividend_event_count: int
    repeatability_sha256: str
    quality_gate_passed: bool


@dataclass(frozen=True, slots=True)
class ApprovedEtfStudyResult:
    """Ordered result of the sequential approved-study workflow."""

    mode: DatasetRunMode
    records: tuple[SymbolDatasetRecord, ...]
    quality_summary_text: Path | None = None
    quality_summary_csv: Path | None = None
    experiment_manifest: Path | None = None
    provider_diagnostic: YahooSpyDiagnostic | None = None

    @property
    def succeeded(self) -> bool:
        """Return whether every requested symbol completed its requested stage."""

        return bool(self.records) and all(
            record.status is DatasetRecordStatus.SUCCESS for record in self.records
        )


def validate_yfinance_version(actual_version: str) -> None:
    """Reject silent upgrades or downgrades at the live integration boundary."""

    if actual_version != YFINANCE_VERSION:
        raise ApprovedDatasetProviderError(
            f"yfinance version must be exactly {YFINANCE_VERSION}; found {actual_version}"
        )


def redact_secret(message: str, secret: str | None = None) -> str:
    """Retain provider-neutral secret redaction for all persisted failures."""

    redacted = message
    for label in (
        "apikey",
        "api_key",
        "token",
        "authorization",
        "cookie",
        "crumb",
        "session",
    ):
        redacted = _redact_label(redacted, label)
    if secret:
        redacted = redacted.replace(secret, _REDACTED)
    return redacted


def _redact_label(message: str, label: str) -> str:
    """Redact one simple key/value representation without regex backtracking."""

    lowered = message.lower()
    start = 0
    result = message
    while True:
        position = lowered.find(label, start)
        if position < 0:
            return result
        separator = position + len(label)
        while separator < len(result) and result[separator] in " \t":
            separator += 1
        if separator >= len(result) or result[separator] not in "=:":
            start = position + len(label)
            continue
        value_start = separator + 1
        while value_start < len(result) and result[value_start] in " \t":
            value_start += 1
        value_end = value_start
        while value_end < len(result) and result[value_end] not in " \t,;&'\"":
            value_end += 1
        result = result[:value_start] + _REDACTED + result[value_end:]
        lowered = result.lower()
        start = value_start + len(_REDACTED)


def _default_downloader(symbol: str, **parameters: object) -> object:
    """Call the exact pinned yfinance boundary without relying on defaults."""

    try:
        actual_version = metadata.version("yfinance")
    except metadata.PackageNotFoundError as exc:
        raise ApprovedDatasetProviderError(
            f"yfinance {YFINANCE_VERSION} is required but is not installed"
        ) from exc
    validate_yfinance_version(actual_version)
    try:
        import yfinance as yf  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ApprovedDatasetProviderError("could not import pinned yfinance") from exc
    return cast(object, yf.download(symbol, **parameters))


def _download_frame(
    symbol: str,
    start: date,
    end_exclusive: date,
    *,
    downloader: Downloader,
    timeout_seconds: float,
) -> tuple[object, tuple[str, ...]]:
    """Perform one explicit sequential download and safely capture diagnostics."""

    stream = io.StringIO()
    logger = logging.getLogger("yfinance")
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    captured_warnings: list[warnings.WarningMessage] = []
    try:
        with (
            warnings.catch_warnings(record=True) as captured_warnings,
            redirect_stdout(stream),
            redirect_stderr(stream),
        ):
            warnings.simplefilter("always")
            frame = downloader(
                symbol,
                start=start.isoformat(),
                end=end_exclusive.isoformat(),
                interval="1d",
                auto_adjust=False,
                actions=True,
                repair=False,
                keepna=True,
                threads=False,
                progress=False,
                rounding=False,
                timeout=timeout_seconds,
                multi_level_index=False,
            )
    except ApprovedDatasetProviderError:
        raise
    except Exception as exc:
        raise ApprovedDatasetProviderError(
            f"Yahoo download failed for {symbol}: {redact_secret(str(exc))}"
        ) from exc
    finally:
        logger.removeHandler(handler)
    messages = tuple(
        dict.fromkeys(
            redact_secret(message)
            for message in (
                *(
                    line.strip()
                    for line in stream.getvalue().splitlines()
                    if line.strip()
                ),
                *(
                    str(item.message).strip()
                    for item in captured_warnings
                    if str(item.message).strip()
                ),
            )
        )
    )
    if any(
        marker in message.lower()
        for message in messages
        for marker in _ERROR_MESSAGE_MARKERS
    ):
        raise ApprovedDatasetProviderError(
            f"Yahoo reported a provider failure for {symbol}: "
            f"{redact_secret(messages[0])}"
        )
    return frame, messages


def snapshot_yfinance_frame(
    frame: object,
    symbol: str,
    *,
    retrieved_at: datetime,
    request: YahooRequestParameters = APPROVED_REQUEST,
    provider_messages: Sequence[str] = (),
    repeatability: RepeatabilityDiagnostic | None = None,
) -> RawYahooDataset:
    """Preserve and validate a yfinance dataframe without display-oriented loss."""

    if symbol not in APPROVED_ETF_SYMBOLS:
        raise ApprovedDatasetError(f"symbol is not in the approved ETF universe: {symbol}")
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("retrieved_at must be timezone-aware")
    dynamic = cast(Any, frame)
    try:
        columns_object = dynamic.columns
        if getattr(columns_object, "nlevels", 1) != 1:
            raise ApprovedDatasetError("Yahoo dataframe must use single-level columns")
        columns = tuple(str(value) for value in list(columns_object))
        dtypes = tuple(str(value) for value in list(dynamic.dtypes))
        index_values = tuple(dynamic.index.tolist())
        matrix = tuple(tuple(row) for row in dynamic.to_numpy(copy=True).tolist())
        index_name_value = dynamic.index.name
        index_dtype = str(dynamic.index.dtype)
        index_timezone_value = getattr(dynamic.index, "tz", None)
    except ApprovedDatasetError:
        raise
    except Exception as exc:
        raise ApprovedDatasetProviderError(
            "Yahoo returned an unsupported dataframe representation"
        ) from exc
    if not columns or not matrix:
        raise ApprovedDatasetProviderError(f"Yahoo returned no observations for {symbol}")
    if len(columns) != len(set(columns)):
        raise ApprovedDatasetError("Yahoo dataframe contains duplicate columns")
    missing = sorted(_REQUIRED_COLUMNS.difference(columns))
    if missing:
        raise ApprovedDatasetProviderError(
            f"Yahoo dataframe is missing required column(s): {', '.join(missing)}"
        )
    if len(columns) != len(dtypes):
        raise ApprovedDatasetProviderError("Yahoo column dtype metadata is incomplete")
    if len(index_values) != len(matrix):
        raise ApprovedDatasetProviderError("Yahoo index and row counts differ")
    if any(len(row) != len(columns) for row in matrix):
        raise ApprovedDatasetProviderError("Yahoo dataframe row width is inconsistent")

    encoded_rows = [
        {
            "index": _encode_index_value(index_value),
            "values": [_encode_scalar(value) for value in values],
        }
        for index_value, values in zip(index_values, matrix, strict=True)
    ]
    data_document = {
        "schema": _RAW_SCHEMA,
        "schema_version": _RAW_SCHEMA_VERSION,
        "source_name": YAHOO_SOURCE_NAME,
        "yfinance_version": YFINANCE_VERSION,
        "request": request.as_serializable(symbol),
        "index": {
            "name": None if index_name_value is None else str(index_name_value),
            "dtype": index_dtype,
            "timezone": (
                None if index_timezone_value is None else str(index_timezone_value)
            ),
            "trading_date_interpretation": "America/New_York exchange session date",
        },
        "columns": [
            {"name": name, "dtype": dtype}
            for name, dtype in zip(columns, dtypes, strict=True)
        ],
        "rows": encoded_rows,
    }
    canonical_bytes = _json_bytes(data_document)
    document = dict(data_document)
    document["retrieved_at"] = retrieved_at.astimezone(UTC).isoformat()
    document["provider_messages"] = list(provider_messages)
    if repeatability is not None:
        document["repeatability"] = {
            "start": repeatability.start.isoformat(),
            "end_exclusive": repeatability.end_exclusive.isoformat(),
            "row_count": repeatability.row_count,
            "canonical_sha256": repeatability.canonical_sha256,
        }
    raw_bytes = _json_bytes(document)
    parsed = parse_raw_yfinance_bytes(raw_bytes)
    return RawYahooDataset(
        symbol=parsed.symbol,
        retrieved_at=parsed.retrieved_at,
        columns=parsed.columns,
        dtypes=parsed.dtypes,
        index_name=parsed.index_name,
        index_dtype=parsed.index_dtype,
        index_timezone=parsed.index_timezone,
        request=parsed.request,
        rows=parsed.rows,
        raw_bytes=raw_bytes,
        canonical_data_sha256=_sha256_bytes(canonical_bytes),
        provider_messages=tuple(provider_messages),
        repeatability=repeatability,
    )


def _encode_index_value(value: object) -> dict[str, str]:
    """Encode one dataframe index value with its exact temporal representation."""

    if isinstance(value, datetime):
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return {"type": type(value).__name__, "value": str(isoformat())}
    raise ApprovedDatasetProviderError("Yahoo dataframe index must contain dates")


def _encode_scalar(value: object) -> dict[str, str]:
    """Encode one dataframe scalar losslessly while retaining a Decimal entry text."""

    if isinstance(value, bool):
        return {"type": "bool", "value": str(value).lower()}
    if isinstance(value, Integral):
        return {"type": "integer", "value": str(int(value))}
    if isinstance(value, Real):
        floating = float(value)
        if not math.isfinite(floating):
            return {"type": "null", "value": str(value)}
        return {
            "type": "float",
            "hex": floating.hex(),
            "decimal": str(value),
        }
    if value is None:
        return {"type": "null", "value": "None"}
    if isinstance(value, str):
        return {"type": "string", "value": value}
    raise ApprovedDatasetProviderError(
        f"Yahoo dataframe contains unsupported scalar type: {type(value).__name__}"
    )


def parse_raw_yfinance_bytes(raw_bytes: bytes) -> RawYahooDataset:
    """Reload and validate the deterministic raw yfinance representation."""

    try:
        document = json.loads(
            raw_bytes.decode("utf-8"),
            parse_float=Decimal,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ApprovedDatasetError("raw Yahoo file is not valid UTF-8 JSON") from exc
    if not isinstance(document, Mapping):
        raise ApprovedDatasetError("raw Yahoo document must be an object")
    if document.get("schema") != _RAW_SCHEMA:
        raise ApprovedDatasetError("raw Yahoo document has an unsupported schema")
    if document.get("schema_version") != _RAW_SCHEMA_VERSION:
        raise ApprovedDatasetError("raw Yahoo document has an unsupported schema version")
    if document.get("source_name") != YAHOO_SOURCE_NAME:
        raise ApprovedDatasetError("raw Yahoo source name is invalid")
    if document.get("yfinance_version") != YFINANCE_VERSION:
        raise ApprovedDatasetError("raw Yahoo file was not created by pinned yfinance")

    request_mapping = document.get("request")
    if not isinstance(request_mapping, Mapping):
        raise ApprovedDatasetError("raw Yahoo request metadata is missing")
    symbol = _required_text(request_mapping.get("symbol"), "request symbol").upper()
    if symbol not in APPROVED_ETF_SYMBOLS:
        raise ApprovedDatasetError(f"raw Yahoo symbol is not approved: {symbol}")
    try:
        request = YahooRequestParameters(
            start=date.fromisoformat(
                _required_text(request_mapping.get("start"), "request start")
            ),
            end_exclusive=date.fromisoformat(
                _required_text(request_mapping.get("end"), "request end")
            ),
        )
    except ValueError as exc:
        raise ApprovedDatasetError("raw Yahoo request dates are invalid") from exc
    expected_request = request.as_serializable(symbol)
    if dict(request_mapping) != expected_request:
        raise ApprovedDatasetError("raw Yahoo request parameters do not match the approved request")

    retrieved_at_value = _required_text(document.get("retrieved_at"), "retrieved_at")
    try:
        retrieved_at = datetime.fromisoformat(retrieved_at_value)
    except ValueError as exc:
        raise ApprovedDatasetError("raw Yahoo retrieval timestamp is invalid") from exc
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ApprovedDatasetError("raw Yahoo retrieval timestamp must be timezone-aware")

    index = document.get("index")
    if not isinstance(index, Mapping):
        raise ApprovedDatasetError("raw Yahoo index metadata is missing")
    index_name_value = index.get("name")
    if index_name_value is not None and not isinstance(index_name_value, str):
        raise ApprovedDatasetError("raw Yahoo index name must be text or null")
    index_dtype = _required_text(index.get("dtype"), "index dtype")
    index_timezone_value = index.get("timezone")
    if index_timezone_value is not None and not isinstance(index_timezone_value, str):
        raise ApprovedDatasetError("raw Yahoo index timezone must be text or null")

    raw_columns = document.get("columns")
    if not isinstance(raw_columns, list) or not raw_columns:
        raise ApprovedDatasetError("raw Yahoo columns metadata is missing")
    columns: list[str] = []
    dtypes: list[str] = []
    for item in raw_columns:
        if not isinstance(item, Mapping):
            raise ApprovedDatasetError("raw Yahoo column metadata must be objects")
        columns.append(_required_text(item.get("name"), "column name"))
        dtypes.append(_required_text(item.get("dtype"), "column dtype"))
    if len(columns) != len(set(columns)):
        raise ApprovedDatasetError("raw Yahoo file contains duplicate columns")
    missing = sorted(_REQUIRED_COLUMNS.difference(columns))
    if missing:
        raise ApprovedDatasetError(
            f"raw Yahoo file is missing required column(s): {', '.join(missing)}"
        )

    raw_rows = document.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ApprovedDatasetError("raw Yahoo file contains no observations")
    column_positions = {name: index for index, name in enumerate(columns)}
    rows: list[RawVendorObservation] = []
    seen_dates: set[date] = set()
    for row_number, item in enumerate(raw_rows, start=1):
        if not isinstance(item, Mapping):
            raise ApprovedDatasetError(f"raw Yahoo row {row_number} must be an object")
        trading_date = _decode_trading_date(item.get("index"), row_number)
        if trading_date in seen_dates:
            raise ApprovedDatasetError(
                f"raw Yahoo observations contain duplicate date {trading_date}"
            )
        seen_dates.add(trading_date)
        values = item.get("values")
        if not isinstance(values, list) or len(values) != len(columns):
            raise ApprovedDatasetError(f"raw Yahoo row {row_number} has invalid width")
        open_price = _decoded_decimal(
            values[column_positions["Open"]],
            row_number,
            "Open",
        )
        high = _decoded_decimal(values[column_positions["High"]], row_number, "High")
        low = _decoded_decimal(values[column_positions["Low"]], row_number, "Low")
        close = _decoded_decimal(values[column_positions["Close"]], row_number, "Close")
        adjusted_close = _decoded_decimal(
            values[column_positions["Adj Close"]],
            row_number,
            "Adj Close",
        )
        volume = _decoded_integer(
            values[column_positions["Volume"]],
            row_number,
            "Volume",
        )
        dividend = _decoded_decimal(
            values[column_positions["Dividends"]],
            row_number,
            "Dividends",
        )
        split = _decoded_decimal(
            values[column_positions["Stock Splits"]],
            row_number,
            "Stock Splits",
        )
        _validate_ohlcv(
            trading_date,
            open_price,
            high,
            low,
            close,
            adjusted_close,
            volume,
            dividend,
            split,
        )
        rows.append(
            RawVendorObservation(
                trading_date=trading_date,
                open=open_price,
                high=high,
                low=low,
                close=close,
                adjusted_close=adjusted_close,
                volume=volume,
                dividend_amount=dividend,
                split_coefficient=split,
            )
        )
    if any(
        current.trading_date <= previous.trading_date
        for previous, current in zip(rows, rows[1:], strict=False)
    ):
        raise ApprovedDatasetError("raw Yahoo observations must be strictly chronological")

    canonical_document = {
        key: value
        for key, value in document.items()
        if key not in {"retrieved_at", "provider_messages", "repeatability"}
    }
    provider_messages_value = document.get("provider_messages", [])
    if not isinstance(provider_messages_value, list) or any(
        not isinstance(item, str) for item in provider_messages_value
    ):
        raise ApprovedDatasetError("raw Yahoo provider messages must be text")
    repeatability_value = document.get("repeatability")
    repeatability = (
        None
        if repeatability_value is None
        else _parse_repeatability(repeatability_value)
    )
    return RawYahooDataset(
        symbol=symbol,
        retrieved_at=retrieved_at.astimezone(UTC),
        columns=tuple(columns),
        dtypes=tuple(dtypes),
        index_name=index_name_value,
        index_dtype=index_dtype,
        index_timezone=index_timezone_value,
        request=request,
        rows=tuple(rows),
        raw_bytes=raw_bytes,
        canonical_data_sha256=_sha256_bytes(_json_bytes(canonical_document)),
        provider_messages=tuple(cast(list[str], provider_messages_value)),
        repeatability=repeatability,
    )


def _parse_repeatability(value: object) -> RepeatabilityDiagnostic:
    """Reload an acquisition control retained with the raw response."""

    if not isinstance(value, Mapping):
        raise ApprovedDatasetError("raw Yahoo repeatability metadata is invalid")
    try:
        start = date.fromisoformat(
            _required_text(value.get("start"), "repeatability start")
        )
        end_exclusive = date.fromisoformat(
            _required_text(
                value.get("end_exclusive"),
                "repeatability end_exclusive",
            )
        )
    except ValueError as exc:
        raise ApprovedDatasetError("raw Yahoo repeatability dates are invalid") from exc
    row_count = value.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1:
        raise ApprovedDatasetError("raw Yahoo repeatability row_count is invalid")
    sha256 = _required_text(
        value.get("canonical_sha256"),
        "repeatability SHA-256",
    )
    if len(sha256) != 64 or any(
        character not in "0123456789abcdef" for character in sha256
    ):
        raise ApprovedDatasetError("raw Yahoo repeatability SHA-256 is invalid")
    return RepeatabilityDiagnostic(
        start=start,
        end_exclusive=end_exclusive,
        row_count=row_count,
        canonical_sha256=sha256,
    )


def _reject_json_constant(value: str) -> Decimal:
    """Reject non-standard JSON numeric constants."""

    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _required_text(value: object, name: str) -> str:
    """Return required non-blank text."""

    if not isinstance(value, str) or not value.strip():
        raise ApprovedDatasetError(f"raw Yahoo {name} must be non-blank text")
    return value.strip()


def _decode_trading_date(value: object, row_number: int) -> date:
    """Decode a preserved dataframe index as an exchange-session date."""

    if not isinstance(value, Mapping):
        raise ApprovedDatasetError(f"raw Yahoo row {row_number} index is invalid")
    index_type = _required_text(value.get("type"), "index type")
    encoded = _required_text(value.get("value"), "index value")
    try:
        if index_type == "date":
            return date.fromisoformat(encoded)
        parsed = datetime.fromisoformat(encoded)
    except ValueError as exc:
        raise ApprovedDatasetError(
            f"raw Yahoo row {row_number} index is not an ISO date"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.date()
    return parsed.astimezone(_market_timezone()).date()


def _decoded_decimal(value: object, row_number: int, column: str) -> Decimal:
    """Decode one finite source scalar directly into Decimal."""

    if not isinstance(value, Mapping):
        raise ApprovedDatasetError(
            f"raw Yahoo row {row_number} {column} scalar is invalid"
        )
    value_type = value.get("type")
    if value_type == "integer":
        encoded = value.get("value")
    elif value_type == "float":
        encoded = value.get("decimal")
        hexadecimal = value.get("hex")
        if not isinstance(hexadecimal, str):
            raise ApprovedDatasetError(
                f"raw Yahoo row {row_number} {column} float encoding is incomplete"
            )
        try:
            restored = float.fromhex(hexadecimal)
        except ValueError as exc:
            raise ApprovedDatasetError(
                f"raw Yahoo row {row_number} {column} float encoding is invalid"
            ) from exc
        if not math.isfinite(restored):
            raise ApprovedDatasetError(
                f"raw Yahoo row {row_number} {column} must be finite"
            )
    else:
        raise ApprovedDatasetError(
            f"raw Yahoo row {row_number} {column} is null or non-numeric"
        )
    if not isinstance(encoded, str):
        raise ApprovedDatasetError(
            f"raw Yahoo row {row_number} {column} numeric text is missing"
        )
    try:
        parsed = Decimal(encoded)
    except InvalidOperation as exc:
        raise ApprovedDatasetError(
            f"raw Yahoo row {row_number} {column} is not numeric"
        ) from exc
    if not parsed.is_finite():
        raise ApprovedDatasetError(
            f"raw Yahoo row {row_number} {column} must be finite"
        )
    return parsed


def _decoded_integer(value: object, row_number: int, column: str) -> int:
    """Decode a non-negative whole-number provider scalar."""

    parsed = _decoded_decimal(value, row_number, column)
    if parsed != parsed.to_integral_value():
        raise ApprovedDatasetError(
            f"raw Yahoo row {row_number} {column} must be an integer"
        )
    return int(parsed)


def _validate_ohlcv(
    trading_date: date,
    open_price: Decimal,
    high: Decimal,
    low: Decimal,
    close: Decimal,
    adjusted_close: Decimal,
    volume: int,
    dividend: Decimal,
    split: Decimal,
) -> None:
    """Enforce the primary price-study schema and volume warning policy."""

    if min(open_price, high, low, close, adjusted_close) <= 0:
        raise ApprovedDatasetError(f"Yahoo contains non-positive price on {trading_date}")
    if low > min(open_price, close) or high < max(open_price, close) or low > high:
        raise ApprovedDatasetError(f"Yahoo contains inconsistent OHLC on {trading_date}")
    if volume <= 0:
        raise ApprovedDatasetError(
            f"Yahoo volume must be positive for this study on {trading_date}"
        )
    if dividend < 0:
        raise ApprovedDatasetError(f"Yahoo contains negative dividend on {trading_date}")
    if split < 0:
        raise ApprovedDatasetError(f"Yahoo contains negative split value on {trading_date}")


def transform_yahoo_rows(
    rows: Sequence[RawVendorObservation],
    symbol: str,
    *,
    approved_start: date = APPROVED_START,
    approved_end: date = APPROVED_END,
) -> tuple[PreparedObservation, ...]:
    """Map Yahoo OHLCV unchanged; never apply a second split adjustment."""

    if symbol not in APPROVED_ETF_SYMBOLS:
        raise ApprovedDatasetError(f"symbol is not in the approved ETF universe: {symbol}")
    if approved_start > approved_end:
        raise ValueError("approved_start must not follow approved_end")
    snapshot = tuple(rows)
    if not snapshot:
        raise ApprovedDatasetError("raw Yahoo data contains no observations")
    if len({row.trading_date for row in snapshot}) != len(snapshot):
        raise ApprovedDatasetError("raw Yahoo data contains duplicate dates")
    prepared = tuple(
        PreparedObservation(
            trading_date=row.trading_date,
            timestamp=datetime.combine(
                row.trading_date,
                _MARKET_CLOSE_TIME,
                tzinfo=_market_timezone(),
            ).astimezone(UTC),
            symbol=symbol,
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
            volume=row.volume,
            adjusted_close=row.adjusted_close,
        )
        for row in sorted(snapshot, key=lambda item: item.trading_date)
        if approved_start <= row.trading_date <= approved_end
    )
    selected_raw = tuple(
        row
        for row in sorted(snapshot, key=lambda item: item.trading_date)
        if approved_start <= row.trading_date <= approved_end
    )
    for raw, output in zip(
        selected_raw,
        prepared,
        strict=True,
    ):
        if (
            raw.open != output.open
            or raw.high != output.high
            or raw.low != output.low
            or raw.close != output.close
            or raw.volume != output.volume
        ):
            raise ApprovedDatasetError("Yahoo OHLCV changed during no-adjustment mapping")
    return prepared


def prepared_csv_bytes(rows: Sequence[PreparedObservation]) -> bytes:
    """Serialize canonical chronological project-schema CSV bytes."""

    snapshot = tuple(rows)
    if not snapshot:
        raise ApprovedDatasetError("prepared dataset contains no observations")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(_PREPARED_COLUMNS)
    for row in snapshot:
        writer.writerow(
            (
                row.timestamp.isoformat(),
                row.symbol,
                _decimal_text(row.open),
                _decimal_text(row.high),
                _decimal_text(row.low),
                _decimal_text(row.close),
                str(row.volume),
            )
        )
    return stream.getvalue().encode("utf-8")


def calculate_corporate_action_diagnostics(
    rows: Sequence[RawVendorObservation],
) -> CorporateActionDiagnostics:
    """Calculate exact actions, returns, and adjusted-close separation."""

    ordered = tuple(sorted(rows, key=lambda row: row.trading_date))
    returns = _close_returns(
        tuple((row.trading_date, row.close) for row in ordered)
    )
    differences = tuple(abs(row.close - row.adjusted_close) for row in ordered)
    return CorporateActionDiagnostics(
        split_events=tuple(
            (row.trading_date, row.split_coefficient)
            for row in ordered
            if row.split_coefficient != 0
        ),
        dividend_events=tuple(
            (row.trading_date, row.dividend_amount)
            for row in ordered
            if row.dividend_amount != 0
        ),
        largest_positive_return=_largest_positive(returns),
        largest_negative_return=_largest_negative(returns),
        suspicious_returns=tuple(
            item for item in returns if abs(item.value) >= _SUSPICIOUS_RETURN_THRESHOLD
        ),
        largest_adjusted_close_difference=max(differences, default=Decimal("0")),
    )


def _close_returns(
    observations: Sequence[tuple[date, Decimal]],
) -> tuple[ReturnDiagnostic, ...]:
    """Return exact adjacent close returns."""

    with localcontext() as context:
        context.prec = 50
        return tuple(
            ReturnDiagnostic(
                trading_date=current[0],
                value=(current[1] / previous[1]) - Decimal("1"),
            )
            for previous, current in zip(observations, observations[1:], strict=False)
        )


def _largest_positive(values: Sequence[ReturnDiagnostic]) -> Decimal | None:
    """Return the largest positive return when present."""

    selected = tuple(item.value for item in values if item.value > 0)
    return max(selected) if selected else None


def _largest_negative(values: Sequence[ReturnDiagnostic]) -> Decimal | None:
    """Return the most negative return when present."""

    selected = tuple(item.value for item in values if item.value < 0)
    return min(selected) if selected else None


def _validate_quality_gate(
    dataset: RawYahooDataset,
    settings: QualityGateSettings,
) -> CorporateActionDiagnostics:
    """Require exact coverage, complete rows, actions, and plausible price returns."""

    rows = dataset.rows
    if len(rows) < settings.minimum_observations:
        raise ApprovedDatasetError(
            f"quality gate failed: {len(rows)} rows is below "
            f"the required {settings.minimum_observations}"
        )
    if rows[0].trading_date != settings.expected_first_date:
        raise ApprovedDatasetError(
            "quality gate failed: first Yahoo trading date must be "
            f"{settings.expected_first_date}; found {rows[0].trading_date}"
        )
    if rows[-1].trading_date != settings.expected_last_date:
        raise ApprovedDatasetError(
            "quality gate failed: final Yahoo trading date must be "
            f"{settings.expected_last_date}; found {rows[-1].trading_date}"
        )
    diagnostics = calculate_corporate_action_diagnostics(rows)
    if diagnostics.suspicious_returns:
        first = diagnostics.suspicious_returns[0]
        raise ApprovedDatasetError(
            "quality gate failed: unexplained extreme close return "
            f"{_decimal_text(first.value)} on {first.trading_date}"
        )
    return diagnostics


def _repeatability_control(
    symbol: str,
    *,
    downloader: Downloader,
    timeout_seconds: float,
    clock: Clock,
) -> RepeatabilityDiagnostic:
    """Download a fixed range twice and require equivalent execution fields."""

    snapshots: list[RawYahooDataset] = []
    request = YahooRequestParameters(
        start=_CONTROL_START,
        end_exclusive=_CONTROL_END_EXCLUSIVE,
    )
    for _ in range(2):
        frame, messages = _download_frame(
            symbol,
            _CONTROL_START,
            _CONTROL_END_EXCLUSIVE,
            downloader=downloader,
            timeout_seconds=timeout_seconds,
        )
        snapshots.append(
            snapshot_yfinance_frame(
                frame,
                symbol,
                retrieved_at=_aware_clock_value(clock),
                request=request,
                provider_messages=messages,
            )
        )
    first, second = snapshots
    first_hash = _execution_data_sha256(first.rows)
    second_hash = _execution_data_sha256(second.rows)
    if first_hash != second_hash:
        raise ApprovedDatasetProviderError(
            f"repeated Yahoo execution-data control downloads disagree for {symbol}"
        )
    if not first.rows:
        raise ApprovedDatasetProviderError(
            f"Yahoo repeatability control returned no rows for {symbol}"
        )
    return RepeatabilityDiagnostic(
        start=_CONTROL_START,
        end_exclusive=_CONTROL_END_EXCLUSIVE,
        row_count=len(first.rows),
        canonical_sha256=first_hash,
    )


def _execution_data_sha256(rows: Sequence[RawVendorObservation]) -> str:
    """Hash approved execution OHLCV and action diagnostics, excluding Adj Close."""

    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Dividends",
            "Stock Splits",
        )
    )
    for row in rows:
        writer.writerow(
            (
                row.trading_date.isoformat(),
                _decimal_text(row.open),
                _decimal_text(row.high),
                _decimal_text(row.low),
                _decimal_text(row.close),
                str(row.volume),
                _decimal_text(row.dividend_amount),
                _decimal_text(row.split_coefficient),
            )
        )
    return _sha256_bytes(stream.getvalue().encode("utf-8"))


def _json_bytes(value: object) -> bytes:
    """Serialize stable UTF-8 JSON without binary floating-point output."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _decimal_text(value: Decimal) -> str:
    """Serialize a finite Decimal without exponent notation."""

    if not value.is_finite():
        raise ApprovedDatasetError("cannot serialize a non-finite Decimal")
    return format(value, "f")


def _sha256_bytes(value: bytes) -> str:
    """Return a lowercase SHA-256 digest."""

    return hashlib.sha256(value).hexdigest()


def _read_bytes(path: Path, description: str) -> bytes:
    """Read one local file with a stable, filename-only error."""

    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise ApprovedDatasetError(f"{description} not found: {path.name}") from exc
    except IsADirectoryError as exc:
        raise ApprovedDatasetError(f"{description} is not a file: {path.name}") from exc
    except OSError as exc:
        raise ApprovedDatasetError(f"could not read {description}: {path.name}") from exc


def _preflight_new_files(paths: Sequence[Path], overwrite: bool) -> None:
    """Reject existing destinations before any replacement."""

    if overwrite:
        return
    for path in paths:
        if path.exists():
            raise ApprovedDatasetError(f"output file already exists: {path.name}")


def _temporary_file(path: Path, data: bytes) -> Path:
    """Write and fsync bytes to a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            return Path(handle.name)
    except OSError as exc:
        raise ApprovedDatasetError(
            f"could not create temporary file for {path.name}"
        ) from exc


def _publish_temporary_file(
    temporary: Path,
    destination: Path,
    overwrite: bool,
) -> None:
    """Atomically publish a temporary file without silent overwrite."""

    try:
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
    except FileExistsError as exc:
        raise ApprovedDatasetError(f"output file already exists: {destination.name}") from exc
    except OSError as exc:
        raise ApprovedDatasetError(f"could not publish {destination.name}") from exc
    finally:
        if temporary.exists():
            with suppress(OSError):
                temporary.unlink()


def _atomic_write(path: Path, data: bytes, *, overwrite: bool) -> None:
    """Atomically publish immutable bytes."""

    if not overwrite and path.exists():
        raise ApprovedDatasetError(f"output file already exists: {path.name}")
    temporary = _temporary_file(path, data)
    _publish_temporary_file(temporary, path, overwrite)


def _validate_prepared_csv(
    symbol: str,
    rows: Sequence[PreparedObservation],
    destination: Path,
    data: bytes,
    *,
    overwrite: bool,
) -> DatasetQualitySummary:
    """Validate prepared bytes through the existing provider before publication."""

    temporary = _temporary_file(destination, data)
    try:
        bars = tuple(
            CsvMarketDataProvider(temporary).get_historical_bars(
                symbol=symbol,
                start=None,
                end=None,
            )
        )
        if len(bars) != len(rows):
            raise ApprovedDatasetError(
                "prepared CSV provider row count differs from Yahoo rows"
            )
        quality = calculate_dataset_quality(bars)
        if quality.duplicate_timestamp_count != 0:
            raise ApprovedDatasetError("prepared CSV contains duplicate timestamps")
        if quality.non_increasing_timestamp_count != 0:
            raise ApprovedDatasetError("prepared CSV is not strictly chronological")
        if quality.zero_volume_count != 0:
            raise ApprovedDatasetError("prepared CSV contains zero volume")
        _publish_temporary_file(temporary, destination, overwrite)
        return quality
    except ApprovedDatasetError:
        raise
    except Exception as exc:
        raise ApprovedDatasetError(
            f"prepared Yahoo CSV failed existing provider validation: {exc}"
        ) from exc
    finally:
        if temporary.exists():
            with suppress(OSError):
                temporary.unlink()


def compare_yahoo_spy_with_fmp(
    yahoo_rows: Sequence[PreparedObservation],
    fmp_path: Path,
) -> FmpReferenceComparison:
    """Compare primary Yahoo SPY data with the retained FMP validation reference."""

    fmp = _read_fmp_reference(fmp_path)
    yahoo = {row.trading_date: (row.close, row.volume) for row in yahoo_rows}
    common = sorted(set(yahoo).intersection(fmp))
    if not common:
        raise ApprovedDatasetError("FMP SPY reference has no dates in common with Yahoo")
    yahoo_only = set(yahoo).difference(fmp)
    fmp_only = set(fmp).difference(yahoo)
    price_differences: list[tuple[Decimal, Decimal]] = []
    with localcontext() as context:
        context.prec = 50
        for trading_date in common:
            yahoo_close = yahoo[trading_date][0]
            fmp_close = fmp[trading_date][0]
            absolute = abs(yahoo_close - fmp_close)
            relative = absolute / abs(fmp_close)
            price_differences.append((absolute, relative))
    relatives = tuple(item[1] for item in price_differences)
    maximum_absolute = max(item[0] for item in price_differences)
    maximum_relative = max(relatives)
    if yahoo_only or fmp_only:
        raise ApprovedDatasetError(
            "Yahoo and FMP SPY validation-reference trading dates differ"
        )
    if maximum_relative > _PRICE_COMPARISON_THRESHOLD:
        raise ApprovedDatasetError(
            "Yahoo SPY Close differs materially from the retained FMP reference"
        )
    try:
        yahoo_volume = yahoo[_KNOWN_VOLUME_WARNING_DATE][1]
        fmp_volume = fmp[_KNOWN_VOLUME_WARNING_DATE][1]
    except KeyError as exc:
        raise ApprovedDatasetError(
            "known SPY cross-provider volume-warning date is unavailable"
        ) from exc
    absolute_volume = abs(yahoo_volume - fmp_volume)
    with localcontext() as context:
        context.prec = 50
        relative_volume = Decimal(absolute_volume) / Decimal(fmp_volume)
    return FmpReferenceComparison(
        common_date_count=len(common),
        yahoo_only_date_count=len(yahoo_only),
        fmp_only_date_count=len(fmp_only),
        maximum_absolute_close_difference=maximum_absolute,
        maximum_relative_close_difference=maximum_relative,
        median_relative_close_difference=statistics.median(relatives),
        price_differences_above_threshold=sum(
            value > _PRICE_COMPARISON_THRESHOLD for value in relatives
        ),
        volume_difference_date=_KNOWN_VOLUME_WARNING_DATE,
        yahoo_volume=yahoo_volume,
        fmp_volume=fmp_volume,
        volume_absolute_difference=absolute_volume,
        volume_relative_difference=relative_volume,
    )


def _read_fmp_reference(path: Path) -> dict[date, tuple[Decimal, int]]:
    """Read the unchanged provider-neutral FMP SPY reference CSV."""

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if tuple(reader.fieldnames or ()) != _PREPARED_COLUMNS:
                raise ApprovedDatasetError("FMP SPY reference schema is invalid")
            result: dict[date, tuple[Decimal, int]] = {}
            for row in reader:
                timestamp = datetime.fromisoformat(row["timestamp"])
                if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                    raise ApprovedDatasetError("FMP SPY timestamps must be timezone-aware")
                trading_date = timestamp.astimezone(_market_timezone()).date()
                if trading_date in result:
                    raise ApprovedDatasetError("FMP SPY contains duplicate dates")
                result[trading_date] = (Decimal(row["close"]), int(row["volume"]))
            return result
    except ApprovedDatasetError:
        raise
    except (OSError, UnicodeError, csv.Error, ValueError, InvalidOperation) as exc:
        raise ApprovedDatasetError("could not validate retained FMP SPY reference") from exc


def _raw_path(output_root: Path, symbol: str) -> Path:
    """Return the independent raw Yahoo dataframe record path."""

    return (
        output_root
        / "incoming"
        / "yahoo"
        / f"yahoo_{symbol}_2005-01-03_2025-12-31.yfinance.json"
    )


def _prepared_paths(output_root: Path, symbol: str) -> tuple[Path, Path, Path]:
    """Return provider-specific primary CSV, provenance, and quality paths."""

    directory = output_root / "prepared" / "approved-etf-study" / "yahoo" / symbol
    stem = f"{symbol.lower()}_daily_split_adjusted"
    return (
        directory / f"{stem}.csv",
        directory / f"{stem}.metadata.toml",
        directory / f"{symbol.lower()}_data_quality.txt",
    )


def _fmp_reference_path(output_root: Path) -> Path:
    """Return the existing FMP SPY validation-reference CSV path."""

    return (
        output_root
        / "prepared"
        / "approved-etf-study"
        / "SPY"
        / "spy_daily_split_adjusted.csv"
    )


def _fmp_reference_metadata_path(output_root: Path) -> Path:
    """Return the existing FMP metadata path without modifying it."""

    return _fmp_reference_path(output_root).with_suffix(".metadata.toml")


def _fmp_reference_descriptor_path(output_root: Path) -> Path:
    """Return a separate immutable classification for the retained FMP reference."""

    return (
        output_root
        / "reference"
        / "approved-etf-study"
        / "fmp-spy-validation-reference.toml"
    )


def _provenance_toml(
    *,
    dataset: RawYahooDataset,
    raw_path: Path,
    raw_sha256: str,
    prepared_path: Path,
    prepared_sha256: str,
    rows: Sequence[PreparedObservation],
    diagnostics: CorporateActionDiagnostics,
    repeatability: RepeatabilityDiagnostic,
    fmp_comparison: FmpReferenceComparison | None,
) -> str:
    """Serialize loader-compatible and machine-readable Yahoo provenance."""

    first, last = rows[0], rows[-1]
    values = [
        "[provenance]",
        (
            "dataset_id = "
            + _toml_string(
                f"approved-etf-study-yahoo-{dataset.symbol.lower()}-daily-split-adjusted"
            )
        ),
        f"source_name = {_toml_string(YAHOO_SOURCE_NAME)}",
        'source_url = "https://finance.yahoo.com/"',
        f"downloaded_at = {_toml_string(dataset.retrieved_at.isoformat())}",
        'price_adjustment = "split-adjusted"',
        'dividend_treatment = "excluded"',
        'split_treatment = "adjusted"',
        'bar_frequency = "daily"',
        'exchange_timezone = "America/New_York"',
        'currency = "USD"',
        (
            'volume_validation_status = '
            '"internally_valid_with_cross_provider_warning"'
        ),
        "volume_approved_for_strategy_signals = false",
        f"notes = {_toml_string(_PERSONAL_RESEARCH_LIMITATION + ' ' + _PRICE_NOTES)}",
        "",
        "[acquisition]",
        f"source_name = {_toml_string(YAHOO_SOURCE_NAME)}",
        f"source_type = {_toml_string(YAHOO_SOURCE_TYPE)}",
        f"usage_scope = {_toml_string(YAHOO_USAGE_SCOPE)}",
        f'yfinance_version = "{YFINANCE_VERSION}"',
        f'symbol = "{dataset.symbol}"',
        f'requested_start = "{APPROVED_START}"',
        f'requested_end_exclusive = "{YFINANCE_REQUEST_END}"',
        f'returned_start = "{first.trading_date}"',
        f'returned_end = "{last.trading_date}"',
        'interval = "1d"',
        "auto_adjust = false",
        "actions = true",
        "repair = false",
        "keepna = true",
        "threads = false",
        "progress = false",
        "sequential_acquisition = true",
        f"raw_filename = {_toml_string(raw_path.name)}",
        f'raw_sha256 = "{raw_sha256}"',
        f'canonical_data_sha256 = "{dataset.canonical_data_sha256}"',
        f"raw_columns = {_toml_array(dataset.columns)}",
        f"raw_dtypes = {_toml_array(dataset.dtypes)}",
        f"raw_index_dtype = {_toml_string(dataset.index_dtype)}",
        (
            "raw_index_timezone = "
            + (
                '""'
                if dataset.index_timezone is None
                else _toml_string(dataset.index_timezone)
            )
        ),
        (
            'raw_preservation_format = "deterministic JSON with ordered columns, '
            'dtype metadata, exact index values, and hexadecimal IEEE-754 floats"'
        ),
        "",
        "[field_semantics]",
        'price_interpretation = "split-adjusted, dividend-excluded OHLC"',
        (
            'adjusted_close_interpretation = "split-and-dividend-adjusted; '
            'excluded from execution"'
        ),
        "additional_split_adjustment = false",
        'dividend_treatment = "excluded from execution prices"',
        'actions_treatment = "diagnostics and provenance only"',
        (
            "column_mapping = "
            + _toml_array(
                (
                    "Open -> open",
                    "High -> high",
                    "Low -> low",
                    "Close -> close",
                    "Volume -> volume",
                )
            )
        ),
        "",
        "[prepared_dataset]",
        f"prepared_filename = {_toml_string(prepared_path.name)}",
        f'prepared_sha256 = "{prepared_sha256}"',
        f"row_count = {len(rows)}",
        f"first_timestamp = {_toml_string(first.timestamp.isoformat())}",
        f"last_timestamp = {_toml_string(last.timestamp.isoformat())}",
        f"prepared_columns = {_toml_array(_PREPARED_COLUMNS)}",
        'timestamp_conversion = "16:00 America/New_York converted to UTC"',
        f'transformation_version = "{TRANSFORMATION_VERSION}"',
        "",
        "[quality]",
        'quality_gate_result = "pass"',
        f"split_event_count = {len(diagnostics.split_events)}",
        f"dividend_event_count = {len(diagnostics.dividend_events)}",
        (
            "largest_positive_close_return = "
            + _toml_optional_decimal(diagnostics.largest_positive_return)
        ),
        (
            "largest_negative_close_return = "
            + _toml_optional_decimal(diagnostics.largest_negative_return)
        ),
        f"repeatability_start = {_toml_string(repeatability.start.isoformat())}",
        (
            "repeatability_end_exclusive = "
            + _toml_string(repeatability.end_exclusive.isoformat())
        ),
        f"repeatability_row_count = {repeatability.row_count}",
        f'repeatability_sha256 = "{repeatability.canonical_sha256}"',
        (
            'repeatability_basis = "Date, Open, High, Low, Close, Volume, '
            'Dividends, and Stock Splits; Adj Close excluded because it is not '
            'approved for execution or returns"'
        ),
        f"personal_research_limitation = {_toml_string(_PERSONAL_RESEARCH_LIMITATION)}",
        "",
    ]
    if fmp_comparison is not None:
        values.extend(
            (
                "[cross_provider_reference]",
                'source_name = "Financial Modeling Prep"',
                "validation_reference_only = true",
                "excluded_from_primary_research_universe = true",
                f"common_date_count = {fmp_comparison.common_date_count}",
                f"yahoo_only_date_count = {fmp_comparison.yahoo_only_date_count}",
                f"fmp_only_date_count = {fmp_comparison.fmp_only_date_count}",
                (
                    "maximum_absolute_close_difference = "
                    + _toml_string(
                        _decimal_text(
                            fmp_comparison.maximum_absolute_close_difference
                        )
                    )
                ),
                (
                    "maximum_relative_close_difference = "
                    + _toml_string(
                        _decimal_text(
                            fmp_comparison.maximum_relative_close_difference
                        )
                    )
                ),
                (
                    "median_relative_close_difference = "
                    + _toml_string(
                        _decimal_text(
                            fmp_comparison.median_relative_close_difference
                        )
                    )
                ),
                (
                    "price_differences_above_0_001_percent = "
                    f"{fmp_comparison.price_differences_above_threshold}"
                ),
                f'volume_warning_date = "{fmp_comparison.volume_difference_date}"',
                f"yahoo_volume = {fmp_comparison.yahoo_volume}",
                f"fmp_volume = {fmp_comparison.fmp_volume}",
                (
                    "volume_absolute_difference = "
                    f"{fmp_comparison.volume_absolute_difference}"
                ),
                (
                    "volume_relative_difference = "
                    + _toml_string(
                        _decimal_text(fmp_comparison.volume_relative_difference)
                    )
                ),
                (
                    'volume_warning_status = "unresolved cross-provider volume '
                    'disagreement; no impact on approved price-only strategies"'
                ),
                "",
            )
        )
    return "\n".join(values)


def _symbol_quality_report(
    *,
    dataset: RawYahooDataset,
    raw_sha256: str,
    prepared_sha256: str,
    rows: Sequence[PreparedObservation],
    diagnostics: CorporateActionDiagnostics,
    repeatability: RepeatabilityDiagnostic,
    quality: DatasetQualitySummary,
    fmp_comparison: FmpReferenceComparison | None,
) -> str:
    """Return a readable per-symbol quality report."""

    lines = [
        f"APPROVED YAHOO ETF DATA QUALITY: {dataset.symbol}",
        "=" * (33 + len(dataset.symbol)),
        "Status: PASS",
        "Approval: PERSONAL PRICE-BASED RESEARCH ONLY",
        f"Requested period: {APPROVED_START} through {APPROVED_END} inclusive",
        f"Provider request end: {YFINANCE_REQUEST_END} exclusive",
        f"Returned period: {rows[0].trading_date} through {rows[-1].trading_date}",
        f"Rows: {len(rows)}",
        f"yfinance version: {YFINANCE_VERSION}",
        f"Raw SHA-256: {raw_sha256}",
        f"Canonical data SHA-256: {dataset.canonical_data_sha256}",
        f"Prepared SHA-256: {prepared_sha256}",
        "Required columns present: true",
        "Duplicate dates: 0",
        "Non-chronological dates: 0",
        "Null OHLCV rows: 0",
        "Invalid OHLC rows: 0",
        "Non-positive OHLC rows: 0",
        "Zero or negative volume rows: 0",
        f"CSV-provider rows: {quality.bar_count}",
        f"Split events: {len(diagnostics.split_events)}",
        f"Dividend events: {len(diagnostics.dividend_events)}",
        (
            "Largest positive close return: "
            f"{_optional_decimal(diagnostics.largest_positive_return)}"
        ),
        (
            "Largest negative close return: "
            f"{_optional_decimal(diagnostics.largest_negative_return)}"
        ),
        f"Suspicious close returns (absolute >= 0.40): "
        f"{len(diagnostics.suspicious_returns)}",
        "Additional split adjustment: false",
        "Execution field: Close",
        "Adj Close used for execution: false",
        "Dividends used in execution prices: false",
        (
            "Volume validation status: "
            "internally_valid_with_cross_provider_warning"
        ),
        "Volume approved for strategy signals: false",
        (
            "Repeatability control: "
            f"{repeatability.start} through {repeatability.end_exclusive} exclusive, "
            f"rows={repeatability.row_count}, sha256={repeatability.canonical_sha256}"
        ),
        f"Provider messages retained: {len(dataset.provider_messages)}",
        "",
        "Research limitation:",
        f"  {_PERSONAL_RESEARCH_LIMITATION}",
        "",
    ]
    if fmp_comparison is not None:
        lines.extend(
            (
                "FMP SPY validation-reference comparison:",
                "  validation_reference_only: true",
                "  excluded_from_primary_research_universe: true",
                f"  common dates: {fmp_comparison.common_date_count}",
                f"  Yahoo-only dates: {fmp_comparison.yahoo_only_date_count}",
                f"  FMP-only dates: {fmp_comparison.fmp_only_date_count}",
                (
                    "  maximum absolute close difference: "
                    + _decimal_text(
                        fmp_comparison.maximum_absolute_close_difference
                    )
                ),
                (
                    "  maximum relative close difference: "
                    + _decimal_text(
                        fmp_comparison.maximum_relative_close_difference
                    )
                ),
                (
                    f"  {_KNOWN_VOLUME_WARNING_DATE} Yahoo volume: "
                    f"{fmp_comparison.yahoo_volume}"
                ),
                (
                    f"  {_KNOWN_VOLUME_WARNING_DATE} FMP volume: "
                    f"{fmp_comparison.fmp_volume}"
                ),
                (
                    "  unresolved volume difference: "
                    f"{fmp_comparison.volume_absolute_difference} shares "
                    f"({_decimal_text(fmp_comparison.volume_relative_difference)})"
                ),
                (
                    "  impact: none for current SMA and Donchian strategies because "
                    "they do not consume volume"
                ),
                "",
            )
        )
    lines.extend((_PRICE_NOTES, ""))
    return "\n".join(lines)


def _toml_string(value: str) -> str:
    """Return a TOML-compatible basic string."""

    return json.dumps(value, ensure_ascii=False)


def _toml_array(values: Sequence[str]) -> str:
    """Return one deterministic TOML string array."""

    return "[" + ", ".join(_toml_string(value) for value in values) + "]"


def _toml_optional_decimal(value: Decimal | None) -> str:
    """Return an optional Decimal as TOML string or null-like explanatory text."""

    return _toml_string("not-observed" if value is None else _decimal_text(value))


def _optional_decimal(value: Decimal | None) -> str:
    """Return readable optional Decimal text."""

    return "not-observed" if value is None else _decimal_text(value)


def _fmp_reference_descriptor(
    output_root: Path,
    comparison: FmpReferenceComparison,
) -> bytes:
    """Classify existing FMP artifacts without altering or replacing them."""

    csv_path = _fmp_reference_path(output_root)
    metadata_path = _fmp_reference_metadata_path(output_root)
    csv_sha256 = _sha256_bytes(_read_bytes(csv_path, "FMP SPY reference"))
    metadata_sha256 = _sha256_bytes(
        _read_bytes(metadata_path, "FMP SPY reference metadata")
    )
    values = (
        "[reference]",
        'source_name = "Financial Modeling Prep"',
        'symbol = "SPY"',
        "validation_reference_only = true",
        "excluded_from_primary_research_universe = true",
        f"dataset_filename = {_toml_string(csv_path.name)}",
        f'metadata_filename = "{metadata_path.name}"',
        f'dataset_sha256 = "{csv_sha256}"',
        f'metadata_sha256 = "{metadata_sha256}"',
        f"common_date_count = {comparison.common_date_count}",
        (
            "maximum_relative_close_difference = "
            + _toml_string(
                _decimal_text(comparison.maximum_relative_close_difference)
            )
        ),
        f'known_volume_warning_date = "{comparison.volume_difference_date}"',
        f"known_volume_absolute_difference = {comparison.volume_absolute_difference}",
        (
            "known_volume_relative_difference = "
            + _toml_string(_decimal_text(comparison.volume_relative_difference))
        ),
        'primary_provider = "Yahoo Finance via yfinance"',
        "",
    )
    return "\n".join(values).encode("utf-8")


def _prepare_symbol(
    dataset: RawYahooDataset,
    *,
    raw_path: Path,
    prepared_path: Path,
    provenance_path: Path,
    quality_path: Path,
    repeatability: RepeatabilityDiagnostic,
    output_root: Path,
    overwrite: bool,
    quality_gate: QualityGateSettings,
) -> SymbolDatasetRecord:
    """Validate and publish one Yahoo prepared-data bundle."""

    output_paths = [prepared_path, provenance_path, quality_path]
    if dataset.symbol == "SPY":
        output_paths.append(_fmp_reference_descriptor_path(output_root))
    _preflight_new_files(output_paths, overwrite)
    diagnostics = _validate_quality_gate(dataset, quality_gate)
    rows = transform_yahoo_rows(dataset.rows, dataset.symbol)
    prepared_bytes = prepared_csv_bytes(rows)
    raw_sha256 = _sha256_bytes(dataset.raw_bytes)
    if _sha256_bytes(_read_bytes(raw_path, "raw Yahoo file")) != raw_sha256:
        raise ApprovedDatasetError("raw Yahoo file changed during preparation")
    fmp_comparison = None
    if dataset.symbol == "SPY":
        fmp_comparison = compare_yahoo_spy_with_fmp(
            rows,
            _fmp_reference_path(output_root),
        )
    quality = _validate_prepared_csv(
        dataset.symbol,
        rows,
        prepared_path,
        prepared_bytes,
        overwrite=overwrite,
    )
    prepared_sha256 = _sha256_bytes(prepared_bytes)
    provenance = _provenance_toml(
        dataset=dataset,
        raw_path=raw_path,
        raw_sha256=raw_sha256,
        prepared_path=prepared_path,
        prepared_sha256=prepared_sha256,
        rows=rows,
        diagnostics=diagnostics,
        repeatability=repeatability,
        fmp_comparison=fmp_comparison,
    )
    quality_report = _symbol_quality_report(
        dataset=dataset,
        raw_sha256=raw_sha256,
        prepared_sha256=prepared_sha256,
        rows=rows,
        diagnostics=diagnostics,
        repeatability=repeatability,
        quality=quality,
        fmp_comparison=fmp_comparison,
    )
    _atomic_write(
        provenance_path,
        provenance.encode("utf-8"),
        overwrite=overwrite,
    )
    _atomic_write(
        quality_path,
        quality_report.encode("utf-8"),
        overwrite=overwrite,
    )
    if fmp_comparison is not None:
        _atomic_write(
            _fmp_reference_descriptor_path(output_root),
            _fmp_reference_descriptor(output_root, fmp_comparison),
            overwrite=overwrite,
        )
    notes = (
        (
            "SPY 2025-09-30 unresolved Yahoo/FMP volume disagreement; "
            "no effect on current price-only strategies."
        )
        if dataset.symbol == "SPY"
        else "Volume is internally valid but not approved for strategy signals."
    )
    return SymbolDatasetRecord(
        symbol=dataset.symbol,
        status=DatasetRecordStatus.SUCCESS,
        raw_path=raw_path,
        prepared_path=prepared_path,
        provenance_path=provenance_path,
        quality_report_path=quality_path,
        raw_sha256=raw_sha256,
        canonical_data_sha256=dataset.canonical_data_sha256,
        prepared_sha256=prepared_sha256,
        row_count=len(rows),
        first_date=rows[0].trading_date,
        last_date=rows[-1].trading_date,
        diagnostics=diagnostics,
        repeatability=repeatability,
        fmp_comparison=fmp_comparison,
        quality_summary=quality,
        quality_gate_passed=True,
        review_notes=(notes,),
    )


def run_approved_etf_study(
    config: ApprovedEtfStudyConfig,
    *,
    downloader: Downloader = _default_downloader,
    clock: Clock = _utc_now,
) -> ApprovedEtfStudyResult:
    """Acquire and prepare requested symbols sequentially without evaluations."""

    if not isinstance(config, ApprovedEtfStudyConfig):
        raise TypeError("config must be ApprovedEtfStudyConfig")
    summary_text = config.quality_report_directory / "data-quality-summary.txt"
    summary_csv = config.quality_report_directory / "data-quality-summary.csv"
    if config.mode is DatasetRunMode.FULL:
        preflight_paths = [summary_text, summary_csv]
        if config.symbols == APPROVED_ETF_SYMBOLS:
            preflight_paths.append(config.experiment_manifest)
        _preflight_new_files(preflight_paths, config.overwrite)
    records: list[SymbolDatasetRecord] = []
    provider_diagnostic: YahooSpyDiagnostic | None = None
    for symbol in config.symbols:
        raw_path = _raw_path(config.output_root, symbol)
        prepared_path, provenance_path, quality_path = _prepared_paths(
            config.output_root,
            symbol,
        )
        try:
            if config.mode is DatasetRunMode.PREPARE_ONLY:
                dataset = parse_raw_yfinance_bytes(
                    _read_bytes(raw_path, "raw Yahoo file")
                )
                if dataset.repeatability is None:
                    raise ApprovedDatasetError(
                        "raw Yahoo file lacks repeatability-control metadata"
                    )
                repeatability = dataset.repeatability
            else:
                destination_paths = [raw_path]
                if config.mode is DatasetRunMode.FULL:
                    destination_paths.extend(
                        (prepared_path, provenance_path, quality_path)
                    )
                    if symbol == "SPY":
                        destination_paths.append(
                            _fmp_reference_descriptor_path(config.output_root)
                        )
                _preflight_new_files(destination_paths, config.overwrite)
                repeatability = _repeatability_control(
                    symbol,
                    downloader=downloader,
                    timeout_seconds=config.timeout_seconds,
                    clock=clock,
                )
                frame, messages = _download_frame(
                    symbol,
                    APPROVED_START,
                    YFINANCE_REQUEST_END,
                    downloader=downloader,
                    timeout_seconds=config.timeout_seconds,
                )
                dataset = snapshot_yfinance_frame(
                    frame,
                    symbol,
                    retrieved_at=_aware_clock_value(clock),
                    provider_messages=messages,
                    repeatability=repeatability,
                )
                _validate_quality_gate(dataset, config.quality_gate)
                _atomic_write(
                    raw_path,
                    dataset.raw_bytes,
                    overwrite=config.overwrite,
                )
                if _sha256_bytes(_read_bytes(raw_path, "raw Yahoo file")) != _sha256_bytes(
                    dataset.raw_bytes
                ):
                    raise ApprovedDatasetError("published raw Yahoo hash mismatch")
            if dataset.symbol != symbol:
                raise ApprovedDatasetError("raw Yahoo symbol does not match requested symbol")
            diagnostics = _validate_quality_gate(dataset, config.quality_gate)
            if config.mode is DatasetRunMode.DOWNLOAD_ONLY:
                record = SymbolDatasetRecord(
                    symbol=symbol,
                    status=DatasetRecordStatus.SUCCESS,
                    raw_path=raw_path,
                    raw_sha256=_sha256_bytes(dataset.raw_bytes),
                    canonical_data_sha256=dataset.canonical_data_sha256,
                    row_count=len(dataset.rows),
                    first_date=dataset.rows[0].trading_date,
                    last_date=dataset.rows[-1].trading_date,
                    diagnostics=diagnostics,
                    repeatability=repeatability,
                    quality_gate_passed=True,
                    review_notes=("Download-only mode; preparation was not run.",),
                )
            else:
                record = _prepare_symbol(
                    dataset,
                    raw_path=raw_path,
                    prepared_path=prepared_path,
                    provenance_path=provenance_path,
                    quality_path=quality_path,
                    repeatability=repeatability,
                    output_root=config.output_root,
                    overwrite=config.overwrite,
                    quality_gate=config.quality_gate,
                )
            records.append(record)
            if symbol == "SPY":
                assert record.diagnostics is not None
                assert record.repeatability is not None
                provider_diagnostic = YahooSpyDiagnostic(
                    valid_data=True,
                    earliest_date=record.first_date or APPROVED_START,
                    latest_date=record.last_date or APPROVED_END,
                    required_fields_present=True,
                    row_count=record.row_count or 0,
                    split_event_count=len(record.diagnostics.split_events),
                    dividend_event_count=len(record.diagnostics.dividend_events),
                    repeatability_sha256=record.repeatability.canonical_sha256,
                    quality_gate_passed=record.quality_gate_passed,
                )
        except ApprovedDatasetProviderError:
            raise
        except (ApprovedDatasetError, OSError, UnicodeError, ValueError) as exc:
            records.append(
                SymbolDatasetRecord(
                    symbol=symbol,
                    status=DatasetRecordStatus.FAILURE,
                    raw_path=raw_path,
                    failure_message=redact_secret(str(exc)),
                )
            )

    if config.mode is DatasetRunMode.DOWNLOAD_ONLY:
        return ApprovedEtfStudyResult(
            mode=config.mode,
            records=tuple(records),
            provider_diagnostic=provider_diagnostic,
        )

    _write_universe_quality_summaries(
        records,
        summary_text,
        summary_csv,
        overwrite=config.overwrite,
    )
    manifest_path: Path | None = None
    if _complete_prepared_universe(config.output_root):
        manifest_bytes = approved_experiment_manifest_bytes(
            config.output_root,
            config.experiment_manifest,
        )
        _atomic_write(
            config.experiment_manifest,
            manifest_bytes,
            overwrite=config.overwrite,
        )
        manifest = load_split_evaluation_manifest(config.experiment_manifest)
        if len(manifest.entries) != 32 or any(
            entry.definition is None for entry in manifest.entries
        ):
            raise ApprovedDatasetError(
                "generated experiment manifest did not validate as 32 evaluations"
            )
        manifest_path = config.experiment_manifest
    return ApprovedEtfStudyResult(
        mode=config.mode,
        records=tuple(records),
        quality_summary_text=summary_text,
        quality_summary_csv=summary_csv,
        experiment_manifest=manifest_path,
        provider_diagnostic=provider_diagnostic,
    )


def _aware_clock_value(clock: Clock) -> datetime:
    """Require acquisition clocks to carry an explicit UTC offset."""

    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ApprovedDatasetError("dataset acquisition clock must be timezone-aware")
    return value.astimezone(UTC)


def _complete_prepared_universe(output_root: Path) -> bool:
    """Return whether all eight Yahoo CSVs and provenance sidecars exist."""

    return all(
        all(path.is_file() for path in _prepared_paths(output_root, symbol)[:2])
        for symbol in APPROVED_ETF_SYMBOLS
    )


_UNIVERSE_SUMMARY_COLUMNS = (
    "symbol",
    "source_name",
    "status",
    "row_count",
    "first_date",
    "last_date",
    "yfinance_version",
    "raw_sha256",
    "canonical_data_sha256",
    "prepared_sha256",
    "split_event_count",
    "dividend_event_count",
    "zero_volume_count",
    "largest_positive_return",
    "largest_negative_return",
    "quality_gate_passed",
    "volume_validation_status",
    "volume_approved_for_strategy_signals",
    "cross_provider_warning",
    "review_notes",
)


def _write_universe_quality_summaries(
    records: Sequence[SymbolDatasetRecord],
    text_path: Path,
    csv_path: Path,
    *,
    overwrite: bool,
) -> None:
    """Write ordered machine-readable and readable universe summaries."""

    _preflight_new_files((text_path, csv_path), overwrite)
    merged = _existing_summary_rows(csv_path) if overwrite else {}
    for record in records:
        merged[record.symbol] = _record_summary_row(record)
    rows = tuple(
        merged[symbol]
        for symbol in APPROVED_ETF_SYMBOLS
        if symbol in merged
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=_UNIVERSE_SUMMARY_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    text_lines = [
        "YAHOO APPROVED ETF STUDY DATA-QUALITY SUMMARY",
        "=============================================",
        "Approval: PERSONAL PRICE-BASED RESEARCH ONLY",
        f"Symbols recorded: {len(rows)}",
        "Volume approved for strategy signals: false",
        "",
    ]
    for row in rows:
        text_lines.extend(
            (
                f"{row['symbol']}: {row['status'].upper()}",
                f"  rows: {row['row_count'] or 'not-available'}",
                (
                    f"  period: {row['first_date'] or 'not-available'} to "
                    f"{row['last_date'] or 'not-available'}"
                ),
                f"  raw sha256: {row['raw_sha256'] or 'not-available'}",
                f"  prepared sha256: {row['prepared_sha256'] or 'not-available'}",
                f"  quality gate: {row['quality_gate_passed']}",
                f"  warning: {row['cross_provider_warning'] or 'none'}",
                "",
            )
        )
    text_lines.extend(
        (
            "Known cross-provider warning:",
            "  Date: 2025-09-30",
            "  Yahoo versus FMP volume difference: 7,863,106 shares",
            "  Relative difference: approximately 8.3516%",
            "  Status: unresolved cross-provider volume disagreement",
            (
                "  Impact: none because current SMA and Donchian strategies "
                "do not use volume"
            ),
            "",
            _PERSONAL_RESEARCH_LIMITATION,
            "",
        )
    )
    _atomic_write(csv_path, stream.getvalue().encode("utf-8"), overwrite=overwrite)
    _atomic_write(
        text_path,
        "\n".join(text_lines).encode("utf-8"),
        overwrite=overwrite,
    )


def _existing_summary_rows(path: Path) -> dict[str, dict[str, str]]:
    """Load validated Yahoo rows before an explicitly authorised subset update."""

    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if tuple(reader.fieldnames or ()) != _UNIVERSE_SUMMARY_COLUMNS:
                raise ApprovedDatasetError(
                    "existing Yahoo quality summary has an incompatible schema"
                )
            result: dict[str, dict[str, str]] = {}
            for row in reader:
                symbol = row.get("symbol")
                if (
                    symbol not in APPROVED_ETF_SYMBOLS
                    or row.get("source_name") != YAHOO_SOURCE_NAME
                ):
                    raise ApprovedDatasetError(
                        "existing quality summary is not a Yahoo approved-study result"
                    )
                result[symbol] = {
                    column: row.get(column, "")
                    for column in _UNIVERSE_SUMMARY_COLUMNS
                }
            return result
    except ApprovedDatasetError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ApprovedDatasetError(
            "could not read existing Yahoo quality summary"
        ) from exc


def _record_summary_row(record: SymbolDatasetRecord) -> dict[str, str]:
    """Convert one symbol result into the stable universe-summary schema."""

    diagnostics = record.diagnostics
    quality = record.quality_summary
    warning = (
        "2025-09-30 unresolved Yahoo/FMP volume disagreement: "
        "7,863,106 shares; approximately 8.3516%"
        if record.symbol == "SPY" and record.status is DatasetRecordStatus.SUCCESS
        else ""
    )
    notes = list(record.review_notes)
    if record.failure_message:
        notes.append(record.failure_message)
    return {
        "symbol": record.symbol,
        "source_name": YAHOO_SOURCE_NAME,
        "status": record.status.value,
        "row_count": "" if record.row_count is None else str(record.row_count),
        "first_date": "" if record.first_date is None else record.first_date.isoformat(),
        "last_date": "" if record.last_date is None else record.last_date.isoformat(),
        "yfinance_version": YFINANCE_VERSION,
        "raw_sha256": record.raw_sha256 or "",
        "canonical_data_sha256": record.canonical_data_sha256 or "",
        "prepared_sha256": record.prepared_sha256 or "",
        "split_event_count": (
            "" if diagnostics is None else str(len(diagnostics.split_events))
        ),
        "dividend_event_count": (
            "" if diagnostics is None else str(len(diagnostics.dividend_events))
        ),
        "zero_volume_count": "" if quality is None else str(quality.zero_volume_count),
        "largest_positive_return": (
            ""
            if diagnostics is None or diagnostics.largest_positive_return is None
            else _decimal_text(diagnostics.largest_positive_return)
        ),
        "largest_negative_return": (
            ""
            if diagnostics is None or diagnostics.largest_negative_return is None
            else _decimal_text(diagnostics.largest_negative_return)
        ),
        "quality_gate_passed": str(record.quality_gate_passed).lower(),
        (
            "volume_validation_status"
        ): "internally_valid_with_cross_provider_warning",
        "volume_approved_for_strategy_signals": "false",
        "cross_provider_warning": warning,
        "review_notes": " ".join(notes),
    }


def approved_experiment_manifest_bytes(
    output_root: Path,
    manifest_path: Path,
) -> bytes:
    """Build exactly 32 ordered evaluations using Yahoo primary CSVs only."""

    output_root = Path(output_root)
    manifest_path = Path(manifest_path)
    lines = ['schema_version = "1.0"', ""]
    configurations = (
        ("sma-20-50", "sma-crossover", ("fast_window", 20), ("slow_window", 50)),
        ("sma-50-200", "sma-crossover", ("fast_window", 50), ("slow_window", 200)),
        (
            "donchian-20-10",
            "donchian-breakout",
            ("entry_window", 20),
            ("exit_window", 10),
        ),
        (
            "donchian-50-20",
            "donchian-breakout",
            ("entry_window", 50),
            ("exit_window", 20),
        ),
    )
    identifiers: set[str] = set()
    for symbol in APPROVED_ETF_SYMBOLS:
        prepared_path = _prepared_paths(output_root, symbol)[0]
        try:
            dataset_path = Path(
                os.path.relpath(prepared_path, start=manifest_path.parent)
            ).as_posix()
        except ValueError as exc:
            raise ApprovedDatasetError(
                "experiment dataset paths must be portable relative paths"
            ) from exc
        if "/yahoo/" not in f"/{dataset_path.lower()}/":
            raise ApprovedDatasetError(
                "approved experiment manifest must use Yahoo primary datasets"
            )
        for suffix, strategy_name, first_parameter, second_parameter in configurations:
            evaluation_id = f"{symbol.lower()}-{suffix}"
            if evaluation_id in identifiers:
                raise ApprovedDatasetError(
                    f"duplicate generated evaluation ID: {evaluation_id}"
                )
            identifiers.add(evaluation_id)
            lines.extend(
                (
                    "[[split_evaluations]]",
                    f'id = "{evaluation_id}"',
                    f"dataset = {_toml_string(dataset_path)}",
                    f'symbol = "{symbol}"',
                    'starting_cash = "100000"',
                    'commission_bps = "1"',
                    'slippage_bps = "5"',
                    'end_of_test = "liquidate"',
                    'benchmark = "buy-and-hold"',
                    "",
                    "[split_evaluations.split]",
                    'development_start = "2005-01-03T00:00:00+00:00"',
                    'development_end = "2018-12-31T23:59:59+00:00"',
                    'holdout_start = "2019-01-02T00:00:00+00:00"',
                    'holdout_end = "2025-12-31T23:59:59+00:00"',
                    'warmup_policy = "carry-history"',
                    "",
                    "[split_evaluations.strategy]",
                    f'name = "{strategy_name}"',
                    f"{first_parameter[0]} = {first_parameter[1]}",
                    f"{second_parameter[0]} = {second_parameter[1]}",
                    "",
                    "[split_evaluations.position_sizing]",
                    'mode = "cash-allocation"',
                    'cash_allocation_ratio = "1.0"',
                    "",
                    "[split_evaluations.risk_metrics]",
                    'periods_per_year = "252"',
                    'risk_free_rate_per_period = "0"',
                    'target_return_per_period = "0"',
                    "",
                )
            )
    if len(identifiers) != 32:
        raise ApprovedDatasetError("approved experiment manifest must contain 32 IDs")
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")
