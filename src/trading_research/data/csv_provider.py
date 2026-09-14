"""Validated, read-only historical market data loaded from local CSV files."""

import csv
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TextIO

from trading_research.data.providers import MarketDataProvider
from trading_research.errors import (
    MarketDataError,
    MarketDataSchemaError,
    MarketDataValidationError,
)
from trading_research.models import MarketBar, normalize_symbol

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = frozenset(
    {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
)
_PRICE_COLUMNS = ("open", "high", "low", "close")
_MAX_ERROR_VALUE_LENGTH = 80


class CsvMarketDataProvider(MarketDataProvider):
    """Load a single symbol's validated OHLCV history from a local CSV file.

    Extra named columns are ignored. Rows must already be strictly chronological,
    and ``start`` and ``end`` filters are inclusive when supplied. The source file
    is opened in read-only mode and is never modified.
    """

    def __init__(self, path: Path) -> None:
        """Store the local CSV path without reading or modifying it."""

        self._path = Path(path)

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime | None,
        end: datetime | None,
    ) -> Sequence[MarketBar]:
        """Return validated bars for one symbol within inclusive time filters.

        Raises:
            MarketDataError: If the file cannot be read.
            MarketDataSchemaError: If the CSV structure is invalid.
            MarketDataValidationError: If a row or request filter is invalid.
        """

        requested_symbol = self._validate_requested_symbol(symbol)
        self._validate_filters(start, end)
        bars = self._read_bars()
        self._validate_sequence(bars, requested_symbol)

        filtered = tuple(
            bar
            for bar in bars
            if (start is None or bar.timestamp >= start)
            and (end is None or bar.timestamp <= end)
        )
        logger.info("Loaded %d filtered bars from %s", len(filtered), self._path)
        return filtered

    def _read_bars(self) -> tuple[MarketBar, ...]:
        """Read and validate every observation before applying filters."""

        try:
            with self._path.open(mode="r", encoding="utf-8-sig", newline="") as handle:
                return self._parse_csv(handle)
        except (MarketDataSchemaError, MarketDataValidationError):
            raise
        except FileNotFoundError as exc:
            raise MarketDataError(f"CSV file not found: {self._path}") from exc
        except IsADirectoryError as exc:
            raise MarketDataError(f"CSV path is not a file: {self._path}") from exc
        except PermissionError as exc:
            raise MarketDataError(f"CSV file is not readable: {self._path}") from exc
        except UnicodeError as exc:
            raise MarketDataSchemaError(f"CSV file is not valid UTF-8: {self._path}") from exc
        except OSError as exc:
            raise MarketDataError(f"Could not read CSV file {self._path}: {exc}") from exc

    def _parse_csv(self, handle: TextIO) -> tuple[MarketBar, ...]:
        """Parse a CSV stream and add file and row context to errors."""

        try:
            reader = csv.DictReader(handle, strict=True)
            self._validate_header(reader.fieldnames)
            bars: list[MarketBar] = []
            for row_number, row in enumerate(reader, start=2):
                self._validate_row_shape(row, row_number)
                bars.append(self._parse_row(row, row_number))
        except csv.Error as exc:
            raise MarketDataSchemaError(f"Malformed CSV in {self._path}: {exc}") from exc

        if not bars:
            raise MarketDataValidationError(
                f"CSV file contains a header but no observations: {self._path}"
            )
        return tuple(bars)

    def _validate_header(self, fieldnames: Sequence[str] | None) -> None:
        """Require one non-duplicated header containing every required column."""

        if fieldnames is None:
            raise MarketDataSchemaError(f"CSV file is empty: {self._path}")
        if len(fieldnames) != len(set(fieldnames)):
            raise MarketDataSchemaError(f"CSV header contains duplicate columns: {self._path}")
        missing = sorted(_REQUIRED_COLUMNS.difference(fieldnames))
        if missing:
            raise MarketDataSchemaError(
                f"CSV file {self._path} is missing required columns: {', '.join(missing)}"
            )

    def _validate_row_shape(
        self,
        row: Mapping[str | None, str | list[str] | None],
        row_number: int,
    ) -> None:
        """Reject rows with more or fewer values than the declared header."""

        if None in row:
            raise self._row_error(row_number, None, row[None], "row has too many values")
        missing_values = [column for column in _REQUIRED_COLUMNS if row.get(column) is None]
        if missing_values:
            raise self._row_error(
                row_number,
                missing_values[0],
                None,
                "row has too few values",
            )

    def _parse_row(
        self,
        row: Mapping[str | None, str | list[str] | None],
        row_number: int,
    ) -> MarketBar:
        """Convert one structurally valid dictionary row into a domain model."""

        timestamp = self._parse_timestamp(self._value(row, "timestamp"), row_number)
        prices = {
            column: self._parse_decimal(self._value(row, column), row_number, column)
            for column in _PRICE_COLUMNS
        }
        volume = self._parse_volume(self._value(row, "volume"), row_number)
        symbol = self._value(row, "symbol")

        try:
            return MarketBar(
                timestamp=timestamp,
                symbol=symbol,
                open=prices["open"],
                high=prices["high"],
                low=prices["low"],
                close=prices["close"],
                volume=volume,
            )
        except (TypeError, ValueError) as exc:
            raise self._row_error(row_number, None, None, str(exc)) from exc

    def _parse_timestamp(self, value: str, row_number: int) -> datetime:
        """Parse an ISO 8601 timestamp and require an explicit UTC offset."""

        try:
            timestamp = datetime.fromisoformat(value.strip())
        except ValueError as exc:
            raise self._row_error(
                row_number,
                "timestamp",
                value,
                "invalid ISO 8601 timestamp",
            ) from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise self._row_error(
                row_number,
                "timestamp",
                value,
                "timestamp must be timezone-aware",
            )
        return timestamp

    def _parse_decimal(self, value: str, row_number: int, column: str) -> Decimal:
        """Parse a finite exact decimal value with row and column context."""

        try:
            parsed = Decimal(value.strip())
        except InvalidOperation as exc:
            raise self._row_error(row_number, column, value, "invalid decimal value") from exc
        if not parsed.is_finite():
            raise self._row_error(row_number, column, value, "decimal value must be finite")
        return parsed

    def _parse_volume(self, value: str, row_number: int) -> int:
        """Parse volume as a base-10 integer."""

        try:
            return int(value.strip())
        except ValueError as exc:
            raise self._row_error(row_number, "volume", value, "invalid integer value") from exc

    def _validate_sequence(self, bars: Sequence[MarketBar], requested_symbol: str) -> None:
        """Validate identity, uniqueness, and source ordering without repairing data."""

        symbols = {bar.symbol for bar in bars}
        if len(symbols) != 1:
            raise MarketDataValidationError(
                f"CSV file {self._path} contains mixed symbols: {', '.join(sorted(symbols))}"
            )
        file_symbol = next(iter(symbols))
        if file_symbol != requested_symbol:
            raise MarketDataValidationError(
                f"CSV file {self._path} contains symbol {file_symbol!r}, "
                f"not requested symbol {requested_symbol!r}"
            )

        seen: set[tuple[str, datetime]] = set()
        previous: MarketBar | None = None
        for row_number, bar in enumerate(bars, start=2):
            identity = (bar.symbol, bar.timestamp)
            if identity in seen:
                raise MarketDataValidationError(
                    f"CSV file {self._path}, row {row_number}: duplicate observation "
                    f"for {bar.symbol} at {bar.timestamp.isoformat()}"
                )
            if previous is not None and bar.timestamp <= previous.timestamp:
                raise MarketDataValidationError(
                    f"CSV file {self._path}, row {row_number}: observations must be "
                    "strictly chronological"
                )
            seen.add(identity)
            previous = bar

    def _validate_requested_symbol(self, symbol: str) -> str:
        """Apply the domain symbol rules to the provider request."""

        try:
            return normalize_symbol(symbol)
        except (TypeError, ValueError) as exc:
            raise MarketDataValidationError(f"Invalid requested symbol: {exc}") from exc

    @staticmethod
    def _validate_filters(start: datetime | None, end: datetime | None) -> None:
        """Require aware optional filters and a non-reversed inclusive range."""

        for name, value in (("start", start), ("end", end)):
            if value is not None and not isinstance(value, datetime):
                raise MarketDataValidationError(f"{name} filter must be a datetime or None")
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise MarketDataValidationError(f"{name} filter must be timezone-aware")
        if start is not None and end is not None and start > end:
            raise MarketDataValidationError("start filter must not be after end filter")

    @staticmethod
    def _value(row: Mapping[str | None, str | list[str] | None], column: str) -> str:
        """Return a required scalar value after structural validation."""

        value = row[column]
        assert isinstance(value, str)
        return value

    def _row_error(
        self,
        row_number: int,
        column: str | None,
        value: object,
        reason: str,
    ) -> MarketDataValidationError:
        """Build a bounded error message containing safe, relevant CSV context."""

        location = f"CSV file {self._path}, row {row_number}"
        if column is not None:
            location += f", column {column!r}"
        value_text = repr(value)
        if len(value_text) > _MAX_ERROR_VALUE_LENGTH:
            value_text = value_text[: _MAX_ERROR_VALUE_LENGTH - 3] + "..."
        return MarketDataValidationError(f"{location}, value {value_text}: {reason}")
