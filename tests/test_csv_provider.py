"""Unit tests for the validated local CSV market-data provider."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from trading_research.data import CsvMarketDataProvider
from trading_research.errors import (
    MarketDataError,
    MarketDataSchemaError,
    MarketDataValidationError,
)
from trading_research.models import MarketBar

HEADER = "timestamp,symbol,open,high,low,close,volume"
BASE_TIMESTAMP = datetime(2025, 1, 1, tzinfo=UTC)


def csv_row(**overrides: str) -> str:
    """Create one valid CSV row with selected string overrides."""

    values = {
        "timestamp": "2025-01-01T00:00:00+00:00",
        "symbol": "AAPL",
        "open": "10.00",
        "high": "11.00",
        "low": "9.00",
        "close": "10.50",
        "volume": "100",
    }
    values.update(overrides)
    return ",".join(values[column] for column in HEADER.split(","))


def write_csv(
    tmp_path: Path,
    rows: Sequence[str],
    header: str = HEADER,
) -> Path:
    """Write a temporary UTF-8 CSV fixture."""

    path = tmp_path / "prices.csv"
    path.write_text("\n".join((header, *rows)) + "\n", encoding="utf-8")
    return path


def load(path: Path, symbol: str = "AAPL") -> Sequence[MarketBar]:
    """Load all rows through the provider with no timestamp filters."""

    return CsvMarketDataProvider(path).get_historical_bars(symbol, None, None)


def test_loads_one_valid_row_with_exact_types(tmp_path: Path) -> None:
    """A valid observation becomes an immutable, exact domain value."""

    bars = load(write_csv(tmp_path, [csv_row()]))

    assert isinstance(bars, tuple)
    assert len(bars) == 1
    bar = bars[0]
    assert bar.symbol == "AAPL"
    assert bar.timestamp == BASE_TIMESTAMP
    assert bar.close == Decimal("10.50")


def test_loads_multiple_rows_in_source_order(tmp_path: Path) -> None:
    """Chronological source order is preserved rather than silently changed."""

    path = write_csv(
        tmp_path,
        [
            csv_row(),
            csv_row(timestamp="2025-01-02T00:00:00+00:00", close="10.75"),
        ],
    )

    bars = CsvMarketDataProvider(path).get_historical_bars(" aapl ", None, None)

    assert [bar.close for bar in bars] == [Decimal("10.50"), Decimal("10.75")]
    assert [bar.timestamp for bar in bars] == sorted(bar.timestamp for bar in bars)


def test_ignores_additional_named_columns(tmp_path: Path) -> None:
    """Named vendor metadata is ignored while required values remain validated."""

    path = write_csv(
        tmp_path,
        [csv_row() + ",synthetic"],
        header=HEADER + ",source",
    )

    assert len(load(path)) == 1


def test_inclusive_timestamp_filters(tmp_path: Path) -> None:
    """Start and end filters include observations exactly on their boundaries."""

    path = write_csv(
        tmp_path,
        [
            csv_row(),
            csv_row(timestamp="2025-01-02T00:00:00+00:00", close="11.00"),
            csv_row(timestamp="2025-01-03T00:00:00+00:00", close="12.00", high="12.00"),
        ],
    )
    provider = CsvMarketDataProvider(path)
    second = BASE_TIMESTAMP + timedelta(days=1)
    third = BASE_TIMESTAMP + timedelta(days=2)

    assert [bar.close for bar in provider.get_historical_bars("AAPL", second, None)] == [
        Decimal("11.00"),
        Decimal("12.00"),
    ]
    assert [bar.close for bar in provider.get_historical_bars("AAPL", None, second)] == [
        Decimal("10.50"),
        Decimal("11.00"),
    ]
    assert [bar.close for bar in provider.get_historical_bars("AAPL", second, third)] == [
        Decimal("11.00"),
        Decimal("12.00"),
    ]


def test_valid_filter_with_no_matches_returns_empty_tuple(tmp_path: Path) -> None:
    """No temporal matches is a valid empty result, not malformed source data."""

    provider = CsvMarketDataProvider(write_csv(tmp_path, [csv_row()]))

    bars = provider.get_historical_bars(
        "AAPL",
        BASE_TIMESTAMP + timedelta(days=1),
        None,
    )

    assert bars == ()


@pytest.mark.parametrize("missing_column", HEADER.split(","))
def test_rejects_each_missing_required_column(tmp_path: Path, missing_column: str) -> None:
    """Every schema field is independently required."""

    header = ",".join(column for column in HEADER.split(",") if column != missing_column)
    path = write_csv(tmp_path, [], header=header)

    with pytest.raises(MarketDataSchemaError, match=missing_column):
        load(path)


def test_rejects_empty_file(tmp_path: Path) -> None:
    """A zero-byte file has no usable schema."""

    path = tmp_path / "empty.csv"
    path.touch()

    with pytest.raises(MarketDataSchemaError, match="empty"):
        load(path)


def test_rejects_header_only_file(tmp_path: Path) -> None:
    """A valid header without observations is not historical data."""

    with pytest.raises(MarketDataValidationError, match="no observations"):
        load(write_csv(tmp_path, []))


def test_rejects_duplicate_header_columns(tmp_path: Path) -> None:
    """Duplicate headings cannot ambiguously overwrite row values."""

    path = write_csv(tmp_path, [], header=HEADER + ",close")

    with pytest.raises(MarketDataSchemaError, match="duplicate columns"):
        load(path)


@pytest.mark.parametrize(
    "contents",
    [
        HEADER + '\n"2025-01-01T00:00:00+00:00,AAPL,10,11,9,10,100\n',
        HEADER + "\n" + csv_row() + ",unexpected\n",
        HEADER + "\n2025-01-01T00:00:00+00:00,AAPL,10\n",
    ],
)
def test_rejects_malformed_csv_structure(tmp_path: Path, contents: str) -> None:
    """Broken quoting and inconsistent row widths fail as structured errors."""

    path = tmp_path / "malformed.csv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises((MarketDataSchemaError, MarketDataValidationError)) as exc_info:
        load(path)

    assert str(path) in str(exc_info.value)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("symbol", "", "supported market-symbol"),
        ("timestamp", "not-a-date", "invalid ISO 8601"),
        ("timestamp", "2025-01-01T00:00:00", "timezone-aware"),
        ("open", "not-a-number", "invalid decimal"),
        ("open", "NaN", "finite"),
        ("open", "Infinity", "finite"),
        ("open", "-Infinity", "finite"),
        ("open", "-1", "positive"),
        ("open", "0", "positive"),
        ("volume", "-1", "non-negative"),
        ("volume", "1.5", "invalid integer"),
        ("high", "9", "greatest OHLC"),
    ],
)
def test_rejects_invalid_row_values(
    tmp_path: Path,
    field: str,
    value: str,
    reason: str,
) -> None:
    """Bad values report their file, row, and validation reason."""

    path = write_csv(tmp_path, [csv_row(**{field: value})])

    with pytest.raises(MarketDataValidationError) as exc_info:
        load(path)

    message = str(exc_info.value)
    assert str(path) in message
    assert "row 2" in message
    assert reason in message


def test_rejects_duplicate_symbol_timestamp(tmp_path: Path) -> None:
    """Repeated observation identities are rejected explicitly."""

    path = write_csv(tmp_path, [csv_row(), csv_row(close="10.75")])

    with pytest.raises(MarketDataValidationError, match=r"row 3.*duplicate"):
        load(path)


def test_rejects_unordered_observations(tmp_path: Path) -> None:
    """The provider never silently sorts source observations."""

    path = write_csv(
        tmp_path,
        [
            csv_row(timestamp="2025-01-02T00:00:00+00:00"),
            csv_row(timestamp="2025-01-01T00:00:00+00:00"),
        ],
    )

    with pytest.raises(MarketDataValidationError, match=r"row 3.*chronological"):
        load(path)


def test_rejects_mixed_symbols(tmp_path: Path) -> None:
    """A single-symbol request cannot consume a mixed-symbol file."""

    path = write_csv(
        tmp_path,
        [
            csv_row(),
            csv_row(timestamp="2025-01-02T00:00:00+00:00", symbol="MSFT"),
        ],
    )

    with pytest.raises(MarketDataValidationError, match="mixed symbols"):
        load(path)


def test_rejects_requested_symbol_mismatch(tmp_path: Path) -> None:
    """A valid single-symbol file must match the requested identity."""

    with pytest.raises(MarketDataValidationError, match="not requested symbol"):
        load(write_csv(tmp_path, [csv_row()]), symbol="MSFT")


def test_rejects_reversed_date_range(tmp_path: Path) -> None:
    """Start cannot be later than end."""

    provider = CsvMarketDataProvider(write_csv(tmp_path, [csv_row()]))

    with pytest.raises(MarketDataValidationError, match="after end"):
        provider.get_historical_bars(
            "AAPL",
            BASE_TIMESTAMP + timedelta(days=1),
            BASE_TIMESTAMP,
        )


@pytest.mark.parametrize(
    ("start", "end", "name"),
    [
        (datetime(2025, 1, 1), None, "start"),
        (None, datetime(2025, 1, 1), "end"),
    ],
)
def test_rejects_timezone_naive_filters(
    tmp_path: Path,
    start: datetime | None,
    end: datetime | None,
    name: str,
) -> None:
    """Ambiguous request boundaries fail before the file is read."""

    provider = CsvMarketDataProvider(write_csv(tmp_path, [csv_row()]))

    with pytest.raises(MarketDataValidationError, match=f"{name} filter"):
        provider.get_historical_bars("AAPL", start, end)


def test_rejects_invalid_requested_symbol(tmp_path: Path) -> None:
    """Request identity uses the same symbol rules as domain observations."""

    provider = CsvMarketDataProvider(write_csv(tmp_path, [csv_row()]))

    with pytest.raises(MarketDataValidationError, match="Invalid requested symbol"):
        provider.get_historical_bars("../AAPL", None, None)


def test_missing_file_uses_market_data_error(tmp_path: Path) -> None:
    """Filesystem failures are converted to the project exception hierarchy."""

    path = tmp_path / "missing.csv"

    with pytest.raises(MarketDataError, match="not found") as exc_info:
        load(path)

    assert str(path) in str(exc_info.value)


def test_shipped_synthetic_sample_loads() -> None:
    """The documented sample remains valid and usable."""

    sample_path = Path(__file__).parents[1] / "data" / "sample_prices.csv"

    bars = CsvMarketDataProvider(sample_path).get_historical_bars("DEMO", None, None)

    assert len(bars) == 8
    assert bars[0].close == Decimal("10")
    assert bars[-1].close == Decimal("9")
