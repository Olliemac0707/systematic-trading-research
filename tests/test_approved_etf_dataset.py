"""Offline tests for Yahoo-backed approved ETF data preparation."""

from __future__ import annotations

import csv
import json
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from trading_research.data.approved_etf import (
    APPROVED_END,
    APPROVED_ETF_SYMBOLS,
    APPROVED_START,
    YFINANCE_VERSION,
    ApprovedEtfStudyConfig,
    DatasetRecordStatus,
    DatasetRunMode,
    QualityGateSettings,
    RawVendorObservation,
    _repeatability_control,
    approved_experiment_manifest_bytes,
    compare_yahoo_spy_with_fmp,
    parse_raw_yfinance_bytes,
    prepared_csv_bytes,
    redact_secret,
    run_approved_etf_study,
    snapshot_yfinance_frame,
    transform_yahoo_rows,
    validate_yfinance_version,
)
from trading_research.errors import ApprovedDatasetError, ApprovedDatasetProviderError
from trading_research.evaluation import (
    DatasetProvenance,
    DividendTreatment,
    PriceAdjustmentPolicy,
    SplitTreatment,
    load_dataset_provenance,
    load_split_evaluation_manifest,
)
from trading_research.evaluation.runner import validate_strategy_data_policy
from trading_research.strategies import (
    DonchianBreakoutParameters,
    SmaCrossoverParameters,
    StrategyName,
    create_strategy,
)
from trading_research.strategies.base import Strategy

_COLUMNS = (
    "Adj Close",
    "Close",
    "Dividends",
    "High",
    "Low",
    "Open",
    "Stock Splits",
    "Volume",
)
_RETRIEVED = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)


class _ListWithMetadata(list[object]):
    """A tiny pandas-like columns/index collection for boundary tests."""

    nlevels = 1
    name: str | None = "Date"
    dtype = "datetime64[ns]"
    tz: str | None = None

    def tolist(self) -> list[object]:
        """Return an independent list like pandas Index.tolist."""

        return list(self)


class _Matrix:
    """A tiny pandas-like matrix returned by ``to_numpy``."""

    def __init__(self, rows: Sequence[Sequence[object]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[object]]:
        """Return an independent row matrix."""

        return [list(row) for row in self._rows]


class _Frame:
    """Minimal immutable dataframe surface consumed by the adapter."""

    def __init__(
        self,
        dates: Sequence[date],
        rows: Sequence[Mapping[str, object]],
        *,
        columns: Sequence[str] = _COLUMNS,
    ) -> None:
        self.columns = _ListWithMetadata(columns)
        self.dtypes = _ListWithMetadata(
            "int64" if column == "Volume" else "float64" for column in columns
        )
        self.index = _ListWithMetadata(
            datetime.combine(day, datetime.min.time()) for day in dates
        )
        self._matrix = tuple(
            tuple(row[column] for column in columns) for row in rows
        )

    def to_numpy(self, *, copy: bool) -> _Matrix:
        """Return a copied matrix and require the production option."""

        assert copy is True
        return _Matrix(self._matrix)


def _row(
    *,
    open_price: object = 100.0,
    high: object = 102.0,
    low: object = 99.0,
    close: object = 101.0,
    adjusted_close: object = 98.5,
    volume: object = 1_000,
    dividend: object = 0.0,
    split: object = 0.0,
) -> dict[str, object]:
    """Return one valid Yahoo row with distinct Close and Adj Close."""

    return {
        "Adj Close": adjusted_close,
        "Close": close,
        "Dividends": dividend,
        "High": high,
        "Low": low,
        "Open": open_price,
        "Stock Splits": split,
        "Volume": volume,
    }


def _full_frame() -> _Frame:
    """Return a compact full-period fixture including the known warning date."""

    return _Frame(
        (APPROVED_START, date(2025, 9, 30), APPROVED_END),
        (
            _row(close=100.0, adjusted_close=80.0, volume=1_000),
            _row(
                open_price=119.0,
                high=121.0,
                low=118.0,
                close=120.0,
                adjusted_close=118.0,
                volume=86_288_000,
                dividend=1.2,
            ),
            _row(
                open_price=120.0,
                high=122.0,
                low=119.0,
                close=121.0,
                adjusted_close=119.0,
                volume=2_000,
            ),
        ),
    )


def _control_frame() -> _Frame:
    """Return the deterministic repeatability control response."""

    return _Frame(
        (date(2025, 12, 15), date(2025, 12, 16)),
        (
            _row(),
            _row(open_price=101.0, high=103.0, low=100.0, close=102.0),
        ),
    )


def _small_gate() -> QualityGateSettings:
    """Return exact endpoints with a compact observation minimum."""

    return QualityGateSettings(
        minimum_observations=3,
        expected_first_date=APPROVED_START,
        expected_last_date=APPROVED_END,
    )


def _clock() -> datetime:
    """Return a deterministic aware acquisition timestamp."""

    return _RETRIEVED


def _write_fmp_reference(root: Path) -> Path:
    """Create the retained provider-neutral SPY reference used in tests."""

    path = (
        root
        / "prepared"
        / "approved-etf-study"
        / "SPY"
        / "spy_daily_split_adjusted.csv"
    )
    path.parent.mkdir(parents=True)
    rows = (
        (APPROVED_START, "100", 1_000),
        (date(2025, 9, 30), "120", 94_151_106),
        (APPROVED_END, "121", 2_000),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("timestamp", "symbol", "open", "high", "low", "close", "volume"))
        for trading_date, close, volume in rows:
            timestamp = datetime(
                trading_date.year,
                trading_date.month,
                trading_date.day,
                21,
                tzinfo=UTC,
            )
            writer.writerow(
                (timestamp.isoformat(), "SPY", close, close, close, close, volume)
            )
    path.with_suffix(".metadata.toml").write_text(
        '[provenance]\nsource_name = "Financial Modeling Prep"\n',
        encoding="utf-8",
    )
    return path


@dataclass
class _Downloader:
    """Deterministic offline yfinance boundary with an auditable call log."""

    calls: list[tuple[str, dict[str, object]]]
    failure_message: str | None = None

    def __call__(self, symbol: str, **kwargs: object) -> object:
        """Return control or full frames without making a network request."""

        self.calls.append((symbol, kwargs))
        if self.failure_message is not None:
            raise RuntimeError(self.failure_message)
        return (
            _control_frame()
            if kwargs["start"] == "2025-12-15"
            else _full_frame()
        )


def _config(
    root: Path,
    *,
    symbols: tuple[str, ...] = ("SPY",),
    mode: DatasetRunMode = DatasetRunMode.FULL,
    overwrite: bool = False,
) -> ApprovedEtfStudyConfig:
    """Return compact approved-study paths and quality settings."""

    return ApprovedEtfStudyConfig(
        output_root=root / "data",
        quality_report_directory=root / "reports",
        experiment_manifest=root / "experiments" / "approved-etf-study.toml",
        symbols=symbols,
        mode=mode,
        overwrite=overwrite,
        quality_gate=_small_gate(),
    )


def _raw_observation(
    trading_date: date,
    *,
    close: str = "101",
    adjusted_close: str = "98",
    dividend: str = "0",
    split: str = "0",
) -> RawVendorObservation:
    """Return one exact provider-neutral Yahoo observation."""

    return RawVendorObservation(
        trading_date=trading_date,
        open=Decimal("100"),
        high=Decimal("102"),
        low=Decimal("99"),
        close=Decimal(close),
        adjusted_close=Decimal(adjusted_close),
        volume=1_000,
        dividend_amount=Decimal(dividend),
        split_coefficient=Decimal(split),
    )


def test_pinned_yfinance_boundary_rejects_silent_version_changes() -> None:
    """The exact empirically validated dependency is mandatory."""

    validate_yfinance_version(YFINANCE_VERSION)
    with pytest.raises(ApprovedDatasetProviderError, match="exactly 1.5.1"):
        validate_yfinance_version("1.5.2")


def test_raw_preservation_is_lossless_deterministic_and_keeps_actions() -> None:
    """Raw JSON retains ordering, dtypes, float hex values, actions, and request."""

    frame = _Frame(
        (date(2020, 8, 31),),
        (_row(split=4.0, dividend=0.82),),
    )
    first = snapshot_yfinance_frame(frame, "SPY", retrieved_at=_RETRIEVED)
    second = snapshot_yfinance_frame(frame, "SPY", retrieved_at=_RETRIEVED)
    parsed = parse_raw_yfinance_bytes(first.raw_bytes)
    document = json.loads(first.raw_bytes)

    assert first.raw_bytes == second.raw_bytes
    assert first.canonical_data_sha256 == second.canonical_data_sha256
    assert parsed.columns == _COLUMNS
    assert parsed.dtypes[-1] == "int64"
    assert parsed.rows[0].split_coefficient == Decimal("4.0")
    assert parsed.rows[0].dividend_amount == Decimal("0.82")
    assert document["rows"][0]["values"][0]["hex"].startswith("0x")
    assert document["request"]["end"] == "2026-01-01"
    assert document["request"]["end_semantics"] == "exclusive"
    assert document["yfinance_version"] == YFINANCE_VERSION


def test_required_columns_and_null_values_are_rejected() -> None:
    """Missing actions/Adj Close and null execution fields cannot be published."""

    missing = tuple(column for column in _COLUMNS if column != "Adj Close")
    with pytest.raises(ApprovedDatasetProviderError, match="Adj Close"):
        snapshot_yfinance_frame(
            _Frame((APPROVED_START,), (_row(),), columns=missing),
            "SPY",
            retrieved_at=_RETRIEVED,
        )
    with pytest.raises(ApprovedDatasetError, match="null or non-numeric"):
        snapshot_yfinance_frame(
            _Frame((APPROVED_START,), (_row(close=float("nan")),)),
            "SPY",
            retrieved_at=_RETRIEVED,
        )


@pytest.mark.parametrize(
    "invalid_row",
    [
        _row(open_price=0),
        _row(high=98),
        _row(low=103),
        _row(volume=0),
        _row(volume=-1),
    ],
    ids=("zero-price", "high-too-low", "low-too-high", "zero-volume", "negative-volume"),
)
def test_invalid_ohlc_and_non_positive_volume_are_rejected(
    invalid_row: Mapping[str, object],
) -> None:
    """The personal-study volume warning never permits invalid Yahoo volume."""

    with pytest.raises(ApprovedDatasetError):
        snapshot_yfinance_frame(
            _Frame((APPROVED_START,), (invalid_row,)),
            "SPY",
            retrieved_at=_RETRIEVED,
        )


def test_duplicate_and_non_chronological_dates_are_rejected() -> None:
    """No date is silently deduplicated or reordered."""

    with pytest.raises(ApprovedDatasetError, match="duplicate"):
        snapshot_yfinance_frame(
            _Frame((APPROVED_START, APPROVED_START), (_row(), _row())),
            "SPY",
            retrieved_at=_RETRIEVED,
        )
    with pytest.raises(ApprovedDatasetError, match="chronological"):
        snapshot_yfinance_frame(
            _Frame((APPROVED_END, APPROVED_START), (_row(), _row())),
            "SPY",
            retrieved_at=_RETRIEVED,
        )


def test_yahoo_prices_are_not_split_adjusted_twice_and_close_is_execution() -> None:
    """Prepared OHLCV is an exact identity mapping and never uses Adj Close."""

    raw = (
        _raw_observation(
            date(2020, 8, 28),
            close="125",
            adjusted_close="120",
        ),
        _raw_observation(
            date(2020, 8, 31),
            close="129",
            adjusted_close="124",
            split="4",
        ),
    )
    prepared = transform_yahoo_rows(raw, "SPY")
    csv_text = prepared_csv_bytes(prepared).decode("utf-8")

    assert prepared[0].close == Decimal("125")
    assert prepared[1].close == Decimal("129")
    assert prepared[1].adjusted_close == Decimal("124")
    assert ",125," in csv_text
    assert ",120," not in csv_text


def test_dividends_are_diagnostic_only_and_dst_conversion_is_exact() -> None:
    """Actions do not affect Close and 16:00 New York follows DST."""

    raw = (
        _raw_observation(date(2025, 1, 2), dividend="1.23"),
        _raw_observation(date(2025, 7, 1), close="102"),
    )
    prepared = transform_yahoo_rows(raw, "SPY")

    assert prepared[0].close == Decimal("101")
    assert prepared[0].timestamp.isoformat() == "2025-01-02T21:00:00+00:00"
    assert prepared[1].timestamp.isoformat() == "2025-07-01T20:00:00+00:00"


def test_partial_download_is_rejected() -> None:
    """Exact approved start and final session are required."""

    downloader_frame = _Frame(
        (date(2005, 1, 4), date(2025, 9, 30), APPROVED_END),
        (_row(), _row(), _row()),
    )

    def partial(symbol: str, **kwargs: object) -> object:
        return _control_frame() if kwargs["start"] == "2025-12-15" else downloader_frame

    result = run_approved_etf_study(
        ApprovedEtfStudyConfig(
            output_root=Path("unused"),
            quality_report_directory=Path("unused-reports"),
            experiment_manifest=Path("unused-manifest.toml"),
            symbols=("QQQ",),
            mode=DatasetRunMode.DOWNLOAD_ONLY,
            quality_gate=_small_gate(),
        ),
        downloader=partial,
        clock=_clock,
    )
    assert result.records[0].status is DatasetRecordStatus.FAILURE
    assert "first Yahoo trading date" in (result.records[0].failure_message or "")


def test_provider_errors_are_classified_without_sensitive_state(tmp_path: Path) -> None:
    """Boundary exceptions fail safely and redact credential-shaped text."""

    downloader = _Downloader(
        calls=[],
        failure_message="token=secret-value provider unavailable",
    )
    with pytest.raises(ApprovedDatasetProviderError) as error:
        run_approved_etf_study(
            _config(tmp_path, symbols=("QQQ",), mode=DatasetRunMode.DOWNLOAD_ONLY),
            downloader=downloader,
            clock=_clock,
        )

    assert "secret-value" not in str(error.value)
    assert redact_secret("cookie: private-value") == "cookie: <redacted>"


def test_acquisition_is_sequential_and_all_parameters_are_explicit(
    tmp_path: Path,
) -> None:
    """Each symbol receives two controls then one full request in declared order."""

    downloader = _Downloader(calls=[])
    result = run_approved_etf_study(
        _config(
            tmp_path,
            symbols=("QQQ", "IWM"),
            mode=DatasetRunMode.DOWNLOAD_ONLY,
        ),
        downloader=downloader,
        clock=_clock,
    )

    assert result.succeeded
    assert [symbol for symbol, _ in downloader.calls] == [
        "QQQ",
        "QQQ",
        "QQQ",
        "IWM",
        "IWM",
        "IWM",
    ]
    full = downloader.calls[2][1]
    assert full == {
        "start": "2005-01-03",
        "end": "2026-01-01",
        "interval": "1d",
        "auto_adjust": False,
        "actions": True,
        "repair": False,
        "keepna": True,
        "threads": False,
        "progress": False,
        "rounding": False,
        "timeout": 30.0,
        "multi_level_index": False,
    }


def test_repeatability_gates_execution_fields_but_excludes_adjusted_close() -> None:
    """Unused dividend-adjusted drift does not invalidate stable execution data."""

    calls = 0

    def varying_adjusted_close(symbol: str, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        adjusted_close = 98.0 if calls == 1 else 98.0001
        return _Frame(
            (date(2025, 12, 15), date(2025, 12, 16)),
            (
                _row(adjusted_close=adjusted_close),
                _row(
                    open_price=101.0,
                    high=103.0,
                    low=100.0,
                    close=102.0,
                    adjusted_close=adjusted_close,
                ),
            ),
        )

    diagnostic = _repeatability_control(
        "SPY",
        downloader=varying_adjusted_close,
        timeout_seconds=30,
        clock=_clock,
    )
    assert diagnostic.row_count == 2

    calls = 0

    def varying_close(symbol: str, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return _Frame(
            (date(2025, 12, 15), date(2025, 12, 16)),
            (
                _row(),
                _row(
                    open_price=101.0,
                    high=103.0,
                    low=100.0,
                    close=102.0 + calls - 1,
                ),
            ),
        )

    with pytest.raises(ApprovedDatasetProviderError, match="execution-data"):
        _repeatability_control(
            "SPY",
            downloader=varying_close,
            timeout_seconds=30,
            clock=_clock,
        )


def test_full_spy_publication_retains_fmp_and_records_volume_warning(
    tmp_path: Path,
) -> None:
    """Yahoo is primary while the unchanged FMP CSV remains validation-only."""

    fmp_path = _write_fmp_reference(tmp_path / "data")
    original_fmp = fmp_path.read_bytes()
    original_metadata = fmp_path.with_suffix(".metadata.toml").read_bytes()
    downloader = _Downloader(calls=[])
    result = run_approved_etf_study(
        _config(tmp_path),
        downloader=downloader,
        clock=_clock,
    )
    record = result.records[0]

    assert result.succeeded
    assert record.fmp_comparison is not None
    assert record.fmp_comparison.volume_absolute_difference == 7_863_106
    assert record.fmp_comparison.volume_relative_difference == Decimal(
        "0.083515811274697081094299625115396944991809230578768"
    )
    assert record.prepared_path is not None
    assert "yahoo" in record.prepared_path.parts
    assert fmp_path.read_bytes() == original_fmp
    assert fmp_path.with_suffix(".metadata.toml").read_bytes() == original_metadata
    descriptor = (
        tmp_path
        / "data"
        / "reference"
        / "approved-etf-study"
        / "fmp-spy-validation-reference.toml"
    )
    descriptor_values = tomllib.loads(descriptor.read_text(encoding="utf-8"))
    assert descriptor_values["reference"]["validation_reference_only"] is True
    assert (
        descriptor_values["reference"]["excluded_from_primary_research_universe"]
        is True
    )


def test_provenance_has_personal_scope_pinned_version_and_volume_policy(
    tmp_path: Path,
) -> None:
    """Every prepared Yahoo sidecar is loader-compatible and policy-explicit."""

    _write_fmp_reference(tmp_path / "data")
    result = run_approved_etf_study(
        _config(tmp_path),
        downloader=_Downloader(calls=[]),
        clock=_clock,
    )
    record = result.records[0]
    assert record.prepared_path is not None
    assert record.provenance_path is not None
    provenance = load_dataset_provenance(record.prepared_path)
    document = tomllib.loads(record.provenance_path.read_text(encoding="utf-8"))

    assert provenance.source_name == "Yahoo Finance via yfinance"
    assert provenance.price_adjustment is PriceAdjustmentPolicy.SPLIT_ADJUSTED
    assert provenance.dividend_treatment is DividendTreatment.EXCLUDED
    assert provenance.split_treatment is SplitTreatment.ADJUSTED
    assert provenance.volume_approved_for_strategy_signals is False
    assert document["acquisition"]["yfinance_version"] == "1.5.1"
    assert document["field_semantics"]["additional_split_adjustment"] is False
    limitation = document["quality"]["personal_research_limitation"]
    assert "personal" in limitation
    assert "research" in limitation


def test_existing_file_protection_precedes_download(tmp_path: Path) -> None:
    """A protected raw destination prevents provider calls."""

    raw = (
        tmp_path
        / "data"
        / "incoming"
        / "yahoo"
        / "yahoo_QQQ_2005-01-03_2025-12-31.yfinance.json"
    )
    raw.parent.mkdir(parents=True)
    raw.write_text("protected", encoding="utf-8")
    downloader = _Downloader(calls=[])
    result = run_approved_etf_study(
        _config(
            tmp_path,
            symbols=("QQQ",),
            mode=DatasetRunMode.DOWNLOAD_ONLY,
        ),
        downloader=downloader,
        clock=_clock,
    )

    assert result.records[0].status is DatasetRecordStatus.FAILURE
    assert downloader.calls == []
    assert raw.read_text(encoding="utf-8") == "protected"


def test_existing_prepared_output_blocks_before_download(tmp_path: Path) -> None:
    """Downstream protection is checked before repeatability or full requests."""

    prepared = (
        tmp_path
        / "data"
        / "prepared"
        / "approved-etf-study"
        / "yahoo"
        / "QQQ"
        / "qqq_daily_split_adjusted.csv"
    )
    prepared.parent.mkdir(parents=True)
    prepared.write_text("protected", encoding="utf-8")
    downloader = _Downloader(calls=[])
    result = run_approved_etf_study(
        _config(tmp_path, symbols=("QQQ",)),
        downloader=downloader,
        clock=_clock,
    )

    assert result.records[0].status is DatasetRecordStatus.FAILURE
    assert downloader.calls == []
    assert prepared.read_text(encoding="utf-8") == "protected"


def test_prepare_only_reuses_retained_repeatability_without_network(
    tmp_path: Path,
) -> None:
    """Raw files retain enough audit state for deterministic offline rebuilding."""

    _write_fmp_reference(tmp_path / "data")
    first = run_approved_etf_study(
        _config(tmp_path),
        downloader=_Downloader(calls=[]),
        clock=_clock,
    )
    record = first.records[0]
    assert record.prepared_path is not None
    assert record.provenance_path is not None
    assert record.quality_report_path is not None
    record.prepared_path.unlink()
    record.provenance_path.unlink()
    record.quality_report_path.unlink()

    def reject_network(*_: object, **__: object) -> object:
        raise AssertionError("prepare-only must not call yfinance")

    rebuilt = run_approved_etf_study(
        _config(tmp_path, mode=DatasetRunMode.PREPARE_ONLY, overwrite=True),
        downloader=reject_network,
        clock=_clock,
    )

    assert rebuilt.succeeded
    assert rebuilt.records[0].repeatability is not None
    assert rebuilt.records[0].repeatability.row_count == 2


def test_authorised_subset_update_preserves_prior_universe_summary(
    tmp_path: Path,
) -> None:
    """Recovery updates replace requested rows without dropping earlier symbols."""

    _write_fmp_reference(tmp_path / "data")
    run_approved_etf_study(
        _config(tmp_path),
        downloader=_Downloader(calls=[]),
        clock=_clock,
    )
    run_approved_etf_study(
        _config(tmp_path, symbols=("QQQ",), overwrite=True),
        downloader=_Downloader(calls=[]),
        clock=_clock,
    )
    with (tmp_path / "reports" / "data-quality-summary.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(
                handle,
            )
        )

    assert [row["symbol"] for row in rows] == ["SPY", "QQQ"]


def test_fmp_comparison_uses_close_and_never_substitutes_volume(tmp_path: Path) -> None:
    """The known volume difference is recorded exactly and Yahoo remains unchanged."""

    fmp_path = _write_fmp_reference(tmp_path)
    rows = transform_yahoo_rows(
        (
            _raw_observation(APPROVED_START, close="100"),
            RawVendorObservation(
                trading_date=date(2025, 9, 30),
                open=Decimal("119"),
                high=Decimal("121"),
                low=Decimal("118"),
                close=Decimal("120"),
                adjusted_close=Decimal("118"),
                volume=86_288_000,
                dividend_amount=Decimal("0"),
                split_coefficient=Decimal("0"),
            ),
            _raw_observation(APPROVED_END, close="121"),
        ),
        "SPY",
    )
    comparison = compare_yahoo_spy_with_fmp(rows, fmp_path)

    assert comparison.maximum_absolute_close_difference == 0
    assert comparison.common_date_count == 3
    assert comparison.yahoo_volume == 86_288_000
    assert rows[1].volume == 86_288_000


class _VolumeStrategy(Strategy):
    """Synthetic future strategy declaring a volume dependency."""

    def required_market_data_fields(self) -> frozenset[str]:
        """Declare the field whose provenance approval is required."""

        return frozenset({"close", "volume"})

    def generate_signals(self, bars: Sequence[Any]) -> Sequence[Any]:
        """Return no signals; this test exercises validation only."""

        return ()


def _yahoo_provenance(*, volume_approved: bool) -> DatasetProvenance:
    """Return minimal Yahoo-compatible provenance."""

    return DatasetProvenance(
        dataset_id="yahoo-spy",
        source_name="Yahoo Finance via yfinance",
        price_adjustment=PriceAdjustmentPolicy.SPLIT_ADJUSTED,
        dividend_treatment=DividendTreatment.EXCLUDED,
        split_treatment=SplitTreatment.ADJUSTED,
        bar_frequency="daily",
        volume_approved_for_strategy_signals=volume_approved,
    )


def test_volume_dependent_strategy_requires_approval_or_explicit_override() -> None:
    """Future volume signals cannot silently consume warning-scoped data."""

    strategy = _VolumeStrategy()
    with pytest.raises(ValueError, match="requires volume"):
        validate_strategy_data_policy(
            strategy,
            _yahoo_provenance(volume_approved=False),
        )
    validate_strategy_data_policy(
        strategy,
        _yahoo_provenance(volume_approved=False),
        allow_unapproved_volume=True,
    )
    validate_strategy_data_policy(
        strategy,
        _yahoo_provenance(volume_approved=True),
    )


def test_current_sma_and_donchian_remain_price_only() -> None:
    """Existing strategies are unaffected by the new dataset policy."""

    strategies = (
        create_strategy(
            StrategyName.SMA_CROSSOVER,
            SmaCrossoverParameters(fast_window=2, slow_window=3),
        ),
        create_strategy(
            StrategyName.DONCHIAN_BREAKOUT,
            DonchianBreakoutParameters(entry_window=2, exit_window=2),
        ),
    )
    provenance = _yahoo_provenance(volume_approved=False)
    for strategy in strategies:
        assert "volume" not in strategy.required_market_data_fields()
        validate_strategy_data_policy(strategy, provenance)


def test_manifest_has_exactly_32_yahoo_evaluations_and_no_fmp_dataset(
    tmp_path: Path,
) -> None:
    """The fixed provider-specific universe remains ordered and complete."""

    manifest_path = tmp_path / "experiments" / "approved-etf-study.toml"
    manifest_path.parent.mkdir()
    manifest_path.write_bytes(
        approved_experiment_manifest_bytes(tmp_path / "data", manifest_path)
    )
    manifest = load_split_evaluation_manifest(manifest_path)
    datasets = tuple(
        str(entry.definition.dataset)
        for entry in manifest.entries
        if entry.definition is not None
    )

    assert len(manifest.entries) == 32
    assert tuple(entry.evaluation_id for entry in manifest.entries[:4]) == (
        "spy-sma-20-50",
        "spy-sma-50-200",
        "spy-donchian-20-10",
        "spy-donchian-50-20",
    )
    assert all("yahoo" in dataset for dataset in datasets)
    assert all("fmp" not in dataset.lower() for dataset in datasets)
    assert {entry.evaluation_id.split("-", 1)[0].upper() for entry in manifest.entries} == set(
        APPROVED_ETF_SYMBOLS
    )
