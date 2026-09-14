"""Pure deterministic dataset-quality diagnostic tests."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

from evaluation_helpers import synthetic_bars

from trading_research.evaluation import calculate_dataset_quality


def test_normal_ordered_data_reports_exact_gap_and_volume_summary() -> None:
    """Daily observations produce exact Decimal gap statistics without warnings."""

    bars = synthetic_bars(closes=("10", "11", "12"))
    summary = calculate_dataset_quality(bars)

    assert summary.bar_count == 3
    assert summary.duplicate_timestamp_count == 0
    assert summary.non_increasing_timestamp_count == 0
    assert summary.minimum_gap == summary.maximum_gap == timedelta(days=1)
    assert summary.median_gap_seconds == Decimal("86400")
    assert summary.zero_volume_count == 0
    assert summary.suspicious_return_count == 0


def test_extreme_returns_zero_volume_and_irregular_gaps_are_diagnostic_only() -> None:
    """Large moves and gaps remain present while their diagnostics are counted."""

    original = synthetic_bars(closes=("10", "15", "6", "6"))
    bars = (
        original[0],
        replace(original[1], timestamp=original[0].timestamp + timedelta(days=1)),
        replace(original[2], timestamp=original[0].timestamp + timedelta(days=4)),
        replace(
            original[3],
            timestamp=original[0].timestamp + timedelta(days=5),
            volume=0,
        ),
    )
    snapshot = tuple(bars)

    summary = calculate_dataset_quality(
        bars,
        suspicious_return_threshold=Decimal("0.40"),
    )

    assert bars == snapshot
    assert len(bars) == 4
    assert summary.suspicious_return_count == 2
    assert summary.largest_positive_close_return == Decimal("0.5")
    assert summary.largest_negative_close_return == Decimal("-0.6")
    assert summary.zero_volume_count == 1
    assert summary.minimum_gap == timedelta(days=1)
    assert summary.maximum_gap == timedelta(days=3)
    assert summary.median_gap_seconds == Decimal("86400")


def test_duplicate_and_non_increasing_observations_are_counted_not_reordered() -> None:
    """Input-order defects are visible and excluded only from adjacent calculations."""

    ordered = synthetic_bars(closes=("10", "11", "12"))
    bars = (ordered[0], ordered[1], ordered[1], ordered[0], ordered[2])

    summary = calculate_dataset_quality(bars)

    assert summary.bar_count == 5
    assert summary.duplicate_timestamp_count == 2
    assert summary.non_increasing_timestamp_count == 2
    assert bars[2] == ordered[1]
    assert bars[3] == ordered[0]


def test_single_observation_has_no_invented_gap_or_return() -> None:
    """No calendar or neighbouring value is inferred for a one-bar dataset."""

    summary = calculate_dataset_quality(synthetic_bars(closes=("10",)))

    assert summary.minimum_gap is None
    assert summary.maximum_gap is None
    assert summary.median_gap_seconds is None
    assert summary.largest_positive_close_return is None
    assert summary.largest_negative_close_return is None
