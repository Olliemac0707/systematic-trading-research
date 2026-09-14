"""Pure, non-mutating local market-data quality diagnostics."""

from collections.abc import Sequence
from datetime import timedelta
from decimal import Decimal

from trading_research.evaluation.models import (
    DEFAULT_SUSPICIOUS_RETURN_THRESHOLD,
    DatasetQualitySummary,
)
from trading_research.models import MarketBar

_ZERO = Decimal("0")
_MICROSECONDS_PER_SECOND = Decimal("1000000")
_SECONDS_PER_DAY = Decimal("86400")


def calculate_dataset_quality(
    bars: Sequence[MarketBar],
    *,
    suspicious_return_threshold: Decimal = DEFAULT_SUSPICIOUS_RETURN_THRESHOLD,
) -> DatasetQualitySummary:
    """Describe observed bars without sorting, repairing, filtering, or interpolating.

    Gap statistics use positive adjacent gaps in the supplied order. Return
    diagnostics likewise use only strictly increasing adjacent observations.
    Non-increasing pairs remain counted and are never silently reinterpreted.
    """

    snapshot = tuple(bars)
    if not snapshot:
        raise ValueError("quality summary requires at least one market bar")
    if any(not isinstance(bar, MarketBar) for bar in snapshot):
        raise TypeError("quality summary requires MarketBar values")
    if not isinstance(suspicious_return_threshold, Decimal):
        raise TypeError("suspicious_return_threshold must be a Decimal")
    if not suspicious_return_threshold.is_finite() or suspicious_return_threshold <= 0:
        raise ValueError("suspicious_return_threshold must be positive and finite")

    timestamps = tuple(bar.timestamp for bar in snapshot)
    duplicate_count = len(timestamps) - len(set(timestamps))
    non_increasing_count = sum(
        current <= previous
        for previous, current in zip(timestamps, timestamps[1:], strict=False)
    )
    positive_gaps = tuple(
        current - previous
        for previous, current in zip(timestamps, timestamps[1:], strict=False)
        if current > previous
    )
    gap_seconds = tuple(_timedelta_seconds(gap) for gap in positive_gaps)

    close_returns: list[Decimal] = []
    for previous, current in zip(snapshot, snapshot[1:], strict=False):
        if current.timestamp <= previous.timestamp:
            continue
        close_returns.append((current.close / previous.close) - Decimal("1"))
    positive_returns = tuple(value for value in close_returns if value > _ZERO)
    negative_returns = tuple(value for value in close_returns if value < _ZERO)

    return DatasetQualitySummary(
        bar_count=len(snapshot),
        first_timestamp=min(timestamps),
        last_timestamp=max(timestamps),
        duplicate_timestamp_count=duplicate_count,
        non_increasing_timestamp_count=non_increasing_count,
        minimum_gap=min(positive_gaps) if positive_gaps else None,
        maximum_gap=max(positive_gaps) if positive_gaps else None,
        median_gap_seconds=_median(gap_seconds),
        zero_volume_count=sum(bar.volume == 0 for bar in snapshot),
        suspicious_return_threshold=suspicious_return_threshold,
        suspicious_return_count=sum(
            abs(value) >= suspicious_return_threshold for value in close_returns
        ),
        largest_positive_close_return=(max(positive_returns) if positive_returns else None),
        largest_negative_close_return=(min(negative_returns) if negative_returns else None),
    )


def _timedelta_seconds(value: timedelta) -> Decimal:
    """Convert a positive timedelta to exact Decimal seconds without float."""

    return (
        Decimal(value.days) * _SECONDS_PER_DAY
        + Decimal(value.seconds)
        + (Decimal(value.microseconds) / _MICROSECONDS_PER_SECOND)
    )


def _median(values: tuple[Decimal, ...]) -> Decimal | None:
    """Return the exact median of observed positive gaps."""

    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal("2")
