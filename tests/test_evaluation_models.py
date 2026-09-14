"""Validation tests for provenance, split boundaries, and frozen fingerprints."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from evaluation_helpers import evaluation_split, frozen_configuration, provenance
from pydantic import ValidationError

from trading_research.evaluation import (
    DatasetProvenance,
    DividendTreatment,
    EvaluationSplit,
    FrozenEvaluationConfiguration,
    PriceAdjustmentPolicy,
    SplitTreatment,
    WarmupPolicy,
    canonical_configuration_json,
    configuration_sha256,
)
from trading_research.experiments import BenchmarkSelection
from trading_research.reporting import StrategyRunConfiguration
from trading_research.strategies import SmaCrossoverParameters, StrategyName


def test_valid_unknown_provenance_is_immutable_and_does_not_verify_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown treatment remains explicit and source URLs are descriptive only."""

    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("provenance validation must not access the network")

    monkeypatch.setattr("socket.create_connection", reject_network)
    model = DatasetProvenance(
        dataset_id="aapl-daily",
        source_name="user supplied",
        source_url="https://example.invalid/aapl.csv",
        downloaded_at=datetime(2025, 1, 1, tzinfo=UTC),
        price_adjustment=PriceAdjustmentPolicy.UNKNOWN,
        dividend_treatment=DividendTreatment.UNKNOWN,
        split_treatment=SplitTreatment.UNKNOWN,
        bar_frequency="daily",
    )

    assert model.price_adjustment is PriceAdjustmentPolicy.UNKNOWN
    assert model.dividend_treatment is DividendTreatment.UNKNOWN
    assert model.split_treatment is SplitTreatment.UNKNOWN
    with pytest.raises(ValidationError, match="frozen"):
        model.__setattr__("dataset_id", "changed")


def test_explicit_total_return_adjustment_requires_complete_consistent_claims() -> None:
    """An adjusted dataset is accepted only with matching dividend and split treatment."""

    model = DatasetProvenance(
        dataset_id="adjusted-daily",
        source_name="documented source",
        price_adjustment=PriceAdjustmentPolicy.TOTAL_RETURN_ADJUSTED,
        dividend_treatment=DividendTreatment.INCLUDED_IN_ADJUSTED_PRICE,
        split_treatment=SplitTreatment.ADJUSTED,
        bar_frequency="daily",
    )

    assert model.price_adjustment is PriceAdjustmentPolicy.TOTAL_RETURN_ADJUSTED


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "price_adjustment": "total-return-adjusted",
                "dividend_treatment": "excluded",
                "split_treatment": "adjusted",
            },
            "require dividends included",
        ),
        (
            {
                "price_adjustment": "split-adjusted",
                "dividend_treatment": "excluded",
                "split_treatment": "unadjusted",
            },
            "require splits adjusted",
        ),
    ],
)
def test_contradictory_provenance_is_rejected(
    overrides: dict[str, str],
    message: str,
) -> None:
    """Adjustment claims cannot contradict their dividend or split declarations."""

    with pytest.raises(ValidationError, match=message):
        DatasetProvenance.model_validate(
            {
                "dataset_id": "contradiction",
                "source_name": "test",
                "bar_frequency": "daily",
                **overrides,
            }
        )


def test_provenance_download_timestamp_must_be_timezone_aware() -> None:
    """Acquisition timestamps without offsets are ambiguous and rejected."""

    with pytest.raises(ValidationError, match="timezone-aware"):
        DatasetProvenance(
            dataset_id="naive",
            source_name="test",
            downloaded_at=datetime(2025, 1, 1),
            bar_frequency="daily",
        )


def test_valid_split_accepts_gap_and_exposes_inclusive_boundaries() -> None:
    """A gap is allowed and retained rather than filled or inferred."""

    split = EvaluationSplit(
        development_start=datetime(2025, 1, 1, tzinfo=UTC),
        development_end=datetime(2025, 1, 5, tzinfo=UTC),
        holdout_start=datetime(2025, 1, 10, tzinfo=UTC),
        holdout_end=datetime(2025, 1, 20, tzinfo=UTC),
        warmup_policy=WarmupPolicy.CARRY_HISTORY,
    )

    assert split.boundary_gap == timedelta(days=5)
    assert split.warmup_policy is WarmupPolicy.CARRY_HISTORY


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("development_end", datetime(2025, 1, 10, tzinfo=UTC), "must not overlap"),
        ("development_start", datetime(2025, 1, 9, tzinfo=UTC), "must not be after"),
        ("holdout_end", datetime(2025, 1, 7, tzinfo=UTC), "must not be after"),
        ("holdout_start", datetime(2025, 1, 9), "timezone-aware"),
    ],
)
def test_invalid_split_boundaries_are_rejected(
    field: str,
    value: datetime,
    message: str,
) -> None:
    """Overlaps, reversals, and naive boundaries fail explicit validation."""

    values = {
        "development_start": datetime(2025, 1, 1, tzinfo=UTC),
        "development_end": datetime(2025, 1, 8, tzinfo=UTC),
        "holdout_start": datetime(2025, 1, 9, tzinfo=UTC),
        "holdout_end": datetime(2025, 1, 16, tzinfo=UTC),
    }
    values[field] = value

    with pytest.raises(ValidationError, match=message):
        EvaluationSplit.model_validate(values)


def test_configuration_fingerprint_is_stable_and_exact() -> None:
    """Canonical keys and Decimal strings produce a repeatable digest."""

    configuration = frozen_configuration(commission_bps=Decimal("0.10"))
    canonical = canonical_configuration_json(configuration)

    assert configuration_sha256(configuration) == configuration_sha256(configuration)
    assert '"commission_bps":"0.10"' in canonical
    assert '"generated_at"' not in canonical
    assert '"run_id"' not in canonical
    assert ".0," not in canonical


def test_strategy_parameter_and_commission_changes_alter_fingerprint() -> None:
    """Material strategy and cost changes cannot share a frozen fingerprint."""

    baseline = frozen_configuration()
    changed_strategy = replace(
        baseline,
        strategy=StrategyRunConfiguration(
            name=StrategyName.SMA_CROSSOVER,
            parameters=SmaCrossoverParameters(fast_window=2, slow_window=4),
        ),
    )
    changed_commission = FrozenEvaluationConfiguration(
        strategy=baseline.strategy,
        backtest=baseline.backtest.model_copy(
            update={"commission_bps": Decimal("2")}
        ),
        risk_metrics=baseline.risk_metrics,
        benchmark=BenchmarkSelection.NONE,
    )

    assert configuration_sha256(changed_strategy) != configuration_sha256(baseline)
    assert configuration_sha256(changed_commission) != configuration_sha256(baseline)


def test_external_timestamps_and_ids_cannot_change_fingerprint() -> None:
    """Run identity is absent from the frozen configuration by construction."""

    configuration = frozen_configuration()
    first = configuration_sha256(configuration)
    generated_at = datetime(2025, 1, 1, tzinfo=UTC)
    run_id = "development-run"

    generated_at = generated_at + timedelta(days=1)
    run_id = "holdout-run"

    assert generated_at != datetime(2025, 1, 1, tzinfo=UTC)
    assert run_id == "holdout-run"
    assert configuration_sha256(configuration) == first


def test_helper_provenance_and_split_are_valid() -> None:
    """Shared deterministic fixtures retain the intended stable conventions."""

    assert provenance().dataset_id == "synthetic-daily-v1"
    assert evaluation_split().development_end < evaluation_split().holdout_start
