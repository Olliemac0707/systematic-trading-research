"""Tests for Decimal return-series and risk-adjusted performance metrics."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from decimal import Decimal, getcontext

import pytest

from trading_research.backtesting import SimpleBacktestEngine
from trading_research.config import BacktestConfig, EndOfTestPolicy
from trading_research.models import BacktestResult, EquityPoint, MarketBar
from trading_research.performance import (
    STATISTICAL_DECIMAL_PRECISION,
    PeriodReturn,
    RiskMetricSettings,
    calculate_performance,
    calculate_risk_adjusted_metrics,
    construct_return_series,
)
from trading_research.reporting import format_backtest_report
from trading_research.strategies import SimpleMovingAverageCrossover

_START = datetime(2025, 1, 1, tzinfo=UTC)
_STARTING_EQUITY = Decimal("100")


def make_unsafe_result(
    equity_values: tuple[Decimal, ...],
    *,
    timestamps: tuple[datetime, ...] | None = None,
    starting_equity: Decimal = _STARTING_EQUITY,
    final_equity: Decimal | None = None,
) -> BacktestResult:
    """Build a minimal BacktestResult shell for pure validation tests."""

    selected_timestamps = timestamps or tuple(
        _START + timedelta(days=index) for index in range(len(equity_values))
    )
    points = tuple(
        unsafe_equity_point(timestamp, equity)
        for timestamp, equity in zip(
            selected_timestamps,
            equity_values,
            strict=True,
        )
    )
    result = object.__new__(BacktestResult)
    object.__setattr__(result, "initial_cash", starting_equity)
    object.__setattr__(
        result,
        "final_equity",
        starting_equity if final_equity is None and not points else (
            points[-1].equity if final_equity is None else final_equity
        ),
    )
    object.__setattr__(result, "equity_curve", points)
    return result


def unsafe_equity_point(timestamp: datetime, equity: Decimal) -> EquityPoint:
    """Create an observation shell so the return layer validates bad inputs."""

    point = object.__new__(EquityPoint)
    object.__setattr__(point, "timestamp", timestamp)
    object.__setattr__(point, "equity", equity)
    return point


def make_market_bars(prices: list[int]) -> list[MarketBar]:
    """Create exact daily bars for end-policy consistency tests."""

    return [
        MarketBar(
            symbol="TEST",
            timestamp=_START + timedelta(days=index),
            open=Decimal(price),
            high=Decimal(price),
            low=Decimal(price),
            close=Decimal(price),
            volume=100,
        )
        for index, price in enumerate(prices)
    ]


def run_end_policy(policy: EndOfTestPolicy) -> BacktestResult:
    """Run a deterministic case that holds or liquidates a final long."""

    return SimpleBacktestEngine().run(
        make_market_bars([10, 10, 10, 12, 13]),
        SimpleMovingAverageCrossover(short_window=2, long_window=3),
        BacktestConfig(
            initial_cash=Decimal("1000"),
            trade_quantity=2,
            commission_bps=Decimal("100"),
            slippage_bps=Decimal("100"),
            end_of_test_policy=policy,
        ),
    )


def report_value(report: str, label: str) -> str:
    """Extract one aligned report value by label."""

    prefix = f"{label}:"
    return next(
        line[len(prefix) :].strip()
        for line in report.splitlines()
        if line.startswith(prefix)
    )


def test_empty_and_single_observation_return_policies() -> None:
    """No observations yield no statistics; one yields only one-period values."""

    empty_result = make_unsafe_result(())
    single_result = make_unsafe_result((Decimal("110"),))

    empty_series = construct_return_series(empty_result)
    empty_metrics = calculate_risk_adjusted_metrics(empty_result)
    single_series = construct_return_series(single_result)
    single_metrics = calculate_risk_adjusted_metrics(single_result)

    assert empty_series.returns == ()
    assert empty_metrics.observation_count == 0
    assert empty_metrics.cumulative_return == Decimal("0")
    assert empty_metrics.arithmetic_mean_return is None
    assert empty_metrics.periodic_volatility is None
    assert empty_metrics.mean_excess_return is None
    assert empty_metrics.downside_deviation is None
    assert empty_metrics.periodic_sharpe_ratio is None
    assert empty_metrics.periodic_sortino_ratio is None

    assert len(single_series.returns) == 1
    assert single_series.returns[0] == PeriodReturn(
        timestamp=_START,
        starting_equity=Decimal("100"),
        ending_equity=Decimal("110"),
        return_ratio=Decimal("0.1"),
    )
    assert single_metrics.arithmetic_mean_return == Decimal("0.1")
    assert single_metrics.periodic_volatility is None
    assert single_metrics.periodic_sharpe_ratio is None
    assert single_metrics.downside_deviation == Decimal("0")
    assert single_metrics.periodic_sortino_ratio is None


def test_multiple_returns_use_starting_cash_and_observation_timestamps() -> None:
    """Starting cash seeds positive, negative, and zero simple returns in order."""

    result = make_unsafe_result(
        (Decimal("110"), Decimal("99"), Decimal("99")),
    )
    original_points = result.equity_curve

    series = construct_return_series(result)

    assert tuple(item.return_ratio for item in series.returns) == (
        Decimal("0.1"),
        Decimal("-0.1"),
        Decimal("0"),
    )
    assert tuple(item.timestamp for item in series.returns) == tuple(
        point.timestamp for point in original_points
    )
    assert series.returns[0].starting_equity == result.initial_cash
    assert series.returns[1].starting_equity == Decimal("110")
    assert result.equity_curve is original_points


def test_compounded_returns_and_cumulative_return_match_authoritative_total() -> None:
    """The sequence compounds to the same ending-equity total return."""

    result = make_unsafe_result(
        (Decimal("110"), Decimal("99"), Decimal("108.9")),
    )
    series = construct_return_series(result)
    metrics = calculate_risk_adjusted_metrics(result)
    compounded = Decimal("1")
    for observation in series.returns:
        compounded *= Decimal("1") + observation.return_ratio

    assert compounded - Decimal("1") == Decimal("0.089")
    assert metrics.cumulative_return == Decimal("0.089")
    assert metrics.cumulative_return == result.total_return


@pytest.mark.parametrize(
    ("timestamps", "message"),
    [
        ((_START, _START), "strictly chronological"),
        ((_START + timedelta(days=1), _START), "strictly chronological"),
        ((datetime(2025, 1, 1),), "timezone-aware"),
    ],
    ids=["duplicate", "unordered", "naive"],
)
def test_return_series_rejects_invalid_timestamp_order(
    timestamps: tuple[datetime, ...],
    message: str,
) -> None:
    """Return observations are aware and strictly ordered without sorting."""

    values = tuple(Decimal("100") for _ in timestamps)
    result = make_unsafe_result(values, timestamps=timestamps)

    with pytest.raises(ValueError, match=message):
        construct_return_series(result)


@pytest.mark.parametrize(
    ("equity_values", "starting_equity", "message"),
    [
        ((Decimal("0"), Decimal("1")), Decimal("100"), "prior equity"),
        ((Decimal("-1"),), Decimal("100"), "non-negative"),
        ((Decimal("NaN"),), Decimal("100"), "finite"),
        ((Decimal("Infinity"),), Decimal("100"), "finite"),
        ((Decimal("100"),), Decimal("0"), "starting_cash must be positive"),
        ((Decimal("100"),), Decimal("-1"), "starting_cash must be positive"),
    ],
    ids=[
        "zero-prior",
        "negative-equity",
        "nan-equity",
        "infinite-equity",
        "zero-start",
        "negative-start",
    ],
)
def test_return_series_rejects_invalid_equity_denominators(
    equity_values: tuple[Decimal, ...],
    starting_equity: Decimal,
    message: str,
) -> None:
    """Invalid percentage-return values fail explicitly rather than being hidden."""

    result = make_unsafe_result(
        equity_values,
        starting_equity=starting_equity,
    )

    with pytest.raises(ValueError, match=message):
        construct_return_series(result)


def test_return_series_rejects_authoritative_ending_equity_mismatch() -> None:
    """A stale terminal observation is rejected rather than supplemented."""

    result = make_unsafe_result(
        (Decimal("100"), Decimal("101")),
        final_equity=Decimal("102"),
    )

    with pytest.raises(ValueError, match="final equity observation"):
        construct_return_series(result)

    empty_changed_result = make_unsafe_result((), final_equity=Decimal("101"))
    with pytest.raises(ValueError, match="empty equity curve"):
        construct_return_series(empty_changed_result)


def test_sample_mean_volatility_and_mixed_returns_are_exact() -> None:
    """Volatility uses the documented sample n-minus-one denominator."""

    metrics = calculate_risk_adjusted_metrics(
        make_unsafe_result((Decimal("110"), Decimal("99"))),
    )

    assert metrics.observation_count == 2
    assert metrics.arithmetic_mean_return == Decimal("0")
    assert metrics.periodic_volatility == Decimal(
        "0.1414213562373095048801688724209698"
    )
    assert metrics.mean_excess_return == Decimal("0")
    assert metrics.periodic_sharpe_ratio == Decimal("0")
    assert metrics.downside_deviation == Decimal(
        "0.07071067811865475244008443621048490"
    )
    assert metrics.periodic_sortino_ratio == Decimal("0")


def test_constant_returns_have_zero_volatility_and_no_infinite_ratios() -> None:
    """Zero standard deviation and zero downside return None ratios."""

    metrics = calculate_risk_adjusted_metrics(
        make_unsafe_result((Decimal("110"), Decimal("121"))),
    )

    assert metrics.arithmetic_mean_return == Decimal("0.1")
    assert metrics.periodic_volatility == Decimal("0")
    assert metrics.periodic_sharpe_ratio is None
    assert metrics.downside_deviation == Decimal("0")
    assert metrics.periodic_sortino_ratio is None


def test_sharpe_uses_per_period_risk_free_rate_and_explicit_annualisation() -> None:
    """Sharpe supports positive and negative mean excess without infinity."""

    result = make_unsafe_result((Decimal("110"), Decimal("132")))
    positive = calculate_risk_adjusted_metrics(
        result,
        RiskMetricSettings(periods_per_year=Decimal("4")),
    )
    negative = calculate_risk_adjusted_metrics(
        result,
        RiskMetricSettings(risk_free_rate_per_period=Decimal("0.20")),
    )

    assert positive.arithmetic_mean_return == Decimal("0.15")
    assert positive.periodic_volatility == Decimal(
        "0.07071067811865475244008443621048490"
    )
    assert positive.periodic_sharpe_ratio == Decimal(
        "2.121320343559642573202533086314547"
    )
    assert positive.annualised_sharpe_ratio == Decimal(
        "4.242640687119285146405066172629094"
    )
    assert positive.annualised_volatility == Decimal(
        "0.1414213562373095048801688724209698"
    )
    assert negative.mean_excess_return == Decimal("-0.05")
    assert negative.periodic_sharpe_ratio == Decimal(
        "-0.7071067811865475244008443621048491"
    )
    assert negative.annualised_sharpe_ratio is None


def test_downside_and_sortino_use_all_observations_and_target_return() -> None:
    """Non-shortfalls remain in n while positive and negative targets are allowed."""

    result = make_unsafe_result((Decimal("110"), Decimal("99")))
    targeted = calculate_risk_adjusted_metrics(
        result,
        RiskMetricSettings(
            periods_per_year=Decimal("4"),
            target_return_per_period=Decimal("0.05"),
        ),
    )
    negative_target = calculate_risk_adjusted_metrics(
        result,
        RiskMetricSettings(target_return_per_period=Decimal("-0.20")),
    )

    assert targeted.downside_deviation == Decimal(
        "0.1060660171779821286601266543157274"
    )
    assert targeted.periodic_sortino_ratio == Decimal(
        "-0.4714045207910316829338962414032325"
    )
    assert targeted.annualised_sortino_ratio == Decimal(
        "-0.9428090415820633658677924828064650"
    )
    assert negative_target.downside_deviation == Decimal("0")
    assert negative_target.periodic_sortino_ratio is None


@pytest.mark.parametrize(
    "periods_per_year",
    [Decimal("0"), Decimal("-1"), Decimal("NaN"), Decimal("Infinity")],
)
def test_settings_reject_invalid_annualisation_factors(
    periods_per_year: Decimal,
) -> None:
    """Annualisation is optional but must be positive and finite when supplied."""

    with pytest.raises(ValueError):
        RiskMetricSettings(periods_per_year=periods_per_year)


@pytest.mark.parametrize("field", ["risk_free_rate_per_period", "target_return_per_period"])
@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity")])
def test_settings_reject_non_finite_per_period_rates(
    field: str,
    value: Decimal,
) -> None:
    """Rates are direct Decimal ratios and must be finite."""

    with pytest.raises(ValueError):
        RiskMetricSettings(**{field: value})


def test_factor_one_and_missing_factor_annualisation_policies() -> None:
    """No factor omits annual metrics and a factor of one preserves periodic ones."""

    result = make_unsafe_result((Decimal("110"), Decimal("99")))
    periodic = calculate_risk_adjusted_metrics(result)
    factor_one = calculate_risk_adjusted_metrics(
        result,
        RiskMetricSettings(periods_per_year=Decimal("1")),
    )

    assert periodic.annualised_volatility is None
    assert periodic.annualised_sharpe_ratio is None
    assert periodic.annualised_sortino_ratio is None
    assert factor_one.annualised_volatility == factor_one.periodic_volatility
    assert factor_one.annualised_sharpe_ratio == factor_one.periodic_sharpe_ratio
    assert factor_one.annualised_sortino_ratio == factor_one.periodic_sortino_ratio


def test_fixed_precision_is_deterministic_and_global_context_is_unchanged() -> None:
    """The documented 34-digit local context isolates calculations from globals."""

    result = make_unsafe_result((Decimal("110"), Decimal("99")))
    original_context = getcontext().copy()
    try:
        getcontext().prec = 7
        first = calculate_risk_adjusted_metrics(
            result,
            RiskMetricSettings(periods_per_year=Decimal("252")),
        )
        getcontext().prec = 50
        second = calculate_risk_adjusted_metrics(
            result,
            RiskMetricSettings(periods_per_year=Decimal("252")),
        )
        assert first == second
        assert STATISTICAL_DECIMAL_PRECISION == 34
        assert getcontext().prec == 50
    finally:
        getcontext().prec = original_context.prec
        getcontext().rounding = original_context.rounding


def test_hold_and_liquidate_use_their_authoritative_final_equity() -> None:
    """Both end policies remain reconciled and produce consistent total returns."""

    held = run_end_policy(EndOfTestPolicy.HOLD)
    liquidated = run_end_policy(EndOfTestPolicy.LIQUIDATE)

    held_metrics = calculate_risk_adjusted_metrics(held)
    liquidated_metrics = calculate_risk_adjusted_metrics(liquidated)

    assert held.equity_curve[-1].equity == held.final_equity
    assert liquidated.equity_curve[-1].equity == liquidated.final_equity
    assert held_metrics.cumulative_return == calculate_performance(held).total_return
    assert (
        liquidated_metrics.cumulative_return
        == calculate_performance(liquidated).total_return
    )
    assert held.reconciliation.is_reconciled
    assert liquidated.reconciliation.is_reconciled
    assert held.trades[0].price == liquidated.trades[0].price == Decimal("13.13")
    assert held.trades[0].commission == liquidated.trades[0].commission == Decimal(
        "0.2626"
    )


def test_reporting_is_opt_in_and_formats_ratios_with_correct_units() -> None:
    """Periodic metrics are optional and Sharpe/Sortino are not percentages."""

    result = run_end_policy(EndOfTestPolicy.HOLD)
    default_report = format_backtest_report(result)
    periodic_report = format_backtest_report(
        result,
        risk_metric_settings=RiskMetricSettings(),
    )
    annualised_report = format_backtest_report(
        result,
        risk_metric_settings=RiskMetricSettings(periods_per_year=Decimal("252")),
    )

    assert "Return and Risk Statistics" not in default_report
    assert "Return and Risk Statistics" in periodic_report
    assert report_value(periodic_report, "Return observations") == "5"
    assert report_value(periodic_report, "Annualised volatility") == "Not available"
    assert report_value(periodic_report, "Annualised Sharpe ratio") == "Not available"
    assert report_value(periodic_report, "Periods per year") == "Not available"
    assert "%" not in report_value(periodic_report, "Periodic Sharpe ratio")
    assert "%" not in report_value(periodic_report, "Periodic Sortino ratio")
    assert report_value(annualised_report, "Periods per year") == "252"
    assert report_value(annualised_report, "Annualised volatility") != "Not available"


def test_return_models_are_immutable() -> None:
    """Return observations and settings cannot be mutated after validation."""

    observation = construct_return_series(
        make_unsafe_result((Decimal("100"),)),
    ).returns[0]
    settings = RiskMetricSettings()
    return_attribute = "return_ratio"
    periods_attribute = "periods_per_year"

    with pytest.raises(FrozenInstanceError):
        setattr(observation, return_attribute, Decimal("1"))
    with pytest.raises(FrozenInstanceError):
        setattr(settings, periods_attribute, Decimal("252"))
