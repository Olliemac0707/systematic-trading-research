"""Pure Decimal return-series and risk-adjusted performance statistics.

All divisions and square roots in this module use a local 34-significant-digit
decimal context. The process-wide Decimal context is never modified.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext

from trading_research.models import BacktestResult, EquityPoint

STATISTICAL_DECIMAL_PRECISION = 34
_ONE = Decimal("1")
_ZERO = Decimal("0")


def _new_statistical_context() -> Context:
    """Return a fresh deterministic context for statistical calculations."""

    return Context(
        prec=STATISTICAL_DECIMAL_PRECISION,
        rounding=ROUND_HALF_EVEN,
    )


def _require_finite_decimal(value: Decimal, name: str) -> None:
    """Require an exact finite Decimal value."""

    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")


def _require_aware_timestamp(timestamp: datetime, name: str) -> None:
    """Require a timezone-aware observation timestamp."""

    if not isinstance(timestamp, datetime):
        raise TypeError(f"{name} must be a datetime")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _simple_return(starting_equity: Decimal, ending_equity: Decimal) -> Decimal:
    """Return one simple arithmetic return at fixed Decimal precision."""

    with localcontext(_new_statistical_context()):
        return (ending_equity / starting_equity) - _ONE


@dataclass(frozen=True, slots=True)
class PeriodReturn:
    """One simple return ending at a timezone-aware equity observation."""

    timestamp: datetime
    starting_equity: Decimal
    ending_equity: Decimal
    return_ratio: Decimal

    def __post_init__(self) -> None:
        """Validate exact inputs and the arithmetic-return relationship."""

        _require_aware_timestamp(self.timestamp, "timestamp")
        financial_values = (
            ("starting_equity", self.starting_equity),
            ("ending_equity", self.ending_equity),
            ("return_ratio", self.return_ratio),
        )
        for name, value in financial_values:
            _require_finite_decimal(value, name)
        if self.starting_equity <= _ZERO:
            raise ValueError("starting_equity must be positive")
        if self.ending_equity < _ZERO:
            raise ValueError("ending_equity must be non-negative")
        if self.return_ratio != _simple_return(
            self.starting_equity,
            self.ending_equity,
        ):
            raise ValueError("return_ratio must match the simple arithmetic return")


@dataclass(frozen=True, slots=True)
class ReturnSeries:
    """Immutable chronological periodic returns from an equity curve."""

    returns: tuple[PeriodReturn, ...]

    def __post_init__(self) -> None:
        """Validate tuple storage, chronology, and observation continuity."""

        if not isinstance(self.returns, tuple):
            raise TypeError("returns must be a tuple")
        previous: PeriodReturn | None = None
        for period_return in self.returns:
            if not isinstance(period_return, PeriodReturn):
                raise TypeError("returns must contain PeriodReturn values")
            if previous is not None:
                if period_return.timestamp <= previous.timestamp:
                    raise ValueError("return timestamps must be strictly chronological")
                if period_return.starting_equity != previous.ending_equity:
                    raise ValueError("return observations must form a continuous series")
            previous = period_return


@dataclass(frozen=True, slots=True)
class RiskMetricSettings:
    """Explicit per-period assumptions for risk-adjusted statistics."""

    periods_per_year: Decimal | None = None
    risk_free_rate_per_period: Decimal = _ZERO
    target_return_per_period: Decimal = _ZERO

    def __post_init__(self) -> None:
        """Validate finite rates and an optional positive annualisation factor."""

        if self.periods_per_year is not None:
            _require_finite_decimal(self.periods_per_year, "periods_per_year")
            if self.periods_per_year <= _ZERO:
                raise ValueError("periods_per_year must be positive")
        _require_finite_decimal(
            self.risk_free_rate_per_period,
            "risk_free_rate_per_period",
        )
        _require_finite_decimal(
            self.target_return_per_period,
            "target_return_per_period",
        )


@dataclass(frozen=True, slots=True)
class RiskAdjustedMetrics:
    """Immutable periodic and explicitly annualised equity-return statistics."""

    observation_count: int
    cumulative_return: Decimal
    arithmetic_mean_return: Decimal | None
    periodic_volatility: Decimal | None
    annualised_volatility: Decimal | None
    mean_excess_return: Decimal | None
    periodic_sharpe_ratio: Decimal | None
    annualised_sharpe_ratio: Decimal | None
    downside_deviation: Decimal | None
    periodic_sortino_ratio: Decimal | None
    annualised_sortino_ratio: Decimal | None
    risk_free_rate_per_period: Decimal
    target_return_per_period: Decimal
    periods_per_year: Decimal | None

    def __post_init__(self) -> None:
        """Validate exact metric types, optionality, and non-negative deviations."""

        if isinstance(self.observation_count, bool) or not isinstance(
            self.observation_count,
            int,
        ):
            raise TypeError("observation_count must be an integer")
        if self.observation_count < 0:
            raise ValueError("observation_count must be non-negative")

        required_decimals = (
            ("cumulative_return", self.cumulative_return),
            ("risk_free_rate_per_period", self.risk_free_rate_per_period),
            ("target_return_per_period", self.target_return_per_period),
        )
        optional_decimals = (
            ("arithmetic_mean_return", self.arithmetic_mean_return),
            ("periodic_volatility", self.periodic_volatility),
            ("annualised_volatility", self.annualised_volatility),
            ("mean_excess_return", self.mean_excess_return),
            ("periodic_sharpe_ratio", self.periodic_sharpe_ratio),
            ("annualised_sharpe_ratio", self.annualised_sharpe_ratio),
            ("downside_deviation", self.downside_deviation),
            ("periodic_sortino_ratio", self.periodic_sortino_ratio),
            ("annualised_sortino_ratio", self.annualised_sortino_ratio),
            ("periods_per_year", self.periods_per_year),
        )
        for required_name, required_value in required_decimals:
            _require_finite_decimal(required_value, required_name)
        for optional_name, optional_value in optional_decimals:
            if optional_value is not None:
                _require_finite_decimal(optional_value, optional_name)
        if self.cumulative_return < Decimal("-1"):
            raise ValueError("cumulative_return cannot be less than -1")
        if self.periodic_volatility is not None and self.periodic_volatility < _ZERO:
            raise ValueError("periodic_volatility must be non-negative")
        if self.annualised_volatility is not None and self.annualised_volatility < _ZERO:
            raise ValueError("annualised_volatility must be non-negative")
        if self.downside_deviation is not None and self.downside_deviation < _ZERO:
            raise ValueError("downside_deviation must be non-negative")
        if self.periods_per_year is not None and self.periods_per_year <= _ZERO:
            raise ValueError("periods_per_year must be positive")
        if self.periods_per_year is None and any(
            value is not None
            for value in (
                self.annualised_volatility,
                self.annualised_sharpe_ratio,
                self.annualised_sortino_ratio,
            )
        ):
            raise ValueError("annualised metrics require periods_per_year")


def construct_return_series(result: BacktestResult) -> ReturnSeries:
    """Construct simple returns, seeding the first with starting cash.

    The timestamp of each return is the ending equity observation timestamp.
    Input order is validated and never silently sorted. A zero ending equity is
    valid only for the terminal observation because it cannot serve as a later
    percentage-return denominator.
    """

    if not isinstance(result, BacktestResult):
        raise TypeError("result must be a BacktestResult")
    _require_finite_decimal(result.initial_cash, "starting_cash")
    _require_finite_decimal(result.final_equity, "ending_equity")
    if result.initial_cash <= _ZERO:
        raise ValueError("starting_cash must be positive")
    if result.final_equity < _ZERO:
        raise ValueError("ending_equity must be non-negative")

    returns: list[PeriodReturn] = []
    prior_equity = result.initial_cash
    prior_timestamp: datetime | None = None
    for point in result.equity_curve:
        if not isinstance(point, EquityPoint):
            raise TypeError("equity_curve must contain EquityPoint values")
        _require_aware_timestamp(point.timestamp, "equity timestamp")
        _require_finite_decimal(point.equity, "equity")
        if prior_timestamp is not None and point.timestamp <= prior_timestamp:
            raise ValueError("equity observations must be strictly chronological")
        if prior_equity <= _ZERO:
            raise ValueError("prior equity must be positive for a percentage return")
        if point.equity < _ZERO:
            raise ValueError("equity observations must be non-negative")
        returns.append(
            PeriodReturn(
                timestamp=point.timestamp,
                starting_equity=prior_equity,
                ending_equity=point.equity,
                return_ratio=_simple_return(prior_equity, point.equity),
            )
        )
        prior_equity = point.equity
        prior_timestamp = point.timestamp

    if returns and returns[-1].ending_equity != result.final_equity:
        raise ValueError("final equity must equal the final equity observation")
    if not returns and result.final_equity != result.initial_cash:
        raise ValueError("an empty equity curve cannot represent changed final equity")
    return ReturnSeries(tuple(returns))


def calculate_risk_adjusted_metrics(
    result: BacktestResult,
    settings: RiskMetricSettings | None = None,
) -> RiskAdjustedMetrics:
    """Calculate periodic and optional annualised equity-return statistics.

    Volatility is sample standard deviation with an ``n - 1`` denominator.
    Downside deviation uses all observations: non-shortfall returns contribute
    zero to the numerator and remain in the ``n`` denominator.
    """

    if not isinstance(result, BacktestResult):
        raise TypeError("result must be a BacktestResult")
    selected_settings = RiskMetricSettings() if settings is None else settings
    if not isinstance(selected_settings, RiskMetricSettings):
        raise TypeError("settings must be a RiskMetricSettings or None")
    return_series = construct_return_series(result)
    return_values = tuple(item.return_ratio for item in return_series.returns)
    observation_count = len(return_values)

    with localcontext(_new_statistical_context()):
        cumulative_return = (result.final_equity / result.initial_cash) - _ONE
        arithmetic_mean_return = _mean(return_values)
        periodic_volatility = _sample_standard_deviation(return_values)

        excess_returns = tuple(
            value - selected_settings.risk_free_rate_per_period
            for value in return_values
        )
        mean_excess_return = _mean(excess_returns)
        excess_volatility = _sample_standard_deviation(excess_returns)
        periodic_sharpe_ratio = _safe_ratio(
            mean_excess_return,
            excess_volatility,
        )

        downside_deviation = _downside_deviation(
            return_values,
            selected_settings.target_return_per_period,
        )
        mean_target_excess = (
            None
            if arithmetic_mean_return is None
            else arithmetic_mean_return - selected_settings.target_return_per_period
        )
        periodic_sortino_ratio = _safe_ratio(
            mean_target_excess,
            downside_deviation,
        )

        annualisation_root = (
            None
            if selected_settings.periods_per_year is None
            else selected_settings.periods_per_year.sqrt()
        )
        annualised_volatility = _scale_optional(
            periodic_volatility,
            annualisation_root,
        )
        annualised_sharpe_ratio = _scale_optional(
            periodic_sharpe_ratio,
            annualisation_root,
        )
        annualised_sortino_ratio = _scale_optional(
            periodic_sortino_ratio,
            annualisation_root,
        )

    return RiskAdjustedMetrics(
        observation_count=observation_count,
        cumulative_return=cumulative_return,
        arithmetic_mean_return=arithmetic_mean_return,
        periodic_volatility=periodic_volatility,
        annualised_volatility=annualised_volatility,
        mean_excess_return=mean_excess_return,
        periodic_sharpe_ratio=periodic_sharpe_ratio,
        annualised_sharpe_ratio=annualised_sharpe_ratio,
        downside_deviation=downside_deviation,
        periodic_sortino_ratio=periodic_sortino_ratio,
        annualised_sortino_ratio=annualised_sortino_ratio,
        risk_free_rate_per_period=selected_settings.risk_free_rate_per_period,
        target_return_per_period=selected_settings.target_return_per_period,
        periods_per_year=selected_settings.periods_per_year,
    )


def _mean(values: tuple[Decimal, ...]) -> Decimal | None:
    """Return an arithmetic mean in the active fixed-precision context."""

    if not values:
        return None
    return sum(values, _ZERO) / Decimal(len(values))


def _sample_standard_deviation(
    values: tuple[Decimal, ...],
) -> Decimal | None:
    """Return sample standard deviation using the ``n - 1`` denominator."""

    if len(values) < 2:
        return None
    mean = _mean(values)
    if mean is None:
        return None
    squared_deviations = sum(
        ((value - mean) ** 2 for value in values),
        _ZERO,
    )
    variance = squared_deviations / Decimal(len(values) - 1)
    return variance.sqrt()


def _downside_deviation(
    values: tuple[Decimal, ...],
    target_return: Decimal,
) -> Decimal | None:
    """Return target downside deviation using all observations in ``n``."""

    if not values:
        return None
    squared_shortfalls = sum(
        (min(value - target_return, _ZERO) ** 2 for value in values),
        _ZERO,
    )
    return (squared_shortfalls / Decimal(len(values))).sqrt()


def _safe_ratio(
    numerator: Decimal | None,
    denominator: Decimal | None,
) -> Decimal | None:
    """Return no ratio for missing or zero denominators instead of infinity."""

    if numerator is None or denominator is None or denominator == _ZERO:
        return None
    return numerator / denominator


def _scale_optional(
    value: Decimal | None,
    factor: Decimal | None,
) -> Decimal | None:
    """Multiply two present values and otherwise preserve missingness."""

    if value is None or factor is None:
        return None
    return value * factor
