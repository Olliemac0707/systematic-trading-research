"""Pure observation-based exposure and pre-trade decision statistics."""

from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext

from trading_research.models import (
    BacktestResult,
    PreTradeDecisionOutcome,
)

EXPOSURE_DECIMAL_PRECISION = 34
_ONE = Decimal("1")
_ZERO = Decimal("0")


def _new_exposure_context() -> Context:
    """Return a fresh fixed-precision context for deterministic ratios."""

    return Context(prec=EXPOSURE_DECIMAL_PRECISION, rounding=ROUND_HALF_EVEN)


def _require_finite_decimal(value: Decimal, name: str) -> None:
    """Require an exact finite Decimal value."""

    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")


def _require_aware_timestamp(value: datetime, name: str) -> None:
    """Require an unambiguous observation timestamp."""

    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _ratio(numerator: Decimal, denominator: Decimal) -> Decimal:
    """Return one ratio in the isolated exposure context."""

    with localcontext(_new_exposure_context()):
        return numerator / denominator


def _composition_ratios(
    position_market_value: Decimal,
    equity: Decimal,
) -> tuple[Decimal, Decimal]:
    """Return utilisation and its exact residual cash share.

    Independent finite-precision division of both account components can make
    their rounded ratios differ infinitesimally from one. The position share is
    canonical and the cash share is its exact residual.
    """

    with localcontext(_new_exposure_context()):
        utilisation = position_market_value / equity
        cash_share = _ONE - utilisation
    return utilisation, cash_share


def _decision_rates(
    full_count: int,
    reduced_count: int,
    actionable_count: int,
) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    """Return full, reduction, rejection, and execution rates that sum exactly.

    Finite-precision ``Decimal`` cannot represent every rational fraction. Full
    and reduction rates use direct division; rejection is the exact residual so
    the three mutually exclusive outcome rates retain their required partition.
    Execution approval is full approval plus reduction.
    """

    count = Decimal(actionable_count)
    with localcontext(_new_exposure_context()):
        full_rate = Decimal(full_count) / count
        reduction_rate = Decimal(reduced_count) / count
        rejection_rate = _ONE - full_rate - reduction_rate
        execution_rate = full_rate + reduction_rate
    return full_rate, reduction_rate, rejection_rate, execution_rate


@dataclass(frozen=True, slots=True)
class ExposureObservation:
    """Cash and long-position composition at one authoritative equity point."""

    timestamp: datetime
    cash: Decimal
    position_quantity: int
    position_market_value: Decimal
    equity: Decimal
    capital_utilisation_ratio: Decimal
    cash_ratio: Decimal
    invested: bool

    def __post_init__(self) -> None:
        """Validate exact account composition and canonical utilisation ratios."""

        _require_aware_timestamp(self.timestamp, "timestamp")
        for name, value in (
            ("cash", self.cash),
            ("position_market_value", self.position_market_value),
            ("equity", self.equity),
            ("capital_utilisation_ratio", self.capital_utilisation_ratio),
            ("cash_ratio", self.cash_ratio),
        ):
            _require_finite_decimal(value, name)
        if self.cash < _ZERO or self.position_market_value < _ZERO:
            raise ValueError("cash and position market value must be non-negative")
        if self.equity <= _ZERO:
            raise ValueError("exposure observations require positive equity")
        if isinstance(self.position_quantity, bool) or not isinstance(
            self.position_quantity,
            int,
        ):
            raise TypeError("position_quantity must be an integer")
        if self.position_quantity < 0:
            raise ValueError("position_quantity must be non-negative")
        if not isinstance(self.invested, bool):
            raise TypeError("invested must be a bool")
        if self.invested != (self.position_quantity > 0):
            raise ValueError("invested must match whether position quantity is positive")
        if self.cash + self.position_market_value != self.equity:
            raise ValueError("cash plus position market value must equal equity")
        if self.position_quantity == 0 and self.position_market_value != _ZERO:
            raise ValueError("a cash-only observation must have zero position value")
        if self.position_quantity > 0 and self.position_market_value <= _ZERO:
            raise ValueError("an invested observation requires positive position value")
        expected_utilisation, expected_cash = _composition_ratios(
            self.position_market_value,
            self.equity,
        )
        if self.capital_utilisation_ratio != expected_utilisation:
            raise ValueError("capital utilisation must equal position value over equity")
        if self.cash_ratio != expected_cash:
            raise ValueError("cash ratio must be the residual account share")
        if not (_ZERO <= self.capital_utilisation_ratio <= _ONE):
            raise ValueError("capital utilisation must be between 0 and 1")
        if not (_ZERO <= self.cash_ratio <= _ONE):
            raise ValueError("cash ratio must be between 0 and 1")
        if self.capital_utilisation_ratio + self.cash_ratio != _ONE:
            raise ValueError("cash and capital utilisation ratios must sum to one")


@dataclass(frozen=True, slots=True)
class ExposureStatistics:
    """Immutable observation exposure and entry-decision outcome statistics."""

    observation_count: int
    invested_observation_count: int
    cash_only_observation_count: int
    time_in_market_ratio: Decimal | None
    average_position_quantity: Decimal | None
    maximum_position_quantity: Decimal
    average_position_market_value: Decimal | None
    maximum_position_market_value: Decimal
    average_position_value_ratio: Decimal | None
    maximum_position_value_ratio: Decimal | None
    average_cash: Decimal | None
    minimum_cash: Decimal | None
    average_cash_ratio: Decimal | None
    minimum_cash_ratio: Decimal | None
    average_capital_utilisation_ratio: Decimal | None
    maximum_capital_utilisation_ratio: Decimal | None
    actionable_signal_count: int
    full_approval_count: int
    reduced_decision_count: int
    rejected_decision_count: int
    proposed_quantity_total: Decimal
    approved_quantity_total: Decimal
    quantity_reduction_total: Decimal
    execution_approval_rate: Decimal | None
    full_approval_rate: Decimal | None
    reduction_rate: Decimal | None
    rejection_rate: Decimal | None

    def __post_init__(self) -> None:
        """Validate count partitions, exact aggregates, and unambiguous rates."""

        counts = (
            self.observation_count,
            self.invested_observation_count,
            self.cash_only_observation_count,
            self.actionable_signal_count,
            self.full_approval_count,
            self.reduced_decision_count,
            self.rejected_decision_count,
        )
        if any(isinstance(count, bool) or not isinstance(count, int) for count in counts):
            raise TypeError("exposure and decision counts must be integers")
        if any(count < 0 for count in counts):
            raise ValueError("exposure and decision counts must be non-negative")
        if self.observation_count != (
            self.invested_observation_count + self.cash_only_observation_count
        ):
            raise ValueError("observation count must equal invested plus cash-only counts")
        if self.actionable_signal_count != (
            self.full_approval_count
            + self.reduced_decision_count
            + self.rejected_decision_count
        ):
            raise ValueError("actionable count must equal all decision outcomes")

        required_decimals = (
            ("maximum_position_quantity", self.maximum_position_quantity),
            ("maximum_position_market_value", self.maximum_position_market_value),
            ("proposed_quantity_total", self.proposed_quantity_total),
            ("approved_quantity_total", self.approved_quantity_total),
            ("quantity_reduction_total", self.quantity_reduction_total),
        )
        observation_optionals = (
            ("time_in_market_ratio", self.time_in_market_ratio),
            ("average_position_quantity", self.average_position_quantity),
            ("average_position_market_value", self.average_position_market_value),
            ("average_position_value_ratio", self.average_position_value_ratio),
            ("maximum_position_value_ratio", self.maximum_position_value_ratio),
            ("average_cash", self.average_cash),
            ("minimum_cash", self.minimum_cash),
            ("average_cash_ratio", self.average_cash_ratio),
            ("minimum_cash_ratio", self.minimum_cash_ratio),
            (
                "average_capital_utilisation_ratio",
                self.average_capital_utilisation_ratio,
            ),
            (
                "maximum_capital_utilisation_ratio",
                self.maximum_capital_utilisation_ratio,
            ),
        )
        decision_rates = (
            ("execution_approval_rate", self.execution_approval_rate),
            ("full_approval_rate", self.full_approval_rate),
            ("reduction_rate", self.reduction_rate),
            ("rejection_rate", self.rejection_rate),
        )
        for required_name, required_value in required_decimals:
            _require_finite_decimal(required_value, required_name)
            if required_value < _ZERO:
                raise ValueError(f"{required_name} must be non-negative")
        for optional_name, optional_value in observation_optionals + decision_rates:
            if optional_value is not None:
                _require_finite_decimal(optional_value, optional_name)
        if self.quantity_reduction_total != (
            self.proposed_quantity_total - self.approved_quantity_total
        ):
            raise ValueError("quantity reduction must reconcile to proposed and approved")

        if self.observation_count == 0:
            if any(value is not None for _, value in observation_optionals):
                raise ValueError("empty exposure statistics must not contain averages")
            if (
                self.maximum_position_quantity != _ZERO
                or self.maximum_position_market_value != _ZERO
            ):
                raise ValueError("empty exposure statistics must have zero maxima")
        else:
            if any(value is None for _, value in observation_optionals):
                raise ValueError("non-empty exposure statistics require all observation values")
            expected_time = _ratio(
                Decimal(self.invested_observation_count),
                Decimal(self.observation_count),
            )
            if self.time_in_market_ratio != expected_time:
                raise ValueError("time in market must use the observation-count policy")

        if (
            self.average_position_value_ratio
            != self.average_capital_utilisation_ratio
            or self.maximum_position_value_ratio
            != self.maximum_capital_utilisation_ratio
        ):
            raise ValueError(
                "position-value and capital-utilisation ratios must use one calculation"
            )

        if self.actionable_signal_count == 0:
            if any(value is not None for _, value in decision_rates):
                raise ValueError("empty decision statistics must not contain rates")
        else:
            if any(value is None for _, value in decision_rates):
                raise ValueError("non-empty decision statistics require all rates")
            (
                expected_full,
                expected_reduced,
                expected_rejected,
                expected_execution,
            ) = _decision_rates(
                self.full_approval_count,
                self.reduced_decision_count,
                self.actionable_signal_count,
            )
            if self.full_approval_rate != expected_full:
                raise ValueError("full approval rate is inconsistent")
            if self.reduction_rate != expected_reduced:
                raise ValueError("reduction rate is inconsistent")
            if self.rejection_rate != expected_rejected:
                raise ValueError("rejection rate is inconsistent")
            if self.execution_approval_rate != expected_execution:
                raise ValueError("execution approval must include full and reduced orders")
            if expected_full + expected_reduced + expected_rejected != _ONE:
                raise ValueError("decision outcome rates must reconcile to one")


def derive_exposure_observations(
    result: BacktestResult,
) -> tuple[ExposureObservation, ...]:
    """Derive ordered account composition from the ledger and equity sequence."""

    if not isinstance(result, BacktestResult):
        raise TypeError("result must be a BacktestResult")
    ledger_entries = result.reconciliation.ledger.entries
    ledger_index = 0
    running_cash = result.initial_cash
    running_quantity = 0
    observations: list[ExposureObservation] = []

    for point in result.equity_curve:
        while (
            ledger_index < len(ledger_entries)
            and ledger_entries[ledger_index].trade.timestamp <= point.timestamp
        ):
            entry = ledger_entries[ledger_index]
            running_cash += entry.cash_change
            running_quantity += entry.position_change
            ledger_index += 1
        position_market_value = point.equity - running_cash
        if point.equity <= _ZERO:
            raise ValueError("exposure calculation requires positive equity")
        utilisation_ratio, cash_ratio = _composition_ratios(
            position_market_value,
            point.equity,
        )
        observations.append(
            ExposureObservation(
                timestamp=point.timestamp,
                cash=running_cash,
                position_quantity=running_quantity,
                position_market_value=position_market_value,
                equity=point.equity,
                capital_utilisation_ratio=utilisation_ratio,
                cash_ratio=cash_ratio,
                invested=running_quantity > 0,
            )
        )

    if ledger_index != len(ledger_entries):
        raise ValueError("trade execution occurs after the final equity observation")
    if observations:
        final = observations[-1]
        if final.cash != result.final_cash:
            raise ValueError("final exposure cash must match authoritative ending cash")
        if final.position_quantity != result.final_position_quantity:
            raise ValueError("final exposure position must match authoritative ending position")
        if final.equity != result.final_equity:
            raise ValueError("final exposure equity must match authoritative ending equity")
    return tuple(observations)


def calculate_exposure_statistics(result: BacktestResult) -> ExposureStatistics:
    """Calculate pure observation-based exposure and audited decision statistics.

    Time in market is the fraction of equity observations with a positive long
    quantity. It is not weighted by elapsed calendar time. Zero-position
    observations remain in every observation average.
    """

    observations = derive_exposure_observations(result)
    observation_count = len(observations)
    invested_count = sum(observation.invested for observation in observations)
    decisions = result.pre_trade_decisions
    full_count = sum(
        decision.outcome is PreTradeDecisionOutcome.APPROVED
        for decision in decisions
    )
    reduced_count = sum(
        decision.outcome is PreTradeDecisionOutcome.REDUCED
        for decision in decisions
    )
    rejected_count = sum(
        decision.outcome is PreTradeDecisionOutcome.REJECTED
        for decision in decisions
    )
    proposed_total = sum(
        (Decimal(decision.proposed_quantity) for decision in decisions),
        _ZERO,
    )
    approved_total = sum(
        (Decimal(decision.approved_quantity) for decision in decisions),
        _ZERO,
    )

    if observation_count == 0:
        average_quantity = None
        maximum_quantity = _ZERO
        average_position_value = None
        maximum_position_value = _ZERO
        average_cash = None
        minimum_cash = None
        average_cash_ratio = None
        minimum_cash_ratio = None
        average_utilisation = None
        maximum_utilisation = None
        time_in_market = None
    else:
        count = Decimal(observation_count)
        with localcontext(_new_exposure_context()):
            average_quantity = sum(
                (Decimal(item.position_quantity) for item in observations),
                _ZERO,
            ) / count
            average_position_value = sum(
                (item.position_market_value for item in observations),
                _ZERO,
            ) / count
            average_cash = sum((item.cash for item in observations), _ZERO) / count
            average_cash_ratio = sum(
                (item.cash_ratio for item in observations),
                _ZERO,
            ) / count
            average_utilisation = sum(
                (item.capital_utilisation_ratio for item in observations),
                _ZERO,
            ) / count
            time_in_market = Decimal(invested_count) / count
        maximum_quantity = max(
            Decimal(item.position_quantity) for item in observations
        )
        maximum_position_value = max(
            item.position_market_value for item in observations
        )
        minimum_cash = min(item.cash for item in observations)
        minimum_cash_ratio = min(item.cash_ratio for item in observations)
        maximum_utilisation = max(
            item.capital_utilisation_ratio for item in observations
        )

    actionable_count = len(decisions)
    if actionable_count == 0:
        execution_approval_rate = None
        full_approval_rate = None
        reduction_rate = None
        rejection_rate = None
    else:
        (
            full_approval_rate,
            reduction_rate,
            rejection_rate,
            execution_approval_rate,
        ) = _decision_rates(full_count, reduced_count, actionable_count)

    return ExposureStatistics(
        observation_count=observation_count,
        invested_observation_count=invested_count,
        cash_only_observation_count=observation_count - invested_count,
        time_in_market_ratio=time_in_market,
        average_position_quantity=average_quantity,
        maximum_position_quantity=maximum_quantity,
        average_position_market_value=average_position_value,
        maximum_position_market_value=maximum_position_value,
        average_position_value_ratio=average_utilisation,
        maximum_position_value_ratio=maximum_utilisation,
        average_cash=average_cash,
        minimum_cash=minimum_cash,
        average_cash_ratio=average_cash_ratio,
        minimum_cash_ratio=minimum_cash_ratio,
        average_capital_utilisation_ratio=average_utilisation,
        maximum_capital_utilisation_ratio=maximum_utilisation,
        actionable_signal_count=actionable_count,
        full_approval_count=full_count,
        reduced_decision_count=reduced_count,
        rejected_decision_count=rejected_count,
        proposed_quantity_total=proposed_total,
        approved_quantity_total=approved_total,
        quantity_reduction_total=proposed_total - approved_total,
        execution_approval_rate=execution_approval_rate,
        full_approval_rate=full_approval_rate,
        reduction_rate=reduction_rate,
        rejection_rate=rejection_rate,
    )
