"""Deterministic pre-trade policies for simulated long entries."""

from dataclasses import dataclass
from decimal import Decimal

from trading_research.models import PreTradeDecisionReason
from trading_research.risk.base import RiskDecision, RiskPolicy
from trading_research.simulation import (
    calculate_proportional_commission,
    largest_affordable_quantity,
)

_ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class AllowAllRiskPolicy:
    """Approve every valid positive whole-share proposal unchanged."""

    def evaluate(
        self,
        *,
        proposed_quantity: int,
        available_cash: Decimal,
        reference_price: Decimal,
        estimated_fill_price: Decimal,
        estimated_commission: Decimal,
        commission_bps: Decimal,
        current_position_quantity: int,
    ) -> RiskDecision:
        """Approve a valid proposal without changing its quantity."""

        _validate_evaluation_inputs(
            proposed_quantity=proposed_quantity,
            available_cash=available_cash,
            reference_price=reference_price,
            estimated_fill_price=estimated_fill_price,
            estimated_commission=estimated_commission,
            commission_bps=commission_bps,
            current_position_quantity=current_position_quantity,
        )
        if proposed_quantity == 0:
            return RiskDecision(
                False,
                0,
                "proposed quantity is zero",
                PreTradeDecisionReason.RISK_POLICY_REJECTED,
            )
        return RiskDecision(True, proposed_quantity)


@dataclass(frozen=True, slots=True)
class MaximumPositionValuePolicy:
    """Reduce a buy to keep estimated post-trade position value within a cap."""

    maximum_position_value: Decimal

    def __post_init__(self) -> None:
        """Require a positive finite simulated position-value limit."""

        _require_finite_decimal(self.maximum_position_value, "maximum_position_value")
        if self.maximum_position_value <= _ZERO:
            raise ValueError("maximum_position_value must be positive")

    def evaluate(
        self,
        *,
        proposed_quantity: int,
        available_cash: Decimal,
        reference_price: Decimal,
        estimated_fill_price: Decimal,
        estimated_commission: Decimal,
        commission_bps: Decimal,
        current_position_quantity: int,
    ) -> RiskDecision:
        """Approve or reduce the proposal using its estimated fill price."""

        _validate_evaluation_inputs(
            proposed_quantity=proposed_quantity,
            available_cash=available_cash,
            reference_price=reference_price,
            estimated_fill_price=estimated_fill_price,
            estimated_commission=estimated_commission,
            commission_bps=commission_bps,
            current_position_quantity=current_position_quantity,
        )
        maximum_total_quantity = int(
            self.maximum_position_value // estimated_fill_price
        )
        permitted_quantity = max(maximum_total_quantity - current_position_quantity, 0)
        approved_quantity = min(proposed_quantity, permitted_quantity)
        if approved_quantity == 0:
            return RiskDecision(
                False,
                0,
                "maximum position value leaves no capacity",
                PreTradeDecisionReason.MAXIMUM_POSITION_VALUE,
            )
        if approved_quantity < proposed_quantity:
            return RiskDecision(
                True,
                approved_quantity,
                "quantity reduced by maximum position value policy",
                PreTradeDecisionReason.MAXIMUM_POSITION_VALUE,
            )
        return RiskDecision(True, approved_quantity)


@dataclass(frozen=True, slots=True)
class MinimumCashReservePolicy:
    """Reduce a buy so estimated post-trade cash retains a configured reserve."""

    minimum_cash_reserve: Decimal

    def __post_init__(self) -> None:
        """Require a finite non-negative cash reserve."""

        _require_finite_decimal(self.minimum_cash_reserve, "minimum_cash_reserve")
        if self.minimum_cash_reserve < _ZERO:
            raise ValueError("minimum_cash_reserve must be non-negative")

    def evaluate(
        self,
        *,
        proposed_quantity: int,
        available_cash: Decimal,
        reference_price: Decimal,
        estimated_fill_price: Decimal,
        estimated_commission: Decimal,
        commission_bps: Decimal,
        current_position_quantity: int,
    ) -> RiskDecision:
        """Approve or reduce using fill-price notional plus commission."""

        _validate_evaluation_inputs(
            proposed_quantity=proposed_quantity,
            available_cash=available_cash,
            reference_price=reference_price,
            estimated_fill_price=estimated_fill_price,
            estimated_commission=estimated_commission,
            commission_bps=commission_bps,
            current_position_quantity=current_position_quantity,
        )
        approved_quantity = min(
            proposed_quantity,
            largest_affordable_quantity(
                budget=available_cash - self.minimum_cash_reserve,
                fill_price=estimated_fill_price,
                commission_bps=commission_bps,
            ),
        )
        if approved_quantity == 0:
            return RiskDecision(
                False,
                0,
                "minimum cash reserve leaves no buying power",
                PreTradeDecisionReason.MINIMUM_CASH_RESERVE,
            )
        if approved_quantity < proposed_quantity:
            return RiskDecision(
                True,
                approved_quantity,
                "quantity reduced by minimum cash reserve policy",
                PreTradeDecisionReason.MINIMUM_CASH_RESERVE,
            )
        return RiskDecision(True, approved_quantity)


@dataclass(frozen=True, slots=True)
class CompositeRiskPolicy:
    """Apply policies in tuple order to each prior approved quantity."""

    policies: tuple[RiskPolicy, ...]

    def __post_init__(self) -> None:
        """Require an immutable tuple of policy-like objects."""

        if not isinstance(self.policies, tuple):
            raise TypeError("policies must be a tuple")
        if any(not callable(getattr(policy, "evaluate", None)) for policy in self.policies):
            raise TypeError("policies must provide an evaluate method")

    def evaluate(
        self,
        *,
        proposed_quantity: int,
        available_cash: Decimal,
        reference_price: Decimal,
        estimated_fill_price: Decimal,
        estimated_commission: Decimal,
        commission_bps: Decimal,
        current_position_quantity: int,
    ) -> RiskDecision:
        """Return the final sequential whole-share policy decision."""

        _validate_evaluation_inputs(
            proposed_quantity=proposed_quantity,
            available_cash=available_cash,
            reference_price=reference_price,
            estimated_fill_price=estimated_fill_price,
            estimated_commission=estimated_commission,
            commission_bps=commission_bps,
            current_position_quantity=current_position_quantity,
        )
        if proposed_quantity == 0:
            return RiskDecision(
                False,
                0,
                "proposed quantity is zero",
                PreTradeDecisionReason.RISK_POLICY_REJECTED,
            )

        approved_quantity = proposed_quantity
        reasons: list[str] = []
        reason_codes: list[PreTradeDecisionReason] = []
        for policy in self.policies:
            decision = policy.evaluate(
                proposed_quantity=approved_quantity,
                available_cash=available_cash,
                reference_price=reference_price,
                estimated_fill_price=estimated_fill_price,
                estimated_commission=calculate_proportional_commission(
                    estimated_fill_price,
                    approved_quantity,
                    commission_bps,
                ),
                commission_bps=commission_bps,
                current_position_quantity=current_position_quantity,
            )
            if decision.approved_quantity > approved_quantity:
                raise ValueError("a risk policy cannot increase the proposed quantity")
            if decision.reason is not None:
                reasons.append(decision.reason)
            if (
                decision.reason_code is not None
                and decision.reason_code not in reason_codes
            ):
                reason_codes.append(decision.reason_code)
            approved_quantity = decision.approved_quantity
            if not decision.approved:
                return RiskDecision(
                    False,
                    0,
                    "; ".join(reasons) or "risk policy rejected order",
                    _composite_reason_code(reason_codes),
                )

        return RiskDecision(
            True,
            approved_quantity,
            "; ".join(reasons) or None,
            _composite_reason_code(reason_codes),
        )


def _composite_reason_code(
    reason_codes: list[PreTradeDecisionReason],
) -> PreTradeDecisionReason | None:
    """Collapse ordered policy codes without inspecting human-readable prose."""

    if len(reason_codes) > 1:
        return PreTradeDecisionReason.MULTIPLE_RISK_POLICIES
    if reason_codes:
        return reason_codes[0]
    return None


def _validate_evaluation_inputs(
    *,
    proposed_quantity: int,
    available_cash: Decimal,
    reference_price: Decimal,
    estimated_fill_price: Decimal,
    estimated_commission: Decimal,
    commission_bps: Decimal,
    current_position_quantity: int,
) -> None:
    """Validate exact proposal context and commission consistency."""

    quantities = (
        ("proposed_quantity", proposed_quantity),
        ("current_position_quantity", current_position_quantity),
    )
    for name, quantity in quantities:
        if isinstance(quantity, bool) or not isinstance(quantity, int):
            raise TypeError(f"{name} must be an integer")
        if quantity < 0:
            raise ValueError(f"{name} must be non-negative")

    financial_values = (
        ("available_cash", available_cash),
        ("reference_price", reference_price),
        ("estimated_fill_price", estimated_fill_price),
        ("estimated_commission", estimated_commission),
        ("commission_bps", commission_bps),
    )
    for name, financial_value in financial_values:
        _require_finite_decimal(financial_value, name)
    if available_cash < _ZERO:
        raise ValueError("available_cash must be non-negative")
    if reference_price <= _ZERO or estimated_fill_price <= _ZERO:
        raise ValueError("reference and estimated fill prices must be positive")
    if estimated_commission < _ZERO or commission_bps < _ZERO:
        raise ValueError("commission inputs must be non-negative")
    expected_commission = calculate_proportional_commission(
        estimated_fill_price,
        proposed_quantity,
        commission_bps,
    )
    if estimated_commission != expected_commission:
        raise ValueError("estimated_commission must match the proposed quantity")


def _require_finite_decimal(value: Decimal, name: str) -> None:
    """Require exact finite Decimal policy configuration and inputs."""

    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")
