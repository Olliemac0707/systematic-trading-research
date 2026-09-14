"""Tests for exact position sizing and deterministic pre-trade risk policies."""

from dataclasses import FrozenInstanceError
from decimal import Decimal
from typing import Never

import pytest

from trading_research.models import PreTradeDecisionReason
from trading_research.risk import (
    AllowAllRiskPolicy,
    CashAllocationSizer,
    CompositeRiskPolicy,
    FixedQuantitySizer,
    MaximumPositionValuePolicy,
    MinimumCashReservePolicy,
    RiskDecision,
    RiskPolicy,
)
from trading_research.simulation import (
    calculate_buy_fill_price,
    calculate_proportional_commission,
)


def calculate_quantity(
    sizer: FixedQuantitySizer | CashAllocationSizer,
    *,
    available_cash: Decimal = Decimal("1000"),
    reference_price: Decimal = Decimal("100"),
    commission_bps: Decimal = Decimal("0"),
    slippage_bps: Decimal = Decimal("0"),
) -> int:
    """Call a position sizer with exact current-state inputs."""

    return sizer.calculate_quantity(
        available_cash=available_cash,
        reference_price=reference_price,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )


def evaluate_policy(
    policy: RiskPolicy,
    *,
    proposed_quantity: int = 10,
    available_cash: Decimal = Decimal("1000"),
    reference_price: Decimal = Decimal("100"),
    slippage_bps: Decimal = Decimal("0"),
    commission_bps: Decimal = Decimal("0"),
    current_position_quantity: int = 0,
) -> RiskDecision:
    """Evaluate one policy with internally consistent exact estimates."""

    fill_price = calculate_buy_fill_price(reference_price, slippage_bps)
    commission = calculate_proportional_commission(
        fill_price,
        proposed_quantity,
        commission_bps,
    )
    return policy.evaluate(
        proposed_quantity=proposed_quantity,
        available_cash=available_cash,
        reference_price=reference_price,
        estimated_fill_price=fill_price,
        estimated_commission=commission,
        commission_bps=commission_bps,
        current_position_quantity=current_position_quantity,
    )


def test_fixed_quantity_sizer_preserves_whole_share_quantity() -> None:
    """Fixed sizing returns the configured quantity without financial mutation."""

    sizer = FixedQuantitySizer(10)

    assert calculate_quantity(sizer) == 10


@pytest.mark.parametrize("quantity", [0, -1, 1.5, True])
def test_fixed_quantity_sizer_rejects_invalid_quantities(quantity: object) -> None:
    """Zero, negative, fractional, and boolean quantities are unsupported."""

    expected_error = TypeError if isinstance(quantity, (float, bool)) else ValueError
    with pytest.raises(expected_error):
        FixedQuantitySizer(quantity)  # type: ignore[arg-type]


def test_cash_allocation_rounds_down_and_includes_both_costs() -> None:
    """Sizing uses adverse fill plus commission before whole-share flooring."""

    quantity = calculate_quantity(
        CashAllocationSizer(Decimal("0.25")),
        available_cash=Decimal("1000"),
        reference_price=Decimal("100"),
        commission_bps=Decimal("100"),
        slippage_bps=Decimal("100"),
    )

    assert quantity == 2
    assert Decimal("101") * quantity + Decimal("2.02") <= Decimal("250")


def test_cash_allocation_commission_can_prevent_one_share() -> None:
    """Commission is included rather than checked after selecting quantity."""

    quantity = calculate_quantity(
        CashAllocationSizer(Decimal("1")),
        available_cash=Decimal("100"),
        reference_price=Decimal("100"),
        commission_bps=Decimal("100"),
    )

    assert quantity == 0


def test_cash_allocation_slippage_can_prevent_one_share() -> None:
    """Adverse buy slippage is included in the affordability estimate."""

    quantity = calculate_quantity(
        CashAllocationSizer(Decimal("1")),
        available_cash=Decimal("100"),
        reference_price=Decimal("100"),
        slippage_bps=Decimal("100"),
    )

    assert quantity == 0


def test_full_cash_allocation_uses_largest_affordable_whole_quantity() -> None:
    """A ratio of one may use all cash but never create a fractional share."""

    assert calculate_quantity(CashAllocationSizer(Decimal("1"))) == 10
    assert calculate_quantity(
        CashAllocationSizer(Decimal("0.01")),
        available_cash=Decimal("50"),
    ) == 0


@pytest.mark.parametrize(
    "ratio",
    [Decimal("0"), Decimal("-0.1"), Decimal("1.01"), Decimal("NaN"), Decimal("Infinity")],
)
def test_cash_allocation_rejects_invalid_ratios(ratio: Decimal) -> None:
    """Allocation ratios must be exact, finite, positive, and no greater than one."""

    with pytest.raises(ValueError):
        CashAllocationSizer(ratio)


def test_cash_allocation_does_not_convert_through_float(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All sizing arithmetic remains Decimal-only."""

    def reject_float(*_: object, **__: object) -> Never:
        raise AssertionError("float conversion is forbidden")

    monkeypatch.setattr("builtins.float", reject_float)

    assert calculate_quantity(
        CashAllocationSizer(Decimal("0.333")),
        reference_price=Decimal("7.25"),
        commission_bps=Decimal("7.5"),
        slippage_bps=Decimal("2.5"),
    ) == 45


def test_allow_all_policy_approves_valid_quantity_unchanged() -> None:
    """The default risk policy preserves prior fixed-size behaviour."""

    decision = evaluate_policy(AllowAllRiskPolicy(), proposed_quantity=7)

    assert decision == RiskDecision(True, 7)


def test_maximum_position_value_approves_or_reduces_using_fill_price() -> None:
    """The cap uses estimated adverse fill value and whole-share reduction."""

    unchanged = evaluate_policy(
        MaximumPositionValuePolicy(Decimal("500")),
        proposed_quantity=4,
    )
    reduced = evaluate_policy(
        MaximumPositionValuePolicy(Decimal("1000")),
        proposed_quantity=10,
        slippage_bps=Decimal("1000"),
    )

    assert unchanged == RiskDecision(True, 4)
    assert reduced.approved_quantity == 9
    assert reduced.reason == "quantity reduced by maximum position value policy"
    assert reduced.reason_code is PreTradeDecisionReason.MAXIMUM_POSITION_VALUE
    assert Decimal(reduced.approved_quantity) * Decimal("110") <= Decimal("1000")


def test_maximum_position_value_rejects_when_existing_position_meets_limit() -> None:
    """No additional shares are approved after existing value fills the cap."""

    decision = evaluate_policy(
        MaximumPositionValuePolicy(Decimal("500")),
        proposed_quantity=3,
        current_position_quantity=5,
    )

    assert decision.approved is False
    assert decision.approved_quantity == 0
    assert decision.reason_code is PreTradeDecisionReason.MAXIMUM_POSITION_VALUE


@pytest.mark.parametrize(
    "maximum",
    [Decimal("0"), Decimal("-1"), Decimal("NaN"), Decimal("Infinity")],
)
def test_maximum_position_value_rejects_invalid_configuration(
    maximum: Decimal,
) -> None:
    """The simulated position-value cap must be positive and finite."""

    with pytest.raises(ValueError):
        MaximumPositionValuePolicy(maximum)


def test_minimum_cash_reserve_approves_reduces_and_handles_exact_boundary() -> None:
    """Reserve sizing includes notional and proportional commission exactly."""

    unchanged = evaluate_policy(
        MinimumCashReservePolicy(Decimal("200")),
        proposed_quantity=5,
    )
    reduced = evaluate_policy(
        MinimumCashReservePolicy(Decimal("200")),
        proposed_quantity=10,
    )
    boundary = evaluate_policy(
        MinimumCashReservePolicy(Decimal("490")),
        proposed_quantity=5,
        commission_bps=Decimal("200"),
    )

    assert unchanged == RiskDecision(True, 5)
    assert reduced.approved_quantity == 8
    assert reduced.reason_code is PreTradeDecisionReason.MINIMUM_CASH_RESERVE
    assert boundary == RiskDecision(True, 5)


def test_minimum_cash_reserve_includes_commission_and_adverse_slippage() -> None:
    """The reserve is checked against the same estimated fill and commission."""

    decision = evaluate_policy(
        MinimumCashReservePolicy(Decimal("0")),
        proposed_quantity=10,
        slippage_bps=Decimal("100"),
        commission_bps=Decimal("100"),
    )

    assert decision.approved_quantity == 9


@pytest.mark.parametrize(
    "reserve",
    [Decimal("-1"), Decimal("NaN"), Decimal("Infinity")],
)
def test_minimum_cash_reserve_rejects_invalid_configuration(reserve: Decimal) -> None:
    """A reserve may be zero but must be finite and non-negative."""

    with pytest.raises(ValueError):
        MinimumCashReservePolicy(reserve)


def test_composite_applies_position_cap_then_cash_reserve() -> None:
    """Each later policy receives the quantity approved by the prior policy."""

    policy = CompositeRiskPolicy(
        (
            MaximumPositionValuePolicy(Decimal("600")),
            MinimumCashReservePolicy(Decimal("500")),
        )
    )

    decision = evaluate_policy(policy, proposed_quantity=10)

    assert decision.approved_quantity == 5
    assert decision.reason == (
        "quantity reduced by maximum position value policy; "
        "quantity reduced by minimum cash reserve policy"
    )
    assert decision.reason_code is PreTradeDecisionReason.MULTIPLE_RISK_POLICIES


class _ReducingPolicy:
    """Test policy that records the proposal received from a composite."""

    def __init__(self, approved_quantity: int, seen: list[int]) -> None:
        self._approved_quantity = approved_quantity
        self._seen = seen

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
        """Record and reduce without using unrelated context fields."""

        self._seen.append(proposed_quantity)
        approved_quantity = min(proposed_quantity, self._approved_quantity)
        if approved_quantity == 0:
            return RiskDecision(False, 0, "test rejection")
        return RiskDecision(True, approved_quantity)


def test_composite_passes_reduced_quantity_and_stops_at_zero() -> None:
    """Sequential evaluation is ordered and rejection stops later policies."""

    seen: list[int] = []
    policy = CompositeRiskPolicy(
        (
            _ReducingPolicy(4, seen),
            _ReducingPolicy(0, seen),
            _ReducingPolicy(2, seen),
        )
    )

    decision = evaluate_policy(policy, proposed_quantity=10)

    assert seen == [10, 4]
    assert decision == RiskDecision(False, 0, "test rejection")


def test_risk_models_are_immutable() -> None:
    """Validated decisions and policy configurations cannot be mutated."""

    decision = RiskDecision(True, 1)
    policy = MinimumCashReservePolicy(Decimal("100"))
    decision_attribute = "approved_quantity"
    policy_attribute = "minimum_cash_reserve"

    with pytest.raises(FrozenInstanceError):
        setattr(decision, decision_attribute, 2)
    with pytest.raises(FrozenInstanceError):
        setattr(policy, policy_attribute, Decimal("0"))
