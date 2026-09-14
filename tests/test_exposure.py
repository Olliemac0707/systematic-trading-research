"""Tests for audited entry decisions and observation-based exposure statistics."""

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from typing import Never

import pytest

from trading_research.backtesting import SimpleBacktestEngine
from trading_research.config import BacktestConfig, EndOfTestPolicy
from trading_research.models import (
    BacktestResult,
    EquityPoint,
    MarketBar,
    PreTradeDecision,
    PreTradeDecisionOutcome,
    PreTradeDecisionReason,
    SignalAction,
    SimulatedTrade,
    StrategySignal,
    TradeSide,
)
from trading_research.performance import (
    ExposureObservation,
    ExposureStatistics,
    calculate_exposure_statistics,
    derive_exposure_observations,
)
from trading_research.reporting import format_backtest_report
from trading_research.risk import RiskDecision
from trading_research.strategies import Strategy

_START = datetime(2025, 1, 1, tzinfo=UTC)
_ZERO = Decimal("0")


class FixedSignals(Strategy):
    """Return an explicit immutable strategy-signal sequence."""

    def __init__(self, signals: Sequence[StrategySignal]) -> None:
        self._signals = tuple(signals)

    def generate_signals(self, bars: Sequence[MarketBar]) -> Sequence[StrategySignal]:
        """Return the configured signals without changing market data."""

        return self._signals


class ZeroSizer:
    """Return an explicit zero proposal for the entry audit boundary."""

    def calculate_quantity(self, **_: object) -> int:
        """Return zero without performing a risk calculation."""

        return 0


class RejectRiskEvaluation:
    """Fail if a zero-sized proposal reaches the risk-policy boundary."""

    def evaluate(self, **_: object) -> RiskDecision:
        """Reject an unexpected call from the engine."""

        raise AssertionError("risk policy must not evaluate a zero proposal")


def bars(prices: Sequence[str]) -> tuple[MarketBar, ...]:
    """Create deterministic daily bars with identical OHLC prices."""

    return tuple(
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
    )


def entry_result(
    config: BacktestConfig,
    *,
    position_sizer: object | None = None,
    risk_policy: object | None = None,
) -> BacktestResult:
    """Run one actionable buy intent at the following bar's open."""

    market_bars = bars(("10", "13", "13"))
    strategy = FixedSignals(
        (
            StrategySignal(
                symbol="TEST",
                timestamp=market_bars[0].timestamp,
                action=SignalAction.BUY,
            ),
        )
    )
    return SimpleBacktestEngine().run(
        market_bars,
        strategy,
        config,
        position_sizer=position_sizer,  # type: ignore[arg-type]
        risk_policy=risk_policy,  # type: ignore[arg-type]
    )


def decision(
    offset: int,
    proposed: int,
    approved: int,
    outcome: PreTradeDecisionOutcome,
    reason: PreTradeDecisionReason | None,
) -> PreTradeDecision:
    """Create one exact zero-cost audit record for aggregate tests."""

    return PreTradeDecision(
        timestamp=_START + timedelta(days=offset),
        symbol="TEST",
        side=TradeSide.BUY,
        reference_price=Decimal("10"),
        proposed_quantity=proposed,
        approved_quantity=approved,
        estimated_fill_price=Decimal("10"),
        estimated_commission=_ZERO,
        estimated_total_cost=Decimal(10 * proposed),
        available_cash_before=Decimal("1000"),
        position_quantity_before=0,
        outcome=outcome,
        reason=reason,
    )


def execution(offset: int, side: TradeSide, quantity: int, price: str) -> SimulatedTrade:
    """Create one exact strategy execution for observation tests."""

    return SimulatedTrade(
        symbol="TEST",
        timestamp=_START + timedelta(days=offset),
        side=side,
        quantity=quantity,
        price=Decimal(price),
        reference_price=Decimal(price),
    )


def mixed_result() -> BacktestResult:
    """Create full, reduced, and rejected decisions over alternating exposure."""

    trades = (
        execution(0, TradeSide.BUY, 10, "10"),
        execution(1, TradeSide.SELL, 10, "10"),
        execution(2, TradeSide.BUY, 5, "10"),
        execution(3, TradeSide.SELL, 5, "10"),
    )
    decisions = (
        decision(0, 10, 10, PreTradeDecisionOutcome.APPROVED, None),
        decision(
            2,
            10,
            5,
            PreTradeDecisionOutcome.REDUCED,
            PreTradeDecisionReason.MAXIMUM_POSITION_VALUE,
        ),
        decision(
            4,
            7,
            0,
            PreTradeDecisionOutcome.REJECTED,
            PreTradeDecisionReason.MINIMUM_CASH_RESERVE,
        ),
    )
    return BacktestResult(
        initial_cash=Decimal("1000"),
        final_cash=Decimal("1000"),
        final_equity=Decimal("1000"),
        final_position_quantity=0,
        trades=trades,
        equity_curve=tuple(
            EquityPoint(_START + timedelta(days=index), Decimal("1000"))
            for index in range(5)
        ),
        pre_trade_decisions=decisions,
    )


def report_value(report: str, label: str) -> str:
    """Return one value from a labelled plain-text report line."""

    prefix = f"{label}:"
    return next(
        line[len(prefix) :].strip()
        for line in report.splitlines()
        if line.startswith(prefix)
    )


def test_fully_approved_decision_retains_exact_execution_inputs() -> None:
    """A full fixed proposal records the same fill and commission used by the trade."""

    result = entry_result(
        BacktestConfig(
            initial_cash=Decimal("1000"),
            trade_quantity=10,
            commission_bps=Decimal("100"),
            slippage_bps=Decimal("100"),
        )
    )
    audit = result.pre_trade_decisions[0]
    trade = result.trades[0]

    assert audit.outcome is PreTradeDecisionOutcome.APPROVED
    assert audit.outcome.value == "approved"
    assert audit.reason is None
    assert audit.proposed_quantity == audit.approved_quantity == trade.quantity == 10
    assert audit.reference_price == trade.reference_price == Decimal("13")
    assert audit.estimated_fill_price == trade.price == Decimal("13.13")
    assert audit.estimated_commission == trade.commission == Decimal("1.3130")
    assert audit.estimated_total_cost == Decimal("132.6130")
    assert audit.available_cash_before == Decimal("1000")
    assert audit.position_quantity_before == 0
    assert result.final_cash == Decimal("867.3870")
    assert result.final_equity == Decimal("997.3870")
    assert result.reconciliation.is_reconciled


def test_existing_backtest_result_positional_arguments_remain_compatible() -> None:
    """Adding audit data does not displace the existing final-mark argument."""

    trade = execution(0, TradeSide.BUY, 2, "100")
    result = BacktestResult(
        Decimal("1000"),
        Decimal("800"),
        Decimal("1000"),
        2,
        (trade,),
        (EquityPoint(_START, Decimal("1000")),),
        Decimal("100"),
    )

    assert result.final_position_mark_price == Decimal("100")
    assert result.pre_trade_decisions == ()
    assert result.reconciliation.is_reconciled


def test_exposure_cash_ratio_is_exact_residual_for_precise_components() -> None:
    """Rounded component ratios retain an exact partition of account equity."""

    final_cash = Decimal("12345.0000000000000000000007919")
    position_value = Decimal("67890.0000000000000000000104729")
    equity = final_cash + position_value
    with localcontext(Context(prec=34, rounding=ROUND_HALF_EVEN)):
        utilisation = position_value / equity
        cash_ratio = Decimal("1") - utilisation
    observation = ExposureObservation(
        timestamp=_START,
        cash=final_cash,
        position_quantity=1,
        position_market_value=position_value,
        equity=equity,
        capital_utilisation_ratio=utilisation,
        cash_ratio=cash_ratio,
        invested=True,
    )

    assert (
        observation.capital_utilisation_ratio + observation.cash_ratio
        == Decimal("1")
    )
    with localcontext(Context(prec=34, rounding=ROUND_HALF_EVEN)):
        assert observation.cash_ratio == (
            Decimal("1") - observation.capital_utilisation_ratio
        )


def test_reduced_decision_matches_the_only_executed_quantity() -> None:
    """A position cap reduces an existing proposal without a second calculation."""

    result = entry_result(
        BacktestConfig(
            initial_cash=Decimal("1000"),
            trade_quantity=10,
            maximum_position_value=Decimal("65"),
        )
    )
    audit = result.pre_trade_decisions[0]

    assert audit.outcome is PreTradeDecisionOutcome.REDUCED
    assert audit.outcome.value == "reduced"
    assert audit.reason is PreTradeDecisionReason.MAXIMUM_POSITION_VALUE
    assert audit.proposed_quantity == 10
    assert audit.approved_quantity == audit.quantity_reduction == 5
    assert result.trades[0].quantity == audit.approved_quantity
    assert result.reconciliation.ledger.executed_trades == result.trades


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        (
            BacktestConfig(
                initial_cash=Decimal("1000"),
                trade_quantity=10,
                maximum_position_value=Decimal("10"),
            ),
            PreTradeDecisionReason.MAXIMUM_POSITION_VALUE,
        ),
        (
            BacktestConfig(
                initial_cash=Decimal("1000"),
                trade_quantity=10,
                minimum_cash_reserve=Decimal("1000"),
            ),
            PreTradeDecisionReason.MINIMUM_CASH_RESERVE,
        ),
        (
            BacktestConfig(initial_cash=Decimal("10"), trade_quantity=1),
            PreTradeDecisionReason.INSUFFICIENT_CASH,
        ),
    ],
    ids=["maximum-position", "minimum-cash", "affordability"],
)
def test_rejected_entry_reasons_are_stable_and_do_not_change_the_account(
    config: BacktestConfig,
    reason: PreTradeDecisionReason,
) -> None:
    """Every rejection branch records a stable reason but no trade or ledger entry."""

    result = entry_result(config)
    audit = result.pre_trade_decisions[0]

    assert audit.outcome is PreTradeDecisionOutcome.REJECTED
    assert audit.outcome.value == "rejected"
    assert audit.reason is reason
    assert audit.reason.value in {
        "maximum_position_value",
        "minimum_cash_reserve",
        "insufficient_cash",
    }
    assert audit.approved_quantity == 0
    assert result.trades == ()
    assert result.reconciliation.ledger.entries == ()
    assert result.final_cash == result.final_equity == config.initial_cash


def test_zero_sizer_is_audited_without_calling_risk_or_creating_a_trade() -> None:
    """A zero proposal is actionable intent with an explicit sizing outcome."""

    result = entry_result(
        BacktestConfig(initial_cash=Decimal("1000"), trade_quantity=1),
        position_sizer=ZeroSizer(),
        risk_policy=RejectRiskEvaluation(),
    )
    audit = result.pre_trade_decisions[0]

    assert audit.proposed_quantity == audit.approved_quantity == 0
    assert audit.estimated_commission == audit.estimated_total_cost == _ZERO
    assert audit.outcome is PreTradeDecisionOutcome.REJECTED
    assert audit.reason is PreTradeDecisionReason.POSITION_SIZER_RETURNED_ZERO
    assert result.trades == ()


@pytest.mark.parametrize(
    (
        "config",
        "expected_counts",
        "expected_rates",
        "expected_quantities",
    ),
    [
        (
            BacktestConfig(initial_cash=Decimal("1000"), trade_quantity=10),
            (1, 0, 0),
            (Decimal("1"), Decimal("1"), _ZERO, _ZERO),
            (Decimal("10"), Decimal("10"), _ZERO),
        ),
        (
            BacktestConfig(
                initial_cash=Decimal("1000"),
                trade_quantity=10,
                maximum_position_value=Decimal("65"),
            ),
            (0, 1, 0),
            (Decimal("1"), _ZERO, Decimal("1"), _ZERO),
            (Decimal("10"), Decimal("5"), Decimal("5")),
        ),
        (
            BacktestConfig(
                initial_cash=Decimal("1000"),
                trade_quantity=10,
                maximum_position_value=Decimal("10"),
            ),
            (0, 0, 1),
            (_ZERO, _ZERO, _ZERO, Decimal("1")),
            (Decimal("10"), _ZERO, Decimal("10")),
        ),
    ],
    ids=["all-full", "all-reduced", "all-rejected"],
)
def test_single_outcome_decision_statistics_have_unambiguous_rates(
    config: BacktestConfig,
    expected_counts: tuple[int, int, int],
    expected_rates: tuple[Decimal, Decimal, Decimal, Decimal],
    expected_quantities: tuple[Decimal, Decimal, Decimal],
) -> None:
    """Homogeneous outcomes define full, reduction, rejection, and execution rates."""

    statistics = calculate_exposure_statistics(entry_result(config))

    assert statistics.actionable_signal_count == 1
    assert (
        statistics.full_approval_count,
        statistics.reduced_decision_count,
        statistics.rejected_decision_count,
    ) == expected_counts
    assert (
        statistics.execution_approval_rate,
        statistics.full_approval_rate,
        statistics.reduction_rate,
        statistics.rejection_rate,
    ) == expected_rates
    assert (
        statistics.proposed_quantity_total,
        statistics.approved_quantity_total,
        statistics.quantity_reduction_total,
    ) == expected_quantities


def test_decisions_are_chronological_and_non_actionable_bars_add_no_records() -> None:
    """Only evaluated entry proposals appear in strict execution-time order."""

    market_bars = bars(("10", "11", "12", "13"))
    signals = (
        StrategySignal("TEST", market_bars[0].timestamp, SignalAction.BUY),
        StrategySignal("TEST", market_bars[1].timestamp, SignalAction.SELL),
        StrategySignal("TEST", market_bars[2].timestamp, SignalAction.BUY),
    )
    result = SimpleBacktestEngine().run(
        market_bars,
        FixedSignals(signals),
        BacktestConfig(initial_cash=Decimal("1000"), trade_quantity=2),
    )

    assert [item.timestamp for item in result.pre_trade_decisions] == [
        market_bars[1].timestamp,
        market_bars[3].timestamp,
    ]
    assert all(
        item.timestamp.tzinfo is not None and item.timestamp.utcoffset() is not None
        for item in result.pre_trade_decisions
    )
    assert len(result.pre_trade_decisions) == 2
    assert len(result.trades) == 3

    inactive = SimpleBacktestEngine().run(
        market_bars,
        FixedSignals(()),
        BacktestConfig(initial_cash=Decimal("1000"), trade_quantity=2),
    )
    assert inactive.pre_trade_decisions == ()


def test_synthetic_liquidation_is_not_an_entry_proposal() -> None:
    """The terminal sell changes exposure but does not fabricate a new decision."""

    result = entry_result(
        BacktestConfig(
            initial_cash=Decimal("1000"),
            trade_quantity=2,
            end_of_test_policy=EndOfTestPolicy.LIQUIDATE,
        )
    )
    statistics = calculate_exposure_statistics(result)

    assert len(result.pre_trade_decisions) == 1
    assert len(result.trades) == 2
    assert result.was_end_of_test_liquidated
    assert result.final_position_quantity == 0
    assert statistics.invested_observation_count == 1
    assert statistics.cash_only_observation_count == 2
    with localcontext(Context(prec=34, rounding=ROUND_HALF_EVEN)):
        expected_time_in_market = Decimal(1) / Decimal(3)
    assert statistics.time_in_market_ratio == expected_time_in_market
    assert derive_exposure_observations(result)[-1].position_quantity == 0


def test_decision_model_is_immutable_aware_and_rejects_inconsistent_records() -> None:
    """The audit model rejects ambiguity before it reaches reports or exports."""

    audit = decision(0, 10, 10, PreTradeDecisionOutcome.APPROVED, None)

    with pytest.raises(FrozenInstanceError):
        audit.approved_quantity = 5  # type: ignore[misc]
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(audit, timestamp=datetime(2025, 1, 1))
    with pytest.raises(ValueError, match="outcome is inconsistent"):
        replace(audit, outcome=PreTradeDecisionOutcome.REDUCED)
    with pytest.raises(TypeError, match="TradeSide"):
        replace(audit, side="buy")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reason detail"):
        replace(audit, reason_detail="unexpected")
    with pytest.raises(TypeError, match="stable reason"):
        replace(
            audit,
            approved_quantity=0,
            outcome=PreTradeDecisionOutcome.REJECTED,
        )
    with pytest.raises(ValueError, match="estimated_total_cost"):
        replace(audit, estimated_commission=Decimal("1"))


def test_alternating_exposure_and_mixed_decisions_use_exact_observation_counts() -> None:
    """Zero-position observations remain in averages and every rate denominator."""

    result = mixed_result()
    observations = derive_exposure_observations(result)
    statistics = calculate_exposure_statistics(result)

    assert [item.position_quantity for item in observations] == [10, 0, 5, 0, 0]
    assert [item.cash for item in observations] == [
        Decimal("900"),
        Decimal("1000"),
        Decimal("950"),
        Decimal("1000"),
        Decimal("1000"),
    ]
    assert [item.position_market_value for item in observations] == [
        Decimal("100"),
        _ZERO,
        Decimal("50"),
        _ZERO,
        _ZERO,
    ]
    assert [item.invested for item in observations] == [True, False, True, False, False]
    assert statistics.observation_count == 5
    assert statistics.invested_observation_count == 2
    assert statistics.cash_only_observation_count == 3
    assert statistics.time_in_market_ratio == Decimal("0.4")
    assert statistics.average_position_quantity == Decimal("3")
    assert statistics.maximum_position_quantity == Decimal("10")
    assert statistics.average_position_market_value == Decimal("30")
    assert statistics.maximum_position_market_value == Decimal("100")
    assert statistics.average_cash == Decimal("970")
    assert statistics.minimum_cash == Decimal("900")
    assert statistics.average_cash_ratio == Decimal("0.97")
    assert statistics.minimum_cash_ratio == Decimal("0.9")
    assert statistics.average_capital_utilisation_ratio == Decimal("0.03")
    assert statistics.maximum_capital_utilisation_ratio == Decimal("0.1")
    assert statistics.average_position_value_ratio == Decimal("0.03")
    assert statistics.maximum_position_value_ratio == Decimal("0.1")
    assert statistics.actionable_signal_count == 3
    assert statistics.full_approval_count == 1
    assert statistics.reduced_decision_count == 1
    assert statistics.rejected_decision_count == 1
    assert statistics.proposed_quantity_total == Decimal("27")
    assert statistics.approved_quantity_total == Decimal("15")
    assert statistics.quantity_reduction_total == Decimal("12")
    with localcontext(Context(prec=34, rounding=ROUND_HALF_EVEN)):
        one_third = Decimal(1) / Decimal(3)
        two_thirds = one_third + one_third
        residual_third = Decimal(1) - two_thirds
    assert statistics.execution_approval_rate == two_thirds
    assert statistics.full_approval_rate == one_third
    assert statistics.reduction_rate == one_third
    assert statistics.rejection_rate == residual_third
    assert (
        statistics.full_approval_rate
        + statistics.reduction_rate
        + statistics.rejection_rate
        == Decimal("1")
    )


def test_text_report_formats_exposure_and_decisions_without_recalculation() -> None:
    """Human output labels observation policy and separates intent from execution."""

    report = format_backtest_report(mixed_result())

    assert "Exposure and Capital Use" in report
    assert "Order Decisions" in report
    assert report_value(report, "Equity observations") == "5"
    assert report_value(report, "Time in market (observations)") == "40.00%"
    assert report_value(report, "Average position quantity") == "3.00"
    assert report_value(report, "Average position value") == "30.00"
    assert report_value(report, "Average capital utilisation") == "3.00%"
    assert report_value(report, "Maximum capital utilisation") == "10.00%"
    assert report_value(report, "Average cash") == "970.00"
    assert report_value(report, "Minimum cash") == "900.00"
    assert report_value(report, "Actionable proposals") == "3"
    assert report_value(report, "Fully approved") == "1"
    assert report_value(report, "Reduced") == "1"
    assert report_value(report, "Rejected") == "1"
    assert report_value(report, "Execution approval rate") == "66.67%"
    assert report_value(report, "Full approval rate") == "33.33%"
    assert report_value(report, "Reduction rate") == "33.33%"
    assert report_value(report, "Rejection rate") == "33.33%"
    assert report_value(report, "Proposed quantity") == "27.00"
    assert report_value(report, "Approved quantity") == "15.00"
    assert report_value(report, "Quantity prevented or reduced") == "12.00"
    assert "Time in market is observation-based" in report


def test_always_cash_has_zero_exposure_and_no_decision_rates() -> None:
    """An inactive account includes every cash-only observation in its averages."""

    market_bars = bars(("10", "11", "12"))
    result = SimpleBacktestEngine().run(
        market_bars,
        FixedSignals(()),
        BacktestConfig(initial_cash=Decimal("1000"), trade_quantity=2),
    )
    statistics = calculate_exposure_statistics(result)

    assert statistics.observation_count == 3
    assert statistics.invested_observation_count == 0
    assert statistics.cash_only_observation_count == 3
    assert statistics.time_in_market_ratio == _ZERO
    assert statistics.average_position_quantity == _ZERO
    assert statistics.maximum_position_quantity == _ZERO
    assert statistics.average_position_market_value == _ZERO
    assert statistics.maximum_position_market_value == _ZERO
    assert statistics.average_cash == statistics.minimum_cash == Decimal("1000")
    assert statistics.average_cash_ratio == statistics.minimum_cash_ratio == Decimal("1")
    assert statistics.average_capital_utilisation_ratio == _ZERO
    assert statistics.maximum_capital_utilisation_ratio == _ZERO
    assert statistics.actionable_signal_count == 0
    assert statistics.execution_approval_rate is None
    assert statistics.full_approval_rate is None
    assert statistics.reduction_rate is None
    assert statistics.rejection_rate is None


def test_always_invested_hold_result_uses_final_open_position_value() -> None:
    """A held long contributes to every observation and the final equity composition."""

    trade = execution(0, TradeSide.BUY, 2, "100")
    result = BacktestResult(
        initial_cash=Decimal("1000"),
        final_cash=Decimal("800"),
        final_equity=Decimal("1020"),
        final_position_quantity=2,
        final_position_mark_price=Decimal("110"),
        trades=(trade,),
        equity_curve=(
            EquityPoint(_START, Decimal("1000")),
            EquityPoint(_START + timedelta(days=1), Decimal("1020")),
        ),
    )
    observations = derive_exposure_observations(result)
    statistics = calculate_exposure_statistics(result)

    assert [item.position_market_value for item in observations] == [
        Decimal("200"),
        Decimal("220"),
    ]
    assert statistics.time_in_market_ratio == Decimal("1")
    assert statistics.average_position_quantity == Decimal("2")
    assert statistics.maximum_position_quantity == Decimal("2")
    assert statistics.average_position_market_value == Decimal("210")
    assert statistics.maximum_position_market_value == Decimal("220")
    assert statistics.average_cash == statistics.minimum_cash == Decimal("800")
    assert observations[-1].cash + observations[-1].position_market_value == (
        result.final_equity
    )


def test_empty_observation_statistics_require_none_averages() -> None:
    """The immutable summary has an explicit empty-observation representation."""

    market_bars = bars(("10",))
    result = SimpleBacktestEngine().run(
        market_bars,
        FixedSignals(()),
        BacktestConfig(initial_cash=Decimal("1000"), trade_quantity=2),
    )
    statistics = replace(
        calculate_exposure_statistics(result),
        observation_count=0,
        invested_observation_count=0,
        cash_only_observation_count=0,
        time_in_market_ratio=None,
        average_position_quantity=None,
        maximum_position_quantity=_ZERO,
        average_position_market_value=None,
        maximum_position_market_value=_ZERO,
        average_position_value_ratio=None,
        maximum_position_value_ratio=None,
        average_cash=None,
        minimum_cash=None,
        average_cash_ratio=None,
        minimum_cash_ratio=None,
        average_capital_utilisation_ratio=None,
        maximum_capital_utilisation_ratio=None,
    )

    assert statistics.observation_count == 0
    assert statistics.time_in_market_ratio is None
    assert statistics.average_position_quantity is None
    assert statistics.average_position_market_value is None
    assert statistics.average_cash is None
    assert statistics.minimum_cash is None
    assert statistics.average_cash_ratio is None
    assert statistics.minimum_cash_ratio is None
    assert statistics.average_capital_utilisation_ratio is None
    assert statistics.maximum_capital_utilisation_ratio is None
    assert statistics.maximum_position_quantity == _ZERO
    assert statistics.maximum_position_market_value == _ZERO


def test_calculation_is_deterministic_immutable_and_never_uses_float(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated pure calculations preserve inputs and avoid binary conversion."""

    result = mixed_result()
    before = repr(result)

    def reject_float(*_: object, **__: object) -> Never:
        raise AssertionError("float conversion is forbidden")

    monkeypatch.setattr("builtins.float", reject_float)
    first = calculate_exposure_statistics(result)
    second = calculate_exposure_statistics(result)

    assert first == second
    assert isinstance(first, ExposureStatistics)
    assert repr(result) == before
    with pytest.raises(FrozenInstanceError):
        first.observation_count = 0  # type: ignore[misc]
