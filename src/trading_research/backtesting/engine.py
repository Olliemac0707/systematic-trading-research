"""A minimal long-only, close-price backtesting engine."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading_research.backtesting.base import BacktestEngine
from trading_research.config import BacktestConfig, EndOfTestPolicy
from trading_research.errors import BacktestError
from trading_research.models import (
    BacktestResult,
    EquityPoint,
    ExecutionReason,
    MarketBar,
    PreTradeDecision,
    PreTradeDecisionOutcome,
    PreTradeDecisionReason,
    SignalAction,
    SimulatedTrade,
    StrategySignal,
    TradeSide,
)
from trading_research.risk import (
    PositionSizer,
    RiskDecision,
    RiskPolicy,
    build_position_sizer,
    build_risk_policy,
)
from trading_research.simulation import (
    calculate_buy_cost,
    calculate_buy_fill_price,
    calculate_proportional_commission,
    calculate_sell_fill_price,
)
from trading_research.strategies import Strategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _PreparedBuy:
    """One audited entry decision plus its unchanged execution commission."""

    decision: PreTradeDecision
    execution_commission: Decimal


class SimpleBacktestEngine(BacktestEngine):
    """Simulate sized, risk-checked long-only trades at the next bar's open."""

    def run(
        self,
        bars: Sequence[MarketBar],
        strategy: Strategy,
        config: BacktestConfig,
        *,
        position_sizer: PositionSizer | None = None,
        risk_policy: RiskPolicy | None = None,
    ) -> BacktestResult:
        """Run the strategy and return trades plus mark-to-market equity.

        Raises:
            BacktestError: If bars are empty, mixed-symbol, non-chronological, or
                the strategy produces ambiguous signals.
        """

        bar_snapshot = tuple(bars)
        self._validate_bars(bar_snapshot)
        signals = strategy.generate_signals(bar_snapshot)
        signals_by_timestamp = self._index_signals(signals, bar_snapshot)
        selected_sizer = (
            build_position_sizer(config) if position_sizer is None else position_sizer
        )
        selected_risk_policy = (
            build_risk_policy(config) if risk_policy is None else risk_policy
        )

        cash = config.initial_cash
        quantity = 0
        trades: list[SimulatedTrade] = []
        pre_trade_decisions: list[PreTradeDecision] = []
        equity_curve: list[EquityPoint] = []
        pending_signal: StrategySignal | None = None

        for bar in bar_snapshot:
            if (
                pending_signal is not None
                and pending_signal.action is SignalAction.BUY
                and quantity == 0
            ):
                prepared = self._prepare_buy(
                    timestamp=bar.timestamp,
                    symbol=bar.symbol,
                    available_cash=cash,
                    reference_price=bar.open,
                    current_position_quantity=quantity,
                    config=config,
                    position_sizer=selected_sizer,
                    risk_policy=selected_risk_policy,
                )
                decision = prepared.decision
                pre_trade_decisions.append(decision)
                approved_quantity = decision.approved_quantity
                if approved_quantity > 0:
                    total_cost = calculate_buy_cost(
                        decision.estimated_fill_price,
                        approved_quantity,
                        config.commission_bps,
                    )
                    cash -= total_cost
                    quantity = approved_quantity
                    trades.append(
                        SimulatedTrade(
                            symbol=bar.symbol,
                            timestamp=bar.timestamp,
                            side=TradeSide.BUY,
                            quantity=quantity,
                            price=decision.estimated_fill_price,
                            reference_price=bar.open,
                            commission=prepared.execution_commission,
                        )
                    )
            elif (
                pending_signal is not None
                and pending_signal.action is SignalAction.SELL
                and quantity > 0
            ):
                fill_price = self._sell_price(bar.open, config.slippage_bps)
                commission = self._commission(fill_price, quantity, config)
                cash += (fill_price * quantity) - commission
                trades.append(
                    SimulatedTrade(
                        symbol=bar.symbol,
                        timestamp=bar.timestamp,
                        side=TradeSide.SELL,
                        quantity=quantity,
                        price=fill_price,
                        reference_price=bar.open,
                        commission=commission,
                    )
                )
                quantity = 0

            equity_curve.append(
                EquityPoint(timestamp=bar.timestamp, equity=cash + (bar.close * quantity))
            )
            pending_signal = signals_by_timestamp.get(bar.timestamp)

        if (
            config.end_of_test_policy is EndOfTestPolicy.LIQUIDATE
            and quantity > 0
        ):
            final_bar = bar_snapshot[-1]
            fill_price = self._sell_price(final_bar.close, config.slippage_bps)
            commission = self._commission(fill_price, quantity, config)
            cash += (fill_price * quantity) - commission
            trades.append(
                SimulatedTrade(
                    symbol=final_bar.symbol,
                    timestamp=final_bar.timestamp,
                    side=TradeSide.SELL,
                    quantity=quantity,
                    price=fill_price,
                    reference_price=final_bar.close,
                    commission=commission,
                    execution_reason=ExecutionReason.END_OF_TEST_LIQUIDATION,
                )
            )
            quantity = 0
            equity_curve[-1] = EquityPoint(
                timestamp=final_bar.timestamp,
                equity=cash,
            )

        final_equity = cash + (bar_snapshot[-1].close * quantity)
        logger.info("Backtest completed with %d simulated trades", len(trades))
        return BacktestResult(
            initial_cash=config.initial_cash,
            final_cash=cash,
            final_equity=final_equity,
            final_position_quantity=quantity,
            final_position_mark_price=bar_snapshot[-1].close,
            trades=tuple(trades),
            equity_curve=tuple(equity_curve),
            pre_trade_decisions=tuple(pre_trade_decisions),
            end_of_test_policy=config.end_of_test_policy,
        )

    @staticmethod
    def _validate_bars(bars: Sequence[MarketBar]) -> None:
        """Validate collection-level backtest assumptions."""

        if not bars:
            raise BacktestError("backtest requires at least one market bar")
        if len({bar.symbol for bar in bars}) != 1:
            raise BacktestError("backtest bars must contain exactly one symbol")
        if any(
            current.timestamp <= previous.timestamp
            for previous, current in zip(bars, bars[1:], strict=False)
        ):
            raise BacktestError("backtest bars must be strictly chronological")

    @staticmethod
    def _index_signals(
        signals: Sequence[StrategySignal], bars: Sequence[MarketBar]
    ) -> dict[datetime, StrategySignal]:
        """Validate and index strategy output by timestamp."""

        indexed: dict[datetime, StrategySignal] = {}
        expected_symbol = bars[0].symbol
        bar_timestamps = {bar.timestamp for bar in bars}
        for signal in signals:
            if signal.symbol != expected_symbol:
                raise BacktestError("strategy returned a signal for an unexpected symbol")
            if signal.timestamp not in bar_timestamps:
                raise BacktestError("strategy returned a signal outside the backtest bars")
            if signal.timestamp in indexed:
                raise BacktestError("strategy returned multiple signals for one timestamp")
            indexed[signal.timestamp] = signal
        return indexed

    @classmethod
    def _prepare_buy(
        cls,
        *,
        timestamp: datetime,
        symbol: str,
        available_cash: Decimal,
        reference_price: Decimal,
        current_position_quantity: int,
        config: BacktestConfig,
        position_sizer: PositionSizer,
        risk_policy: RiskPolicy,
    ) -> _PreparedBuy:
        """Size, risk-check, audit, and verify affordability before a buy."""

        proposed_quantity = position_sizer.calculate_quantity(
            available_cash=available_cash,
            reference_price=reference_price,
            commission_bps=config.commission_bps,
            slippage_bps=config.slippage_bps,
        )
        cls._validate_policy_quantity(proposed_quantity, "position sizer")
        fill_price = cls._buy_price(reference_price, config.slippage_bps)
        if proposed_quantity == 0:
            return _PreparedBuy(
                decision=cls._entry_decision(
                    timestamp=timestamp,
                    symbol=symbol,
                    reference_price=reference_price,
                    proposed_quantity=0,
                    approved_quantity=0,
                    fill_price=fill_price,
                    proposed_commission=Decimal("0"),
                    available_cash=available_cash,
                    current_position_quantity=current_position_quantity,
                    outcome=PreTradeDecisionOutcome.REJECTED,
                    reason=PreTradeDecisionReason.POSITION_SIZER_RETURNED_ZERO,
                ),
                execution_commission=Decimal("0"),
            )

        proposed_commission = cls._commission(fill_price, proposed_quantity, config)
        decision = risk_policy.evaluate(
            proposed_quantity=proposed_quantity,
            available_cash=available_cash,
            reference_price=reference_price,
            estimated_fill_price=fill_price,
            estimated_commission=proposed_commission,
            commission_bps=config.commission_bps,
            current_position_quantity=current_position_quantity,
        )
        if not isinstance(decision, RiskDecision):
            raise BacktestError("risk policy must return a RiskDecision")
        approved_quantity = decision.approved_quantity
        cls._validate_policy_quantity(approved_quantity, "risk policy")
        if approved_quantity > proposed_quantity:
            raise BacktestError("risk policy cannot increase the proposed quantity")
        if not decision.approved:
            return _PreparedBuy(
                decision=cls._entry_decision(
                    timestamp=timestamp,
                    symbol=symbol,
                    reference_price=reference_price,
                    proposed_quantity=proposed_quantity,
                    approved_quantity=0,
                    fill_price=fill_price,
                    proposed_commission=proposed_commission,
                    available_cash=available_cash,
                    current_position_quantity=current_position_quantity,
                    outcome=PreTradeDecisionOutcome.REJECTED,
                    reason=cls._risk_reason(decision, reduced=False),
                    reason_detail=decision.reason,
                ),
                execution_commission=Decimal("0"),
            )

        commission = cls._commission(fill_price, approved_quantity, config)
        total_cost = calculate_buy_cost(
            fill_price,
            approved_quantity,
            config.commission_bps,
        )
        if total_cost > available_cash:
            logger.warning("Skipped simulated buy: approved quantity is unaffordable")
            return _PreparedBuy(
                decision=cls._entry_decision(
                    timestamp=timestamp,
                    symbol=symbol,
                    reference_price=reference_price,
                    proposed_quantity=proposed_quantity,
                    approved_quantity=0,
                    fill_price=fill_price,
                    proposed_commission=proposed_commission,
                    available_cash=available_cash,
                    current_position_quantity=current_position_quantity,
                    outcome=PreTradeDecisionOutcome.REJECTED,
                    reason=PreTradeDecisionReason.INSUFFICIENT_CASH,
                ),
                execution_commission=Decimal("0"),
            )
        outcome = (
            PreTradeDecisionOutcome.APPROVED
            if approved_quantity == proposed_quantity
            else PreTradeDecisionOutcome.REDUCED
        )
        return _PreparedBuy(
            decision=cls._entry_decision(
                timestamp=timestamp,
                symbol=symbol,
                reference_price=reference_price,
                proposed_quantity=proposed_quantity,
                approved_quantity=approved_quantity,
                fill_price=fill_price,
                proposed_commission=proposed_commission,
                available_cash=available_cash,
                current_position_quantity=current_position_quantity,
                outcome=outcome,
                reason=(
                    None
                    if outcome is PreTradeDecisionOutcome.APPROVED
                    else cls._risk_reason(decision, reduced=True)
                ),
                reason_detail=(
                    None
                    if outcome is PreTradeDecisionOutcome.APPROVED
                    else decision.reason
                ),
            ),
            execution_commission=commission,
        )

    @staticmethod
    def _entry_decision(
        *,
        timestamp: datetime,
        symbol: str,
        reference_price: Decimal,
        proposed_quantity: int,
        approved_quantity: int,
        fill_price: Decimal,
        proposed_commission: Decimal,
        available_cash: Decimal,
        current_position_quantity: int,
        outcome: PreTradeDecisionOutcome,
        reason: PreTradeDecisionReason | None,
        reason_detail: str | None = None,
    ) -> PreTradeDecision:
        """Create one audit record from already-computed decision inputs."""

        return PreTradeDecision(
            timestamp=timestamp,
            symbol=symbol,
            side=TradeSide.BUY,
            reference_price=reference_price,
            proposed_quantity=proposed_quantity,
            approved_quantity=approved_quantity,
            estimated_fill_price=fill_price,
            estimated_commission=proposed_commission,
            estimated_total_cost=(
                fill_price * proposed_quantity + proposed_commission
            ),
            available_cash_before=available_cash,
            position_quantity_before=current_position_quantity,
            outcome=outcome,
            reason=reason,
            reason_detail=reason_detail,
        )

    @staticmethod
    def _risk_reason(
        decision: RiskDecision,
        *,
        reduced: bool,
    ) -> PreTradeDecisionReason:
        """Map existing policy output to a stable reason without re-evaluation."""

        if decision.reason_code is not None:
            return decision.reason_code
        if reduced:
            return PreTradeDecisionReason.RISK_POLICY_REDUCED
        return PreTradeDecisionReason.RISK_POLICY_REJECTED

    @staticmethod
    def _validate_policy_quantity(quantity: int, source: str) -> None:
        """Require non-negative whole shares from sizing and risk policies."""

        if isinstance(quantity, bool) or not isinstance(quantity, int):
            raise BacktestError(f"{source} quantity must be an integer")
        if quantity < 0:
            raise BacktestError(f"{source} quantity must be non-negative")

    @staticmethod
    def _buy_price(close: Decimal, slippage_bps: Decimal) -> Decimal:
        """Apply adverse simulated slippage to a buy."""

        return calculate_buy_fill_price(close, slippage_bps)

    @staticmethod
    def _sell_price(close: Decimal, slippage_bps: Decimal) -> Decimal:
        """Apply adverse simulated slippage to a sell."""

        return calculate_sell_fill_price(close, slippage_bps)

    @staticmethod
    def _commission(
        price: Decimal,
        quantity: int,
        config: BacktestConfig,
    ) -> Decimal:
        """Calculate a proportional simulated commission."""

        return calculate_proportional_commission(
            price,
            quantity,
            config.commission_bps,
        )
