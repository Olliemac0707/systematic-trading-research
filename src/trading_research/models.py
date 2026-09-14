"""Immutable domain models shared by data, strategies, and simulations."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9^][A-Z0-9._^:-]{0,31}$")
_RECONCILIATION_TOLERANCE = Decimal("0")


def normalize_symbol(symbol: str) -> str:
    """Normalize a stock symbol and reject unsafe or ambiguous characters."""

    if not isinstance(symbol, str):
        raise TypeError("symbol must be a string")
    normalized = symbol.strip().upper()
    if not _SYMBOL_PATTERN.fullmatch(normalized):
        raise ValueError("symbol must be 1-32 supported market-symbol characters")
    return normalized


def _require_finite_decimal(value: Decimal, name: str) -> None:
    """Require exact, finite decimal values at domain boundaries."""

    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")


def _require_aware_timestamp(timestamp: datetime) -> None:
    """Reject ambiguous timestamps that have no UTC offset."""

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")


class SignalAction(StrEnum):
    """Actions a strategy may recommend to a simulator."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class TradeSide(StrEnum):
    """Sides supported by the long-only simulation."""

    BUY = "buy"
    SELL = "sell"


class PreTradeDecisionOutcome(StrEnum):
    """Final outcomes of a simulated long-entry decision workflow."""

    APPROVED = "approved"
    REDUCED = "reduced"
    REJECTED = "rejected"


class PreTradeDecisionReason(StrEnum):
    """Stable machine-readable explanations for reductions and rejections."""

    POSITION_SIZER_RETURNED_ZERO = "position_sizer_returned_zero"
    MAXIMUM_POSITION_VALUE = "maximum_position_value"
    MINIMUM_CASH_RESERVE = "minimum_cash_reserve"
    MULTIPLE_RISK_POLICIES = "multiple_risk_policies"
    RISK_POLICY_REDUCED = "risk_policy_reduced"
    RISK_POLICY_REJECTED = "risk_policy_rejected"
    INSUFFICIENT_CASH = "insufficient_cash"


class ExecutionReason(StrEnum):
    """Reasons a simulated execution can be created."""

    STRATEGY_SIGNAL = "strategy_signal"
    END_OF_TEST_LIQUIDATION = "end_of_test_liquidation"


class EndOfTestPolicy(StrEnum):
    """Policies for a long position remaining after the final market bar."""

    HOLD = "hold"
    LIQUIDATE = "liquidate"


@dataclass(frozen=True, slots=True)
class MarketBar:
    """A validated OHLCV observation for one symbol and time period."""

    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int

    def __post_init__(self) -> None:
        """Validate symbol, timestamp, prices, and volume."""

        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _require_aware_timestamp(self.timestamp)

        named_prices = (
            ("open", self.open),
            ("high", self.high),
            ("low", self.low),
            ("close", self.close),
        )
        for name, price in named_prices:
            _require_finite_decimal(price, name)
        prices = tuple(price for _, price in named_prices)
        if any(price <= 0 for price in prices):
            raise ValueError("OHLC prices must be positive")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be the greatest OHLC value")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be the least OHLC value")
        if isinstance(self.volume, bool) or not isinstance(self.volume, int):
            raise TypeError("volume must be an integer")
        if self.volume < 0:
            raise ValueError("volume must be non-negative")


@dataclass(frozen=True, slots=True)
class StrategySignal:
    """A strategy recommendation associated with a market bar."""

    symbol: str
    timestamp: datetime
    action: SignalAction
    reason: str = ""

    def __post_init__(self) -> None:
        """Normalize the symbol and validate the timestamp."""

        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _require_aware_timestamp(self.timestamp)
        if not isinstance(self.action, SignalAction):
            raise TypeError("action must be a SignalAction")
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")


@dataclass(frozen=True, slots=True)
class SimulatedTrade:
    """A fill generated inside a simulation; it is never a live order."""

    symbol: str
    timestamp: datetime
    side: TradeSide
    quantity: int
    price: Decimal
    commission: Decimal = Decimal("0")
    reference_price: Decimal | None = None
    execution_reason: ExecutionReason = ExecutionReason.STRATEGY_SIGNAL

    def __post_init__(self) -> None:
        """Validate simulated fill values."""

        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _require_aware_timestamp(self.timestamp)
        if not isinstance(self.side, TradeSide):
            raise TypeError("side must be a TradeSide")
        if not isinstance(self.execution_reason, ExecutionReason):
            raise TypeError("execution_reason must be an ExecutionReason")
        if (
            self.execution_reason is ExecutionReason.END_OF_TEST_LIQUIDATION
            and self.side is not TradeSide.SELL
        ):
            raise ValueError("end-of-test liquidation must be a sell execution")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise TypeError("quantity must be an integer")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        _require_finite_decimal(self.price, "price")
        if self.price <= 0:
            raise ValueError("price must be positive")
        reference_price = self.price if self.reference_price is None else self.reference_price
        _require_finite_decimal(reference_price, "reference_price")
        if reference_price <= 0:
            raise ValueError("reference_price must be positive")
        object.__setattr__(self, "reference_price", reference_price)
        _require_finite_decimal(self.commission, "commission")
        if self.commission < 0:
            raise ValueError("commission must be non-negative")

    @property
    def notional(self) -> Decimal:
        """Return the fill-price value of the simulated execution."""

        return self.price * self.quantity

    @property
    def slippage_cost(self) -> Decimal:
        """Return signed execution slippage without changing account cash.

        Positive values are adverse to the simulated account. Fill-price
        notional already includes this effect, so callers must not deduct it
        from cash a second time.
        """

        reference_price = self.price if self.reference_price is None else self.reference_price
        if self.side is TradeSide.BUY:
            return (self.price - reference_price) * self.quantity
        return (reference_price - self.price) * self.quantity


@dataclass(frozen=True, slots=True)
class PreTradeDecision:
    """Immutable audit of one actionable simulated long-entry proposal.

    Estimated commission and total cost describe the position sizer's proposed
    quantity supplied to the risk policy. Actual executed commission remains on
    the resulting ``SimulatedTrade`` and is calculated exactly as before.
    """

    timestamp: datetime
    symbol: str
    side: TradeSide
    reference_price: Decimal
    proposed_quantity: int
    approved_quantity: int
    estimated_fill_price: Decimal
    estimated_commission: Decimal
    estimated_total_cost: Decimal
    available_cash_before: Decimal
    position_quantity_before: int
    outcome: PreTradeDecisionOutcome
    reason: PreTradeDecisionReason | None = None
    reason_detail: str | None = None

    def __post_init__(self) -> None:
        """Validate exact estimates, whole-share quantities, and outcome rules."""

        _require_aware_timestamp(self.timestamp)
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        if not isinstance(self.side, TradeSide):
            raise TypeError("side must be a TradeSide")
        if self.side is not TradeSide.BUY:
            raise ValueError("pre-trade decisions currently support buy proposals only")
        quantities = (
            ("proposed_quantity", self.proposed_quantity),
            ("approved_quantity", self.approved_quantity),
            ("position_quantity_before", self.position_quantity_before),
        )
        for name, quantity in quantities:
            if isinstance(quantity, bool) or not isinstance(quantity, int):
                raise TypeError(f"{name} must be an integer")
            if quantity < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.approved_quantity > self.proposed_quantity:
            raise ValueError("approved_quantity cannot exceed proposed_quantity")

        exact_values = (
            ("reference_price", self.reference_price),
            ("estimated_fill_price", self.estimated_fill_price),
            ("estimated_commission", self.estimated_commission),
            ("estimated_total_cost", self.estimated_total_cost),
            ("available_cash_before", self.available_cash_before),
        )
        for name, value in exact_values:
            _require_finite_decimal(value, name)
        if self.reference_price <= 0 or self.estimated_fill_price <= 0:
            raise ValueError("reference and estimated fill prices must be positive")
        if self.estimated_commission < 0 or self.estimated_total_cost < 0:
            raise ValueError("estimated costs must be non-negative")
        if self.available_cash_before < 0:
            raise ValueError("available_cash_before must be non-negative")
        expected_total = (
            self.estimated_fill_price * self.proposed_quantity
            + self.estimated_commission
        )
        if self.estimated_total_cost != expected_total:
            raise ValueError("estimated_total_cost must match the proposed quantity")
        if self.proposed_quantity == 0 and self.estimated_commission != 0:
            raise ValueError("a zero proposal must have zero estimated commission")

        if not isinstance(self.outcome, PreTradeDecisionOutcome):
            raise TypeError("outcome must be a PreTradeDecisionOutcome")
        if self.outcome is PreTradeDecisionOutcome.APPROVED:
            consistent = (
                self.proposed_quantity > 0
                and self.approved_quantity == self.proposed_quantity
            )
        elif self.outcome is PreTradeDecisionOutcome.REDUCED:
            consistent = 0 < self.approved_quantity < self.proposed_quantity
        else:
            consistent = self.approved_quantity == 0
        if not consistent:
            raise ValueError("outcome is inconsistent with proposed and approved quantities")
        if self.outcome is PreTradeDecisionOutcome.APPROVED:
            if self.reason is not None:
                raise ValueError("a fully approved decision must not have a reason")
            if self.reason_detail is not None:
                raise ValueError("a fully approved decision must not have reason detail")
        elif not isinstance(self.reason, PreTradeDecisionReason):
            raise TypeError("reduced and rejected decisions require a stable reason")
        if self.reason_detail is not None:
            if not isinstance(self.reason_detail, str):
                raise TypeError("reason_detail must be a string or None")
            if not self.reason_detail.strip() or not self.reason_detail.isprintable():
                raise ValueError("reason_detail must be non-empty printable text")

    @property
    def quantity_reduction(self) -> int:
        """Return units prevented by sizing, risk, or affordability checks."""

        return self.proposed_quantity - self.approved_quantity


@dataclass(frozen=True, slots=True)
class AccountLedgerEntry:
    """Auditable cash and position effects derived from one simulated fill."""

    trade: SimulatedTrade

    def __post_init__(self) -> None:
        """Require a validated simulated fill."""

        if not isinstance(self.trade, SimulatedTrade):
            raise TypeError("trade must be a SimulatedTrade")

    @property
    def gross_notional(self) -> Decimal:
        """Return fill-price notional before commission."""

        return self.trade.notional

    @property
    def cash_change(self) -> Decimal:
        """Return the exact cash movement, including commission once."""

        if self.trade.side is TradeSide.BUY:
            return -(self.gross_notional + self.trade.commission)
        return self.gross_notional - self.trade.commission

    @property
    def position_change(self) -> int:
        """Return the signed unit change caused by the fill."""

        if self.trade.side is TradeSide.BUY:
            return self.trade.quantity
        return -self.trade.quantity

    @property
    def slippage_cost(self) -> Decimal:
        """Return execution attribution that is not an extra cash movement."""

        return self.trade.slippage_cost


@dataclass(frozen=True, slots=True)
class AccountLedger:
    """Exact account movements for one-symbol, long-only simulation fills."""

    starting_cash: Decimal
    entries: tuple[AccountLedgerEntry, ...]

    def __post_init__(self) -> None:
        """Validate chronology and prevent leverage, shorts, or portfolios."""

        _require_finite_decimal(self.starting_cash, "starting_cash")
        if self.starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        if not isinstance(self.entries, tuple):
            raise TypeError("entries must be a tuple")

        running_cash = self.starting_cash
        running_position = 0
        previous_timestamp: datetime | None = None
        symbols: set[str] = set()
        for entry in self.entries:
            if not isinstance(entry, AccountLedgerEntry):
                raise TypeError("entries must contain AccountLedgerEntry values")
            trade = entry.trade
            symbols.add(trade.symbol)
            if previous_timestamp is not None and trade.timestamp < previous_timestamp:
                raise ValueError("ledger entries must be chronological")
            previous_timestamp = trade.timestamp

            running_cash += entry.cash_change
            running_position += entry.position_change
            if running_cash < 0:
                raise ValueError("ledger entries cannot create negative cash")
            if running_position < 0:
                raise ValueError("ledger entries cannot create a short position")

        if len(symbols) > 1:
            raise ValueError("account ledger supports exactly one symbol")

    @classmethod
    def from_trades(
        cls,
        starting_cash: Decimal,
        trades: tuple[SimulatedTrade, ...],
    ) -> "AccountLedger":
        """Build immutable ledger entries from executed simulated trades."""

        return cls(
            starting_cash=starting_cash,
            entries=tuple(AccountLedgerEntry(trade=trade) for trade in trades),
        )

    @property
    def executed_trades(self) -> tuple[SimulatedTrade, ...]:
        """Return the fills represented by the ledger."""

        return tuple(entry.trade for entry in self.entries)

    @property
    def total_buy_notional(self) -> Decimal:
        """Return gross fill-price notional for buys."""

        return sum(
            (
                entry.gross_notional
                for entry in self.entries
                if entry.trade.side is TradeSide.BUY
            ),
            Decimal("0"),
        )

    @property
    def total_sell_notional(self) -> Decimal:
        """Return gross fill-price notional for sells."""

        return sum(
            (
                entry.gross_notional
                for entry in self.entries
                if entry.trade.side is TradeSide.SELL
            ),
            Decimal("0"),
        )

    @property
    def total_commissions(self) -> Decimal:
        """Return commissions charged across all fills."""

        return sum(
            (entry.trade.commission for entry in self.entries),
            Decimal("0"),
        )

    @property
    def net_trade_cash_flow(self) -> Decimal:
        """Return the ledger-replayed change in cash."""

        return self.calculated_ending_cash - self.starting_cash

    @property
    def calculated_ending_cash(self) -> Decimal:
        """Replay trade cash movements in their original execution order."""

        cash = self.starting_cash
        for entry in self.entries:
            cash += entry.cash_change
        return cash

    @property
    def ending_position_quantity(self) -> int:
        """Return remaining long units after every ledger entry."""

        return sum(entry.position_change for entry in self.entries)

    @property
    def total_slippage_cost(self) -> Decimal:
        """Return execution attribution without deducting it from cash again."""

        return sum((entry.slippage_cost for entry in self.entries), Decimal("0"))


@dataclass(frozen=True, slots=True)
class AccountReconciliation:
    """Explain ending cash and equity using an exact-zero tolerance policy."""

    ledger: AccountLedger
    ending_cash: Decimal
    ending_position_mark_price: Decimal
    ending_equity: Decimal

    def __post_init__(self) -> None:
        """Validate reported account values and the independent position mark."""

        if not isinstance(self.ledger, AccountLedger):
            raise TypeError("ledger must be an AccountLedger")
        _require_finite_decimal(self.ending_cash, "ending_cash")
        _require_finite_decimal(
            self.ending_position_mark_price, "ending_position_mark_price"
        )
        _require_finite_decimal(self.ending_equity, "ending_equity")
        if self.ending_cash < 0 or self.ending_equity < 0:
            raise ValueError("ending cash and equity must be non-negative")
        if self.ending_position_mark_price < 0:
            raise ValueError("ending_position_mark_price must be non-negative")
        if (
            self.remaining_position_quantity > 0
            and self.ending_position_mark_price == 0
        ):
            raise ValueError("an open position requires a positive ending mark price")

    @property
    def remaining_position_quantity(self) -> int:
        """Return the ledger-derived ending position."""

        return self.ledger.ending_position_quantity

    @property
    def remaining_position_value(self) -> Decimal:
        """Return exact mark-to-market value of remaining units."""

        return self.ending_position_mark_price * self.remaining_position_quantity

    @property
    def calculated_ending_cash(self) -> Decimal:
        """Return cash explained by starting cash and ledger entries."""

        return self.ledger.calculated_ending_cash

    @property
    def calculated_ending_equity(self) -> Decimal:
        """Return ledger cash plus the remaining marked position."""

        return self.calculated_ending_cash + self.remaining_position_value

    @property
    def cash_difference(self) -> Decimal:
        """Return reported cash less ledger-explained cash."""

        return self.ending_cash - self.calculated_ending_cash

    @property
    def reconciliation_difference(self) -> Decimal:
        """Return reported equity less ledger-explained ending equity."""

        return self.ending_equity - self.calculated_ending_equity

    @property
    def is_reconciled(self) -> bool:
        """Return whether both cash and equity differences are exactly zero."""

        return (
            self.cash_difference == _RECONCILIATION_TOLERANCE
            and self.reconciliation_difference == _RECONCILIATION_TOLERANCE
        )


@dataclass(frozen=True, slots=True)
class EquityPoint:
    """Portfolio equity at a point in simulated time."""

    timestamp: datetime
    equity: Decimal

    def __post_init__(self) -> None:
        """Validate the timestamp and equity."""

        _require_aware_timestamp(self.timestamp)
        _require_finite_decimal(self.equity, "equity")
        if self.equity < 0:
            raise ValueError("equity must be non-negative")


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Aggregate output of a completed simulation."""

    initial_cash: Decimal
    final_cash: Decimal
    final_equity: Decimal
    final_position_quantity: int
    trades: tuple[SimulatedTrade, ...]
    equity_curve: tuple[EquityPoint, ...]
    final_position_mark_price: Decimal | None = None
    end_of_test_policy: EndOfTestPolicy = EndOfTestPolicy.HOLD
    pre_trade_decisions: tuple[PreTradeDecision, ...] = ()
    reconciliation: AccountReconciliation = field(init=False)

    def __post_init__(self) -> None:
        """Validate portfolio totals and the equity curve."""

        _require_finite_decimal(self.initial_cash, "initial_cash")
        _require_finite_decimal(self.final_cash, "final_cash")
        _require_finite_decimal(self.final_equity, "final_equity")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.final_cash < 0 or self.final_equity < 0:
            raise ValueError("final cash and equity must be non-negative")
        if isinstance(self.final_position_quantity, bool) or not isinstance(
            self.final_position_quantity, int
        ):
            raise TypeError("final_position_quantity must be an integer")
        if self.final_position_quantity < 0:
            raise ValueError("final_position_quantity must be non-negative")
        if not isinstance(self.pre_trade_decisions, tuple):
            raise TypeError("pre_trade_decisions must be a tuple")
        previous_decision_timestamp: datetime | None = None
        for decision in self.pre_trade_decisions:
            if not isinstance(decision, PreTradeDecision):
                raise TypeError(
                    "pre_trade_decisions must contain PreTradeDecision values"
                )
            if (
                previous_decision_timestamp is not None
                and decision.timestamp <= previous_decision_timestamp
            ):
                raise ValueError("pre_trade_decisions must be strictly chronological")
            previous_decision_timestamp = decision.timestamp
        if not isinstance(self.end_of_test_policy, EndOfTestPolicy):
            raise TypeError("end_of_test_policy must be an EndOfTestPolicy")
        final_position_mark_price = self.final_position_mark_price
        if final_position_mark_price is None:
            if self.final_position_quantity > 0:
                raise ValueError(
                    "final_position_mark_price is required for an open position"
                )
            final_position_mark_price = Decimal("0")
            object.__setattr__(
                self, "final_position_mark_price", final_position_mark_price
            )
        _require_finite_decimal(
            final_position_mark_price, "final_position_mark_price"
        )
        if final_position_mark_price < 0:
            raise ValueError("final_position_mark_price must be non-negative")
        if self.final_position_quantity > 0 and final_position_mark_price == 0:
            raise ValueError("an open position requires a positive final mark price")
        ledger = AccountLedger.from_trades(self.initial_cash, self.trades)
        if ledger.ending_position_quantity != self.final_position_quantity:
            raise ValueError("final_position_quantity must match simulated trades")
        if not self.equity_curve:
            raise ValueError("equity_curve must contain at least one point")
        if any(
            current.timestamp <= previous.timestamp
            for previous, current in zip(self.equity_curve, self.equity_curve[1:], strict=False)
        ):
            raise ValueError("equity_curve must be strictly chronological")
        if self.equity_curve[-1].equity != self.final_equity:
            raise ValueError("final_equity must equal the last equity-curve value")
        self._validate_pre_trade_decisions()
        liquidation_trades = tuple(
            trade
            for trade in self.trades
            if trade.execution_reason is ExecutionReason.END_OF_TEST_LIQUIDATION
        )
        if len(liquidation_trades) > 1:
            raise ValueError("a backtest can contain at most one end liquidation")
        if liquidation_trades:
            if self.end_of_test_policy is not EndOfTestPolicy.LIQUIDATE:
                raise ValueError("end liquidation requires the liquidate policy")
            if (
                self.trades[-1].execution_reason
                is not ExecutionReason.END_OF_TEST_LIQUIDATION
            ):
                raise ValueError("end liquidation must be the final execution")
            if liquidation_trades[0].timestamp != self.equity_curve[-1].timestamp:
                raise ValueError("end liquidation must use the final valuation timestamp")
        if (
            self.end_of_test_policy is EndOfTestPolicy.LIQUIDATE
            and self.final_position_quantity > 0
        ):
            raise ValueError("liquidate policy cannot leave an open position")
        if self.final_position_quantity == 0 and self.final_cash != self.final_equity:
            raise ValueError("final cash and equity must match when the position is flat")
        if self.final_position_quantity > 0 and self.final_equity <= self.final_cash:
            raise ValueError("a long ending position must add positive marked-to-market value")

        reconciliation = AccountReconciliation(
            ledger=ledger,
            ending_cash=self.final_cash,
            ending_position_mark_price=final_position_mark_price,
            ending_equity=self.final_equity,
        )
        if not reconciliation.is_reconciled:
            raise ValueError(
                "completed backtest must have zero reconciliation difference "
                "for cash and equity"
            )
        object.__setattr__(self, "reconciliation", reconciliation)

    def _validate_pre_trade_decisions(self) -> None:
        """Require recorded entry decisions to match executions without inferring them."""

        if not self.pre_trade_decisions:
            return
        equity_timestamps = {point.timestamp for point in self.equity_curve}
        strategy_buys = tuple(
            trade
            for trade in self.trades
            if trade.side is TradeSide.BUY
            and trade.execution_reason is ExecutionReason.STRATEGY_SIGNAL
        )
        matched_buys: set[int] = set()
        for decision in self.pre_trade_decisions:
            if decision.timestamp not in equity_timestamps:
                raise ValueError("pre-trade decision must align with an equity observation")
            matching_indexes = tuple(
                index
                for index, trade in enumerate(strategy_buys)
                if trade.timestamp == decision.timestamp
                and trade.symbol == decision.symbol
            )
            if decision.approved_quantity == 0:
                if matching_indexes:
                    raise ValueError("a rejected decision cannot create a buy execution")
                continue
            if len(matching_indexes) != 1:
                raise ValueError("an approved decision requires one matching buy execution")
            match_index = matching_indexes[0]
            trade = strategy_buys[match_index]
            if trade.quantity != decision.approved_quantity:
                raise ValueError("approved decision quantity must match its buy execution")
            if trade.reference_price != decision.reference_price:
                raise ValueError("decision reference price must match its buy execution")
            if trade.price != decision.estimated_fill_price:
                raise ValueError("decision fill estimate must match its buy execution")
            matched_buys.add(match_index)
        if len(matched_buys) != len(strategy_buys):
            raise ValueError("every strategy buy must have one recorded decision")

    @property
    def total_return(self) -> Decimal:
        """Return the unannualized portfolio return as a decimal ratio."""

        return (self.final_equity / self.initial_cash) - Decimal("1")

    @property
    def was_end_of_test_liquidated(self) -> bool:
        """Return whether a remaining position received a synthetic final exit."""

        return bool(
            self.trades
            and self.trades[-1].execution_reason
            is ExecutionReason.END_OF_TEST_LIQUIDATION
        )
