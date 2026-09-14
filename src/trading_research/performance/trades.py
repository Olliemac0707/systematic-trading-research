"""Pure closed-trade reconstruction and deterministic trade statistics."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from trading_research.models import (
    BacktestResult,
    ExecutionReason,
    SimulatedTrade,
    TradeSide,
    normalize_symbol,
)
from trading_research.performance._decimal import exact_decimal_sum
from trading_research.performance.metrics import calculate_performance

_ZERO = Decimal("0")


def _require_finite_decimal(value: Decimal, name: str) -> None:
    """Require an exact, finite Decimal value."""

    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")


def _require_aware_timestamp(timestamp: datetime, name: str) -> None:
    """Require a timezone-aware execution timestamp."""

    if not isinstance(timestamp, datetime):
        raise TypeError(f"{name} must be a datetime")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    """One completed long-only entry and full exit."""

    symbol: str
    entry_timestamp: datetime
    exit_timestamp: datetime
    quantity: int
    entry_reference_price: Decimal
    entry_fill_price: Decimal
    exit_reference_price: Decimal
    exit_fill_price: Decimal
    gross_entry_value: Decimal
    gross_exit_value: Decimal
    entry_commission: Decimal
    exit_commission: Decimal
    total_commission: Decimal
    entry_slippage_cost: Decimal
    exit_slippage_cost: Decimal
    total_slippage_cost: Decimal
    gross_pnl: Decimal
    net_pnl: Decimal
    return_ratio: Decimal
    holding_period: timedelta
    exit_execution_reason: ExecutionReason = ExecutionReason.STRATEGY_SIGNAL

    def __post_init__(self) -> None:
        """Validate exact execution-derived values and relationships."""

        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _require_aware_timestamp(self.entry_timestamp, "entry_timestamp")
        _require_aware_timestamp(self.exit_timestamp, "exit_timestamp")
        if not isinstance(self.exit_execution_reason, ExecutionReason):
            raise TypeError("exit_execution_reason must be an ExecutionReason")
        if self.exit_timestamp < self.entry_timestamp:
            raise ValueError("exit_timestamp must not precede entry_timestamp")
        if (
            self.exit_timestamp == self.entry_timestamp
            and self.exit_execution_reason
            is not ExecutionReason.END_OF_TEST_LIQUIDATION
        ):
            raise ValueError("equal execution timestamps require end liquidation")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise TypeError("quantity must be an integer")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if not isinstance(self.holding_period, timedelta):
            raise TypeError("holding_period must be a timedelta")
        if self.holding_period < timedelta(0):
            raise ValueError("holding_period must be non-negative")
        if self.holding_period != self.exit_timestamp - self.entry_timestamp:
            raise ValueError("holding_period must match the execution timestamps")

        financial_values = (
            ("entry_reference_price", self.entry_reference_price),
            ("entry_fill_price", self.entry_fill_price),
            ("exit_reference_price", self.exit_reference_price),
            ("exit_fill_price", self.exit_fill_price),
            ("gross_entry_value", self.gross_entry_value),
            ("gross_exit_value", self.gross_exit_value),
            ("entry_commission", self.entry_commission),
            ("exit_commission", self.exit_commission),
            ("total_commission", self.total_commission),
            ("entry_slippage_cost", self.entry_slippage_cost),
            ("exit_slippage_cost", self.exit_slippage_cost),
            ("total_slippage_cost", self.total_slippage_cost),
            ("gross_pnl", self.gross_pnl),
            ("net_pnl", self.net_pnl),
            ("return_ratio", self.return_ratio),
        )
        for name, value in financial_values:
            _require_finite_decimal(value, name)
        if min(
            self.entry_reference_price,
            self.entry_fill_price,
            self.exit_reference_price,
            self.exit_fill_price,
        ) <= 0:
            raise ValueError("trade prices must be positive")
        if self.entry_commission < 0 or self.exit_commission < 0:
            raise ValueError("trade commissions must be non-negative")

        quantity = Decimal(self.quantity)
        if self.gross_entry_value != self.entry_fill_price * quantity:
            raise ValueError("gross_entry_value must equal entry fill value")
        if self.gross_exit_value != self.exit_fill_price * quantity:
            raise ValueError("gross_exit_value must equal exit fill value")
        if self.total_commission != self.entry_commission + self.exit_commission:
            raise ValueError("total_commission must equal both execution commissions")
        if self.entry_slippage_cost != (
            self.entry_fill_price - self.entry_reference_price
        ) * quantity:
            raise ValueError("entry_slippage_cost must match buy execution attribution")
        if self.exit_slippage_cost != (
            self.exit_reference_price - self.exit_fill_price
        ) * quantity:
            raise ValueError("exit_slippage_cost must match sell execution attribution")
        if (
            self.total_slippage_cost
            != self.entry_slippage_cost + self.exit_slippage_cost
        ):
            raise ValueError("total_slippage_cost must equal both execution costs")
        if self.gross_pnl != (
            self.exit_fill_price - self.entry_fill_price
        ) * quantity:
            raise ValueError("gross_pnl must equal fill-price movement times quantity")
        if self.net_pnl != self.gross_pnl - self.total_commission:
            raise ValueError("net_pnl must deduct commission exactly once")
        if self.return_ratio != self.net_pnl / self.gross_entry_value:
            raise ValueError("return_ratio must use gross entry value")


@dataclass(frozen=True, slots=True)
class TradeStatistics:
    """Immutable aggregate statistics for completed long-only trades."""

    completed_trade_count: int
    winning_trade_count: int
    losing_trade_count: int
    breakeven_trade_count: int
    win_rate: Decimal | None
    loss_rate: Decimal | None
    gross_profit: Decimal
    gross_loss: Decimal
    net_closed_trade_pnl: Decimal
    average_net_pnl: Decimal | None
    average_winner: Decimal | None
    average_loser: Decimal | None
    largest_winner: Decimal | None
    largest_loser: Decimal | None
    profit_factor: Decimal | None
    payoff_ratio: Decimal | None
    expectancy: Decimal | None
    average_return_ratio: Decimal | None
    average_holding_period: timedelta | None
    longest_holding_period: timedelta | None
    shortest_holding_period: timedelta | None
    total_trade_commission: Decimal
    total_trade_slippage_cost: Decimal

    def __post_init__(self) -> None:
        """Validate count, rate, P&L, and duration relationships."""

        counts = (
            self.completed_trade_count,
            self.winning_trade_count,
            self.losing_trade_count,
            self.breakeven_trade_count,
        )
        if any(isinstance(count, bool) or not isinstance(count, int) for count in counts):
            raise TypeError("trade counts must be integers")
        if any(count < 0 for count in counts):
            raise ValueError("trade counts must be non-negative")
        if self.completed_trade_count != sum(counts[1:]):
            raise ValueError("completed trade count must equal all classifications")

        decimal_values = (
            ("gross_profit", self.gross_profit),
            ("gross_loss", self.gross_loss),
            ("net_closed_trade_pnl", self.net_closed_trade_pnl),
            ("total_trade_commission", self.total_trade_commission),
            ("total_trade_slippage_cost", self.total_trade_slippage_cost),
        )
        optional_decimals = (
            ("win_rate", self.win_rate),
            ("loss_rate", self.loss_rate),
            ("average_net_pnl", self.average_net_pnl),
            ("average_winner", self.average_winner),
            ("average_loser", self.average_loser),
            ("largest_winner", self.largest_winner),
            ("largest_loser", self.largest_loser),
            ("profit_factor", self.profit_factor),
            ("payoff_ratio", self.payoff_ratio),
            ("expectancy", self.expectancy),
            ("average_return_ratio", self.average_return_ratio),
        )
        for decimal_name, decimal_value in decimal_values:
            _require_finite_decimal(decimal_value, decimal_name)
        for optional_name, optional_value in optional_decimals:
            if optional_value is not None:
                _require_finite_decimal(optional_value, optional_name)
        if self.gross_profit < 0 or self.gross_loss < 0:
            raise ValueError("gross profit and loss must be non-negative")
        if self.total_trade_commission < 0:
            raise ValueError("total_trade_commission must be non-negative")
        if (
            exact_decimal_sum(
                (self.gross_profit, self.gross_loss.copy_negate())
            )
            != self.net_closed_trade_pnl
        ):
            raise ValueError("gross profit less gross loss must equal net P&L")

        durations = (
            self.average_holding_period,
            self.longest_holding_period,
            self.shortest_holding_period,
        )
        if any(
            duration is not None and not isinstance(duration, timedelta)
            for duration in durations
        ):
            raise TypeError("holding-period statistics must be timedeltas")
        if any(duration is not None and duration < timedelta(0) for duration in durations):
            raise ValueError("holding-period statistics must be non-negative")

        if self.completed_trade_count == 0:
            if any(value is not None for _, value in optional_decimals) or any(
                duration is not None for duration in durations
            ):
                raise ValueError("empty trade statistics must not contain averages")
            return

        count = Decimal(self.completed_trade_count)
        if self.win_rate != Decimal(self.winning_trade_count) / count:
            raise ValueError("win_rate must include every completed trade")
        if self.loss_rate != Decimal(self.losing_trade_count) / count:
            raise ValueError("loss_rate must include every completed trade")
        if self.average_net_pnl != self.net_closed_trade_pnl / count:
            raise ValueError("average_net_pnl must match net closed P&L")
        if self.expectancy != self.average_net_pnl:
            raise ValueError("expectancy must equal average currency P&L")
        if any(duration is None for duration in durations):
            raise ValueError("non-empty trade statistics require holding periods")
        if (
            self.shortest_holding_period is not None
            and self.longest_holding_period is not None
            and self.shortest_holding_period > self.longest_holding_period
        ):
            raise ValueError("shortest holding period cannot exceed longest")


def reconstruct_closed_trades(result: BacktestResult) -> tuple[ClosedTrade, ...]:
    """Pair chronological full-size BUY/SELL executions from the current engine.

    The engine permits only one long position at a time and sells it in full.
    Repeated entries and partial exits are rejected rather than silently assigned
    a cost-allocation policy. A final unmatched buy remains an open position.
    """

    if not isinstance(result, BacktestResult):
        raise TypeError("result must be a BacktestResult")

    closed_trades: list[ClosedTrade] = []
    open_entry: SimulatedTrade | None = None
    previous_timestamp: datetime | None = None
    expected_symbol: str | None = None
    for execution in result.trades:
        if not isinstance(execution, SimulatedTrade):
            raise TypeError("backtest trades must contain SimulatedTrade values")
        if expected_symbol is None:
            expected_symbol = execution.symbol
        elif execution.symbol != expected_symbol:
            raise ValueError("all executions must have the same symbol")
        if previous_timestamp is not None and execution.timestamp < previous_timestamp:
            raise ValueError("executions must be strictly chronological")
        if (
            previous_timestamp is not None
            and execution.timestamp == previous_timestamp
            and execution.execution_reason
            is not ExecutionReason.END_OF_TEST_LIQUIDATION
        ):
            raise ValueError("equal execution timestamps require end liquidation")
        previous_timestamp = execution.timestamp

        if execution.side is TradeSide.BUY:
            if open_entry is not None:
                raise ValueError("repeated entries are unsupported while a position is open")
            open_entry = execution
            continue

        if open_entry is None:
            raise ValueError("sell before buy would create unsupported short exposure")
        if execution.symbol != open_entry.symbol:
            raise ValueError("paired executions must have the same symbol")
        if execution.quantity > open_entry.quantity:
            raise ValueError("sell quantity cannot exceed the open position")
        if execution.quantity < open_entry.quantity:
            raise ValueError("partial exits are unsupported by the current engine")
        closed_trades.append(_create_closed_trade(open_entry, execution))
        open_entry = None

    return tuple(closed_trades)


def calculate_trade_statistics(result: BacktestResult) -> TradeStatistics:
    """Calculate exact statistics for all reconstructed completed trades."""

    closed_trades = reconstruct_closed_trades(result)
    performance = calculate_performance(result)
    closed_commission = exact_decimal_sum(
        trade.total_commission for trade in closed_trades
    )
    closed_net_pnl = exact_decimal_sum(
        (
            performance.gross_realised_pnl,
            closed_commission.copy_negate(),
        )
    )
    return _summarise_closed_trades(
        closed_trades,
        reconciled_net_pnl=closed_net_pnl,
    )


def _create_closed_trade(entry: SimulatedTrade, exit_: SimulatedTrade) -> ClosedTrade:
    """Create one validated closed trade from a matched execution pair."""

    quantity = Decimal(entry.quantity)
    entry_reference_price = (
        entry.price if entry.reference_price is None else entry.reference_price
    )
    exit_reference_price = (
        exit_.price if exit_.reference_price is None else exit_.reference_price
    )
    gross_entry_value = entry.price * quantity
    gross_exit_value = exit_.price * quantity
    total_commission = entry.commission + exit_.commission
    gross_pnl = (exit_.price - entry.price) * quantity
    net_pnl = gross_pnl - total_commission
    return ClosedTrade(
        symbol=entry.symbol,
        entry_timestamp=entry.timestamp,
        exit_timestamp=exit_.timestamp,
        quantity=entry.quantity,
        entry_reference_price=entry_reference_price,
        entry_fill_price=entry.price,
        exit_reference_price=exit_reference_price,
        exit_fill_price=exit_.price,
        gross_entry_value=gross_entry_value,
        gross_exit_value=gross_exit_value,
        entry_commission=entry.commission,
        exit_commission=exit_.commission,
        total_commission=total_commission,
        entry_slippage_cost=entry.slippage_cost,
        exit_slippage_cost=exit_.slippage_cost,
        total_slippage_cost=entry.slippage_cost + exit_.slippage_cost,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        return_ratio=net_pnl / gross_entry_value,
        holding_period=exit_.timestamp - entry.timestamp,
        exit_execution_reason=exit_.execution_reason,
    )


def _summarise_closed_trades(
    closed_trades: tuple[ClosedTrade, ...],
    *,
    reconciled_net_pnl: Decimal,
) -> TradeStatistics:
    """Aggregate deterministic statistics from immutable closed trades."""

    completed_trade_count = len(closed_trades)
    if completed_trade_count == 0:
        return TradeStatistics(
            completed_trade_count=0,
            winning_trade_count=0,
            losing_trade_count=0,
            breakeven_trade_count=0,
            win_rate=None,
            loss_rate=None,
            gross_profit=_ZERO,
            gross_loss=_ZERO,
            net_closed_trade_pnl=_ZERO,
            average_net_pnl=None,
            average_winner=None,
            average_loser=None,
            largest_winner=None,
            largest_loser=None,
            profit_factor=None,
            payoff_ratio=None,
            expectancy=None,
            average_return_ratio=None,
            average_holding_period=None,
            longest_holding_period=None,
            shortest_holding_period=None,
            total_trade_commission=_ZERO,
            total_trade_slippage_cost=_ZERO,
        )

    winners = tuple(trade.net_pnl for trade in closed_trades if trade.net_pnl > _ZERO)
    losers = tuple(-trade.net_pnl for trade in closed_trades if trade.net_pnl < _ZERO)
    breakeven_trade_count = sum(trade.net_pnl == _ZERO for trade in closed_trades)
    count = Decimal(completed_trade_count)
    gross_profit = exact_decimal_sum(winners)
    gross_loss = exact_decimal_sum(losers)
    net_closed_trade_pnl = reconciled_net_pnl
    if (
        exact_decimal_sum((gross_profit, gross_loss.copy_negate()))
        != net_closed_trade_pnl
    ):
        # Preserve account-reconciled net P&L and place only the finite-context
        # aggregation residual in the loss magnitude.
        gross_loss = exact_decimal_sum(
            (gross_profit, net_closed_trade_pnl.copy_negate())
        )
    average_winner = (
        gross_profit / Decimal(len(winners)) if winners else None
    )
    average_loser = gross_loss / Decimal(len(losers)) if losers else None
    payoff_ratio = (
        average_winner / average_loser
        if average_winner is not None
        and average_loser is not None
        and average_loser != _ZERO
        else None
    )
    holding_periods = tuple(trade.holding_period for trade in closed_trades)
    total_holding_period = sum(holding_periods, timedelta(0))
    return TradeStatistics(
        completed_trade_count=completed_trade_count,
        winning_trade_count=len(winners),
        losing_trade_count=len(losers),
        breakeven_trade_count=breakeven_trade_count,
        win_rate=Decimal(len(winners)) / count,
        loss_rate=Decimal(len(losers)) / count,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_closed_trade_pnl=net_closed_trade_pnl,
        average_net_pnl=net_closed_trade_pnl / count,
        average_winner=average_winner,
        average_loser=average_loser,
        largest_winner=max(winners) if winners else None,
        largest_loser=max(losers) if losers else None,
        profit_factor=gross_profit / gross_loss if gross_loss != _ZERO else None,
        payoff_ratio=payoff_ratio,
        expectancy=net_closed_trade_pnl / count,
        average_return_ratio=exact_decimal_sum(
            trade.return_ratio for trade in closed_trades
        )
        / count,
        average_holding_period=total_holding_period / completed_trade_count,
        longest_holding_period=max(holding_periods),
        shortest_holding_period=min(holding_periods),
        total_trade_commission=exact_decimal_sum(
            trade.total_commission for trade in closed_trades
        ),
        total_trade_slippage_cost=exact_decimal_sum(
            trade.total_slippage_cost for trade in closed_trades
        ),
    )
