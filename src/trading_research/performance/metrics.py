"""Pure, deterministic performance metrics for reconciled backtests."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading_research.models import (
    AccountLedger,
    BacktestResult,
    EquityPoint,
    TradeSide,
)
from trading_research.performance._decimal import exact_decimal_sum

_ZERO = Decimal("0")
_MINIMUM_DRAWDOWN_RATIO = Decimal("-1")


def _require_finite_decimal(value: Decimal, name: str) -> None:
    """Require an exact, finite Decimal in a performance result."""

    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")


def _require_aware_timestamp(timestamp: datetime | None, name: str) -> None:
    """Require optional drawdown timestamps to include a UTC offset."""

    if timestamp is not None and (
        timestamp.tzinfo is None or timestamp.utcoffset() is None
    ):
        raise ValueError(f"{name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    """Immutable core metrics derived from one reconciled backtest result."""

    starting_cash: Decimal
    ending_cash: Decimal
    ending_equity: Decimal
    net_profit: Decimal
    total_return: Decimal
    gross_realised_pnl: Decimal
    unrealised_pnl: Decimal
    total_commission: Decimal
    total_adverse_slippage_cost: Decimal
    commission_ratio: Decimal
    slippage_cost_ratio: Decimal
    maximum_absolute_drawdown: Decimal
    maximum_percentage_drawdown: Decimal
    drawdown_peak_timestamp: datetime | None
    drawdown_trough_timestamp: datetime | None
    drawdown_recovery_timestamp: datetime | None
    open_quantity: int
    open_position_market_value: Decimal

    def __post_init__(self) -> None:
        """Validate metric types, signs, relationships, and timestamps."""

        financial_values = (
            ("starting_cash", self.starting_cash),
            ("ending_cash", self.ending_cash),
            ("ending_equity", self.ending_equity),
            ("net_profit", self.net_profit),
            ("total_return", self.total_return),
            ("gross_realised_pnl", self.gross_realised_pnl),
            ("unrealised_pnl", self.unrealised_pnl),
            ("total_commission", self.total_commission),
            ("total_adverse_slippage_cost", self.total_adverse_slippage_cost),
            ("commission_ratio", self.commission_ratio),
            ("slippage_cost_ratio", self.slippage_cost_ratio),
            ("maximum_absolute_drawdown", self.maximum_absolute_drawdown),
            ("maximum_percentage_drawdown", self.maximum_percentage_drawdown),
            ("open_position_market_value", self.open_position_market_value),
        )
        for name, value in financial_values:
            _require_finite_decimal(value, name)

        if self.starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        if self.ending_cash < 0 or self.ending_equity < 0:
            raise ValueError("ending cash and equity must be non-negative")
        if isinstance(self.open_quantity, bool) or not isinstance(self.open_quantity, int):
            raise TypeError("open_quantity must be an integer")
        if self.open_quantity < 0:
            raise ValueError("open_quantity must be non-negative")
        if self.open_position_market_value < 0:
            raise ValueError("open_position_market_value must be non-negative")
        if self.total_commission < 0 or self.total_adverse_slippage_cost < 0:
            raise ValueError("cost attribution must be non-negative")
        if self.commission_ratio < 0 or self.slippage_cost_ratio < 0:
            raise ValueError("cost ratios must be non-negative")
        if self.maximum_absolute_drawdown < 0:
            raise ValueError("maximum_absolute_drawdown must be non-negative")
        if not (
            _MINIMUM_DRAWDOWN_RATIO
            <= self.maximum_percentage_drawdown
            <= _ZERO
        ):
            raise ValueError("maximum_percentage_drawdown must be between -1 and 0")

        timestamps = (
            ("drawdown_peak_timestamp", self.drawdown_peak_timestamp),
            ("drawdown_trough_timestamp", self.drawdown_trough_timestamp),
            ("drawdown_recovery_timestamp", self.drawdown_recovery_timestamp),
        )
        for name, timestamp in timestamps:
            _require_aware_timestamp(timestamp, name)
        if self.maximum_absolute_drawdown == _ZERO:
            if self.maximum_percentage_drawdown != _ZERO or any(
                timestamp is not None for _, timestamp in timestamps
            ):
                raise ValueError("zero drawdown must have zero ratio and no timestamps")
        elif self.drawdown_trough_timestamp is None:
            raise ValueError("a positive drawdown requires a trough timestamp")
        if (
            self.drawdown_peak_timestamp is not None
            and self.drawdown_trough_timestamp is not None
            and self.drawdown_peak_timestamp >= self.drawdown_trough_timestamp
        ):
            raise ValueError("drawdown peak must precede its trough")
        if (
            self.drawdown_recovery_timestamp is not None
            and self.drawdown_trough_timestamp is not None
            and self.drawdown_recovery_timestamp <= self.drawdown_trough_timestamp
        ):
            raise ValueError("drawdown recovery must follow its trough")

        if self.net_profit != self.ending_equity - self.starting_cash:
            raise ValueError("net_profit must reconcile to ending equity")
        if self.total_return != self.net_profit / self.starting_cash:
            raise ValueError("total_return must equal net profit divided by starting cash")
        if self.commission_ratio != self.total_commission / self.starting_cash:
            raise ValueError("commission_ratio must reconcile to starting cash")
        if (
            self.slippage_cost_ratio
            != self.total_adverse_slippage_cost / self.starting_cash
        ):
            raise ValueError("slippage_cost_ratio must reconcile to starting cash")
        if (
            exact_decimal_sum(
                (
                    self.gross_realised_pnl,
                    self.unrealised_pnl,
                    self.total_commission.copy_negate(),
                )
            )
            != self.net_profit
        ):
            raise ValueError("gross and unrealised P&L must reconcile to net profit")


@dataclass(frozen=True, slots=True)
class _DrawdownStatistics:
    """Internal drawdown calculation result."""

    maximum_absolute: Decimal
    maximum_percentage: Decimal
    peak_timestamp: datetime | None
    trough_timestamp: datetime | None
    recovery_timestamp: datetime | None


def calculate_performance(result: BacktestResult) -> PerformanceSummary:
    """Calculate core metrics without changing a reconciled backtest result.

    Drawdown calculation seeds its running peak with starting cash. Timestamps
    describe the maximum absolute drawdown; equal depths keep the earliest
    trough. A seed-only peak has no timestamp, and a zero drawdown has no
    peak, trough, or recovery timestamps.
    """

    if not isinstance(result, BacktestResult):
        raise TypeError("result must be a BacktestResult")
    reconciliation = result.reconciliation
    if not reconciliation.is_reconciled:
        raise ValueError("performance requires an exactly reconciled backtest")

    ledger = reconciliation.ledger
    gross_realised_pnl, open_fill_cost = _calculate_pnl_attribution(ledger)
    open_position_market_value = reconciliation.remaining_position_value
    unrealised_pnl = open_position_market_value - open_fill_cost
    total_commission = ledger.total_commissions
    total_adverse_slippage_cost = sum(
        (
            entry.slippage_cost
            for entry in ledger.entries
            if entry.slippage_cost > _ZERO
        ),
        _ZERO,
    )
    net_profit = reconciliation.ending_equity - ledger.starting_cash
    attributed_net_profit = exact_decimal_sum(
        (
            gross_realised_pnl,
            unrealised_pnl,
            total_commission.copy_negate(),
        )
    )
    if attributed_net_profit != net_profit:
        # The account replay is authoritative. Finite-precision cash updates can
        # differ infinitesimally from regrouped P&L components, so realised P&L
        # carries that exact residual instead of weakening reconciliation.
        gross_realised_pnl = exact_decimal_sum(
            (net_profit, unrealised_pnl.copy_negate(), total_commission)
        )

    drawdown = _calculate_drawdown(ledger.starting_cash, result.equity_curve)
    return PerformanceSummary(
        starting_cash=ledger.starting_cash,
        ending_cash=reconciliation.ending_cash,
        ending_equity=reconciliation.ending_equity,
        net_profit=net_profit,
        total_return=net_profit / ledger.starting_cash,
        gross_realised_pnl=gross_realised_pnl,
        unrealised_pnl=unrealised_pnl,
        total_commission=total_commission,
        total_adverse_slippage_cost=total_adverse_slippage_cost,
        commission_ratio=total_commission / ledger.starting_cash,
        slippage_cost_ratio=total_adverse_slippage_cost / ledger.starting_cash,
        maximum_absolute_drawdown=drawdown.maximum_absolute,
        maximum_percentage_drawdown=drawdown.maximum_percentage,
        drawdown_peak_timestamp=drawdown.peak_timestamp,
        drawdown_trough_timestamp=drawdown.trough_timestamp,
        drawdown_recovery_timestamp=drawdown.recovery_timestamp,
        open_quantity=reconciliation.remaining_position_quantity,
        open_position_market_value=open_position_market_value,
    )


def _calculate_pnl_attribution(ledger: AccountLedger) -> tuple[Decimal, Decimal]:
    """Return gross realised P&L and remaining weighted-average fill cost."""

    open_quantity = 0
    open_fill_cost = _ZERO
    realised_movements: list[Decimal] = []

    for entry in ledger.entries:
        trade = entry.trade
        if trade.side is TradeSide.BUY:
            open_quantity += trade.quantity
            open_fill_cost += entry.gross_notional
            continue

        if trade.quantity == open_quantity:
            allocated_fill_cost = open_fill_cost
        else:
            allocated_fill_cost = (
                open_fill_cost * Decimal(trade.quantity) / Decimal(open_quantity)
            )
        realised_movements.append(entry.gross_notional - allocated_fill_cost)
        open_fill_cost -= allocated_fill_cost
        open_quantity -= trade.quantity
        if open_quantity == 0:
            open_fill_cost = _ZERO

    if open_quantity != ledger.ending_position_quantity:
        raise ValueError("performance position cost does not match the account ledger")
    return exact_decimal_sum(realised_movements), open_fill_cost


def _calculate_drawdown(
    starting_cash: Decimal,
    equity_curve: tuple[EquityPoint, ...],
) -> _DrawdownStatistics:
    """Calculate exact drawdowns with starting cash as the initial peak."""

    if not equity_curve:
        raise ValueError("performance requires a non-empty equity curve")

    running_peak = starting_cash
    running_peak_timestamp: datetime | None = None
    maximum_absolute = _ZERO
    maximum_percentage = _ZERO
    worst_peak = starting_cash
    worst_peak_timestamp: datetime | None = None
    worst_trough_timestamp: datetime | None = None
    worst_trough_index: int | None = None

    for index, point in enumerate(equity_curve):
        if point.equity > running_peak:
            running_peak = point.equity
            running_peak_timestamp = point.timestamp
            continue
        if point.equity == running_peak:
            if running_peak_timestamp is None:
                running_peak_timestamp = point.timestamp
            continue

        absolute_drawdown = running_peak - point.equity
        percentage_drawdown = (point.equity - running_peak) / running_peak
        if percentage_drawdown < maximum_percentage:
            maximum_percentage = percentage_drawdown
        if absolute_drawdown > maximum_absolute:
            maximum_absolute = absolute_drawdown
            worst_peak = running_peak
            worst_peak_timestamp = running_peak_timestamp
            worst_trough_timestamp = point.timestamp
            worst_trough_index = index

    if maximum_absolute == _ZERO:
        return _DrawdownStatistics(_ZERO, _ZERO, None, None, None)

    recovery_timestamp: datetime | None = None
    if worst_trough_index is not None:
        for point in equity_curve[worst_trough_index + 1 :]:
            if point.equity >= worst_peak:
                recovery_timestamp = point.timestamp
                break

    return _DrawdownStatistics(
        maximum_absolute=maximum_absolute,
        maximum_percentage=maximum_percentage,
        peak_timestamp=worst_peak_timestamp,
        trough_timestamp=worst_trough_timestamp,
        recovery_timestamp=recovery_timestamp,
    )
