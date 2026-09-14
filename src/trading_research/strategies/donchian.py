"""Long-only closing-price Donchian breakout strategy."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from trading_research.errors import StrategyError
from trading_research.models import MarketBar, SignalAction, StrategySignal
from trading_research.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DonchianBreakoutStrategy(Strategy):
    """Emit long entry and exit signals from prior closing-price channels.

    The current close is excluded from both channels. Strict inequalities apply,
    and logical signal state prevents repeated entries, pyramiding, short signals,
    and repeated exits. The backtesting engine remains authoritative for whether a
    signal results in a sized, risk-approved simulated execution.
    """

    entry_window: int
    exit_window: int

    def __post_init__(self) -> None:
        """Require independent positive whole-number channel windows."""

        _validate_window(self.entry_window, "entry_window")
        _validate_window(self.exit_window, "exit_window")

    def generate_signals(self, bars: Sequence[MarketBar]) -> tuple[StrategySignal, ...]:
        """Return chronological next-bar-executable breakout signals."""

        snapshot = tuple(bars)
        if not snapshot:
            return ()
        _validate_bars(snapshot)

        return self._generate_signals(snapshot, evaluation_start=None)

    def required_warmup_bars(self) -> int:
        """Return the larger channel needed at a fresh evaluation boundary."""

        return max(self.entry_window, self.exit_window)

    def generate_signals_for_period(
        self,
        bars: Sequence[MarketBar],
        evaluation_start: datetime,
    ) -> tuple[StrategySignal, ...]:
        """Use prior channels but reset logical position state at the boundary."""

        snapshot = tuple(bars)
        if not snapshot:
            return ()
        _validate_bars(snapshot)
        if evaluation_start.tzinfo is None or evaluation_start.utcoffset() is None:
            raise ValueError("evaluation_start must be timezone-aware")
        return self._generate_signals(snapshot, evaluation_start=evaluation_start)

    def _generate_signals(
        self,
        snapshot: tuple[MarketBar, ...],
        *,
        evaluation_start: datetime | None,
    ) -> tuple[StrategySignal, ...]:
        """Generate channels, optionally suppressing all pre-boundary signal state."""

        signals: list[StrategySignal] = []
        logically_long = False
        closes = tuple(bar.close for bar in snapshot)

        for index, bar in enumerate(snapshot):
            if evaluation_start is not None and bar.timestamp < evaluation_start:
                continue
            if not logically_long and index >= self.entry_window:
                channel_high = max(closes[index - self.entry_window : index])
                if bar.close > channel_high:
                    signals.append(
                        StrategySignal(
                            symbol=bar.symbol,
                            timestamp=bar.timestamp,
                            action=SignalAction.BUY,
                            reason=(
                                f"close ({bar.close}) broke above prior "
                                f"{self.entry_window}-bar closing high ({channel_high})"
                            ),
                        )
                    )
                    logically_long = True
            elif logically_long and index >= self.exit_window:
                channel_low = min(closes[index - self.exit_window : index])
                if bar.close < channel_low:
                    signals.append(
                        StrategySignal(
                            symbol=bar.symbol,
                            timestamp=bar.timestamp,
                            action=SignalAction.SELL,
                            reason=(
                                f"close ({bar.close}) broke below prior "
                                f"{self.exit_window}-bar closing low ({channel_low})"
                            ),
                        )
                    )
                    logically_long = False

        logger.info("Generated %d Donchian breakout signals", len(signals))
        return tuple(signals)


def _validate_window(value: int, name: str) -> None:
    """Reject booleans, non-integers, and non-positive window values."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")


def _validate_bars(bars: tuple[MarketBar, ...]) -> None:
    """Require one symbol and strictly chronological observations."""

    if any(not isinstance(bar, MarketBar) for bar in bars):
        raise TypeError("strategy input must contain MarketBar values")
    if len({bar.symbol for bar in bars}) != 1:
        raise StrategyError("strategy input must contain exactly one symbol")
    if any(
        current.timestamp <= previous.timestamp
        for previous, current in zip(bars, bars[1:], strict=False)
    ):
        raise StrategyError("strategy input bars must be strictly chronological")
