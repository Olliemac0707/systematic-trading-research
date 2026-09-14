"""Base strategy contract."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime

from trading_research.models import MarketBar, StrategySignal


class Strategy(ABC):
    """Contract for pure strategies that transform bars into signals."""

    def required_market_data_fields(self) -> frozenset[str]:
        """Declare provider fields consumed by the strategy.

        Price-only is the safe default for current and legacy strategies. A future
        strategy that reads volume must explicitly include ``"volume"`` here so
        dataset provenance policy can reject unapproved use before a backtest.
        """

        return frozenset({"open", "high", "low", "close"})

    @abstractmethod
    def generate_signals(self, bars: Sequence[MarketBar]) -> Sequence[StrategySignal]:
        """Generate chronological signals without placing or submitting orders."""

    def required_warmup_bars(self) -> int:
        """Return the exact preceding observations needed for indicator state."""

        return 0

    def generate_signals_for_period(
        self,
        bars: Sequence[MarketBar],
        evaluation_start: datetime,
    ) -> Sequence[StrategySignal]:
        """Generate signals with history while exposing only evaluation-period signals."""

        return tuple(
            signal
            for signal in self.generate_signals(bars)
            if signal.timestamp >= evaluation_start
        )
