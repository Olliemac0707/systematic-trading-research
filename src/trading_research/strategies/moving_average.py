"""A dependency-free simple moving-average crossover example."""

import logging
from collections.abc import Sequence
from decimal import Decimal

from trading_research.errors import StrategyError
from trading_research.models import MarketBar, SignalAction, StrategySignal
from trading_research.strategies.base import Strategy

logger = logging.getLogger(__name__)


class SimpleMovingAverageCrossover(Strategy):
    """Emit signals when a short SMA crosses a longer SMA.

    At least two fully formed average pairs are required to identify a crossover.
    An existing trend at the start of a dataset does not itself produce a signal.
    """

    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        """Create the strategy with positive, ordered window lengths."""

        if short_window <= 0 or long_window <= 0:
            raise ValueError("moving-average windows must be positive")
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, bars: Sequence[MarketBar]) -> Sequence[StrategySignal]:
        """Return crossover signals for chronological bars from a single symbol.

        Raises:
            StrategyError: If bars contain multiple symbols or are out of order.
        """

        if not bars:
            return ()

        symbols = {bar.symbol for bar in bars}
        if len(symbols) != 1:
            raise StrategyError("strategy input must contain exactly one symbol")
        if any(
            current.timestamp <= previous.timestamp
            for previous, current in zip(bars, bars[1:], strict=False)
        ):
            raise StrategyError("strategy input bars must be strictly chronological")

        if len(bars) < self.long_window:
            logger.debug("Not enough bars to form the long moving average")
            return ()

        signals: list[StrategySignal] = []
        previous_relation: int | None = None
        closes = [bar.close for bar in bars]

        for index in range(self.long_window - 1, len(bars)):
            short_sma = self._mean(closes[index - self.short_window + 1 : index + 1])
            long_sma = self._mean(closes[index - self.long_window + 1 : index + 1])
            relation = (short_sma > long_sma) - (short_sma < long_sma)

            action: SignalAction | None = None
            if previous_relation is not None:
                if relation > 0 and previous_relation <= 0:
                    action = SignalAction.BUY
                elif relation < 0 and previous_relation >= 0:
                    action = SignalAction.SELL

            if action is not None:
                signals.append(
                    StrategySignal(
                        symbol=bars[index].symbol,
                        timestamp=bars[index].timestamp,
                        action=action,
                        reason=(
                            f"short SMA ({short_sma}) crossed "
                            f"{'above' if action is SignalAction.BUY else 'below'} "
                            f"long SMA ({long_sma})"
                        ),
                    )
                )

            previous_relation = relation

        logger.info("Generated %d crossover signals", len(signals))
        return tuple(signals)

    def required_warmup_bars(self) -> int:
        """Return enough closes to retain the relation before the boundary."""

        return self.long_window

    @staticmethod
    def _mean(values: Sequence[Decimal]) -> Decimal:
        """Calculate an exact arithmetic mean for decimal prices."""

        return sum(values, start=Decimal("0")) / Decimal(len(values))
