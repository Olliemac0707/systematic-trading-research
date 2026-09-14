"""Base interface for replaceable backtesting engines."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from trading_research.config import BacktestConfig
from trading_research.models import BacktestResult, MarketBar
from trading_research.risk import PositionSizer, RiskPolicy
from trading_research.strategies import Strategy


class BacktestEngine(ABC):
    """Contract for engines that simulate a strategy over historical bars."""

    @abstractmethod
    def run(
        self,
        bars: Sequence[MarketBar],
        strategy: Strategy,
        config: BacktestConfig,
        *,
        position_sizer: PositionSizer | None = None,
        risk_policy: RiskPolicy | None = None,
    ) -> BacktestResult:
        """Run a simulation without contacting a broker or submitting orders."""
