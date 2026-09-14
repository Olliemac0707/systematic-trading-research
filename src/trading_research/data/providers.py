"""Interfaces implemented by replaceable historical market-data providers."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime

from trading_research.models import MarketBar


class MarketDataProvider(ABC):
    """Contract for fetching normalized, read-only historical market bars."""

    @abstractmethod
    def get_historical_bars(
        self,
        symbol: str,
        start: datetime | None,
        end: datetime | None,
    ) -> Sequence[MarketBar]:
        """Return chronological bars for ``symbol`` within optional time boundaries."""
