"""Project-specific exception hierarchy."""


class TradingResearchError(Exception):
    """Base exception for recoverable project errors."""


class MarketDataError(TradingResearchError):
    """Raised when market data cannot be loaded or validated as a collection."""


class MarketDataSchemaError(MarketDataError):
    """Raised when a market-data file has an invalid structure."""


class MarketDataValidationError(MarketDataError):
    """Raised when market-data observations or request filters are invalid."""


class ApprovedDatasetError(MarketDataError):
    """Raised when the approved ETF dataset workflow cannot complete safely."""


class ApprovedDatasetProviderError(ApprovedDatasetError):
    """Raised for provider-level transport, authentication, or schema failures."""


class StrategyError(TradingResearchError):
    """Raised when a strategy cannot evaluate its input."""


class BacktestError(TradingResearchError):
    """Raised when a backtest cannot be completed safely."""


class ReportWriteError(TradingResearchError):
    """Raised when a requested local report cannot be written safely."""


class ExperimentManifestError(TradingResearchError):
    """Raised when a batch experiment manifest is structurally invalid."""


class EvaluationManifestError(TradingResearchError):
    """Raised when a split-evaluation manifest is structurally invalid."""
