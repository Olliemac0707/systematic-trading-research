"""Deterministic shared fixtures for split-evaluation tests."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from trading_research.config import BacktestConfig, PositionSizingMode
from trading_research.evaluation import (
    DatasetProvenance,
    DividendTreatment,
    EvaluationSplit,
    FrozenEvaluationConfiguration,
    PriceAdjustmentPolicy,
    SplitTreatment,
    WarmupPolicy,
)
from trading_research.experiments import BenchmarkSelection
from trading_research.models import EndOfTestPolicy, MarketBar
from trading_research.performance import RiskMetricSettings
from trading_research.reporting import StrategyRunConfiguration
from trading_research.strategies import (
    DonchianBreakoutParameters,
    SmaCrossoverParameters,
    StrategyName,
)

SYNTHETIC_CLOSES = (
    "10",
    "9",
    "8",
    "9",
    "10",
    "11",
    "10",
    "9",
    "8",
    "9",
    "10",
    "11",
    "10",
    "9",
    "8",
    "9",
)
SYNTHETIC_START = datetime(2025, 1, 1, tzinfo=UTC)


def synthetic_bars(
    *,
    closes: tuple[str, ...] = SYNTHETIC_CLOSES,
    symbol: str = "TEST",
) -> tuple[MarketBar, ...]:
    """Return chronological daily bars with exact Decimal OHLC values."""

    bars: list[MarketBar] = []
    for index, close_text in enumerate(closes):
        close = Decimal(close_text)
        bars.append(
            MarketBar(
                symbol=symbol,
                timestamp=SYNTHETIC_START + timedelta(days=index),
                open=close,
                high=close + Decimal("0.5"),
                low=close - Decimal("0.5"),
                close=close,
                volume=0 if index == 5 else 1000,
            )
        )
    return tuple(bars)


def write_bars_csv(path: Path, bars: tuple[MarketBar, ...]) -> Path:
    """Write validated synthetic bars in the existing provider schema."""

    rows = ["symbol,timestamp,open,high,low,close,volume"]
    rows.extend(
        ",".join(
            (
                bar.symbol,
                bar.timestamp.isoformat(),
                str(bar.open),
                str(bar.high),
                str(bar.low),
                str(bar.close),
                str(bar.volume),
            )
        )
        for bar in bars
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def provenance() -> DatasetProvenance:
    """Return explicit unknown-adjustment provenance for synthetic bars."""

    return DatasetProvenance(
        dataset_id="synthetic-daily-v1",
        source_name="unit-test fixture",
        source_url=None,
        downloaded_at=datetime(2025, 1, 20, tzinfo=UTC),
        price_adjustment=PriceAdjustmentPolicy.UNKNOWN,
        dividend_treatment=DividendTreatment.UNKNOWN,
        split_treatment=SplitTreatment.UNKNOWN,
        bar_frequency="daily",
        exchange_timezone="UTC",
        currency="USD",
        notes="Deterministic local test data.",
    )


def evaluation_split(
    warmup_policy: WarmupPolicy = WarmupPolicy.ISOLATED,
) -> EvaluationSplit:
    """Return two inclusive eight-observation periods."""

    return EvaluationSplit(
        development_start=SYNTHETIC_START,
        development_end=SYNTHETIC_START + timedelta(days=7),
        holdout_start=SYNTHETIC_START + timedelta(days=8),
        holdout_end=SYNTHETIC_START + timedelta(days=15),
        warmup_policy=warmup_policy,
    )


def frozen_configuration(
    *,
    strategy_name: StrategyName = StrategyName.SMA_CROSSOVER,
    benchmark: BenchmarkSelection = BenchmarkSelection.NONE,
    risk_metrics: bool = False,
    end_policy: EndOfTestPolicy = EndOfTestPolicy.HOLD,
    commission_bps: Decimal = Decimal("1"),
) -> FrozenEvaluationConfiguration:
    """Return one exact fixed-size configuration for both evaluation periods."""

    if strategy_name is StrategyName.SMA_CROSSOVER:
        parameters: SmaCrossoverParameters | DonchianBreakoutParameters = (
            SmaCrossoverParameters(fast_window=2, slow_window=3)
        )
    else:
        parameters = DonchianBreakoutParameters(entry_window=2, exit_window=2)
    return FrozenEvaluationConfiguration(
        strategy=StrategyRunConfiguration(
            name=strategy_name,
            parameters=parameters,
        ),
        backtest=BacktestConfig(
            initial_cash=Decimal("10000"),
            trade_quantity=10,
            commission_bps=commission_bps,
            slippage_bps=Decimal("5"),
            position_sizing_mode=PositionSizingMode.FIXED,
            end_of_test_policy=end_policy,
        ),
        risk_metrics=(
            RiskMetricSettings(
                periods_per_year=Decimal("252"),
                risk_free_rate_per_period=Decimal("0"),
                target_return_per_period=Decimal("0"),
            )
            if risk_metrics
            else None
        ),
        benchmark=benchmark,
    )
