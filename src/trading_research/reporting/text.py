"""Plain-text reporting for reconciled backtest results."""

from datetime import datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal

from trading_research.benchmarks import BenchmarkAnalysis, BenchmarkComparison
from trading_research.models import BacktestResult, EndOfTestPolicy, ExecutionReason
from trading_research.performance import (
    ClosedTrade,
    ExposureStatistics,
    PerformanceSummary,
    RiskAdjustedMetrics,
    RiskMetricSettings,
    TradeStatistics,
    calculate_exposure_statistics,
    calculate_performance,
    calculate_risk_adjusted_metrics,
    calculate_trade_statistics,
    reconstruct_closed_trades,
)
from trading_research.reporting.metadata import StrategyRunConfiguration
from trading_research.strategies import (
    DonchianBreakoutParameters,
    SmaCrossoverParameters,
)

_CURRENCY_QUANTUM = Decimal("0.01")
_DISPLAY_QUANTUM = Decimal("0.01")
_HUNDRED = Decimal("100")
_ZERO = Decimal("0")
_MISSING = "Not available"
_LABEL_WIDTH = 35


def format_decimal_currency(
    value: Decimal,
    currency_symbol: str | None = None,
) -> str:
    """Format an exact financial value with deterministic two-place rounding.

    ``None`` selects the neutral default with no symbol. A supplied symbol is
    placed after any minus sign; no currency is inferred from a market symbol.
    """

    _require_finite_decimal(value, "value")
    symbol = _validate_currency_symbol(currency_symbol)
    rounded = value.quantize(_CURRENCY_QUANTUM, rounding=ROUND_HALF_EVEN)
    if rounded == _ZERO:
        rounded = _ZERO.quantize(_CURRENCY_QUANTUM)
    sign = "-" if rounded < _ZERO else ""
    magnitude = -rounded if rounded < _ZERO else rounded
    return f"{sign}{symbol}{magnitude:,.2f}"


def format_backtest_report(
    result: BacktestResult,
    *,
    currency_symbol: str | None = None,
    include_closed_trades: bool = False,
    risk_metric_settings: RiskMetricSettings | None = None,
    strategy_configuration: StrategyRunConfiguration | None = None,
    benchmark: BenchmarkAnalysis | None = None,
    benchmark_comparison: BenchmarkComparison | None = None,
) -> str:
    """Return a deterministic report derived from one reconciled backtest.

    The function performs no output or I/O. Performance and trade summaries are
    calculated through their existing public APIs so accounting formulas remain
    outside the reporting layer.
    """

    if not isinstance(result, BacktestResult):
        raise TypeError("result must be a BacktestResult")
    if not isinstance(include_closed_trades, bool):
        raise TypeError("include_closed_trades must be a bool")
    if strategy_configuration is not None and not isinstance(
        strategy_configuration,
        StrategyRunConfiguration,
    ):
        raise TypeError("strategy_configuration must be StrategyRunConfiguration or None")
    if (benchmark is None) != (benchmark_comparison is None):
        raise ValueError("benchmark and benchmark_comparison must be supplied together")
    _validate_currency_symbol(currency_symbol)
    symbol = currency_symbol

    reconciliation = result.reconciliation
    if (
        not reconciliation.is_reconciled
        or reconciliation.cash_difference != _ZERO
        or reconciliation.reconciliation_difference != _ZERO
    ):
        raise ValueError("reporting requires an exactly reconciled backtest")

    performance = calculate_performance(result)
    statistics = calculate_trade_statistics(result)
    exposure = calculate_exposure_statistics(result)
    report_sections = [
        _report_header(),
        _format_period(result, strategy_configuration),
        _format_account_summary(performance, symbol),
        _format_profit_and_loss(performance, symbol),
        _format_risk(performance, symbol),
        _format_exposure(exposure, symbol),
        _format_order_decisions(exposure),
    ]
    risk_metrics = (
        None
        if risk_metric_settings is None
        else calculate_risk_adjusted_metrics(result, risk_metric_settings)
    )
    if risk_metrics is not None:
        report_sections.append(
            _format_return_risk_statistics(risk_metrics)
        )
    if benchmark is not None and benchmark_comparison is not None:
        if benchmark_comparison.benchmark_name != benchmark.name:
            raise ValueError("benchmark comparison must describe the supplied benchmark")
        if benchmark.assumptions.risk_metric_settings != risk_metric_settings:
            raise ValueError("benchmark and strategy risk metric settings must match")
        _validate_benchmark_report_inputs(
            benchmark,
            benchmark_comparison,
            performance,
            exposure,
            risk_metrics,
        )
        report_sections.append(
            _format_benchmark_comparison(
                benchmark,
                benchmark_comparison,
                symbol,
            )
        )
    report_sections.extend(
        (
            _format_trade_statistics(statistics, symbol),
            _format_open_position(performance, symbol),
            _format_reconciliation(result, symbol),
        )
    )
    if include_closed_trades:
        report_sections.append(
            _format_closed_trade_details(reconstruct_closed_trades(result), symbol)
        )
    if result.was_end_of_test_liquidated:
        report_sections.append(
            "Final open position was synthetically liquidated at the final bar close."
        )
    report_sections.append(
        "Note: Slippage is already reflected in execution fill prices and is shown "
        "as attribution only."
    )
    return "\n\n".join(report_sections) + "\n"


def _validate_benchmark_report_inputs(
    benchmark: BenchmarkAnalysis,
    comparison: BenchmarkComparison,
    strategy_performance: PerformanceSummary,
    strategy_exposure: ExposureStatistics,
    strategy_risk: RiskAdjustedMetrics | None,
) -> None:
    """Reject benchmark text assembled from unrelated calculation results."""

    expected_values = {
        "strategy_ending_equity": strategy_performance.ending_equity,
        "benchmark_ending_equity": benchmark.performance.ending_equity,
        "strategy_total_return": strategy_performance.total_return,
        "benchmark_total_return": benchmark.performance.total_return,
        "strategy_maximum_percentage_drawdown": (
            strategy_performance.maximum_percentage_drawdown
        ),
        "benchmark_maximum_percentage_drawdown": (
            benchmark.performance.maximum_percentage_drawdown
        ),
        "strategy_time_in_market_ratio": strategy_exposure.time_in_market_ratio,
        "benchmark_time_in_market_ratio": (
            benchmark.exposure_statistics.time_in_market_ratio
        ),
    }
    for attribute in ("periodic_sharpe_ratio", "periodic_sortino_ratio"):
        expected_values[f"strategy_{attribute}"] = (
            None if strategy_risk is None else getattr(strategy_risk, attribute)
        )
        expected_values[f"benchmark_{attribute}"] = (
            None
            if benchmark.risk_adjusted_metrics is None
            else getattr(benchmark.risk_adjusted_metrics, attribute)
        )
    for attribute, expected in expected_values.items():
        if getattr(comparison, attribute) != expected:
            raise ValueError(f"comparison {attribute} must match report calculations")


def _format_benchmark_comparison(
    benchmark: BenchmarkAnalysis,
    comparison: BenchmarkComparison,
    currency_symbol: str | None,
) -> str:
    """Format an opt-in, capital-matched buy-and-hold comparison."""

    lines = [
        _line("Benchmark convention", "Next-bar executable"),
        _line(
            "Benchmark ending equity",
            format_decimal_currency(
                comparison.benchmark_ending_equity,
                currency_symbol,
            ),
        ),
        _line(
            "Strategy ending equity",
            format_decimal_currency(
                comparison.strategy_ending_equity,
                currency_symbol,
            ),
        ),
        _line(
            "Ending-equity difference",
            format_decimal_currency(
                comparison.ending_equity_difference,
                currency_symbol,
            ),
        ),
        _line("Benchmark total return", _format_percentage(comparison.benchmark_total_return)),
        _line("Strategy total return", _format_percentage(comparison.strategy_total_return)),
        _line("Excess return", _format_percentage(comparison.excess_return)),
        _line(
            "Benchmark maximum drawdown",
            _format_percentage(comparison.benchmark_maximum_percentage_drawdown),
        ),
        _line(
            "Strategy maximum drawdown",
            _format_percentage(comparison.strategy_maximum_percentage_drawdown),
        ),
        _line(
            "Drawdown improvement",
            _format_percentage(comparison.maximum_drawdown_improvement),
        ),
        _line(
            "Benchmark time in market",
            _format_percentage(comparison.benchmark_time_in_market_ratio),
        ),
        _line(
            "Strategy time in market",
            _format_percentage(comparison.strategy_time_in_market_ratio),
        ),
        _line(
            "Periodic Sharpe difference",
            _format_decimal(comparison.periodic_sharpe_difference),
        ),
        _line(
            "Periodic Sortino difference",
            _format_decimal(comparison.periodic_sortino_difference),
        ),
        "Note: The benchmark uses the same costs, sizing, and risk controls.",
        (
            "Note: Positive drawdown improvement means the strategy's "
            "non-positive maximum drawdown was less severe."
        ),
    ]
    if benchmark.result.was_end_of_test_liquidated:
        lines.append("Benchmark final position was synthetically liquidated.")
    return _section("Buy-and-Hold Benchmark", tuple(lines))


def _report_header() -> str:
    """Return the stable report title."""

    title = "BACKTEST PERFORMANCE REPORT"
    return f"{title}\n{'=' * len(title)}"


def _format_period(
    result: BacktestResult,
    strategy_configuration: StrategyRunConfiguration | None,
) -> str:
    """Format metadata that is actually available on a backtest result."""

    market_symbol = result.trades[0].symbol if result.trades else _MISSING
    strategy_lines = _format_strategy_metadata(strategy_configuration)
    return _section(
        "Period",
        (
            _line("Symbol", market_symbol),
            _line("Start", _format_timestamp(result.equity_curve[0].timestamp)),
            _line("End", _format_timestamp(result.equity_curve[-1].timestamp)),
            _line("Market bars", str(len(result.equity_curve))),
        )
        + strategy_lines
        + (
            _line("End-of-test policy", _format_end_of_test_policy(result)),
            _line("Open position remains", _yes_no(result.final_position_quantity > 0)),
        ),
    )


def _format_strategy_metadata(
    strategy_configuration: StrategyRunConfiguration | None,
) -> tuple[str, ...]:
    """Return display labels for only the selected strategy's parameters."""

    if strategy_configuration is None:
        return (_line("Strategy", _MISSING),)
    parameters = strategy_configuration.parameters
    if isinstance(parameters, SmaCrossoverParameters):
        return (
            _line("Strategy", "SMA crossover"),
            _line("Fast window", str(parameters.fast_window)),
            _line("Slow window", str(parameters.slow_window)),
        )
    if isinstance(parameters, DonchianBreakoutParameters):
        return (
            _line("Strategy", "Donchian breakout"),
            _line("Entry window", str(parameters.entry_window)),
            _line("Exit window", str(parameters.exit_window)),
        )
    raise TypeError("unsupported strategy parameter model")


def _format_account_summary(
    performance: PerformanceSummary,
    currency_symbol: str | None,
) -> str:
    """Format cash, position value, and equity as distinct values."""

    return _section(
        "Account Summary",
        (
            _line(
                "Starting cash",
                format_decimal_currency(performance.starting_cash, currency_symbol),
            ),
            _line(
                "Ending cash",
                format_decimal_currency(performance.ending_cash, currency_symbol),
            ),
            _line(
                "Open-position market value",
                format_decimal_currency(
                    performance.open_position_market_value,
                    currency_symbol,
                ),
            ),
            _line(
                "Ending equity",
                format_decimal_currency(performance.ending_equity, currency_symbol),
            ),
            _line(
                "Net profit",
                format_decimal_currency(performance.net_profit, currency_symbol),
            ),
            _line("Total return", _format_percentage(performance.total_return)),
            _line(
                "Equity composition",
                "Ending cash + open-position market value",
            ),
        ),
    )


def _format_profit_and_loss(
    performance: PerformanceSummary,
    currency_symbol: str | None,
) -> str:
    """Format realised, unrealised, and cost attribution values."""

    return _section(
        "Profit and Loss",
        (
            _line(
                "Gross realised P&L",
                format_decimal_currency(
                    performance.gross_realised_pnl,
                    currency_symbol,
                ),
            ),
            _line(
                "Unrealised P&L",
                format_decimal_currency(performance.unrealised_pnl, currency_symbol),
            ),
            _line(
                "Total commissions",
                format_decimal_currency(performance.total_commission, currency_symbol),
            ),
            _line(
                "Adverse slippage attribution",
                format_decimal_currency(
                    performance.total_adverse_slippage_cost,
                    currency_symbol,
                ),
            ),
            _line(
                "Net profit",
                format_decimal_currency(performance.net_profit, currency_symbol),
            ),
        ),
    )


def _format_risk(
    performance: PerformanceSummary,
    currency_symbol: str | None,
) -> str:
    """Format maximum drawdown and its observed lifecycle."""

    if (
        performance.maximum_absolute_drawdown > _ZERO
        and performance.drawdown_recovery_timestamp is None
    ):
        recovery = "Not recovered by end of backtest"
    else:
        recovery = _format_timestamp(performance.drawdown_recovery_timestamp)
    return _section(
        "Risk",
        (
            _line(
                "Maximum absolute drawdown",
                format_decimal_currency(
                    performance.maximum_absolute_drawdown,
                    currency_symbol,
                ),
            ),
            _line(
                "Maximum percentage drawdown",
                _format_percentage(performance.maximum_percentage_drawdown),
            ),
            _line(
                "Drawdown peak timestamp",
                _format_timestamp(performance.drawdown_peak_timestamp),
            ),
            _line(
                "Drawdown trough timestamp",
                _format_timestamp(performance.drawdown_trough_timestamp),
            ),
            _line("Drawdown recovery timestamp", recovery),
        ),
    )


def _format_trade_statistics(
    statistics: TradeStatistics,
    currency_symbol: str | None,
) -> str:
    """Format all existing completed-trade statistics."""

    return _section(
        "Trade Statistics",
        (
            _line("Completed trades", str(statistics.completed_trade_count)),
            _line("Winning trades", str(statistics.winning_trade_count)),
            _line("Losing trades", str(statistics.losing_trade_count)),
            _line("Breakeven trades", str(statistics.breakeven_trade_count)),
            _line("Win rate", _format_percentage(statistics.win_rate)),
            _line(
                "Gross profit",
                format_decimal_currency(statistics.gross_profit, currency_symbol),
            ),
            _line(
                "Gross loss",
                format_decimal_currency(statistics.gross_loss, currency_symbol),
            ),
            _line(
                "Net closed-trade P&L",
                format_decimal_currency(
                    statistics.net_closed_trade_pnl,
                    currency_symbol,
                ),
            ),
            _line(
                "Average trade P&L",
                _format_optional_currency(statistics.average_net_pnl, currency_symbol),
            ),
            _line(
                "Average winner",
                _format_optional_currency(statistics.average_winner, currency_symbol),
            ),
            _line(
                "Average loser",
                _format_optional_currency(statistics.average_loser, currency_symbol),
            ),
            _line(
                "Largest winner",
                _format_optional_currency(statistics.largest_winner, currency_symbol),
            ),
            _line(
                "Largest loser",
                _format_optional_currency(statistics.largest_loser, currency_symbol),
            ),
            _line("Profit factor", _format_decimal(statistics.profit_factor)),
            _line("Payoff ratio", _format_decimal(statistics.payoff_ratio)),
            _line(
                "Expectancy",
                _format_optional_currency(statistics.expectancy, currency_symbol),
            ),
            _line(
                "Average trade return",
                _format_percentage(statistics.average_return_ratio),
            ),
            _line(
                "Average holding period",
                _format_duration(statistics.average_holding_period),
            ),
            _line(
                "Shortest holding period",
                _format_duration(statistics.shortest_holding_period),
            ),
            _line(
                "Longest holding period",
                _format_duration(statistics.longest_holding_period),
            ),
            _line(
                "Trade commissions",
                format_decimal_currency(
                    statistics.total_trade_commission,
                    currency_symbol,
                ),
            ),
            _line(
                "Trade slippage attribution",
                format_decimal_currency(
                    statistics.total_trade_slippage_cost,
                    currency_symbol,
                ),
            ),
        ),
    )


def _format_exposure(
    statistics: ExposureStatistics,
    currency_symbol: str | None,
) -> str:
    """Format observation-based long exposure and account composition."""

    return _section(
        "Exposure and Capital Use",
        (
            _line("Equity observations", str(statistics.observation_count)),
            _line(
                "Invested observations",
                str(statistics.invested_observation_count),
            ),
            _line(
                "Cash-only observations",
                str(statistics.cash_only_observation_count),
            ),
            _line(
                "Time in market (observations)",
                _format_percentage(statistics.time_in_market_ratio),
            ),
            _line(
                "Average position quantity",
                _format_decimal(statistics.average_position_quantity),
            ),
            _line(
                "Maximum position quantity",
                _format_decimal(statistics.maximum_position_quantity),
            ),
            _line(
                "Average position value",
                _format_optional_currency(
                    statistics.average_position_market_value,
                    currency_symbol,
                ),
            ),
            _line(
                "Maximum position value",
                format_decimal_currency(
                    statistics.maximum_position_market_value,
                    currency_symbol,
                ),
            ),
            _line(
                "Average capital utilisation",
                _format_percentage(
                    statistics.average_capital_utilisation_ratio,
                ),
            ),
            _line(
                "Maximum capital utilisation",
                _format_percentage(
                    statistics.maximum_capital_utilisation_ratio,
                ),
            ),
            _line(
                "Average cash",
                _format_optional_currency(statistics.average_cash, currency_symbol),
            ),
            _line(
                "Minimum cash",
                _format_optional_currency(statistics.minimum_cash, currency_symbol),
            ),
            _line(
                "Average cash ratio",
                _format_percentage(statistics.average_cash_ratio),
            ),
            _line(
                "Minimum cash ratio",
                _format_percentage(statistics.minimum_cash_ratio),
            ),
            "Note: Time in market is observation-based; it is not elapsed-time weighted.",
        ),
    )


def _format_order_decisions(statistics: ExposureStatistics) -> str:
    """Format audited entry proposals separately from executed trades."""

    return _section(
        "Order Decisions",
        (
            _line("Actionable proposals", str(statistics.actionable_signal_count)),
            _line("Fully approved", str(statistics.full_approval_count)),
            _line("Reduced", str(statistics.reduced_decision_count)),
            _line("Rejected", str(statistics.rejected_decision_count)),
            _line(
                "Execution approval rate",
                _format_percentage(statistics.execution_approval_rate),
            ),
            _line(
                "Full approval rate",
                _format_percentage(statistics.full_approval_rate),
            ),
            _line("Reduction rate", _format_percentage(statistics.reduction_rate)),
            _line("Rejection rate", _format_percentage(statistics.rejection_rate)),
            _line(
                "Proposed quantity",
                _format_decimal(statistics.proposed_quantity_total),
            ),
            _line(
                "Approved quantity",
                _format_decimal(statistics.approved_quantity_total),
            ),
            _line(
                "Quantity prevented or reduced",
                _format_decimal(statistics.quantity_reduction_total),
            ),
        ),
    )


def _format_return_risk_statistics(metrics: RiskAdjustedMetrics) -> str:
    """Format pre-calculated return and risk statistics without recomputing them."""

    return _section(
        "Return and Risk Statistics",
        (
            _line("Return observations", str(metrics.observation_count)),
            _line("Cumulative return", _format_percentage(metrics.cumulative_return)),
            _line(
                "Mean periodic return",
                _format_percentage(metrics.arithmetic_mean_return),
            ),
            _line(
                "Periodic volatility",
                _format_percentage(metrics.periodic_volatility),
            ),
            _line(
                "Annualised volatility",
                _format_percentage(metrics.annualised_volatility),
            ),
            _line(
                "Mean excess return",
                _format_percentage(metrics.mean_excess_return),
            ),
            _line(
                "Periodic Sharpe ratio",
                _format_decimal(metrics.periodic_sharpe_ratio),
            ),
            _line(
                "Annualised Sharpe ratio",
                _format_decimal(metrics.annualised_sharpe_ratio),
            ),
            _line(
                "Downside deviation",
                _format_percentage(metrics.downside_deviation),
            ),
            _line(
                "Periodic Sortino ratio",
                _format_decimal(metrics.periodic_sortino_ratio),
            ),
            _line(
                "Annualised Sortino ratio",
                _format_decimal(metrics.annualised_sortino_ratio),
            ),
            _line(
                "Risk-free rate per period",
                _format_percentage(metrics.risk_free_rate_per_period),
            ),
            _line(
                "Target return per period",
                _format_percentage(metrics.target_return_per_period),
            ),
            _line(
                "Periods per year",
                _format_periods_per_year(metrics.periods_per_year),
            ),
        ),
    )


def _format_open_position(
    performance: PerformanceSummary,
    currency_symbol: str | None,
) -> str:
    """Always state whether the result contains a remaining position."""

    if performance.open_quantity == 0:
        return _section("Open Position", ("No open position",))
    return _section(
        "Open Position",
        (
            _line("Open quantity", str(performance.open_quantity)),
            _line(
                "Open-position market value",
                format_decimal_currency(
                    performance.open_position_market_value,
                    currency_symbol,
                ),
            ),
            _line(
                "Unrealised P&L",
                format_decimal_currency(performance.unrealised_pnl, currency_symbol),
            ),
        ),
    )


def _format_reconciliation(
    result: BacktestResult,
    currency_symbol: str | None,
) -> str:
    """Surface both exact reconciliation differences and their status."""

    reconciliation = result.reconciliation
    return _section(
        "Reconciliation",
        (
            _line(
                "Cash reconciliation difference",
                format_decimal_currency(
                    reconciliation.cash_difference,
                    currency_symbol,
                ),
            ),
            _line(
                "Equity reconciliation difference",
                format_decimal_currency(
                    reconciliation.reconciliation_difference,
                    currency_symbol,
                ),
            ),
            _line("Reconciliation status", "PASS"),
        ),
    )


def _format_closed_trade_details(
    closed_trades: tuple[ClosedTrade, ...],
    currency_symbol: str | None,
) -> str:
    """Format reconstructed closed trades in chronological order."""

    if not closed_trades:
        return _section("Closed Trade Details", ("No completed trades",))
    header = (
        "Entry | Exit | Quantity | Entry Fill | Exit Fill | Net P&L | Return | "
        "Holding Period | Exit Reason"
    )
    separator = "-" * len(header)
    rows = tuple(
        _format_closed_trade_row(trade, currency_symbol) for trade in closed_trades
    )
    return _section("Closed Trade Details", (header, separator, *rows))


def _format_closed_trade_row(
    trade: ClosedTrade,
    currency_symbol: str | None,
) -> str:
    """Format one immutable reconstructed trade without recalculating it."""

    return " | ".join(
        (
            _format_timestamp(trade.entry_timestamp),
            _format_timestamp(trade.exit_timestamp),
            str(trade.quantity),
            format_decimal_currency(trade.entry_fill_price, currency_symbol),
            format_decimal_currency(trade.exit_fill_price, currency_symbol),
            format_decimal_currency(trade.net_pnl, currency_symbol),
            _format_percentage(trade.return_ratio),
            _format_duration(trade.holding_period),
            _format_execution_reason(trade.exit_execution_reason),
        )
    )


def _format_end_of_test_policy(result: BacktestResult) -> str:
    """Describe the configured final-position convention."""

    if result.end_of_test_policy is EndOfTestPolicy.HOLD:
        return "Hold open position"
    return "Liquidate at final close"


def _format_execution_reason(reason: ExecutionReason) -> str:
    """Return a concise human-readable simulated execution reason."""

    if reason is ExecutionReason.END_OF_TEST_LIQUIDATION:
        return "End-of-test liquidation"
    return "Strategy signal"


def _format_optional_currency(
    value: Decimal | None,
    currency_symbol: str | None,
) -> str:
    """Format an optional financial value using the missing-value policy."""

    if value is None:
        return _MISSING
    return format_decimal_currency(value, currency_symbol)


def _format_percentage(value: Decimal | None) -> str:
    """Format an optional exact ratio as a two-place percentage."""

    if value is None:
        return _MISSING
    _require_finite_decimal(value, "ratio")
    percentage = (value * _HUNDRED).quantize(
        _DISPLAY_QUANTUM,
        rounding=ROUND_HALF_EVEN,
    )
    if percentage == _ZERO:
        percentage = _ZERO.quantize(_DISPLAY_QUANTUM)
    return f"{percentage:,.2f}%"


def _format_decimal(value: Decimal | None) -> str:
    """Format an optional unitless Decimal without converting through float."""

    if value is None:
        return _MISSING
    _require_finite_decimal(value, "value")
    rounded = value.quantize(_DISPLAY_QUANTUM, rounding=ROUND_HALF_EVEN)
    if rounded == _ZERO:
        rounded = _ZERO.quantize(_DISPLAY_QUANTUM)
    return f"{rounded:,.2f}"


def _format_periods_per_year(value: Decimal | None) -> str:
    """Format an explicit annualisation factor without implying percentage units."""

    if value is None:
        return _MISSING
    _require_finite_decimal(value, "periods_per_year")
    return format(value, "f")


def _format_timestamp(value: datetime | None) -> str:
    """Format an optional timezone-aware timestamp as ISO 8601."""

    if value is None:
        return _MISSING
    if not isinstance(value, datetime):
        raise TypeError("timestamp must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.isoformat()


def _format_duration(value: timedelta | None) -> str:
    """Format an optional duration without dropping sub-day precision."""

    if value is None:
        return _MISSING
    if not isinstance(value, timedelta):
        raise TypeError("duration must be a timedelta")
    if value < timedelta(0):
        raise ValueError("duration must be non-negative")
    hours, remaining_seconds = divmod(value.seconds, 3600)
    minutes, seconds = divmod(remaining_seconds, 60)
    fractional = f".{value.microseconds:06d}" if value.microseconds else ""
    day_label = "day" if value.days == 1 else "days"
    return (
        f"{value.days} {day_label}, {hours:02d}:{minutes:02d}:{seconds:02d}"
        f"{fractional}"
    )


def _section(title: str, lines: tuple[str, ...]) -> str:
    """Build one plain-text report section."""

    return "\n".join((title, "-" * len(title), *lines))


def _line(label: str, value: str) -> str:
    """Align a label and its already formatted display value."""

    return f"{label + ':':<{_LABEL_WIDTH}}{value}"


def _yes_no(value: bool) -> str:
    """Format a boolean for human readers."""

    return "Yes" if value else "No"


def _validate_currency_symbol(currency_symbol: str | None) -> str:
    """Return a safe display symbol, using an empty string only for ``None``."""

    if currency_symbol is None:
        return ""
    if not isinstance(currency_symbol, str):
        raise TypeError("currency_symbol must be a string or None")
    if not currency_symbol or any(
        character.isspace() or not character.isprintable()
        for character in currency_symbol
    ):
        raise ValueError(
            "currency_symbol must be non-empty printable text without whitespace"
        )
    return currency_symbol


def _require_finite_decimal(value: Decimal, name: str) -> None:
    """Require exact finite Decimal input at a formatting boundary."""

    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")
