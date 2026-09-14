"""Safe command-line orchestration for local simulated backtests."""

import argparse
import logging
import re
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Never, cast

from trading_research.backtesting.pipeline import (
    BacktestRunRequest,
    build_execution_report,
    run_backtest_pipeline,
)
from trading_research.config import BacktestConfig, EndOfTestPolicy, PositionSizingMode
from trading_research.data.approved_etf import (
    APPROVED_ETF_SYMBOLS,
    ApprovedEtfStudyConfig,
    ApprovedEtfStudyResult,
    DatasetRecordStatus,
    DatasetRunMode,
    run_approved_etf_study,
)
from trading_research.errors import (
    ApprovedDatasetError,
    BacktestError,
    EvaluationManifestError,
    ExperimentManifestError,
    MarketDataError,
    ReportWriteError,
    StrategyError,
)
from trading_research.evaluation import (
    DevelopmentEvaluationBatchResult,
    DevelopmentEvaluationStatus,
    SealedHoldoutBatchResult,
    SplitEvaluationBatchResult,
    SplitEvaluationStatus,
    load_split_evaluation_manifest,
    run_development_evaluation_batch,
    run_sealed_holdout_batch,
    run_split_evaluation_batch,
)
from trading_research.experiments import (
    ExperimentBatchResult,
    ExperimentStatus,
    load_experiment_manifest,
    run_experiment_batch,
)
from trading_research.models import normalize_symbol
from trading_research.performance import RiskMetricSettings
from trading_research.reporting import (
    StrategyRunConfiguration,
    StructuredBacktestReport,
    current_git_commit,
    format_backtest_report,
    format_decimal_currency,
    installed_application_version,
    write_benchmark_equity_csv,
    write_closed_trades_csv,
    write_decisions_csv,
    write_equity_curve_csv,
    write_json_report,
    write_summary_csv,
)
from trading_research.strategies import (
    DEFAULT_STRATEGY_NAME,
    DonchianBreakoutParameters,
    SmaCrossoverParameters,
    StrategyName,
)

EXIT_SUCCESS = 0
EXIT_USAGE_ERROR = 2
EXIT_DATA_ERROR = 3
EXIT_BACKTEST_ERROR = 4
EXIT_REPORT_ERROR = 5
EXIT_EXPERIMENT_FAILURE = 6
EXIT_EVALUATION_FAILURE = 7

_MAXIMUM_COMMISSION_BPS = Decimal("10000")
_MAXIMUM_SLIPPAGE_BPS = Decimal("10000")
_ZERO = Decimal("0")
_CLI_SECRET_ARGUMENT_PATTERN = re.compile(
    r"(?i)(--(?:api-key|token|fmp-api-key)(?:=|\s+))\S+"
)


@dataclass(frozen=True, slots=True)
class _BacktestOptions:
    """Typed arguments for one local historical simulation."""

    data: Path
    symbol: str
    strategy: str
    fast_window: int | None
    slow_window: int | None
    entry_window: int | None
    exit_window: int | None
    starting_cash: Decimal
    position_sizing_mode: str
    quantity: int | None
    cash_allocation_ratio: Decimal | None
    commission_bps: Decimal
    slippage_bps: Decimal
    maximum_position_value: Decimal | None
    minimum_cash_reserve: Decimal | None
    end_of_test: str
    periods_per_year: Decimal | None
    risk_free_rate_per_period: Decimal | None
    target_return_per_period: Decimal | None
    start: datetime | None
    end: datetime | None
    include_closed_trades: bool
    currency_symbol: str | None
    json_report: Path | None
    summary_csv: Path | None
    equity_csv: Path | None
    closed_trades_csv: Path | None
    decisions_csv: Path | None
    benchmark: str
    benchmark_equity_csv: Path | None
    overwrite_reports: bool


@dataclass(frozen=True, slots=True)
class _CompletedBacktest:
    """Text output plus optional structured data for requested exports."""

    text_report: str
    structured_report: StructuredBacktestReport | None


@dataclass(frozen=True, slots=True)
class _ExperimentOptions:
    """Typed arguments for one predefined sequential experiment batch."""

    manifest: Path
    output_directory: Path
    overwrite_output_directory: bool


@dataclass(frozen=True, slots=True)
class _EvaluationOptions:
    """Typed arguments for one sequential split-evaluation batch."""

    manifest: Path
    output_directory: Path
    overwrite_output_directory: bool
    development_only: bool


@dataclass(frozen=True, slots=True)
class _HoldoutOptions:
    """Typed inputs for the one-shot sealed ETF holdout."""

    manifest: Path
    selection_record: Path
    development_summary: Path
    output_directory: Path


@dataclass(frozen=True, slots=True)
class _DatasetOptions:
    """Typed arguments for the approved historical ETF dataset workflow."""

    output_root: Path
    quality_report_directory: Path
    experiment_manifest: Path
    symbols: tuple[str, ...]
    mode: DatasetRunMode
    overwrite: bool


class _CliUsageError(ValueError):
    """A semantic command-line validation failure."""


class _SafeArgumentParser(argparse.ArgumentParser):
    """Redact credential-shaped unknown arguments from parser diagnostics."""

    def error(self, message: str) -> Never:
        """Exit through argparse after removing any supplied API-key value."""

        super().error(_CLI_SECRET_ARGUMENT_PATTERN.sub(r"\1<redacted>", message))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a deterministic process exit code."""

    parser = _build_parser()
    try:
        namespace = parser.parse_args(None if argv is None else list(argv))
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE_ERROR

    if namespace.command == "dataset":
        return _run_dataset_command(_dataset_options_from_namespace(namespace))
    if namespace.command == "experiment":
        return _run_experiment_command(_experiment_options_from_namespace(namespace))
    if namespace.command == "evaluate":
        return _run_evaluation_command(_evaluation_options_from_namespace(namespace))
    if namespace.command == "holdout":
        return _run_holdout_command(_holdout_options_from_namespace(namespace))
    if namespace.command != "backtest":
        parser.print_usage(sys.stderr)
        return EXIT_USAGE_ERROR

    options = _options_from_namespace(namespace)
    try:
        completed = _run_backtest(options)
        _write_requested_reports(options, completed.structured_report)
    except _CliUsageError as exc:
        _print_error(str(exc))
        return EXIT_USAGE_ERROR
    except MarketDataError as exc:
        _print_error(str(exc))
        return EXIT_DATA_ERROR
    except (BacktestError, StrategyError) as exc:
        _print_error(str(exc))
        return EXIT_BACKTEST_ERROR
    except ReportWriteError as exc:
        _print_error(str(exc))
        return EXIT_REPORT_ERROR

    print(completed.text_report, end="")
    _print_written_reports(options)
    return EXIT_SUCCESS


def _build_parser() -> argparse.ArgumentParser:
    """Create the standard-library command parser."""

    parser = _SafeArgumentParser(
        prog="trading-research",
        allow_abbrev=False,
        description=(
            "Run educational, simulated research using validated local market data."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {installed_application_version()}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    backtest = subparsers.add_parser(
        "backtest",
        allow_abbrev=False,
        help="run one configured simulated backtest",
        description=(
            "Read a local CSV, run the existing simulated backtester, and print "
            "its human-readable report."
        ),
    )
    backtest.add_argument(
        "--data",
        type=Path,
        required=True,
        help="path to a local historical OHLCV CSV file",
    )
    backtest.add_argument(
        "--symbol",
        required=True,
        help="single market symbol expected in the CSV",
    )
    backtest.add_argument(
        "--strategy",
        choices=tuple(strategy.value for strategy in StrategyName),
        default=DEFAULT_STRATEGY_NAME.value,
        help="built-in strategy to simulate (default: sma-crossover)",
    )
    backtest.add_argument(
        "--fast-window",
        type=_parse_positive_integer,
        help="positive short moving-average window for sma-crossover",
    )
    backtest.add_argument(
        "--slow-window",
        type=_parse_positive_integer,
        help="positive long moving-average window for sma-crossover",
    )
    backtest.add_argument(
        "--entry-window",
        type=_parse_positive_integer,
        help="positive prior-close entry channel for donchian-breakout",
    )
    backtest.add_argument(
        "--exit-window",
        type=_parse_positive_integer,
        help="positive prior-close exit channel for donchian-breakout",
    )
    backtest.add_argument(
        "--starting-cash",
        type=_parse_positive_decimal,
        required=True,
        help="positive starting cash parsed as an exact decimal",
    )
    backtest.add_argument(
        "--position-size-mode",
        choices=("fixed", "cash-allocation"),
        default="fixed",
        help="entry sizing policy (default: fixed)",
    )
    backtest.add_argument(
        "--quantity",
        type=_parse_positive_integer,
        help="positive whole-share quantity required in fixed mode",
    )
    backtest.add_argument(
        "--cash-allocation-ratio",
        type=_parse_allocation_ratio,
        help="exact cash fraction in (0, 1] required in cash-allocation mode",
    )
    backtest.add_argument(
        "--commission-bps",
        dest="commission_bps",
        type=_parse_commission_bps,
        required=True,
        help="non-negative proportional commission rate in basis points",
    )
    backtest.add_argument(
        "--slippage-bps",
        type=_parse_slippage_bps,
        required=True,
        help="non-negative adverse simulated slippage in basis points",
    )
    backtest.add_argument(
        "--maximum-position-value",
        type=_parse_positive_decimal,
        help="optional positive simulated post-trade position-value cap",
    )
    backtest.add_argument(
        "--minimum-cash-reserve",
        type=_parse_nonnegative_decimal,
        help="optional non-negative cash that must remain after a simulated buy",
    )
    backtest.add_argument(
        "--end-of-test",
        choices=("hold", "liquidate"),
        default="hold",
        help=(
            "final-position policy: hold at the final close (default), or "
            "synthetically liquidate using the final close"
        ),
    )
    backtest.add_argument(
        "--periods-per-year",
        type=_parse_positive_decimal,
        help=(
            "optional positive annualisation factor selected for the data frequency"
        ),
    )
    backtest.add_argument(
        "--risk-free-rate-per-period",
        type=_parse_finite_decimal,
        help="optional per-period decimal return ratio, not a percentage or annual rate",
    )
    backtest.add_argument(
        "--target-return-per-period",
        type=_parse_finite_decimal,
        help="optional per-period decimal target ratio for downside statistics",
    )
    backtest.add_argument(
        "--start",
        type=_parse_timestamp,
        help="inclusive timezone-aware ISO 8601 start timestamp",
    )
    backtest.add_argument(
        "--end",
        type=_parse_timestamp,
        help="inclusive timezone-aware ISO 8601 end timestamp",
    )
    backtest.add_argument(
        "--include-closed-trades",
        action="store_true",
        help="append reconstructed closed-trade detail rows",
    )
    backtest.add_argument(
        "--currency-symbol",
        type=_parse_currency_symbol,
        help="optional display-only currency symbol; no currency is inferred",
    )
    backtest.add_argument(
        "--json-report",
        type=Path,
        help="optional path for a complete structured JSON report",
    )
    backtest.add_argument(
        "--summary-csv",
        type=Path,
        help="optional path for a one-row raw summary CSV",
    )
    backtest.add_argument(
        "--equity-csv",
        type=Path,
        help="optional path for a chronological equity-curve CSV",
    )
    backtest.add_argument(
        "--closed-trades-csv",
        type=Path,
        help="optional path for reconstructed closed trades (header-only if empty)",
    )
    backtest.add_argument(
        "--decisions-csv",
        type=Path,
        help="optional path for audited simulated entry decisions",
    )
    backtest.add_argument(
        "--benchmark",
        choices=("none", "buy-and-hold"),
        default="none",
        help="optional capital-matched benchmark (default: none)",
    )
    backtest.add_argument(
        "--benchmark-equity-csv",
        type=Path,
        help="optional aligned strategy-versus-benchmark equity CSV",
    )
    backtest.add_argument(
        "--overwrite-reports",
        action="store_true",
        help="allow replacement of only the explicitly requested report files",
    )
    experiment = subparsers.add_parser(
        "experiment",
        allow_abbrev=False,
        help="run a predefined sequential JSON or TOML experiment batch",
        description=(
            "Run explicitly configured backtests sequentially and write one JSON "
            "report per success plus ordered aggregate outputs."
        ),
    )
    experiment.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="path to a predefined .json or .toml experiment manifest",
    )
    experiment.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="dedicated directory for individual and aggregate batch reports",
    )
    experiment.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="explicitly allow replacement of files in an existing output directory",
    )
    evaluate = subparsers.add_parser(
        "evaluate",
        allow_abbrev=False,
        help="run predefined development and holdout evaluations",
        description=(
            "Run frozen configurations over explicit development and holdout ranges "
            "without optimisation, selection, tuning, or network access."
        ),
    )
    evaluate.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="path to a .json or .toml manifest containing split_evaluations",
    )
    evaluate.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="dedicated directory for ordered split-evaluation outputs",
    )
    evaluate.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="explicitly allow replacement of known files in an existing directory",
    )
    evaluate.add_argument(
        "--development-only",
        action="store_true",
        help=(
            "run and export only the declared development periods; no future-period "
            "strategy run or result is created"
        ),
    )
    holdout = subparsers.add_parser(
        "holdout",
        allow_abbrev=False,
        help="run the single pre-registered approved-ETF holdout",
        description=(
            "Run exactly the frozen universal SMA 50/200 configuration over the "
            "eight approved ETF holdout periods without alternatives or tuning."
        ),
    )
    holdout.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="frozen approved-ETF split manifest",
    )
    holdout.add_argument(
        "--selection-record",
        type=Path,
        required=True,
        help="sealed development selection TOML",
    )
    holdout.add_argument(
        "--development-summary",
        type=Path,
        required=True,
        help="frozen development-summary.csv used only for comparison",
    )
    holdout.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new protected directory for the one-shot holdout outputs",
    )
    dataset = subparsers.add_parser(
        "dataset",
        allow_abbrev=False,
        help="acquire and prepare approved local historical research datasets",
        description=(
            "Download and validate approved historical datasets without running "
            "strategies, backtests, or evaluations."
        ),
    )
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    approved_etfs = dataset_commands.add_parser(
        "acquire-approved-etf-study",
        allow_abbrev=False,
        help="prepare the fixed eight-ETF personal-research universe",
        description=(
            "Acquire pinned yfinance daily data and create Yahoo prepared data, "
            "provenance, quality reports, and a frozen manifest for personal research."
        ),
    )
    approved_etfs.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="local data root containing incoming and prepared directories",
    )
    approved_etfs.add_argument(
        "--quality-report-dir",
        type=Path,
        required=True,
        help="directory for universe data-quality summaries",
    )
    approved_etfs.add_argument(
        "--experiment-manifest",
        type=Path,
        required=True,
        help="path for the generated 32-evaluation TOML manifest",
    )
    approved_etfs.add_argument(
        "--symbols",
        nargs="+",
        type=_parse_approved_etf_symbol,
        default=APPROVED_ETF_SYMBOLS,
        help="optional approved-symbol subset for deterministic recovery",
    )
    mode = approved_etfs.add_mutually_exclusive_group()
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="use existing raw files and perform no network requests",
    )
    mode.add_argument(
        "--download-only",
        action="store_true",
        help="download and validate raw files without transforming them",
    )
    approved_etfs.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly allow replacement of known dataset workflow outputs",
    )
    return parser


def _options_from_namespace(namespace: argparse.Namespace) -> _BacktestOptions:
    """Convert argparse's dynamic namespace to typed immutable options."""

    return _BacktestOptions(
        data=cast(Path, namespace.data),
        symbol=cast(str, namespace.symbol),
        strategy=cast(str, namespace.strategy),
        fast_window=cast(int | None, namespace.fast_window),
        slow_window=cast(int | None, namespace.slow_window),
        entry_window=cast(int | None, namespace.entry_window),
        exit_window=cast(int | None, namespace.exit_window),
        starting_cash=cast(Decimal, namespace.starting_cash),
        position_sizing_mode=cast(str, namespace.position_size_mode),
        quantity=cast(int | None, namespace.quantity),
        cash_allocation_ratio=cast(
            Decimal | None,
            namespace.cash_allocation_ratio,
        ),
        commission_bps=cast(Decimal, namespace.commission_bps),
        slippage_bps=cast(Decimal, namespace.slippage_bps),
        maximum_position_value=cast(
            Decimal | None,
            namespace.maximum_position_value,
        ),
        minimum_cash_reserve=cast(
            Decimal | None,
            namespace.minimum_cash_reserve,
        ),
        end_of_test=cast(str, namespace.end_of_test),
        periods_per_year=cast(Decimal | None, namespace.periods_per_year),
        risk_free_rate_per_period=cast(
            Decimal | None,
            namespace.risk_free_rate_per_period,
        ),
        target_return_per_period=cast(
            Decimal | None,
            namespace.target_return_per_period,
        ),
        start=cast(datetime | None, namespace.start),
        end=cast(datetime | None, namespace.end),
        include_closed_trades=cast(bool, namespace.include_closed_trades),
        currency_symbol=cast(str | None, namespace.currency_symbol),
        json_report=cast(Path | None, namespace.json_report),
        summary_csv=cast(Path | None, namespace.summary_csv),
        equity_csv=cast(Path | None, namespace.equity_csv),
        closed_trades_csv=cast(Path | None, namespace.closed_trades_csv),
        decisions_csv=cast(Path | None, namespace.decisions_csv),
        benchmark=cast(str, namespace.benchmark),
        benchmark_equity_csv=cast(Path | None, namespace.benchmark_equity_csv),
        overwrite_reports=cast(bool, namespace.overwrite_reports),
    )


def _experiment_options_from_namespace(
    namespace: argparse.Namespace,
) -> _ExperimentOptions:
    """Convert argparse's experiment namespace to immutable typed options."""

    return _ExperimentOptions(
        manifest=cast(Path, namespace.manifest),
        output_directory=cast(Path, namespace.output_dir),
        overwrite_output_directory=cast(bool, namespace.overwrite_output_dir),
    )


def _evaluation_options_from_namespace(
    namespace: argparse.Namespace,
) -> _EvaluationOptions:
    """Convert argparse's evaluation namespace to immutable typed options."""

    return _EvaluationOptions(
        manifest=cast(Path, namespace.manifest),
        output_directory=cast(Path, namespace.output_dir),
        overwrite_output_directory=cast(bool, namespace.overwrite_output_dir),
        development_only=cast(bool, namespace.development_only),
    )


def _holdout_options_from_namespace(namespace: argparse.Namespace) -> _HoldoutOptions:
    """Create typed one-shot holdout options from parsed arguments."""

    return _HoldoutOptions(
        manifest=cast(Path, namespace.manifest),
        selection_record=cast(Path, namespace.selection_record),
        development_summary=cast(Path, namespace.development_summary),
        output_directory=cast(Path, namespace.output_dir),
    )


def _dataset_options_from_namespace(namespace: argparse.Namespace) -> _DatasetOptions:
    """Convert approved-dataset arguments to a typed immutable configuration."""

    mode = DatasetRunMode.FULL
    if cast(bool, namespace.prepare_only):
        mode = DatasetRunMode.PREPARE_ONLY
    elif cast(bool, namespace.download_only):
        mode = DatasetRunMode.DOWNLOAD_ONLY
    return _DatasetOptions(
        output_root=cast(Path, namespace.output_root),
        quality_report_directory=cast(Path, namespace.quality_report_dir),
        experiment_manifest=cast(Path, namespace.experiment_manifest),
        symbols=tuple(cast(Sequence[str], namespace.symbols)),
        mode=mode,
        overwrite=cast(bool, namespace.overwrite),
    )


def _run_dataset_command(options: _DatasetOptions) -> int:
    """Acquire or prepare approved historical files without running strategies."""

    try:
        result = run_approved_etf_study(
            ApprovedEtfStudyConfig(
                output_root=options.output_root,
                quality_report_directory=options.quality_report_directory,
                experiment_manifest=options.experiment_manifest,
                symbols=options.symbols,
                mode=options.mode,
                overwrite=options.overwrite,
            )
        )
    except (ApprovedDatasetError, ValueError) as exc:
        _print_error(str(exc))
        return EXIT_DATA_ERROR
    _print_dataset_result(result)
    return EXIT_SUCCESS if result.succeeded else EXIT_DATA_ERROR


def _print_dataset_result(result: ApprovedEtfStudyResult) -> None:
    """Print every requested symbol outcome without secrets or absolute paths."""

    print("APPROVED ETF DATASET WORKFLOW")
    print("=============================")
    print(f"Mode: {result.mode.value}")
    if result.provider_diagnostic is not None:
        diagnostic = result.provider_diagnostic
        print("Yahoo Finance via yfinance SPY full-history validation:")
        print(f"- Valid data: {diagnostic.valid_data}")
        print(
            "- Available dates: "
            f"{diagnostic.earliest_date} through {diagnostic.latest_date}"
        )
        print(f"- Rows: {diagnostic.row_count}")
        print(f"- Required fields present: {diagnostic.required_fields_present}")
        print(f"- Split events: {diagnostic.split_event_count}")
        print(f"- Dividend events: {diagnostic.dividend_event_count}")
        print(f"- Repeatability SHA-256: {diagnostic.repeatability_sha256}")
        print(f"- Quality gate passed: {diagnostic.quality_gate_passed}")
    for record in result.records:
        if record.status is DatasetRecordStatus.SUCCESS:
            detail = (
                f"raw_sha256={record.raw_sha256}"
                if record.prepared_sha256 is None
                else (
                    f"rows={record.row_count} "
                    f"raw_sha256={record.raw_sha256} "
                    f"prepared_sha256={record.prepared_sha256}"
                )
            )
            print(f"- {record.symbol}: SUCCESS ({detail})")
        else:
            print(f"- {record.symbol}: FAILURE ({record.failure_message})")
    if result.quality_summary_csv is not None:
        print(f"Quality CSV: {_safe_display_path(result.quality_summary_csv)}")
    if result.quality_summary_text is not None:
        print(f"Quality text: {_safe_display_path(result.quality_summary_text)}")
    if result.experiment_manifest is not None:
        print(f"Manifest: {_safe_display_path(result.experiment_manifest)}")


def _run_evaluation_command(options: _EvaluationOptions) -> int:
    """Load, execute, export, and summarize declared split evaluations."""

    try:
        manifest = load_split_evaluation_manifest(options.manifest)
        if options.development_only:
            development_result = run_development_evaluation_batch(
                manifest,
                options.output_directory,
                overwrite=options.overwrite_output_directory,
                repository=Path.cwd(),
            )
            _print_development_evaluation_result(
                development_result,
                options.output_directory,
            )
            return (
                EXIT_SUCCESS
                if development_result.succeeded
                else EXIT_EVALUATION_FAILURE
            )
        result = run_split_evaluation_batch(
            manifest,
            options.output_directory,
            overwrite=options.overwrite_output_directory,
            repository=Path.cwd(),
        )
    except EvaluationManifestError as exc:
        _print_error(str(exc))
        return EXIT_USAGE_ERROR
    except ReportWriteError as exc:
        _print_error(str(exc))
        return EXIT_REPORT_ERROR
    except BacktestError as exc:
        _print_error(str(exc))
        return EXIT_BACKTEST_ERROR
    _print_evaluation_result(result, options.output_directory)
    return EXIT_SUCCESS if result.succeeded else EXIT_EVALUATION_FAILURE


def _print_development_evaluation_result(
    result: DevelopmentEvaluationBatchResult,
    output_directory: Path,
) -> None:
    """Print development outcomes without future-period results."""

    print("DEVELOPMENT-ONLY EVALUATION BATCH")
    print("=================================")
    print(f"Evaluations: {len(result.records)}")
    print(f"Completed:   {result.success_count}")
    print(f"Failed:      {result.failure_count}")
    for record in result.records:
        if record.status is DevelopmentEvaluationStatus.SUCCESS:
            print(f"- {record.evaluation_id}: COMPLETED")
        else:
            kind = (
                "failure"
                if record.failure_kind is None
                else record.failure_kind.value
            )
            print(
                f"- {record.evaluation_id}: FAILURE "
                f"({kind}) {record.failure_message}"
            )
    print(f"Selection: {result.selection_status}")
    print(f"Output: {_safe_display_path(output_directory)}")


def _run_holdout_command(options: _HoldoutOptions) -> int:
    """Run and report only the completed one-shot sealed holdout batch."""

    try:
        manifest = load_split_evaluation_manifest(options.manifest)
        result = run_sealed_holdout_batch(
            manifest,
            selection_record=options.selection_record,
            development_summary=options.development_summary,
            output_directory=options.output_directory,
            repository=Path.cwd(),
        )
    except EvaluationManifestError as exc:
        _print_error(str(exc))
        return EXIT_USAGE_ERROR
    except ReportWriteError as exc:
        _print_error(str(exc))
        return EXIT_REPORT_ERROR
    except BacktestError as exc:
        _print_error(str(exc))
        return EXIT_BACKTEST_ERROR
    _print_holdout_result(result, options.output_directory)
    return (
        EXIT_SUCCESS
        if result.aggregate.integrity_gate_passed
        else EXIT_EVALUATION_FAILURE
    )


def _print_holdout_result(
    result: SealedHoldoutBatchResult,
    output_directory: Path,
) -> None:
    """Print no result until every holdout and aggregate export is complete."""

    reconciled = sum(
        item.report.backtest_result.reconciliation.is_reconciled
        for item in result.results
    )
    print("SEALED HOLDOUT EVALUATION COMPLETE")
    print("==================================")
    print(f"Evaluations:     {len(result.results)}")
    print(f"Reconciliations: {reconciled}")
    print(
        "Integrity gate: "
        f"{'PASS' if result.aggregate.integrity_gate_passed else 'FAIL'}"
    )
    print(f"Classification: {result.aggregate.classification.value}")
    print(f"Output: {_safe_display_path(output_directory)}")


def _print_evaluation_result(
    result: SplitEvaluationBatchResult,
    output_directory: Path,
) -> None:
    """Print every declared outcome without an automatic research verdict."""

    print("OUT-OF-SAMPLE EVALUATION BATCH")
    print("==============================")
    print(f"Evaluations: {len(result.records)}")
    print(f"Completed:   {result.success_count}")
    print(f"Failed:      {result.failure_count}")
    for record in result.records:
        if record.status is SplitEvaluationStatus.SUCCESS:
            print(f"- {record.evaluation_id}: COMPLETED")
        else:
            kind = "failure" if record.failure_kind is None else record.failure_kind.value
            print(f"- {record.evaluation_id}: FAILURE ({kind}) {record.failure_message}")
    print(f"Output: {_safe_display_path(output_directory)}")


def _run_experiment_command(options: _ExperimentOptions) -> int:
    """Load, execute, export, and summarize one predefined experiment batch."""

    try:
        manifest = load_experiment_manifest(options.manifest)
        result = run_experiment_batch(
            manifest,
            options.output_directory,
            overwrite=options.overwrite_output_directory,
            repository=Path.cwd(),
        )
    except ExperimentManifestError as exc:
        _print_error(str(exc))
        return EXIT_USAGE_ERROR
    except ReportWriteError as exc:
        _print_error(str(exc))
        return EXIT_REPORT_ERROR
    except BacktestError as exc:
        _print_error(str(exc))
        return EXIT_BACKTEST_ERROR
    _print_experiment_result(result, options.output_directory)
    return EXIT_SUCCESS if result.succeeded else EXIT_EXPERIMENT_FAILURE


def _print_experiment_result(
    result: ExperimentBatchResult,
    output_directory: Path,
) -> None:
    """Print every ordered status without exposing absolute machine paths."""

    print("EXPERIMENT BATCH RESULT")
    print("=======================")
    print(f"Experiments: {len(result.records)}")
    print(f"Succeeded:   {result.success_count}")
    print(f"Failed:      {result.failure_count}")
    for record in result.records:
        if record.status is ExperimentStatus.SUCCESS:
            print(f"- {record.experiment_id}: SUCCESS")
        else:
            kind = "failure" if record.failure_kind is None else record.failure_kind.value
            print(f"- {record.experiment_id}: FAILURE ({kind}) {record.failure_message}")
    print(f"Output: {_safe_display_path(output_directory)}")


def _run_backtest(options: _BacktestOptions) -> _CompletedBacktest:
    """Run existing components and prepare structured data only when requested."""

    _validate_options(options)
    try:
        symbol = normalize_symbol(options.symbol)
    except (TypeError, ValueError) as exc:
        raise _CliUsageError(f"invalid symbol: {exc}") from exc
    strategy_configuration = _strategy_configuration(options)

    sizing_mode = (
        PositionSizingMode.FIXED
        if options.position_sizing_mode == "fixed"
        else PositionSizingMode.CASH_ALLOCATION
    )
    configured_quantity = options.quantity if options.quantity is not None else 10
    config = BacktestConfig(
        initial_cash=options.starting_cash,
        trade_quantity=configured_quantity,
        commission_bps=options.commission_bps,
        slippage_bps=options.slippage_bps,
        position_sizing_mode=sizing_mode,
        cash_allocation_ratio=options.cash_allocation_ratio,
        maximum_position_value=options.maximum_position_value,
        minimum_cash_reserve=options.minimum_cash_reserve,
        end_of_test_policy=EndOfTestPolicy(options.end_of_test),
    )
    risk_metric_settings = _build_risk_metric_settings(options)
    request = BacktestRunRequest(
        data_path=options.data,
        symbol=symbol,
        strategy_configuration=strategy_configuration,
        backtest_configuration=config,
        start=options.start,
        end=options.end,
        risk_metric_settings=risk_metric_settings,
        include_buy_and_hold_benchmark=options.benchmark == "buy-and-hold",
    )
    with _suppress_project_logging():
        execution = run_backtest_pipeline(request)
        if execution.benchmark is None and risk_metric_settings is None:
            text_report = format_backtest_report(
                execution.result,
                currency_symbol=options.currency_symbol,
                include_closed_trades=options.include_closed_trades,
                strategy_configuration=strategy_configuration,
            )
        elif execution.benchmark is None:
            text_report = format_backtest_report(
                execution.result,
                currency_symbol=options.currency_symbol,
                include_closed_trades=options.include_closed_trades,
                risk_metric_settings=risk_metric_settings,
                strategy_configuration=strategy_configuration,
            )
        elif risk_metric_settings is None:
            text_report = format_backtest_report(
                execution.result,
                currency_symbol=options.currency_symbol,
                include_closed_trades=options.include_closed_trades,
                strategy_configuration=strategy_configuration,
                benchmark=execution.benchmark,
                benchmark_comparison=execution.benchmark_comparison,
            )
        else:
            text_report = format_backtest_report(
                execution.result,
                currency_symbol=options.currency_symbol,
                include_closed_trades=options.include_closed_trades,
                risk_metric_settings=risk_metric_settings,
                strategy_configuration=strategy_configuration,
                benchmark=execution.benchmark,
                benchmark_comparison=execution.benchmark_comparison,
            )
        if not _has_requested_reports(options):
            return _CompletedBacktest(text_report=text_report, structured_report=None)
        try:
            structured_report = build_execution_report(
                execution,
                git_commit=current_git_commit(Path.cwd()),
                base_directory=Path.cwd(),
            )
        except OSError as exc:
            raise MarketDataError(str(exc)) from exc
        return _CompletedBacktest(
            text_report=text_report,
            structured_report=structured_report,
        )


def _build_risk_metric_settings(
    options: _BacktestOptions,
) -> RiskMetricSettings | None:
    """Build opt-in return-metric settings without inferring annualisation."""

    if all(
        value is None
        for value in (
            options.periods_per_year,
            options.risk_free_rate_per_period,
            options.target_return_per_period,
        )
    ):
        return None
    return RiskMetricSettings(
        periods_per_year=options.periods_per_year,
        risk_free_rate_per_period=(
            _ZERO
            if options.risk_free_rate_per_period is None
            else options.risk_free_rate_per_period
        ),
        target_return_per_period=(
            _ZERO
            if options.target_return_per_period is None
            else options.target_return_per_period
        ),
    )


def _strategy_configuration(
    options: _BacktestOptions,
) -> StrategyRunConfiguration:
    """Build one typed strategy descriptor without ignoring irrelevant flags."""

    try:
        strategy_name = StrategyName(options.strategy)
    except ValueError as exc:
        raise _CliUsageError(f"unsupported strategy: {options.strategy}") from exc

    if strategy_name is StrategyName.SMA_CROSSOVER:
        if options.entry_window is not None or options.exit_window is not None:
            raise _CliUsageError(
                "--entry-window and --exit-window apply only to donchian-breakout"
            )
        if options.fast_window is None or options.slow_window is None:
            raise _CliUsageError(
                "--fast-window and --slow-window are required for sma-crossover"
            )
        try:
            sma_parameters = SmaCrossoverParameters(
                fast_window=options.fast_window,
                slow_window=options.slow_window,
            )
        except (TypeError, ValueError) as exc:
            raise _CliUsageError(str(exc)) from exc
        return StrategyRunConfiguration(name=strategy_name, parameters=sma_parameters)

    if options.fast_window is not None or options.slow_window is not None:
        raise _CliUsageError(
            "--fast-window and --slow-window apply only to sma-crossover"
        )
    if options.entry_window is None or options.exit_window is None:
        raise _CliUsageError(
            "--entry-window and --exit-window are required for donchian-breakout"
        )
    donchian_parameters = DonchianBreakoutParameters(
        entry_window=options.entry_window,
        exit_window=options.exit_window,
    )
    return StrategyRunConfiguration(
        name=strategy_name,
        parameters=donchian_parameters,
    )


def _validate_options(options: _BacktestOptions) -> None:
    """Validate cross-field constraints and inspect the supplied local path."""

    if not options.data.exists():
        raise _CliUsageError(f"data file does not exist: {options.data}")
    if not options.data.is_file():
        raise _CliUsageError(f"data path is not a file: {options.data}")
    _strategy_configuration(options)
    if options.position_sizing_mode == "fixed":
        if options.quantity is None:
            raise _CliUsageError("--quantity is required in fixed position-size mode")
        if options.cash_allocation_ratio is not None:
            raise _CliUsageError(
                "--cash-allocation-ratio is only valid in cash-allocation mode"
            )
    else:
        if options.quantity is not None:
            raise _CliUsageError("--quantity is only valid in fixed position-size mode")
        if options.cash_allocation_ratio is None:
            raise _CliUsageError(
                "--cash-allocation-ratio is required in cash-allocation mode"
            )
    if options.start is not None and options.end is not None and options.start > options.end:
        raise _CliUsageError("start timestamp must not be after end timestamp")
    if options.benchmark_equity_csv is not None and options.benchmark == "none":
        raise _CliUsageError(
            "--benchmark-equity-csv requires --benchmark buy-and-hold"
        )
    report_paths = tuple(path for _, path in _requested_report_paths(options))
    normalized_paths = tuple(path.resolve(strict=False) for path in report_paths)
    if len(normalized_paths) != len(set(normalized_paths)):
        raise _CliUsageError("each report output must use a different path")


def _has_requested_reports(options: _BacktestOptions) -> bool:
    """Return whether at least one structured output path was supplied."""

    return bool(_requested_report_paths(options))


def _requested_report_paths(
    options: _BacktestOptions,
) -> tuple[tuple[str, Path], ...]:
    """Return explicitly requested outputs in stable display and write order."""

    candidates = (
        ("JSON", options.json_report),
        ("Summary CSV", options.summary_csv),
        ("Equity CSV", options.equity_csv),
        ("Closed trades CSV", options.closed_trades_csv),
        ("Decisions CSV", options.decisions_csv),
        ("Benchmark equity CSV", options.benchmark_equity_csv),
    )
    return tuple((label, path) for label, path in candidates if path is not None)


def _write_requested_reports(
    options: _BacktestOptions,
    report: StructuredBacktestReport | None,
) -> None:
    """Write all explicitly requested files before printing success output."""

    requested = _requested_report_paths(options)
    if not requested:
        return
    if report is None:
        raise ReportWriteError("structured report was not prepared")
    _preflight_report_paths(options)
    common = {
        "overwrite": options.overwrite_reports,
        "create_parents": True,
    }
    if options.json_report is not None:
        write_json_report(report, options.json_report, **common)
    if options.summary_csv is not None:
        write_summary_csv(report, options.summary_csv, **common)
    if options.equity_csv is not None:
        write_equity_curve_csv(report, options.equity_csv, **common)
    if options.closed_trades_csv is not None:
        write_closed_trades_csv(report, options.closed_trades_csv, **common)
    if options.decisions_csv is not None:
        write_decisions_csv(report, options.decisions_csv, **common)
    if options.benchmark_equity_csv is not None:
        write_benchmark_equity_csv(report, options.benchmark_equity_csv, **common)


def _preflight_report_paths(options: _BacktestOptions) -> None:
    """Reject predictable destination failures before writing any report."""

    for _, path in _requested_report_paths(options):
        if path.exists():
            if path.is_dir():
                raise ReportWriteError(f"report destination is a directory: {path}")
            if not options.overwrite_reports:
                raise ReportWriteError(f"report file already exists: {path}")
        parent = path.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if parent.exists() and not parent.is_dir():
            raise ReportWriteError(f"report parent path is not a directory: {parent}")


def _print_written_reports(options: _BacktestOptions) -> None:
    """Print concise relative destinations only after every write succeeds."""

    requested = _requested_report_paths(options)
    if not requested:
        return
    print("\nReports written:")
    for label, path in requested:
        print(f"- {label}: {_safe_display_path(path)}")


def _safe_display_path(path: Path) -> str:
    """Prefer a current-directory-relative path and otherwise show only the name."""

    try:
        return path.resolve(strict=False).relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _parse_positive_integer(value: str) -> int:
    """Parse a strictly positive base-10 whole number."""

    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a whole number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parse_approved_etf_symbol(value: str) -> str:
    """Normalize and restrict recovery subsets to the fixed approved universe."""

    normalized = value.strip().upper()
    if normalized not in APPROVED_ETF_SYMBOLS:
        choices = ", ".join(APPROVED_ETF_SYMBOLS)
        raise argparse.ArgumentTypeError(f"must be one of: {choices}")
    return normalized


def _parse_finite_decimal(value: str) -> Decimal:
    """Parse an exact finite Decimal directly from command-line text."""

    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("must be a decimal number") from exc
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _parse_positive_decimal(value: str) -> Decimal:
    """Parse a finite Decimal greater than zero."""

    parsed = _parse_finite_decimal(value)
    if parsed <= _ZERO:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parse_nonnegative_decimal(value: str) -> Decimal:
    """Parse a finite Decimal greater than or equal to zero."""

    parsed = _parse_finite_decimal(value)
    if parsed < _ZERO:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _parse_allocation_ratio(value: str) -> Decimal:
    """Parse an exact cash-allocation ratio in the interval ``(0, 1]``."""

    parsed = _parse_finite_decimal(value)
    if not (_ZERO < parsed <= Decimal("1")):
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


def _parse_commission_bps(value: str) -> Decimal:
    """Parse the existing proportional commission basis-point rate."""

    parsed = _parse_finite_decimal(value)
    if parsed < _ZERO:
        raise argparse.ArgumentTypeError("must be non-negative")
    if parsed > _MAXIMUM_COMMISSION_BPS:
        raise argparse.ArgumentTypeError("must not exceed 10000 basis points")
    return parsed


def _parse_slippage_bps(value: str) -> Decimal:
    """Parse the existing adverse-slippage basis-point rate."""

    parsed = _parse_finite_decimal(value)
    if parsed < _ZERO:
        raise argparse.ArgumentTypeError("must be non-negative")
    if parsed >= _MAXIMUM_SLIPPAGE_BPS:
        raise argparse.ArgumentTypeError("must be less than 10000 basis points")
    return parsed


def _parse_timestamp(value: str) -> datetime:
    """Parse a timezone-aware ISO 8601 timestamp."""

    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an ISO 8601 timestamp") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise argparse.ArgumentTypeError("must include a timezone offset")
    return timestamp


def _parse_currency_symbol(value: str) -> str:
    """Reuse the report formatter's currency-symbol validation policy."""

    try:
        format_decimal_currency(_ZERO, value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


@contextmanager
def _suppress_project_logging() -> Iterator[None]:
    """Prevent library log records from contaminating CLI report output."""

    project_logger = logging.getLogger("trading_research")
    previous_level = project_logger.level
    project_logger.setLevel(logging.CRITICAL + 1)
    try:
        yield
    finally:
        project_logger.setLevel(previous_level)


def _print_error(message: str) -> None:
    """Write one concise expected-error message without a traceback."""

    print(f"error: {message}", file=sys.stderr)
