"""Validated, immutable identity and configuration for one backtest run."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import uuid4

from trading_research._version import application_version
from trading_research.config import BacktestConfig, PositionSizingMode
from trading_research.models import EndOfTestPolicy, MarketBar, normalize_symbol
from trading_research.performance import RiskMetricSettings
from trading_research.strategies import (
    DonchianBreakoutParameters,
    SmaCrossoverParameters,
    StrategyName,
    StrategyParameters,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ZERO = Decimal("0")


def _require_text(value: str, name: str) -> None:
    """Require non-empty printable text without control characters."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip() or any(not character.isprintable() for character in value):
        raise ValueError(f"{name} must be non-empty printable text")


def _require_aware_timestamp(value: datetime | None, name: str) -> None:
    """Require optional timestamps to carry an explicit UTC offset."""

    if value is not None and not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime or None")
    if value is not None and (value.tzinfo is None or value.utcoffset() is None):
        raise ValueError(f"{name} must be timezone-aware")


def _require_finite_decimal(value: Decimal, name: str) -> None:
    """Require exact finite Decimal configuration values."""

    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    """Identity of the exact validated dataset used by a backtest."""

    source_type: str
    identifier: str
    sha256: str | None
    symbol: str
    requested_start: datetime | None
    requested_end: datetime | None
    actual_start: datetime
    actual_end: datetime
    bar_count: int

    def __post_init__(self) -> None:
        """Validate safe identity text, timestamps, digest, and observation count."""

        _require_text(self.source_type, "source_type")
        _require_text(self.identifier, "identifier")
        if PurePosixPath(self.identifier).is_absolute() or PureWindowsPath(
            self.identifier
        ).is_absolute():
            raise ValueError("identifier must not be an absolute path")
        if ".." in PurePosixPath(self.identifier).parts or ".." in PureWindowsPath(
            self.identifier
        ).parts:
            raise ValueError("identifier must not contain parent-directory traversal")
        if self.sha256 is not None and not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("sha256 must be a lowercase 64-character digest or None")
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        for name, timestamp in (
            ("requested_start", self.requested_start),
            ("requested_end", self.requested_end),
            ("actual_start", self.actual_start),
            ("actual_end", self.actual_end),
        ):
            _require_aware_timestamp(timestamp, name)
        if (
            self.requested_start is not None
            and self.requested_end is not None
            and self.requested_start > self.requested_end
        ):
            raise ValueError("requested_start must not be after requested_end")
        if self.actual_start > self.actual_end:
            raise ValueError("actual_start must not be after actual_end")
        if self.requested_start is not None and self.actual_start < self.requested_start:
            raise ValueError("actual_start must not precede requested_start")
        if self.requested_end is not None and self.actual_end > self.requested_end:
            raise ValueError("actual_end must not follow requested_end")
        if isinstance(self.bar_count, bool) or not isinstance(self.bar_count, int):
            raise TypeError("bar_count must be an integer")
        if self.bar_count <= 0:
            raise ValueError("bar_count must be positive")


@dataclass(frozen=True, slots=True)
class StrategyRunConfiguration:
    """Stable strategy identity paired with its typed parameter model."""

    name: StrategyName
    parameters: StrategyParameters

    def __post_init__(self) -> None:
        """Require parameters belonging to the selected built-in strategy."""

        if not isinstance(self.name, StrategyName):
            raise TypeError("strategy name must be a StrategyName")
        if self.name is StrategyName.SMA_CROSSOVER:
            if not isinstance(self.parameters, SmaCrossoverParameters):
                raise TypeError("sma-crossover requires SmaCrossoverParameters")
        elif not isinstance(self.parameters, DonchianBreakoutParameters):
            raise TypeError("donchian-breakout requires DonchianBreakoutParameters")

    @property
    def fast_window(self) -> int | None:
        """Return the SMA fast window, or None for another strategy."""

        if isinstance(self.parameters, SmaCrossoverParameters):
            return self.parameters.fast_window
        return None

    @property
    def slow_window(self) -> int | None:
        """Return the SMA slow window, or None for another strategy."""

        if isinstance(self.parameters, SmaCrossoverParameters):
            return self.parameters.slow_window
        return None

    @property
    def entry_window(self) -> int | None:
        """Return the Donchian entry window, or None for another strategy."""

        if isinstance(self.parameters, DonchianBreakoutParameters):
            return self.parameters.entry_window
        return None

    @property
    def exit_window(self) -> int | None:
        """Return the Donchian exit window, or None for another strategy."""

        if isinstance(self.parameters, DonchianBreakoutParameters):
            return self.parameters.exit_window
        return None


@dataclass(frozen=True, slots=True)
class SimulationAssumptions:
    """Financial and terminal-position assumptions for one simulation."""

    starting_cash: Decimal
    commission_bps: Decimal
    slippage_bps: Decimal
    end_of_test_policy: EndOfTestPolicy

    def __post_init__(self) -> None:
        """Validate exact financial inputs and the explicit end policy."""

        for name, value in (
            ("starting_cash", self.starting_cash),
            ("commission_bps", self.commission_bps),
            ("slippage_bps", self.slippage_bps),
        ):
            _require_finite_decimal(value, name)
        if self.starting_cash <= _ZERO:
            raise ValueError("starting_cash must be positive")
        if self.commission_bps < _ZERO or self.commission_bps > Decimal("10000"):
            raise ValueError("commission_bps must be between 0 and 10000")
        if self.slippage_bps < _ZERO or self.slippage_bps >= Decimal("10000"):
            raise ValueError("slippage_bps must be between 0 and less than 10000")
        if not isinstance(self.end_of_test_policy, EndOfTestPolicy):
            raise TypeError("end_of_test_policy must be an EndOfTestPolicy")


@dataclass(frozen=True, slots=True)
class PositionSizingRunConfiguration:
    """Exact parameters for the selected whole-share sizing mode."""

    mode: PositionSizingMode
    quantity: int | None
    cash_allocation_ratio: Decimal | None

    def __post_init__(self) -> None:
        """Require only the parameter meaningful to the selected mode."""

        if not isinstance(self.mode, PositionSizingMode):
            raise TypeError("mode must be a PositionSizingMode")
        if self.mode is PositionSizingMode.FIXED:
            if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
                raise TypeError("fixed sizing quantity must be an integer")
            if self.quantity <= 0:
                raise ValueError("fixed sizing quantity must be positive")
            if self.cash_allocation_ratio is not None:
                raise ValueError("cash_allocation_ratio is only valid for cash allocation")
            return
        if self.quantity is not None:
            raise ValueError("quantity is only valid for fixed sizing")
        if self.cash_allocation_ratio is None:
            raise ValueError("cash_allocation_ratio is required for cash allocation")
        _require_finite_decimal(self.cash_allocation_ratio, "cash_allocation_ratio")
        if not (_ZERO < self.cash_allocation_ratio <= Decimal("1")):
            raise ValueError("cash_allocation_ratio must be greater than 0 and at most 1")


@dataclass(frozen=True, slots=True)
class RiskPolicyRunConfiguration:
    """Configured pre-trade risk policy identity and exact limits."""

    name: str
    maximum_position_value: Decimal | None
    minimum_cash_reserve: Decimal | None

    def __post_init__(self) -> None:
        """Validate the policy label and optional Decimal limits."""

        _require_text(self.name, "risk policy name")
        if self.maximum_position_value is not None:
            _require_finite_decimal(
                self.maximum_position_value,
                "maximum_position_value",
            )
            if self.maximum_position_value <= _ZERO:
                raise ValueError("maximum_position_value must be positive")
        if self.minimum_cash_reserve is not None:
            _require_finite_decimal(
                self.minimum_cash_reserve,
                "minimum_cash_reserve",
            )
            if self.minimum_cash_reserve < _ZERO:
                raise ValueError("minimum_cash_reserve must be non-negative")
        expected_name = _risk_policy_name_from_limits(
            self.maximum_position_value,
            self.minimum_cash_reserve,
        )
        if self.name != expected_name:
            raise ValueError(
                f"risk policy name must be {expected_name!r} for the supplied limits"
            )


@dataclass(frozen=True, slots=True)
class BacktestRunMetadata:
    """Immutable reproducibility record for one completed backtest."""

    run_id: str
    generated_at: datetime
    application_version: str
    git_commit: str | None
    dataset: DatasetIdentity
    strategy: StrategyRunConfiguration
    simulation: SimulationAssumptions
    position_sizing: PositionSizingRunConfiguration
    risk_policy: RiskPolicyRunConfiguration
    risk_metric_settings: RiskMetricSettings | None

    def __post_init__(self) -> None:
        """Validate run identity and every strongly typed metadata component."""

        _require_text(self.run_id, "run_id")
        _require_aware_timestamp(self.generated_at, "generated_at")
        _require_text(self.application_version, "application_version")
        if self.git_commit is not None:
            _require_text(self.git_commit, "git_commit")
        if not isinstance(self.dataset, DatasetIdentity):
            raise TypeError("dataset must be a DatasetIdentity")
        if not isinstance(self.strategy, StrategyRunConfiguration):
            raise TypeError("strategy must be a StrategyRunConfiguration")
        if not isinstance(self.simulation, SimulationAssumptions):
            raise TypeError("simulation must be SimulationAssumptions")
        if not isinstance(self.position_sizing, PositionSizingRunConfiguration):
            raise TypeError("position_sizing must be PositionSizingRunConfiguration")
        if not isinstance(self.risk_policy, RiskPolicyRunConfiguration):
            raise TypeError("risk_policy must be RiskPolicyRunConfiguration")
        if self.risk_metric_settings is not None and not isinstance(
            self.risk_metric_settings,
            RiskMetricSettings,
        ):
            raise TypeError("risk_metric_settings must be RiskMetricSettings or None")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's unmodified raw bytes."""

    source = Path(path)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"data file does not exist: {source}") from exc
    except IsADirectoryError as exc:
        raise IsADirectoryError(f"data path is not a file: {source}") from exc
    except OSError as exc:
        raise OSError(f"could not hash data file {source}: {exc}") from exc
    return digest.hexdigest()


def safe_data_identifier(path: Path, *, base_directory: Path | None = None) -> str:
    """Return a project-relative POSIX identifier, or only the file name."""

    source = Path(path)
    base = Path.cwd() if base_directory is None else Path(base_directory)
    try:
        relative = source.resolve(strict=False).relative_to(base.resolve(strict=False))
    except ValueError:
        return source.name
    identifier = relative.as_posix()
    return identifier if identifier and identifier != "." else source.name


def installed_application_version() -> str:
    """Return the single application version from installed project metadata."""

    return application_version()


def current_git_commit(repository: Path) -> str | None:
    """Best-effort current commit lookup isolated from domain calculations."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repository),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else None


def create_backtest_run_metadata(
    *,
    data_path: Path,
    bars: tuple[MarketBar, ...],
    symbol: str,
    requested_start: datetime | None,
    requested_end: datetime | None,
    fast_window: int | None = None,
    slow_window: int | None = None,
    strategy_configuration: StrategyRunConfiguration | None = None,
    config: BacktestConfig,
    risk_metric_settings: RiskMetricSettings | None,
    run_id: str | None = None,
    generated_at: datetime | None = None,
    application_version: str | None = None,
    git_commit: str | None = None,
    base_directory: Path | None = None,
) -> BacktestRunMetadata:
    """Construct metadata once from validated bars and explicit run settings.

    ``fast_window`` and ``slow_window`` remain a compatibility path for existing
    SMA callers. New multi-strategy callers supply ``strategy_configuration``.
    """

    if not bars:
        raise ValueError("bars must contain at least one validated observation")
    if any(not isinstance(bar, MarketBar) for bar in bars):
        raise TypeError("bars must contain MarketBar values")
    selected_run_id = str(uuid4()) if run_id is None else run_id
    selected_timestamp = datetime.now(UTC) if generated_at is None else generated_at
    selected_version = (
        installed_application_version()
        if application_version is None
        else application_version
    )
    selected_strategy = _resolve_strategy_configuration(
        strategy_configuration=strategy_configuration,
        fast_window=fast_window,
        slow_window=slow_window,
    )
    sizing = PositionSizingRunConfiguration(
        mode=config.position_sizing_mode,
        quantity=(
            config.trade_quantity
            if config.position_sizing_mode is PositionSizingMode.FIXED
            else None
        ),
        cash_allocation_ratio=config.cash_allocation_ratio,
    )
    risk_name = _risk_policy_name(config)
    return BacktestRunMetadata(
        run_id=selected_run_id,
        generated_at=selected_timestamp,
        application_version=selected_version,
        git_commit=git_commit,
        dataset=DatasetIdentity(
            source_type="local_csv",
            identifier=safe_data_identifier(
                data_path,
                base_directory=base_directory,
            ),
            sha256=sha256_file(data_path),
            symbol=symbol,
            requested_start=requested_start,
            requested_end=requested_end,
            actual_start=bars[0].timestamp,
            actual_end=bars[-1].timestamp,
            bar_count=len(bars),
        ),
        strategy=selected_strategy,
        simulation=SimulationAssumptions(
            starting_cash=config.initial_cash,
            commission_bps=config.commission_bps,
            slippage_bps=config.slippage_bps,
            end_of_test_policy=config.end_of_test_policy,
        ),
        position_sizing=sizing,
        risk_policy=RiskPolicyRunConfiguration(
            name=risk_name,
            maximum_position_value=config.maximum_position_value,
            minimum_cash_reserve=config.minimum_cash_reserve,
        ),
        risk_metric_settings=risk_metric_settings,
    )


def _resolve_strategy_configuration(
    *,
    strategy_configuration: StrategyRunConfiguration | None,
    fast_window: int | None,
    slow_window: int | None,
) -> StrategyRunConfiguration:
    """Resolve the typed strategy metadata without ignoring supplied parameters."""

    if strategy_configuration is not None:
        if not isinstance(strategy_configuration, StrategyRunConfiguration):
            raise TypeError("strategy_configuration must be StrategyRunConfiguration")
        if fast_window is not None or slow_window is not None:
            raise ValueError(
                "fast_window and slow_window cannot accompany strategy_configuration"
            )
        return strategy_configuration
    if fast_window is None or slow_window is None:
        raise ValueError("legacy SMA metadata requires fast_window and slow_window")
    return StrategyRunConfiguration(
        name=StrategyName.SMA_CROSSOVER,
        parameters=SmaCrossoverParameters(
            fast_window=fast_window,
            slow_window=slow_window,
        ),
    )


def _risk_policy_name(config: BacktestConfig) -> str:
    """Return the stable name of the configured pre-trade policy composition."""

    return _risk_policy_name_from_limits(
        config.maximum_position_value,
        config.minimum_cash_reserve,
    )


def _risk_policy_name_from_limits(
    maximum_position_value: Decimal | None,
    minimum_cash_reserve: Decimal | None,
) -> str:
    """Return a policy name that cannot disagree with its supplied limits."""

    if maximum_position_value is not None and minimum_cash_reserve is not None:
        return "composite"
    if maximum_position_value is not None:
        return "maximum_position_value"
    if minimum_cash_reserve is not None:
        return "minimum_cash_reserve"
    return "allow_all"
