"""Canonical exact configuration serialization and SHA-256 fingerprints."""

import hashlib
import json
from decimal import Decimal

from trading_research.config import PositionSizingMode
from trading_research.evaluation.models import FrozenEvaluationConfiguration
from trading_research.strategies import (
    DonchianBreakoutParameters,
    SmaCrossoverParameters,
)


def canonical_configuration_json(configuration: FrozenEvaluationConfiguration) -> str:
    """Return stable JSON with strings for every Decimal and value for every enum."""

    if not isinstance(configuration, FrozenEvaluationConfiguration):
        raise TypeError("configuration must be FrozenEvaluationConfiguration")
    parameters = configuration.strategy.parameters
    if isinstance(parameters, SmaCrossoverParameters):
        strategy_parameters = {
            "fast_window": parameters.fast_window,
            "slow_window": parameters.slow_window,
        }
    elif isinstance(parameters, DonchianBreakoutParameters):
        strategy_parameters = {
            "entry_window": parameters.entry_window,
            "exit_window": parameters.exit_window,
        }
    else:
        raise TypeError("unsupported strategy parameter model")

    backtest = configuration.backtest
    sizing_parameters: dict[str, object]
    if backtest.position_sizing_mode is PositionSizingMode.FIXED:
        sizing_parameters = {"quantity": backtest.trade_quantity}
    else:
        sizing_parameters = {
            "cash_allocation_ratio": _decimal_string(backtest.cash_allocation_ratio)
        }
    risk_metrics = configuration.risk_metrics
    payload: dict[str, object] = {
        "benchmark": configuration.benchmark.value,
        "end_of_test_policy": backtest.end_of_test_policy.value,
        "risk_metrics": (
            None
            if risk_metrics is None
            else {
                "periods_per_year": _decimal_string(risk_metrics.periods_per_year),
                "risk_free_rate_per_period": _decimal_string(
                    risk_metrics.risk_free_rate_per_period
                ),
                "target_return_per_period": _decimal_string(
                    risk_metrics.target_return_per_period
                ),
            }
        ),
        "risk_policy": {
            "maximum_position_value": _decimal_string(
                backtest.maximum_position_value
            ),
            "minimum_cash_reserve": _decimal_string(backtest.minimum_cash_reserve),
        },
        "simulation": {
            "commission_bps": _decimal_string(backtest.commission_bps),
            "slippage_bps": _decimal_string(backtest.slippage_bps),
            "starting_cash": _decimal_string(backtest.initial_cash),
        },
        "strategy": {
            "name": configuration.strategy.name.value,
            "parameters": strategy_parameters,
        },
        "position_sizing": {
            "mode": backtest.position_sizing_mode.value,
            "parameters": sizing_parameters,
        },
    }
    if configuration.allow_unapproved_volume:
        payload["data_policy"] = {"allow_unapproved_volume": True}
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def configuration_sha256(configuration: FrozenEvaluationConfiguration) -> str:
    """Return a deterministic fingerprint excluding paths, timestamps, and run IDs."""

    canonical = canonical_configuration_json(configuration)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _decimal_string(value: Decimal | None) -> str | None:
    """Return the exact Decimal representation without float conversion."""

    if value is None:
        return None
    if not isinstance(value, Decimal):
        raise TypeError("configuration financial values must be Decimal")
    if not value.is_finite():
        raise ValueError("configuration financial values must be finite")
    return str(value)
