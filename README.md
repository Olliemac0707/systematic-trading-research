# Systematic Trading Research

[![CI](https://github.com/Olliemac0707/systematic-trading-research/actions/workflows/ci.yml/badge.svg)](https://github.com/Olliemac0707/systematic-trading-research/actions/workflows/ci.yml)

A Python 3.12 framework for reproducible systematic-strategy research, event-driven
backtesting, benchmark comparison, risk analysis and sealed out-of-sample evaluation.

The project asks whether one trend-following configuration, selected across a
diversified eight-ETF development universe, generalises to a later sealed holdout.
It emphasises experimental discipline: frozen configurations, next-bar execution,
exact accounting, data and artifact hashes, and no retuning after holdout inspection.

> Research-only software. All results are simulated. There is no brokerage
> integration, live-order path or claim of future profitability.

## What this demonstrates

- Modular market-data, strategy, backtesting, risk, benchmark, evaluation and
  reporting components.
- SMA crossover and Donchian breakout strategies evaluated across SPY, QQQ, IWM,
  XLF, XLE, XLV, TLT and GLD.
- Explicit development/holdout separation with configuration fingerprints and
  safeguards against look-ahead bias and data leakage.
- Whole-share sizing, commissions, adverse slippage, exposure analysis, drawdown,
  return/risk metrics, trade reconstruction and capital-matched benchmarking.
- Offline automated tests, strict typing and reproducible structured outputs.

## Study at a glance

| Stage | Period | Scope |
|---|---|---:|
| Development | 2005-01-03 to 2018-12-31 | 8 assets × 4 declared candidates |
| Sealed holdout | 2019-01-02 to 2025-12-31 | 8 assets × 1 frozen candidate |

The development procedure selected one universal SMA 50/200 configuration. The
first and only holdout run passed its integrity gate but **did not pass the complete
pre-registered generalisation gate**: positive return and Sharpe requirements were
met, while drawdown improvement was observed in 3 of 8 assets versus the required
5. No alternative was substituted and the holdout was not reused for selection.

That negative conclusion is retained because this repository demonstrates a
research process, not a search for an attractive backtest.

## Architecture

```text
market data → strategies → sizing/risk → simulated execution
                                      ├── performance
                                      ├── benchmarks
                                      ├── evaluation
                                      └── reporting
```

The package uses typed interfaces and immutable domain models so financial logic
can be tested independently. Normal strategy signals are calculated from completed
bars and execute at the following bar's open.

## Run the offline example

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python -m trading_research experiment \
  --manifest experiments/example.toml \
  --output-dir reports/example
```

The example uses committed synthetic prices. Historical vendor datasets are not
redistributed.

## Verify

```bash
pytest --cov=trading_research --cov-report=term-missing
ruff check .
mypy src tests
```

CI is offline and does not download market data.

## Read more

- [Architecture](docs/architecture.md)
- [Research methodology](docs/methodology.md)
- [Data provenance and hashes](docs/data-provenance.md)
- [Execution assumptions](docs/execution-assumptions.md)
- [Complete study results](docs/results.md)
- [Reproducibility](docs/reproducibility.md)
- [Public-snapshot provenance](docs/public-snapshot-provenance.md)

## License

Code is available under the [MIT License](LICENSE). Third-party market data is not
included or licensed by this repository.
