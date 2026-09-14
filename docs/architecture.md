# Architecture

The project separates financial decisions from orchestration and presentation. Data
providers return immutable market bars; strategies consume completed bars and emit
signals; sizing and risk policies determine an approved whole-share quantity; and the
backtesting engine records simulated fills, cash, positions and equity.

```text
CsvMarketDataProvider
        │ MarketBar
        ▼
Strategy ──► PositionSizer ──► RiskPolicy ──► BacktestEngine
                                                  │
                    ┌─────────────────────────────┼─────────────────────────┐
                    ▼                             ▼                         ▼
              performance                   benchmarks                reporting
                    └─────────────────────────────┬─────────────────────────┘
                                                  ▼
                                             evaluation
```

## Packages

| Package | Responsibility |
|---|---|
| `data` | Provider contracts, strict CSV parsing and approved dataset preparation |
| `strategies` | Typed strategy interface, closed registry, SMA and Donchian signals |
| `simulation` | Shared exact commission and slippage arithmetic |
| `risk` | Fixed/cash-allocation sizing and composable pre-trade constraints |
| `backtesting` | Long-only, next-bar event simulation and ledger reconciliation |
| `performance` | Returns, drawdown, risk ratios, exposure and closed-trade reconstruction |
| `benchmarks` | Capital-matched buy-and-hold execution and comparison |
| `evaluation` | Split manifests, fingerprints, provenance and sealed holdout gates |
| `experiments` | Ordered, explicitly declared batch execution |
| `reporting` | Human-readable and structured JSON/CSV exports |

Interfaces allow providers, strategies, sizing rules and risk policies to be tested
independently. Pydantic validates external configuration; frozen dataclasses and
tuples represent internal state. Financial values use `Decimal` rather than binary
floating point.

## Dependency discipline

Performance and reporting code consume completed backtest results and cannot alter
fills. Benchmark execution uses the same cost and sizing arithmetic as the strategy.
The experiment and evaluation runners call the same single-run pipeline as the CLI,
so batch research does not contain a second implementation of financial formulas.

The public snapshot contains no brokerage adapter, wallet support or live-order
interface.
