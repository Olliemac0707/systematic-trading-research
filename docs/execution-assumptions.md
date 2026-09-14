# Execution assumptions

## Signal timing

Strategies evaluate completed bars. A normal signal produced at bar `t` executes at
bar `t+1` open; the engine never fills it on the close that generated the signal.

SMA crossover compares fast and slow rolling close averages. Donchian entry compares
the current close with the maximum of the preceding entry-window closes, and exit
compares it with the minimum of the preceding exit-window closes. The current close is
excluded from both Donchian channels and equality is not a breakout.

## Capital and sizing

Each asset evaluation starts with an independent simulated account containing
`100000` cash. The study uses cash-allocation sizing at ratio `1.0`, rounded down to
whole shares after estimated buy slippage and proportional commission. An affordability
guard prevents negative cash. The framework also implements fixed-quantity sizing,
maximum-position-value limits and minimum-cash-reserve constraints.

No leverage, short selling or fractional shares are used in the registered study.

## Costs

- Commission: 1 basis point of fill notional
- Slippage: 5 basis points, adverse on both buys and sells
- Cash return: zero

Shared `Decimal` helpers calculate expected sizing costs and actual execution costs,
preventing a second slippage or commission implementation from diverging.

## End-of-test policy

Open strategy positions are liquidated at the final eligible bar's close. This is an
explicit backtest convention, not a fabricated future bar, and is recorded separately
from next-open strategy execution. It affects realised profit, costs and trade counts.

## Benchmark

Every study run uses a capital-matched buy-and-hold benchmark with the same starting
cash, whole-share constraint, commission, slippage and final-liquidation convention.
Strategy-minus-benchmark results therefore compare like-for-like simulated execution
assumptions.

## Analytics

The public implementation includes:

- starting and ending cash/equity, realised and unrealised P&L and net return;
- absolute and percentage maximum drawdown;
- period returns, mean return, volatility, Sharpe and Sortino ratios when configured;
- exposure, time in market, cash share and capital utilisation;
- closed-trade reconstruction, win/loss counts, holding periods and cost attribution;
- benchmark returns, drawdown and exact strategy-minus-benchmark differences; and
- immutable pre-trade decisions showing proposed, reduced, approved or rejected size.

These are analytical descriptions of a simulation. They are not forecasts or risk
guarantees.
