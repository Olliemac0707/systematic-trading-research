# Research methodology

## Question and scope

The study asked whether one universal trend-following configuration could be selected
on a development sample and then generalise across eight exchange-traded funds in a
sealed later holdout.

The universe was fixed as SPY, QQQ, IWM, XLF, XLE, XLV, TLT and GLD. Four candidates
were declared before selection:

| Candidate | Parameters |
|---|---|
| SMA crossover | 20-session fast, 50-session slow |
| SMA crossover | 50-session fast, 200-session slow |
| Donchian breakout | 20-session entry, 10-session exit |
| Donchian breakout | 50-session entry, 20-session exit |

The development interval was 2005-01-03 through 2018-12-31. The holdout interval was
2019-01-02 through 2025-12-31. Both boundaries were inclusive. Pre-holdout observations
could supply indicator history under the registered carry-history policy, but could
not create holdout trades, returns or equity.

## Development eligibility

A candidate was eligible only when all eight asset evaluations completed and
reconciled, with:

- positive net return on at least six assets;
- positive periodic Sharpe ratio on at least six assets;
- a smaller maximum-drawdown magnitude than buy-and-hold on at least five assets;
- at least three closed trades for every asset; and
- no unresolved integrity warnings.

## Selection

Eligible candidates were ranked separately for each asset on net return descending,
periodic Sharpe ratio descending and maximum-drawdown magnitude ascending. The overall
score was the arithmetic mean of those 24 ranks. Exact ties received average ranks.

The registered tie-breakers, in order, were higher median periodic Sharpe ratio,
smaller median drawdown magnitude, higher median net return, lower aggregate simulated
cost and fewer closed trades. No tie-breaker was required. SMA 50/200 was selected
with an overall development rank score of
`1.583333333333333333333333333`.

## Sealed holdout

Exactly one SMA 50/200 evaluation was authorised for each ETF. The holdout integrity
gate required all eight evaluations to use the frozen candidate, exact configuration
fingerprint and registered datasets; reconcile financially; complete the required
period; and avoid any alternative candidate execution.

Generalisation required at least six positive returns, six positive Sharpe ratios and
five drawdown improvements. The first two thresholds passed; drawdown improvement was
observed for only three assets. The registered conclusion was therefore
`GENERALISATION NOT SUPPORTED BY PRE-REGISTERED GATE`.

The holdout is not reusable for selecting a modified strategy. Any new hypothesis
requires a new protocol and genuinely fresh evidence.

## Bias and leakage controls

- Strategy parameters and the universe were explicit rather than dynamically loaded.
- The program did not optimise parameters or choose the split.
- Normal signals used only completed bars and executed at the following bar's open.
- Development and holdout outputs carried the same configuration SHA-256.
- Dataset provenance and exact input hashes were checked before sealed execution.
- The holdout runner rejected non-selected candidates and changed assumptions.
- No candidate or parameter was substituted after the holdout outcome was known.

These controls improve the validity of the experiment, but they do not remove market
selection bias, multiple-researcher degrees of freedom outside the registered study,
model risk or the limitations of simulated execution.
