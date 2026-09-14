# Complete eight-ETF study result

## Development selection

All four declared candidates met the development eligibility gate. Their overall
rank scores were:

| Candidate | Overall rank | Positive returns | Positive Sharpe | Drawdown improvements | Minimum closed trades |
|---|---:|---:|---:|---:|---:|
| SMA 20/50 | 2.7500 | 7/8 | 8/8 | 8/8 | 34 |
| **SMA 50/200** | **1.5833** | **8/8** | **8/8** | **8/8** | **5** |
| Donchian 20/10 | 2.8750 | 8/8 | 8/8 | 8/8 | 76 |
| Donchian 50/20 | 2.7917 | 6/8 | 7/8 | 8/8 | 34 |

SMA 50/200 was selected once for all eight assets. It was selected for the registered
cross-asset rank, risk control and lower turnover—not because it maximised absolute
return.

## Holdout gate

The first and only holdout attempt completed all eight authorised evaluations and
reconciled them exactly.

| Criterion | Observed | Required | Result |
|---|---:|---:|---|
| Positive net return | 6/8 | 6/8 | Pass |
| Positive periodic Sharpe | 7/8 | 6/8 | Pass |
| Smaller maximum drawdown than buy-and-hold | 3/8 | 5/8 | **Fail** |

**Registered classification:** `GENERALISATION NOT SUPPORTED BY PRE-REGISTERED GATE`.

The failure was caused solely by the drawdown criterion. Drawdown advantage persisted
for QQQ, XLE and TLT. IWM and XLV produced negative net returns; IWM produced the only
negative Sharpe ratio. Strategy return exceeded buy-and-hold only for TLT.

## Aggregate descriptive statistics

The values below aggregate eight independent accounts; they are not one continuously
compounded portfolio.

| Measure | Value |
|---|---:|
| Median net return | 35.3221% |
| Mean net return | 62.8244% |
| Median strategy maximum drawdown | -32.8892% |
| Median buy-and-hold return | 106.5146% |
| Median benchmark maximum drawdown | -38.3284% |
| Mean exposure | 65.7386% |
| Closed trades | 48 |
| Total commission | 1,168.84 |
| Total slippage attribution | 5,844.34 |
| Total simulated cost | 7,013.19 |

## Interpretation

The result shows that a configuration can satisfy development criteria and retain
some favourable holdout properties while still failing the complete generalisation
claim. The appropriate conclusion is not to search the holdout for a replacement.
It is to retain the negative result and require a new protocol plus fresh evidence for
any modified hypothesis.

The results are simulated, dividend-excluded price returns. Costs are modelled, market
data was historically available, and no result is evidence of future profitability.
