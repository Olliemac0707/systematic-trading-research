# Data provenance and hashes

## Historical source

The study used daily Yahoo Finance observations acquired through pinned
`yfinance==1.5.1` for SPY, QQQ, IWM, XLF, XLE, XLV, TLT and GLD. The approved source
interval was 2005-01-03 through 2025-12-31.

The request fixed `auto_adjust=False`, `actions=True`, `repair=False`, `keepna=True`
and `threads=False`. Raw observations retained Open, High, Low, Close, Adj Close,
Volume, Dividends and Stock Splits for validation and provenance.

Prepared research CSVs mapped Yahoo Open, High, Low, Close and Volume without applying
a second split adjustment. `Close` was treated as split-adjusted and dividend-excluded.
`Adj Close` and dividends were excluded from signals, fills, valuation, benchmarks and
returns. Volume was validated but was not an input to either strategy.

The holdout ran from prepared local files and made no live network request.

## Exact dataset identities

| Symbol | SHA-256 |
|---|---|
| SPY | `d1edbc93f2e89cff7827b96c2b7a820281a07cabac1dffa6443037011ae6a2ec` |
| QQQ | `c3a23aa983a3a0cd5080edf11bcf7e1ea1c1a927b72fdddcae5e6fe446371034` |
| IWM | `eef32e78bceaa966a5ed767d359d0db64223fe12a42b251b16e1a1eab985e804` |
| XLF | `61d7384e4374830786a716e414201e801c49605d8a02575dad34d896df258e1f` |
| XLE | `7e98ad15262bfc88c657f6a803d757b487a14aa2c9dd2e6763910bb5877b6b94` |
| XLV | `8d5917f63c07e23b628db87a691e894256a8ea735265d98c38ac5b7a17ced99d` |
| TLT | `66aa6d3ef6cff63fd6a8db3a05879e863d91797bb041fc6124f41db705d3cfd4` |
| GLD | `e3ddffa9c366ac8385883dca6c38d9805e1ab4b2360476a724b88020a70a5656` |

## Registered artifact identities

| Artifact | SHA-256 |
|---|---|
| Study manifest | `73769c33454cc59b59c22a0091940eaa84e83167d7a1a910ceeabeeb9c1bc236` |
| Development selection record | `d663a2ce3e0787fc10d2441fae9de44c3a93c6f20898d851df78178a519a0cff` |
| Development summary CSV | `a60d20551c9aa5f079427ef0c34135df79cae52116b7ae2624307863b66fe9df` |
| Holdout configuration | `2264fbf414804be31651f207e98c0a618346a783132c0352eec65337e3b499d7` |

The tracked compact holdout record also lists hashes for every original detailed
output. Generated reports and vendor datasets are intentionally excluded from this
public snapshot. Their absence prevents full numerical reproduction without the
registered inputs, but does not prevent inspection of the protocol, archived result
or complete offline software pipeline.

Raw third-party market data is not redistributed or licensed by this repository.
