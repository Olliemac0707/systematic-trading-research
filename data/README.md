# Data policy

`sample_prices.csv` is a small synthetic dataset committed solely for offline
examples and tests. It is not market data and is not investment information.

The eight-ETF study used separately acquired Yahoo Finance observations. Those
vendor files and the generated detailed reports are deliberately not redistributed.
The approved date range, transformation policy and SHA-256 identities of the exact
research inputs are recorded in [data provenance](../docs/data-provenance.md).

To reproduce the software pipeline without third-party data, use the synthetic
example documented in the top-level README. Reproducing the archived historical
numbers additionally requires input files matching every registered dataset hash.
