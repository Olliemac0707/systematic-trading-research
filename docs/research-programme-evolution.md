# Research programme evolution: from strategy testing to implementation research

## Overview

This project began as a reproducible framework for testing systematic investment ideas under explicit historical assumptions. The research programme has since shifted toward a broader question:

> How can a systematic strategy be evaluated, frozen, stress-tested and translated into an implementation study without allowing later results to alter the original research decision?

The current phase treats the strategy itself as a **frozen research object**. Rather than continuing to search for improved parameters, the work now focuses on whether the same frozen decision process survives increasingly realistic accounting, currency and implementation layers.

Exact strategy parameters, candidate identifiers and performance outputs are intentionally excluded from this public document. The purpose of the public repository is to demonstrate the **research process, software architecture and experimental controls**, rather than disclose a reproducible trading specification.

## Research progression

The programme is structured as a sequence of increasingly demanding research layers:

```text
strategy hypothesis
    ↓
historical evaluation
    ↓
pre-registered candidate selection
    ↓
robustness diagnostics
    ↓
frozen strategy specification
    ↓
prospective observation
    ↓
currency / economic-exposure modelling
    ↓
implementation and proxy-divergence research
```

Each layer answers a different question. Passing one layer does not imply that later layers will pass, and later analysis is not permitted to retroactively redefine the earlier hypothesis.

## Experimental discipline

The framework emphasises controls that are common in formal empirical research but are often absent from hobbyist backtests.

### Frozen specifications

Once a research candidate is selected, its signal logic, selection rules, execution timing, cost assumptions and evaluation protocol are frozen. Later diagnostics may challenge the candidate, but they do not silently modify it.

### Outcome-blind diagnostics

Before one-shot evaluations, the implementation can be exercised through structural and redacted shadow runs. These runs test parsers, accounting identities, calendars, distributions, FX alignment, serialization and publication logic while withholding the economic outcome.

This separates two questions:

1. **Will the software execute the registered experiment correctly?**
2. **What result does the registered experiment produce?**

The first should be answered before the second is revealed.

### Reconciliation

Portfolio accounting is treated as an auditable identity rather than a display calculation. Holdings, cash, transaction costs, execution prices and equity are reconciled against frozen authority records before downstream results are accepted.

### Provenance

Research inputs and outputs are bound to cryptographic identities. The programme uses hashes, immutable protocol files, execution receipts, manifests and completion attestations so that a reported result can be tied back to the exact code, data and assumptions that produced it.

### Fail-closed execution

Registered studies use no-clobber output rules and one-shot execution controls. If an authorized run fails after consumption, that failed attempt is preserved rather than erased and silently retried.

This is intentionally conservative: failed execution evidence is part of the research record.

## Validation programme

The case study has been subjected to multiple classes of diagnostics, including:

- walk-forward / fold-based historical evaluation;
- placebo-style ranking diagnostics;
- moving-block bootstrap analysis;
- parameter-neighbourhood checks;
- transaction-cost sensitivity;
- synthetic-price Monte Carlo diagnostics;
- exact accounting reconciliation;
- distribution and corporate-action handling;
- FX alignment and currency translation checks;
- redacted full-path execution diagnostics;
- static typing, linting and extensive automated tests.

These diagnostics do not prove future profitability. Their role is to identify instability, implementation errors, data leakage, hidden assumptions and research-process weaknesses.

## Shift to economic-exposure research

A later research phase asked how a frozen strategy expressed in one currency should be interpreted from the perspective of an investor whose base currency differs from that of the underlying assets.

That required separating several effects that are often conflated in simple backtests:

- price-only market movement;
- distributions and total-economic return;
- currency translation;
- cash treatment;
- transaction-boundary ownership;
- corporate-action treatment.

The purpose of this stage was **accounting equivalence**, not strategy re-selection.

The resulting model is deliberately described as an economic-exposure model rather than as historical performance of a particular investment product.

## Why product implementation is a separate study

Economic exposure and investable product performance are not the same thing.

A practical implementation may introduce additional differences through:

- fund or instrument tracking;
- product-specific fees;
- market spreads;
- trading-session overlap;
- whole-share constraints;
- residual cash;
- execution timing;
- product structure;
- local-market conventions.

These effects are therefore deferred to a distinct **implementation / proxy-divergence study**.

Keeping implementation research separate from signal research reduces the risk of using implementation details as a post-hoc excuse to retune the strategy.

## Current research question

The next stage asks:

> How closely can a UK-accessible implementation reproduce the economic exposure of the already-frozen research strategy once product tracking, fees, market microstructure and execution constraints are introduced?

This is an implementation question, not a search for a new strategy.

The planned work includes:

- measuring divergence between theoretical exposure and accessible proxies;
- modelling product-specific costs separately from the original signal;
- examining execution during overlapping trading hours;
- handling calendar and daylight-saving differences;
- preserving cash and whole-share constraints;
- distinguishing structural index differences from ordinary tracking error;
- maintaining a full audit trail from frozen signal to implementation outcome.

## Public/private boundary

This public repository intentionally documents the **methodology and engineering discipline** while withholding details that would make the current frozen strategy straightforward to reproduce.

Public material may describe:

- research architecture;
- experimental controls;
- validation methods;
- software design;
- accounting and provenance principles;
- implementation questions;
- high-level conclusions.

Private research records retain:

- exact signal parameters;
- exact candidate identifiers;
- detailed instrument mappings;
- exact fold-level performance;
- decomposition values;
- canonical result artifacts;
- execution receipts and attestations;
- implementation details that materially reveal the trading specification.

This separation is intentional. It allows the project to remain academically useful and auditable without turning the public repository into a recipe for reproducing the underlying strategy.

## Research status

The current frozen case study has completed its economic-exposure equivalence stage under the registered protocol.

The next phase is implementation research.

No conclusion in this repository should be interpreted as a forecast, investment recommendation or claim of future profitability.
