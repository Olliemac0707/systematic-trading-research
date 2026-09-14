# Reproducibility

## Requirements

- Python 3.12
- Git
- A virtual environment

## Install and verify

PowerShell:

```powershell
py -3.12 -m venv .venv
$venv = (Resolve-Path -LiteralPath '.venv').Path
$env:VIRTUAL_ENV = $venv
$env:Path = "$venv\Scripts;$env:Path"
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python -m pytest --cov=trading_research --cov-report=term-missing
ruff check .
mypy src tests
```

POSIX shells:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python -m pytest --cov=trading_research --cov-report=term-missing
ruff check .
mypy src tests
```

The test suite uses synthetic in-memory or temporary-file inputs and must not download
market data.

## Representative experiment

```bash
python -m trading_research experiment \
  --manifest experiments/example.toml \
  --output-dir reports/example
```

This executes declared SMA and Donchian examples against committed synthetic prices.
It writes individual structured reports, an aggregate CSV and a batch manifest. The
output directory is ignored by Git and protected against accidental overwrite.

## Historical result verification

The complete eight-ETF numerical study cannot be rerun from this repository alone
because vendor market data is intentionally excluded. A valid reconstruction requires
all eight prepared files to match the SHA-256 identities in `data-provenance.md`.

Without those files, reviewers can still:

1. inspect the full 32-entry registered manifest;
2. verify the tracked selection and holdout records;
3. run every software test and synthetic integration path; and
4. reproduce representative backtests and structured reports offline.

This distinction avoids claiming that a vendor-data-dependent historical result is
fully reproducible from public inputs when it is not.
