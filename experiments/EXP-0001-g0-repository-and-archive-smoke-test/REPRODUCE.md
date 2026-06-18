# Reproduce

## Clean environment

```powershell
git checkout <G0-commit>
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

On Linux/WSL, use `python3 -m venv .venv`, activate the venv, and run the same
Python module commands. The full `.[research,dev]` install is intended for
later research work; the G0 doctor/test/validate/fetch path requires base/dev
dependencies only.

## Verify inputs

```powershell
.\.venv\Scripts\python.exe -m hkg_tmax doctor
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m hkg_tmax validate all
.\.venv\Scripts\python.exe -m hkg_tmax manifest
```

## Run

```powershell
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m hkg_tmax sources fetch --tag bootstrap_now
.\.venv\Scripts\python.exe -m hkg_tmax sources fetch --tag bootstrap_now
.\.venv\Scripts\python.exe -m hkg_tmax sources report
.\.venv\Scripts\python.exe -m pytest tests/test_fetch.py tests/test_hko.py
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m mypy src
```

## Expected outputs

| File | SHA-256 or tolerance |
|---|---|
| reports/source_inventory.md | deterministic from `config/data_sources.yaml`; observed `0feb442378de719d374352e4db7c1b221ee85d59f9c325c7404262e60cec6674` |
| results/metrics.json | exact JSON values except timestamps/commit fields if rerun after a new commit |
| data/raw/* | raw live payload hashes may change if provider payloads change; each accepted payload must have a recomputable SHA-256 and sidecar |
| results/predictions.parquet | not produced in G0 |

## Expected metric tolerances

Deterministic pass/fail. Expected test count after this experiment: 34 passing
tests.

## External immutable data locations

Raw snapshots are local under `data/raw/` and are intentionally excluded from
Git except `.gitkeep` and README files. Exact snapshot hashes are recorded in
`DATA_MANIFEST.yaml`.

## Known platform differences

`make` was not installed in the PowerShell environment, so direct Python module
commands were used. The initial Windows full research install hit a long-path
failure in `statsmodels`; this does not affect the base/dev G0 path. The
PowerShell bootstrap script was patched to stop on native command failures.

## No undocumented steps

No manual data editing was performed. `.env` is copied from `.env.example` and
is ignored by Git.
