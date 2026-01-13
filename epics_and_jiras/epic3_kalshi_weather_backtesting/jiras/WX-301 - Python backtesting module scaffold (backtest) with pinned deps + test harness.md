# WX-301 — Python backtesting module scaffold (backtest/) with pinned deps + test harness

## Objective
Create the Python project scaffold under `backtest/` that will host all Epic #3 code, with a minimal runnable CLI and test harness.

## Why this matters
Epic #3 will ship a non-trivial data + simulation engine. We need:
- a clean module boundary
- repeatable installs (pinned deps)
- a deterministic test harness from day 1

## Requirements
### Folder structure
Create:
- `backtest/`
  - `pyproject.toml` (preferred) OR `requirements.txt` + lock file
  - `src/weather_backtest/`
    - `__init__.py`
    - `cli.py` (or `run_backtest.py`)
    - `kalshi_client/` (empty placeholder)
    - `ingest/` (empty placeholder)
    - `engine/` (empty placeholder)
    - `reporting/` (empty placeholder)
  - `tests/`
    - `test_smoke.py` (placeholder)
  - `configs/`
    - `backtest.example.yaml`
  - `README.md`

### Dependencies (minimum)
Pin versions (or lock):
- requests or httpx
- pydantic (optional but recommended)
- python-dateutil
- pytz or zoneinfo (py3.9+ zoneinfo ok)
- pandas
- numpy
- sqlalchemy + mysql driver (pymysql or mysqlclient)
- pyyaml
- matplotlib (for optional plotting)
- pytest

### CLI entrypoint (stub is ok)
A runnable entrypoint must exist:
- `python -m weather_backtest.cli --help`
- Or `python -m weather_backtest.run_backtest --help`

The CLI can be a stub that prints “not implemented” but must parse a config path.

### Root build compatibility
- Root `mvn clean install` must still pass.
- Python module is not compiled by Maven; it must not break the Java build.

## Acceptance Criteria
- [ ] Installing deps succeeds in a clean environment (`pip install -r ...` or `uv sync` / `poetry install`).
- [ ] `pytest` runs and passes with at least one smoke test.
- [ ] CLI help command works and exits 0.
- [ ] `backtest/README.md` includes:
  - how to create venv
  - how to install deps
  - how to run tests
  - how to run a backtest (placeholder ok)
