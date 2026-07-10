# KLGA Tmax research system

Leakage-safe KLGA daily maximum-temperature research, provider acquisition,
forecast evaluation, and weather-market analysis.

Read [START_HERE.md](START_HERE.md) and [AGENTS.md](AGENTS.md) before working.

```text
src/             Python package
tests/           offline tests
alembic/         database migrations
config/          governed configuration
experiments/     compact experiment code/evidence
docs/            strategy, context, implementation docs, status
scripts/         bounded entry points
```

Generated artifacts belong under `KLGA_RUN_ROOT` (recommended on this machine:
`C:/klga_tmax_data/runs`). Credentials and `KLGA_DB_URL` stay in the ignored
local environment. Research/backtest is the default; this project does not
place orders by default.

Setup:

```powershell
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
```
