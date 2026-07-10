# HKG Tmax research system

The active HKG mini-project for leakage-safe daily maximum-temperature
forecasting, probability calibration, and weather-market research. It is one
component of the weather-markets monorepo, not a nested repository.

The governing entry point is the root constitution plus project
[AGENTS.md](AGENTS.md). Follow its section 2 exactly; that sequence routes you
through [START_HERE.md](START_HERE.md), this README, the code map, and current
state without loading the historical corpus.

## Setup

```powershell
Copy-Item .env.example .env
# Set external data/run roots and local credentials in .env.
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,research]"
.\.venv\Scripts\python.exe -m hkg_tmax doctor
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_bootstrap_safety_contract.py `
  tests/test_config_and_sources.py `
  tests/test_experiments.py `
  tests/test_validation.py `
  tests/test_hko_backfill.py `
  tests/hkg_t24/test_h24n_contract_policy.py `
  tests/hkg_t24/test_schema_sql_contract.py `
  tests/test_demo_trading_migration.py
.\.venv\Scripts\python.exe -m hkg_tmax validate all
.\.venv\Scripts\python.exe scripts/manage_campaign_documentation.py check
```

On systems with `make`, `make doctor-fast`, `make test-fast`, and
`make validate` are equivalent aliases for these bounded gates.

All collectors, backfills, schedulers, database stacks, and network calls are
disabled or dry-run by default. `--execute` acknowledges intent but does not
replace provider/request/runtime budgets.

## Repository map

| Path | Ownership |
|---|---|
| `src/` | importable production/research packages |
| `tests/` | offline tests and contract checks |
| `config/` | project, sources, acquisition, and experiment configuration |
| `db/` | database migrations, schemas, and SQL |
| `experiments/` | campaigns, registry, and standard template |
| `docs/` | one canonical documentation tree |
| `planning/` | task specifications and work packages |
| `scripts/` | bounded command entry points; see `scripts/README.md` |
| `apps/polymarket-backtester/` | local demo/backtester UI |

Raw data, generated reports, logs, models, predictions, and large experiment
outputs belong under the configured external roots, never in Git.

## Research boundary

Settlement parity, publication-time semantics, locked out-of-sample validation,
probability calibration, and reproducibility remain mandatory gates. Market
edge is not guaranteed profit, and this project does not place orders.
