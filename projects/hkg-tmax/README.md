# HKG Tmax research system

The active HKG mini-project for leakage-safe daily maximum-temperature
forecasting, probability calibration, and weather-market research. It is one
component of the weather-markets monorepo, not a nested repository.

Start with [START_HERE.md](START_HERE.md), then follow [AGENTS.md](AGENTS.md).

## Setup

```powershell
Copy-Item .env.example .env
# Set external data/run roots and local credentials in .env.
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,research]"
make doctor-fast
make test-fast
make validate
```

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
