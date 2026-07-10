# HKG Tmax: start here

This is the active HKG Tmax research project inside the weather-markets
monorepo. It forecasts the contract-authoritative Hong Kong Observatory daily
maximum temperature with point-in-time-safe inputs and calibrated probability
distributions.

## Safe orientation

1. Read `../../AGENTS.md` and `AGENTS.md` completely.
2. Read `docs/status/CURRENT_STATE.md` and `docs/INDEX.md`.
3. Confirm Git root, branch, remote, and tracked-only status.
4. Confirm `.env` points `HKG_TMAX_DATA_ROOT` and `HKG_TMAX_RUN_ROOT` outside
   the repository. Never print credential values.
5. Run `make doctor-fast` and only the focused tests required by the task.

Orientation is offline. It must not start collectors, backfills, schedulers,
servers, databases, containers, training, or market calls.

## Working loop

```text
orient -> identify governing contract -> predeclare experiment if research ->
make the smallest coherent change -> focused verification ->
record evidence and negative results -> update current documentation
```

## Canonical locations

```text
src/                 Python packages
tests/               offline and focused tests
config/              governed project/source/acquisition/experiment config
db/                  migrations, schemas, SQL
experiments/         campaign protocols, compact evidence, registry/templates
docs/                architecture, contracts, data, research, runbooks, status
planning/            task and work-package specifications
scripts/             bounded operator/research entry points
apps/                HKG-specific user interfaces
```

Data and generated runs are external. See `docs/data/STORAGE_LAYOUT.md`.
