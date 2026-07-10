# HKG Tmax: start here

This is the active HKG Tmax research project inside the weather-markets
monorepo. It forecasts the contract-authoritative Hong Kong Observatory daily
maximum temperature with point-in-time-safe inputs and calibrated probability
distributions.

## Safe orientation

`AGENTS.md` section 2 is the single mandatory startup sequence. Follow it from
the bounded Git identity proof through the root safety documents, applicable
agent contracts, this file, `README.md`, the code map, current state, and only
then the task-specific contracts/code/tests/config. Do not replace it with a
shortened order from another document.

After that read sequence, confirm `.env` resolves `HKG_TMAX_DATA_ROOT` and
`HKG_TMAX_RUN_ROOT` outside the repository without printing credential values.
The bounded project health command is:

```powershell
.\.venv\Scripts\python.exe -m hkg_tmax doctor
```

`make doctor-fast` is an optional alias on systems where `make` is installed.
Run only the focused tests warranted by the task.

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
