# KLGA Tmax agent operating contract

This file applies to `projects/klga-tmax`. Read the monorepo-level
`../../AGENTS.md` first; both contracts apply.

## Start every task

Read, in order:

1. `../../AGENTS.md`
2. `START_HERE.md`
3. `docs/status/CURRENT_STATE.md`
4. `docs/INDEX.md`
5. `docs/specifications/strategy/KLGA_TMAX_TRADING_STRATEGY_SPEC.md`
6. the supplemental strategy patches in the same directory
7. `docs/context/KLGA_TMAX_TASK_IMPLEMENTATION_QUEUE.md`
8. the exact numbered task under `docs/specifications/data-acquisition/`

For an implemented task, read its context/deep-dive before changing it. Prove
Git root, branch, remote, and tracked-only status. Orientation is offline: do
not connect to a provider/database, run a backfill, migrate a schema, train a
model, start a service, or place an order.

## Canonical ownership

- source: `src/klga_tmax/`
- tests: `tests/`
- Alembic migrations: `alembic/versions/`
- configuration: `config/`
- bounded entry points: `scripts/`
- compact research records: `experiments/`
- strategy/task/context documentation: `docs/`
- generated data/logs/models/reports: `${KLGA_RUN_ROOT}`

Do not create source under an `implementation/` wrapper, a nested Git root, or
another bootstrap directory.

## Data, leakage, and database contracts

- A feature is eligible only if the system could have known it at the forecast
  cutoff. Preserve issue, valid, provider availability, ingestion, revision,
  request identity, record identity, and raw-payload hash semantics.
- `registry.station_registry` is the canonical station universe. Do not copy
  station lists or provider IDs into provider modules.
- `KLGA_DB_URL` is the only application DSN. Never hardcode or document a real
  username/password URL.
- Prefer additive Alembic migrations, parameterized SQL, explicit constraints,
  and reversible notes. Never delete user data without explicit authorization.
- This project is research/backtest by default. Live/prod mode is fail-closed
  and requires the exact reviewed acknowledgement; do not place orders.

## Provider and resource safety

- Every provider command requires `--execute` before credentials, DB writes, or
  network clients are initialized.
- Require explicit dates and provider/model/station scope plus request/chunk,
  byte, runtime, retry, and worker budgets.
- Default workers and numerical-library threads to one; maximum two without
  explicit user approval and a reviewed code change.
- GribStream calls are one-threaded, spaced at least 12 seconds, and stop on
  authentication failure or rate limiting.
- Wunderground defaults to one worker, short chunks, bounded retries, and no
  more than a 31-day command window.
- A dry run performs zero provider calls. Do not call provider APIs from tests.

## Task workflow

1. Read the governing task and existing implementation.
2. Identify target, as-of, persistence, and external-cost impact.
3. Implement a narrow coherent slice in `src/` and `tests/`.
4. Add an Alembic migration when persistence changes.
5. Add CLI/validation surfaces only with safe defaults and audit records.
6. Update the task deep dive and queue status from verified evidence.
7. Run focused syntax, Ruff, and pytest checks; run DB validation only when the
   user authorized it and a safe local DSN is configured.

## Process and Git safety

- Background work needs an external run ledger entry with exact command, PID,
  owner, scope, budgets, logs, and stop command.
- Stop only verified owned PIDs, children first; never kill all Python/Java.
- Never use destructive/broad Git cleanup, broad staging, or history rewriting.
- Preserve unrelated work and inspect staged paths/diff before commit.
- Do not recursively scan data, artifacts, virtualenvs, `.git`, or Parquet
  stores during startup.

## Verification

From this directory, prefer:

```powershell
python -m compileall -q src tests
python -m pytest -q <focused-test-path>
python -m klga_tmax.cli --help
```

Run the full offline suite only for release/cutover or broad changes. Report
exactly which checks were run and never claim DB/provider validation from mocks.
