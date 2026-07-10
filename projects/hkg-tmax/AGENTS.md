# HKG Tmax agent operating contract

This file is mandatory reading for every agent working anywhere below
`projects/hkg-tmax`. The repository-level `../../AGENTS.md` also applies. If a
deeper `AGENTS.md` exists, follow all non-conflicting rules and use the deepest
file for local details.

## Start every task here

Read, in order:

1. `../../AGENTS.md`
2. `START_HERE.md`
3. `docs/status/CURRENT_STATE.md`
4. `docs/INDEX.md`
5. the relevant contract under `docs/contracts/` or
   `docs/specifications/hkg-t24/`
6. the relevant experiment README decision and `EXPERIMENT_INDEX.md`

Then prove repository identity without a deep scan:

```powershell
git rev-parse --show-toplevel
git branch --show-current
git remote -v
git status --short --untracked-files=no
```

Do not run a collector, backfill, scheduler, server, database migration,
container stack, model training job, or market request during orientation.

## Source-of-truth hierarchy

When facts conflict, use this order:

1. exact market settlement rules and archived source payloads;
2. point-in-time database records and immutable manifests;
3. governed configuration in `config/`;
4. source and tests in `src/` and `tests/`;
5. current status and experiment conclusions;
6. historical reports and archived documents.

Historical paths and prose are evidence, not current instructions. Preserve
their provenance; do not silently rewrite them to look current.

## Route work to the correct home

- production/library code: `src/`
- tests: `tests/`
- runnable utilities: `scripts/`
- governed configuration: `config/{project,sources,acquisition,experiments}`
- database assets: `db/{migrations,schemas,sql}`
- research protocols/results summaries: `experiments/campaigns/`
- specifications and runbooks: `docs/`
- work packages: `planning/`
- UI: `apps/polymarket-backtester/`
- raw/derived data: `${HKG_TMAX_DATA_ROOT}`
- logs/models/predictions/run artifacts: `${HKG_TMAX_RUN_ROOT}`

Never create a second project root, nested Git repository, junction, or copy of
the source tree inside an experiment or handoff.

## Data and experiment invariants

- The settlement target and as-of cutoff are contracts, not assumptions.
- A feature is eligible only when its real publication/availability timestamp
  is at or before the forecast cutoff.
- Preserve raw bytes, content hashes, source metadata, and legacy absolute
  provenance. New records must also contain relocation-safe relative paths.
- Never overwrite a completed experiment. Create a new governed ID.
- Record negative, null, blocked, and rejected outcomes.
- Large outputs stay external; Git receives one compact README dossier per
  campaign/experiment plus machine-readable manifests, metrics, and audits.
- Do not use market prices as meteorological features unless an explicit
  research contract authorizes it. This repository does not place orders.

## Resource and network safety

- Default workers, BLAS threads, model threads, and subprocess fan-out: `1`.
- Maximum without explicit user approval: `2`.
- Never use `n_jobs=-1`, `pytest -n auto`, Maven `-T`, unbounded process pools,
  or recursive compilation from the monorepo root.
- Never recursively scan `${HKG_TMAX_DATA_ROOT}`, Parquet payloads, raw object
  stores, `.git`, virtualenvs, or experiment artifacts during startup.
- Network commands require an explicit `--execute`, a narrow provider/source
  scope, and hard request/runtime/byte budgets.
- Collectors are globally disabled in
  `config/acquisition/collector_schedules.yaml`; enabling individual sources is
  insufficient. Scheduler installation is dry-run and disabled by default.
- Respect provider spacing and `Retry-After`. Stop on authentication errors,
  repeated rate limits, or budget exhaustion.

## Process and Git safety

- Background work needs a run ID, exact command, PID, start time, owner, budget,
  log path, and stop command in the external run ledger.
- Stop only verified owned PIDs, children first. Never kill all Python, Java,
  Node, Docker, or browser processes.
- Never use `git reset --hard`, `git clean`, bulk checkout/revert, broad
  `git add .`, or history rewriting without explicit authorization.
- Stage explicit paths. Before commit, inspect staged names and staged diff.
- Preserve unrelated user work and verify the actual Git root/remote before any
  commit or push.

## Verification ladder

Use the cheapest sufficient checks first:

1. `make doctor-fast`
2. focused Ruff/type checks for changed modules
3. focused tests for changed behavior
4. `make test-fast`
5. `make validate`
6. `make test-full` only for release/cutover or when broad impact justifies it

All provider tests must use mocked transports unless the user explicitly asks
for a bounded live probe. Report exactly what passed, failed, timed out, or was
not run.

## Definition of done

A change is complete only when behavior, focused tests, relevant contracts,
path references, and operator documentation agree. Update:

- `CHANGELOG.md` for material behavior;
- `docs/architecture/PROJECT_STRUCTURE_AND_CODE_MAP.md` for layout changes;
- the experiment conclusion/index for research work;
- `docs/status/CURRENT_STATE.md` only for verified current-state changes.

Do not claim production, leakage safety, reproducibility, or live readiness from
code inspection alone.
