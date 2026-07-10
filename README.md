# Weather Markets Workspace

This repository is the single source-controlled workspace for weather data acquisition,
forecast research, and weather-market integrations. Deployable services, shared packages,
city research projects, operational tools, and repository governance are separated at the
top level so that each kind of work has one obvious home.

## Start here

1. Read [`AGENTS.md`](AGENTS.md).
2. Read [`docs/START_HERE.md`](docs/START_HERE.md).
3. Confirm the Git root, branch, and remote before making changes.
4. Read the closest scoped `AGENTS.md` and project/component start document.
5. Run only the focused, offline checks relevant to the requested work.

Repository health can be checked without starting services or contacting providers:

```powershell
python tools/repo/doctor.py
python -m unittest discover -s tools/repo/tests -p "test_*.py"
```

Use `python tools/repo/doctor.py --strict` for the CI gate. The doctor is read-only. It does
not fetch, install, start, stop, migrate, trade, backfill, or modify Git state.

## Repository map

| Path | Responsibility |
|---|---|
| `apps/` | Deployable ingestion and market services |
| `packages/` | Shared Java and Python libraries |
| `projects/hkg-tmax/` | Hong Kong Tmax research product |
| `projects/klga-tmax/` | KLGA Tmax research product |
| `tools/` | Bounded operational, backfill, discovery, and repository utilities |
| `tests/smoke/` | Cross-component offline smoke checks |
| `config/examples/` | Non-secret configuration examples |
| `docs/` | Canonical repository architecture, operations, security, and migration docs |
| `legacy/` | Retained historical code that is not an active authority |
| `var/` | Ignored local runtime state only |

The detailed ownership map is in
[`docs/architecture/REPOSITORY_MAP.md`](docs/architecture/REPOSITORY_MAP.md).

## Data and runtime state

Large datasets, caches, models, predictions, logs, exports, and run artifacts do not belong
in Git. HKG data is configured through `HKG_TMAX_DATA_ROOT`; the established local data root
is `C:\hkg_tmax_data`. The repository keeps only small fixtures, schemas, catalogs, and
manifests needed to understand and reproduce work.

Local process metadata belongs in ignored `var/run/`. Every background process must have a
PID, exact command, start time, owner, log path, and stop command recorded there.

## Safe verification

Verification is progressive:

1. Repository doctor and focused syntax/static checks.
2. Focused unit tests for changed behavior.
3. Relevant integration or contract tests with external effects disabled.
4. Broader project checks only when the change warrants them.
5. Live provider, database, scheduler, service, or trading checks only with explicit scope,
   budgets, credentials, and authorization.

Do not use root-wide compilation, test discovery, hashing, recursive listing, or automatic
parallelism as a startup check. See
[`docs/operations/SAFE_COMMANDS.md`](docs/operations/SAFE_COMMANDS.md).

## Security

Secrets are environment-provided and must never be committed, printed, copied into reports,
or embedded in URLs. Live ingestion, backfills, schedulers, and trading must be fail-closed.
See [`docs/security/SECURITY_BASELINE.md`](docs/security/SECURITY_BASELINE.md).
