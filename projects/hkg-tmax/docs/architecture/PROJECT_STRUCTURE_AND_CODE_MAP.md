# Project structure and code map

Current as of the 2026-07-10 consolidation. Historical mappings are preserved
in `docs/archive/HISTORICAL_PROJECT_STRUCTURE_AND_CODE_MAP.md`.

## Runtime flow

```text
config + external raw archive
          |
          v
src/hkg_tmax acquisition, contracts, normalization, feature/model logic
          |
          +--> external bronze/silver/gold + manifests
          +--> experiments/campaigns compact evidence
          +--> external reports/models/predictions/logs
          |
          v
src/hkg_tmax_demo_trading + apps/polymarket-backtester (local demo only)
```

## Ownership map

| Path | Purpose | Principal verification |
|---|---|---|
| `src/hkg_tmax/` | core acquisition, contracts, storage, research/model code | `tests/test_*.py` |
| `src/hkg_tmax/experiment_registry.py` | governed campaign/ID/path/scaffold contracts and atomic registry primitives | experiment registry and corruption tests |
| `src/hkg_tmax/experiment_transaction.py` | locked allocation, staging, journal, rollback, and interrupted-create recovery | experiment failure-injection and recovery tests |
| `src/hkg_tmax/experiment_index.py`, `experiments.py` | bounded index/status projection plus backward-compatible public façade | index, façade, and reparse-path tests |
| `src/hkg_t24/` | T-24 strategy CLI, DB/features/models/validation | `tests/hkg_t24/`, T-24 focused tests |
| `src/hkg_tmax/hkg_t24/` | legacy locked-test guard/governance/peak/moisture support only | legacy HKG T-24 focused tests |
| `src/hkg_tmax_probability/` | bucket, PMF, distribution, calibration, scoring, and probability inference | probability V1/V2 focused tests |
| `src/hkg_tmax_db/` | database audit/ingestion CLI | DB migration contract tests |
| `src/hkg_tmax_demo_trading/` | local probability/demo API and store | demo migration/probability tests |
| `pyproject.toml` | Python dependencies, entry points, version floor, and tool policy | install, Ruff, typing, and focused test gates |
| `.env.example` | placeholder-only environment key contract and safe defaults | doctor/path/security checks; never real secrets |
| `apps/polymarket-backtester/package.json`, `apps/polymarket-backtester/vite.config.js` | frontend dependencies, scripts, and Vite build configuration | package build plus browser QA |
| `Makefile` | optional bounded aliases for project checks and governed commands | compare each alias with canonical README commands |
| `Dockerfile`, `compose.yaml`, `.dockerignore` | local container build, topology, limits, and build context | config validation and explicit container smoke tests |
| `.gitignore`, `.pre-commit-config.yaml` | project runtime exclusions and scoped hook policy | repository doctor and bootstrap safety tests |
| `SECURITY.md`, `docs/security/` | project security entry point and detailed controls | secret scan and production-safety review |
| `tests/` | offline unit/integration/contract suite | `make test-fast`, `make test-full` |
| `config/project/` | target, as-of, evaluation, goals, buckets | validation and settlement tests |
| `config/sources/` | source and station catalogs | source/config tests |
| `config/acquisition/` | acquisition policy and disabled schedules | runtime-safety tests |
| `config/experiments/` | campaign model/probability configs | campaign-specific tests |
| `config/hkg_t24/` | T-24 strategy configuration | T-24 governance tests |
| `db/migrations/postgres/` | ordered database migrations | migration/schema tests |
| `db/schemas/`, `db/sql/` | JSON schemas and canonical SQL | schema contract tests |
| `experiments/campaigns/` | one README dossier per campaign/experiment plus machine evidence and a retired-doc hash ledger | campaign documentation layout and provenance checks |
| `experiments/registry/` | governed global ID allocation and campaign-relative directory ledger | campaign-aware experiment/registry tests |
| `experiments/templates/standard/` | mandatory new-experiment contract | repository validation |
| `scripts/` | thin bounded entry points | same-name focused tests where available |
| `planning/tasks/` | implementation task packages | task status indexes |
| `planning/work-packages/` | bounded handoff/work specifications | package-local manifests |
| `docs/` | current docs plus explicit archive/evidence zones | link/path checks |

## Relocation and storage

`src/hkg_tmax/paths.py` is the canonical root resolver. Production code must not
derive the repository from a fixed number of parent directories. Runtime data
and generated runs resolve from environment-configured external roots. Archive
records carry a storage-root ID and root-relative path while retaining legacy
absolute provenance for audit.

## Change rule

Any move or new top-level responsibility updates this file, `docs/INDEX.md`,
focused path tests, and the migration ledger in the same change.
