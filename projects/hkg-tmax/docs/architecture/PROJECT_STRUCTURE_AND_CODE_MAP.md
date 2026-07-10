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
| `src/hkg_t24/` | T-24 strategy CLI, DB/features/models/validation | `tests/hkg_t24/`, T-24 focused tests |
| `src/hkg_tmax_db/` | database audit/ingestion CLI | DB migration contract tests |
| `src/hkg_tmax_demo_trading/` | local probability/demo API and store | demo migration/probability tests |
| `tests/` | offline unit/integration/contract suite | `make test-fast`, `make test-full` |
| `config/project/` | target, as-of, evaluation, goals, buckets | validation and settlement tests |
| `config/sources/` | source and station catalogs | source/config tests |
| `config/acquisition/` | acquisition policy and disabled schedules | runtime-safety tests |
| `config/experiments/` | campaign model/probability configs | campaign-specific tests |
| `config/hkg_t24/` | T-24 strategy configuration | T-24 governance tests |
| `db/migrations/postgres/` | ordered database migrations | migration/schema tests |
| `db/schemas/`, `db/sql/` | JSON schemas and canonical SQL | schema contract tests |
| `experiments/campaigns/` | immutable compact research evidence | experiment index/manifest checks |
| `experiments/registry/` | governed ID allocation | experiment tests |
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
