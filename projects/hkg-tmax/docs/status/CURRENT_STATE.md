# Current state

Last structural verification: 2026-07-10.

## Canonical architecture

- HKG is a component at `projects/hkg-tmax` in one standalone monorepo.
- Python packages and tests use `src/` and `tests/`.
- Configuration is grouped under `config/project`, `config/sources`,
  `config/acquisition`, and `config/experiments`.
- Database assets use `db/migrations`, `db/schemas`, and `db/sql`.
- Documentation has one canonical `docs/` tree.
- Experiments are grouped by campaign with a registry and standard template.
- Task packages use `planning/`.

## Runtime safety

- Data resolves through `HKG_TMAX_DATA_ROOT`; generated runs resolve through
  `HKG_TMAX_RUN_ROOT`.
- New archive records preserve legacy absolute provenance and add portable
  root-relative fields.
- Collectors and schedules are disabled by default. Network acquisition needs
  `--execute` and explicit source/request budgets.
- Default worker/model/BLAS concurrency is one; local Codex concurrency is two.
- Docker database ports bind to loopback and passwords come from `.env`.

## Evidence boundary

Historical research evidence and milestones were preserved under `docs/archive`
and `docs/evidence`. Their scientific claims were not revalidated by the
repository-structure migration. Consult experiment conclusions and live data
manifests before treating a result as current.
