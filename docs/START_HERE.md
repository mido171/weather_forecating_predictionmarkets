# Start Here

This is the short orientation document for the whole repository. It routes work to the
correct owner without requiring agents to read the entire historical corpus.

## Read order

1. Root `AGENTS.md`.
2. `docs/architecture/REPOSITORY_MAP.md`.
3. `docs/operations/SAFE_COMMANDS.md`.
4. The closest scoped `AGENTS.md`.
5. The relevant project/component `START_HERE.md`, `README.md`, and current-state document.
6. The exact task specification and only the canonical references it names.

Do not begin by reading every experiment, report, milestone, handoff, or archive. Those are
routed evidence, not startup material.

## Choose the owner

| Work | Owner |
|---|---|
| Java ingestion/provider work | `apps/ingestion-service` |
| Kalshi API, orderbook, or trading integration | `apps/kalshi-market-service` |
| Shared Java contracts/entities | `packages/java` |
| Shared Python ML/live utilities | `packages/python` |
| Hong Kong Tmax research | `projects/hkg-tmax` |
| KLGA Tmax research | `projects/klga-tmax` |
| Bounded backfills or discovery utilities | `tools` |
| Repository safety and structure | `tools/repo`, root `docs` |

## Source authority

The authority chain is root `AGENTS.md`, closest scoped `AGENTS.md`, canonical specifications,
then verified code/tests/schemas/live state. Accepted experiment conclusions are evidence.
Archives, copied contracts, generated reports, handoffs, and `legacy/` are non-authoritative
unless a canonical document explicitly promotes them.

## Safe first checks

```powershell
git rev-parse --show-toplevel
git branch --show-current
git remote -v
git -c core.fsmonitor=false status --short --branch --untracked-files=no
python tools/repo/doctor.py
```

These checks are local and read-only. Startup must not fetch, install, start services, contact
providers, run backfills, mutate databases, install schedulers, or enable trading.

## Canonical root documents

- Repository ownership: `docs/architecture/REPOSITORY_MAP.md`
- Safe commands and verification: `docs/operations/SAFE_COMMANDS.md`
- Local virtual environments: `docs/operations/ENVIRONMENT_SETUP.md`
- Runtime/resource ownership: `docs/operations/RUNTIME_SAFETY.md`
- Secrets and production safety: `docs/security/SECURITY_BASELINE.md`
- Reorganization record: `docs/migrations/2026-07-10-repository-restructure.md`
