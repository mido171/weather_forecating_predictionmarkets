# Repository Map

## Design rule

Each durable responsibility has one canonical home. Deployable applications may depend on
shared packages; shared packages must not depend on applications. Research projects may use
shared packages but own their city-specific contracts, experiments, and decisions.

```text
weather_data_extraction/
├── apps/                 deployable services
├── packages/             shared Java and Python libraries
├── projects/             city-specific research products
├── tools/                bounded operational and repository utilities
├── tests/smoke/          cross-component offline smoke checks
├── config/examples/      non-secret configuration examples
├── docs/                 repository-wide canonical documentation
├── legacy/               retained non-authoritative history
└── var/                  ignored lightweight runtime state
```

## Ownership

| Path | Owns | Must not own |
|---|---|---|
| `apps/ingestion-service` | Provider clients, ingestion orchestration, service configuration | Shared city research policy |
| `apps/kalshi-market-service` | Kalshi transport, orderbook, account/trading boundaries | Generic forecast research |
| `packages/java/weather-common` | Shared Java time/HTTP primitives | Application startup |
| `packages/java/weather-models` | Shared Java persistence/domain contracts | Generated station model output |
| `packages/python/weather-ml` | Shared offline ML utilities | City experiment ledgers |
| `packages/python/weather-live` | Installable `ml_live` adapters and runtime paths | Trading authorization |
| `projects/hkg-tmax` | HKG domain, research, experiments, probability work | Repository-wide operations |
| `projects/klga-tmax` | KLGA domain, research, experiments | HKG contracts |
| `tools/backfills` | Explicit, bounded, resumable acquisition entrypoints | Reusable domain libraries |
| `tools/live` | Promotion boundary for verified live/operator utilities; currently empty | Incomplete runners or startup behavior |
| `tools/repo` | Read-only repository health checks | Product behavior |
| `legacy` | Historical reference | Current authority |

Reusable logic belongs in a package under `src`, not in an executable script. Scripts parse
arguments, validate budgets, call package code, and report results. Tests import package code,
not arbitrary run scripts.

## Documentation placement

- `docs/architecture`: repository boundaries and decisions.
- `docs/operations`: safe commands, processes, storage, and runbooks.
- `docs/security`: credentials, production, and financial safety.
- `docs/migrations`: structural moves, path maps, verification, and rollback.
- Project `docs`: domain-specific architecture, research, runbooks, and decisions.

Canonical specifications are linked, not copied. Generated indexes should be derived from
manifests rather than edited independently.

## Data boundary

The repository keeps schemas, catalogs, manifests, and small deterministic fixtures. Bulk
raw/bronze/silver/gold data, caches, models, predictions, logs, and run outputs live in an
environment-configured external store. Reparse points/junctions are prohibited beneath the
repository because they create ambiguous ownership and duplicate scans.
