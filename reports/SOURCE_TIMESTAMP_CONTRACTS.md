# Source Timestamp Contracts

Primary operational target: forecast Hong Kong Observatory daily maximum temperature for target date T.

Primary cutoff: `T-1 15:00 HKT`. A feature is usable only if the feature value and all source information needed to compute it are available no later than this cutoff.

| Source family | Source time | Issue time | Valid time | Available-at contract | T-24 role |
|---|---|---|---|---|---|
| HKO official daily Tmax | Local calendar day T | Publication after observation day | Day T | after target day completion/publication | TARGET_ONLY |
| HKO other official daily climate | Local calendar day T | Publication after observation day | Day T | after target day completion/publication | RETROSPECTIVE_MECHANISM_ONLY for same-day target analysis |
| HKO high-frequency station observations | Observation timestamp in HKT | Same feed publication event | Observation instant | observed_at + 20 minutes unless live `retrieved_at` proves earlier/later | OPERATIONAL_WITH_CONSERVATIVE_LATENCY |
| HKO since-midnight max/min | Observation timestamp in HKT | Same feed publication event | Partial day up to observation time | observed_at + 20 minutes; only T-1 or earlier is allowed at this cutoff | OPERATIONAL_WITH_CONSERVATIVE_LATENCY |
| HKO forecast/warning JSON/RSS/image current feeds | Provider issue timestamp where present, otherwise retrieval timestamp | Feed issue/retrieval | Forecast/warning validity fields | retrieved_at or parsed issue availability | OPERATIONAL_POINT_IN_TIME for archived vintages only |
| Current-only NWP subsets | Model cycle time | Model cycle issuance | Forecast lead valid time | cycle-specific release lag must be proven | PROSPECTIVE_ONLY_NOT_YET_BACKTESTABLE until historical cycles exist |
| Reanalysis/final gridded products | Analysis valid time | Final product release | Historical valid time | release lag after valid time | RETROSPECTIVE_MECHANISM_ONLY unless release lag makes it eligible |
| Static geospatial context | Static | Static | Static | always available after station metadata freeze | STATIC_DETERMINISTIC |
