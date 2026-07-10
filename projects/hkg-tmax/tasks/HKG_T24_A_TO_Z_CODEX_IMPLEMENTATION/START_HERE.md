# HKG T+24 Tmax — A-to-Z Exact-Vintage Data, MOS, Router, Specialist, and Production Implementation Package

## Mission

Implement, validate, and document the complete leakage-safe system for forecasting Hong Kong Observatory daily maximum temperature for target date **T**, with the operational decision frozen at **15:00 HKT on T−1** unless the repository already contains a different explicit contract. The final point forecast must be derived only from information provably available at that cutoff.

This package is deliberately not a loose research suggestion. It is a sequenced set of **40 executable Codex tasks**, plus database contracts, model/source dispositions, feature requirements, router and specialist specifications, validation rules, and evidence files. Completing the tasks creates the full system from data acquisition through sealed confirmation and live operation.

## Critical correction incorporated

The current database contains a near-continuous official HKO forecast archive in `public.hko_historical_forecasts_2000_2026`. The supplied direct database facts are:

- clean filter: `row_quality_status = 'usable_local_minmax'`;
- 115,795 usable local min/max rows;
- issue span reported as 2000-01-01 16:22 through 2026-06-20 23:45;
- target-date span 2000-01-02 through 2026-06-21;
- 9,667 distinct target dates;
- one missing target date, 2003-02-02;
- 324,179 raw rows across local, 5-day, 7-day, and 9-day products.

These facts **supersede all earlier assumptions that the official forecast history is sparse or contains a decade-long gap**. Task T00 must verify the facts directly in Postgres before they become canonical. The strategy treats the official HKO forecast as the central long-history anchor, including every eligible pre-cutoff vintage and its revision path.

## Repository paths

```text
Repository: C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex
Datasets:   C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets
Experiments:C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\experiments
```

## Mandatory read order

1. `START_HERE.md`
2. `CODEX_GLOBAL_EXECUTION_CONTRACT.md`
3. `T24_POINT_IN_TIME_CONSTITUTION.md`
4. `CURRENT_DB_CANONICAL_FACTS.md`
5. `MASTER_EXECUTION_DAG.md`
6. `design/GRIBSTREAM_ACQUISITION_SPECIFICATION.md`
7. `design/DATABASE_SCHEMA_AND_LINEAGE.md`
8. `design/FEATURE_ENGINEERING_CATALOG.md`
9. `design/EXPERT_MODEL_CATALOG.md`
10. `design/ROUTER_TRAINING_SPECIFICATION.md`
11. `design/SPECIALIST_TRAINING_SPECIFICATION.md`
12. `design/VALIDATION_PROTOCOL.md`
13. Execute tasks in `TASK_INDEX.csv` dependency order.

## Non-negotiable outcome

The implementation is not complete merely because code exists. It is complete only when:

- acquisition has actually run for all backfillable core models;
- prospective collectors are deployed for short-retention sources;
- all raw payloads and normalized rows are lineage-traceable;
- exact-vintage eligibility is enforced in database views and code;
- every expert has genuine out-of-fold predictions;
- router weights are trained from out-of-fold expected-error estimates;
- specialists have learned regime, correction, and benefit gates;
- the integrated point forecast beats the proper baselines on the declared frames;
- 2024, 2025, and 2026 evidence is opened only under the sealing protocol;
- every result is reproducible from immutable manifests.

No task may claim 0.45°C MAE unless an untouched, declared evaluation frame proves it.
