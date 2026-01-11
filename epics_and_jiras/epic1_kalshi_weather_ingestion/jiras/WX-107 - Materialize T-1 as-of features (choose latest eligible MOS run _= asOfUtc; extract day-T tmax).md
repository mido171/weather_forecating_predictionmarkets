# WX-107 — Materialize T-1 as-of features (choose latest eligible MOS run <= asOfUtc; extract day-T tmax)

## Objective
Given:
- station
- target date T (local date)
- asOf policy (local time on T-1)

Produce, for each MOS model, a single feature:
- `tmax_f_model_asof(T-1 → T)`

and store it in `mos_asof_feature`, including:
- asOfUtc
- chosen MOS runtimeUtc (<= asOfUtc)
- retrieval timestamps and raw refs

## Inputs
- `station_id` (e.g., KMIA)
- `target_date_local` (DATE)
- `asof_policy_id`
- models = [GFS, MEX, NAM, NBS, NBE]

## Step-by-step algorithm (MUST FOLLOW)
For each day T:
1) Compute `asOfLocal = (T-1) at asof_policy.asof_local_time` in station ZoneId
2) Compute `asOfUtc = asOfLocal.toInstant()`

For each model:
3) Query `mos_run` for (station_id, model) where runtime_utc <= asOfUtc
4) Choose `chosen_runtime_utc = max(runtime_utc)`
5) Load the MOS data for that run (raw payload or parsed values)
6) Extract the daily max temp forecast for target date T:
   - Use MOS variable `n_x` (max/min temp [F]) as defined by IEM.
   - Extract the “max” component for the specific day matching T.
7) Persist:
   - mos_asof_feature row with tmax_f, chosen_runtime_utc, asof_utc
8) If no eligible run exists:
   - persist row with tmax_f=NULL and missing_reason="NO_ELIGIBLE_RUN"

## No-leakage guard (hard requirement)
- If chosen_runtime_utc > asof_utc, abort and log a structured error.

## Acceptance Criteria
- [ ] For each station and each day in a sample range (e.g., 30 days), features are produced for all 5 models (unless missing for a legitimate reason).
- [ ] No row exists with chosen_runtime_utc > asof_utc.
- [ ] Re-running materialization is idempotent (unique key; upsert).
- [ ] Every stored feature includes:
  - asof_utc
  - chosen_runtime_utc
  - retrieved_at_utc
- [ ] A “feature completeness report” is generated after the run:
  - % missing per model per station
  - top missing reasons

## Source references
- MOS variable definitions: `n_x` is max/min temp [F] (IEM MOS download interface).
- MOS archives and models: IEM MOS pages.

