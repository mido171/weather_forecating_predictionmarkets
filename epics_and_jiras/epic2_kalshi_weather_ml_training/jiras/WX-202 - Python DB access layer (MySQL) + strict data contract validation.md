# WX-202 — Python DB access layer (MySQL) + strict data contract validation

## Objective
Implement the Python DB access layer that reads:
- CLI labels from `cli_daily`
- MOS as-of features from `mos_asof_feature`
for any station/date range/as-of policy.

## Requirements
- Implement connection handling via SQLAlchemy engine (or equivalent).
- Implement a dataset extraction function:
  build_dataset(stations, start_date, end_date, asof_policy_id) -> DataFrame
- MUST validate:
  - required columns exist
  - station_id and date range are respected
  - asof_policy_id filter applied

## Mandatory leakage check (hard fail)
For every record used:
- chosen_runtime_utc <= asof_utc
If any row violates this, abort with detailed diagnostics.

## Output schema (minimum columns)
- station_id
- target_date_local
- asof_policy_id
- gfs_mos_tmax_f
- mex_tmax_f
- nam_mos_tmax_f
- nbs_tmax_f
- nbe_tmax_f
- cli_tmax_f (label)

## Acceptance Criteria
- [ ] Works for at least one station (KMIA) and one full month.
- [ ] Pivot from long (rows per model) to wide columns per model is correct.
- [ ] Leakage check is implemented and tested.
- [ ] Missing model rows result in NULL in corresponding columns (handled later by strategy).
- [ ] Logs row counts and missing counts by model.

