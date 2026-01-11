# Data Contract (Epic #1 → Epic #2)

Epic #2 assumes Epic #1 has populated these MySQL tables:

- `station_registry`
- `asof_policy`
- `cli_daily`
- `mos_asof_feature`

## Required columns

### cli_daily (label)
- station_id (e.g., KMIA)
- target_date_local (DATE)  <-- this is the settlement “day T”
- tmax_f (DECIMAL)          <-- settlement label y(T)
- retrieved_at_utc
- raw_payload_hash

Unique key:
- (station_id, target_date_local)

### mos_asof_feature (features)
One row per:
- station_id
- target_date_local (DATE)  <-- the forecast target day T
- asof_policy_id
- model (GFS/MEX/NAM/NBS/NBE)
Contains:
- asof_utc (TIMESTAMP)
- chosen_runtime_utc (TIMESTAMP)  <-- must be <= asof_utc (no leakage)
- tmax_f (DECIMAL)                <-- MOS daily max feature for day T (as-of T-1)
- missing_reason (nullable)
- retrieved_at_utc

Unique key:
- (station_id, target_date_local, asof_policy_id, model)

## Dataset extraction query shape

Epic #2 training dataset must be “one row per (station_id, target_date_local, asof_policy_id)”.

It is produced by:
1) join cli_daily as label
2) pivot mos_asof_feature by model into columns

Example pivot logic (SQL concept):

SELECT
  f.station_id,
  f.target_date_local,
  f.asof_policy_id,
  MAX(CASE WHEN f.model='GFS' THEN f.tmax_f END) AS gfs_mos_tmax_f,
  MAX(CASE WHEN f.model='MEX' THEN f.tmax_f END) AS mex_tmax_f,
  MAX(CASE WHEN f.model='NAM' THEN f.tmax_f END) AS nam_mos_tmax_f,
  MAX(CASE WHEN f.model='NBS' THEN f.tmax_f END) AS nbs_tmax_f,
  MAX(CASE WHEN f.model='NBE' THEN f.tmax_f END) AS nbe_tmax_f,
  c.tmax_f AS cli_tmax_f
FROM mos_asof_feature f
JOIN cli_daily c
  ON c.station_id=f.station_id AND c.target_date_local=f.target_date_local
WHERE f.asof_policy_id = :asof_policy_id
  AND f.station_id IN (:stations)
  AND f.target_date_local BETWEEN :start AND :end
GROUP BY f.station_id, f.target_date_local, f.asof_policy_id, c.tmax_f;

## Leakage safety checks (MUST RUN)
- For every feature record used:
  chosen_runtime_utc <= asof_utc

Epic #2 will hard-fail training if any row violates this, because it indicates forward-looking leakage.

## Missing data
mos_asof_feature.tmax_f may be NULL with missing_reason.
Epic #2 must implement a configurable strategy:
- drop rows with missing any model feature (default)
- or impute missing with:
  - station climatology
  - or per-model mean

The chosen strategy must be recorded in run metadata.

