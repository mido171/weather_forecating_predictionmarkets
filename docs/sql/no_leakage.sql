-- No-leakage audit: find any MOS feature rows that used a runtime after as-of.
SELECT
  station_id,
  target_date_local,
  asof_policy_id,
  model,
  asof_utc,
  chosen_runtime_utc
FROM mos_asof_feature
WHERE chosen_runtime_utc IS NOT NULL
  AND chosen_runtime_utc > asof_utc
ORDER BY chosen_runtime_utc DESC
LIMIT 100;
