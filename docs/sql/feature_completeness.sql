-- Feature completeness by model for a station/date range.
SET @station_id = 'KMIA';
SET @start_date = '2024-07-01';
SET @end_date = '2024-07-30';
SET @asof_policy_id = 1;

SELECT
  model,
  COUNT(*) AS total_days,
  SUM(CASE WHEN tmax_f IS NULL THEN 1 ELSE 0 END) AS missing_days,
  ROUND(SUM(CASE WHEN tmax_f IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS missing_pct
FROM mos_asof_feature
WHERE station_id = @station_id
  AND asof_policy_id = @asof_policy_id
  AND target_date_local BETWEEN @start_date AND @end_date
GROUP BY model
ORDER BY model;
