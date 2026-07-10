ALTER TABLE mos_daily_value
  ADD COLUMN asof_utc TIMESTAMP NULL AFTER model;

CREATE INDEX idx_mos_daily_asof
  ON mos_daily_value (asof_utc);
