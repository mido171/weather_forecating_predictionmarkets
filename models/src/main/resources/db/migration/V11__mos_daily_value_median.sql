ALTER TABLE mos_daily_value
  ADD COLUMN value_median DECIMAL(10, 4) NULL AFTER value_mean;
