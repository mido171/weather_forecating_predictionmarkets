CREATE TABLE IF NOT EXISTS model_catalog (
  model_code TEXT PRIMARY KEY,
  family TEXT NOT NULL,
  role TEXT NOT NULL,
  archive_start DATE NOT NULL,
  snapshot_var_name TEXT NOT NULL,
  snapshot_var_level TEXT NOT NULL,
  snapshot_var_info TEXT,
  native_tmax_var_name TEXT,
  native_tmax_var_level TEXT,
  native_tmax_var_info TEXT,
  native_tmax_available_from DATE,
  ensemble_members_json TEXT,
  enabled_backtest INTEGER NOT NULL,
  enabled_live INTEGER NOT NULL,
  notes TEXT,
  created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nws_daily_settlements (
  station_id TEXT NOT NULL,
  settlement_date_local DATE NOT NULL,
  timezone TEXT NOT NULL,
  local_day_start_utc TEXT NOT NULL,
  local_day_end_utc TEXT NOT NULL,
  actual_tmax_native REAL,
  actual_tmax_native_unit TEXT,
  actual_tmax_f REAL NOT NULL,
  source TEXT NOT NULL,
  ingested_at_utc TEXT NOT NULL,
  PRIMARY KEY (station_id, settlement_date_local)
);

CREATE TABLE IF NOT EXISTS gribstream_requests (
  request_id TEXT PRIMARY KEY,
  model_code TEXT NOT NULL,
  station_id TEXT NOT NULL,
  settlement_date_local DATE NOT NULL,
  endpoint TEXT NOT NULL,
  as_of_utc TEXT NOT NULL,
  from_time_utc TEXT NOT NULL,
  until_time_utc TEXT NOT NULL,
  http_status INTEGER,
  attempts INTEGER NOT NULL,
  success INTEGER NOT NULL,
  row_count INTEGER NOT NULL,
  error_text TEXT,
  started_at_utc TEXT NOT NULL,
  finished_at_utc TEXT,
  response_format TEXT NOT NULL,
  response_compressed INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS gribstream_raw_forecasts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id TEXT NOT NULL,
  model_code TEXT NOT NULL,
  station_id TEXT NOT NULL,
  settlement_date_local DATE NOT NULL,
  as_of_utc TEXT NOT NULL,
  forecasted_at_utc TEXT NOT NULL,
  forecasted_time_utc TEXT NOT NULL,
  forecasted_time_local TEXT NOT NULL,
  forecasted_date_local DATE NOT NULL,
  lat REAL NOT NULL,
  lon REAL NOT NULL,
  coord_name TEXT,
  variable_name TEXT NOT NULL,
  variable_level TEXT NOT NULL,
  variable_info TEXT,
  member INTEGER,
  value_native REAL NOT NULL,
  unit_native TEXT NOT NULL,
  value_f REAL,
  lead_minutes INTEGER,
  inserted_at_utc TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_gribstream_raw_forecasts_natural
  ON gribstream_raw_forecasts (
    model_code,
    station_id,
    settlement_date_local,
    as_of_utc,
    forecasted_at_utc,
    forecasted_time_utc,
    variable_name,
    variable_level,
    IFNULL(variable_info, ''),
    IFNULL(member, -1)
  );

CREATE TABLE IF NOT EXISTS daily_model_tmax (
  station_id TEXT NOT NULL,
  settlement_date_local DATE NOT NULL,
  model_code TEXT NOT NULL,
  family TEXT NOT NULL,
  as_of_utc TEXT NOT NULL,
  local_day_start_utc TEXT NOT NULL,
  local_day_end_utc TEXT NOT NULL,
  native_tmax_f REAL,
  snapshot_tmax_f REAL,
  interpolated_tmax_f REAL,
  selected_raw_tmax_f REAL,
  selected_method TEXT NOT NULL,
  snapshot_row_count INTEGER NOT NULL,
  native_row_count INTEGER NOT NULL,
  model_available INTEGER NOT NULL,
  notes TEXT,
  created_at_utc TEXT NOT NULL,
  PRIMARY KEY (station_id, settlement_date_local, model_code)
);

CREATE TABLE IF NOT EXISTS model_daily_errors (
  station_id TEXT NOT NULL,
  settlement_date_local DATE NOT NULL,
  model_code TEXT NOT NULL,
  selected_raw_tmax_f REAL NOT NULL,
  actual_tmax_f REAL NOT NULL,
  error_f REAL NOT NULL,
  abs_error_f REAL NOT NULL,
  squared_error_f REAL NOT NULL,
  created_at_utc TEXT NOT NULL,
  PRIMARY KEY (station_id, settlement_date_local, model_code)
);

CREATE TABLE IF NOT EXISTS daily_model_weights (
  station_id TEXT NOT NULL,
  settlement_date_local DATE NOT NULL,
  model_code TEXT NOT NULL,
  family TEXT NOT NULL,
  train_start_date DATE,
  train_end_date DATE,
  train_n_days INTEGER NOT NULL,
  ew_bias_f REAL,
  ew_mae_f REAL,
  ew_rmse_f REAL,
  bias_corrected_tmax_f REAL,
  raw_weight REAL,
  model_cap_applied INTEGER NOT NULL,
  family_cap_applied INTEGER NOT NULL,
  final_weight REAL,
  included_in_blend INTEGER NOT NULL,
  exclusion_reason TEXT,
  created_at_utc TEXT NOT NULL,
  PRIMARY KEY (station_id, settlement_date_local, model_code)
);

CREATE TABLE IF NOT EXISTS daily_prediction_components (
  station_id TEXT NOT NULL,
  settlement_date_local DATE NOT NULL,
  model_code TEXT NOT NULL,
  family TEXT NOT NULL,
  selected_raw_tmax_f REAL,
  bias_corrected_tmax_f REAL,
  final_weight REAL,
  weighted_contribution_f REAL,
  created_at_utc TEXT NOT NULL,
  PRIMARY KEY (station_id, settlement_date_local, model_code)
);

CREATE TABLE IF NOT EXISTS daily_predictions (
  station_id TEXT NOT NULL,
  settlement_date_local DATE NOT NULL,
  as_of_utc TEXT NOT NULL,
  actual_tmax_f REAL NOT NULL,
  equal_weight_blend_f REAL,
  inverse_rmse_blend_f REAL,
  family_capped_blend_f REAL,
  nbm_only_f REAL,
  hrrr_only_f REAL,
  rap_only_f REAL,
  gfs_only_f REAL,
  best_single_model_code TEXT,
  best_single_model_pred_f REAL,
  family_capped_error_f REAL,
  family_capped_abs_error_f REAL,
  created_at_utc TEXT NOT NULL,
  PRIMARY KEY (station_id, settlement_date_local)
);

CREATE TABLE IF NOT EXISTS metrics_summary (
  metric_scope TEXT NOT NULL,
  metric_name TEXT NOT NULL,
  evaluation_start DATE NOT NULL,
  evaluation_end DATE NOT NULL,
  n_days INTEGER NOT NULL,
  mae_f REAL,
  rmse_f REAL,
  bias_f REAL,
  median_abs_error_f REAL,
  within_0_5f REAL,
  within_1f REAL,
  within_2f REAL,
  created_at_utc TEXT NOT NULL,
  PRIMARY KEY (metric_scope, metric_name, evaluation_start, evaluation_end)
);

CREATE TABLE IF NOT EXISTS coverage_summary (
  model_code TEXT PRIMARY KEY,
  role TEXT NOT NULL,
  archive_start DATE NOT NULL,
  first_date_fetched DATE,
  last_date_fetched DATE,
  fetched_day_count INTEGER NOT NULL,
  scored_day_count INTEGER NOT NULL,
  notes TEXT,
  created_at_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nws_daily_settlements_station_date
  ON nws_daily_settlements (station_id, settlement_date_local);

CREATE INDEX IF NOT EXISTS idx_gribstream_requests_model_date
  ON gribstream_requests (model_code, settlement_date_local);

CREATE INDEX IF NOT EXISTS idx_gribstream_requests_station_date
  ON gribstream_requests (station_id, settlement_date_local);

CREATE INDEX IF NOT EXISTS idx_gribstream_raw_forecasts_station_date_model
  ON gribstream_raw_forecasts (station_id, settlement_date_local, model_code);

CREATE INDEX IF NOT EXISTS idx_gribstream_raw_forecasts_model_var
  ON gribstream_raw_forecasts (model_code, variable_name, settlement_date_local);

CREATE INDEX IF NOT EXISTS idx_daily_model_tmax_station_date
  ON daily_model_tmax (station_id, settlement_date_local);

CREATE INDEX IF NOT EXISTS idx_model_daily_errors_station_date
  ON model_daily_errors (station_id, settlement_date_local);

CREATE INDEX IF NOT EXISTS idx_daily_model_weights_station_date
  ON daily_model_weights (station_id, settlement_date_local);

CREATE INDEX IF NOT EXISTS idx_daily_predictions_station_date
  ON daily_predictions (station_id, settlement_date_local);
