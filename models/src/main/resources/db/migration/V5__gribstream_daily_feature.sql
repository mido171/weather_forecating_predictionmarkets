CREATE TABLE gribstream_daily_feature (
  id BIGINT NOT NULL AUTO_INCREMENT,
  station_id VARCHAR(32) NOT NULL,
  zone_id VARCHAR(64) NOT NULL,
  target_date_local DATE NOT NULL,
  asof_utc TIMESTAMP NOT NULL,
  model_code VARCHAR(32) NOT NULL,
  metric VARCHAR(64) NOT NULL,
  value_f DOUBLE NULL,
  value_k DOUBLE NULL,
  source_forecasted_at_utc TIMESTAMP NULL,
  window_start_utc TIMESTAMP NOT NULL,
  window_end_utc TIMESTAMP NOT NULL,
  min_horizon_hours INT NOT NULL,
  max_horizon_hours INT NOT NULL,
  request_json TEXT NOT NULL,
  request_sha256 CHAR(64) NOT NULL,
  response_sha256 CHAR(64) NOT NULL,
  retrieved_at_utc TIMESTAMP NOT NULL,
  notes VARCHAR(512) NULL,
  PRIMARY KEY (id),
  UNIQUE (station_id, target_date_local, asof_utc, model_code, metric)
);

CREATE INDEX idx_gribstream_station_target
  ON gribstream_daily_feature (station_id, target_date_local);

CREATE INDEX idx_gribstream_model_target
  ON gribstream_daily_feature (model_code, target_date_local);

CREATE INDEX idx_gribstream_asof
  ON gribstream_daily_feature (asof_utc);
