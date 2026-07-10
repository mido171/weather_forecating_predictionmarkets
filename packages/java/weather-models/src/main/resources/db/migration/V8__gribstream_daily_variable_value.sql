CREATE TABLE gribstream_daily_variable_value (
  id BIGINT NOT NULL AUTO_INCREMENT,
  station_id VARCHAR(32) NOT NULL,
  zone_id VARCHAR(64) NOT NULL,
  model_code VARCHAR(32) NOT NULL,
  target_date_local DATE NOT NULL,
  asof_utc TIMESTAMP NOT NULL,
  member_stat VARCHAR(16) NOT NULL,
  member INT NOT NULL,
  variable_name VARCHAR(32) NOT NULL,
  variable_level VARCHAR(64) NOT NULL,
  variable_info VARCHAR(64) NOT NULL,
  variable_alias VARCHAR(64) NOT NULL,
  reducer VARCHAR(16) NOT NULL,
  value_num DOUBLE NULL,
  value_text VARCHAR(128) NULL,
  sample_count INT NULL,
  expected_count INT NULL,
  window_start_utc TIMESTAMP NOT NULL,
  window_end_utc TIMESTAMP NOT NULL,
  request_json TEXT NOT NULL,
  request_sha256 CHAR(64) NOT NULL,
  response_sha256 CHAR(64) NOT NULL,
  retrieved_at_utc TIMESTAMP NOT NULL,
  notes VARCHAR(512) NULL,
  PRIMARY KEY (id),
  UNIQUE (station_id, model_code, target_date_local, asof_utc, member_stat, member,
          variable_alias, reducer)
);

CREATE INDEX idx_gribstream_daily_var_station_target
  ON gribstream_daily_variable_value (station_id, target_date_local);

CREATE INDEX idx_gribstream_daily_var_model_target
  ON gribstream_daily_variable_value (model_code, target_date_local);

CREATE INDEX idx_gribstream_daily_var_asof
  ON gribstream_daily_variable_value (asof_utc);
