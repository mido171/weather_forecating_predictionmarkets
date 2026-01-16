CREATE TABLE mos_forecast_value (
  id BIGINT NOT NULL AUTO_INCREMENT,
  station_id VARCHAR(8) NOT NULL,
  model VARCHAR(16) NOT NULL,
  runtime_utc TIMESTAMP NOT NULL,
  forecast_time_utc TIMESTAMP NOT NULL,
  variable_code VARCHAR(16) NOT NULL,
  value_num DECIMAL(10, 4) NULL,
  value_text VARCHAR(64) NULL,
  value_raw VARCHAR(128) NOT NULL,
  raw_payload_hash_ref CHAR(64) NOT NULL,
  retrieved_at_utc TIMESTAMP NOT NULL,
  PRIMARY KEY (id),
  UNIQUE (station_id, model, runtime_utc, forecast_time_utc, variable_code),
  CONSTRAINT fk_mos_forecast_station
    FOREIGN KEY (station_id) REFERENCES station_registry(station_id),
  CONSTRAINT ck_mos_forecast_model
    CHECK (model IN ('GFS', 'MEX', 'NAM', 'NBS', 'NBE'))
);

CREATE INDEX idx_mos_forecast_station_time
  ON mos_forecast_value (station_id, forecast_time_utc);

CREATE INDEX idx_mos_forecast_runtime
  ON mos_forecast_value (runtime_utc);

CREATE TABLE gribstream_forecast_value (
  id BIGINT NOT NULL AUTO_INCREMENT,
  station_id VARCHAR(32) NOT NULL,
  zone_id VARCHAR(64) NOT NULL,
  model_code VARCHAR(32) NOT NULL,
  asof_utc TIMESTAMP NOT NULL,
  forecasted_at_utc TIMESTAMP NOT NULL,
  forecasted_time_utc TIMESTAMP NOT NULL,
  member INT NULL,
  variable_name VARCHAR(32) NOT NULL,
  variable_level VARCHAR(64) NOT NULL,
  variable_info VARCHAR(64) NOT NULL,
  variable_alias VARCHAR(64) NOT NULL,
  value_num DOUBLE NULL,
  value_text VARCHAR(128) NULL,
  request_json TEXT NOT NULL,
  request_sha256 CHAR(64) NOT NULL,
  response_sha256 CHAR(64) NOT NULL,
  retrieved_at_utc TIMESTAMP NOT NULL,
  notes VARCHAR(512) NULL,
  PRIMARY KEY (id),
  UNIQUE (station_id, model_code, asof_utc, forecasted_at_utc, forecasted_time_utc,
          variable_alias, member)
);

CREATE INDEX idx_gribstream_forecast_station_time
  ON gribstream_forecast_value (station_id, forecasted_time_utc);

CREATE INDEX idx_gribstream_forecast_model_time
  ON gribstream_forecast_value (model_code, forecasted_time_utc);

CREATE INDEX idx_gribstream_forecast_asof
  ON gribstream_forecast_value (asof_utc);
