CREATE TABLE mos_daily_value (
  id BIGINT NOT NULL AUTO_INCREMENT,
  station_id VARCHAR(8) NOT NULL,
  station_zoneid VARCHAR(64) NOT NULL,
  model VARCHAR(16) NOT NULL,
  runtime_utc TIMESTAMP NOT NULL,
  target_date_local DATE NOT NULL,
  variable_code VARCHAR(16) NOT NULL,
  value_min DECIMAL(10, 4) NULL,
  value_max DECIMAL(10, 4) NULL,
  value_mean DECIMAL(10, 4) NULL,
  sample_count INT NOT NULL,
  first_forecast_time_utc TIMESTAMP NULL,
  last_forecast_time_utc TIMESTAMP NULL,
  raw_payload_hash_ref CHAR(64) NOT NULL,
  retrieved_at_utc TIMESTAMP NOT NULL,
  PRIMARY KEY (id),
  UNIQUE (station_id, model, runtime_utc, target_date_local, variable_code),
  CONSTRAINT fk_mos_daily_station
    FOREIGN KEY (station_id) REFERENCES station_registry(station_id),
  CONSTRAINT ck_mos_daily_model
    CHECK (model IN ('GFS', 'MEX', 'NAM', 'NBS', 'NBE'))
);

CREATE INDEX idx_mos_daily_station_target
  ON mos_daily_value (station_id, target_date_local);

CREATE INDEX idx_mos_daily_runtime
  ON mos_daily_value (runtime_utc);
