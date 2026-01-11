CREATE TABLE kalshi_series (
  series_ticker VARCHAR(32) NOT NULL,
  title VARCHAR(256) NOT NULL,
  category VARCHAR(64) NOT NULL,
  settlement_source_name VARCHAR(128) NOT NULL,
  settlement_source_url VARCHAR(1024) NOT NULL,
  contract_terms_url VARCHAR(1024),
  contract_url VARCHAR(1024),
  retrieved_at_utc TIMESTAMP NOT NULL,
  raw_payload_hash CHAR(64) NOT NULL,
  raw_json LONGTEXT,
  PRIMARY KEY (series_ticker)
);

CREATE TABLE station_registry (
  station_id VARCHAR(8) NOT NULL,
  issuedby VARCHAR(8) NOT NULL,
  wfo_site VARCHAR(8) NOT NULL,
  series_ticker VARCHAR(32) NOT NULL,
  zone_id VARCHAR(64) NOT NULL,
  standard_offset_minutes INT NOT NULL,
  mapping_status VARCHAR(32) NOT NULL,
  created_at_utc TIMESTAMP NOT NULL,
  updated_at_utc TIMESTAMP NOT NULL,
  PRIMARY KEY (station_id),
  CONSTRAINT fk_station_series
    FOREIGN KEY (series_ticker) REFERENCES kalshi_series(series_ticker),
  CONSTRAINT ck_station_mapping_status
    CHECK (mapping_status IN ('AUTO_OK', 'NEEDS_OVERRIDE'))
);

CREATE TABLE station_override (
  issuedby VARCHAR(8) NOT NULL,
  station_id_override VARCHAR(8) NOT NULL,
  zone_id_override VARCHAR(64) NOT NULL,
  notes VARCHAR(512),
  PRIMARY KEY (issuedby)
);

CREATE TABLE asof_policy (
  id BIGINT NOT NULL AUTO_INCREMENT,
  name VARCHAR(64) NOT NULL,
  asof_local_time TIME NOT NULL,
  enabled BOOLEAN NOT NULL,
  PRIMARY KEY (id),
  UNIQUE (name)
);

CREATE TABLE cli_daily (
  station_id VARCHAR(8) NOT NULL,
  target_date_local DATE NOT NULL,
  tmax_f DECIMAL(5, 2),
  tmin_f DECIMAL(5, 2),
  report_issued_at_utc TIMESTAMP NULL,
  raw_payload_hash CHAR(64) NOT NULL,
  retrieved_at_utc TIMESTAMP NOT NULL,
  PRIMARY KEY (station_id, target_date_local),
  CONSTRAINT fk_cli_station
    FOREIGN KEY (station_id) REFERENCES station_registry(station_id)
);

CREATE TABLE mos_run (
  station_id VARCHAR(8) NOT NULL,
  model VARCHAR(16) NOT NULL,
  runtime_utc TIMESTAMP NOT NULL,
  raw_payload_hash CHAR(64) NOT NULL,
  retrieved_at_utc TIMESTAMP NOT NULL,
  PRIMARY KEY (station_id, model, runtime_utc),
  CONSTRAINT fk_mos_run_station
    FOREIGN KEY (station_id) REFERENCES station_registry(station_id),
  CONSTRAINT ck_mos_model
    CHECK (model IN ('GFS', 'MEX', 'NAM', 'NBS', 'NBE'))
);

CREATE TABLE mos_asof_feature (
  station_id VARCHAR(8) NOT NULL,
  target_date_local DATE NOT NULL,
  asof_policy_id BIGINT NOT NULL,
  model VARCHAR(16) NOT NULL,
  asof_utc TIMESTAMP NOT NULL,
  asof_local TIMESTAMP NOT NULL,
  station_zoneid VARCHAR(64) NOT NULL,
  chosen_runtime_utc TIMESTAMP NULL,
  tmax_f DECIMAL(5, 2),
  missing_reason VARCHAR(128),
  raw_payload_hash_ref CHAR(64),
  retrieved_at_utc TIMESTAMP NOT NULL,
  PRIMARY KEY (station_id, target_date_local, asof_policy_id, model),
  CONSTRAINT fk_asof_station
    FOREIGN KEY (station_id) REFERENCES station_registry(station_id),
  CONSTRAINT fk_asof_policy
    FOREIGN KEY (asof_policy_id) REFERENCES asof_policy(id),
  CONSTRAINT ck_mos_asof_model
    CHECK (model IN ('GFS', 'MEX', 'NAM', 'NBS', 'NBE'))
);

CREATE TABLE ingest_checkpoint (
  id BIGINT NOT NULL AUTO_INCREMENT,
  job_name VARCHAR(128) NOT NULL,
  station_id VARCHAR(8) NOT NULL,
  model VARCHAR(16),
  cursor_date DATE,
  cursor_runtime_utc TIMESTAMP,
  status VARCHAR(16) NOT NULL,
  updated_at_utc TIMESTAMP NOT NULL,
  PRIMARY KEY (id),
  UNIQUE (job_name, station_id, model),
  CONSTRAINT fk_checkpoint_station
    FOREIGN KEY (station_id) REFERENCES station_registry(station_id),
  CONSTRAINT ck_checkpoint_model
    CHECK (model IS NULL OR model IN ('GFS', 'MEX', 'NAM', 'NBS', 'NBE')),
  CONSTRAINT ck_checkpoint_status
    CHECK (status IN ('RUNNING', 'COMPLETE', 'FAILED'))
);
