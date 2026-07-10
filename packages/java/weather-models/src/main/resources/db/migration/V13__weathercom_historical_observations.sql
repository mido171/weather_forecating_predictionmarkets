CREATE TABLE weathercom_location (
  id BIGINT NOT NULL AUTO_INCREMENT,
  location_id VARCHAR(64) NOT NULL,
  display_name VARCHAR(255) NULL,
  active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at_utc TIMESTAMP NOT NULL,
  updated_at_utc TIMESTAMP NOT NULL,
  PRIMARY KEY (id),
  UNIQUE (location_id)
);

CREATE INDEX idx_weathercom_location_active
  ON weathercom_location (active);

CREATE TABLE weathercom_ingestion_run (
  id BIGINT NOT NULL AUTO_INCREMENT,
  status VARCHAR(32) NOT NULL,
  started_at_utc TIMESTAMP NOT NULL,
  finished_at_utc TIMESTAMP NULL,
  requested_by VARCHAR(128) NULL,
  request_payload_json LONGTEXT NOT NULL,
  total_tasks INT NOT NULL,
  succeeded_tasks INT NOT NULL DEFAULT 0,
  failed_tasks INT NOT NULL DEFAULT 0,
  created_at_utc TIMESTAMP NOT NULL,
  updated_at_utc TIMESTAMP NOT NULL,
  PRIMARY KEY (id),
  CONSTRAINT ck_weathercom_ingestion_run_status
    CHECK (status IN ('RUNNING', 'SUCCEEDED', 'PARTIAL_SUCCESS', 'FAILED'))
);

CREATE INDEX idx_weathercom_ingestion_run_started_at
  ON weathercom_ingestion_run (started_at_utc);

CREATE TABLE weathercom_api_call (
  id BIGINT NOT NULL AUTO_INCREMENT,
  ingestion_run_id BIGINT NOT NULL,
  request_location_id VARCHAR(64) NOT NULL,
  units CHAR(1) NOT NULL,
  start_date DATE NOT NULL,
  end_date DATE NOT NULL,
  response_location_id VARCHAR(64) NULL,
  response_units CHAR(1) NULL,
  response_language VARCHAR(32) NULL,
  transaction_id VARCHAR(128) NULL,
  api_version VARCHAR(32) NULL,
  expire_time_gmt BIGINT NULL,
  http_status INT NOT NULL,
  fetched_at_utc TIMESTAMP NOT NULL,
  duration_ms INT NULL,
  error_type VARCHAR(64) NULL,
  error_message LONGTEXT NULL,
  response_body_json LONGTEXT NULL,
  response_body_hash CHAR(64) NULL,
  created_at_utc TIMESTAMP NOT NULL,
  updated_at_utc TIMESTAMP NOT NULL,
  PRIMARY KEY (id),
  CONSTRAINT fk_weathercom_api_call_run
    FOREIGN KEY (ingestion_run_id) REFERENCES weathercom_ingestion_run(id)
);

CREATE INDEX idx_weathercom_api_call_request_location_date
  ON weathercom_api_call (request_location_id, start_date);

CREATE INDEX idx_weathercom_api_call_run_id
  ON weathercom_api_call (ingestion_run_id);

CREATE TABLE weathercom_observation (
  id BIGINT NOT NULL AUTO_INCREMENT,
  api_call_id BIGINT NOT NULL,
  request_location_id VARCHAR(64) NOT NULL,
  obs_id VARCHAR(32) NOT NULL,
  obs_key VARCHAR(32) NULL,
  obs_name VARCHAR(255) NULL,
  valid_time_gmt BIGINT NOT NULL,
  valid_time_utc TIMESTAMP NOT NULL,
  day_ind CHAR(1) NULL,
  temp INT NULL,
  dew_pt INT NULL,
  heat_index INT NULL,
  rh INT NULL,
  pressure DECIMAL(10, 4) NULL,
  pressure_tend INT NULL,
  pressure_desc VARCHAR(64) NULL,
  vis DECIMAL(10, 4) NULL,
  wc INT NULL,
  wdir INT NULL,
  wdir_cardinal VARCHAR(16) NULL,
  gust INT NULL,
  wspd INT NULL,
  wx_phrase VARCHAR(255) NULL,
  wx_icon INT NULL,
  icon_extd INT NULL,
  precip_total DECIMAL(10, 4) NULL,
  precip_hrly DECIMAL(10, 4) NULL,
  snow_hrly DECIMAL(10, 4) NULL,
  max_temp INT NULL,
  min_temp INT NULL,
  uv_desc VARCHAR(64) NULL,
  uv_index INT NULL,
  feels_like INT NULL,
  clds VARCHAR(64) NULL,
  qualifier VARCHAR(64) NULL,
  qualifier_svrty VARCHAR(64) NULL,
  blunt_phrase VARCHAR(255) NULL,
  terse_phrase VARCHAR(255) NULL,
  observation_class VARCHAR(32) NULL,
  water_temp INT NULL,
  primary_wave_period DECIMAL(10, 4) NULL,
  primary_wave_height DECIMAL(10, 4) NULL,
  primary_swell_period DECIMAL(10, 4) NULL,
  primary_swell_height DECIMAL(10, 4) NULL,
  primary_swell_direction INT NULL,
  secondary_swell_period DECIMAL(10, 4) NULL,
  secondary_swell_height DECIMAL(10, 4) NULL,
  secondary_swell_direction INT NULL,
  created_at_utc TIMESTAMP NOT NULL,
  updated_at_utc TIMESTAMP NOT NULL,
  PRIMARY KEY (id),
  UNIQUE (request_location_id, obs_id, valid_time_gmt),
  CONSTRAINT fk_weathercom_observation_api_call
    FOREIGN KEY (api_call_id) REFERENCES weathercom_api_call(id)
);

CREATE INDEX idx_weathercom_observation_request_valid
  ON weathercom_observation (request_location_id, valid_time_gmt);

CREATE INDEX idx_weathercom_observation_obs_valid
  ON weathercom_observation (obs_id, valid_time_gmt);

CREATE INDEX idx_weathercom_observation_api_call_id
  ON weathercom_observation (api_call_id);
