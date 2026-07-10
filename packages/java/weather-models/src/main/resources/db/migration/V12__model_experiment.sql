CREATE TABLE model_experiment (
  id BIGINT NOT NULL AUTO_INCREMENT,
  experiment_key VARCHAR(255) NOT NULL,
  experiment_name VARCHAR(255) NULL,
  station_id VARCHAR(32) NULL,
  model_name VARCHAR(128) NULL,
  model_family VARCHAR(64) NULL,
  source_path VARCHAR(1024) NOT NULL,
  metadata_json LONGTEXT NOT NULL,
  metrics_train_json LONGTEXT NULL,
  metrics_validation_json LONGTEXT NULL,
  metrics_test_json LONGTEXT NULL,
  train_mae DOUBLE NULL,
  train_rmse DOUBLE NULL,
  train_bias DOUBLE NULL,
  train_median_ae DOUBLE NULL,
  train_max_ae DOUBLE NULL,
  train_corr DOUBLE NULL,
  train_n INT NULL,
  validation_mae DOUBLE NULL,
  validation_rmse DOUBLE NULL,
  validation_bias DOUBLE NULL,
  validation_median_ae DOUBLE NULL,
  validation_max_ae DOUBLE NULL,
  validation_corr DOUBLE NULL,
  validation_n INT NULL,
  test_mae DOUBLE NULL,
  test_rmse DOUBLE NULL,
  test_bias DOUBLE NULL,
  test_median_ae DOUBLE NULL,
  test_max_ae DOUBLE NULL,
  test_corr DOUBLE NULL,
  test_n INT NULL,
  description_text TEXT NOT NULL,
  raw_payload_hash CHAR(64) NULL,
  retrieved_at_utc TIMESTAMP NOT NULL,
  created_at_utc TIMESTAMP NOT NULL,
  updated_at_utc TIMESTAMP NOT NULL,
  PRIMARY KEY (id),
  UNIQUE (experiment_key)
);

CREATE INDEX idx_model_experiment_station
  ON model_experiment (station_id);

CREATE INDEX idx_model_experiment_test_mae
  ON model_experiment (test_mae);

CREATE INDEX idx_model_experiment_model_family
  ON model_experiment (model_family);
