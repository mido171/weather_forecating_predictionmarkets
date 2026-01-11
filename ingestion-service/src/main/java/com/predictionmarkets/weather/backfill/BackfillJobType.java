package com.predictionmarkets.weather.backfill;

import java.util.Locale;

public enum BackfillJobType {
  KALSHI_SERIES_SYNC("kalshi_series_sync"),
  CLI_INGEST_YEAR("cli_ingest_year"),
  MOS_INGEST_WINDOW("mos_ingest_window"),
  MOS_ASOF_MATERIALIZE_RANGE("mos_asof_materialize_range");

  private final String jobName;

  BackfillJobType(String jobName) {
    this.jobName = jobName;
  }

  public String jobName() {
    return jobName;
  }

  public static BackfillJobType fromJobName(String jobName) {
    if (jobName == null || jobName.isBlank()) {
      throw new IllegalArgumentException("job is required");
    }
    String normalized = jobName.trim().toLowerCase(Locale.ROOT);
    for (BackfillJobType type : values()) {
      if (type.jobName.equals(normalized)) {
        return type;
      }
    }
    throw new IllegalArgumentException("Unsupported job: " + jobName);
  }
}
