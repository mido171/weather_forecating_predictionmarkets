package com.predictionmarkets.weather.models;

import java.io.Serializable;
import java.time.LocalDate;
import java.util.Objects;
import jakarta.persistence.Column;
import jakarta.persistence.Embeddable;

@Embeddable
public class CliDailyId implements Serializable {
  @Column(name = "station_id", nullable = false, length = 8)
  private String stationId;

  @Column(name = "target_date_local", nullable = false)
  private LocalDate targetDateLocal;

  public CliDailyId() {
  }

  public CliDailyId(String stationId, LocalDate targetDateLocal) {
    this.stationId = stationId;
    this.targetDateLocal = targetDateLocal;
  }

  public String getStationId() {
    return stationId;
  }

  public void setStationId(String stationId) {
    this.stationId = stationId;
  }

  public LocalDate getTargetDateLocal() {
    return targetDateLocal;
  }

  public void setTargetDateLocal(LocalDate targetDateLocal) {
    this.targetDateLocal = targetDateLocal;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    CliDailyId cliDailyId = (CliDailyId) o;
    return Objects.equals(stationId, cliDailyId.stationId)
        && Objects.equals(targetDateLocal, cliDailyId.targetDateLocal);
  }

  @Override
  public int hashCode() {
    return Objects.hash(stationId, targetDateLocal);
  }
}
