package com.predictionmarkets.weather.config;

import java.time.LocalDate;
import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "cli-settlement")
public class CliSettlementIngestionProperties {
  private List<String> stationIds;
  private LocalDate startDateLocal;
  private LocalDate endDateLocal;

  public List<String> getStationIds() {
    return stationIds;
  }

  public void setStationIds(List<String> stationIds) {
    this.stationIds = stationIds;
  }

  public LocalDate getStartDateLocal() {
    return startDateLocal;
  }

  public void setStartDateLocal(LocalDate startDateLocal) {
    this.startDateLocal = startDateLocal;
  }

  public LocalDate getEndDateLocal() {
    return endDateLocal;
  }

  public void setEndDateLocal(LocalDate endDateLocal) {
    this.endDateLocal = endDateLocal;
  }
}
