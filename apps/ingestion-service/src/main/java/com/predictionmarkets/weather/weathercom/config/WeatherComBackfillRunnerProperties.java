package com.predictionmarkets.weather.weathercom.config;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "weathercom.backfill-runner")
public class WeatherComBackfillRunnerProperties {
  private boolean enabled;
  private List<String> locationIds = new ArrayList<>();
  private LocalDate startDate;
  private LocalDate endDate;
  private String units = "e";
  private String requestedBy = "system";
  private int pollIntervalSeconds = 15;
  private boolean failOnNonSucceeded = true;

  public boolean isEnabled() {
    return enabled;
  }

  public void setEnabled(boolean enabled) {
    this.enabled = enabled;
  }

  public List<String> getLocationIds() {
    return locationIds;
  }

  public void setLocationIds(List<String> locationIds) {
    this.locationIds = locationIds;
  }

  public LocalDate getStartDate() {
    return startDate;
  }

  public void setStartDate(LocalDate startDate) {
    this.startDate = startDate;
  }

  public LocalDate getEndDate() {
    return endDate;
  }

  public void setEndDate(LocalDate endDate) {
    this.endDate = endDate;
  }

  public String getUnits() {
    return units;
  }

  public void setUnits(String units) {
    this.units = units;
  }

  public String getRequestedBy() {
    return requestedBy;
  }

  public void setRequestedBy(String requestedBy) {
    this.requestedBy = requestedBy;
  }

  public int getPollIntervalSeconds() {
    return pollIntervalSeconds;
  }

  public void setPollIntervalSeconds(int pollIntervalSeconds) {
    this.pollIntervalSeconds = pollIntervalSeconds;
  }

  public boolean isFailOnNonSucceeded() {
    return failOnNonSucceeded;
  }

  public void setFailOnNonSucceeded(boolean failOnNonSucceeded) {
    this.failOnNonSucceeded = failOnNonSucceeded;
  }
}
