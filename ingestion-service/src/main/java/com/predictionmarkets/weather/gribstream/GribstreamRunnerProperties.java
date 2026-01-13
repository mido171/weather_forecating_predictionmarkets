package com.predictionmarkets.weather.gribstream;

import com.predictionmarkets.weather.models.AsofTimeZone;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalTime;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app.runners.gribstream-example")
public class GribstreamRunnerProperties {
  private boolean enabled;
  private LocalDate startDateLocal;
  private LocalDate endDateLocal;
  private Instant asOfUtc;
  private LocalTime asOfLocalTime;
  private AsofTimeZone asOfTimeZone;

  public boolean isEnabled() {
    return enabled;
  }

  public void setEnabled(boolean enabled) {
    this.enabled = enabled;
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

  public Instant getAsOfUtc() {
    return asOfUtc;
  }

  public void setAsOfUtc(Instant asOfUtc) {
    this.asOfUtc = asOfUtc;
  }

  public LocalTime getAsOfLocalTime() {
    return asOfLocalTime;
  }

  public void setAsOfLocalTime(LocalTime asOfLocalTime) {
    this.asOfLocalTime = asOfLocalTime;
  }

  public AsofTimeZone getAsOfTimeZone() {
    return asOfTimeZone;
  }

  public void setAsOfTimeZone(AsofTimeZone asOfTimeZone) {
    this.asOfTimeZone = asOfTimeZone;
  }
}
