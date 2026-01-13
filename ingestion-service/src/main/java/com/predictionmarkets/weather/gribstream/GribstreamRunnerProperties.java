package com.predictionmarkets.weather.gribstream;

import java.time.Instant;
import java.time.LocalDate;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app.runners.gribstream-example")
public class GribstreamRunnerProperties {
  private boolean enabled;
  private LocalDate startDateLocal;
  private LocalDate endDateLocal;
  private Instant asOfUtc;

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
}
