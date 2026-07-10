package com.predictionmarkets.weather.pilot.config;

public class SourceConfig {
  private String sourceName;
  private String sourceFamily;
  private boolean enabled;
  private String baseUrl;

  public String getSourceName() {
    return sourceName;
  }

  public void setSourceName(String sourceName) {
    this.sourceName = sourceName;
  }

  public String getSourceFamily() {
    return sourceFamily;
  }

  public void setSourceFamily(String sourceFamily) {
    this.sourceFamily = sourceFamily;
  }

  public boolean isEnabled() {
    return enabled;
  }

  public void setEnabled(boolean enabled) {
    this.enabled = enabled;
  }

  public String getBaseUrl() {
    return baseUrl;
  }

  public void setBaseUrl(String baseUrl) {
    this.baseUrl = baseUrl;
  }
}
