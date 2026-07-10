package com.predictionmarkets.weather.security;

import jakarta.annotation.PostConstruct;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "ingestion.admin-api")
public class AdminApiProperties {
  private boolean enabled;
  private String controlToken;

  @PostConstruct
  public void validate() {
    if (enabled && (controlToken == null || controlToken.isBlank())) {
      throw new IllegalStateException(
          "INGESTION_LOCAL_CONTROL_TOKEN is required when ingestion.admin-api.enabled=true");
    }
  }

  public boolean isEnabled() {
    return enabled;
  }

  public void setEnabled(boolean enabled) {
    this.enabled = enabled;
  }

  public String getControlToken() {
    return controlToken;
  }

  public void setControlToken(String controlToken) {
    this.controlToken = controlToken;
  }
}
