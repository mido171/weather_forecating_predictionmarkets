package com.predictionmarkets.weather.kalshiapi.security;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "kalshi.local-control")
public class LocalControlProperties {
  private String token;

  public String getToken() {
    return token;
  }

  public void setToken(String token) {
    this.token = token;
  }

  public boolean hasToken() {
    return token != null && !token.isBlank();
  }
}
