package com.predictionmarkets.weather.kalshi;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "kalshi")
public class KalshiProperties {
  private String baseUrl = "https://api.elections.kalshi.com/trade-api/v2";

  public String getBaseUrl() {
    return baseUrl;
  }

  public void setBaseUrl(String baseUrl) {
    this.baseUrl = baseUrl;
  }
}
